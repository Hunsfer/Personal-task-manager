"""Runs one model and prompt over the sample and reports aggregate numbers.

Answers go to files in eval/results, which are as sensitive as the dataset. Only counters and rates
are printed, so reports can be shared.
"""

import re
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from task_manager.analyzer.ollama import ChatResult, OllamaClient, think_setting
from task_manager.analyzer.prompt import load_template, render_system, render_user, template_id
from task_manager.analyzer.verdict import (
    InvalidVerdictError,
    TaskVerdict,
    has_placeholder,
    parse_verdict,
    verdict_schema,
)
from task_manager.common.config import ScopeConfig
from task_manager.evaluation import metrics
from task_manager.evaluation.sampling import Example, Sample
from task_manager.evaluation.split import Part

THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
ANSWER_SUFFIX = ".jsonl"
THINKING_SUFFIX = "-think"


class Answer(BaseModel):
    message_id: int
    content: str
    error: str | None
    seconds: float
    prompt_tokens: int
    output_tokens: int
    truncated_input: bool


def answer_path(results_dir: Path, part: Part, model: str, prompt_id: str) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    return results_dir / f"{part}__{safe_model}__{prompt_id}{ANSWER_SUFFIX}"


def read_answers(path: Path) -> dict[int, Answer]:
    if not path.is_file():
        return {}
    answers = [Answer.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]
    return {answer.message_id: answer for answer in answers}


def _classify(result: ChatResult) -> str | None:
    if result.truncated_output:
        return "output_too_long"
    try:
        parse_verdict(result.content)
    except InvalidVerdictError:
        return "schema"
    return None


async def run_model(
    client: OllamaClient,
    model: str,
    prompt_path: Path,
    scope: ScopeConfig,
    examples: list[Example],
    part: Part,
    results_dir: Path,
    thinking: bool = False,
) -> Path:
    """Answer every example not answered yet, so an interrupted run continues where it stopped.

    If Ollama goes away this raises, rather than recording the outage as a wrong answer.
    """
    template = load_template(prompt_path)
    system = render_system(template, scope)
    schema = verdict_schema()
    think = True if thinking else think_setting(await client.show(model))

    results_dir.mkdir(parents=True, exist_ok=True)
    prompt_id = template_id(prompt_path, template) + (THINKING_SUFFIX if thinking else "")
    path = answer_path(results_dir, part, model, prompt_id)
    done = read_answers(path)
    with path.open("a", encoding="utf-8") as output:
        for example in examples:
            if example.message_id in done:
                continue
            started = time.monotonic()
            result = await client.chat(model, system, render_user(example.thread), schema, think)
            answer = Answer(
                message_id=example.message_id,
                content=result.content,
                error=_classify(result),
                seconds=time.monotonic() - started,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
                truncated_input=result.truncated_input,
            )
            output.write(answer.model_dump_json() + "\n")
            output.flush()
    return path


def _verdict(answer: Answer) -> TaskVerdict | None:
    return None if answer.error else parse_verdict(answer.content)


def score_rows(
    answers: dict[int, Answer], sample: Sample, labels: dict[int, int]
) -> list[metrics.Scored]:
    strata = {example.message_id: example.stratum for example in sample.examples}
    return [
        metrics.Scored(
            message_id=answer.message_id,
            label=labels[answer.message_id],
            stratum=strata[answer.message_id],
            verdict=_verdict(answer),
            error=answer.error,
            seconds=answer.seconds,
            truncated=answer.truncated_input,
        )
        for answer in answers.values()
        if answer.message_id in labels and answer.message_id in strata
    ]


def _percent(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.1%}"


def _rate(rate: metrics.Rate) -> str:
    return (
        f"{_percent(rate.value)} ({rate.hits}/{rate.total}, "
        f"интервал {rate.low:.0%}–{rate.high:.0%})"
    )


def format_report(title: str, rows: list[metrics.Scored], sample: Sample, threshold: float) -> str:
    """Aggregates only. No message text, no answers, no identifiers."""
    if not rows:
        return f"{title}\n  нет ответов"
    lines = [title]
    labels = Counter(row.label for row in rows)
    errors = Counter(row.error for row in rows if row.error)
    seconds = [row.seconds for row in rows]
    lines.append(f"  ответов {len(rows)}, метки {dict(sorted(labels.items()))}")
    lines.append(
        f"  ошибки {dict(errors)}, обрезан ввод {sum(row.truncated for row in rows)}, "
        f"время: среднее {sum(seconds) / len(seconds):.1f} с, "
        f"p95 {metrics.percentile(seconds, 0.95):.1f} с"
    )
    lines.append(f"  порог  {'precision':<38} recall")
    for candidate in sorted({*THRESHOLDS, threshold}):
        result = metrics.score(rows, candidate, sample.population)
        lines.append(
            f"  {candidate:4.2f}   {_rate(result.precision):<38} {_rate(result.recall)}"
            f"  взвеш. {_percent(result.weighted_precision)} / {_percent(result.weighted_recall)}"
        )
    chosen = metrics.score(rows, threshold, sample.population)
    lines.append(f"  при пороге {threshold:.2f}:")
    lines.append(f"    метка «возможно мне»: {chosen.maybe_label_destinations}")
    for reason, rate in metrics.precision_by_reason(rows, threshold).items():
        lines.append(f"    precision {reason}: {_rate(rate)}")
    missed_by_prefilter = sum(
        row.label == metrics.LABEL_MINE and row.stratum == "none" for row in rows
    )
    lines.append(f"    сообщений «задача мне» без единого признака: {missed_by_prefilter}")

    threads = {example.message_id: example.thread for example in sample.examples}
    drafts = [row for row in rows if metrics.is_draft(row.verdict, threshold)]
    placeholders = sum(has_placeholder(row.verdict.description) for row in drafts)
    copied = sum(
        metrics.copies_source(row.verdict.description, threads[row.message_id].lines())
        for row in drafts
    )
    lines.append(
        f"    описания: черновиков {len(drafts)}, с плейсхолдерами {placeholders}, "
        f"с фрагментом чата от {metrics.COPIED_WORDS} слов {copied}"
    )
    return "\n".join(lines)


def write_review(
    path: Path, rows: list[metrics.Scored], sample: Sample, threshold: float, limit: int = 30
) -> int:
    """A file for the owner: descriptions next to their source, in code blocks."""
    threads = {example.message_id: example.thread for example in sample.examples}
    drafts = [row for row in rows if metrics.is_draft(row.verdict, threshold)][:limit]
    parts = [
        "# Проверка описаний",
        "",
        "Файл содержит обезличенную переписку. Удалите его после просмотра.",
        "",
        "Понятно ли описание без чата? Нет ли цитат и имён?",
        "",
    ]
    for number, row in enumerate(drafts, 1):
        thread = threads[row.message_id]
        context = render_user(thread)
        fence = "`" * max(3, 1 + max(map(len, re.findall(r"`+", context)), default=0))
        parts += [
            f"## {number}. {row.verdict.title}",
            "",
            f"```text\n{row.verdict.description}\n```",
            "",
            f"{fence}text\n{context}\n{fence}",
            "",
        ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return len(drafts)


_LABEL_NAMES = {
    metrics.LABEL_MINE: "задача мне",
    metrics.LABEL_MAYBE: "возможно мне",
    metrics.LABEL_OTHERS: "задача не мне",
    metrics.LABEL_NOT_TASK: "не задача",
    metrics.LABEL_PART: "часть задачи",
}


def write_disagreements(
    path: Path, rows: list[metrics.Scored], sample: Sample, threshold: float
) -> tuple[int, int]:
    """A file for the owner: where the model and the labels differ, to find patterns or fix labels.

    Only for the tuning part: looking at final examples would spoil the final check.
    """
    threads = {example.message_id: example.thread for example in sample.examples}
    extra = [
        row
        for row in rows
        if metrics.is_draft(row.verdict, threshold)
        and row.label not in (metrics.LABEL_MINE, metrics.LABEL_PART)
    ]
    missed = [
        row
        for row in rows
        if row.label == metrics.LABEL_MINE and not metrics.is_draft(row.verdict, threshold)
    ]
    parts = [
        "# Расхождения модели с разметкой",
        "",
        "Файл содержит обезличенную переписку. Удалите его после просмотра.",
        "",
        "Ищите повторяющиеся причины. Если метка поставлена неверно, запишите номер в скобках.",
        "",
    ]
    for title, group in (
        (f"Лишние черновики ({len(extra)})", extra),
        (f"Пропущенные задачи ({len(missed)})", missed),
    ):
        parts += [f"## {title}", ""]
        for row in group:
            context = render_user(threads[row.message_id])
            fence = "`" * max(3, 1 + max(map(len, re.findall(r"`+", context)), default=0))
            answer = row.verdict
            summary = (
                f"Ошибка ответа: {row.error}"
                if answer is None
                else f"is_task {answer.is_task}, мне {answer.assigned_to_me}, "
                f"причина {answer.assignment_reason.value}, уверенность {answer.confidence}"
            )
            parts += [
                f"### ({row.message_id}) ваша метка: {_LABEL_NAMES[row.label]}",
                "",
                f"Модель: {summary}",
                "",
                f"{fence}text\n{context}\n{fence}",
                "",
            ]
            if answer is not None and answer.description:
                parts += [f"```text\nОписание модели: {answer.description}\n```", ""]
    path.write_text("\n".join(parts), encoding="utf-8")
    return len(extra), len(missed)
