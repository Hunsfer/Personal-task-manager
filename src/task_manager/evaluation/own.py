"""Checks the promise prompt on the owner's own messages, labelled separately.

There is no held-out part: about 50 messages are too few to split, so these numbers are optimistic.
"""

import json
import re
import time
from pathlib import Path

from task_manager.analyzer.classifier import AnswerError, Classifier
from task_manager.analyzer.prompt import render_user
from task_manager.evaluation.metrics import LABEL_MINE, LABEL_PART, Rate, wilson
from task_manager.evaluation.sampling import Example


def answers_path(results_dir: Path, model: str, prompt_id: str) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    return results_dir / f"own__{safe_model}__{prompt_id}.jsonl"


def read_answers(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    return {row["message_id"]: row for row in rows}


async def run_own(classifier: Classifier, examples: list[Example], path: Path) -> None:
    """Resumes where it stopped. An outage raises rather than being recorded as an answer."""
    done = read_answers(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for example in examples:
            if example.message_id in done:
                continue
            started = time.monotonic()
            row = {"message_id": example.message_id, "error": None, "draft": False}
            try:
                decision = await classifier.classify(example.thread, own=True)
            except AnswerError as error:
                row["error"] = str(error)
            else:
                row |= {
                    "draft": decision.creates_draft,
                    "title": decision.title,
                    "description": decision.description,
                }
            row["seconds"] = time.monotonic() - started
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()


def score_own(answers: dict[int, dict], labels: dict[int, int]) -> tuple[Rate, Rate, int]:
    """Precision, recall and how many answers failed. Counts only.

    Clarifying a task that already exists counts neither way, as it does for incoming messages.
    """
    rows = [
        row
        for message_id, row in answers.items()
        if message_id in labels and labels[message_id] != LABEL_PART
    ]
    drafts = [row for row in rows if row["draft"]]
    correct = sum(labels[row["message_id"]] == LABEL_MINE for row in drafts)
    mine = sum(labels[row["message_id"]] == LABEL_MINE for row in rows)
    return wilson(correct, len(drafts)), wilson(correct, mine), sum(bool(r["error"]) for r in rows)


def write_own_errors(
    path: Path, answers: dict[int, dict], labels: dict[int, int], examples: list[Example]
) -> tuple[int, int]:
    """A file for the owner: where the model and the labels differ."""
    threads = {example.message_id: example.thread for example in examples}
    extra = [
        m
        for m, row in answers.items()
        if row["draft"] and labels.get(m) not in (LABEL_MINE, LABEL_PART)
    ]
    missed = [m for m, row in answers.items() if not row["draft"] and labels.get(m) == LABEL_MINE]
    parts = ["# Обещания: расхождения с разметкой", "", "Файл содержит обезличенную переписку.", ""]
    for title, group in (
        (f"Лишние ({len(extra)})", extra),
        (f"Пропущенные ({len(missed)})", missed),
    ):
        parts += [f"## {title}", ""]
        for message_id in group:
            context = render_user(threads[message_id])
            fence = "`" * max(3, 1 + max(map(len, re.findall(r"`+", context)), default=0))
            parts += [f"### ({message_id})", "", f"{fence}text\n{context}\n{fence}", ""]
            description = answers[message_id].get("description")
            if description:
                parts += [f"```text\nОписание модели: {description}\n```", ""]
    path.write_text("\n".join(parts), encoding="utf-8")
    return len(extra), len(missed)
