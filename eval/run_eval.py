"""Compare models and prompts on the owner's labelled messages.

    uv run python eval/run_eval.py split
    uv run python eval/run_eval.py run --model qwen3:14b --prompt config/prompts/task_v1.md
    uv run python eval/run_eval.py report --threshold 0.8
    uv run python eval/run_eval.py run --part final --model ... --prompt ...
    uv run python eval/run_eval.py review --model ... --prompt ... --out temp/descriptions.md
    uv run python eval/run_eval.py errors --model ... --prompt ... --out temp/errors.md
    uv run python eval/run_eval.py prefilter
    uv run python eval/run_eval.py own-run
    uv run python eval/run_eval.py own-report
    uv run python eval/run_eval.py own-errors --out temp/own_errors.md

Reads only eval/dataset, never the database. Printed output is aggregate numbers, safe to share.
Answers stay in eval/results. The final part is meant to run once, for the chosen combination.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from task_manager.analyzer.classifier import Classifier
from task_manager.analyzer.ollama import OllamaClient, OllamaUnavailableError
from task_manager.analyzer.prefilter import Prefilter
from task_manager.analyzer.prompt import load_template, template_id
from task_manager.common.config import (
    ConfigError,
    load_analyzer_config,
    load_app_config,
    load_settings,
)
from task_manager.evaluation import own, runner
from task_manager.evaluation.labeling import LABELS_FILE, read_labels
from task_manager.evaluation.sampling import read_sample
from task_manager.evaluation.split import group_examples, split_examples

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
RESULTS = ROOT / "results"
SPLIT_FILE = DATASET / "split.json"
OWN_DATASET = DATASET / "own"


def _read_split() -> dict[int, str]:
    if not SPLIT_FILE.is_file():
        raise ConfigError("run the split command first")
    return {int(key): value for key, value in json.loads(SPLIT_FILE.read_text("utf-8")).items()}


def _split() -> int:
    sample = read_sample(DATASET)
    labels = read_labels(DATASET / LABELS_FILE)
    parts = split_examples(sample.examples, labels)
    SPLIT_FILE.write_text(json.dumps(parts), encoding="utf-8")

    labelled = [example for example in sample.examples if example.message_id in labels]
    groups = group_examples(labelled)
    print(f"размечено {len(labelled)} из {len(sample.examples)}")
    print(f"групп {len(groups)}, самая большая {max(map(len, groups), default=0)}")
    for part in ("dev", "final"):
        chosen = [example for example in labelled if parts[example.message_id] == part]
        counts = Counter(labels[example.message_id] for example in chosen)
        print(f"{part}: {len(chosen)}, метки {dict(sorted(counts.items()))}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    scope = load_app_config(settings.config_dir).scope
    sample = read_sample(DATASET)
    parts = _read_split()
    examples = [e for e in sample.examples if parts.get(e.message_id) == args.part]
    client = OllamaClient(settings.ollama_url)
    try:
        path = await runner.run_model(
            client, args.model, args.prompt, scope, examples, args.part, RESULTS, args.think
        )
    finally:
        await client.aclose()
    print(f"готово: {path.name}, ответов {len(runner.read_answers(path))} из {len(examples)}")
    return 0


def _report(args: argparse.Namespace) -> int:
    sample = read_sample(DATASET)
    labels = read_labels(DATASET / LABELS_FILE)
    paths = sorted(RESULTS.glob(f"*{runner.ANSWER_SUFFIX}"))
    final_runs = [path for path in paths if path.name.startswith("final__")]
    if len(final_runs) > 1:
        print(f"ВНИМАНИЕ: финальная часть прогонялась {len(final_runs)} раз, цифры завышены\n")
    for path in paths:
        rows = runner.score_rows(runner.read_answers(path), sample, labels)
        print(runner.format_report(path.stem, rows, sample, args.threshold))
        print()
    return 0


def _rows_for(args: argparse.Namespace):
    sample = read_sample(DATASET)
    labels = read_labels(DATASET / LABELS_FILE)
    prompt_id = template_id(args.prompt, load_template(args.prompt))
    if args.think:
        prompt_id += runner.THINKING_SUFFIX
    path = runner.answer_path(RESULTS, args.part, args.model, prompt_id)
    return runner.score_rows(runner.read_answers(path), sample, labels), sample


def _errors(args: argparse.Namespace) -> int:
    if args.part == "final":
        raise ConfigError("looking at final examples would spoil the final check")
    rows, sample = _rows_for(args)
    extra, missed = runner.write_disagreements(args.out, rows, sample, args.threshold)
    print(f"лишних {extra}, пропущенных {missed}. Файл {args.out} содержит обезличенную переписку.")
    return 0


def _review(args: argparse.Namespace) -> int:
    rows, sample = _rows_for(args)
    count = runner.write_review(args.out, rows, sample, args.threshold)
    print(f"описаний: {count}, файл {args.out}. Он содержит обезличенную переписку.")
    return 0


def _prefilter() -> int:
    """What the prefilter would drop from the labelled messages. Counts only."""
    settings = load_settings()
    prefilter = Prefilter(load_analyzer_config(settings.config_dir).noise_phrases)
    for name, directory in (("входящие", DATASET), ("свои", OWN_DATASET)):
        if not (directory / LABELS_FILE).is_file():
            print(f"{name}: разметки нет")
            continue
        labels = read_labels(directory / LABELS_FILE)
        dropped: Counter[int] = Counter()
        examples = read_sample(directory).examples
        for example in examples:
            if prefilter.noise_reason(example.thread.target) is not None:
                dropped[labels.get(example.message_id, 0)] += 1
        print(
            f"{name}: сообщений {len(examples)}, отсеяно по меткам {dict(sorted(dropped.items()))}"
        )
    return 0


def _own_setup():
    settings = load_settings()
    client = OllamaClient(settings.ollama_url)
    classifier = Classifier(
        client,
        settings.analyzer_model,
        settings.config_dir / "prompts",
        settings.analyzer_prompt,
        load_app_config(settings.config_dir).scope,
    )
    path = own.answers_path(RESULTS, classifier.model, classifier.promise_prompt_id)
    return client, classifier, path


async def _own_run() -> int:
    client, classifier, path = _own_setup()
    examples = read_sample(OWN_DATASET).examples
    try:
        await own.run_own(classifier, examples, path)
    finally:
        await client.aclose()
    print(f"готово: {path.name}, ответов {len(own.read_answers(path))} из {len(examples)}")
    return 0


def _own_report() -> int:
    _, _, path = _own_setup()
    labels = read_labels(OWN_DATASET / LABELS_FILE)
    precision, recall, failed = own.score_own(own.read_answers(path), labels)
    print(path.stem)
    print(f"  метки {dict(sorted(Counter(labels.values()).items()))}, ошибок ответа {failed}")
    print(f"  precision {runner._rate(precision)}")
    print(f"  recall    {runner._rate(recall)}")
    print("  Отложенной части нет: цифры оптимистичны.")
    return 0


def _own_errors(args: argparse.Namespace) -> int:
    _, _, path = _own_setup()
    labels = read_labels(OWN_DATASET / LABELS_FILE)
    extra, missed = own.write_own_errors(
        args.out, own.read_answers(path), labels, read_sample(OWN_DATASET).examples
    )
    print(f"лишних {extra}, пропущенных {missed}. Файл {args.out} содержит обезличенную переписку.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("split")
    commands.add_parser("prefilter")
    commands.add_parser("own-run")
    commands.add_parser("own-report")
    commands.add_parser("own-errors").add_argument("--out", required=True, type=Path)
    for name in ("run", "review", "errors"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--prompt", required=True, type=Path)
        command.add_argument("--part", choices=("dev", "final"), default="dev")
        command.add_argument("--think", action="store_true", help="let the model reason first")
    report = commands.add_parser("report")
    report.add_argument("--threshold", type=float, default=0.8)
    for name in ("review", "errors"):
        command = commands.choices[name]
        command.add_argument("--threshold", type=float, default=0.8)
        command.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        match args.command:
            case "split":
                return _split()
            case "prefilter":
                return _prefilter()
            case "own-run":
                return asyncio.run(_own_run())
            case "own-report":
                return _own_report()
            case "own-errors":
                return _own_errors(args)
            case "run":
                return asyncio.run(_run(args))
            case "report":
                return _report(args)
            case "errors":
                return _errors(args)
            case _:
                return _review(args)
    except (ConfigError, ValueError, FileNotFoundError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except OllamaUnavailableError as error:
        print(f"Ollama недоступна: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
