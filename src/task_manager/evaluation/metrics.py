"""Scores answers against the owner's labels. Works on data only and prints nothing."""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from task_manager.analyzer.thread import ThreadLine
from task_manager.analyzer.verdict import AssignmentReason, TaskVerdict, creates_draft

LABEL_MINE = 1
LABEL_MAYBE = 2
LABEL_OTHERS = 3
LABEL_NOT_TASK = 4
# Not a task by itself, but part of a conversation that leads to one. Later stages merge such
# messages into one draft, and the model can't see what comes next, so it's neither hit nor error.
LABEL_PART = 5

COPIED_WORDS = 5
_Z_95 = 1.96
_WORD = re.compile(r"[^\W_]+")
_PERSON_PLACEHOLDER = re.compile(r"<(?:PERSON|ORG|LOC|PHONE|EMAIL|URL|IP|ID)_\d+>")


@dataclass(frozen=True)
class Scored:
    message_id: int
    label: int
    stratum: str
    verdict: TaskVerdict | None
    error: str | None = None
    seconds: float = 0.0
    truncated: bool = False


@dataclass(frozen=True)
class Rate:
    hits: int
    total: int
    low: float
    high: float

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None


@dataclass(frozen=True)
class Metrics:
    precision: Rate
    recall: Rate
    weighted_precision: float | None
    weighted_recall: float | None
    maybe_label_destinations: dict[str, int]


def wilson(hits: int, total: int) -> Rate:
    """95% interval that stays honest on the small counts rare tasks produce."""
    if total == 0:
        return Rate(0, 0, 0.0, 1.0)
    share = hits / total
    z2 = _Z_95**2
    denominator = 1 + z2 / total
    centre = (share + z2 / (2 * total)) / denominator
    margin = _Z_95 * math.sqrt(share * (1 - share) / total + z2 / (4 * total**2)) / denominator
    return Rate(hits, total, max(0.0, centre - margin), min(1.0, centre + margin))


def is_draft(verdict: TaskVerdict | None, threshold: float) -> bool:
    """The rule the analyzer uses, plus a threshold to compare candidates."""
    return verdict is not None and creates_draft(verdict) and verdict.confidence >= threshold


def destination(verdict: TaskVerdict | None, threshold: float) -> str:
    if is_draft(verdict, threshold):
        return "draft"
    if (
        verdict is not None
        and verdict.is_task
        and (verdict.assigned_to_me or verdict.assignment_reason == AssignmentReason.COLLECTIVE)
    ):
        return "maybe"
    return "nothing"


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def score(rows: Sequence[Scored], threshold: float, population: dict[str, int]) -> Metrics:
    """A draft on a "maybe mine" message is an error, and missing one isn't.

    Parts of a larger task don't count either way. The sample over-represents messages with signs,
    so weighted numbers scale each group of signs back to its share of all messages.
    """
    drawn = Counter(row.stratum for row in rows)
    weights = {name: population.get(name, 0) / count for name, count in drawn.items()}

    rows = [row for row in rows if row.label != LABEL_PART]
    drafts = [row for row in rows if is_draft(row.verdict, threshold)]
    correct = [row for row in drafts if row.label == LABEL_MINE]
    mine = [row for row in rows if row.label == LABEL_MINE]

    def weigh(items: Iterable[Scored]) -> float:
        return sum(weights[row.stratum] for row in items)

    return Metrics(
        precision=wilson(len(correct), len(drafts)),
        recall=wilson(len(correct), len(mine)),
        weighted_precision=_ratio(weigh(correct), weigh(drafts)),
        weighted_recall=_ratio(weigh(correct), weigh(mine)),
        maybe_label_destinations=dict(
            Counter(destination(row.verdict, threshold) for row in rows if row.label == LABEL_MAYBE)
        ),
    )


def precision_by_reason(rows: Iterable[Scored], threshold: float) -> dict[str, Rate]:
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for row in rows:
        if row.label != LABEL_PART and is_draft(row.verdict, threshold):
            reason = row.verdict.assignment_reason.value
            totals[reason] += 1
            hits[reason] += row.label == LABEL_MINE
    return {reason: wilson(hits[reason], totals[reason]) for reason in sorted(totals)}


def percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(share * len(ordered)) - 1)]


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(_PERSON_PLACEHOLDER.sub(" ", text))]


def copies_source(description: str, source: Iterable[ThreadLine]) -> bool:
    """A run of words taken as is from the chat, which must not reach Jira."""
    words = _words(description)
    runs = {tuple(words[i : i + COPIED_WORDS]) for i in range(len(words) - COPIED_WORDS + 1)}
    for line in source:
        original = _words(line.text)
        for i in range(len(original) - COPIED_WORDS + 1):
            if tuple(original[i : i + COPIED_WORDS]) in runs:
                return True
    return False
