"""Divides labelled examples into a tuning part and a final check.

Examples that share any message stay together: otherwise the final check would contain text the
prompt was tuned on, and its numbers would look better than they are.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Literal

from task_manager.evaluation.sampling import Example

FINAL_SHARE = 0.3
SEED = 20261104
Part = Literal["dev", "final"]


def group_examples(examples: Sequence[Example]) -> list[list[Example]]:
    parent = list(range(len(examples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner: dict[tuple[int, int], int] = {}
    for index, example in enumerate(examples):
        for line in example.thread.lines():
            key = (example.chat_ref, line.msg_id)
            if key in owner:
                parent[find(index)] = find(owner[key])
            else:
                owner[key] = index

    groups: dict[int, list[Example]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[find(index)].append(example)
    return list(groups.values())


def _keys(example: Example, labels: dict[int, int]) -> list[tuple[str, object]]:
    return [("label", labels[example.message_id]), ("stratum", example.stratum), ("size", "all")]


def split_examples(
    examples: Sequence[Example], labels: dict[int, int], seed: int = SEED
) -> dict[int, Part]:
    """Largest groups first, each to the part that keeps labels and signs closest to target."""
    labelled = [example for example in examples if example.message_id in labels]
    targets: Counter[tuple[str, object]] = Counter()
    for example in labelled:
        targets.update(_keys(example, labels))

    groups = group_examples(labelled)
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)

    def error(counts: Counter[tuple[str, object]]) -> float:
        return sum((FINAL_SHARE * targets[key] - counts[key]) ** 2 for key in targets)

    group_keys = [
        Counter(key for example in group for key in _keys(example, labels)) for group in groups
    ]
    in_final = [False] * len(groups)
    final_counts: Counter[tuple[str, object]] = Counter()
    # Moving one group at a time can only lower the error, so this always settles.
    changed = True
    while changed:
        changed = False
        for index, keys in enumerate(group_keys):
            moved = final_counts - keys if in_final[index] else final_counts + keys
            if error(moved) < error(final_counts):
                final_counts = moved
                in_final[index] = not in_final[index]
                changed = True

    parts: dict[int, Part] = {}
    for group, final in zip(groups, in_final, strict=True):
        for example in group:
            parts[example.message_id] = "final" if final else "dev"
    return parts
