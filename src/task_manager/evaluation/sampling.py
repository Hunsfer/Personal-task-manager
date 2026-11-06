"""Chooses the messages the owner labels. Anonymized text only, through build_thread."""

import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.analyzer.signals import ME_TOKEN, NO_SIGNAL, SignalDetector
from task_manager.analyzer.thread import Thread, build_thread
from task_manager.anonymizer.runner import live_rules
from task_manager.common.config import AnonymizerConfig, MeConfig, ScopeConfig
from task_manager.storage.models import Message, MessageStatus

SEED = 20261104
RANDOM_SIZE = 100
# Tasks are rare, so a purely random sample would hold too few of them. These quotas enrich the
# sample without letting direct messages crowd out the rest. Unused places are shared out.
SIGNAL_QUOTAS = {
    "mention": 50,
    "reply": 20,
    "dm": 50,
    "section": 35,
    "role": 10,
    "collective": 15,
    "request": 20,
}
SAMPLE_FILE = "sample.json"

OWN_SEED = SEED + 1
OWN_RANDOM_SIZE = 20
# Promises are rare among the owner's messages, so a plain random draw would hold too few.
OWN_QUOTAS = {"promise_word": 15, "reply_to_other": 15}
_OWN_OTHER = "other"
_PROMISE_WORDS = re.compile(
    r"\b(сделаю|посмотрю|гляну|поправлю|исправлю|доделаю|переделаю|добавлю|сверстаю|проверю|"
    r"займусь|возьму|возьмусь|сделаем|посмотрим|поправим)\b",
    re.I,
)


class Example(BaseModel):
    message_id: int
    chat_ref: int
    stratum: str
    signals: list[str]
    thread: Thread


class Sample(BaseModel):
    seed: int
    population: dict[str, int]
    skipped: int
    examples: list[Example]


async def collect_examples(
    engine: AsyncEngine,
    chat_ids: Iterable[int],
    me: MeConfig,
    scope: ScopeConfig,
    config: AnonymizerConfig,
    own: bool = False,
) -> tuple[list[Example], int]:
    """Every message eligible for labelling, and how many had to be skipped as not ready.

    Only archived messages with text: incoming ones, or the owner's own with `own`, which are judged
    by a different prompt. The database may still hold a chat removed from the whitelist, which must
    not be analyzed any more.
    """
    detector = SignalDetector(scope)
    async with AsyncSession(engine) as session:
        rules = await live_rules(session, me, scope, config)
        rows = await session.scalars(
            select(Message)
            .where(
                Message.status == MessageStatus.ARCHIVED,
                Message.is_outgoing.is_(own),
                Message.text != "",
                Message.chat_id.in_(list(chat_ids)),
            )
            .order_by(Message.id)
        )
        messages = list(rows)
        chat_refs = {
            chat_id: ref for ref, chat_id in enumerate(sorted({m.chat_id for m in messages}))
        }

        examples: list[Example] = []
        skipped = 0
        for message in messages:
            thread = await build_thread(session, message, rules)
            if thread is None:
                skipped += 1
                continue
            if (thread.target.sender == ME_TOKEN) != own or not thread.target.text.strip():
                continue
            signals = _own_signals(thread) if own else detector.detect(thread)
            examples.append(
                Example(
                    message_id=message.id,
                    chat_ref=chat_refs[message.chat_id],
                    stratum=signals[0] if signals else NO_SIGNAL,
                    signals=signals,
                    thread=thread,
                )
            )
    return examples, skipped


def _own_signals(thread: Thread) -> list[str]:
    if _PROMISE_WORDS.search(thread.target.text):
        return ["promise_word"]
    parent = next(
        (line for line in thread.reply_chain if line.msg_id == thread.target.reply_to), None
    )
    if parent is not None and parent.sender != ME_TOKEN:
        return ["reply_to_other"]
    return [_OWN_OTHER]


def select_sample(
    examples: list[Example],
    skipped: int,
    seed: int = SEED,
    random_size: int = RANDOM_SIZE,
    quotas: dict[str, int] = SIGNAL_QUOTAS,
) -> Sample:
    """A uniform draw from everything, then uniform draws within each group of signs.

    Within a group every message has the same chance, so results can be weighted back to the whole
    population by group size.
    """
    picker = random.Random(seed)
    ordered = sorted(examples, key=lambda example: example.message_id)
    chosen = picker.sample(ordered, min(random_size, len(ordered)))
    chosen_ids = {example.message_id for example in chosen}

    pools = {
        stratum: [e for e in ordered if e.stratum == stratum and e.message_id not in chosen_ids]
        for stratum in quotas
    }
    for pool in pools.values():
        picker.shuffle(pool)

    for stratum, quota in quotas.items():
        chosen.extend(pools[stratum][:quota])
        del pools[stratum][:quota]
    remaining = sum(quotas.values()) - (len(chosen) - min(random_size, len(ordered)))
    while remaining > 0 and any(pools.values()):
        for pool in pools.values():
            if pool and remaining > 0:
                chosen.append(pool.pop())
                remaining -= 1

    picker.shuffle(chosen)
    return Sample(
        seed=seed,
        population=dict(Counter(example.stratum for example in ordered)),
        skipped=skipped,
        examples=chosen,
    )


def refresh_sample(existing: Sample, examples: list[Example], skipped: int) -> Sample:
    """The same messages with current text, minus those no longer eligible. Labels stay valid."""
    current = {example.message_id: example for example in examples}
    return Sample(
        seed=existing.seed,
        population=dict(Counter(example.stratum for example in examples)),
        skipped=skipped,
        examples=[
            current[example.message_id]
            for example in existing.examples
            if example.message_id in current
        ],
    )


def write_sample(sample: Sample, directory: Path) -> Path:
    path = directory / SAMPLE_FILE
    path.write_text(sample.model_dump_json(indent=1), encoding="utf-8")
    return path


def read_sample(directory: Path) -> Sample:
    return Sample.model_validate_json((directory / SAMPLE_FILE).read_text(encoding="utf-8"))


def sample_counters(sample: Sample) -> dict[str, int]:
    """Safe to print: no text, no identifiers."""
    drawn = Counter(example.stratum for example in sample.examples)
    counters = {"sampled": len(sample.examples), "skipped_not_ready": sample.skipped}
    for stratum in sorted(sample.population):
        counters[f"population_{stratum}"] = sample.population[stratum]
        counters[f"sampled_{stratum}"] = drawn[stratum]
    return counters
