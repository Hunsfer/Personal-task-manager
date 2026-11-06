"""Sample of before/after pairs for checking anonymization on real messages by eye.

The file holds real correspondence, so it goes only where the owner points it. The console output
is counters only, which are safe to share.
"""

import random
import re
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.common.config import ConfigError
from task_manager.storage.models import AnonymizedMessage, Message

SAMPLE_SIZE = 150
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_HTML_SUFFIXES = {".html", ".htm"}
_PLACEHOLDER = re.compile(r"<([A-Z]+)(?:_\d+)?>")
_CAPITALIZED_MID_TEXT = re.compile(r"[^.!?\n]\s+[А-ЯЁ][а-яё]+")


def _fenced(text: str) -> list[str]:
    """A code block, so viewers show <ME> instead of hiding it as an HTML tag.

    The fence is longer than any run of backticks in the message, which can't then close it.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", text, fence]


def _entry(markdown: bool, sender: str | None, parts: list[tuple[str, str]]) -> list[str]:
    if not markdown:
        lines = [f"автор: {sender or '-'}"]
        lines += [f"{label}: {text}" for label, text in parts]
        return [*lines, ""]
    lines = [f"**Автор:** `{sender or '-'}`", ""]
    for label, text in parts:
        lines += [f"{label}:", *_fenced(text), ""]
    return [*lines, "---", ""]


async def write_preview(engine: AsyncEngine, path: Path) -> dict[str, int]:
    """Write the sample and return safe counters.

    Half the sample has replacements, to spot over-masking. The other half has none but contains
    capitalized words mid-sentence, which is where missed names are most likely.
    """
    if path.suffix.lower() in _HTML_SUFFIXES:
        raise ConfigError(
            f"{path}: use a .md or .txt file. Browsers treat <ME> and <SECRET> as HTML tags and "
            "silently hide them."
        )
    async with AsyncSession(engine) as session:
        rows = (
            await session.execute(
                select(Message.text, AnonymizedMessage.text, AnonymizedMessage.sender)
                .join(AnonymizedMessage, AnonymizedMessage.message_id == Message.id)
                .where(Message.text != "")
            )
        ).all()
        total = len((await session.scalars(select(Message.id))).all())

    counters: Counter[str] = Counter()
    replaced = []
    suspicious = []
    for original, anonymized, sender in rows:
        found = _PLACEHOLDER.findall(anonymized)
        counters.update(found)
        if found:
            replaced.append((original, anonymized, sender))
        elif _CAPITALIZED_MID_TEXT.search(original):
            suspicious.append((original, anonymized, sender))

    picker = random.Random()
    half = SAMPLE_SIZE // 2
    sample_a = picker.sample(replaced, min(half, len(replaced)))
    sample_b = picker.sample(suspicious, min(half, len(suspicious)))

    markdown = path.suffix.lower() in _MARKDOWN_SUFFIXES
    heading = "## " if markdown else "=== "
    lines = [
        "# Проверка обезличивания" if markdown else "ПРОВЕРКА ОБЕЗЛИЧИВАНИЯ",
        "",
        "Файл содержит настоящую переписку. Удалите его после просмотра.",
        "",
        "Автор показан так, как его увидит модель. Ваши сообщения подписаны как ME.",
        "",
        f"{heading}С заменами: нет ли лишних и всё ли заменено ({len(sample_a)})",
        "",
    ]
    for original, anonymized, sender in sample_a:
        lines += _entry(markdown, sender, [("было", original), ("стало", anonymized)])
    lines += [f"{heading}Без замен: не пропущены ли имена ({len(sample_b)})", ""]
    for original, _, sender in sample_b:
        lines += _entry(markdown, sender, [("текст", original)])
    path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "messages": total,
        "anonymized_with_text": len(rows),
        "with_replacements": len(replaced),
        "without_replacements_but_capitalized": len(suspicious),
        **{f"placeholder_{kind}": count for kind, count in sorted(counters.items())},
    }
