"""What a task looks like in Jira: real values back in the text, in Atlassian's document format."""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.anonymizer.store import restore

ME_MARK = "<ME>"
ME_TEXT = "я"
_PARAGRAPHS = re.compile(r"\n\s*\n")

# Limits from Jira for a summary. Longer text is cut rather than refused.
MAX_SUMMARY = 255


@dataclass(frozen=True)
class PreparedText:
    text: str
    unknown: tuple[str, ...]


async def prepare_text(session: AsyncSession, text: str) -> PreparedText:
    """Placeholders become real values only here: the owner saw them when approving."""
    restored, unknown = await restore(session, text)
    return PreparedText(restored.replace(ME_MARK, ME_TEXT), tuple(unknown))


def summary(text: str) -> str:
    single_line = " ".join(text.split())
    return single_line[:MAX_SUMMARY]


def _paragraph(text: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for index, line in enumerate(text.split("\n")):
        if index:
            content.append({"type": "hardBreak"})
        if line:
            content.append({"type": "text", "text": line})
    return {"type": "paragraph", "content": content}


def text_nodes(text: str) -> list[dict[str, Any]]:
    """Empty text nodes are invalid in this format, so empty paragraphs are left out."""
    return [_paragraph(part.strip()) for part in _PARAGRAPHS.split(text) if part.strip()]


def code_node(code: str) -> dict[str, Any]:
    return {"type": "codeBlock", "content": [{"type": "text", "text": code}]} if code else {}


def heading_node(text: str) -> dict[str, Any]:
    return {"type": "heading", "attrs": {"level": 3}, "content": [{"type": "text", "text": text}]}


def document(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "doc", "version": 1, "content": [node for node in nodes if node]}


def extended_document(
    existing: dict[str, Any] | None, nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keeps everything already in the description: someone may have edited it in Jira."""
    content = list((existing or {}).get("content", []))
    return document([*content, *nodes])


def mentions(existing: dict[str, Any] | None, marker: str) -> bool:
    """Whether a heading was already added, so a retry after a crash doesn't add it twice."""
    return marker in repr(existing or {})
