"""One-off operations around the queue. Console output is counters only."""

import asyncio
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.analyzer.drafts import draft_lines
from task_manager.analyzer.prompt import format_line
from task_manager.analyzer.runner import QueueWorker
from task_manager.storage.db import connect_raw
from task_manager.storage.models import (
    Analysis,
    Draft,
    DraftDecision,
    DraftMessage,
    DraftRole,
    Message,
    MessageStatus,
)

_ROLE_NAMES = {
    DraftRole.TRIGGER: "задача",
    DraftRole.EVIDENCE: "основание",
    DraftRole.CONTEXT: "уточнение",
}


async def retry_failed(engine: AsyncEngine) -> int:
    """Put failed messages back in the queue, for example after changing the prompt."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            update(Message)
            .where(Message.status == MessageStatus.FAILED)
            .values(status=MessageStatus.PENDING)
        )
        await session.commit()
    return result.rowcount


async def archive_unlisted(engine: AsyncEngine, chat_ids: list[int]) -> int:
    """Take waiting messages of chats off the whitelist out of the queue. Nothing is deleted."""
    async with AsyncSession(engine) as session:
        result = await session.execute(
            update(Message)
            .where(Message.status == MessageStatus.PENDING, Message.chat_id.not_in(chat_ids))
            .values(status=MessageStatus.ARCHIVED)
        )
        await session.commit()
    return result.rowcount


async def status_counts(engine: AsyncEngine) -> dict[str, int]:
    async with AsyncSession(engine) as session:
        rows = await session.execute(select(Message.status, func.count()).group_by(Message.status))
        return {status.value: count for status, count in rows.all()}


async def draft_counters(engine: AsyncEngine) -> dict[str, int]:
    """Safe to print: no text, no identifiers."""
    async with AsyncSession(engine) as session:
        outcomes = await session.execute(
            select(Analysis.outcome, Analysis.detail, func.count()).group_by(
                Analysis.outcome, Analysis.detail
            )
        )
        counters: Counter[str] = Counter()
        for outcome, detail, count in outcomes.all():
            counters[f"analysis_{outcome.value}"] += count
            if detail:
                counters[f"detail_{detail}"] += count
        reasons = await session.execute(
            select(Analysis.assignment_reason, func.count())
            .where(Analysis.outcome == "draft")
            .group_by(Analysis.assignment_reason)
        )
        for reason, count in reasons.all():
            counters[f"reason_{reason}"] = count

        rows = await session.execute(
            select(DraftMessage.draft_id, DraftMessage.role, func.count()).group_by(
                DraftMessage.draft_id, DraftMessage.role
            )
        )
        triggers: Counter[int] = Counter()
        sizes: Counter[int] = Counter()
        for draft_id, role, count in rows.all():
            sizes[draft_id] += count
            if role is DraftRole.TRIGGER:
                triggers[draft_id] += count
        drafts = list(await session.scalars(select(Draft)))
    counters["drafts"] = len(drafts)
    counters["drafts_merged"] = sum(count > 1 for count in triggers.values())
    counters["drafts_with_placeholders"] = sum(draft.has_placeholders for draft in drafts)
    counters["drafts_with_stale_text"] = sum(draft.text_is_stale for draft in drafts)
    counters["largest_draft_messages"] = max(sizes.values(), default=0)
    return dict(sorted(counters.items()))


async def feedback_export(engine: AsyncEngine, path: Path) -> dict[str, int]:
    """The owner's decisions as labels the evaluation can read. Only clear-cut ones count.

    A rejection for a reason such as "duplicate" says nothing about whether the message is a task.
    """
    async with AsyncSession(engine) as session:
        rows = await session.execute(
            select(DraftMessage.message_id, DraftDecision.decision, DraftDecision.detail)
            .join(DraftDecision, DraftDecision.draft_id == DraftMessage.draft_id)
            .where(
                DraftMessage.role == DraftRole.TRIGGER,
                DraftDecision.decision.in_(("approved", "rejected")),
            )
            .order_by(DraftDecision.id)
        )
        labels: dict[int, int] = {}
        for message_id, action, detail in rows.all():
            label = {"approved": 1, "not_mine": 3, "not_task": 4}.get(
                "approved" if action == "approved" else detail
            )
            if label is not None:
                labels[message_id] = label
    path.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    counts = Counter(labels.values())
    return {f"label_{label}": counts[label] for label in sorted(counts)}


def _fenced(text: str) -> str:
    fence = "`" * max(3, 1 + max(map(len, re.findall(r"`+", text)), default=0))
    return f"{fence}text\n{text}\n{fence}"


async def write_drafts_preview(engine: AsyncEngine, path: Path, rules: str) -> int:
    """A file for the owner: drafts next to their messages, in code blocks to keep <ME> visible."""
    parts = [
        "# Черновики задач",
        "",
        "Файл содержит обезличенную переписку. Удалите его после просмотра.",
        "",
        "Понятен ли черновик без чата? Верно ли объединены сообщения? Не склеены ли разные задачи?",
        "",
    ]
    async with AsyncSession(engine) as session:
        drafts = list(await session.scalars(select(Draft).order_by(Draft.last_message_at)))
        for number, draft in enumerate(drafts, 1):
            lines = await draft_lines(session, draft.id, rules)
            rows = await session.execute(
                select(Message.msg_id, DraftMessage.role)
                .join(DraftMessage, DraftMessage.message_id == Message.id)
                .where(DraftMessage.draft_id == draft.id)
            )
            roles = {msg_id: role for msg_id, role in rows.all()}
            notes = [f"приоритет {draft.priority}"]
            if draft.deadline:
                notes.append(f"срок (догадка модели) {draft.deadline}")
            if draft.has_placeholders:
                notes.append("ЕСТЬ ПЛЕЙСХОЛДЕРЫ")
            if draft.text_is_stale:
                notes.append("ТЕКСТ НЕ ОБНОВЛЁН")
            source = (
                "\n\n".join(
                    f"({_ROLE_NAMES[roles[line.msg_id]]}) {format_line(line)}" for line in lines
                )
                if lines is not None
                else "(сообщения ещё обезличиваются)"
            )
            parts += [
                f"## {number}. Сообщений: {len(roles)}",
                "",
                ", ".join(notes),
                "",
                _fenced(f"{draft.title}\n\n{draft.description}"),
                "",
                _fenced(source),
                "",
            ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return len(drafts)


def make_replay_copy(
    source: Path, target: Path, chat_ids: list[int], days: int | None = None
) -> int:
    """A private copy where history waits in the queue. The original is only read.

    Returns how many messages the replay will handle.
    """
    original = connect_raw(source, read_only=True)
    copy = connect_raw(target)
    try:
        original.backup(copy)
        for table in ("attachment_uploads", "file_requests", "bot_notices", "draft_decisions"):
            copy.execute(f"DELETE FROM {table}")
        copy.execute("DELETE FROM draft_messages")
        copy.execute("DELETE FROM drafts")
        copy.execute("DELETE FROM analyses")
        copy.execute("UPDATE messages SET status = 'done' WHERE status != 'archived'")
        marks = ",".join("?" * len(chat_ids))
        query = f"UPDATE messages SET status = 'pending' WHERE chat_id IN ({marks})"
        parameters: list[object] = list(chat_ids)
        if days is not None:
            since = datetime.now(UTC) - timedelta(days=days)
            query += " AND sent_at >= ?"
            parameters.append(since.strftime("%Y-%m-%d %H:%M:%S"))
        count = copy.execute(query + " AND status = 'archived'", parameters).rowcount
        copy.commit()
    finally:
        copy.close()
        original.close()
    return count


def remove_replay_copy(target: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{target}{suffix}").unlink(missing_ok=True)


async def run_replay(worker: QueueWorker) -> Counter[str]:
    """Process the whole copy. An outage raises, so a partial replay isn't taken for a result."""
    return await worker.drain(asyncio.Event())
