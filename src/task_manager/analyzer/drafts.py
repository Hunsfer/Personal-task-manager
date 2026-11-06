"""Drafts in the database: finding the one a message belongs to and storing the result."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.analyzer.classifier import Decision
from task_manager.analyzer.thread import ThreadLine, build_line
from task_manager.analyzer.verdict import has_placeholder
from task_manager.anonymizer.store import get_ready
from task_manager.storage.models import (
    Draft,
    DraftDecision,
    DraftKind,
    DraftMessage,
    DraftRole,
    DraftStatus,
    Message,
)

# A task is usually discussed within a day. Older drafts are left alone, so a later task on the same
# subject gets its own draft rather than disappearing into an old one.
MERGE_WINDOW = timedelta(hours=24)
# The model compares a new task with this many of the most recent drafts.
MAX_CANDIDATES = 5
UNDECIDED = (DraftStatus.NEW, DraftStatus.SENT)
REJECTED_ACTION = "rejected"


@dataclass(frozen=True)
class OpenDraft:
    id: int
    title: str
    description: str
    msg_ids: frozenset[int]
    message_ids: frozenset[int]
    status: DraftStatus = DraftStatus.NEW
    parent_id: int | None = None
    edited_by_owner: bool = False
    rejection: str | None = None

    @property
    def decided(self) -> bool:
        return self.status not in UNDECIDED


@dataclass(frozen=True)
class DraftUpdate:
    """What to write once the model has answered. Nothing is stored before that."""

    draft_id: int | None
    title: str
    description: str
    text_is_stale: bool
    kind: DraftKind = DraftKind.TASK
    parent_id: int | None = None
    # Messages a new ticket inherits from the rejected draft it brings back.
    inherited: frozenset[int] = frozenset()


async def open_drafts(session: AsyncSession, chat_id: int, around: datetime) -> list[OpenDraft]:
    """Drafts of this chat close enough in time to be the same conversation, newest first."""
    drafts = await session.scalars(
        select(Draft)
        .where(
            Draft.chat_id == chat_id,
            Draft.last_message_at >= around - MERGE_WINDOW,
            Draft.last_message_at <= around + MERGE_WINDOW,
        )
        .order_by(Draft.last_message_at.desc(), Draft.id.desc())
    )
    found = []
    for draft in drafts:
        rows = await session.execute(
            select(Message.id, Message.msg_id)
            .join(DraftMessage, DraftMessage.message_id == Message.id)
            .where(DraftMessage.draft_id == draft.id)
        )
        pairs = rows.all()
        found.append(
            OpenDraft(
                id=draft.id,
                title=draft.title,
                description=draft.description,
                msg_ids=frozenset(pair.msg_id for pair in pairs),
                message_ids=frozenset(pair.id for pair in pairs),
                status=draft.status,
                parent_id=draft.parent_draft_id,
                edited_by_owner=draft.edited_by_owner,
                rejection=(
                    await _rejection(session, draft.id)
                    if draft.status is DraftStatus.REJECTED
                    else None
                ),
            )
        )
    return found


async def _rejection(session: AsyncSession, draft_id: int) -> str | None:
    row = (
        await session.execute(
            select(DraftDecision.detail, DraftDecision.note)
            .where(DraftDecision.draft_id == draft_id, DraftDecision.decision == REJECTED_ACTION)
            .order_by(DraftDecision.id.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        return None
    return f"{row.detail}: {row.note}" if row.note else row.detail


def undecided_child(drafts: Iterable[OpenDraft], draft: OpenDraft) -> OpenDraft:
    """A decided draft may already have an open follow-up, which is where new messages belong."""
    if not draft.decided:
        return draft
    return next((d for d in drafts if d.parent_id == draft.id and not d.decided), draft)


def linked_draft(drafts: Iterable[OpenDraft], msg_ids: Iterable[int]) -> OpenDraft | None:
    """The most recent draft already holding one of these messages."""
    wanted = set(msg_ids)
    return next((draft for draft in drafts if draft.msg_ids & wanted), None)


def holding_draft(drafts: Iterable[OpenDraft], message_id: int) -> OpenDraft | None:
    """The draft this very message is already part of, which happens when it was edited."""
    return next((draft for draft in drafts if message_id in draft.message_ids), None)


async def draft_lines(session: AsyncSession, draft_id: int, rules: str) -> list[ThreadLine] | None:
    """A draft's messages for the model, or None if any isn't anonymized under current rules."""
    messages = await session.scalars(
        select(Message)
        .join(DraftMessage, DraftMessage.message_id == Message.id)
        .where(DraftMessage.draft_id == draft_id)
        .order_by(Message.sent_at, Message.id)
    )
    lines = []
    for message in messages:
        ready = await get_ready(session, message.id, rules)
        if ready is None:
            return None
        lines.append(build_line(message, ready))
    return lines


async def _message_ids(session: AsyncSession, chat_id: int, msg_ids: Iterable[int]) -> list[int]:
    rows = await session.scalars(
        select(Message.id).where(Message.chat_id == chat_id, Message.msg_id.in_(list(msg_ids)))
    )
    return list(rows)


async def _link_target(
    session: AsyncSession, draft_id: int, message_id: int, role: DraftRole
) -> None:
    """The latest decision on this message wins: after an edit it may no longer be a task."""
    link = await session.get(DraftMessage, (draft_id, message_id))
    if link is None:
        session.add(DraftMessage(draft_id=draft_id, message_id=message_id, role=role))
    else:
        link.role = role


async def _link_evidence(session: AsyncSession, draft_id: int, message_id: int) -> None:
    """Evidence never demotes a message that was itself judged to be a task."""
    link = await session.get(DraftMessage, (draft_id, message_id))
    if link is None:
        session.add(DraftMessage(draft_id=draft_id, message_id=message_id, role=DraftRole.EVIDENCE))
    elif link.role is DraftRole.CONTEXT:
        link.role = DraftRole.EVIDENCE


async def apply_update(
    session: AsyncSession,
    update: DraftUpdate,
    decision: Decision,
    message: Message,
    evidence: Iterable[int],
    model: str,
) -> int:
    """Create or extend a draft. The caller owns the transaction. Returns the draft's ID."""
    now = datetime.now(UTC)
    # A new ticket is about this message, even when it isn't a task by itself.
    is_trigger = decision.creates_draft or update.draft_id is None
    target_role = DraftRole.TRIGGER if is_trigger else DraftRole.CONTEXT
    marked = has_placeholder(update.title) or has_placeholder(update.description)

    if update.draft_id is None:
        draft = Draft(
            chat_id=message.chat_id,
            title=update.title,
            description=update.description,
            deadline=decision.deadline,
            priority=decision.priority,
            has_placeholders=marked,
            text_is_stale=False,
            kind=update.kind,
            parent_draft_id=update.parent_id,
            model=model,
            prompt_id=decision.prompt_id,
            last_message_at=message.sent_at,
            created_at=now,
            updated_at=now,
        )
        session.add(draft)
        await session.flush()
    else:
        draft = await session.get_one(Draft, update.draft_id)
        draft.title = update.title
        draft.description = update.description
        draft.has_placeholders = marked
        draft.text_is_stale = update.text_is_stale
        draft.updated_at = now
        # A draft already in the bot changes there too, silently.
        draft.needs_refresh = draft.status is DraftStatus.SENT
        draft.last_message_at = max(draft.last_message_at, message.sent_at)
        if decision.creates_draft:
            draft.deadline = decision.deadline or draft.deadline
            if decision.priority == "high":
                draft.priority = "high"

    await _link_target(session, draft.id, message.id, target_role)
    for message_id in update.inherited:
        if message_id != message.id:
            await _link_evidence(session, draft.id, message_id)
    if decision.creates_draft:
        for message_id in await _message_ids(session, message.chat_id, evidence):
            if message_id != message.id:
                await _link_evidence(session, draft.id, message_id)
    return draft.id
