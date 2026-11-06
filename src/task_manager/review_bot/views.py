"""Reads drafts for the bot. Real values appear only here, for showing to the owner."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.analyzer.drafts import REJECTED_ACTION
from task_manager.review_bot.render import (
    MAX_MESSAGE,
    REJECTION_REASONS,
    DraftCard,
    FileInfo,
    escape,
    extract_code,
    find_marks,
    format_time,
)
from task_manager.storage.attachments import HIDDEN_KINDS, goes_to_jira
from task_manager.storage.models import (
    Analysis,
    AnalysisOutcome,
    AnonymizedMessage,
    Attachment,
    Chat,
    Draft,
    DraftDecision,
    DraftMessage,
    DraftRole,
    Message,
    MessageStatus,
    Person,
    Pseudonym,
)

_ROLE_NAMES = {
    DraftRole.TRIGGER: "задача",
    DraftRole.EVIDENCE: "основание",
    DraftRole.CONTEXT: "уточнение",
}
# Only these can be a detection mistake. A contact, link or secret is matched by its form.
MARKABLE_KINDS = ("PERSON", "ORG", "LOC")
# Known by Telegram ID, so this is a real person whatever the name looks like.
KNOWN_PERSON_KEY = "user:"


@dataclass(frozen=True)
class Markable:
    id: int
    placeholder: str
    value: str


async def _values(session: AsyncSession, marks: list[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for mark in marks:
        values[mark] = None
        kind, _, number = mark.strip("<>").partition("_")
        if number:
            values[mark] = await session.scalar(
                select(Pseudonym.value).where(
                    Pseudonym.kind == kind, Pseudonym.number == int(number)
                )
            )
    return values


async def _draft_messages(session: AsyncSession, draft_id: int) -> list[tuple[Message, DraftRole]]:
    rows = await session.execute(
        select(Message, DraftMessage.role)
        .join(DraftMessage, DraftMessage.message_id == Message.id)
        .where(DraftMessage.draft_id == draft_id)
        .order_by(Message.sent_at, Message.id)
    )
    return [(row[0], row[1]) for row in rows.all()]


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
    reason = REJECTION_REASONS.get(row.detail, row.detail)
    return f"{reason}: {row.note}" if row.note else reason


async def load_card(session: AsyncSession, draft_id: int) -> DraftCard | None:
    draft = await session.get(Draft, draft_id, populate_existing=True)
    if draft is None:
        return None
    chat = await session.get(Chat, draft.chat_id)
    messages = await _draft_messages(session, draft_id)
    trigger = next((m for m, role in messages if role is DraftRole.TRIGGER), None)
    analysis = await session.get(Analysis, trigger.id) if trigger is not None else None
    files = await session.scalars(
        select(Attachment)
        .join(DraftMessage, DraftMessage.message_id == Attachment.message_id)
        .where(DraftMessage.draft_id == draft_id)
        .order_by(Attachment.id)
    )
    parent = await session.get(Draft, draft.parent_draft_id) if draft.parent_draft_id else None
    return DraftCard(
        deadline_by_owner=draft.deadline_by_owner,
        jira_key=draft.jira_key,
        jira_link=draft.jira_link,
        export_error=draft.export_error,
        parent_key=parent.jira_key if parent is not None else None,
        parent_status=parent.jira_status if parent is not None else None,
        parent_locked=parent.jira_locked if parent is not None else False,
        parent_waiting=(
            parent is not None and parent.export_action is not None and not parent.jira_key
        ),
        id=draft.id,
        status=draft.status,
        kind=draft.kind,
        parent_id=draft.parent_draft_id,
        title=draft.title,
        description=draft.description,
        deadline=draft.deadline,
        priority=draft.priority,
        reason=analysis.assignment_reason if analysis is not None else None,
        chat_title=chat.title if chat is not None else None,
        last_message_at=draft.last_message_at,
        message_count=len(messages),
        text_is_stale=draft.text_is_stale,
        rejection=await _rejection(session, draft_id),
        values=await _values(session, find_marks(f"{draft.title}\n{draft.description}")),
        code=tuple(
            snippet
            for message, _ in messages
            for snippet in extract_code(message.text, message.code_entities)
        ),
        files=tuple(
            FileInfo(
                f.id,
                f.kind,
                f.file_name,
                f.size_bytes,
                not goes_to_jira(f.kind, f.excluded, f.opted_in),
            )
            for f in files
            if f.kind not in HIDDEN_KINDS
        ),
    )


async def source_parts(session: AsyncSession, draft_id: int, utc_offset_hours: int) -> list[str]:
    """A draft's messages with real names, for the owner only. One HTML part per message."""
    parts = []
    for message, role in await _draft_messages(session, draft_id):
        sender = "Вы" if message.is_outgoing else "неизвестно"
        if not message.is_outgoing and message.sender_id is not None:
            person = await session.get(Person, message.sender_id)
            if person is not None:
                names = (person.first_name, person.last_name)
                sender = " ".join(n for n in names if n) or person.username or sender
        head = (
            f"<b>{_ROLE_NAMES[role]}</b> · {escape(sender)} · "
            f"{format_time(message.sent_at, utc_offset_hours)}"
        )
        if message.link:
            head += f' · <a href="{escape(message.link)}">открыть</a>'
        text = message.text or "(без текста)"
        # Room for the header. A very long message is still shown, only shortened.
        limit = MAX_MESSAGE - len(head) - 50
        body = escape(text if len(text) <= limit else f"{text[:limit]}…")
        parts.append(f"{head}\n{body}")
    return parts


async def markable(session: AsyncSession, draft_id: int) -> list[Markable]:
    """Placeholders in a draft and its messages that may be a detection mistake."""
    draft = await session.get(Draft, draft_id)
    if draft is None:
        return []
    records = await session.scalars(
        select(AnonymizedMessage.text)
        .join(DraftMessage, DraftMessage.message_id == AnonymizedMessage.message_id)
        .where(DraftMessage.draft_id == draft_id)
    )
    marks = find_marks("\n".join([draft.title, draft.description, *records]))
    found = []
    for mark in marks:
        kind, _, number = mark.strip("<>").partition("_")
        if kind not in MARKABLE_KINDS or not number:
            continue
        row = await session.scalar(
            select(Pseudonym).where(Pseudonym.kind == kind, Pseudonym.number == int(number))
        )
        if row is not None and not row.key.startswith(KNOWN_PERSON_KEY):
            found.append(Markable(row.id, mark, row.value))
    return found


async def queue_counters(session: AsyncSession) -> dict[str, int]:
    pending = await session.scalar(
        select(func.count()).select_from(Message).where(Message.status == MessageStatus.PENDING)
    )
    failed = await session.scalar(
        select(func.count()).select_from(Message).where(Message.status == MessageStatus.FAILED)
    )
    drafts = await session.execute(select(Draft.status, func.count()).group_by(Draft.status))
    analyses = await session.scalar(
        select(func.count()).select_from(Analysis).where(Analysis.outcome == AnalysisOutcome.DRAFT)
    )
    return {
        "pending": pending or 0,
        "failed": failed or 0,
        "task_decisions": analyses or 0,
        **{f"drafts_{status.value}": count for status, count in drafts.all()},
    }
