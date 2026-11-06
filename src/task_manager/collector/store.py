"""Collector writes to the database. Callers own the transaction and commit it."""

from collections.abc import Iterable
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from task_manager.collector.types import AttachmentInfo, ChatInfo, IncomingMessage, PersonInfo
from task_manager.storage.models import (
    AnonymizedMessage,
    Attachment,
    Chat,
    Message,
    MessageStatus,
    Person,
)


async def sync_chats(session: AsyncSession, chats: Iterable[ChatInfo]) -> None:
    """Add new chats and refresh names. Cursors of known chats stay as they are."""
    for chat in chats:
        existing = await session.get(Chat, chat.id)
        if existing is None:
            session.add(Chat(chat_id=chat.id, title=chat.title, kind=chat.kind))
        else:
            existing.title = chat.title
            existing.kind = chat.kind


async def upsert_people(
    session: AsyncSession, people: Iterable[PersonInfo], is_me: bool = False
) -> None:
    """Add people or refresh their names. A person once marked as me stays marked."""
    for person in people:
        statement = insert(Person).values(
            user_id=person.user_id,
            first_name=person.first_name,
            last_name=person.last_name,
            username=person.username,
            is_me=is_me,
        )
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[Person.user_id],
                set_={
                    "first_name": statement.excluded.first_name,
                    "last_name": statement.excluded.last_name,
                    "username": statement.excluded.username,
                    "is_me": func.max(Person.is_me, statement.excluded.is_me),
                },
            )
        )


def _attachment_rows(attachments: Iterable[AttachmentInfo]) -> list[Attachment]:
    return [
        Attachment(kind=item.kind, file_name=item.file_name, size_bytes=item.size_bytes)
        for item in attachments
    ]


async def save_message(
    session: AsyncSession, message: IncomingMessage, status: MessageStatus
) -> bool:
    """Insert a message. A message that is already stored stays untouched: returns False."""
    message_id = await session.scalar(
        insert(Message)
        .values(
            chat_id=message.chat_id,
            msg_id=message.msg_id,
            reply_to_msg_id=message.reply_to_msg_id,
            sender_id=message.sender_id,
            sent_at=message.sent_at,
            edited_at=message.edited_at,
            text=message.text,
            mentions_me=message.mentions_me,
            is_outgoing=message.is_outgoing,
            link=message.link,
            code_entities=message.code_entities,
            status=status,
        )
        .on_conflict_do_nothing(index_elements=[Message.chat_id, Message.msg_id])
        .returning(Message.id)
    )
    if message.sender is not None:
        await upsert_people(session, [message.sender])
    if message_id is None:
        return False
    for attachment in _attachment_rows(message.attachments):
        attachment.message_id = message_id
        session.add(attachment)
    return True


async def apply_edit(session: AsyncSession, message: IncomingMessage) -> None:
    """Store an edit. Changed content goes back to the analysis queue, except archived history."""
    stored = await session.scalar(
        select(Message)
        .where(Message.chat_id == message.chat_id, Message.msg_id == message.msg_id)
        .options(selectinload(Message.attachments))
    )
    if stored is None:
        await save_message(session, message, MessageStatus.PENDING)
        return

    stored.edited_at = message.edited_at
    stored.code_entities = message.code_entities
    stored_attachments = {(a.kind, a.file_name, a.size_bytes) for a in stored.attachments}
    new_attachments = {(a.kind, a.file_name, a.size_bytes) for a in message.attachments}
    # Telegram also reports changes such as reactions as edits. Those aren't worth re-analysing.
    if stored.text == message.text and stored_attachments == new_attachments:
        return

    stored.text = message.text
    stored.attachments = _attachment_rows(message.attachments)
    # In the same transaction, so the old anonymized text can't outlive the change.
    await session.execute(
        delete(AnonymizedMessage).where(AnonymizedMessage.message_id == stored.id)
    )
    if stored.status is not MessageStatus.ARCHIVED:
        stored.status = MessageStatus.PENDING


async def advance_cursor(session: AsyncSession, chat_id: int, msg_id: int) -> None:
    """Move the cursor forward. It never moves back."""
    await session.execute(
        update(Chat)
        .where(Chat.chat_id == chat_id)
        .values(last_msg_id=func.max(func.coalesce(Chat.last_msg_id, 0), msg_id))
    )


async def mark_history_loaded(session: AsyncSession, chat_id: int, at: datetime) -> None:
    await session.execute(update(Chat).where(Chat.chat_id == chat_id).values(history_loaded_at=at))
