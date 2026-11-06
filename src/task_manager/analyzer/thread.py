"""A message with its surroundings, the way the model sees it. Anonymized text only."""

from datetime import timedelta

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.anonymizer.store import ReadyMessage, get_ready
from task_manager.storage.models import Chat, ChatKind, Message

PREVIOUS_MESSAGES = 5
MAX_REPLY_DEPTH = 5
# Older messages are usually another conversation, however few came in between.
CONTEXT_WINDOW = timedelta(hours=12)


class ThreadLine(BaseModel):
    msg_id: int
    sender: str | None
    text: str
    sent_at: str
    reply_to: int | None = None
    files: list[str] = []


class Thread(BaseModel):
    chat_kind: ChatKind
    mentions_me: bool
    target: ThreadLine
    reply_chain: list[ThreadLine]
    previous: list[ThreadLine]

    def lines(self) -> list[ThreadLine]:
        return [*self.reply_chain, *self.previous, self.target]


def build_line(message: Message, ready: ReadyMessage) -> ThreadLine:
    return ThreadLine(
        msg_id=message.msg_id,
        sender=ready.sender,
        text=ready.text,
        sent_at=message.sent_at.isoformat(timespec="minutes"),
        reply_to=message.reply_to_msg_id,
        files=list(ready.file_names),
    )


async def build_thread(session: AsyncSession, message: Message, rules: str) -> Thread | None:
    """Everything the model reads about a message, or None if any part isn't ready.

    All of it passes through get_ready, so a context message without a current record blocks the
    whole thread instead of reaching the model in its original form.
    """
    target = await get_ready(session, message.id, rules)
    chat = await session.get(Chat, message.chat_id)
    if target is None or chat is None:
        return None

    chain: list[ThreadLine] = []
    seen = {message.msg_id}
    parent_id = message.reply_to_msg_id
    while parent_id is not None and parent_id not in seen and len(chain) < MAX_REPLY_DEPTH:
        seen.add(parent_id)
        parent = await session.scalar(
            select(Message).where(Message.chat_id == message.chat_id, Message.msg_id == parent_id)
        )
        # Replies to messages older than the collected history simply end the chain.
        if parent is None:
            break
        ready = await get_ready(session, parent.id, rules)
        if ready is None:
            return None
        chain.insert(0, build_line(parent, ready))
        parent_id = parent.reply_to_msg_id

    rows = await session.scalars(
        select(Message)
        .where(Message.chat_id == message.chat_id, Message.msg_id < message.msg_id)
        .where(Message.sent_at >= message.sent_at - CONTEXT_WINDOW)
        .order_by(Message.msg_id.desc())
        .limit(PREVIOUS_MESSAGES)
    )
    previous: list[ThreadLine] = []
    for row in reversed(list(rows)):
        if row.msg_id in seen:
            continue
        ready = await get_ready(session, row.id, rules)
        if ready is None:
            return None
        previous.append(build_line(row, ready))

    return Thread(
        chat_kind=chat.kind,
        mentions_me=message.mentions_me,
        target=build_line(message, target),
        reply_chain=chain,
        previous=previous,
    )
