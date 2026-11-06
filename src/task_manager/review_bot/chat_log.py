"""Remembers which messages are in the bot's chat, because Telegram can't list them for a bot."""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from task_manager.review_bot.messenger import Keyboard, MessageGone, Messenger, MessengerError
from task_manager.storage.models import BotChatMessage, Draft


class ChatLog:
    """A messenger that keeps a record of the chat. Only IDs are stored, never text."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], messenger: Messenger) -> None:
        self._sessions = sessions
        self._messenger = messenger

    async def remember(self, message_id: int) -> None:
        async with self._sessions() as session:
            await session.execute(
                insert(BotChatMessage)
                .values(message_id=message_id, created_at=datetime.now(UTC))
                .on_conflict_do_nothing(index_elements=[BotChatMessage.message_id])
            )
            await session.commit()

    async def _forget(self, message_id: int) -> None:
        async with self._sessions() as session:
            await session.execute(
                delete(BotChatMessage).where(BotChatMessage.message_id == message_id)
            )
            await session.commit()

    async def send(self, text: str, keyboard: Keyboard | None = None, silent: bool = False) -> int:
        message_id = await self._messenger.send(text, keyboard, silent)
        await self.remember(message_id)
        return message_id

    async def send_file(self, path: Path, name: str) -> int:
        message_id = await self._messenger.send_file(path, name)
        await self.remember(message_id)
        return message_id

    async def edit(self, message_id: int, text: str, keyboard: Keyboard | None = None) -> None:
        await self._messenger.edit(message_id, text, keyboard)

    async def delete(self, message_id: int) -> None:
        try:
            await self._messenger.delete(message_id)
        except MessageGone:
            await self._forget(message_id)
            raise
        await self._forget(message_id)

    async def clear(self) -> tuple[int, int]:
        """Delete everything in the chat. Returns how many went and how many were too old to.

        Drafts and their statuses stay. They only stop pointing at a message that's gone, so the
        next change shows them as a new message.
        """
        async with self._sessions() as session:
            ids = list(await session.scalars(select(BotChatMessage.message_id)))
        removed = too_old = 0
        for message_id in ids:
            try:
                await self.delete(message_id)
                removed += 1
            except MessageGone:
                too_old += 1
            except MessengerError:
                # Stays on the list for the next /clear.
                continue
        async with self._sessions() as session:
            await session.execute(update(Draft).values(bot_message_id=None))
            await session.commit()
        return removed, too_old
