"""Collector logic: history load, catch-up after downtime, and realtime messages."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from task_manager.collector import store
from task_manager.collector.files import file_loop
from task_manager.collector.types import (
    IncomingMessage,
    NotAuthorizedError,
    RateLimitedError,
    Reader,
)
from task_manager.monitoring import state_loop
from task_manager.storage.db import current_revision
from task_manager.storage.models import Chat, MessageStatus

CATCH_UP_INTERVAL_S = 300.0
HEARTBEAT_INTERVAL_S = 10.0

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class Collector:
    def __init__(
        self,
        reader: Reader,
        sessions: async_sessionmaker[AsyncSession],
        chat_ids: Collection[int],
        history_depth_days: int,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._reader = reader
        self._sessions = sessions
        self._whitelist = frozenset(chat_ids)
        self._history_depth = timedelta(days=history_depth_days)
        self._sleep = sleep
        self._active: list[int] = []

    async def prepare(self) -> None:
        """Match the whitelist against the account's chats and start receiving messages."""
        chats = await self._with_rate_limit(self._reader.list_chats)
        found = [chat for chat in chats if chat.id in self._whitelist]
        for chat_id in self._whitelist - {chat.id for chat in found}:
            logger.warning("chat_not_found", extra={"chat_id": chat_id})

        async with self._sessions() as session:
            await store.sync_chats(session, found)
            await session.commit()

        self._active = [chat.id for chat in found]
        await self._sync_people()
        self._reader.subscribe(self._active, self.handle_new, self.handle_edited)
        logger.info("collector_ready", extra={"count": len(self._active)})

    async def _sync_people(self) -> None:
        """Learn who is in each chat, so the anonymizer can find their names.

        This covers people who are mentioned but never write. A chat that hides its members is
        skipped: its authors are still learned as their messages arrive.
        """
        me = await self._with_rate_limit(self._reader.me)
        async with self._sessions() as session:
            await store.upsert_people(session, [me], is_me=True)
            await session.commit()

        for chat_id in self._active:
            try:
                people = await self._with_rate_limit(
                    lambda chat_id=chat_id: self._reader.list_people(chat_id)
                )
                async with self._sessions() as session:
                    await store.upsert_people(session, people)
                    await session.commit()
            except NotAuthorizedError:
                raise
            except Exception:
                logger.exception("chat_people_failed", extra={"chat_id": chat_id})

    async def catch_up(self) -> None:
        """Bring every chat up to date. A failure in one chat doesn't stop the others."""
        for chat_id in self._active:
            try:
                await self._with_rate_limit(lambda chat_id=chat_id: self._catch_up_chat(chat_id))
            except NotAuthorizedError:
                raise
            except Exception:
                logger.exception("chat_catch_up_failed", extra={"chat_id": chat_id})

    async def handle_new(self, message: IncomingMessage) -> None:
        if message.chat_id not in self._whitelist:
            return
        # A message lost here is recovered by the next catch-up, because the cursor stays behind.
        try:
            async with self._sessions() as session:
                await store.save_message(session, message, MessageStatus.PENDING)
                await session.commit()
        except Exception:
            logger.exception("message_save_failed", extra={"chat_id": message.chat_id})

    async def handle_edited(self, message: IncomingMessage) -> None:
        if message.chat_id not in self._whitelist:
            return
        try:
            async with self._sessions() as session:
                await store.apply_edit(session, message)
                await session.commit()
        except Exception:
            logger.exception("message_edit_failed", extra={"chat_id": message.chat_id})

    async def _with_rate_limit[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Wait out Telegram's rate limits. Safe to retry: every write is idempotent."""
        while True:
            try:
                return await operation()
            except RateLimitedError as error:
                logger.warning("rate_limited", extra={"seconds": error.seconds})
                await self._sleep(error.seconds)

    async def _catch_up_chat(self, chat_id: int) -> None:
        async with self._sessions() as session:
            chat = await session.get(Chat, chat_id)
            loaded = chat is not None and chat.history_loaded_at is not None
        if not loaded:
            await self._load_history(chat_id)
        await self._read_new(chat_id)

    async def _load_history(self, chat_id: int) -> None:
        """Store recent history as archived, so old tasks don't flood the review queue."""
        now = datetime.now(UTC)
        # Anything newer than this is left for _read_new, so it isn't archived by mistake.
        latest = await self._reader.latest_msg_id(chat_id)
        count = 0
        async for message in self._reader.iter_since(chat_id, now - self._history_depth):
            if message.msg_id > latest:
                break
            async with self._sessions() as session:
                await store.save_message(session, message, MessageStatus.ARCHIVED)
                await session.commit()
            count += 1

        async with self._sessions() as session:
            await store.advance_cursor(session, chat_id, latest)
            await store.mark_history_loaded(session, chat_id, now)
            await session.commit()
        logger.info("history_loaded", extra={"chat_id": chat_id, "count": count})

    async def _read_new(self, chat_id: int) -> None:
        async with self._sessions() as session:
            chat = await session.get(Chat, chat_id)
            cursor = (chat.last_msg_id if chat else None) or 0

        async for message in self._reader.iter_after(chat_id, cursor):
            # One transaction: the cursor moves only if the message is stored.
            async with self._sessions() as session:
                await store.save_message(session, message, MessageStatus.PENDING)
                await store.advance_cursor(session, chat_id, message.msg_id)
                await session.commit()


async def _catch_up_loop(collector: Collector, reconnected: asyncio.Event) -> None:
    while True:
        await collector.catch_up()
        reconnected.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(reconnected.wait(), CATCH_UP_INTERVAL_S)


async def _heartbeat_loop(
    reader: Reader,
    engine: AsyncEngine,
    path: Path,
    reconnected: asyncio.Event,
    shutdown: asyncio.Event,
) -> None:
    """Report health only while Telegram is connected and the database is reachable.

    This runs separately from catch-up, so long history loads and rate-limit waits don't make
    the service look dead. It ends on the shutdown event rather than by cancellation: cancelling
    while a database connection is opening leaves aiosqlite's worker thread without an owner.
    """
    lost_connection = False
    ever_connected = False
    while not shutdown.is_set():
        if reader.is_connected():
            await current_revision(engine)
            path.touch()
            if lost_connection:
                logger.info("telegram_reconnected")
                reconnected.set()
            lost_connection = False
            ever_connected = True
        else:
            lost_connection = ever_connected
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(shutdown.wait(), HEARTBEAT_INTERVAL_S)


async def run_collector(
    reader: Reader,
    engine: AsyncEngine,
    chat_ids: Collection[int],
    history_depth_days: int,
    heartbeat: Path,
    stop: asyncio.Event,
    attachments_dir: Path | None = None,
) -> None:
    """Run until stopped. Raises NotAuthorizedError if the session is missing or revoked."""
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    collector = Collector(reader, sessions, chat_ids, history_depth_days)
    reconnected = asyncio.Event()
    shutdown = asyncio.Event()

    async def main() -> None:
        await reader.connect()
        if not await reader.is_authorized():
            raise NotAuthorizedError("no valid Telegram session")
        await collector.prepare()
        await _catch_up_loop(collector, reconnected)

    main_task = asyncio.create_task(main())
    stop_task = asyncio.create_task(stop.wait())
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(reader, engine, heartbeat, reconnected, shutdown)
    )
    state_task = asyncio.create_task(
        state_loop(
            sessions,
            "collector",
            shutdown,
            lambda: None if reader.is_connected() else "telegram_disconnected",
        )
    )
    tasks = [main_task, stop_task, heartbeat_task, state_task]
    if attachments_dir is not None:
        tasks.append(asyncio.create_task(file_loop(sessions, reader, attachments_dir, shutdown)))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        shutdown.set()
        main_task.cancel()
        stop_task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await reader.disconnect()
        heartbeat.unlink(missing_ok=True)
