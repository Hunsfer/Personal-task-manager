import asyncio
import io
import logging
import threading
import time
from collections.abc import AsyncIterator, Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from task_manager.collector import runner, store
from task_manager.collector.runner import Collector, run_collector
from task_manager.collector.types import (
    AttachmentInfo,
    ChatInfo,
    IncomingMessage,
    MessageHandler,
    NotAuthorizedError,
    PersonInfo,
    RateLimitedError,
)
from task_manager.common.log import SafeJsonFormatter
from task_manager.storage.db import make_engine
from task_manager.storage.models import Chat, ChatKind, Message, MessageStatus, Person

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
CHAT = -100
OTHER_CHAT = -200
OUTSIDE_CHAT = -999
SECRET = "Иван Петров, позвони на +79001234567"


def now() -> datetime:
    return datetime.now(UTC)


def incoming(msg_id: int, chat_id: int = CHAT, age_days: int = 0, text: str = "текст", **fields):
    return IncomingMessage(
        chat_id=chat_id,
        msg_id=msg_id,
        sent_at=now() - timedelta(days=age_days),
        text=text,
        **fields,
    )


class FakeReader:
    def __init__(self, messages: Collection[IncomingMessage] = ()) -> None:
        self.messages = list(messages)
        self.chats = [
            ChatInfo(CHAT, "Чат", ChatKind.SUPERGROUP),
            ChatInfo(OTHER_CHAT, "Другой", ChatKind.PRIVATE),
            ChatInfo(OUTSIDE_CHAT, "Не в списке", ChatKind.GROUP),
        ]
        self.rate_limits: list[int] = []
        self.connected = True
        self.authorized = True
        self.disconnected = False
        self.subscribed: set[int] = set()
        self.fail_after: int | None = None
        self.me_info = PersonInfo(99, "Я", None, "me_name")
        self.people: dict[int, list[PersonInfo]] = {}
        self.people_requests: list[int] = []
        self.hidden_chats: set[int] = set()

    def _limit(self) -> None:
        if self.rate_limits:
            raise RateLimitedError(self.rate_limits.pop(0))

    async def connect(self) -> None: ...

    async def disconnect(self) -> None:
        self.disconnected = True

    def is_connected(self) -> bool:
        return self.connected

    async def is_authorized(self) -> bool:
        return self.authorized

    async def me(self) -> PersonInfo:
        return self.me_info

    async def list_people(self, chat_id: int) -> list[PersonInfo]:
        self.people_requests.append(chat_id)
        if chat_id in self.hidden_chats:
            raise PermissionError("members are hidden")
        return self.people.get(chat_id, [])

    async def list_chats(self) -> list[ChatInfo]:
        self._limit()
        return self.chats

    async def latest_msg_id(self, chat_id: int) -> int:
        self._limit()
        ids = [m.msg_id for m in self.messages if m.chat_id == chat_id]
        return max(ids, default=0)

    async def iter_after(self, chat_id: int, msg_id: int) -> AsyncIterator[IncomingMessage]:
        for message in sorted(self.messages, key=lambda m: m.msg_id):
            if message.chat_id == chat_id and message.msg_id > msg_id:
                yield message

    async def iter_since(self, chat_id: int, since: datetime) -> AsyncIterator[IncomingMessage]:
        served = 0
        for message in sorted(self.messages, key=lambda m: m.msg_id):
            if message.chat_id == chat_id and message.sent_at >= since:
                if self.fail_after is not None and served == self.fail_after:
                    self.fail_after = None
                    raise ConnectionError("connection lost")
                served += 1
                yield message

    def subscribe(
        self, chat_ids: Collection[int], on_new: MessageHandler, on_edited: MessageHandler
    ) -> None:
        self.subscribed = set(chat_ids)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return make_engine(path)


def make_collector(engine: AsyncEngine, reader: FakeReader, sleeps: list[float] | None = None):
    async def fake_sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return Collector(
        reader,
        async_sessionmaker(engine, expire_on_commit=False),
        chat_ids=[CHAT, OTHER_CHAT],
        history_depth_days=30,
        sleep=fake_sleep,
    )


async def stored_messages(engine: AsyncEngine) -> dict[int, Message]:
    async with AsyncSession(engine) as session:
        rows = await session.scalars(select(Message).options(selectinload(Message.attachments)))
        return {row.msg_id: row for row in rows}


async def get_chat(engine: AsyncEngine, chat_id: int = CHAT) -> Chat:
    async with AsyncSession(engine) as session:
        chat = await session.get(Chat, chat_id)
        assert chat is not None
        return chat


def run(coroutine):
    async def wrapper():
        return await coroutine

    return asyncio.run(wrapper())


def test_people_are_learned_from_whitelisted_chats_only(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader()
        reader.people = {
            CHAT: [PersonInfo(1, "Ольга", "Смирнова", "olga")],
            OUTSIDE_CHAT: [PersonInfo(2, "Чужой")],
        }
        await make_collector(engine, reader).prepare()

        assert OUTSIDE_CHAT not in reader.people_requests
        async with AsyncSession(engine) as session:
            people = {row.user_id: row for row in await session.scalars(select(Person))}
        assert set(people) == {1, 99}
        assert people[1].first_name == "Ольга"
        assert people[1].is_me is False
        assert people[99].is_me is True
        await engine.dispose()

    run(scenario())


def test_chat_hiding_its_members_does_not_stop_the_collector(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader()
        reader.hidden_chats = {CHAT}
        reader.people = {OTHER_CHAT: [PersonInfo(3, "Дмитрий")]}
        collector = make_collector(engine, reader)
        await collector.prepare()
        assert reader.subscribed == {CHAT, OTHER_CHAT}
        async with AsyncSession(engine) as session:
            assert await session.get(Person, 3) is not None
        await engine.dispose()

    run(scenario())


def test_message_author_is_learned_and_renames_are_tracked(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1, sender=PersonInfo(5, "Вася")))
        await collector.handle_new(incoming(2, sender=PersonInfo(5, "Василий", "Новый")))
        async with AsyncSession(engine) as session:
            person = await session.get(Person, 5)
        assert (person.first_name, person.last_name) == ("Василий", "Новый")
        await engine.dispose()

    run(scenario())


def test_me_stays_marked_when_seen_as_an_author(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1, sender=PersonInfo(99, "Я", None, "me_name")))
        async with AsyncSession(engine) as session:
            assert (await session.get(Person, 99)).is_me is True
        await engine.dispose()

    run(scenario())


def test_only_whitelisted_chats_are_synced_and_subscribed(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader()
        await make_collector(engine, reader).prepare()
        assert reader.subscribed == {CHAT, OTHER_CHAT}
        chat = await get_chat(engine)
        assert (chat.title, chat.kind) == ("Чат", ChatKind.SUPERGROUP)
        await engine.dispose()

    run(scenario())


def test_message_outside_the_whitelist_is_dropped(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1, chat_id=OUTSIDE_CHAT))
        await collector.handle_edited(incoming(2, chat_id=OUTSIDE_CHAT))
        assert await stored_messages(engine) == {}
        await engine.dispose()

    run(scenario())


def test_realtime_message_is_stored_with_all_fields(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        attachment = AttachmentInfo("photo", "a.jpg", 10)
        await collector.handle_new(
            incoming(
                1,
                reply_to_msg_id=9,
                sender_id=5,
                mentions_me=True,
                is_outgoing=False,
                link="https://t.me/c/1/1",
                attachments=(attachment,),
            )
        )
        message = (await stored_messages(engine))[1]
        assert message.status is MessageStatus.PENDING
        assert (message.reply_to_msg_id, message.sender_id) == (9, 5)
        assert message.mentions_me is True
        assert message.link == "https://t.me/c/1/1"
        assert [(a.kind, a.file_name, a.size_bytes) for a in message.attachments] == [
            ("photo", "a.jpg", 10)
        ]
        await engine.dispose()

    run(scenario())


def test_same_message_twice_keeps_one_row_and_its_status(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1, attachments=(AttachmentInfo("photo", None, 1),)))
        async with AsyncSession(engine) as session:
            stored = await session.scalar(select(Message))
            stored.status = MessageStatus.DONE
            await session.commit()

        await collector.handle_new(incoming(1, attachments=(AttachmentInfo("photo", None, 1),)))
        messages = await stored_messages(engine)
        assert len(messages) == 1
        assert messages[1].status is MessageStatus.DONE
        assert len(messages[1].attachments) == 1
        await engine.dispose()

    run(scenario())


def test_first_read_archives_recent_history_only(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=40), incoming(2, age_days=5), incoming(3)])
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        messages = await stored_messages(engine)
        assert set(messages) == {2, 3}
        assert {m.status for m in messages.values()} == {MessageStatus.ARCHIVED}
        chat = await get_chat(engine)
        assert chat.last_msg_id == 3
        assert chat.history_loaded_at is not None
        await engine.dispose()

    run(scenario())


def test_messages_after_history_load_enter_the_queue(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=5)])
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        reader.messages.append(incoming(2))
        await collector.catch_up()

        messages = await stored_messages(engine)
        assert messages[1].status is MessageStatus.ARCHIVED
        assert messages[2].status is MessageStatus.PENDING
        assert (await get_chat(engine)).last_msg_id == 2
        await engine.dispose()

    run(scenario())


def test_message_arriving_during_history_load_is_not_archived(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=5)])
        original = reader.latest_msg_id

        async def latest_then_new_message(chat_id: int) -> int:
            latest = await original(chat_id)
            if chat_id == CHAT:
                reader.messages.append(incoming(2))
            return latest

        reader.latest_msg_id = latest_then_new_message
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        messages = await stored_messages(engine)
        assert messages[1].status is MessageStatus.ARCHIVED
        assert messages[2].status is MessageStatus.PENDING
        await engine.dispose()

    run(scenario())


def test_quiet_chat_does_not_pull_its_whole_history_into_the_queue(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=100), incoming(2, age_days=90)])
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        assert await stored_messages(engine) == {}
        assert (await get_chat(engine)).last_msg_id == 2
        await engine.dispose()

    run(scenario())


def test_interrupted_history_load_starts_over_without_duplicates(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(i, age_days=5) for i in (1, 2, 3)])
        reader.fail_after = 2
        collector = make_collector(engine, reader)
        await collector.prepare()

        await collector.catch_up()
        assert (await get_chat(engine)).history_loaded_at is None
        assert set(await stored_messages(engine)) == {1, 2}

        await collector.catch_up()
        assert (await get_chat(engine)).history_loaded_at is not None
        messages = await stored_messages(engine)
        assert set(messages) == {1, 2, 3}
        assert {m.status for m in messages.values()} == {MessageStatus.ARCHIVED}
        await engine.dispose()

    run(scenario())


def test_failed_write_keeps_the_cursor_and_is_recovered(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        reader = FakeReader()
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        reader.messages += [incoming(1), incoming(2)]
        save = store.save_message

        async def failing_save(session, message, status):
            if message.msg_id == 2:
                raise OSError("disk full")
            return await save(session, message, status)

        monkeypatch.setattr(store, "save_message", failing_save)
        await collector.catch_up()
        assert (await get_chat(engine)).last_msg_id == 1

        monkeypatch.setattr(store, "save_message", save)
        await collector.catch_up()
        assert (await get_chat(engine)).last_msg_id == 2
        assert set(await stored_messages(engine)) == {1, 2}
        await engine.dispose()

    run(scenario())


def test_realtime_message_does_not_hide_a_gap_before_it(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader()
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        # Message 1 was missed, and message 2 arrived in real time.
        reader.messages += [incoming(1), incoming(2)]
        await collector.handle_new(incoming(2))
        assert not (await get_chat(engine)).last_msg_id

        await collector.catch_up()
        assert set(await stored_messages(engine)) == {1, 2}
        await engine.dispose()

    run(scenario())


def test_failure_in_one_chat_does_not_stop_the_others(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=1), incoming(2, OTHER_CHAT, age_days=1)])
        collector = make_collector(engine, reader)
        await collector.prepare()
        original = reader.latest_msg_id

        async def failing_latest(chat_id: int) -> int:
            if chat_id == CHAT:
                raise ConnectionError("chat unavailable")
            return await original(chat_id)

        reader.latest_msg_id = failing_latest
        await collector.catch_up()
        assert set(await stored_messages(engine)) == {2}
        await engine.dispose()

    run(scenario())


def test_edit_updates_text_and_requeues(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1))
        async with AsyncSession(engine) as session:
            (await session.scalar(select(Message))).status = MessageStatus.DONE
            await session.commit()

        edited_at = now()
        await collector.handle_edited(incoming(1, text="новый текст", edited_at=edited_at))
        message = (await stored_messages(engine))[1]
        assert message.text == "новый текст"
        assert message.edited_at is not None
        assert message.status is MessageStatus.PENDING
        await engine.dispose()

    run(scenario())


def test_edit_replaces_attachments(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1, attachments=(AttachmentInfo("photo", "a", 1),)))
        await collector.handle_edited(incoming(1, attachments=(AttachmentInfo("video", "b", 2),)))
        message = (await stored_messages(engine))[1]
        assert [(a.kind, a.file_name) for a in message.attachments] == [("video", "b")]
        await engine.dispose()

    run(scenario())


def test_edit_of_archived_message_stays_archived(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=1)])
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()

        await collector.handle_edited(incoming(1, text="правка"))
        message = (await stored_messages(engine))[1]
        assert message.text == "правка"
        assert message.status is MessageStatus.ARCHIVED
        await engine.dispose()

    run(scenario())


def test_edit_without_content_change_is_not_requeued(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_new(incoming(1))
        async with AsyncSession(engine) as session:
            (await session.scalar(select(Message))).status = MessageStatus.DONE
            await session.commit()

        await collector.handle_edited(incoming(1))
        assert (await stored_messages(engine))[1].status is MessageStatus.DONE
        await engine.dispose()

    run(scenario())


def test_edit_of_unknown_message_is_stored(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        collector = make_collector(engine, FakeReader())
        await collector.prepare()
        await collector.handle_edited(incoming(1, text="правка"))
        assert (await stored_messages(engine))[1].status is MessageStatus.PENDING
        await engine.dispose()

    run(scenario())


def test_rate_limit_waits_and_retries(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=1)])
        reader.rate_limits = [120]
        sleeps: list[float] = []
        collector = make_collector(engine, reader, sleeps)
        await collector.prepare()
        assert sleeps == [120]

        reader.rate_limits = [300, 60]
        await collector.catch_up()
        assert sleeps == [120, 300, 60]
        assert set(await stored_messages(engine)) == {1}
        await engine.dispose()

    run(scenario())


def test_revoked_session_stops_catch_up(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        reader = FakeReader()
        collector = make_collector(engine, reader)
        await collector.prepare()

        async def revoked(_chat_id: int) -> int:
            raise NotAuthorizedError("revoked")

        reader.latest_msg_id = revoked
        with pytest.raises(NotAuthorizedError):
            await collector.catch_up()
        await engine.dispose()

    run(scenario())


def test_message_text_stays_out_of_the_logs(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(SafeJsonFormatter("collector"))
    root = logging.getLogger()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.DEBUG)

    async def scenario() -> None:
        reader = FakeReader([incoming(1, age_days=1, text=SECRET)])
        reader.rate_limits = [1]
        collector = make_collector(engine, reader)
        await collector.prepare()
        await collector.catch_up()
        await collector.handle_new(incoming(2, text=SECRET))

        async def failing_save(session, message, status):
            raise ValueError(message.text)

        monkeypatch.setattr(store, "save_message", failing_save)
        await collector.handle_new(incoming(3, text=SECRET))
        await engine.dispose()

    try:
        run(scenario())
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    logs = output.getvalue()
    assert "message_save_failed" in logs
    assert SECRET not in logs
    assert "79001234567" not in logs


def test_run_collector_heartbeats_and_stops_cleanly(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_S", 0.01)
    path = tmp_path / "collector.heartbeat"
    reader = FakeReader([incoming(1, age_days=1)])

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(run_collector(reader, engine, [CHAT], 30, path, stop))
        for _ in range(200):
            if path.exists() and await stored_messages(engine):
                break
            await asyncio.sleep(0.01)
        assert path.exists()
        assert set(await stored_messages(engine)) == {1}
        stop.set()
        await task
        await engine.dispose()

    run(scenario())
    assert reader.disconnected
    assert not path.exists()


def test_run_collector_without_connection_reports_no_heartbeat(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_S", 0.01)
    path = tmp_path / "collector.heartbeat"
    reader = FakeReader()
    reader.connected = False

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(run_collector(reader, engine, [CHAT], 30, path, stop))
        await asyncio.sleep(0.1)
        assert not path.exists()
        stop.set()
        await task
        await engine.dispose()

    run(scenario())


def test_reconnect_triggers_catch_up(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_S", 0.01)
    path = tmp_path / "collector.heartbeat"
    reader = FakeReader()

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(run_collector(reader, engine, [CHAT], 30, path, stop))
        for _ in range(200):
            async with AsyncSession(engine) as session:
                chat = await session.get(Chat, CHAT)
            if chat is not None and chat.history_loaded_at is not None:
                break
            await asyncio.sleep(0.01)

        reader.connected = False
        await asyncio.sleep(0.05)
        reader.messages.append(incoming(1))
        reader.connected = True
        for _ in range(200):
            if await stored_messages(engine):
                break
            await asyncio.sleep(0.01)

        assert (await stored_messages(engine))[1].status is MessageStatus.PENDING
        stop.set()
        await task
        await engine.dispose()

    run(scenario())


def test_stopping_at_any_moment_leaves_no_worker_threads(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover non-daemon aiosqlite thread would keep the process from exiting."""
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_S", 0.001)
    threads_before = threading.active_count()

    async def scenario(delay: float) -> None:
        reader = FakeReader([incoming(i, age_days=1) for i in range(1, 40)])
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_collector(reader, engine, [CHAT], 30, tmp_path / "heartbeat", stop)
        )
        await asyncio.sleep(delay)
        stop.set()
        await asyncio.wait_for(task, timeout=10)
        await engine.dispose()

    for attempt in range(25):
        asyncio.run(scenario(attempt * 0.002))

    for _ in range(200):
        if threading.active_count() <= threads_before:
            break
        time.sleep(0.01)
    assert threading.active_count() <= threads_before


def test_run_collector_rejects_a_missing_session(engine: AsyncEngine, tmp_path: Path) -> None:
    reader = FakeReader()
    reader.authorized = False

    async def scenario() -> None:
        with pytest.raises(NotAuthorizedError):
            await run_collector(reader, engine, [CHAT], 30, tmp_path / "heartbeat", asyncio.Event())
        await engine.dispose()

    run(scenario())
    assert reader.disconnected
