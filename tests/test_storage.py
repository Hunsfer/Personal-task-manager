import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.storage.db import BUSY_TIMEOUT_MS, current_revision, make_engine
from task_manager.storage.models import Attachment, Chat, Message, MessageStatus

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
SENT_AT = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def alembic_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "nested" / "app.db"))
    return Config(str(ALEMBIC_INI))


@pytest.fixture
def database(tmp_path: Path, alembic_config: Config) -> Path:
    command.upgrade(alembic_config, "head")
    return tmp_path / "nested" / "app.db"


def _message(msg_id: int = 1) -> Message:
    return Message(chat_id=-100, msg_id=msg_id, sent_at=SENT_AT, text="Иван, посмотри")


async def _seed(session: AsyncSession) -> None:
    session.add(Chat(chat_id=-100, title="Чат"))
    await session.flush()


def test_migration_creates_the_tables(database: Path) -> None:
    async def scenario() -> tuple[set[str], str | None]:
        engine = make_engine(database)
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda c: set(inspect(c).get_table_names()))
        revision = await current_revision(engine)
        await engine.dispose()
        return tables, revision

    tables, revision = asyncio.run(scenario())
    assert {"chats", "messages", "attachments"} <= tables
    assert revision is not None


def test_models_match_migrations(alembic_config: Config, database: Path) -> None:
    command.check(alembic_config)


def test_downgrade_removes_the_tables(alembic_config: Config, database: Path) -> None:
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")


def test_recreating_a_table_keeps_the_rows_pointing_at_it(
    alembic_config: Config, tmp_path: Path
) -> None:
    """A batch migration drops the old table, which must not cascade to its dependents."""
    command.upgrade(alembic_config, "ee7c46bd9685")
    path = tmp_path / "nested" / "app.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        INSERT INTO chats (chat_id, kind) VALUES (1, 'group');
        INSERT INTO messages (id, chat_id, msg_id, sent_at, text, status, mentions_me, is_outgoing)
            VALUES (1, 1, 1, '2026-11-04 12:00:00', 'т', 'done', 0, 0);
        INSERT INTO drafts (id, chat_id, status, title, description, priority, has_placeholders,
                            text_is_stale, model, prompt_id, last_message_at,
                            created_at, updated_at)
            VALUES (1, 1, 'new', 'т', 'о', 'medium', 0, 0, 'm', 'p', '2026-11-04 12:00:00',
                    '2026-11-04 12:00:00', '2026-11-04 12:00:00');
        INSERT INTO draft_messages (draft_id, message_id, role) VALUES (1, 1, 'trigger');
        """
    )
    connection.close()

    command.upgrade(alembic_config, "head")

    connection = sqlite3.connect(path)
    links = connection.execute("SELECT draft_id, message_id FROM draft_messages").fetchall()
    connection.close()
    assert links == [(1, 1)]


def test_current_revision_fails_without_migrations(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine = make_engine(tmp_path / "empty.db")
        try:
            await current_revision(engine)
        finally:
            await engine.dispose()

    with pytest.raises(Exception, match="alembic_version"):
        asyncio.run(scenario())


def test_connections_use_the_shared_database_pragmas(database: Path) -> None:
    async def scenario() -> tuple[str, int, int]:
        engine = make_engine(database)
        async with engine.connect() as connection:
            journal = (await connection.execute(text("PRAGMA journal_mode"))).scalar_one()
            timeout = (await connection.execute(text("PRAGMA busy_timeout"))).scalar_one()
            foreign_keys = (await connection.execute(text("PRAGMA foreign_keys"))).scalar_one()
        await engine.dispose()
        return journal, timeout, foreign_keys

    assert asyncio.run(scenario()) == ("wal", BUSY_TIMEOUT_MS, 1)


def test_new_message_is_pending_and_survives_reconnect(database: Path) -> None:
    async def write() -> None:
        engine = make_engine(database)
        async with AsyncSession(engine) as session:
            await _seed(session)
            session.add(_message())
            await session.commit()
        await engine.dispose()

    async def read() -> Message:
        engine = make_engine(database)
        async with AsyncSession(engine) as session:
            message = (await session.execute(select(Message))).scalar_one()
        await engine.dispose()
        return message

    asyncio.run(write())
    message = asyncio.run(read())
    assert message.status is MessageStatus.PENDING
    assert message.text == "Иван, посмотри"


def test_same_message_cannot_be_saved_twice(database: Path) -> None:
    async def scenario() -> None:
        engine = make_engine(database)
        try:
            async with AsyncSession(engine) as session:
                await _seed(session)
                session.add_all([_message(), _message()])
                await session.commit()
        finally:
            await engine.dispose()

    with pytest.raises(IntegrityError):
        asyncio.run(scenario())


def test_deleting_a_message_removes_its_attachments(database: Path) -> None:
    async def scenario() -> list[Attachment]:
        engine = make_engine(database)
        async with AsyncSession(engine) as session:
            await _seed(session)
            message = _message()
            message.attachments.append(Attachment(kind="photo", size_bytes=10))
            session.add(message)
            await session.commit()

            await session.execute(text("DELETE FROM messages"))
            await session.commit()
            remaining = (await session.execute(select(Attachment))).scalars().all()
        await engine.dispose()
        return list(remaining)

    assert asyncio.run(scenario()) == []
