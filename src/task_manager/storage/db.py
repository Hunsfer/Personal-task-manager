"""SQLite engines. Several containers share one database file, so every connection uses WAL.

With a key configured the whole file, its WAL included, is encrypted by SQLCipher. Without one it's
a plain database, which only tests and local development may use: REQUIRE_ENCRYPTED_DB stops a
service from quietly creating an unencrypted database next to the real one.
"""

import re
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
import aiosqlite.core
import sqlcipher3
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.dialects.sqlite.aiosqlite import AsyncAdapt_aiosqlite_dbapi
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from task_manager.common.config import ConfigError, load_settings

BUSY_TIMEOUT_MS = 5000
# A raw 256-bit key. SQLCipher then skips its slow key derivation, which runs per connection.
_RAW_KEY = re.compile(r"^[0-9a-f]{64}$")


def _configured_key() -> str | None:
    settings = load_settings()
    key = settings.database_key.get_secret_value() if settings.database_key else None
    if key is None:
        if settings.require_encrypted_db:
            raise ConfigError("the database key is missing: start the system with `tm start`")
        return None
    if not _RAW_KEY.match(key):
        raise ConfigError("the database key must be 64 lowercase hexadecimal characters")
    return key


def _unlock(cursor: Any, key: str | None) -> None:
    """Must run before anything else on the connection. The key's format was checked above."""
    if key is not None:
        cursor.execute(f"PRAGMA key = \"x'{key}'\"")


def _make_listener(key: str | None):
    def apply_pragmas(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        _unlock(cursor, key)
        # busy_timeout goes first: switching to WAL itself may have to wait for another writer.
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
        except (sqlcipher3.dbapi2.DatabaseError, sqlite3.DatabaseError) as error:
            if key is not None:
                raise ConfigError("the database can't be opened with this key") from None
            raise error
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return apply_pragmas


def make_engine(database_path: Path) -> AsyncEngine:
    key = _configured_key()
    url = f"sqlite+aiosqlite:///{database_path.as_posix()}"
    # aiosqlite opens connections through this module-level name and offers no other way in.
    aiosqlite.core.sqlite3 = sqlite3 if key is None else sqlcipher3.dbapi2
    if key is None:
        engine = create_async_engine(url)
    else:
        engine = create_async_engine(
            url, module=AsyncAdapt_aiosqlite_dbapi(aiosqlite, sqlcipher3.dbapi2)
        )
    event.listen(engine.sync_engine, "connect", _make_listener(key))
    return engine


def make_sync_engine(database_path: Path) -> Engine:
    """Engine for migrations, which run outside the event loop."""
    key = _configured_key()
    url = f"sqlite:///{database_path.as_posix()}"
    engine = create_engine(url) if key is None else create_engine(url, module=sqlcipher3.dbapi2)
    event.listen(engine, "connect", _make_listener(key))
    return engine


def connect_raw(path: Path, read_only: bool = False) -> Any:
    """A plain DB-API connection with the same key, for copying whole databases."""
    key = _configured_key()
    module = sqlite3 if key is None else sqlcipher3.dbapi2
    if read_only:
        connection = module.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    else:
        connection = module.connect(str(path))
    _unlock(connection, key)
    return connection


async def current_revision(engine: AsyncEngine) -> str | None:
    """Return the applied migration revision. Raises if migrations have never run."""
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT version_num FROM alembic_version"))
        return result.scalar_one_or_none()
