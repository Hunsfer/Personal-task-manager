"""Backups of the database. A copy is encrypted with the same key as the original.

A backup nobody has restored isn't known to work, so verifying one means opening it and reading it.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from task_manager.storage.db import connect_raw

PREFIX = "app-"
SUFFIX = ".db"
KEEP = 14


@dataclass(frozen=True)
class BackupCheck:
    messages: int
    drafts: int


def create_backup(database: Path, directory: Path, now: datetime | None = None) -> Path:
    """Consistent even while services are writing. Written aside first, so it's never half done."""
    moment = now or datetime.now(UTC)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{PREFIX}{moment:%Y%m%d-%H%M%S}{SUFFIX}"
    partial = target.with_suffix(".part")
    partial.unlink(missing_ok=True)
    source = connect_raw(database, read_only=True)
    copy = connect_raw(partial)
    try:
        source.backup(copy)
        # The copy inherits WAL, and merely opening it read-only would then leave files beside it.
        copy.execute("PRAGMA journal_mode=DELETE").fetchall()
    finally:
        copy.close()
        source.close()
    partial.replace(target)
    return target


def verify_backup(path: Path) -> BackupCheck:
    """Raises if the file is damaged or the key doesn't fit."""
    connection = connect_raw(path, read_only=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ValueError("the backup failed the integrity check")
        return BackupCheck(
            messages=connection.execute("SELECT count(*) FROM messages").fetchone()[0],
            drafts=connection.execute("SELECT count(*) FROM drafts").fetchone()[0],
        )
    finally:
        connection.close()


def rotate_backups(directory: Path, keep: int = KEEP) -> int:
    backups = sorted(directory.glob(f"{PREFIX}*{SUFFIX}"))
    old = backups[:-keep] if keep else backups
    for path in old:
        path.unlink()
        for suffix in ("-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
    return len(old)


def restore_backup(path: Path, database: Path) -> None:
    """The caller stops the services first. The current database stays until it's replaced."""
    verify_backup(path)
    partial = database.with_suffix(".restoring")
    partial.unlink(missing_ok=True)
    source = connect_raw(path, read_only=True)
    target = connect_raw(partial)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    for suffix in ("-wal", "-shm"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)
    partial.replace(database)
