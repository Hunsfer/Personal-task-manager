"""Downloads files on request, because only the collector holds the Telegram session.

Files are written with a neutral name and stay only until the bot or the export has used them.
"""

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from task_manager.collector.types import RateLimitedError, Reader
from task_manager.storage.models import Attachment, FileRequest, FileRequestStatus, Message

FILE_LOOP_INTERVAL_S = 5.0
FILE_MAX_AGE_S = 3600.0
REQUEST_MAX_AGE = timedelta(hours=1)
PART_SUFFIX = ".part"

logger = logging.getLogger(__name__)


def request_path(directory: Path, request_id: int) -> Path:
    return directory / f"request-{request_id}"


def remove_forgotten_files(directory: Path, now: float | None = None) -> int:
    """Anything older than an hour was forgotten, for example by a crash."""
    current = time.time() if now is None else now
    removed = 0
    for path in directory.glob("request-*"):
        with contextlib.suppress(FileNotFoundError):
            if current - path.stat().st_mtime > FILE_MAX_AGE_S:
                path.unlink()
                removed += 1
    return removed


async def _finish(
    sessions: async_sessionmaker[AsyncSession],
    request_id: int,
    status: FileRequestStatus,
    path: str | None = None,
    error: str | None = None,
) -> None:
    async with sessions() as session:
        await session.execute(
            update(FileRequest)
            .where(FileRequest.id == request_id)
            .values(status=status, path=path, error=error)
        )
        await session.commit()


async def process_requests(
    sessions: async_sessionmaker[AsyncSession], reader: Reader, directory: Path
) -> int:
    """Handle what is waiting. Raises RateLimitedError so the caller can wait it out."""
    async with sessions() as session:
        rows = (
            await session.execute(
                select(FileRequest.id, Message.chat_id, Message.msg_id)
                .join(Attachment, Attachment.id == FileRequest.attachment_id)
                .join(Message, Message.id == Attachment.message_id)
                .where(FileRequest.status == FileRequestStatus.PENDING)
                .order_by(FileRequest.id)
            )
        ).all()
    for request_id, chat_id, msg_id in rows:
        target = request_path(directory, request_id)
        part = target.with_suffix(PART_SUFFIX)
        try:
            found = await reader.download(chat_id, msg_id, part)
        except RateLimitedError:
            part.unlink(missing_ok=True)
            raise
        except Exception:
            part.unlink(missing_ok=True)
            logger.exception("file_download_failed", extra={"msg_id": msg_id})
            await _finish(sessions, request_id, FileRequestStatus.FAILED, error="download_failed")
            continue
        if not found:
            # The message or its file was deleted from the chat.
            await _finish(sessions, request_id, FileRequestStatus.FAILED, error="message_gone")
            continue
        part.replace(target)
        await _finish(sessions, request_id, FileRequestStatus.READY, path=str(target))
    return len(rows)


async def expire_requests(sessions: async_sessionmaker[AsyncSession]) -> None:
    """A ready file nobody picked up is gone by now, so the request can't be used any more."""
    cutoff = datetime.now(UTC) - REQUEST_MAX_AGE
    async with sessions() as session:
        await session.execute(
            update(FileRequest)
            .where(
                FileRequest.status == FileRequestStatus.READY,
                FileRequest.created_at < cutoff,
            )
            .values(status=FileRequestStatus.FAILED, path=None, error="expired")
        )
        await session.commit()


async def file_loop(
    sessions: async_sessionmaker[AsyncSession],
    reader: Reader,
    directory: Path,
    shutdown: asyncio.Event,
) -> None:
    """Ends on the shutdown event, like the heartbeat, so a database call is never cancelled."""
    directory.mkdir(parents=True, exist_ok=True)
    while not shutdown.is_set():
        delay = FILE_LOOP_INTERVAL_S
        try:
            if reader.is_connected():
                await process_requests(sessions, reader, directory)
            await expire_requests(sessions)
            remove_forgotten_files(directory)
        except RateLimitedError as error:
            logger.warning("rate_limited", extra={"seconds": error.seconds})
            delay = float(error.seconds)
        except Exception:
            logger.exception("file_loop_error")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(shutdown.wait(), delay)
