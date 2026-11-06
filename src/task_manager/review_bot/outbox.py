"""Brings the bot up to date with the database: new drafts, changed drafts, service alerts."""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from task_manager.analyzer.ollama import OllamaUnavailableError
from task_manager.review_bot.messenger import (
    Button,
    MessageGone,
    Messenger,
    MessengerError,
    RetryLater,
)
from task_manager.review_bot.render import card_html, card_keyboard, escape
from task_manager.review_bot.views import load_card
from task_manager.storage.attachments import upload_name
from task_manager.storage.models import (
    Attachment,
    BotNotice,
    Draft,
    DraftStatus,
    FileRequest,
    FileRequestStatus,
    Message,
    MessageStatus,
    ServiceState,
)

VIEW_PURPOSE = "view"
SERVICE_SILENT = timedelta(minutes=3)
QUEUE_STUCK = timedelta(hours=1)
TICK_S = 5.0
OLLAMA_CHECK_S = 60.0

logger = logging.getLogger(__name__)


def _format_duration(duration: timedelta) -> str:
    hours, seconds = divmod(int(duration.total_seconds()), 3600)
    return f"{hours} ч {seconds // 60} мин" if hours else f"{seconds // 60} мин"


class Outbox:
    """State lives in the database, so a draft waits here however long the bot was down."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        messenger: Messenger,
        ping: Callable[[], Awaitable[None]],
        utc_offset_hours: int,
        ollama_alert_s: float,
        watched: tuple[str, ...] = (),
        announce_start: bool = False,
    ) -> None:
        self._sessions = sessions
        self._messenger = messenger
        self._ping = ping
        self._offset = utc_offset_hours
        self._alert_after = ollama_alert_s
        self._lock = asyncio.Lock()
        self._known_failed = 0
        self._down_since: float | None = None
        self._alerted = False
        self._checked_at = 0.0
        self._start_reported = not announce_start
        self._watched = watched
        self._alerts: set[str] = set()

    @property
    def ollama_available(self) -> bool:
        return self._down_since is None

    async def _render(self, draft_id: int):
        async with self._sessions() as session:
            card = await load_card(session, draft_id)
        if card is None:
            return None
        return card_html(card, self._offset), card_keyboard(card)

    async def _send_new(self) -> None:
        async with self._sessions() as session:
            ids = list(
                await session.scalars(
                    select(Draft.id)
                    .where(Draft.status == DraftStatus.NEW)
                    .order_by(Draft.last_message_at, Draft.id)
                )
            )
        for draft_id in ids:
            rendered = await self._render(draft_id)
            if rendered is None:
                continue
            message_id = await self._messenger.send(*rendered)
            async with self._sessions() as session:
                await session.execute(
                    update(Draft)
                    .where(Draft.id == draft_id)
                    .values(status=DraftStatus.SENT, bot_message_id=message_id, needs_refresh=False)
                )
                await session.commit()

    async def _replace(self, draft_id: int, old_message_id: int | None, rendered) -> int:
        """A changed draft moves to the bottom of the chat, without a sound."""
        new_id = await self._messenger.send(*rendered, silent=True)
        if old_message_id is None:
            return new_id
        try:
            await self._messenger.delete(old_message_id)
        except MessageGone:
            # Too old to delete. Its buttons still work safely, but it must not look current.
            with contextlib.suppress(MessengerError):
                await self._messenger.edit(
                    old_message_id, f"Задача №{draft_id} обновлена, актуальная версия ниже."
                )
        except MessengerError as error:
            logger.warning("bot_delete_failed", extra={"reason": str(error)})
        return new_id

    async def _refresh(self) -> None:
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(Draft.id, Draft.bot_message_id, Draft.updated_at).where(
                        Draft.needs_refresh.is_(True), Draft.status != DraftStatus.NEW
                    )
                )
            ).all()
        for draft_id, message_id, updated_at in rows:
            rendered = await self._render(draft_id)
            if rendered is None:
                continue
            new_id = await self._replace(draft_id, message_id, rendered)
            async with self._sessions() as session:
                await session.execute(
                    update(Draft).where(Draft.id == draft_id).values(bot_message_id=new_id)
                )
                # A change made while this was being sent keeps the flag and is shown next time.
                await session.execute(
                    update(Draft)
                    .where(Draft.id == draft_id, Draft.updated_at == updated_at)
                    .values(needs_refresh=False)
                )
                await session.commit()

    async def _send_notices(self) -> None:
        """Messages from services that have no bot token. Sent once, in order."""
        async with self._sessions() as session:
            notices = list(await session.scalars(select(BotNotice).order_by(BotNotice.id)))
        for notice in notices:
            keyboard = None
            if notice.keyboard:
                keyboard = [
                    [Button(text, data) for text, data in row]
                    for row in json.loads(notice.keyboard)
                ]
            await self._messenger.send(notice.text, keyboard)
            async with self._sessions() as session:
                await session.execute(delete(BotNotice).where(BotNotice.id == notice.id))
                await session.commit()

    async def _send_files(self) -> None:
        """Files the owner asked to see. Each leaves the disk as soon as it has been sent."""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(FileRequest, Attachment)
                    .join(Attachment, Attachment.id == FileRequest.attachment_id)
                    .where(
                        FileRequest.purpose == VIEW_PURPOSE,
                        FileRequest.status.in_((FileRequestStatus.READY, FileRequestStatus.FAILED)),
                    )
                )
            ).all()
        for request, file in rows:
            name = upload_name(file.file_name, file.kind, file.id)
            if request.status is FileRequestStatus.FAILED:
                await self._messenger.send(f"Файл {escape(name)} получить не удалось.")
            elif request.path and not Path(request.path).is_file():
                # Files live in memory, and a restart of Docker clears them.
                await self._messenger.send(f"Файл {escape(name)} пропал, запросите его снова.")
            elif request.path:
                path = Path(request.path)
                try:
                    await self._messenger.send_file(path, name)
                finally:
                    path.unlink(missing_ok=True)
            async with self._sessions() as session:
                await session.execute(
                    update(FileRequest)
                    .where(FileRequest.id == request.id)
                    .values(status=FileRequestStatus.DONE, path=None)
                )
                await session.commit()

    async def _report_start(self) -> None:
        """Once, when the bot comes up: the owner sees everything rose, and how long it was off."""
        if self._start_reported:
            return
        self._start_reported = True
        async with self._sessions() as session:
            previous = await session.scalar(
                select(ServiceState.last_seen).where(ServiceState.service == "review_bot")
            )
            pending = await session.scalar(
                select(func.count())
                .select_from(Message)
                .where(Message.status == MessageStatus.PENDING)
            )
        line = "✅ Система запущена."
        if previous is not None:
            seen = previous if previous.tzinfo else previous.replace(tzinfo=UTC)
            down = datetime.now(UTC) - seen
            if down >= timedelta(minutes=5):
                line += f" Простой: {_format_duration(down)}."
        await self._messenger.send(f"{line} В очереди на анализ: {pending or 0}.")

    async def _check_services(self) -> None:
        """A service that has gone quiet, or reports a problem. Each is reported once."""
        now = datetime.now(UTC)
        async with self._sessions() as session:
            rows = {row.service: row for row in await session.scalars(select(ServiceState))}
            oldest = await session.scalar(
                select(func.min(Message.sent_at)).where(Message.status == MessageStatus.PENDING)
            )
        problems: dict[str, str] = {}
        for service in self._watched:
            row = rows.get(service)
            if row is None:
                continue
            seen = row.last_seen if row.last_seen.tzinfo else row.last_seen.replace(tzinfo=UTC)
            if now - seen > SERVICE_SILENT:
                problems[service] = f"⚠️ Сервис {service} не отвечает."
            elif row.detail == "telegram_disconnected":
                problems[service] = "⚠️ Нет связи с Telegram, новые сообщения не читаются."
        if oldest is not None and self.ollama_available:
            started = oldest if oldest.tzinfo else oldest.replace(tzinfo=UTC)
            if now - started > QUEUE_STUCK:
                problems["queue"] = "⚠️ Очередь анализа стоит: сообщения ждут больше часа."
        for name, text in problems.items():
            if name not in self._alerts:
                await self._messenger.send(text)
                self._alerts.add(name)
        for name in list(self._alerts - set(problems)):
            await self._messenger.send(f"✅ Восстановлено: {name}.")
            self._alerts.discard(name)

    async def _alert_failed(self) -> None:
        async with self._sessions() as session:
            failed = await session.scalar(
                select(func.count())
                .select_from(Message)
                .where(Message.status == MessageStatus.FAILED)
            )
        failed = failed or 0
        if failed > self._known_failed:
            await self._messenger.send(
                f"⚠️ Модель не смогла разобрать сообщений: {failed}.",
                [[Button("Повторить", "retry")]],
            )
        self._known_failed = failed

    async def _check_ollama(self) -> None:
        now = time.monotonic()
        if now - self._checked_at < OLLAMA_CHECK_S:
            return
        self._checked_at = now
        try:
            await self._ping()
        except OllamaUnavailableError:
            if self._down_since is None:
                self._down_since = now
            if not self._alerted and now - self._down_since >= self._alert_after:
                minutes = round(self._alert_after / 60)
                await self._messenger.send(
                    f"⚠️ Модель недоступна больше {minutes} мин. Сообщения ждут в очереди."
                )
                self._alerted = True
            return
        if self._alerted:
            await self._messenger.send("✅ Модель снова доступна, очередь разбирается.")
        self._down_since = None
        self._alerted = False

    async def flush(self) -> None:
        """Send everything due. Telegram being unavailable only postpones it."""
        async with self._lock:
            try:
                await self._send_new()
                await self._refresh()
                await self._send_notices()
                await self._send_files()
                await self._report_start()
                await self._check_services()
                await self._alert_failed()
                await self._check_ollama()
            except RetryLater as error:
                await asyncio.sleep(error.seconds)
            except MessengerError as error:
                logger.warning("bot_send_failed", extra={"reason": str(error)})

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.flush()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), TICK_S)
