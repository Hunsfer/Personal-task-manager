"""Moves approved drafts to Jira. Every step is recorded, so a restart repeats nothing.

Text becomes real here for the first time, right before it leaves. It must never be logged.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from task_manager.anonymizer.engine import Span
from task_manager.common.config import ConfigError, JiraConfig
from task_manager.jira_export import content
from task_manager.jira_export.client import (
    JiraAuthError,
    JiraClient,
    JiraNotFoundError,
    JiraRateLimitedError,
    JiraRejectedError,
    JiraUnavailableError,
)
from task_manager.monitoring import state_loop
from task_manager.review_bot.render import escape, extract_code
from task_manager.service import beat
from task_manager.storage.attachments import goes_to_jira, upload_name
from task_manager.storage.models import (
    Attachment,
    AttachmentUpload,
    BotNotice,
    Draft,
    DraftDecision,
    DraftKind,
    DraftMessage,
    DraftStatus,
    ExportAction,
    FileRequest,
    FileRequestStatus,
    Message,
)

TICK_S = 15.0
STATUS_SYNC_S = 300.0
# How long a task created just before a crash may take to appear in Jira's search.
SEARCH_SETTLE = timedelta(minutes=2)
FILE_WAIT = timedelta(minutes=30)

logger = logging.getLogger(__name__)

SecretFinder = Callable[[str], list[Span]]


class LockedStatusError(Exception):
    """The task is in a status that must not change."""


def check_project(config: JiraConfig, statuses: dict[str, list[str]]) -> None:
    """A misspelt status would silently stop protecting finished tasks, so it stops the service."""
    if config.issue_type not in statuses:
        raise ConfigError(f"issue type {config.issue_type!r} isn't in the project")
    known = {status.casefold() for names in statuses.values() for status in names}
    missing = [s for s in config.locked_statuses if s.casefold() not in known]
    if missing:
        raise ConfigError(f"locked_statuses not found in the project: {missing}")
    missing = [s for s in config.auto_append_statuses if s.casefold() not in known]
    if missing:
        raise ConfigError(f"auto_append_statuses not found in the project: {missing}")
    locked = {s.casefold() for s in config.locked_statuses}
    both = [s for s in config.auto_append_statuses if s.casefold() in locked]
    if both:
        raise ConfigError(f"statuses both locked and auto-appended: {both}")


def mask_secrets(code: str, find: SecretFinder) -> tuple[str, int]:
    """Code goes to Jira as it is, except for what must never leave: passwords and card numbers."""
    spans = sorted(find(code), key=lambda span: span.start, reverse=True)
    for span in spans:
        code = f"{code[: span.start]}<{span.kind}>{code[span.end :]}"
    return code, len(spans)


class Exporter:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        client: JiraClient,
        config: JiraConfig,
        account_id: str,
        find_secrets: SecretFinder,
    ) -> None:
        self.sessions = sessions
        self._sessions = sessions
        self._client = client
        self._config = config
        self._account_id = account_id
        self._find_secrets = find_secrets
        self._locked = {status.casefold() for status in config.locked_statuses}
        self._auto_append = {status.casefold() for status in config.auto_append_statuses}
        self._auth_reported = False
        self._statuses_at = 0.0

    def is_locked(self, status: str | None) -> bool:
        return status is not None and status.casefold() in self._locked

    async def _notify(self, text: str, keyboard: list[list[tuple[str, str]]] | None = None) -> None:
        async with self._sessions() as session:
            session.add(
                BotNotice(
                    text=text,
                    keyboard=json.dumps(keyboard, ensure_ascii=False) if keyboard else None,
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()

    async def _set(self, draft_id: int, **values: Any) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Draft)
                .where(Draft.id == draft_id)
                .values(**values, needs_refresh=True, updated_at=datetime.now(UTC))
            )
            await session.commit()

    async def _record(self, draft_id: int, **values: Any) -> None:
        """Bookkeeping the card doesn't show, so it doesn't move the card down the chat."""
        async with self._sessions() as session:
            await session.execute(update(Draft).where(Draft.id == draft_id).values(**values))
            await session.commit()

    async def _code_nodes(self, session: AsyncSession, draft_id: int) -> tuple[list, int]:
        rows = await session.scalars(
            select(Message)
            .join(DraftMessage, DraftMessage.message_id == Message.id)
            .where(DraftMessage.draft_id == draft_id)
            .order_by(Message.sent_at, Message.id)
        )
        nodes = []
        masked = 0
        for message in rows:
            for code in extract_code(message.text, message.code_entities):
                clean, count = mask_secrets(code, self._find_secrets)
                masked += count
                nodes.append(content.code_node(clean))
        return nodes, masked

    async def _fields(self, draft: Draft, parent_note: str | None) -> tuple[dict[str, Any], int]:
        async with self._sessions() as session:
            title = await content.prepare_text(session, draft.title)
            description = await content.prepare_text(session, draft.description)
            code, masked = await self._code_nodes(session, draft.id)
        nodes = content.text_nodes(description.text)
        if parent_note:
            nodes += content.text_nodes(parent_note)
        fields: dict[str, Any] = {
            "project": {"key": self._config.project_key},
            "issuetype": {"name": self._config.issue_type},
            "summary": content.summary(title.text),
            "description": content.document([*nodes, *code]),
            "assignee": {"accountId": self._account_id},
        }
        priority = self._config.priorities.get(draft.priority)
        if priority:
            fields["priority"] = {"name": priority}
        # The model's date arithmetic is unreliable, so only a date the owner set goes in the field.
        if draft.deadline and draft.deadline_by_owner:
            fields["duedate"] = draft.deadline
        return fields, masked

    async def _existing_key(self, draft: Draft) -> tuple[str | None, bool]:
        """After a crash the task may already exist. Returns its key and whether to wait."""
        if draft.export_started_at is None:
            return None, False
        async with self._sessions() as session:
            title = await content.prepare_text(session, draft.title)
        key = await self._client.find_my_recent_task(
            self._config.project_key, content.summary(title.text)
        )
        if key is not None:
            return key, False
        started = draft.export_started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        # Jira's search catches up with a short delay, and a second task is worse than a wait.
        return None, datetime.now(UTC) - started < SEARCH_SETTLE

    async def _parent(self, draft: Draft) -> Draft | None:
        if draft.parent_draft_id is None:
            return None
        async with self._sessions() as session:
            return await session.get(Draft, draft.parent_draft_id)

    async def _create(self, draft: Draft) -> None:
        key, wait = await self._existing_key(draft)
        if wait:
            return
        parent = await self._parent(draft)
        link_to = None
        parent_note = None
        if key is None and parent is not None and parent.jira_key:
            # Even a link changes the original, so a locked task only gets mentioned in the new one.
            status = await self._status_or_none(parent.jira_key)
            if status is not None and not self.is_locked(status):
                link_to = parent.jira_key
            else:
                parent_note = f"Продолжение задачи {parent.jira_key}"
        if key is None:
            fields, masked = await self._fields(draft, parent_note)
            await self._record(draft.id, export_started_at=datetime.now(UTC))
            key = await self._client.create_issue(fields)
            if masked:
                await self._notify(
                    f"В коде задачи {escape(key)} скрыты секреты: {masked}. "
                    "Проверьте код в Jira, если он нужен целиком."
                )
        await self._set(
            draft.id,
            status=DraftStatus.EXPORTED,
            jira_key=key,
            jira_link=f"{self._client.base_url}/browse/{key}",
            export_error=None,
        )
        await self._refresh_children(draft.id)
        if link_to is not None:
            try:
                await self._client.link_issues(self._config.link_type, key, link_to)
            except (JiraRejectedError, JiraNotFoundError):
                logger.warning("jira_link_failed")
                await self._notify(
                    f"Задача {escape(key)} создана, но не связана с {escape(link_to)}. "
                    "Свяжите их вручную."
                )

    async def _status_or_none(self, key: str) -> str | None:
        try:
            status, _ = await self._client.issue_state(key)
        except JiraNotFoundError:
            return None
        return status

    async def _write_to_existing(self, key: str, nodes: list, marker: str) -> None:
        """The only way to change a task that already exists.

        It reads the status immediately before writing, whatever a card showed earlier: the task
        may have moved since the owner pressed the button.
        """
        status, description = await self._client.issue_state(key)
        if self.is_locked(status):
            raise LockedStatusError(status)
        if content.mentions(description, marker):
            return
        await self._client.update_description(key, content.extended_document(description, nodes))

    async def _append(self, draft: Draft) -> None:
        parent = await self._parent(draft)
        if parent is None or not parent.jira_key:
            return
        when = f"{draft.last_message_at:%d.%m}"
        marker = f"черновик {draft.id}"
        async with self._sessions() as session:
            description = await content.prepare_text(session, draft.description)
            code, masked = await self._code_nodes(session, draft.id)
        nodes = [
            content.heading_node(f"Изменение от {when}, {marker}"),
            *content.text_nodes(description.text),
            *code,
        ]
        try:
            await self._write_to_existing(parent.jira_key, nodes, marker)
        except (LockedStatusError, JiraNotFoundError) as error:
            await self._refused(draft, parent, error)
            return
        if masked:
            await self._notify(f"В коде дополнения к {escape(parent.jira_key)} скрыты секреты.")
        await self._set(
            draft.id,
            status=DraftStatus.EXPORTED,
            jira_key=parent.jira_key,
            jira_link=parent.jira_link,
            export_error=None,
        )

    async def _refused(self, draft: Draft, parent: Draft, error: Exception) -> None:
        """Nothing was changed. The decision goes back to the owner, who can make a new task."""
        locked = isinstance(error, LockedStatusError)
        await self._set(
            parent.id,
            jira_status=str(error) if locked else None,
            jira_locked=locked,
        )
        await self._set(draft.id, status=DraftStatus.SENT, export_action=None)
        reason = "уже в статусе, который нельзя менять" if locked else "удалена в Jira"
        await self._notify(
            f"Задача {escape(parent.jira_key or '')} {reason}, поэтому она не изменена. "
            f"Для запроса №{draft.id} можно создать новую задачу."
        )

    async def _needs_the_owners_eye(self, draft: Draft) -> bool:
        """Real values and unanonymized files must be seen before they leave, so these wait."""
        if draft.has_placeholders:
            return True
        async with self._sessions() as session:
            return bool(await self._draft_attachments(session, draft.id))

    async def _append_automatically(self) -> None:
        """A task nobody has started on can take changes without asking the owner.

        The owner already approved the task itself. Everything else about the request stays on
        the buttons, and a later status change is caught by the check in _write_to_existing.
        """
        if not self._auto_append:
            return
        async with self._sessions() as session:
            drafts = list(
                await session.scalars(
                    select(Draft)
                    .where(
                        Draft.kind == DraftKind.CHANGE_REQUEST,
                        Draft.status.in_((DraftStatus.NEW, DraftStatus.SENT)),
                        Draft.parent_draft_id.is_not(None),
                    )
                    .order_by(Draft.updated_at, Draft.id)
                )
            )
        for draft in drafts:
            parent = await self._parent(draft)
            if parent is None or not parent.jira_key or await self._needs_the_owners_eye(draft):
                continue
            status = await self._status_or_none(parent.jira_key)
            if status is None or status.casefold() not in self._auto_append:
                continue
            # A press in the bot may have got here first, and then it's no longer ours to apply.
            async with self._sessions() as session:
                claimed = await session.execute(
                    update(Draft)
                    .where(
                        Draft.id == draft.id,
                        Draft.status.in_((DraftStatus.NEW, DraftStatus.SENT)),
                    )
                    .values(export_action=ExportAction.APPEND)
                )
                session.add(
                    DraftDecision(
                        draft_id=draft.id,
                        decision="auto_appended",
                        detail=status,
                        decided_at=datetime.now(UTC),
                    )
                )
                if claimed.rowcount == 0:
                    await session.rollback()
                    continue
                await session.commit()
            await self._append(draft)
            await self._notify(
                f"Задача {escape(parent.jira_key)} в статусе «{escape(status)}» дополнена "
                f"автоматически: {escape(draft.title)}"
            )

    async def _export_approved(self) -> None:
        async with self._sessions() as session:
            drafts = list(
                await session.scalars(
                    select(Draft)
                    .where(
                        Draft.status == DraftStatus.APPROVED,
                        # Drafts approved before export existed have no action and stay put.
                        Draft.export_action.is_not(None),
                        Draft.export_error.is_(None),
                    )
                    .order_by(Draft.updated_at, Draft.id)
                )
            )
        for draft in drafts:
            parent = await self._parent(draft)
            if parent is not None and parent.export_action is not None and not parent.jira_key:
                # The original goes first: there's nothing to link to or extend yet.
                continue
            try:
                if draft.export_action is ExportAction.APPEND:
                    await self._append(draft)
                else:
                    await self._create(draft)
            except JiraRejectedError as error:
                logger.warning("jira_export_rejected", extra={"status": error.status})
                await self._set(draft.id, export_error=str(error))
                await self._notify(
                    f"Jira не приняла задачу №{draft.id}: {escape(str(error))}.",
                    [[("Повторить", f"xr:{draft.id}")]],
                )

    async def _refresh_children(self, parent_id: int) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Draft)
                .where(
                    Draft.parent_draft_id == parent_id,
                    Draft.status == DraftStatus.SENT,
                )
                .values(needs_refresh=True, updated_at=datetime.now(UTC))
            )
            await session.commit()

    async def _sync_statuses(self, now: float) -> None:
        """Keeps the status shown on change requests current, because the bot has no token."""
        if now - self._statuses_at < STATUS_SYNC_S:
            return
        self._statuses_at = now
        async with self._sessions() as session:
            parent_ids = set(
                await session.scalars(
                    select(Draft.parent_draft_id).where(
                        Draft.kind == DraftKind.CHANGE_REQUEST,
                        Draft.status.in_((DraftStatus.NEW, DraftStatus.SENT)),
                        Draft.parent_draft_id.is_not(None),
                    )
                )
            )
            parents = (
                list(await session.scalars(select(Draft).where(Draft.id.in_(parent_ids))))
                if parent_ids
                else []
            )
        for parent in parents:
            if not parent.jira_key:
                continue
            status = await self._status_or_none(parent.jira_key)
            locked = status is None or self.is_locked(status)
            if status != parent.jira_status or locked != parent.jira_locked:
                await self._set(parent.id, jira_status=status, jira_locked=locked)
                await self._refresh_children(parent.id)

    async def _draft_attachments(self, session: AsyncSession, draft_id: int) -> list[Attachment]:
        rows = await session.scalars(
            select(Attachment)
            .join(DraftMessage, DraftMessage.message_id == Attachment.message_id)
            .where(DraftMessage.draft_id == draft_id)
            .order_by(Attachment.id)
        )
        return [row for row in rows if goes_to_jira(row.kind, row.excluded, row.opted_in)]

    async def _upload_one(self, draft: Draft, file: Attachment) -> str | None:
        """Moves one file along. Returns a failure code once it has finally failed."""
        async with self._sessions() as session:
            request = await session.scalar(
                select(FileRequest).where(
                    FileRequest.attachment_id == file.id,
                    FileRequest.draft_id == draft.id,
                    FileRequest.purpose == "upload",
                )
            )
            if request is None:
                session.add(
                    FileRequest(
                        attachment_id=file.id,
                        draft_id=draft.id,
                        purpose="upload",
                        status=FileRequestStatus.PENDING,
                        created_at=datetime.now(UTC),
                    )
                )
                await session.commit()
                return None
            status, path, error = request.status, request.path, request.error
            created = request.created_at
        if status is FileRequestStatus.PENDING:
            started = created if created.tzinfo else created.replace(tzinfo=UTC)
            return "download_timeout" if datetime.now(UTC) - started > FILE_WAIT else None
        if status is FileRequestStatus.FAILED or not path:
            return error or "download_failed"
        local = Path(path)
        if not local.is_file():
            # Files live in memory, and a restart of Docker clears them.
            return "file_lost"
        try:
            await self._client.upload_attachment(
                draft.jira_key, local, upload_name(file.file_name, file.kind, file.id)
            )
        except JiraRejectedError as rejected:
            return f"jira_{rejected.status}"
        finally:
            # Files are only ever kept for as long as it takes to pass them on.
            local.unlink(missing_ok=True)
        return ""

    async def _upload_files(self) -> None:
        async with self._sessions() as session:
            drafts = list(
                await session.scalars(
                    select(Draft).where(
                        Draft.status == DraftStatus.EXPORTED,
                        Draft.files_done.is_(False),
                    )
                )
            )
        for draft in drafts:
            async with self._sessions() as session:
                files = await self._draft_attachments(session, draft.id)
                finished = set(
                    await session.scalars(
                        select(AttachmentUpload.attachment_id).where(
                            AttachmentUpload.draft_id == draft.id
                        )
                    )
                )
            waiting = False
            failed = []
            for file in files:
                if file.id in finished:
                    continue
                outcome = await self._upload_one(draft, file)
                if outcome is None:
                    waiting = True
                    continue
                async with self._sessions() as session:
                    session.add(
                        AttachmentUpload(
                            draft_id=draft.id,
                            attachment_id=file.id,
                            error=outcome or None,
                            finished_at=datetime.now(UTC),
                        )
                    )
                    await session.commit()
                if outcome:
                    failed.append(upload_name(file.file_name, file.kind, file.id))
            if failed:
                names = ", ".join(escape(name) for name in failed)
                await self._notify(
                    f"В задачу {escape(draft.jira_key or '')} не попали файлы: {names}. "
                    "Задача создана.",
                    [[("Повторить файлы", f"ur:{draft.id}")]],
                )
            if not waiting:
                await self._record(draft.id, files_done=True)

    async def tick(self, now: float) -> None:
        """One pass over everything due. A bad token stops it with one message to the owner."""
        try:
            await self._export_approved()
            await self._append_automatically()
            await self._upload_files()
            await self._sync_statuses(now)
        except JiraAuthError:
            logger.warning("jira_auth_failed")
            if not self._auth_reported:
                self._auth_reported = True
                await self._notify(
                    "⚠️ Jira отказала в доступе. Проверьте API-токен в .env.jira: "
                    "экспорт ждёт, ничего не потеряно."
                )
            return
        except JiraRateLimitedError as error:
            logger.warning("jira_rate_limited", extra={"seconds": error.seconds})
            await asyncio.sleep(error.seconds)
            return
        except JiraUnavailableError as error:
            logger.warning("jira_unavailable", extra={"reason": str(error)})
            return
        if self._auth_reported:
            self._auth_reported = False
            await self._notify("✅ Доступ к Jira восстановлен.")


async def run_exporter(exporter: Exporter, heartbeat: Path, stop: asyncio.Event) -> None:
    finished = asyncio.Event()
    beating = asyncio.create_task(beat(heartbeat, finished))
    reporting = asyncio.create_task(state_loop(exporter.sessions, "jira_export", finished))
    logger.info("jira_export_ready")
    loop = asyncio.get_running_loop()
    try:
        while not stop.is_set():
            try:
                await exporter.tick(loop.time())
            except Exception:
                # One bad draft must not stop the others. The stack holds no message text.
                logger.exception("jira_export_error")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), TICK_S)
    finally:
        finished.set()
        await beating
        await reporting
        heartbeat.unlink(missing_ok=True)
