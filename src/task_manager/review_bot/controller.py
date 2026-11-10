"""What each button and message in the bot does. No Telegram library here.

Only the owner is ever answered. Anything the owner types passes through the anonymizer before it
can reach the model or a draft, because the invariant holds for typed text too.
"""

import contextlib
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from task_manager.analyzer.classifier import AnswerError
from task_manager.analyzer.drafts import REJECTED_ACTION, UNDECIDED, draft_lines
from task_manager.analyzer.ollama import OllamaUnavailableError
from task_manager.analyzer.thread import build_thread
from task_manager.analyzer.verdict import DraftText, has_placeholder
from task_manager.review_bot import views
from task_manager.review_bot.chat_log import ChatLog
from task_manager.review_bot.messenger import Button, Keyboard, MessengerError
from task_manager.review_bot.outbox import VIEW_PURPOSE
from task_manager.review_bot.render import (
    PRIORITIES,
    REJECTION_REASONS,
    DraftCard,
    escape,
    format_size,
    format_time,
    split_message,
)
from task_manager.storage.attachments import HIDDEN_KINDS, OPT_IN_KINDS
from task_manager.storage.models import (
    AnonymizerOverride,
    Attachment,
    AttachmentUpload,
    Draft,
    DraftDecision,
    DraftKind,
    DraftMessage,
    DraftRole,
    DraftStatus,
    ExportAction,
    FileRequest,
    FileRequestStatus,
    Message,
    MessageStatus,
    Pseudonym,
)

APPROVED_ACTION = "approved"
OWNER_CREATED_ACTION = "created_by_owner"
RECENT_PAGE = 8
RECENT_PREVIEW = 300
EDITED_ACTION = "edited"
MODEL_EDITED_ACTION = "model_edited"
OTHER_REASON = "other"
MAX_TITLE = 200
MAX_BOT_FILE_BYTES = 50 * 1024 * 1024
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

logger = logging.getLogger(__name__)


class TextEditor(Protocol):
    model: str

    async def edit_text(
        self, lines, title: str, description: str, instruction: str
    ) -> DraftText: ...

    async def rewrite(self, lines) -> DraftText | None: ...


class TextAnonymizer(Protocol):
    async def anonymize(self, session: AsyncSession, text: str) -> str: ...


@dataclass(frozen=True)
class _WaitingField:
    draft_id: int
    field: str


@dataclass(frozen=True)
class _WaitingNote:
    draft_id: int


@dataclass(frozen=True)
class _WaitingInstruction:
    draft_id: int


@dataclass(frozen=True)
class _Proposal:
    draft_id: int
    text: DraftText


@dataclass(frozen=True)
class _NameChoice:
    draft_id: int
    field: str
    typed: str
    anonymized: str


class Controller:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        messenger: ChatLog,
        editor: TextEditor,
        names: TextAnonymizer,
        rules: Callable[[AsyncSession], Awaitable[str]],
        flush: Callable[[], Awaitable[None]],
        ollama_available: Callable[[], bool],
        owner_id: int,
        utc_offset_hours: int,
    ) -> None:
        self._sessions = sessions
        self._messenger = messenger
        self._editor = editor
        self._names = names
        self._rules = rules
        self._flush = flush
        self._ollama_available = ollama_available
        self._owner_id = owner_id
        self._offset = utc_offset_hours
        self._waiting: _WaitingField | _WaitingNote | _WaitingInstruction | None = None
        self._proposal: _Proposal | None = None
        self._name_choice: _NameChoice | None = None
        # Menus and questions, removed once answered so the chat holds only tasks.
        self._prompts: list[int] = []
        # Messages whose draft the model is writing now, so a second press makes no duplicate.
        self._making: set[int] = set()

    # Entry points

    def is_owner(self, user_id: int | None) -> bool:
        return user_id == self._owner_id

    async def on_text(self, user_id: int | None, text: str, message_id: int | None = None) -> None:
        if not self.is_owner(user_id):
            logger.info("bot_foreign_user")
            return
        if message_id is not None:
            await self._messenger.remember(message_id)
        command = text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""
        if command.startswith("/"):
            await self._reset()
            match command.split("@")[0]:
                case "/list":
                    await self._list()
                case "/recent":
                    await self._recent()
                case "/status":
                    await self._status()
                case "/overrides":
                    await self._overrides()
                case "/export_old":
                    await self._export_old()
                case "/clear":
                    await self._clear()
                case "/cancel":
                    await self._say("Отменено.")
                case _:
                    await self._say(
                        "Новые черновики приходят сами.\n"
                        "/list нерешённые черновики\n"
                        "/recent что модель не сочла задачей, можно сделать черновик\n"
                        "/status очередь и модель\n"
                        "/overrides слова, отмеченные как не имена\n"
                        "/clear очистить переписку, статусы задач не меняются\n"
                        "/export_old задачи, принятые до подключения Jira"
                    )
            return
        match self._waiting:
            case _WaitingField():
                await self._field_typed(self._waiting, text)
            case _WaitingNote():
                await self._reject(self._waiting.draft_id, OTHER_REASON, text)
            case _WaitingInstruction():
                await self._ask_model(self._waiting.draft_id, text)
            case _:
                await self._say("Выберите действие кнопкой под черновиком или отправьте /list.")

    async def on_callback(self, user_id: int | None, data: str) -> None:
        if not self.is_owner(user_id):
            logger.info("bot_foreign_user")
            return
        action, _, rest = data.partition(":")
        arguments = rest.split(":") if rest else []
        handler = self._handlers().get(action)
        if handler is not None:
            await handler(*arguments)

    def _handlers(self):
        return {
            "ok": self._approve_asked,
            "okc": self._approve_confirmed,
            "ap": self._append_asked,
            "apc": self._append_confirmed,
            "xo": self._export_one_old,
            "xr": self._export_retry,
            "ur": self._uploads_retry,
            "fv": self._view_file,
            "rj": self._reject_asked,
            "rjr": self._reject_reason,
            "rjs": self._reject_without_note,
            "ed": self._edit_asked,
            "edf": self._edit_field,
            "edp": self._edit_priority,
            "edn": self._name_chosen,
            "ai": self._model_asked,
            "aia": self._proposal_accepted,
            "src": self._source,
            "ph": self._placeholders,
            "phx": self._mark_asked,
            "phy": self._mark_confirmed,
            "ovd": self._override_removed,
            "att": self._attachments,
            "atg": self._attachment_toggled,
            "open": self._open,
            "rc": self._recent,
            "mk": self._make_task,
            "retry": self._retry,
            "cancel": self._cancel,
        }

    # Helpers

    async def _reset(self) -> None:
        self._waiting = None
        self._proposal = None
        self._name_choice = None
        prompts, self._prompts = self._prompts, []
        for message_id in prompts:
            # A menu left behind is harmless: its buttons are safe to press.
            with contextlib.suppress(MessengerError):
                await self._messenger.delete(message_id)

    async def _say(self, text: str, keyboard: Keyboard | None = None) -> None:
        await self._messenger.send(text, keyboard)

    async def _ask(self, text: str, keyboard: Keyboard | None = None) -> None:
        """A question or menu, which goes away as soon as it's answered."""
        self._prompts.append(await self._messenger.send(text, keyboard))

    async def _card(self, draft_id: int) -> DraftCard | None:
        async with self._sessions() as session:
            return await views.load_card(session, draft_id)

    async def _open_card(self, draft_id: int | str) -> DraftCard | None:
        """A draft that can still be changed, or a message saying why not."""
        card = await self._card(int(draft_id))
        if card is None:
            await self._say("Черновик не найден.")
            return None
        if card.decided:
            await self._say(f"По задаче №{card.id} решение уже принято.")
            return None
        return card

    @staticmethod
    def _cancel_row() -> list[Button]:
        return [Button("Отмена", "cancel")]

    async def _cancel(self) -> None:
        await self._reset()
        await self._say("Отменено.")

    async def _change_text(
        self, draft_id: int, values: dict[str, object], action: str, detail: str
    ) -> bool:
        """Edit an undecided draft. The owner's wording is kept from now on."""
        now = datetime.now(UTC)
        async with self._sessions() as session:
            result = await session.execute(
                update(Draft)
                .where(Draft.id == draft_id, Draft.status.in_(UNDECIDED))
                .values(**values, edited_by_owner=True, needs_refresh=True, updated_at=now)
            )
            if result.rowcount == 0:
                await session.rollback()
                return False
            draft = await session.get_one(Draft, draft_id)
            draft.has_placeholders = has_placeholder(draft.title) or has_placeholder(
                draft.description
            )
            session.add(
                DraftDecision(draft_id=draft_id, decision=action, detail=detail, decided_at=now)
            )
            await session.commit()
        return True

    async def _finish(
        self,
        draft_id: int,
        status: DraftStatus,
        action: str,
        detail: str | None = None,
        note: str | None = None,
        export_action: ExportAction | None = None,
    ) -> bool:
        """Decide a draft once. A repeated press of an old button changes nothing."""
        now = datetime.now(UTC)
        async with self._sessions() as session:
            result = await session.execute(
                update(Draft)
                .where(Draft.id == draft_id, Draft.status.in_(UNDECIDED))
                .values(
                    status=status, export_action=export_action, needs_refresh=True, updated_at=now
                )
            )
            if result.rowcount == 0:
                await session.rollback()
                await self._say(f"По задаче №{draft_id} решение уже принято.")
                return False
            session.add(
                DraftDecision(
                    draft_id=draft_id, decision=action, detail=detail, note=note, decided_at=now
                )
            )
            await session.commit()
        await self._reset()
        await self._flush()
        return True

    # Approving

    async def _append_asked(self, draft_id: str) -> None:
        card = await self._open_card(draft_id)
        if card is None:
            return
        if not card.can_append:
            await self._say("Эту задачу сейчас дополнить нельзя. Можно создать новую.")
            return
        await self._approve_flow(card, "apc")

    async def _append_confirmed(self, draft_id: str) -> None:
        card = await self._open_card(draft_id)
        if card is None:
            return
        if not card.can_append:
            await self._say("Эту задачу сейчас дополнить нельзя. Можно создать новую.")
            return
        await self._finish(
            card.id,
            DraftStatus.APPROVED,
            APPROVED_ACTION,
            detail="append",
            export_action=ExportAction.APPEND,
        )

    async def _approve_asked(self, draft_id: str) -> None:
        card = await self._open_card(draft_id)
        if card is None:
            return
        if card.parent_waiting:
            await self._say("Исходная задача ещё отправляется в Jira. Подождите немного.")
            return
        await self._approve_flow(card, "okc")

    async def _approve_flow(self, card: DraftCard, confirm: str) -> None:
        if card.has_irreversible:
            await self._say(
                "В тексте есть <code>&lt;SECRET&gt;</code> или <code>&lt;CARD&gt;</code>. "
                "Их значение нельзя восстановить: сначала уберите их через «Править»."
            )
            return
        if not card.values:
            await self._handlers()[confirm](str(card.id))
            return
        lines = [
            f"<code>{escape(mark)}</code> → <b>{escape(value)}</b>"
            if value
            else f"<code>{escape(mark)}</code> → значение неизвестно, останется как есть"
            for mark, value in card.values.items()
        ]
        await self._ask(
            "В задаче остаются плейсхолдеры. В Jira вместо них будут настоящие значения:\n"
            + "\n".join(lines),
            [
                [
                    Button("✅ Подтвердить", f"{confirm}:{card.id}"),
                    Button("✏️ Править", f"ed:{card.id}"),
                ]
            ],
        )

    async def _approve_confirmed(self, draft_id: str) -> None:
        await self._finish(
            int(draft_id), DraftStatus.APPROVED, APPROVED_ACTION, export_action=ExportAction.CREATE
        )

    # Rejecting

    async def _reject_asked(self, draft_id: str) -> None:
        card = await self._open_card(draft_id)
        if card is None:
            return
        await self._ask(
            f"Почему отклонить задачу №{card.id}?",
            [[Button(name, f"rjr:{card.id}:{code}")] for code, name in REJECTION_REASONS.items()]
            + [self._cancel_row()],
        )

    async def _reject_reason(self, draft_id: str, reason: str) -> None:
        if reason not in REJECTION_REASONS:
            return
        if reason == OTHER_REASON:
            card = await self._open_card(draft_id)
            if card is None:
                return
            await self._reset()
            self._waiting = _WaitingNote(card.id)
            await self._ask(
                "Напишите причину следующим сообщением или пропустите.",
                [[Button("Пропустить", f"rjs:{card.id}")], self._cancel_row()],
            )
            return
        await self._finish(int(draft_id), DraftStatus.REJECTED, REJECTED_ACTION, reason)

    async def _reject_without_note(self, draft_id: str) -> None:
        await self._finish(int(draft_id), DraftStatus.REJECTED, REJECTED_ACTION, OTHER_REASON)

    async def _reject(self, draft_id: int, reason: str, note: str) -> None:
        # The reason later reaches the model together with the draft, so it's anonymized as well.
        async with self._sessions() as session:
            anonymized = await self._names.anonymize(session, note)
            await session.commit()
        await self._finish(draft_id, DraftStatus.REJECTED, REJECTED_ACTION, reason, anonymized)

    # Editing by hand

    async def _edit_asked(self, draft_id: str) -> None:
        card = await self._open_card(draft_id)
        if card is None:
            return
        await self._ask(
            f"Что править в задаче №{card.id}?",
            [
                [
                    Button("Заголовок", f"edf:{card.id}:title"),
                    Button("Описание", f"edf:{card.id}:description"),
                ],
                [
                    Button("Приоритет", f"edf:{card.id}:priority"),
                    Button("Срок", f"edf:{card.id}:deadline"),
                ],
                self._cancel_row(),
            ],
        )

    async def _edit_field(self, draft_id: str, field: str) -> None:
        card = await self._open_card(draft_id)
        if card is None:
            return
        await self._reset()
        if field == "priority":
            await self._ask(
                "Выберите приоритет:",
                [
                    [Button(name, f"edp:{card.id}:{code}") for code, name in PRIORITIES.items()],
                    self._cancel_row(),
                ],
            )
            return
        if field == "deadline":
            self._waiting = _WaitingField(card.id, field)
            await self._ask(
                f"Текущий срок: {card.deadline or 'нет'}.\n"
                "Пришлите дату как ГГГГ-ММ-ДД или «-», чтобы убрать.",
                [self._cancel_row()],
            )
            return
        if field not in ("title", "description"):
            return
        self._waiting = _WaitingField(card.id, field)
        current = card.title if field == "title" else card.description
        await self._ask(
            "Текущий текст, плейсхолдеры можно оставить как есть:\n"
            f"<code>{escape(current[:3500])}</code>\n\nПришлите новый текст следующим сообщением.",
            [self._cancel_row()],
        )

    async def _edit_priority(self, draft_id: str, priority: str) -> None:
        if priority not in PRIORITIES:
            return
        await self._apply_edit(int(draft_id), "priority", priority)

    async def _field_typed(self, waiting: _WaitingField, text: str) -> None:
        typed = text.strip()
        if waiting.field == "deadline":
            if typed != "-" and not _DATE.match(typed):
                await self._say("Нужна дата как ГГГГ-ММ-ДД или «-». Попробуйте ещё раз.")
                return
            await self._apply_edit(waiting.draft_id, "deadline", None if typed == "-" else typed)
            return
        if not typed:
            await self._say("Текст пустой. Пришлите текст или нажмите «Отмена».")
            return
        if waiting.field == "title" and len(typed) > MAX_TITLE:
            await self._say(f"Заголовок длиннее {MAX_TITLE} символов. Сократите его.")
            return
        async with self._sessions() as session:
            anonymized = await self._names.anonymize(session, typed)
            await session.commit()
        if anonymized == typed:
            await self._apply_edit(waiting.draft_id, waiting.field, typed)
            return
        self._waiting = None
        self._name_choice = _NameChoice(waiting.draft_id, waiting.field, typed, anonymized)
        await self._ask(
            "Похоже, в тексте есть имена или другие личные данные. Заменить на плейсхолдеры?\n"
            f"<code>{escape(anonymized[:3500])}</code>",
            [
                [
                    Button("Заменить", f"edn:{waiting.draft_id}:yes"),
                    Button("Оставить как написал", f"edn:{waiting.draft_id}:no"),
                ],
                self._cancel_row(),
            ],
        )

    async def _name_chosen(self, draft_id: str, answer: str) -> None:
        choice = self._name_choice
        if choice is None or choice.draft_id != int(draft_id):
            await self._say("Этот выбор уже не актуален.")
            return
        text = choice.anonymized if answer == "yes" else choice.typed
        await self._apply_edit(choice.draft_id, choice.field, text)

    async def _apply_edit(self, draft_id: int, field: str, value: str | None) -> None:
        await self._reset()
        values: dict[str, object] = {field: value}
        if field == "deadline":
            values["deadline_by_owner"] = True
        changed = await self._change_text(draft_id, values, EDITED_ACTION, field)
        if not changed:
            await self._say(f"По задаче №{draft_id} решение уже принято.")
            return
        await self._flush()

    # Editing through the model

    async def _model_asked(self, draft_id: str, more: str | None = None) -> None:
        """With `more`, the next request builds on the proposal shown, not on the saved text."""
        card = await self._open_card(draft_id)
        if card is None:
            return
        # Asking for more keeps the proposal on screen, so its wording can be read while typing.
        if not more:
            await self._reset()
        self._waiting = _WaitingInstruction(card.id)
        await self._ask(
            "Напишите, что изменить в тексте. Например: «короче» или «добавь, что нужен макет».",
            [self._cancel_row()],
        )

    async def _ask_model(self, draft_id: int, instruction: str) -> None:
        self._waiting = None
        current = (
            self._proposal.text if self._proposal and self._proposal.draft_id == draft_id else None
        )
        async with self._sessions() as session:
            draft = await session.get(Draft, draft_id, populate_existing=True)
            if draft is None or draft.status not in UNDECIDED:
                await self._say(f"По задаче №{draft_id} решение уже принято.")
                return
            title = current.title if current else draft.title
            description = current.description if current else draft.description
            anonymized = await self._names.anonymize(session, instruction)
            await session.commit()
            lines = await draft_lines(session, draft_id, await self._rules(session))
        if lines is None:
            await self._say("Сообщения этой задачи ещё обезличиваются. Попробуйте через минуту.")
            return
        await self._ask("Спрашиваю модель…")
        try:
            text = await self._editor.edit_text(lines, title, description, anonymized)
        except OllamaUnavailableError:
            await self._say("Модель недоступна: возможно, ПК выключен. Черновик не изменён.")
            return
        except AnswerError:
            await self._say("Модель не дала понятного ответа. Черновик не изменён, попробуйте ещё.")
            return
        self._proposal = _Proposal(draft_id, text)
        await self._ask(
            f"<b>{escape(text.title)}</b>\n{escape(text.description[:3000])}",
            [
                [
                    Button("✅ Принять", f"aia:{draft_id}"),
                    Button("✨ Ещё правка", f"ai:{draft_id}:more"),
                ],
                self._cancel_row(),
            ],
        )

    async def _proposal_accepted(self, draft_id: str) -> None:
        proposal = self._proposal
        if proposal is None or proposal.draft_id != int(draft_id):
            await self._say("Это предложение уже не актуально.")
            return
        values = {"title": proposal.text.title, "description": proposal.text.description}
        await self._reset()
        changed = await self._change_text(
            proposal.draft_id, values, MODEL_EDITED_ACTION, "title,description"
        )
        if not changed:
            await self._say(f"По задаче №{draft_id} решение уже принято.")
            return
        await self._flush()

    # Looking

    async def _source(self, draft_id: str) -> None:
        async with self._sessions() as session:
            parts = await views.source_parts(session, int(draft_id), self._offset)
        for message in split_message(parts) or ["Сообщений нет."]:
            await self._say(message)

    async def _attachments(self, draft_id: str) -> None:
        card = await self._card(int(draft_id))
        if card is None:
            return
        if not card.files:
            await self._say("Вложений нет.")
            return
        rows = []
        for file in card.files:
            mark = "🚫" if file.excluded else "📎"
            name = file.name or file.kind
            label = f"{mark} {name} ({format_size(file.size_bytes)})"
            rows.append(
                [
                    Button(label[:60], f"atg:{card.id}:{file.id}"),
                    Button("👁", f"fv:{card.id}:{file.id}"),
                ]
            )
        await self._say("Нажмите на файл, чтобы исключить его или вернуть. 👁 покажет файл.", rows)

    async def _attachment_toggled(self, draft_id: str, file_id: str) -> None:
        async with self._sessions() as session:
            file = await session.get(Attachment, int(file_id))
            if file is None:
                return
            if file.kind in OPT_IN_KINDS:
                file.opted_in = not file.opted_in
            else:
                file.excluded = not file.excluded
            await session.execute(
                update(Draft)
                .where(Draft.id == int(draft_id))
                .values(needs_refresh=True, updated_at=datetime.now(UTC))
            )
            await session.commit()
        await self._attachments(draft_id)
        await self._flush()

    async def _open(self, draft_id: str) -> None:
        """Show a task at the bottom of the chat, replacing its card if it still has one."""
        async with self._sessions() as session:
            await session.execute(
                update(Draft)
                .where(Draft.id == int(draft_id))
                .values(needs_refresh=True, updated_at=datetime.now(UTC))
            )
            await session.commit()
        await self._flush()

    async def _clear(self) -> None:
        removed, too_old = await self._messenger.clear()
        async with self._sessions() as session:
            counters = await views.queue_counters(session)
        waiting = counters.get("drafts_sent", 0)
        lines = [f"Чат очищен, удалено сообщений: {removed}."]
        if too_old:
            lines.append(
                f"Telegram не дал удалить сообщения старше 48 часов: {too_old}. "
                "Их можно удалить вручную."
            )
        if waiting:
            lines.append(f"Статусы не менялись. Ждут решения: {waiting}, откройте через /list.")
        await self._say("\n".join(lines))

    async def _list(self) -> None:
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(Draft.id, Draft.title)
                    .where(Draft.status.in_(UNDECIDED))
                    .order_by(Draft.last_message_at)
                )
            ).all()
        if not rows:
            await self._say("Нерешённых черновиков нет.")
            return
        keyboard = [[Button(f"№{row.id} {row.title}"[:60], f"open:{row.id}")] for row in rows[:50]]
        await self._say(f"Нерешённых черновиков: {len(rows)}", keyboard)

    async def _status(self) -> None:
        async with self._sessions() as session:
            counters = await views.queue_counters(session)
        model = "доступна" if self._ollama_available() else "недоступна"
        drafted, rejected = counters.get("day_draft", 0), counters.get("day_no_draft", 0)
        noise, broken = counters.get("day_skipped", 0), counters.get("day_failed", 0)
        await self._say(
            f"Модель: {model}\n"
            f"В очереди на анализ: {counters['pending']}\n"
            f"Не разобрано: {counters['failed']}\n"
            f"Проанализировано за сутки: {drafted + rejected + noise + broken}\n"
            f"• в черновик: {drafted}\n"
            f"• отклонено моделью: {rejected}, посмотреть: /recent\n"
            f"• отсеяно как шум: {noise}\n"
            f"• с ошибкой: {broken}\n"
            f"Черновики: ждут решения {counters.get('drafts_sent', 0)}, "
            f"приняты {counters.get('drafts_approved', 0)}, "
            f"отклонены {counters.get('drafts_rejected', 0)}"
        )

    async def _recent(self, offset: str = "0") -> None:
        start = max(int(offset), 0)
        async with self._sessions() as session:
            items, more = await views.rejected_messages(session, RECENT_PAGE, start)
        if not items:
            await self._say("Больше таких сообщений нет." if start else "Таких сообщений нет.")
            return
        parts, keyboard = [], []
        for number, item in enumerate(items, start + 1):
            text = item.text or "(без текста)"
            preview = text if len(text) <= RECENT_PREVIEW else f"{text[:RECENT_PREVIEW]}…"
            parts.append(
                f"<b>{number}.</b> {format_time(item.sent_at, self._offset)} · "
                f"{escape(item.chat_title or 'чат')} · {escape(item.sender)}\n{escape(preview)}"
            )
            keyboard.append([Button(f"➕ Сделать задачей: {number}", f"mk:{item.message_id}")])
        if more:
            keyboard.append([Button("Ещё", f"rc:{start + RECENT_PAGE}")])
        await self._say(
            "Модель решила, что это не задачи. Новые сверху:\n\n" + "\n\n".join(parts), keyboard
        )

    async def _make_task(self, message_id: str) -> None:
        """A draft from a message the model turned down. Its text still comes from the model.

        Jira gets only the model's rephrasing, never the chat's own words, so without the model
        nothing is created.
        """
        key = int(message_id)
        if key in self._making:
            await self._say("Черновик по этому сообщению уже готовится.")
            return
        self._making.add(key)
        try:
            await self._make_task_once(key)
        finally:
            self._making.discard(key)

    async def _make_task_once(self, message_id: int) -> None:
        async with self._sessions() as session:
            if await self._already_drafted(session, message_id):
                return
            message = await session.get(Message, message_id)
            if message is None:
                await self._say("Сообщение не найдено.")
                return
            thread = await build_thread(session, message, await self._rules(session))
        if thread is None:
            await self._say("Сообщение ещё обезличивается. Попробуйте через минуту.")
            return
        await self._ask("Модель пишет черновик…")
        try:
            # The reply chain explains what the message answers, so the text doesn't lose it.
            text = await self._editor.rewrite([*thread.reply_chain, thread.target])
        except OllamaUnavailableError:
            await self._say("Модель недоступна: возможно, ПК выключен. Черновик не создан.")
            return
        if text is None:
            await self._say("Модель не дала понятного ответа. Черновик не создан, попробуйте ещё.")
            return
        now = datetime.now(UTC)
        async with self._sessions() as session:
            if await self._already_drafted(session, message_id):
                return
            message = await session.get_one(Message, message_id)
            draft = Draft(
                chat_id=message.chat_id,
                title=text.title,
                description=text.description,
                priority="medium",
                has_placeholders=has_placeholder(text.title) or has_placeholder(text.description),
                kind=DraftKind.TASK,
                model=self._editor.model,
                prompt_id=views.OWNER_PROMPT_ID,
                last_message_at=message.sent_at,
                created_at=now,
                updated_at=now,
            )
            session.add(draft)
            await session.flush()
            session.add(
                DraftMessage(draft_id=draft.id, message_id=message_id, role=DraftRole.TRIGGER)
            )
            session.add(
                DraftDecision(draft_id=draft.id, decision=OWNER_CREATED_ACTION, decided_at=now)
            )
            await session.commit()
        await self._reset()
        await self._flush()

    async def _already_drafted(self, session: AsyncSession, message_id: int) -> bool:
        draft_id = await views.draft_of(session, message_id)
        if draft_id is None:
            return False
        await self._say(
            f"Это сообщение уже в задаче №{draft_id}.", [[Button("Открыть", f"open:{draft_id}")]]
        )
        return True

    async def _retry(self) -> None:
        async with self._sessions() as session:
            result = await session.execute(
                update(Message)
                .where(Message.status == MessageStatus.FAILED)
                .values(status=MessageStatus.PENDING)
            )
            await session.commit()
        await self._say(f"Возвращено в очередь: {result.rowcount}.")

    # Jira

    async def _view_file(self, draft_id: str, file_id: str) -> None:
        async with self._sessions() as session:
            file = await session.get(Attachment, int(file_id))
            if file is None:
                return
            if file.kind in HIDDEN_KINDS:
                await self._say("Такие файлы не показываются.")
                return
            if (file.size_bytes or 0) > MAX_BOT_FILE_BYTES:
                await self._say(
                    "Файл больше 50 МБ, бот не может его отправить. Откройте его в «Источнике»."
                )
                return
            session.add(
                FileRequest(
                    attachment_id=file.id,
                    draft_id=int(draft_id),
                    purpose=VIEW_PURPOSE,
                    status=FileRequestStatus.PENDING,
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
        await self._say("Скачиваю файл, это может занять немного времени.")

    async def _export_old(self) -> None:
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(Draft.id, Draft.title).where(
                        Draft.status == DraftStatus.APPROVED, Draft.export_action.is_(None)
                    )
                )
            ).all()
        if not rows:
            await self._say("Таких задач нет.")
            return
        await self._say(
            "Эти задачи приняты до подключения Jira и сами не уйдут. Нажмите, чтобы отправить:",
            [[Button(f"№{row.id} {row.title}"[:60], f"xo:{row.id}")] for row in rows[:50]],
        )

    async def _export_one_old(self, draft_id: str) -> None:
        async with self._sessions() as session:
            result = await session.execute(
                update(Draft)
                .where(
                    Draft.id == int(draft_id),
                    Draft.status == DraftStatus.APPROVED,
                    Draft.export_action.is_(None),
                )
                .values(export_action=ExportAction.CREATE)
            )
            await session.commit()
        await self._say(
            f"Задача №{draft_id} отправляется в Jira." if result.rowcount else "Уже отправляется."
        )

    async def _export_retry(self, draft_id: str) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(Draft)
                .where(Draft.id == int(draft_id), Draft.status == DraftStatus.APPROVED)
                .values(export_error=None, needs_refresh=True, updated_at=datetime.now(UTC))
            )
            await session.commit()
        await self._flush()

    async def _uploads_retry(self, draft_id: str) -> None:
        """Only files that failed: the task itself and uploaded files stay as they are."""
        async with self._sessions() as session:
            await session.execute(
                delete(AttachmentUpload).where(
                    AttachmentUpload.draft_id == int(draft_id), AttachmentUpload.error.is_not(None)
                )
            )
            await session.execute(
                delete(FileRequest).where(
                    FileRequest.draft_id == int(draft_id), FileRequest.purpose == "upload"
                )
            )
            await session.execute(
                update(Draft).where(Draft.id == int(draft_id)).values(files_done=False)
            )
            await session.commit()
        await self._say("Файлы загружаются повторно.")

    # Anonymizer mistakes

    async def _placeholders(self, draft_id: str) -> None:
        async with self._sessions() as session:
            found = await views.markable(session, int(draft_id))
        if not found:
            await self._say("Здесь нет плейсхолдеров, которые можно отметить.")
            return
        lines = [f"<code>{escape(m.placeholder)}</code> → <b>{escape(m.value)}</b>" for m in found]
        await self._say(
            "Если слово скрыто по ошибке, нажмите на него:\n" + "\n".join(lines),
            [[Button(f"Это не имя: {m.value}"[:60], f"phx:{m.id}")] for m in found],
        )

    async def _mark_asked(self, pseudonym_id: str) -> None:
        async with self._sessions() as session:
            row = await session.get(Pseudonym, int(pseudonym_id))
        if row is None:
            await self._say("Это значение уже не скрывается.")
            return
        await self._say(
            f"Слово <b>{escape(row.value)}</b> станет читаемым во всех сообщениях: "
            "и для модели, и в текстах задач. Точно?",
            [[Button("Да, это не имя", f"phy:{row.id}"), Button("Отмена", "cancel")]],
        )

    async def _mark_confirmed(self, pseudonym_id: str) -> None:
        now = datetime.now(UTC)
        async with self._sessions() as session:
            row = await session.get(Pseudonym, int(pseudonym_id))
            if row is None or row.kind not in views.MARKABLE_KINDS:
                await self._say("Это значение уже не скрывается.")
                return
            if row.key.startswith(views.KNOWN_PERSON_KEY):
                await self._say("Это настоящий человек из Telegram, его отметить нельзя.")
                return
            value = row.value
            placeholder = f"<{row.kind}_{row.number}>"
            await session.execute(
                insert(AnonymizerOverride)
                .values(term=value, created_at=now)
                .on_conflict_do_nothing(index_elements=[AnonymizerOverride.term])
            )
            # In one transaction with the override: once rules change, cleanup may remove this
            # placeholder from the map, and a draft still using it could no longer be read.
            for draft in await session.scalars(select(Draft)):
                if placeholder in draft.title or placeholder in draft.description:
                    draft.title = draft.title.replace(placeholder, value)
                    draft.description = draft.description.replace(placeholder, value)
                    draft.has_placeholders = has_placeholder(draft.title) or has_placeholder(
                        draft.description
                    )
                    draft.needs_refresh = True
                    draft.updated_at = now
            for decision in await session.scalars(
                select(DraftDecision).where(DraftDecision.note.contains(placeholder))
            ):
                decision.note = decision.note.replace(placeholder, value)
            await session.commit()
        await self._reset()
        await self._say(
            f"Готово: <b>{escape(value)}</b> больше не скрывается. Сообщения пересобираются, "
            "анализ подождёт. Снять отметку можно в /overrides."
        )
        await self._flush()

    async def _overrides(self) -> None:
        async with self._sessions() as session:
            rows = list(
                await session.scalars(select(AnonymizerOverride).order_by(AnonymizerOverride.id))
            )
        if not rows:
            await self._say("Отмеченных слов нет.")
            return
        await self._say(
            "Эти слова не скрываются. Нажмите, чтобы снова скрывать:",
            [[Button(f"Вернуть: {row.term}"[:60], f"ovd:{row.id}")] for row in rows],
        )

    async def _override_removed(self, override_id: str) -> None:
        async with self._sessions() as session:
            result = await session.execute(
                delete(AnonymizerOverride).where(AnonymizerOverride.id == int(override_id))
            )
            await session.commit()
        await self._say(
            "Отметка снята, сообщения пересобираются. В текстах уже созданных черновиков "
            "слово останется."
            if result.rowcount
            else "Отметка уже снята."
        )
