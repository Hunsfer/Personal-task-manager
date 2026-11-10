"""Turns drafts into the bot's HTML. Everything from chats and the model is escaped here."""

import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from task_manager.review_bot.messenger import Button, Keyboard
from task_manager.storage.models import DraftKind, DraftStatus

# Telegram's limit is 4096 characters after parsing. These leave room for the rest of the message.
MAX_TEXT = 2600
MAX_CODE = 800
MAX_MESSAGE = 3800

REJECTION_REASONS = {
    "not_task": "Не задача",
    "not_mine": "Не мне",
    "duplicate": "Дубль",
    "outdated": "Не актуально",
    "other": "Другое",
}
PRIORITIES = {"high": "высокий", "medium": "обычный", "low": "низкий"}
STATUS_NAMES = {
    DraftStatus.APPROVED: "✅ Принято, ждёт отправки в Jira",
    DraftStatus.REJECTED: "❌ Отклонено",
    DraftStatus.EXPORTED: "🟢 В Jira",
}
REASON_NAMES = {
    "mention": "упоминание",
    "reply": "ответ на ваше сообщение",
    "dm": "личные сообщения",
    "role": "фронтенд",
    "section": "ваш раздел",
    "promise": "ваше обещание",
    "owner": "вы отметили сами",
}
_MARK = re.compile(r"<(?:(?:PERSON|ORG|LOC|PHONE|EMAIL|URL|IP|ID)_\d+|SECRET|CARD)>")
_IRREVERSIBLE = ("<SECRET>", "<CARD>")


@dataclass(frozen=True)
class FileInfo:
    id: int
    kind: str
    name: str | None
    size_bytes: int | None
    excluded: bool


@dataclass(frozen=True)
class DraftCard:
    id: int
    status: DraftStatus
    kind: DraftKind
    parent_id: int | None
    title: str
    description: str
    deadline: str | None
    priority: str
    reason: str | None
    chat_title: str | None
    last_message_at: datetime
    message_count: int
    text_is_stale: bool
    rejection: str | None
    # Placeholders in the text with what they stand for. None means nothing can be restored.
    values: dict[str, str | None]
    code: tuple[str, ...]
    files: tuple[FileInfo, ...]
    deadline_by_owner: bool = False
    jira_key: str | None = None
    jira_link: str | None = None
    export_error: str | None = None
    parent_key: str | None = None
    parent_status: str | None = None
    parent_locked: bool = False
    # The original is on its way to Jira, so there's nothing to extend or link to yet.
    parent_waiting: bool = False

    @property
    def decided(self) -> bool:
        return self.status in (DraftStatus.APPROVED, DraftStatus.REJECTED, DraftStatus.EXPORTED)

    @property
    def can_append(self) -> bool:
        return bool(self.parent_key and self.parent_status) and not self.parent_locked

    @property
    def has_irreversible(self) -> bool:
        return any(mark in self.title + self.description for mark in _IRREVERSIBLE)

    @property
    def restorable(self) -> dict[str, str]:
        return {mark: value for mark, value in self.values.items() if value is not None}


def escape(text: str) -> str:
    return html.escape(text, quote=False)


def find_marks(text: str) -> list[str]:
    return list(dict.fromkeys(_MARK.findall(text)))


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"


def highlight(text: str, values: dict[str, str | None]) -> str:
    """Escaped text where a placeholder shows its real value, visibly, next to the placeholder."""
    parts = []
    position = 0
    for match in _MARK.finditer(text):
        parts.append(escape(text[position : match.start()]))
        mark = escape(match.group())
        value = values.get(match.group())
        parts.append(
            f"<b>{escape(value)}</b> <code>{mark}</code>"
            if value
            else f"<b><code>{mark}</code></b>"
        )
        position = match.end()
    parts.append(escape(text[position:]))
    return "".join(parts)


def extract_code(text: str, entities: str | None) -> list[str]:
    """The pieces of a message marked as code. Telegram counts offsets in UTF-16 code units."""
    if not entities:
        return []
    encoded = text.encode("utf-16-le")
    return [
        encoded[item["offset"] * 2 : (item["offset"] + item["length"]) * 2].decode(
            "utf-16-le", errors="ignore"
        )
        for item in json.loads(entities)
    ]


def format_size(size: int | None) -> str:
    if size is None:
        return "размер неизвестен"
    if size < 1024 * 1024:
        return f"{max(1, size // 1024)} КБ"
    return f"{size / (1024 * 1024):.1f} МБ"


def format_time(moment: datetime, utc_offset_hours: int) -> str:
    return f"{moment + timedelta(hours=utc_offset_hours):%d.%m %H:%M}"


def _heading(card: DraftCard) -> str:
    if card.kind is DraftKind.CHANGE_REQUEST:
        return f"🔁 Запрос на изменение задачи №{card.parent_id}"
    if card.parent_id is not None:
        return f"♻️ Снова: задача №{card.parent_id} была отклонена"
    return f"📝 Задача №{card.id}"


def _parent_line(card: DraftCard) -> str:
    if card.parent_waiting:
        return "Исходная задача ещё отправляется в Jira, кнопки появятся после этого."
    if card.parent_key is None:
        return "Исходной задачи нет в Jira, можно только создать новую."
    key = escape(card.parent_key)
    if card.parent_status is None:
        return f"Исходная задача {key}: статус пока неизвестен."
    status = escape(card.parent_status)
    if card.parent_locked:
        return f"Исходная задача {key} в статусе «{status}»: её менять нельзя."
    return f"Исходная задача {key}, статус «{status}»."


def card_html(card: DraftCard, utc_offset_hours: int) -> str:
    lines = [f"{_heading(card)}", ""]
    lines.append(f"<b>{highlight(_cut(card.title, 200), card.values)}</b>")
    lines.append(highlight(_cut(card.description, MAX_TEXT), card.values))
    lines.append("")
    details = [f"Приоритет: {PRIORITIES.get(card.priority, card.priority)}"]
    if card.deadline:
        word = "Срок" if card.deadline_by_owner else "Срок (догадка модели, в Jira не уйдёт)"
        details.append(f"{word}: {card.deadline}")
    lines.append(" · ".join(details))
    source = escape(card.chat_title or "чат без названия")
    time = format_time(card.last_message_at, utc_offset_hours)
    lines.append(f"{source} · {time} · сообщений: {card.message_count}")
    if card.reason:
        lines.append(f"Почему: {REASON_NAMES.get(card.reason, escape(card.reason))}")
    if card.kind is DraftKind.CHANGE_REQUEST and not card.decided:
        lines.append(_parent_line(card))
    if card.files:
        included = sum(not file.excluded for file in card.files)
        lines.append(f"📎 Вложений: {len(card.files)}, в задачу войдёт: {included}")
    if card.text_is_stale:
        lines.append("⚠️ Есть новые сообщения, текст по ним не обновлён: см. «Источник».")
    if card.has_irreversible:
        lines.append("⚠️ В тексте скрытое значение, которое нельзя восстановить. Уберите его.")
    budget = MAX_CODE
    for snippet in card.code:
        if budget <= 0:
            lines.append("Есть ещё код: см. «Источник».")
            break
        shown = _cut(snippet, budget)
        budget -= len(shown)
        lines += ["", f"<pre>{escape(shown)}</pre>"]
    if card.code and not card.decided:
        lines.append("Блоки кода уйдут в Jira, пароли в них будут скрыты.")
    if card.status in STATUS_NAMES:
        status = STATUS_NAMES[card.status]
        if card.jira_key and card.jira_link:
            status += f': <a href="{escape(card.jira_link)}">{escape(card.jira_key)}</a>'
        lines += ["", f"<b>{status}</b>"]
        if card.rejection:
            lines.append(escape(card.rejection))
    if card.export_error:
        lines.append(f"⚠️ Jira не приняла задачу: {escape(card.export_error)}")
    return "\n".join(lines)


def card_keyboard(card: DraftCard) -> Keyboard:
    extras = [Button("📄 Источник", f"src:{card.id}"), Button("🏷 Плейсхолдеры", f"ph:{card.id}")]
    if card.decided:
        return [extras]
    reject = Button("❌ Отклонить", f"rj:{card.id}")
    if card.kind is DraftKind.CHANGE_REQUEST:
        first = []
        if card.can_append:
            first.append([Button(f"➕ Дополнить {card.parent_key}", f"ap:{card.id}")])
        if not card.parent_waiting:
            first.append([Button("🆕 Новая задача", f"ok:{card.id}")])
        first.append([reject])
    else:
        first = [[Button("✅ Создать", f"ok:{card.id}"), reject]]
    rows = [
        *first,
        [Button("✏️ Править", f"ed:{card.id}"), Button("✨ Попросить модель", f"ai:{card.id}")],
        extras,
    ]
    if card.files:
        rows.append([Button("📎 Вложения", f"att:{card.id}")])
    return rows


def split_message(parts: list[str]) -> list[str]:
    """Whole parts per message, so a tag is never cut. A single huge part is cut by the caller."""
    messages: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) + 2 > MAX_MESSAGE:
            messages.append(current)
            current = ""
        current = f"{current}\n\n{part}" if current else part
    if current:
        messages.append(current)
    return messages
