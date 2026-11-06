"""Builds the model's input. Templates live in config/prompts and contain made-up examples only."""

import hashlib
from datetime import datetime
from pathlib import Path
from string import Template

from task_manager.analyzer.thread import Thread, ThreadLine
from task_manager.common.config import ConfigError, ScopeConfig
from task_manager.storage.models import ChatKind

_CHAT_KINDS = {
    ChatKind.PRIVATE: "личные сообщения",
    ChatKind.GROUP: "группа",
    ChatKind.SUPERGROUP: "группа",
    ChatKind.CHANNEL: "канал",
    ChatKind.UNKNOWN: "неизвестно",
}
_WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def load_template(path: Path) -> str:
    if not path.is_file():
        raise ConfigError(f"{path} not found")
    return path.read_text(encoding="utf-8")


def template_id(path: Path, text: str) -> str:
    """Names a prompt by content, so an edited prompt never reuses older results."""
    return f"{path.stem}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def render_system(template: str, scope: ScopeConfig) -> str:
    sections = "\n".join(
        f"- {section.name}: {', '.join(section.aliases)}" for section in scope.sections
    )
    try:
        return Template(template).substitute(role=", ".join(scope.role), sections=sections)
    except (KeyError, ValueError) as error:
        raise ConfigError(f"invalid prompt template: {error}") from None


def format_line(line: ThreadLine) -> str:
    sent = datetime.fromisoformat(line.sent_at)
    stamp = f"{sent:%Y-%m-%d} {_WEEKDAYS[sent.weekday()]} {sent:%H:%M}"
    reply = f" (ответ на {line.reply_to})" if line.reply_to is not None else ""
    parts = [f"[{line.msg_id}] {stamp} {line.sender or 'неизвестно'}{reply}:"]
    parts.append(line.text or "(без текста)")
    if line.files:
        parts.append(f"Файлы: {', '.join(line.files)}")
    return "\n".join(parts)


def _block(title: str, lines: list[ThreadLine]) -> list[str]:
    return [title, *(format_line(line) for line in lines), ""] if lines else []


def render_user(thread: Thread) -> str:
    """Untrusted chat text goes here, never through the template's substitution."""
    return "\n".join(
        [
            f"Тип чата: {_CHAT_KINDS[thread.chat_kind]}",
            f"Telegram отметил упоминание <ME>: {'да' if thread.mentions_me else 'нет'}",
            "",
            *_block("Цепочка ответов:", thread.reply_chain),
            *_block("Предыдущие сообщения чата:", thread.previous),
            "Оцени только это сообщение:",
            format_line(thread.target),
        ]
    )
