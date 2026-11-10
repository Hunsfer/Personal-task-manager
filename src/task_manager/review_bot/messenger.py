"""What the bot needs from Telegram, so its logic doesn't depend on aiogram."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Button:
    text: str
    data: str


Keyboard = list[list[Button]]

# Shown in the bot's menu button, so they needn't be typed.
COMMANDS = (
    ("list", "Нерешённые задачи"),
    ("recent", "Что модель не сочла задачей"),
    ("status", "Очередь и модель"),
    ("overrides", "Слова, отмеченные как не имена"),
    ("clear", "Очистить переписку, статусы не меняются"),
    ("export_old", "Задачи, принятые до подключения Jira"),
)


class MessengerError(Exception):
    """Telegram can't be used right now. Its message is an error type, never message text."""


class RetryLater(MessengerError):
    """Telegram asked to wait."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"retry after {seconds} seconds")
        self.seconds = seconds


class MessageGone(MessengerError):
    """The message to edit no longer exists, for example because the chat was cleared."""


class Messenger(Protocol):
    async def send(
        self, text: str, keyboard: Keyboard | None = None, silent: bool = False
    ) -> int: ...
    async def edit(self, message_id: int, text: str, keyboard: Keyboard | None = None) -> None: ...
    async def delete(self, message_id: int) -> None: ...
    async def send_file(self, path: Path, name: str) -> int: ...
