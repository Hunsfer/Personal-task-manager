"""Telegram data as the collector sees it, independent of Telethon."""

from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from task_manager.storage.models import ChatKind


class RateLimitedError(Exception):
    """Telegram asked us to wait before the next request."""

    def __init__(self, seconds: int) -> None:
        super().__init__(f"rate limited for {seconds} seconds")
        self.seconds = seconds


class NotAuthorizedError(Exception):
    """The Telegram session is missing, or it has been revoked."""


@dataclass(frozen=True)
class ChatInfo:
    id: int
    title: str | None
    kind: ChatKind
    username: str | None = None


@dataclass(frozen=True)
class PersonInfo:
    user_id: int
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None


@dataclass(frozen=True)
class AttachmentInfo:
    kind: str
    file_name: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class IncomingMessage:
    chat_id: int
    msg_id: int
    sent_at: datetime
    text: str = ""
    reply_to_msg_id: int | None = None
    sender_id: int | None = None
    edited_at: datetime | None = None
    mentions_me: bool = False
    is_outgoing: bool = False
    link: str | None = None
    attachments: tuple[AttachmentInfo, ...] = ()
    sender: PersonInfo | None = None
    # JSON list of code ranges, or None when the message has no code.
    code_entities: str | None = None


MessageHandler = Callable[[IncomingMessage], Awaitable[None]]


class Reader(Protocol):
    """Everything the collector may do in Telegram. It only ever reads."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    async def is_authorized(self) -> bool: ...
    async def me(self) -> PersonInfo: ...
    async def list_chats(self) -> list[ChatInfo]: ...
    async def list_people(self, chat_id: int) -> list[PersonInfo]: ...
    async def latest_msg_id(self, chat_id: int) -> int: ...
    def iter_after(self, chat_id: int, msg_id: int) -> AsyncIterator[IncomingMessage]: ...
    def iter_since(self, chat_id: int, since: datetime) -> AsyncIterator[IncomingMessage]: ...
    async def download(self, chat_id: int, msg_id: int, path: Path) -> bool: ...
    def subscribe(
        self, chat_ids: Collection[int], on_new: MessageHandler, on_edited: MessageHandler
    ) -> None: ...
