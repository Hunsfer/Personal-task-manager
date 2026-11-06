"""Read-only wrapper around Telethon, and the only module allowed to import it.

The Telegram session belongs to the user's personal account, so this module must never send,
edit, delete or forward anything, or mark anything as read. tests/test_telegram.py enforces it.
"""

import json
import os
from collections.abc import AsyncIterator, Collection, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from telethon import TelegramClient, errors, events, utils
from telethon.sessions import StringSession
from telethon.tl import types

from task_manager.collector.types import (
    AttachmentInfo,
    ChatInfo,
    IncomingMessage,
    MessageHandler,
    NotAuthorizedError,
    PersonInfo,
    RateLimitedError,
)
from task_manager.storage.models import ChatKind

RETRY_DELAY_S = 5
PARTICIPANTS_LIMIT = 5000
# Most specific first: a video is also a document.
ATTACHMENT_KINDS = ("photo", "sticker", "gif", "voice", "video_note", "video", "audio", "document")


def chat_kind(entity: Any) -> ChatKind:
    if isinstance(entity, types.User):
        return ChatKind.PRIVATE
    if isinstance(entity, types.Channel):
        return ChatKind.SUPERGROUP if entity.megagroup else ChatKind.CHANNEL
    if isinstance(entity, types.Chat):
        return ChatKind.GROUP
    return ChatKind.UNKNOWN


def message_link(chat: ChatInfo, msg_id: int) -> str | None:
    """Only supergroups and channels have message links."""
    if chat.kind not in (ChatKind.SUPERGROUP, ChatKind.CHANNEL):
        return None
    if chat.username:
        return f"https://t.me/{chat.username}/{msg_id}"
    return f"https://t.me/c/{utils.resolve_id(chat.id)[0]}/{msg_id}"


def _attachments(message: types.Message) -> tuple[AttachmentInfo, ...]:
    # Telethon's message.photo and message.file also expose a link preview's image, which isn't
    # something the sender attached.
    file = None if isinstance(message.media, types.MessageMediaWebPage) else message.file
    if file is None:
        return ()
    kind = next((name for name in ATTACHMENT_KINDS if getattr(message, name, None)), "document")
    return (AttachmentInfo(kind=kind, file_name=file.name, size_bytes=file.size),)


def code_entities(message: types.Message) -> str | None:
    """Which ranges of the text Telegram shows as code. Offsets are in UTF-16 code units."""
    found = [
        {"offset": entity.offset, "length": entity.length}
        for entity in message.entities or ()
        if isinstance(entity, types.MessageEntityCode | types.MessageEntityPre)
    ]
    return json.dumps(found) if found else None


def person_info(entity: Any) -> PersonInfo | None:
    """Real people only. A bot's name isn't personal data and would mask ordinary words."""
    if not isinstance(entity, types.User) or entity.bot:
        return None
    return PersonInfo(entity.id, entity.first_name, entity.last_name, entity.username)


def to_incoming(message: Any, chat: ChatInfo) -> IncomingMessage | None:
    """Convert a Telethon message. Returns None for service messages such as a title change."""
    if not isinstance(message, types.Message):
        return None
    return IncomingMessage(
        chat_id=chat.id,
        msg_id=message.id,
        sent_at=message.date,
        text=message.message or "",
        reply_to_msg_id=message.reply_to_msg_id,
        sender_id=message.sender_id,
        edited_at=message.edit_date,
        mentions_me=bool(message.mentioned),
        is_outgoing=bool(message.out),
        link=message_link(chat, message.id),
        attachments=_attachments(message),
        sender=person_info(message.sender),
        code_entities=code_entities(message),
    )


@contextmanager
def _translate_errors() -> Iterator[None]:
    """Turn Telethon errors that callers must act on into the collector's own exceptions."""
    try:
        yield
    except errors.FloodWaitError as error:
        raise RateLimitedError(error.seconds) from None
    except errors.UnauthorizedError:
        raise NotAuthorizedError("the Telegram session is no longer valid") from None


class TelegramReader:
    def __init__(
        self, api_id: int, api_hash: str, session_path: Path, session_string: str | None = None
    ) -> None:
        """With a session string nothing granting access to the account is written to disk.

        Telethon then keeps its entity cache in memory, and list_chats refills it at startup.
        Missed messages don't depend on it: catch-up reads from the collector's own cursor.
        """
        self._session_path = session_path
        self._uses_string = session_string is not None
        if not self._uses_string:
            session_path.parent.mkdir(parents=True, exist_ok=True)
        # connection_retries=None keeps reconnecting through network outages.
        self._client = TelegramClient(
            StringSession(session_string) if self._uses_string else str(session_path),
            api_id,
            api_hash,
            connection_retries=None,
            retry_delay=RETRY_DELAY_S,
        )
        self._chats: dict[int, tuple[ChatInfo, Any]] = {}

    async def login(self) -> None:
        """Interactive login: asks for the phone number, code and 2FA password in the terminal."""
        with _translate_errors():
            await self._client.start()
        if not self._uses_string:
            # The session file gives full access to the account.
            os.chmod(self._session_path.with_suffix(".session"), 0o600)

    def export_session(self) -> str:
        """The session as a string, to be kept with the other secrets. It grants full access."""
        return StringSession.save(self._client.session)

    async def connect(self) -> None:
        await self._client.connect()

    async def disconnect(self) -> None:
        await self._client.disconnect()

    def is_connected(self) -> bool:
        return self._client.is_connected()

    async def is_authorized(self) -> bool:
        return await self._client.is_user_authorized()

    async def me(self) -> PersonInfo:
        with _translate_errors():
            me = await self._client.get_me()
        return PersonInfo(me.id, me.first_name, me.last_name, me.username)

    async def list_people(self, chat_id: int) -> list[PersonInfo]:
        """The people in a chat. Large groups may hide their members, which raises an error."""
        entity = self._chats[chat_id][1]
        if isinstance(entity, types.User):
            person = person_info(entity)
            return [person] if person else []
        people = []
        with _translate_errors():
            async for user in self._client.iter_participants(entity, limit=PARTICIPANTS_LIMIT):
                person = person_info(user)
                if person:
                    people.append(person)
        return people

    async def list_chats(self) -> list[ChatInfo]:
        """List all dialogs. This also caches the entities that later requests need."""
        chats: dict[int, tuple[ChatInfo, Any]] = {}
        with _translate_errors():
            async for dialog in self._client.iter_dialogs():
                entity = dialog.entity
                info = ChatInfo(
                    id=dialog.id,
                    title=dialog.name,
                    kind=chat_kind(entity),
                    username=getattr(entity, "username", None),
                )
                chats[dialog.id] = (info, entity)
        self._chats = chats
        return [info for info, _ in chats.values()]

    async def latest_msg_id(self, chat_id: int) -> int:
        with _translate_errors():
            messages = await self._client.get_messages(self._chats[chat_id][1], limit=1)
        return messages[0].id if messages else 0

    def iter_after(self, chat_id: int, msg_id: int) -> AsyncIterator[IncomingMessage]:
        """Messages newer than msg_id, oldest first."""
        return self._iterate(chat_id, min_id=msg_id)

    def iter_since(self, chat_id: int, since: datetime) -> AsyncIterator[IncomingMessage]:
        """Messages sent after the given time, oldest first."""
        return self._iterate(chat_id, offset_date=since)

    async def _iterate(self, chat_id: int, **query: Any) -> AsyncIterator[IncomingMessage]:
        info, entity = self._chats[chat_id]
        with _translate_errors():
            async for message in self._client.iter_messages(entity, reverse=True, **query):
                incoming = to_incoming(message, info)
                if incoming is not None:
                    yield incoming

    async def download(self, chat_id: int, msg_id: int, path: Path) -> bool:
        """Save a message's file. False means the message or its file no longer exists."""
        entity = self._chats[chat_id][1]
        with _translate_errors():
            message = await self._client.get_messages(entity, ids=msg_id)
            if message is None or message.media is None:
                return False
            saved = await message.download_media(file=str(path))
        return saved is not None

    def subscribe(
        self, chat_ids: Collection[int], on_new: MessageHandler, on_edited: MessageHandler
    ) -> None:
        allowed = frozenset(chat_ids)

        def dispatcher(handler: MessageHandler) -> Any:
            async def dispatch(event: Any) -> None:
                # Filter before converting, so other chats' messages are never even processed.
                if event.chat_id not in allowed:
                    return
                incoming = to_incoming(event.message, self._chats[event.chat_id][0])
                if incoming is not None:
                    await handler(incoming)

            return dispatch

        self._client.add_event_handler(dispatcher(on_new), events.NewMessage())
        self._client.add_event_handler(dispatcher(on_edited), events.MessageEdited())
