"""The conversion runs on real Telethon types, so it fails if their attributes change."""

import ast
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from telethon import errors
from telethon.tl import types

from task_manager.collector.telegram import (
    TelegramReader,
    chat_kind,
    message_link,
    person_info,
    to_incoming,
)
from task_manager.collector.types import (
    ChatInfo,
    NotAuthorizedError,
    PersonInfo,
    RateLimitedError,
)
from task_manager.storage.models import ChatKind

SRC = Path(__file__).parent.parent / "src" / "task_manager"
DATE = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
SUPERGROUP = ChatInfo(-1001234567890, "Чат", ChatKind.SUPERGROUP)
PEER = types.PeerChannel(1234567890)

# Everything in Telethon that writes to the account or marks messages as read.
FORBIDDEN = {
    "action",
    "click",
    "delete",
    "delete_dialog",
    "delete_messages",
    "edit",
    "edit_admin",
    "edit_folder",
    "edit_message",
    "edit_permissions",
    "forward_messages",
    "forward_to",
    "kick_participant",
    "mark_read",
    "pin",
    "pin_message",
    "reply",
    "respond",
    "send_file",
    "send_message",
    "send_read_acknowledge",
    "unpin_message",
}
READER_API = {
    "export_session",
    "download",
    "connect",
    "disconnect",
    "is_authorized",
    "is_connected",
    "iter_after",
    "iter_since",
    "latest_msg_id",
    "list_chats",
    "list_people",
    "login",
    "me",
    "subscribe",
}


def message(**fields) -> types.Message:
    defaults = {"id": 7, "peer_id": PEER, "date": DATE, "message": "текст"}
    return types.Message(**{**defaults, **fields})


def photo() -> types.Photo:
    return types.Photo(
        id=1,
        access_hash=1,
        file_reference=b"",
        date=DATE,
        sizes=[types.PhotoSize(type="x", w=10, h=10, size=100)],
        dc_id=1,
    )


def document(attributes: list, size: int = 2048) -> types.Document:
    return types.Document(
        id=1,
        access_hash=1,
        file_reference=b"",
        date=DATE,
        mime_type="application/pdf",
        size=size,
        dc_id=1,
        attributes=attributes,
    )


def test_message_fields_are_converted() -> None:
    result = to_incoming(
        message(
            mentioned=True,
            out=True,
            edit_date=DATE,
            from_id=types.PeerUser(55),
            reply_to=types.MessageReplyHeader(reply_to_msg_id=4),
        ),
        SUPERGROUP,
    )
    assert result is not None
    assert result.chat_id == SUPERGROUP.id
    assert result.msg_id == 7
    assert result.text == "текст"
    assert result.sent_at == DATE
    assert result.edited_at == DATE
    assert result.reply_to_msg_id == 4
    assert result.sender_id == 55
    assert result.mentions_me is True
    assert result.is_outgoing is True
    assert result.link == "https://t.me/c/1234567890/7"
    assert result.attachments == ()


def test_service_message_is_skipped() -> None:
    service = types.MessageService(
        id=8, peer_id=PEER, date=DATE, action=types.MessageActionChatEditTitle(title="новое")
    )
    assert to_incoming(service, SUPERGROUP) is None


def test_message_with_only_an_attachment_has_empty_text() -> None:
    media = types.MessageMediaPhoto(photo=photo())
    result = to_incoming(message(message="", media=media), SUPERGROUP)
    assert result is not None
    assert result.text == ""
    assert [(a.kind, a.size_bytes) for a in result.attachments] == [("photo", 100)]


def test_document_keeps_its_name_and_size() -> None:
    attributes = [types.DocumentAttributeFilename(file_name="договор.pdf")]
    media = types.MessageMediaDocument(document=document(attributes))
    result = to_incoming(message(media=media), SUPERGROUP)
    assert result is not None
    attachment = result.attachments[0]
    assert (attachment.kind, attachment.file_name, attachment.size_bytes) == (
        "document",
        "договор.pdf",
        2048,
    )


def test_video_is_not_reported_as_a_document() -> None:
    attributes = [types.DocumentAttributeVideo(duration=1, w=1, h=1)]
    media = types.MessageMediaDocument(document=document(attributes))
    result = to_incoming(message(media=media), SUPERGROUP)
    assert result is not None
    assert result.attachments[0].kind == "video"


def test_link_preview_is_not_an_attachment() -> None:
    page = types.WebPage(
        id=1, url="https://example.com", display_url="example.com", hash=1, photo=photo()
    )
    result = to_incoming(message(media=types.MessageMediaWebPage(webpage=page)), SUPERGROUP)
    assert result is not None
    assert result.attachments == ()


@pytest.mark.parametrize(
    ("kind", "username", "expected"),
    [
        (ChatKind.SUPERGROUP, None, "https://t.me/c/1234567890/7"),
        (ChatKind.CHANNEL, None, "https://t.me/c/1234567890/7"),
        (ChatKind.SUPERGROUP, "public_chat", "https://t.me/public_chat/7"),
        (ChatKind.PRIVATE, "someone", None),
        (ChatKind.GROUP, None, None),
    ],
)
def test_message_links(kind: ChatKind, username: str | None, expected: str | None) -> None:
    chat = ChatInfo(SUPERGROUP.id, "Чат", kind, username)
    assert message_link(chat, 7) == expected


def test_real_people_are_kept_and_bots_are_not() -> None:
    user = types.User(id=1, first_name="Ольга", last_name="Смирнова", username="olga")
    assert person_info(user) == PersonInfo(1, "Ольга", "Смирнова", "olga")
    assert person_info(types.User(id=2, first_name="Jira Bot", bot=True)) is None
    assert person_info(None) is None


def test_chat_kinds() -> None:
    user = types.User(id=1)
    group = types.Chat(
        id=1, title="г", photo=types.ChatPhotoEmpty(), participants_count=2, date=DATE, version=1
    )
    common = {"id": 1, "title": "к", "photo": types.ChatPhotoEmpty(), "date": DATE}
    supergroup = types.Channel(megagroup=True, **common)
    channel = types.Channel(megagroup=False, **common)

    assert chat_kind(user) is ChatKind.PRIVATE
    assert chat_kind(group) is ChatKind.GROUP
    assert chat_kind(supergroup) is ChatKind.SUPERGROUP
    assert chat_kind(channel) is ChatKind.CHANNEL
    assert chat_kind(object()) is ChatKind.UNKNOWN


class _FailingClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def iter_messages(self, *_args, **_kwargs):
        raise self._error
        yield

    async def disconnect(self) -> None: ...


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (errors.FloodWaitError(request=None, capture=30), RateLimitedError),
        (
            errors.UnauthorizedError(request=None, message="AUTH_KEY_UNREGISTERED"),
            NotAuthorizedError,
        ),
    ],
)
def test_telethon_errors_are_translated(
    tmp_path: Path, error: Exception, expected: type[Exception]
) -> None:
    async def scenario() -> None:
        reader = TelegramReader(1, "hash", tmp_path / "session")
        reader._client = _FailingClient(error)
        reader._chats[1] = (SUPERGROUP, object())
        async for _ in reader.iter_after(1, 0):
            pass

    with pytest.raises(expected):
        asyncio.run(scenario())


def test_rate_limit_keeps_the_wait_time(tmp_path: Path) -> None:
    async def scenario() -> None:
        reader = TelegramReader(1, "hash", tmp_path / "session")
        reader._client = _FailingClient(errors.FloodWaitError(request=None, capture=30))
        reader._chats[1] = (SUPERGROUP, object())
        async for _ in reader.iter_after(1, 0):
            pass

    with pytest.raises(RateLimitedError) as caught:
        asyncio.run(scenario())
    assert caught.value.seconds == 30


# Read-only guarantees


def _violations(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
            found.append(f"forbidden attribute: {node.attr}")
        # Calling the client directly sends a raw request, which bypasses the list above.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_client"
        ):
            found.append("raw client request")
        if isinstance(node, ast.ImportFrom) and node.module == "telethon.tl.functions":
            found.append("raw request import")
    return found


def _source_files() -> list[Path]:
    """The review bot sends with its own token, so it's checked by what it may import instead."""
    return sorted(path for path in SRC.rglob("*.py") if "review_bot" not in path.parts)


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_review_bot_cannot_reach_the_account_session() -> None:
    """Its permission to send rests on this: nothing in it can touch the owner's account."""
    for path in (SRC / "review_bot").rglob("*.py"):
        imported = _imports(path)
        assert not {n for n in imported if n.split(".")[0] == "telethon"}, path
        assert not {n for n in imported if n.startswith("task_manager.collector")}, path


def test_only_the_bot_wrapper_imports_aiogram() -> None:
    wrapper = SRC / "review_bot" / "telegram_api.py"
    for path in SRC.rglob("*.py"):
        uses = any(name.split(".")[0] == "aiogram" for name in _imports(path))
        assert uses == (path == wrapper), path


@pytest.mark.parametrize(
    "source",
    [
        "client.send_message(1, 'x')",
        "message.mark_read()",
        "event.respond('x')",
        "self._client(request)",
        "from telethon.tl.functions import messages",
    ],
)
def test_guard_detects_write_operations(source: str) -> None:
    assert _violations(source)


def test_code_uses_no_write_operations() -> None:
    violations = {
        str(path.relative_to(SRC)): found
        for path in _source_files()
        if (found := _violations(path.read_text(encoding="utf-8")))
    }
    assert violations == {}


def test_only_the_wrapper_imports_telethon() -> None:
    importers = set()
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".")[0] == "telethon" for name in names):
                importers.add(path.relative_to(SRC).as_posix())
    assert importers == {"collector/telegram.py"}


def test_reader_exposes_only_read_methods() -> None:
    public = {name for name in vars(TelegramReader) if not name.startswith("_")}
    assert public == READER_API


def _with_entities(*entities) -> types.Message:
    return types.Message(
        id=1, peer_id=PEER, date=DATE, message="код: print(1) и слово", entities=list(entities)
    )


def test_code_formatting_is_kept_and_other_formatting_is_not() -> None:
    message = _with_entities(
        types.MessageEntityCode(offset=5, length=8),
        types.MessageEntityPre(offset=0, length=3, language="python"),
        types.MessageEntityBold(offset=14, length=1),
    )
    assert json.loads(to_incoming(message, SUPERGROUP).code_entities) == [
        {"offset": 5, "length": 8},
        {"offset": 0, "length": 3},
    ]


def test_message_without_code_stores_nothing() -> None:
    assert to_incoming(_with_entities(), SUPERGROUP).code_entities is None
