"""Which attachments go where. Shared by the bot, which shows them, and the export."""

# Never offered: the owner doesn't want audio in tasks, and it can't be anonymized.
HIDDEN_KINDS = frozenset({"voice", "audio", "video_note"})
# Off unless switched on: rarely part of a task, though sometimes they are.
OPT_IN_KINDS = frozenset({"sticker", "gif"})
_EXTENSIONS = {"photo": "jpg", "video": "mp4", "gif": "mp4", "sticker": "webp"}


def goes_to_jira(kind: str, excluded: bool, opted_in: bool) -> bool:
    if kind in HIDDEN_KINDS:
        return False
    if kind in OPT_IN_KINDS:
        return opted_in
    return not excluded


def upload_name(file_name: str | None, kind: str, attachment_id: int) -> str:
    """Telegram gives no name to photos. A readable one, with no chat data in it, is made up."""
    if file_name:
        return file_name
    extension = _EXTENSIONS.get(kind, "bin")
    return f"{kind}-{attachment_id}.{extension}"
