"""Database tables. Timestamps are stored in UTC."""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class MessageStatus(StrEnum):
    """Processing state. The `messages` table doubles as the work queue."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    # Old history kept for the evaluation dataset. It never enters the analysis queue.
    ARCHIVED = "archived"


class AnalysisOutcome(StrEnum):
    # The prefilter's call: no model was asked.
    SKIPPED = "skipped"
    NO_DRAFT = "no_draft"
    DRAFT = "draft"
    FAILED = "failed"


class DraftStatus(StrEnum):
    # Not in the bot yet: the queue that survives the bot being down.
    NEW = "new"
    SENT = "sent"
    APPROVED = "approved"
    REJECTED = "rejected"
    # In Jira, or added to a task there.
    EXPORTED = "exported"


class ExportAction(StrEnum):
    """The owner's decision. None on an approved draft means it predates export and stays put."""

    CREATE = "create"
    APPEND = "append"


class FileRequestStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    DONE = "done"


class DraftKind(StrEnum):
    TASK = "task"
    # New information about a task the owner has already accepted.
    CHANGE_REQUEST = "change_request"


class DraftRole(StrEnum):
    TRIGGER = "trigger"
    EVIDENCE = "evidence"
    CONTEXT = "context"


class ChatKind(StrEnum):
    PRIVATE = "private"
    GROUP = "group"
    SUPERGROUP = "supergroup"
    CHANNEL = "channel"
    UNKNOWN = "unknown"


def _enum_column(enum_type: type[StrEnum]) -> Enum:
    return Enum(
        enum_type,
        native_enum=False,
        length=16,
        values_callable=lambda members: [member.value for member in members],
    )


class Chat(Base):
    __tablename__ = "chats"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str | None]
    kind: Mapped[ChatKind] = mapped_column(
        _enum_column(ChatKind), default=ChatKind.UNKNOWN, server_default=ChatKind.UNKNOWN.value
    )
    # Backfill resumes from this cursor, so downtime loses no messages.
    last_msg_id: Mapped[int | None] = mapped_column(BigInteger)
    # Empty until the chat's history is fully loaded. An interrupted load starts over.
    history_loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list["Message"]] = relationship(back_populates="chat")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("chat_id", "msg_id", name="uq_messages_chat_id_msg_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.chat_id", ondelete="CASCADE"))
    msg_id: Mapped[int] = mapped_column(BigInteger)
    reply_to_msg_id: Mapped[int | None] = mapped_column(BigInteger)
    sender_id: Mapped[int | None] = mapped_column(BigInteger)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    text: Mapped[str] = mapped_column(Text, default="")
    # Telegram's own flag: a mention by name without @ can't be recovered from the text later.
    mentions_me: Mapped[bool] = mapped_column(default=False, server_default=false())
    is_outgoing: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Only supergroups and channels have message links.
    link: Mapped[str | None]
    # Telegram's code formatting as JSON, since the text itself carries no marks.
    code_entities: Mapped[str | None] = mapped_column(Text)
    status: Mapped[MessageStatus] = mapped_column(
        _enum_column(MessageStatus),
        default=MessageStatus.PENDING,
        index=True,
    )

    chat: Mapped[Chat] = relationship(back_populates="messages")
    attachments: Mapped[list["Attachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", passive_deletes=True
    )


class Attachment(Base):
    """File metadata only. The files themselves are never stored."""

    __tablename__ = "attachments"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str]
    file_name: Mapped[str | None]
    anonymized_file_name: Mapped[str | None]
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    # The owner's choice in the review bot: this file must not go to Jira.
    excluded: Mapped[bool] = mapped_column(default=False, server_default=false())
    # For kinds that are off by default, such as stickers.
    opted_in: Mapped[bool] = mapped_column(default=False, server_default=false())

    message: Mapped[Message] = relationship(back_populates="attachments")


class Person(Base):
    """People known from Telegram, used to find their names in message text."""

    __tablename__ = "people"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    first_name: Mapped[str | None]
    last_name: Mapped[str | None]
    username: Mapped[str | None]
    is_me: Mapped[bool] = mapped_column(default=False, server_default=false())


class Pseudonym(Base):
    """Reversible replacements. The same value always gets the same placeholder, in any chat.

    This table reverses anonymization, so it is as sensitive as the original text.
    """

    __tablename__ = "pseudonyms"
    __table_args__ = (
        UniqueConstraint("kind", "key", name="uq_pseudonyms_kind_key"),
        UniqueConstraint("kind", "number", name="uq_pseudonyms_kind_number"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    key: Mapped[str]
    number: Mapped[int]
    value: Mapped[str]


class AnonymizedMessage(Base):
    """The only form of a message's text that may reach the LLM."""

    __tablename__ = "anonymized_messages"

    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True, autoincrement=False
    )
    text: Mapped[str] = mapped_column(Text)
    sender: Mapped[str | None]
    # Detects a record that no longer matches the source message or the current rules.
    source_hash: Mapped[str] = mapped_column(String(64))
    rules_version: Mapped[str] = mapped_column(String(64), index=True)


class Analysis(Base):
    """What was decided about a message and why. No text, so decisions can be audited later."""

    __tablename__ = "analyses"

    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True, autoincrement=False
    )
    source_hash: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[AnalysisOutcome] = mapped_column(_enum_column(AnalysisOutcome), index=True)
    # Prefilter reason or failure code, always from a fixed set of values.
    detail: Mapped[str | None]
    model: Mapped[str | None]
    prompt_id: Mapped[str | None]
    is_task: Mapped[bool | None]
    assigned_to_me: Mapped[bool | None]
    assignment_reason: Mapped[str | None]
    confidence: Mapped[float | None]
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Draft(Base):
    """A task waiting for the owner's decision. Its text is anonymized, like the model's input."""

    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(
        ForeignKey("chats.chat_id", ondelete="CASCADE"), index=True
    )
    status: Mapped[DraftStatus] = mapped_column(
        _enum_column(DraftStatus), default=DraftStatus.NEW, index=True
    )
    kind: Mapped[DraftKind] = mapped_column(
        _enum_column(DraftKind), default=DraftKind.TASK, server_default=DraftKind.TASK.value
    )
    # The decided draft this one follows up on: a change request or a rejected task brought back.
    parent_draft_id: Mapped[int | None] = mapped_column(
        ForeignKey("drafts.id", ondelete="SET NULL")
    )
    bot_message_id: Mapped[int | None] = mapped_column(BigInteger)
    # The message in the bot no longer shows the current state of this draft.
    needs_refresh: Mapped[bool] = mapped_column(default=False, server_default=false(), index=True)
    # The owner's wording is never overwritten by rebuilding the text from messages.
    edited_by_owner: Mapped[bool] = mapped_column(default=False, server_default=false())
    deadline_by_owner: Mapped[bool] = mapped_column(default=False, server_default=false())
    export_action: Mapped[ExportAction | None] = mapped_column(_enum_column(ExportAction))
    # Set before asking Jira to create, so a retry knows the task may already exist.
    export_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # A status code and field names only. Stops retries until the owner asks for one.
    export_error: Mapped[str | None]
    jira_key: Mapped[str | None]
    jira_link: Mapped[str | None]
    jira_status: Mapped[str | None]
    jira_locked: Mapped[bool] = mapped_column(default=False, server_default=false())
    files_done: Mapped[bool] = mapped_column(default=False, server_default=false())
    title: Mapped[str]
    description: Mapped[str] = mapped_column(Text)
    # The model's guess. Its date arithmetic is unreliable, so nothing uses this automatically.
    deadline: Mapped[str | None]
    priority: Mapped[str]
    has_placeholders: Mapped[bool]
    # The text couldn't be rebuilt after the last message joined, so it may not cover that message.
    text_is_stale: Mapped[bool] = mapped_column(default=False, server_default=false())
    model: Mapped[str]
    prompt_id: Mapped[str]
    # Time in the conversation rather than on the clock, so replaying history groups the same way.
    last_message_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DraftMessage(Base):
    """Messages behind a draft. Merging only adds rows, so a wrong merge can be seen and undone."""

    __tablename__ = "draft_messages"

    draft_id: Mapped[int] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"), primary_key=True
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    role: Mapped[DraftRole] = mapped_column(_enum_column(DraftRole))


class DraftDecision(Base):
    """What the owner did with a draft: material for tuning later. Free text is anonymized."""

    __tablename__ = "draft_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("drafts.id", ondelete="CASCADE"), index=True)
    decision: Mapped[str] = mapped_column(String(16))
    # A rejection reason from a fixed set, or the names of the fields that were edited.
    detail: Mapped[str | None]
    note: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ServiceState(Base):
    """When each service last reported, so the bot can notice one has gone quiet."""

    __tablename__ = "service_states"

    service: Mapped[str] = mapped_column(String(32), primary_key=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # A fixed code such as telegram_disconnected. Never message text.
    detail: Mapped[str | None]


class BotChatMessage(Base):
    """A message in the bot's chat. Telegram can't list them, and /clear needs to find them."""

    __tablename__ = "bot_chat_messages"

    message_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class FileRequest(Base):
    """Asks the collector to download a file. Holds a path only while the file exists."""

    __tablename__ = "file_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    attachment_id: Mapped[int] = mapped_column(
        ForeignKey("attachments.id", ondelete="CASCADE"), index=True
    )
    draft_id: Mapped[int] = mapped_column(ForeignKey("drafts.id", ondelete="CASCADE"), index=True)
    purpose: Mapped[str] = mapped_column(String(16))
    status: Mapped[FileRequestStatus] = mapped_column(
        _enum_column(FileRequestStatus), default=FileRequestStatus.PENDING, index=True
    )
    path: Mapped[str | None]
    error: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AttachmentUpload(Base):
    """A file already handled for a task, so a retry never uploads it twice."""

    __tablename__ = "attachment_uploads"

    draft_id: Mapped[int] = mapped_column(
        ForeignKey("drafts.id", ondelete="CASCADE"), primary_key=True
    )
    attachment_id: Mapped[int] = mapped_column(
        ForeignKey("attachments.id", ondelete="CASCADE"), primary_key=True
    )
    error: Mapped[str | None]
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BotNotice(Base):
    """A message for the owner from a service that has no bot token."""

    __tablename__ = "bot_notices"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    keyboard: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AnonymizerOverride(Base):
    """A word of the owner's that must stay readable. It works like an allowed term in the config.

    Lives here rather than in config/anonymizer.yaml because config is mounted read-only and that
    file holds only the general list, while these words come from real messages.
    """

    __tablename__ = "anonymizer_overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    term: Mapped[str] = mapped_column(unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
