"""Pseudonym map and stored results. The way into and out of anonymized text."""

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from task_manager.anonymizer.engine import FIXED_KINDS, KnownPerson, Span
from task_manager.storage.models import (
    AnonymizedMessage,
    AnonymizerOverride,
    Attachment,
    Draft,
    DraftDecision,
    Message,
    Person,
    Pseudonym,
)

# Raise this when the detection logic changes, so stored results are rebuilt.
ENGINE_VERSION = 5
PLACEHOLDER = re.compile(r"<(PERSON|ORG|LOC|PHONE|EMAIL|URL|IP|ID)_(\d+)>")


def source_hash(text: str, file_names: Iterable[str | None]) -> str:
    payload = json.dumps([text, sorted(name or "" for name in file_names)], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def rules_version(
    me_names: Iterable[str],
    allowed_terms: Iterable[str],
    extra_names: Iterable[str],
    people: Iterable[KnownPerson],
    organizations: Iterable[tuple[str, Iterable[str]]] = (),
    same_people: Iterable[Iterable[str]] = (),
    secrets: Iterable[str] = (),
) -> str:
    """Changes whenever the rules or the known people change, so old results get rebuilt.

    A person learned today may be named in messages collected earlier.
    """
    payload = json.dumps(
        [
            ENGINE_VERSION,
            sorted(me_names),
            sorted(allowed_terms),
            sorted(extra_names),
            [
                [p.user_id, p.first_name, p.last_name, p.username, p.is_me]
                for p in sorted(people, key=lambda person: person.user_id)
            ],
            [[name, sorted(aliases)] for name, aliases in organizations],
            [sorted(group) for group in same_people],
            sorted(secrets),
        ],
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


async def load_people(session: AsyncSession) -> list[KnownPerson]:
    rows = await session.scalars(select(Person).order_by(Person.user_id))
    return [
        KnownPerson(row.user_id, row.first_name, row.last_name, row.username, row.is_me)
        for row in rows
    ]


async def load_overrides(session: AsyncSession) -> list[str]:
    """Words the owner marked as wrongly masked."""
    return list(
        await session.scalars(select(AnonymizerOverride.term).order_by(AnonymizerOverride.id))
    )


async def _placeholder(session: AsyncSession, kind: str, key: str, value: str) -> str:
    if kind in FIXED_KINDS:
        return f"<{kind}>"
    number = await session.scalar(
        select(Pseudonym.number).where(Pseudonym.kind == kind, Pseudonym.key == key)
    )
    if number is None:
        highest = await session.scalar(
            select(func.max(Pseudonym.number)).where(Pseudonym.kind == kind)
        )
        number = (highest or 0) + 1
        session.add(Pseudonym(kind=kind, key=key, number=number, value=value))
        await session.flush()
    return f"<{kind}_{number}>"


async def render(session: AsyncSession, text: str, spans: list[Span]) -> str:
    parts = []
    position = 0
    for span in spans:
        parts.append(text[position : span.start])
        parts.append(await _placeholder(session, span.kind, span.key, span.value))
        position = span.end
    parts.append(text[position:])
    return "".join(parts)


async def sender_placeholder(
    session: AsyncSession, sender_id: int | None, is_outgoing: bool
) -> str | None:
    if is_outgoing:
        return "<ME>"
    if sender_id is None:
        return None
    person = await session.get(Person, sender_id)
    if person is not None and person.is_me:
        return "<ME>"
    name = (
        KnownPerson(person.user_id, person.first_name, person.last_name, person.username)
        if person is not None
        else KnownPerson(sender_id)
    )
    return await _placeholder(session, "PERSON", f"user:{sender_id}", name.display_name)


@dataclass(frozen=True)
class ReadyMessage:
    message_id: int
    text: str
    sender: str | None
    file_names: tuple[str, ...]


async def get_ready(
    session: AsyncSession, message_id: int, current_rules: str
) -> ReadyMessage | None:
    """The only way to get a message's text for the LLM. Fails closed.

    Returns None unless a record exists for exactly this version of the message and these rules.
    """
    message = await session.scalar(
        select(Message).where(Message.id == message_id).options(selectinload(Message.attachments))
    )
    record = await session.get(AnonymizedMessage, message_id)
    if message is None or record is None or record.rules_version != current_rules:
        return None
    if record.source_hash != source_hash(message.text, [a.file_name for a in message.attachments]):
        return None
    return ReadyMessage(
        message_id=message_id,
        text=record.text,
        sender=record.sender,
        file_names=tuple(
            attachment.anonymized_file_name
            for attachment in message.attachments
            if attachment.anonymized_file_name
        ),
    )


async def prune_pseudonyms(session: AsyncSession) -> int:
    """Delete map entries nothing refers to any more, and return how many.

    A value that newer rules treat differently must not linger: earlier rules may have stored
    something as a name that is now known to be a password. Drafts count as references: the model
    or the owner may have left a placeholder there, and it must still resolve for review.
    """
    used: set[tuple[str, int]] = set()
    sources = (
        select(AnonymizedMessage.text),
        select(AnonymizedMessage.sender).where(AnonymizedMessage.sender.is_not(None)),
        select(Attachment.anonymized_file_name).where(Attachment.anonymized_file_name.is_not(None)),
        select(Draft.title),
        select(Draft.description),
        select(DraftDecision.note).where(DraftDecision.note.is_not(None)),
    )
    for source in sources:
        for text in await session.scalars(source):
            used.update((kind, int(number)) for kind, number in PLACEHOLDER.findall(text))

    rows = (await session.execute(select(Pseudonym.id, Pseudonym.kind, Pseudonym.number))).all()
    stale = [row.id for row in rows if (row.kind, row.number) not in used]
    if stale:
        await session.execute(delete(Pseudonym).where(Pseudonym.id.in_(stale)))
    return len(stale)


async def restore(session: AsyncSession, text: str) -> tuple[str, list[str]]:
    """Put real values back, for the user's own review only.

    Returns the text and any placeholders with no map entry, for example ones the LLM invented.
    These stay in the text as they are.
    """
    unknown: list[str] = []
    parts = []
    position = 0
    for match in PLACEHOLDER.finditer(text):
        value = await session.scalar(
            select(Pseudonym.value).where(
                Pseudonym.kind == match.group(1), Pseudonym.number == int(match.group(2))
            )
        )
        parts.append(text[position : match.start()])
        if value is None:
            unknown.append(match.group())
            parts.append(match.group())
        else:
            parts.append(value)
        position = match.end()
    parts.append(text[position:])
    return "".join(parts), unknown
