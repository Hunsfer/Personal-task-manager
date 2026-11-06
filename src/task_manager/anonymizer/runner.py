"""Anonymizer service: keeps an up-to-date anonymized record for every message."""

import asyncio
import contextlib
import logging
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from task_manager.anonymizer import store
from task_manager.anonymizer.engine import Anonymizer, KnownPerson, NerModel
from task_manager.common.config import AnonymizerConfig, MeConfig, ScopeConfig
from task_manager.monitoring import state_loop
from task_manager.storage.models import AnonymizedMessage, Message

BATCH_SIZE = 100
IDLE_INTERVAL_S = 10.0

logger = logging.getLogger(__name__)


def allowed_terms(
    scope: ScopeConfig, config: AnonymizerConfig, overrides: Iterable[str] = ()
) -> list[str]:
    terms = [*config.allow, *scope.role, *overrides]
    for section in scope.sections:
        terms.append(section.name)
        terms.extend(section.aliases)
    return terms


def current_rules(
    people: list[KnownPerson],
    me: MeConfig,
    scope: ScopeConfig,
    config: AnonymizerConfig,
    overrides: Iterable[str] = (),
) -> str:
    """The rules version that a record must carry for its text to be readable."""
    return store.rules_version(
        me.names,
        allowed_terms(scope, config, overrides),
        config.names,
        people,
        [(item.name, item.aliases) for item in me.organizations],
        me.same_people,
        me.secrets,
    )


async def live_rules(
    session: AsyncSession, me: MeConfig, scope: ScopeConfig, config: AnonymizerConfig
) -> str:
    """The current rules version, with everything the owner changes while services run."""
    return current_rules(
        await store.load_people(session), me, scope, config, await store.load_overrides(session)
    )


async def _anonymize_message(
    sessions: async_sessionmaker[AsyncSession],
    anonymizer: Anonymizer,
    rules: str,
    message_id: int,
) -> None:
    """Build and store one record, all or nothing."""
    async with sessions() as session:
        message = await session.scalar(
            select(Message)
            .where(Message.id == message_id)
            .options(selectinload(Message.attachments))
        )
        if message is None:
            return

        text = await store.render(session, message.text, anonymizer.find_spans(message.text))
        for attachment in message.attachments:
            name = attachment.file_name
            attachment.anonymized_file_name = (
                await store.render(session, name, anonymizer.find_spans(name)) if name else None
            )

        record = await session.get(AnonymizedMessage, message_id)
        if record is None:
            record = AnonymizedMessage(
                message_id=message_id, text="", source_hash="", rules_version=""
            )
            session.add(record)
        record.text = text
        record.sender = await store.sender_placeholder(
            session, message.sender_id, message.is_outgoing
        )
        record.source_hash = store.source_hash(
            message.text, [a.file_name for a in message.attachments]
        )
        record.rules_version = rules
        await session.commit()


async def process_batch(
    sessions: async_sessionmaker[AsyncSession],
    anonymizer: Anonymizer,
    rules: str,
    skip: set[int],
) -> int:
    """Process up to one batch. Failed messages get no record and are added to `skip`.

    A failed message can never reach the LLM, because it has no record to read.
    """
    async with sessions() as session:
        result = await session.scalars(
            select(Message.id)
            .outerjoin(AnonymizedMessage, AnonymizedMessage.message_id == Message.id)
            .where(
                or_(
                    AnonymizedMessage.message_id.is_(None),
                    AnonymizedMessage.rules_version != rules,
                )
            )
            .where(Message.id.not_in(skip))
            .order_by(Message.id)
            .limit(BATCH_SIZE)
        )
        message_ids = list(result)

    for message_id in message_ids:
        try:
            await _anonymize_message(sessions, anonymizer, rules, message_id)
        except Exception:
            logger.exception("message_anonymize_failed", extra={"msg_id": message_id})
            skip.add(message_id)
    return len(message_ids)


async def run_anonymizer(
    engine: AsyncEngine,
    me: MeConfig,
    scope: ScopeConfig,
    config: AnonymizerConfig,
    heartbeat: Path,
    stop: asyncio.Event,
) -> None:
    """Run until stopped. It stops between messages, so a database call is never cancelled."""
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    ner = NerModel()
    organizations = [(item.name, item.aliases) for item in me.organizations]
    logger.info("anonymizer_ready")

    anonymizer: Anonymizer | None = None
    rules = ""
    rebuilt = False
    finished = asyncio.Event()
    reporting = asyncio.create_task(state_loop(sessions, "anonymizer", finished))
    try:
        while not stop.is_set():
            async with sessions() as session:
                people = await store.load_people(session)
                overrides = await store.load_overrides(session)
            current = current_rules(people, me, scope, config, overrides)
            if current != rules:
                rules = current
                rebuilt = False
                anonymizer = Anonymizer(
                    ner,
                    me.names,
                    allowed_terms(scope, config, overrides),
                    config.names,
                    people,
                    organizations,
                    me.same_people,
                    me.secrets,
                )

            # Failures are retried on the next idle pass, not in a tight loop.
            skip: set[int] = set()
            total = 0
            while not stop.is_set():
                heartbeat.touch()
                count = await process_batch(sessions, anonymizer, rules, skip)
                total += count
                if count < BATCH_SIZE:
                    break
            if total:
                logger.info("messages_anonymized", extra={"count": total - len(skip)})
            # Only once everything is on the new rules: older records still use the old entries.
            if not rebuilt and not skip and not stop.is_set():
                async with sessions() as session:
                    pruned = await store.prune_pseudonyms(session)
                    await session.commit()
                logger.info("pseudonyms_pruned", extra={"count": pruned})
                rebuilt = True
            heartbeat.touch()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), IDLE_INTERVAL_S)
    finally:
        finished.set()
        await reporting
        heartbeat.unlink(missing_ok=True)
