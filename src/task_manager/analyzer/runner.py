"""Analyzer service: works through the message queue and keeps drafts up to date.

A message stays pending until its result is stored, so a crash or an outage in the middle loses
nothing. All model calls happen outside database transactions.
"""

import asyncio
import contextlib
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from task_manager.analyzer import drafts
from task_manager.analyzer.classifier import AnswerError, Classifier, Decision, DraftSummary
from task_manager.analyzer.ollama import OllamaUnavailableError
from task_manager.analyzer.prefilter import Prefilter
from task_manager.analyzer.signals import ME_TOKEN
from task_manager.analyzer.thread import Thread, ThreadLine, build_thread
from task_manager.anonymizer.runner import live_rules
from task_manager.anonymizer.store import source_hash
from task_manager.common.config import AnonymizerConfig, MeConfig, ScopeConfig
from task_manager.monitoring import state_loop
from task_manager.service import beat
from task_manager.storage.models import (
    Analysis,
    AnalysisOutcome,
    DraftKind,
    DraftStatus,
    Message,
    MessageStatus,
)

BATCH_SIZE = 50
IDLE_INTERVAL_S = 10.0
FIRST_RETRY_S = 5.0
FOLLOWUP_IGNORED = "followup_ignored"
FOLLOWUP_TICKET = "followup_ticket"
MAX_RETRY_S = 300.0

logger = logging.getLogger(__name__)


class Result(StrEnum):
    STORED = "stored"
    # Anonymization hasn't caught up, or the message is gone or no longer waiting.
    WAITING = "waiting"
    # Edited while the model was thinking: the answer is about text that no longer exists.
    CHANGED = "changed"


@dataclass(frozen=True)
class RulesSource:
    me: MeConfig
    scope: ScopeConfig
    config: AnonymizerConfig


@dataclass(frozen=True)
class _Snapshot:
    message_id: int
    chat_id: int
    sent_at: datetime
    hash: str
    own: bool
    thread: Thread
    drafts: list[drafts.OpenDraft]


class QueueWorker:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        classifier: Classifier,
        prefilter: Prefilter,
        rules_source: RulesSource,
        chat_ids: list[int],
    ) -> None:
        self.sessions = sessions
        self._sessions = sessions
        self._classifier = classifier
        self._prefilter = prefilter
        self._rules_source = rules_source
        self._chat_ids = chat_ids

    async def _rules(self) -> str:
        source = self._rules_source
        async with self._sessions() as session:
            return await live_rules(session, source.me, source.scope, source.config)

    async def _snapshot(self, message_id: int, rules: str) -> _Snapshot | None:
        async with self._sessions() as session:
            message = await session.scalar(
                select(Message)
                .where(Message.id == message_id, Message.status == MessageStatus.PENDING)
                .options(selectinload(Message.attachments))
            )
            if message is None:
                return None
            thread = await build_thread(session, message, rules)
            if thread is None:
                return None
            return _Snapshot(
                message_id=message.id,
                chat_id=message.chat_id,
                sent_at=message.sent_at,
                hash=source_hash(message.text, [a.file_name for a in message.attachments]),
                own=message.is_outgoing or thread.target.sender == ME_TOKEN,
                thread=thread,
                drafts=await drafts.open_drafts(session, message.chat_id, message.sent_at),
            )

    @staticmethod
    def _evidence(snapshot: _Snapshot, decision: Decision) -> set[int]:
        """Numbers the model was actually shown: an invented one must not attach foreign files."""
        shown = {line.msg_id for line in snapshot.thread.lines()}
        return shown & set(decision.evidence)

    async def _choose_draft(
        self, snapshot: _Snapshot, decision: Decision, rules: str
    ) -> drafts.OpenDraft | None:
        """Which draft the message joins. Links in the chat come first, as they can't be wrong."""
        current = drafts.holding_draft(snapshot.drafts, snapshot.message_id)
        if current is not None:
            return current
        reply_to = snapshot.thread.target.reply_to
        by_reply = drafts.linked_draft(snapshot.drafts, [reply_to] if reply_to is not None else [])
        if by_reply is not None or not decision.creates_draft:
            return by_reply
        by_evidence = drafts.linked_draft(snapshot.drafts, self._evidence(snapshot, decision))
        if by_evidence is not None or not snapshot.drafts:
            return by_evidence

        candidates = snapshot.drafts[: drafts.MAX_CANDIDATES]
        summaries = []
        async with self._sessions() as session:
            for draft in candidates:
                # A draft whose messages aren't ready is still compared, by its text alone.
                lines = await drafts.draft_lines(session, draft.id, rules) or []
                summaries.append(
                    DraftSummary(draft.id, draft.title, draft.description, tuple(lines))
                )
        chosen = await self._classifier.same_task(
            decision.title, decision.description, summaries, snapshot.thread.lines()
        )
        return next((draft for draft in candidates if draft.id == chosen), None)

    async def _build_update(
        self, snapshot: _Snapshot, decision: Decision, target: drafts.OpenDraft | None, rules: str
    ) -> drafts.DraftUpdate | None:
        """None means a message this draft depends on isn't anonymized yet, so everything waits."""
        if target is None:
            return drafts.DraftUpdate(None, decision.title, decision.description, False)
        async with self._sessions() as session:
            existing = await drafts.draft_lines(session, target.id, rules)
        if existing is None:
            return None
        if target.edited_by_owner:
            # The owner's wording stays. The draft only shows that newer messages joined it.
            return drafts.DraftUpdate(target.id, target.title, target.description, True)
        ordered = self._merged_lines(snapshot, decision, existing)
        text = await self._classifier.rewrite(ordered)
        if text is None:
            logger.warning("draft_rewrite_failed", extra={"count": len(ordered)})
            return drafts.DraftUpdate(target.id, target.title, target.description, True)
        return drafts.DraftUpdate(target.id, text.title, text.description, False)

    def _merged_lines(
        self, snapshot: _Snapshot, decision: Decision, existing: list[ThreadLine]
    ) -> list[ThreadLine]:
        joining = {snapshot.thread.target.msg_id}
        if decision.creates_draft:
            joining |= self._evidence(snapshot, decision)
        lines = {line.msg_id: line for line in existing}
        lines |= {line.msg_id: line for line in snapshot.thread.lines() if line.msg_id in joining}
        return [lines[msg_id] for msg_id in sorted(lines)]

    async def _follow_up(
        self, snapshot: _Snapshot, decision: Decision, target: drafts.OpenDraft, rules: str
    ) -> Result:
        """A decided draft never changes. The model decides whether this needs a new ticket."""
        async with self._sessions() as session:
            existing = await drafts.draft_lines(session, target.id, rules)
        if existing is None:
            return Result.WAITING
        answer = await self._classifier.follow_up(
            target.status.value,
            DraftSummary(target.id, target.title, target.description, tuple(existing)),
            target.rejection,
            snapshot.thread.lines(),
        )
        if not answer.needs_ticket:
            return await self._store(
                snapshot, AnalysisOutcome.NO_DRAFT, detail=FOLLOWUP_IGNORED, decision=decision
            )

        # After a failed answer there's no text, and the ticket must still say what it's about.
        title = answer.title or decision.title or f"Проверить: {target.title}"
        description = answer.description or decision.description or target.description
        if target.status in (DraftStatus.APPROVED, DraftStatus.EXPORTED):
            update = drafts.DraftUpdate(
                None, title, description, False, DraftKind.CHANGE_REQUEST, target.id
            )
        else:
            text = await self._classifier.rewrite(self._merged_lines(snapshot, decision, existing))
            if text is not None:
                title, description = text.title, text.description
            update = drafts.DraftUpdate(
                None, title, description, False, DraftKind.TASK, target.id, target.message_ids
            )
        return await self._store(
            snapshot,
            AnalysisOutcome.DRAFT,
            detail=FOLLOWUP_TICKET,
            decision=decision,
            draft_update=update,
        )

    async def _store(
        self,
        snapshot: _Snapshot,
        outcome: AnalysisOutcome,
        detail: str | None = None,
        decision: Decision | None = None,
        draft_update: drafts.DraftUpdate | None = None,
    ) -> Result:
        status = MessageStatus.FAILED if outcome is AnalysisOutcome.FAILED else MessageStatus.DONE
        async with self._sessions() as session:
            # A write first takes the lock, so an edit can't slip in between this check and commit.
            claimed = await session.execute(
                update(Message)
                .where(Message.id == snapshot.message_id, Message.status == MessageStatus.PENDING)
                .values(status=status)
            )
            if claimed.rowcount == 0:
                await session.rollback()
                return Result.WAITING
            message = await session.scalar(
                select(Message)
                .where(Message.id == snapshot.message_id)
                .options(selectinload(Message.attachments))
                .execution_options(populate_existing=True)
            )
            current = source_hash(message.text, [a.file_name for a in message.attachments])
            if current != snapshot.hash:
                await session.rollback()
                return Result.CHANGED

            await session.merge(
                Analysis(
                    message_id=snapshot.message_id,
                    source_hash=snapshot.hash,
                    outcome=outcome,
                    detail=detail,
                    model=self._classifier.model if decision else None,
                    prompt_id=decision.prompt_id if decision else None,
                    is_task=decision.is_task if decision else None,
                    assigned_to_me=decision.assigned_to_me if decision else None,
                    assignment_reason=decision.reason if decision else None,
                    confidence=decision.confidence if decision else None,
                    analyzed_at=datetime.now(UTC),
                )
            )
            if draft_update is not None:
                await drafts.apply_update(
                    session,
                    draft_update,
                    decision,
                    message,
                    self._evidence(snapshot, decision),
                    self._classifier.model,
                )
            await session.commit()
        return Result.STORED

    async def process(self, message_id: int, rules: str) -> Result:
        snapshot = await self._snapshot(message_id, rules)
        if snapshot is None:
            return Result.WAITING

        noise = self._prefilter.noise_reason(snapshot.thread.target)
        if noise is not None:
            return await self._store(snapshot, AnalysisOutcome.SKIPPED, detail=noise)

        try:
            decision = await self._classifier.classify(snapshot.thread, snapshot.own)
        except AnswerError as error:
            logger.warning("analysis_failed", extra={"msg_id": message_id, "reason": str(error)})
            return await self._store(snapshot, AnalysisOutcome.FAILED, detail=str(error))

        target = await self._choose_draft(snapshot, decision, rules)
        if target is not None:
            target = drafts.undecided_child(snapshot.drafts, target)
            if target.decided:
                return await self._follow_up(snapshot, decision, target, rules)
        draft_update = None
        if decision.creates_draft or target is not None:
            draft_update = await self._build_update(snapshot, decision, target, rules)
            if draft_update is None:
                return Result.WAITING
        outcome = AnalysisOutcome.DRAFT if decision.creates_draft else AnalysisOutcome.NO_DRAFT
        return await self._store(snapshot, outcome, decision=decision, draft_update=draft_update)

    async def _pending(self, skip: set[int]) -> list[int]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(Message.id)
                .where(
                    Message.status == MessageStatus.PENDING,
                    Message.chat_id.in_(self._chat_ids),
                    Message.id.not_in(skip),
                )
                # History may have been loaded newest first, and context depends on the real order.
                .order_by(Message.sent_at, Message.id)
                .limit(BATCH_SIZE)
            )
            return list(rows)

    async def drain(self, stop: asyncio.Event) -> Counter[str]:
        """Handle everything that is ready now. OllamaUnavailableError means try again later."""
        counters: Counter[str] = Counter()
        skip: set[int] = set()
        while not stop.is_set():
            rules = await self._rules()
            batch = await self._pending(skip)
            if not batch:
                break
            for message_id in batch:
                if stop.is_set():
                    break
                try:
                    result = await self.process(message_id, rules)
                except OllamaUnavailableError:
                    raise
                except Exception:
                    # One bad message must not block the queue. It's retried on the next pass.
                    logger.exception("message_analysis_error", extra={"msg_id": message_id})
                    skip.add(message_id)
                    counters["errors"] += 1
                    continue
                counters[result.value] += 1
                if result is Result.WAITING:
                    skip.add(message_id)
        return counters


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), seconds)


async def run_analyzer(worker: QueueWorker, heartbeat: Path, stop: asyncio.Event) -> None:
    """Run until stopped. A model call in progress at shutdown is simply repeated next time."""
    finished = asyncio.Event()
    # Separate from the queue, because one model call can far outlast the heartbeat's limit.
    beating = asyncio.create_task(beat(heartbeat, finished))
    reporting = asyncio.create_task(state_loop(worker.sessions, "analyzer", finished))
    logger.info("analyzer_ready")
    retry = FIRST_RETRY_S
    unavailable = False
    try:
        while not stop.is_set():
            try:
                counters = await worker.drain(stop)
            except OllamaUnavailableError:
                # Messages wait in the queue, and collection continues without the model.
                if not unavailable:
                    logger.warning("ollama_unavailable")
                    unavailable = True
                await _sleep_or_stop(stop, retry)
                retry = min(retry * 2, MAX_RETRY_S)
                continue
            if unavailable:
                logger.info("ollama_restored")
                unavailable = False
            retry = FIRST_RETRY_S
            if counters[Result.STORED.value]:
                logger.info("messages_analyzed", extra={"count": counters[Result.STORED.value]})
            await _sleep_or_stop(stop, IDLE_INTERVAL_S)
    finally:
        finished.set()
        await beating
        await reporting
        heartbeat.unlink(missing_ok=True)


def make_sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
