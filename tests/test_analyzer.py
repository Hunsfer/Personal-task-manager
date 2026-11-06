"""The analyzer service. Made-up data and a fake model only."""

import asyncio
import io
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.analyzer import commands, runner
from task_manager.analyzer.classifier import (
    ATTEMPTS,
    AnswerError,
    Classifier,
    Decision,
    DraftSummary,
)
from task_manager.analyzer.ollama import SEED, OllamaClient, OllamaUnavailableError
from task_manager.analyzer.prefilter import NO_TEXT, NO_WORDS, PHRASE, Prefilter
from task_manager.analyzer.prompt import load_template, render_system
from task_manager.analyzer.runner import QueueWorker, Result, RulesSource, make_sessions
from task_manager.analyzer.thread import ThreadLine
from task_manager.analyzer.verdict import (
    DraftText,
    InvalidVerdictError,
    TaskVerdict,
    creates_draft,
    parse_answer,
    verdict_schema,
)
from task_manager.anonymizer.runner import current_rules
from task_manager.anonymizer.store import source_hash
from task_manager.common.config import AnonymizerConfig, MeConfig, ScopeConfig, Section
from task_manager.common.log import SafeJsonFormatter
from task_manager.evaluation import labeling, own, sampling
from task_manager.storage.db import make_engine
from task_manager.storage.models import (
    Analysis,
    AnalysisOutcome,
    AnonymizedMessage,
    Attachment,
    Chat,
    ChatKind,
    Draft,
    DraftMessage,
    DraftRole,
    Message,
    MessageStatus,
)

ROOT = Path(__file__).parent.parent
ALEMBIC_INI = ROOT / "alembic.ini"
PROMPTS = ROOT / "config" / "prompts"
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
CHAT = 1
OTHER_CHAT = 2
UNLISTED_CHAT = 3

ME = MeConfig(names=["Тестовый"])
SCOPE = ScopeConfig(
    role=["фронтенд", "front-end"], sections=[Section(name="Статьи", aliases=["блог", "/blog"])]
)
ANON = AnonymizerConfig(allow=[], names=[])
RULES = current_rules([], ME, SCOPE, ANON)
SOURCE = RulesSource(ME, SCOPE, ANON)


def _decision(creates: bool = True, **overrides) -> Decision:
    values = {
        "creates_draft": creates,
        "is_task": creates,
        "assigned_to_me": creates,
        "reason": "mention" if creates else "none",
        "confidence": 0.9,
        "title": "Заголовок" if creates else "",
        "description": "Описание" if creates else "",
        "deadline": None,
        "priority": "medium",
        "evidence": (),
        "prompt_id": "task-1",
    }
    return Decision(**(values | overrides))


class FakeClassifier:
    """Answers by message number. Everything it was asked is kept for the assertions."""

    model = "fake"

    def __init__(self, answers=None, same=None, rewritten=None) -> None:
        self.answers = answers or {}
        self.same = same
        self.rewritten = (
            rewritten if rewritten is not None else DraftText(title="Т", description="О")
        )
        self.classified: list[tuple[int, bool]] = []
        self.same_calls: list[list[DraftSummary]] = []
        self.same_contexts: list[list[ThreadLine]] = []
        self.rewrite_calls: list[list[ThreadLine]] = []
        self.on_classify = None

    async def classify(self, thread, own):
        self.classified.append((thread.target.msg_id, own))
        if self.on_classify is not None:
            await self.on_classify(thread)
        answer = self.answers.get(thread.target.msg_id, _decision(False))
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def same_task(self, title, description, drafts, context=()):
        self.same_calls.append(list(drafts))
        self.same_contexts.append(list(context))
        return self.same

    async def rewrite(self, lines):
        self.rewrite_calls.append(list(lines))
        return self.rewritten


@pytest.fixture
def engine(tmp_path, monkeypatch) -> AsyncEngine:
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return make_engine(path)


def _worker(engine: AsyncEngine, classifier, chat_ids=(CHAT, OTHER_CHAT)) -> QueueWorker:
    return QueueWorker(
        make_sessions(engine), classifier, Prefilter(["ок", "спасибо"]), SOURCE, list(chat_ids)
    )


async def _chats(session: AsyncSession) -> None:
    for chat_id in (CHAT, OTHER_CHAT, UNLISTED_CHAT):
        session.add(Chat(chat_id=chat_id, kind=ChatKind.GROUP))
    await session.flush()


async def _add(session: AsyncSession, msg_id: int, text: str = "поправь шапку", **extra) -> Message:
    anonymized = extra.pop("anonymized", text)
    rules = extra.pop("rules", RULES)
    outgoing = extra.pop("is_outgoing", False)
    sender = extra.pop("sender", "<ME>" if outgoing else "<PERSON_1>")
    files = extra.pop("files", [])
    message = Message(
        chat_id=extra.pop("chat_id", CHAT),
        msg_id=msg_id,
        sent_at=extra.pop("sent_at", NOW + timedelta(minutes=msg_id)),
        text=text,
        is_outgoing=outgoing,
        status=extra.pop("status", MessageStatus.PENDING),
        **extra,
    )
    session.add(message)
    await session.flush()
    for name in files:
        session.add(
            Attachment(
                message_id=message.id, kind="document", file_name=name, anonymized_file_name=name
            )
        )
    if anonymized is not None:
        session.add(
            AnonymizedMessage(
                message_id=message.id,
                text=anonymized,
                sender=sender,
                source_hash=source_hash(text, files),
                rules_version=rules,
            )
        )
    await session.flush()
    return message


def _run(engine: AsyncEngine, scenario):
    async def wrapper():
        try:
            return await scenario()
        finally:
            await engine.dispose()

    return asyncio.run(wrapper())


async def _seed(engine: AsyncEngine, *messages: dict) -> None:
    async with AsyncSession(engine) as session:
        if await session.get(Chat, CHAT) is None:
            await _chats(session)
        for fields in messages:
            await _add(session, **fields)
        await session.commit()


async def _drain(engine: AsyncEngine, classifier, **kwargs):
    return await _worker(engine, classifier, **kwargs).drain(asyncio.Event())


async def _rows(engine: AsyncEngine, table) -> list:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        return list(await session.scalars(select(table)))


async def _status(engine: AsyncEngine, msg_id: int, chat_id: int = CHAT) -> MessageStatus:
    async with AsyncSession(engine) as session:
        return await session.scalar(
            select(Message.status).where(Message.msg_id == msg_id, Message.chat_id == chat_id)
        )


async def _links(engine: AsyncEngine) -> dict[int, DraftRole]:
    async with AsyncSession(engine) as session:
        rows = await session.execute(
            select(Message.msg_id, DraftMessage.role).join(
                DraftMessage, DraftMessage.message_id == Message.id
            )
        )
        return {msg_id: role for msg_id, role in rows.all()}


# Prefilter


@pytest.mark.parametrize(
    ("text", "files", "expected"),
    [
        ("", [], NO_TEXT),
        ("   ", ["макет.png"], NO_TEXT),
        ("👍", [], NO_WORDS),
        ("+++", [], NO_WORDS),
        ("Ок!", [], PHRASE),
        ("спасибо 🙏", [], PHRASE),
        ("ок", ["макет.png"], None),
        ("ок, сделай", [], None),
        ("да", [], None),
        ("<URL_1>", [], None),
        ("в шапке съехала кнопка", [], None),
    ],
)
def test_prefilter_drops_only_certain_noise(text, files, expected):
    line = ThreadLine(
        msg_id=1, sender="<PERSON_1>", text=text, sent_at="2026-11-04T12:00", files=files
    )
    assert Prefilter(["ок", "Спасибо"]).noise_reason(line) == expected


def test_noise_list_in_the_repository_keeps_confirmations_out():
    import yaml

    phrases = yaml.safe_load((ROOT / "config" / "analyzer.yaml").read_text("utf-8"))[
        "noise_phrases"
    ]
    assert "ок" in phrases
    assert "да" not in phrases
    assert "нет" not in phrases


def test_noise_never_reaches_the_model(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "text": "ок"}, {"msg_id": 2, "text": ""})
        fake = FakeClassifier()
        await _drain(engine, fake)
        return fake.classified, await _rows(engine, Analysis), await _status(engine, 1)

    classified, analyses, status = _run(engine, scenario)
    assert classified == []
    assert {row.outcome for row in analyses} == {AnalysisOutcome.SKIPPED}
    assert {row.detail for row in analyses} == {PHRASE, NO_TEXT}
    assert status is MessageStatus.DONE


# Queue


def test_task_creates_a_draft_and_records_the_decision(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision(title="Шапка", description="Поправить")}))
        return (
            await _rows(engine, Draft),
            await _rows(engine, Analysis),
            await _links(engine),
            await _status(engine, 1),
        )

    drafts, analyses, links, status = _run(engine, scenario)
    assert [(d.title, d.description, d.chat_id) for d in drafts] == [("Шапка", "Поправить", CHAT)]
    assert analyses[0].outcome is AnalysisOutcome.DRAFT
    assert analyses[0].assignment_reason == "mention"
    assert analyses[0].model == "fake"
    assert links == {1: DraftRole.TRIGGER}
    assert status is MessageStatus.DONE


def test_not_a_task_creates_nothing(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier())
        return await _rows(engine, Draft), await _rows(engine, Analysis)

    drafts, analyses = _run(engine, scenario)
    assert drafts == []
    assert analyses[0].outcome is AnalysisOutcome.NO_DRAFT


@pytest.mark.parametrize("reason", ["collective", "none"])
def test_weak_reasons_never_create_a_draft(reason):
    verdict = TaskVerdict(
        is_task=True,
        assigned_to_me=True,
        assignment_reason=reason,
        confidence=1.0,
        title="т",
        description="о",
        deadline=None,
        priority="medium",
        evidence=[],
    )
    assert not creates_draft(verdict)


def test_low_confidence_still_creates_a_draft(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision(confidence=0.1)}))
        return await _rows(engine, Draft)

    assert len(_run(engine, scenario)) == 1


def test_unlisted_chat_is_not_analyzed(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "chat_id": UNLISTED_CHAT})
        fake = FakeClassifier({1: _decision()})
        await _drain(engine, fake)
        return fake.classified, await _status(engine, 1, UNLISTED_CHAT)

    classified, status = _run(engine, scenario)
    assert classified == []
    assert status is MessageStatus.PENDING


def test_archived_history_never_enters_the_queue(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "status": MessageStatus.ARCHIVED})
        fake = FakeClassifier()
        await _drain(engine, fake)
        return fake.classified

    assert _run(engine, scenario) == []


def test_messages_are_handled_in_the_order_they_were_sent(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 5, "sent_at": NOW + timedelta(minutes=5)},
            {"msg_id": 2, "sent_at": NOW + timedelta(minutes=2)},
            {"msg_id": 9, "sent_at": NOW + timedelta(minutes=9)},
        )
        fake = FakeClassifier()
        await _drain(engine, fake)
        return [msg_id for msg_id, _ in fake.classified]

    assert _run(engine, scenario) == [2, 5, 9]


def test_message_waits_until_it_is_anonymized(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "anonymized": None})
        fake = FakeClassifier({1: _decision()})
        counters = await _drain(engine, fake)
        return fake.classified, counters, await _status(engine, 1)

    classified, counters, status = _run(engine, scenario)
    assert classified == []
    assert counters[Result.WAITING.value] == 1
    assert status is MessageStatus.PENDING


def test_message_waits_while_its_context_is_not_anonymized(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.DONE, "anonymized": None, "text": "сырой текст"},
            {"msg_id": 2},
        )
        fake = FakeClassifier({2: _decision()})
        await _drain(engine, fake)
        return fake.classified, await _status(engine, 2)

    classified, status = _run(engine, scenario)
    assert classified == []
    assert status is MessageStatus.PENDING


def test_outdated_rules_keep_a_message_waiting(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "rules": "старые правила"})
        fake = FakeClassifier({1: _decision()})
        await _drain(engine, fake)
        return fake.classified

    assert _run(engine, scenario) == []


def test_outage_leaves_the_message_in_the_queue(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        with pytest.raises(OllamaUnavailableError):
            await _drain(engine, FakeClassifier({1: OllamaUnavailableError("ConnectError")}))
        return await _status(engine, 1), await _rows(engine, Analysis)

    status, analyses = _run(engine, scenario)
    assert status is MessageStatus.PENDING
    assert analyses == []


def test_bad_answer_fails_the_message_and_retry_returns_it(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: AnswerError("schema")}))
        failed = await _status(engine, 1)
        analysis = (await _rows(engine, Analysis))[0]
        returned = await commands.retry_failed(engine)
        await _drain(engine, FakeClassifier({1: _decision()}))
        return failed, analysis, returned, await _status(engine, 1), await _rows(engine, Draft)

    failed, analysis, returned, status, drafts = _run(engine, scenario)
    assert failed is MessageStatus.FAILED
    assert (analysis.outcome, analysis.detail) == (AnalysisOutcome.FAILED, "schema")
    assert returned == 1
    assert status is MessageStatus.DONE
    assert len(drafts) == 1


def test_edit_during_the_request_discards_the_answer(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "text": "поправь шапку"})
        fake = FakeClassifier({1: _decision()})
        worker = _worker(engine, fake)

        async def edit(_thread):
            async with AsyncSession(engine) as session:
                await session.execute(
                    update(Message).where(Message.msg_id == 1).values(text="уже не нужно")
                )
                await session.commit()

        fake.on_classify = edit
        async with AsyncSession(engine) as session:
            message_id = await session.scalar(select(Message.id).where(Message.msg_id == 1))
        result = await worker.process(message_id, RULES)
        return (
            result,
            await _status(engine, 1),
            await _rows(engine, Draft),
            await _rows(engine, Analysis),
        )

    result, status, drafts, analyses = _run(engine, scenario)
    assert result is Result.CHANGED
    assert status is MessageStatus.PENDING
    assert drafts == []
    assert analyses == []


def test_edited_message_updates_its_draft_instead_of_making_another(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "text": "поправь шапку"})
        await _drain(engine, FakeClassifier({1: _decision()}))
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Message).where(Message.msg_id == 1).values(status=MessageStatus.PENDING)
            )
            await session.commit()
        fake = FakeClassifier({1: _decision(False)})
        await _drain(engine, fake)
        return await _rows(engine, Draft), await _links(engine), await _rows(engine, Analysis)

    drafts, links, analyses = _run(engine, scenario)
    assert len(drafts) == 1
    assert links == {1: DraftRole.CONTEXT}
    assert [row.outcome for row in analyses] == [AnalysisOutcome.NO_DRAFT]


def test_one_broken_message_does_not_block_the_rest(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2})
        fake = FakeClassifier({1: RuntimeError("boom"), 2: _decision()})
        counters = await _drain(engine, fake)
        return counters, await _status(engine, 1), await _status(engine, 2)

    counters, first, second = _run(engine, scenario)
    assert counters["errors"] == 1
    assert first is MessageStatus.PENDING
    assert second is MessageStatus.DONE


# Drafts


def test_evidence_the_model_invented_is_dropped(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.DONE, "text": "контекст"},
            {"msg_id": 2},
            # A real message of this chat, but the model wasn't shown it.
            {
                "msg_id": 50,
                "status": MessageStatus.DONE,
                "text": "чужое",
                "sent_at": NOW + timedelta(days=3),
            },
        )
        await _drain(engine, FakeClassifier({2: _decision(evidence=(1, 50, 999))}))
        return await _links(engine)

    assert _run(engine, scenario) == {1: DraftRole.EVIDENCE, 2: DraftRole.TRIGGER}


def test_placeholder_in_a_draft_is_flagged(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2, "chat_id": OTHER_CHAT})
        await _drain(
            engine,
            FakeClassifier(
                {
                    1: _decision(description="Спросить у <PERSON_3>"),
                    2: _decision(description="Спросить у дизайнера"),
                }
            ),
        )
        return await _rows(engine, Draft)

    flags = {draft.description: draft.has_placeholders for draft in _run(engine, scenario)}
    assert flags == {"Спросить у <PERSON_3>": True, "Спросить у дизайнера": False}


# Merging


def _reply(msg_id: int, reply_to: int) -> dict:
    return {"msg_id": msg_id, "reply_to_msg_id": reply_to}


def test_reply_to_a_draft_message_joins_it_without_asking_the_model(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, _reply(2, 1))
        fake = FakeClassifier(
            {1: _decision(), 2: _decision()}, rewritten=DraftText(title="Вместе", description="Всё")
        )
        await _drain(engine, fake)
        return fake, await _rows(engine, Draft), await _links(engine)

    fake, drafts, links = _run(engine, scenario)
    assert fake.same_calls == []
    assert [line.msg_id for line in fake.rewrite_calls[0]] == [1, 2]
    assert [(d.title, d.description) for d in drafts] == [("Вместе", "Всё")]
    assert links == {1: DraftRole.TRIGGER, 2: DraftRole.TRIGGER}


def test_shared_evidence_joins_a_draft(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.DONE, "text": "контекст"},
            {"msg_id": 2},
            {"msg_id": 3},
        )
        fake = FakeClassifier({2: _decision(evidence=(1,)), 3: _decision(evidence=(1,))})
        await _drain(engine, fake)
        return fake.same_calls, await _rows(engine, Draft)

    same_calls, drafts = _run(engine, scenario)
    assert same_calls == []
    assert len(drafts) == 1


def test_model_decides_when_nothing_links_the_messages(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2})
        fake = FakeClassifier({1: _decision(), 2: _decision()})
        fake.same = 1
        await _drain(engine, fake)
        return fake, await _rows(engine, Draft)

    fake, drafts = _run(engine, scenario)
    assert [summary.id for summary in fake.same_calls[0]] == [1]
    assert [line.msg_id for line in fake.same_calls[0][0].lines] == [1]
    assert 2 in [line.msg_id for line in fake.same_contexts[0]]
    assert len(drafts) == 1
    assert len(fake.rewrite_calls) == 1


def test_unsure_model_keeps_two_drafts(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2})
        await _drain(engine, FakeClassifier({1: _decision(), 2: _decision()}, same=None))
        return await _rows(engine, Draft)

    assert len(_run(engine, scenario)) == 2


def test_old_draft_is_not_merged_into(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1},
            {"msg_id": 2, "sent_at": NOW + timedelta(hours=30), "reply_to_msg_id": 1},
        )
        fake = FakeClassifier({1: _decision(), 2: _decision()}, same=1)
        await _drain(engine, fake)
        return fake.same_calls, await _rows(engine, Draft)

    same_calls, drafts = _run(engine, scenario)
    assert same_calls == []
    assert len(drafts) == 2


def test_draft_of_another_chat_is_not_merged_into(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2, "chat_id": OTHER_CHAT})
        fake = FakeClassifier({1: _decision(), 2: _decision()}, same=1)
        await _drain(engine, fake)
        return fake.same_calls, await _rows(engine, Draft)

    same_calls, drafts = _run(engine, scenario)
    assert same_calls == []
    assert len(drafts) == 2


def test_clarifying_reply_becomes_context_of_the_draft(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, _reply(2, 1))
        fake = FakeClassifier({1: _decision()})
        await _drain(engine, fake)
        return fake, await _links(engine), await _rows(engine, Analysis)

    fake, links, analyses = _run(engine, scenario)
    assert links == {1: DraftRole.TRIGGER, 2: DraftRole.CONTEXT}
    assert len(fake.rewrite_calls) == 1
    assert sorted(row.outcome.value for row in analyses) == ["draft", "no_draft"]


def test_unrelated_message_does_not_touch_a_draft(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2})
        fake = FakeClassifier({1: _decision()})
        await _drain(engine, fake)
        return fake.rewrite_calls, fake.same_calls, await _links(engine)

    rewrites, same_calls, links = _run(engine, scenario)
    assert rewrites == []
    assert same_calls == []
    assert links == {1: DraftRole.TRIGGER}


def test_noise_reply_does_not_join_a_draft(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2, "text": "ок", "reply_to_msg_id": 1})
        fake = FakeClassifier({1: _decision()})
        await _drain(engine, fake)
        return await _links(engine)

    assert _run(engine, scenario) == {1: DraftRole.TRIGGER}


def test_failed_rewrite_keeps_the_text_and_says_so(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1}, _reply(2, 1))
        fake = FakeClassifier({1: _decision(title="Первое"), 2: _decision()})
        fake.rewritten = None
        original = fake.rewrite

        async def fail(lines):
            await original(lines)
            return None

        fake.rewrite = fail
        await _drain(engine, fake)
        return await _rows(engine, Draft), await _links(engine)

    drafts, links = _run(engine, scenario)
    assert [(d.title, d.text_is_stale) for d in drafts] == [("Первое", True)]
    assert set(links) == {1, 2}


def test_rewrite_only_sees_anonymized_text(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "text": "Иван, поправь", "anonymized": "<PERSON_1>, поправь"},
            _reply(2, 1),
        )
        fake = FakeClassifier({1: _decision(), 2: _decision()})
        await _drain(engine, fake)
        return fake.rewrite_calls[0]

    lines = _run(engine, scenario)
    assert "<PERSON_1>, поправь" in [line.text for line in lines]
    assert all("Иван" not in line.text for line in lines)


def test_rewrite_waits_when_a_draft_message_lost_its_record(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision()}))
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        async with AsyncSession(engine) as session:
            message_id = await session.scalar(select(Message.id).where(Message.msg_id == 1))
            record = await session.get(AnonymizedMessage, message_id)
            await session.delete(record)
            await session.commit()
        fake = FakeClassifier({2: _decision()})
        await _drain(engine, fake)
        return fake.rewrite_calls, await _status(engine, 2)

    rewrites, status = _run(engine, scenario)
    assert rewrites == []
    assert status is MessageStatus.PENDING


# Promises


def test_own_message_goes_through_the_promise_path_and_joins_the_request(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1},
            {"msg_id": 2, "text": "поправлю сегодня", "is_outgoing": True, "reply_to_msg_id": 1},
        )
        fake = FakeClassifier({1: _decision(), 2: _decision(reason="promise")})
        await _drain(engine, fake)
        return fake.classified, await _rows(engine, Draft), await _links(engine)

    classified, drafts, links = _run(engine, scenario)
    assert classified == [(1, False), (2, True)]
    assert len(drafts) == 1
    assert links == {1: DraftRole.TRIGGER, 2: DraftRole.TRIGGER}


def test_incoming_schema_cannot_answer_promise():
    assert "promise" not in json.dumps(verdict_schema())
    with pytest.raises(InvalidVerdictError):
        parse_answer(
            TaskVerdict,
            json.dumps(
                {
                    "is_task": True,
                    "assigned_to_me": True,
                    "assignment_reason": "promise",
                    "confidence": 1,
                    "title": "т",
                    "description": "о",
                    "deadline": None,
                    "priority": "medium",
                    "evidence": [],
                }
            ),
        )


# Classifier against a fake server


def _ollama(handler) -> OllamaClient:
    return OllamaClient("http://test", httpx.MockTransport(handler))


def _reply_with(content: str, **extra) -> httpx.Response:
    body = {"message": {"content": content}, "done_reason": "stop", "prompt_eval_count": 10}
    return httpx.Response(200, json=body | extra)


def _classifier(client: OllamaClient) -> Classifier:
    return Classifier(client, "gemma4:12b", PROMPTS, "task_v4.md", SCOPE)


def _thread(sender: str = "<PERSON_1>"):
    from task_manager.analyzer.thread import Thread

    return Thread(
        chat_kind=ChatKind.GROUP,
        mentions_me=False,
        target=ThreadLine(msg_id=1, sender=sender, text="текст", sent_at="2026-11-04T12:00"),
        reply_chain=[],
        previous=[],
    )


TASK_ANSWER = json.dumps(
    {
        "is_task": True,
        "assigned_to_me": True,
        "assignment_reason": "mention",
        "confidence": 0.9,
        "title": "т",
        "description": "о",
        "deadline": None,
        "priority": "medium",
        "evidence": [1],
    }
)
PROMISE_ANSWER = json.dumps(
    {
        "is_promise": True,
        "confidence": 0.9,
        "title": "т",
        "description": "о",
        "deadline": None,
        "priority": "medium",
        "evidence": [1],
    }
)


def _chat_handler(chat):
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={})
        return chat(json.loads(request.content))

    return handler


def test_first_attempt_uses_the_seed_the_evaluation_used():
    seen = []

    def chat(body):
        seen.append(body["options"]["seed"])
        return _reply_with(TASK_ANSWER)

    async def scenario():
        client = _ollama(_chat_handler(chat))
        decision = await _classifier(client).classify(_thread(), own=False)
        await client.aclose()
        return decision

    decision = asyncio.run(scenario())
    assert seen == [SEED]
    assert decision.creates_draft
    assert decision.reason == "mention"


def test_bad_answer_is_retried_with_another_seed():
    seen = []

    def chat(body):
        seen.append(body["options"]["seed"])
        return _reply_with("не json" if len(seen) == 1 else TASK_ANSWER)

    async def scenario():
        client = _ollama(_chat_handler(chat))
        decision = await _classifier(client).classify(_thread(), own=False)
        await client.aclose()
        return decision

    assert asyncio.run(scenario()).creates_draft
    assert seen == [SEED, SEED + 1]


@pytest.mark.parametrize(
    ("reply", "code", "calls"),
    [
        (lambda: _reply_with("не json"), "schema", ATTEMPTS),
        (lambda: _reply_with("{", done_reason="length"), "output_too_long", ATTEMPTS),
        (lambda: _reply_with(TASK_ANSWER, prompt_eval_count=10_000), "input_too_long", 1),
    ],
)
def test_answer_errors_carry_a_fixed_code(reply, code, calls):
    seen = []

    def chat(_body):
        seen.append(1)
        return reply()

    async def scenario():
        client = _ollama(_chat_handler(chat))
        try:
            await _classifier(client).classify(_thread(), own=False)
        finally:
            await client.aclose()

    with pytest.raises(AnswerError) as error:
        asyncio.run(scenario())
    assert str(error.value) == code
    assert len(seen) == calls


def test_task_without_text_is_a_bad_answer():
    empty = json.loads(TASK_ANSWER) | {"title": " ", "description": ""}

    async def scenario():
        client = _ollama(_chat_handler(lambda _body: _reply_with(json.dumps(empty))))
        try:
            await _classifier(client).classify(_thread(), own=False)
        finally:
            await client.aclose()

    with pytest.raises(AnswerError):
        asyncio.run(scenario())


def test_outage_is_not_an_answer_error():
    def handler(request):
        raise httpx.ConnectError("down")

    async def scenario():
        client = _ollama(handler)
        try:
            await _classifier(client).classify(_thread(), own=False)
        finally:
            await client.aclose()

    with pytest.raises(OllamaUnavailableError):
        asyncio.run(scenario())


def test_own_message_is_asked_with_the_promise_prompt():
    systems = []

    def chat(body):
        systems.append(body["messages"][0]["content"])
        return _reply_with(PROMISE_ANSWER)

    async def scenario():
        client = _ollama(_chat_handler(chat))
        decision = await _classifier(client).classify(_thread("<ME>"), own=True)
        await client.aclose()
        return decision

    decision = asyncio.run(scenario())
    assert "пообещал" in systems[0]
    assert decision.reason == "promise"
    assert decision.creates_draft


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"same_draft": 3}', 3),
        ('{"same_draft": null}', None),
        ('{"same_draft": 99}', None),
        ("мусор", None),
    ],
)
def test_same_task_accepts_only_a_listed_draft(content, expected):
    async def scenario():
        client = _ollama(_chat_handler(lambda _body: _reply_with(content)))
        result = await _classifier(client).same_task("т", "о", [DraftSummary(3, "т", "о")])
        await client.aclose()
        return result

    assert asyncio.run(scenario()) == expected


def test_same_task_shows_the_model_the_messages_of_both_sides():
    users = []

    def chat(body):
        users.append(body["messages"][1]["content"])
        return _reply_with('{"same_draft": 3}')

    def line(msg_id, text):
        return ThreadLine(msg_id=msg_id, sender="<ME>", text=text, sent_at="2026-11-04T12:00")

    async def scenario():
        client = _ollama(_chat_handler(chat))
        result = await _classifier(client).same_task(
            "т",
            "о",
            [DraftSummary(3, "т", "о", (line(1, "из черновика"),))],
            [line(2, "из обсуждения")],
        )
        await client.aclose()
        return result

    assert asyncio.run(scenario()) == 3
    assert "из черновика" in users[0]
    assert "из обсуждения" in users[0]


def test_failed_rewrite_returns_nothing():
    async def scenario():
        client = _ollama(_chat_handler(lambda _body: _reply_with("мусор")))
        result = await _classifier(client).rewrite(
            [ThreadLine(msg_id=1, sender="<ME>", text="т", sent_at="2026-11-04T12:00")]
        )
        await client.aclose()
        return result

    assert asyncio.run(scenario()) is None


@pytest.mark.parametrize("name", ["promise_v1.md", "promise_v2.md", "merge_v1.md", "rewrite_v1.md"])
def test_helper_prompts_render_without_leftovers(name):
    system = render_system(load_template(PROMPTS / name), SCOPE)
    assert "$" not in system


# Service


def _capture():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter("test"))
    return stream, handler


def test_logs_hold_no_text_from_messages_or_answers(engine):
    secret_message = "поправь шапку у Иннокентия"
    stream, handler = _capture()
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "text": secret_message},
            {"msg_id": 2, "text": secret_message + " ещё"},
            _reply(3, 1),
        )
        fake = FakeClassifier(
            {1: _decision(title=secret_message), 2: AnswerError("schema"), 3: _decision()},
        )
        fake.rewritten = None

        async def fail(_lines):
            return None

        fake.rewrite = fail
        await _drain(engine, fake)

    try:
        _run(engine, scenario)
    finally:
        root.removeHandler(handler)
    assert "Иннокентия" not in stream.getvalue()
    assert "analysis_failed" in stream.getvalue()


def test_heartbeat_stays_fresh_during_a_slow_model_call(engine, tmp_path, monkeypatch):
    monkeypatch.setattr("task_manager.service.HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(runner, "IDLE_INTERVAL_S", 0.01)
    path = tmp_path / "analyzer.heartbeat"

    async def scenario():
        await _seed(engine, {"msg_id": 1})
        stop = asyncio.Event()
        fake = FakeClassifier({1: _decision()})
        touched = []

        async def slow(_thread):
            first = path.stat().st_mtime_ns
            await asyncio.sleep(0.2)
            touched.append(path.stat().st_mtime_ns > first)
            stop.set()

        fake.on_classify = slow
        await runner.run_analyzer(_worker(engine, fake), path, stop)
        return touched

    assert _run(engine, scenario) == [True]
    assert not path.exists()


def test_service_survives_an_outage_and_recovers(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FIRST_RETRY_S", 0.01)
    monkeypatch.setattr(runner, "IDLE_INTERVAL_S", 0.01)
    stream, handler = _capture()
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    async def scenario():
        await _seed(engine, {"msg_id": 1})
        stop = asyncio.Event()
        fake = FakeClassifier({1: OllamaUnavailableError("ConnectError")})
        statuses = []

        async def control(_thread):
            if len(fake.classified) == 3:
                statuses.append(await _status(engine, 1))
                fake.answers[1] = _decision()
                fake.on_classify = None

        fake.on_classify = control
        task = asyncio.create_task(
            runner.run_analyzer(_worker(engine, fake), tmp_path / "beat", stop)
        )
        while await _status(engine, 1) is not MessageStatus.DONE:
            await asyncio.sleep(0.01)
        stop.set()
        await task
        return statuses, await _rows(engine, Draft)

    try:
        statuses, drafts = _run(engine, scenario)
    finally:
        root.removeHandler(handler)
    assert statuses == [MessageStatus.PENDING]
    assert len(drafts) == 1
    logged = stream.getvalue()
    assert logged.count("ollama_unavailable") == 1
    assert logged.count("ollama_restored") == 1


# Commands


def test_archive_unlisted_touches_only_waiting_messages_of_unlisted_chats(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "chat_id": UNLISTED_CHAT},
            {"msg_id": 2, "chat_id": UNLISTED_CHAT, "status": MessageStatus.DONE},
            {"msg_id": 3},
        )
        count = await commands.archive_unlisted(engine, [CHAT, OTHER_CHAT])
        return (
            count,
            await _status(engine, 1, UNLISTED_CHAT),
            await _status(engine, 2, UNLISTED_CHAT),
            await _status(engine, 3),
        )

    assert _run(engine, scenario) == (
        1,
        MessageStatus.ARCHIVED,
        MessageStatus.DONE,
        MessageStatus.PENDING,
    )


def test_preview_keeps_me_visible_and_counters_hold_no_text(engine, tmp_path):
    path = tmp_path / "drafts.md"

    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "text": "Тестовый, поправь шапку", "anonymized": "<ME>, поправь шапку"},
            {"msg_id": 2, "reply_to_msg_id": 1, "text": "```код```", "anonymized": "```код```"},
        )
        fake = FakeClassifier(
            {1: _decision(title="Шапка"), 2: _decision(title="Шапка")},
            rewritten=DraftText(title="Шапка целиком", description="Спросить у <PERSON_2>"),
        )
        await _drain(engine, fake)
        count = await commands.write_drafts_preview(engine, path, RULES)
        return count, await commands.draft_counters(engine)

    count, counters = _run(engine, scenario)
    text = path.read_text("utf-8")
    assert count == 1
    assert "<ME>, поправь шапку" in text
    assert "Шапка целиком" in text
    assert "ЕСТЬ ПЛЕЙСХОЛДЕРЫ" in text
    assert "````text" in text
    assert counters["drafts"] == 1
    assert counters["drafts_merged"] == 1
    assert counters["analysis_draft"] == 2
    assert "шапку" not in json.dumps(counters, ensure_ascii=False)


def test_replay_copy_leaves_the_original_alone(engine, tmp_path):
    source = tmp_path / "app.db"
    target = tmp_path / "copy.db"

    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.ARCHIVED},
            {"msg_id": 2, "status": MessageStatus.ARCHIVED, "chat_id": UNLISTED_CHAT},
            {"msg_id": 3},
            {"msg_id": 4, "status": MessageStatus.ARCHIVED, "chat_id": OTHER_CHAT},
        )
        await _drain(engine, FakeClassifier({3: _decision()}))
        before = await commands.status_counts(engine)
        count = commands.make_replay_copy(source, target, [CHAT, OTHER_CHAT])
        copy_engine = make_engine(target)
        try:
            copy_counts = await commands.status_counts(copy_engine)
            copy_drafts = await _rows(copy_engine, Draft)
            fake = FakeClassifier({1: _decision()})
            await commands.run_replay(_worker(copy_engine, fake))
            replayed = await _rows(copy_engine, Draft)
        finally:
            await copy_engine.dispose()
        return (
            before,
            count,
            copy_counts,
            copy_drafts,
            replayed,
            await commands.status_counts(engine),
            await _rows(engine, Draft),
            fake.classified,
        )

    before, count, copy_counts, copy_drafts, replayed, after, real_drafts, classified = _run(
        engine, scenario
    )
    assert count == 2
    assert copy_counts == {"pending": 2, "archived": 1, "done": 1}
    assert copy_drafts == []
    assert len(replayed) == 1
    assert sorted(msg_id for msg_id, _ in classified) == [1, 4]
    assert before == after
    assert len(real_drafts) == 1
    commands.remove_replay_copy(target)
    assert not target.exists()


def test_replay_copy_can_cover_only_recent_days(engine, tmp_path):
    source = tmp_path / "app.db"
    target = tmp_path / "copy.db"

    async def scenario():
        now = datetime.now(UTC)
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.ARCHIVED, "sent_at": now - timedelta(days=40)},
            {"msg_id": 2, "status": MessageStatus.ARCHIVED, "sent_at": now - timedelta(days=2)},
        )
        return commands.make_replay_copy(source, target, [CHAT], days=7)

    assert _run(engine, scenario) == 1
    copy = sqlite3.connect(target)
    assert copy.execute("SELECT msg_id FROM messages WHERE status = 'pending'").fetchall() == [(2,)]
    copy.close()


# Own messages for labelling


def test_own_sample_holds_only_the_owners_messages(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.ARCHIVED},
            {
                "msg_id": 2,
                "status": MessageStatus.ARCHIVED,
                "text": "сделаю завтра",
                "is_outgoing": True,
            },
            {
                "msg_id": 3,
                "status": MessageStatus.ARCHIVED,
                "text": "думаю, нормально",
                "is_outgoing": True,
                "reply_to_msg_id": 1,
            },
            {"msg_id": 4, "status": MessageStatus.ARCHIVED, "text": "вот так", "is_outgoing": True},
        )
        return await sampling.collect_examples(engine, [CHAT], ME, SCOPE, ANON, own=True)

    examples, skipped = _run(engine, scenario)
    assert skipped == 0
    assert {e.thread.target.msg_id: e.stratum for e in examples} == {
        2: "promise_word",
        3: "reply_to_other",
        4: "other",
    }


def test_own_labelling_page_offers_only_its_labels(engine):
    async def scenario():
        await _seed(
            engine,
            {"msg_id": 1, "status": MessageStatus.ARCHIVED, "is_outgoing": True, "text": "сделаю"},
        )
        examples, skipped = await sampling.collect_examples(
            engine, [CHAT], ME, SCOPE, ANON, own=True
        )
        return sampling.select_sample(
            examples, skipped, sampling.OWN_SEED, sampling.OWN_RANDOM_SIZE, sampling.OWN_QUOTAS
        )

    page = labeling.render_labeling_page(
        _run(engine, scenario), None, labeling.OWN_LABELS, labeling.OWN_LABELS_HINT
    )
    assert "Задача мне: я взял работу на себя" in page
    assert "Возможно мне" not in page


def test_score_own_counts_drafts_against_labels():
    answers = {
        1: {"message_id": 1, "draft": True, "error": None},
        2: {"message_id": 2, "draft": True, "error": None},
        3: {"message_id": 3, "draft": False, "error": None},
        4: {"message_id": 4, "draft": False, "error": "schema"},
    }
    precision, recall, failed = own.score_own(answers, {1: 1, 2: 4, 3: 1, 4: 4})
    assert (precision.hits, precision.total) == (1, 2)
    assert (recall.hits, recall.total) == (1, 2)
    assert failed == 1


def test_clarifying_a_task_counts_neither_way_for_own_messages():
    answers = {
        1: {"message_id": 1, "draft": True, "error": None},
        2: {"message_id": 2, "draft": True, "error": None},
        3: {"message_id": 3, "draft": False, "error": None},
    }
    precision, recall, _ = own.score_own(answers, {1: 1, 2: 5, 3: 5})
    assert (precision.hits, precision.total) == (1, 1)
    assert (recall.hits, recall.total) == (1, 1)
