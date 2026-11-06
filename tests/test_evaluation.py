"""Model evaluation code. All messages here are made up, and Ollama is replaced by a fake."""

import asyncio
import io
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.analyzer.ollama import (
    CloudModelError,
    OllamaClient,
    OllamaUnavailableError,
    ensure_local,
    think_setting,
)
from task_manager.analyzer.prompt import load_template, render_system, render_user
from task_manager.analyzer.signals import SignalDetector, TermMatcher
from task_manager.analyzer.thread import Thread, ThreadLine, build_thread
from task_manager.analyzer.verdict import (
    AssignmentReason,
    InvalidVerdictError,
    Priority,
    TaskVerdict,
    has_placeholder,
    parse_verdict,
    verdict_schema,
)
from task_manager.anonymizer import runner as anonymizer_runner
from task_manager.common.config import (
    AnonymizerConfig,
    MeConfig,
    ScopeConfig,
    Section,
)
from task_manager.common.log import SafeJsonFormatter
from task_manager.evaluation import metrics, runner
from task_manager.evaluation.labeling import read_labels, render_labeling_page
from task_manager.evaluation.sampling import (
    RANDOM_SIZE,
    SIGNAL_QUOTAS,
    Example,
    Sample,
    collect_examples,
    refresh_sample,
    sample_counters,
    select_sample,
)
from task_manager.evaluation.split import group_examples, split_examples
from task_manager.storage.db import make_engine
from task_manager.storage.models import (
    AnonymizedMessage,
    Chat,
    ChatKind,
    Message,
    MessageStatus,
)

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
REPO_PROMPTS = Path(__file__).parent.parent / "config" / "prompts"
SCOPE = ScopeConfig(
    role=["фронтенд", "front-end"],
    sections=[Section(name="Статьи", aliases=["блог", "/blog", "личный кабинет"])],
)
ME = MeConfig(names=["Тест"])
ANON = AnonymizerConfig(allow=[], names=[])
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def line(
    msg_id: int, text: str = "текст", sender: str | None = "<PERSON_1>", **extra
) -> ThreadLine:
    return ThreadLine(
        msg_id=msg_id, sender=sender, text=text, sent_at="2026-10-10T12:00:00", **extra
    )


def thread(msg_id: int = 10, text: str = "текст", kind=ChatKind.GROUP, **extra) -> Thread:
    return Thread(
        chat_kind=kind,
        mentions_me=extra.pop("mentions_me", False),
        target=line(msg_id, text),
        reply_chain=extra.pop("reply_chain", []),
        previous=extra.pop("previous", []),
    )


def example(message_id: int, stratum: str = "none", chat_ref: int = 0, **extra) -> Example:
    return Example(
        message_id=message_id,
        chat_ref=chat_ref,
        stratum=stratum,
        signals=[] if stratum == "none" else [stratum],
        thread=extra.pop("thread", thread(message_id * 100)),
    )


def verdict(**changes) -> TaskVerdict:
    fields = {
        "is_task": True,
        "assigned_to_me": True,
        "assignment_reason": AssignmentReason.MENTION,
        "confidence": 0.9,
        "title": "Заголовок",
        "description": "Поправить отступы в шапке сайта",
        "deadline": None,
        "priority": Priority.MEDIUM,
        "evidence": [1],
    }
    return TaskVerdict(**{**fields, **changes})


def scored(label: int, result: TaskVerdict | None, stratum: str = "none") -> metrics.Scored:
    return metrics.Scored(message_id=1, label=label, stratum=stratum, verdict=result)


# Answer schema


def test_valid_answer_is_accepted():
    assert parse_verdict(verdict().model_dump_json()).is_task


@pytest.mark.parametrize(
    "change",
    [
        {"assignment_reason": "boss"},
        {"confidence": 1.5},
        {"deadline": "в пятницу"},
        {"priority": "urgent"},
        {"extra_field": 1},
    ],
)
def test_answer_outside_the_schema_is_an_error(change):
    payload = {**json.loads(verdict().model_dump_json()), **change}
    with pytest.raises(InvalidVerdictError):
        parse_verdict(json.dumps(payload))


def test_invalid_answer_error_does_not_repeat_the_text():
    with pytest.raises(InvalidVerdictError) as error:
        parse_verdict('{"title": "секретная фраза"}')
    assert "секретная фраза" not in str(error.value)
    with pytest.raises(InvalidVerdictError):
        parse_verdict("не json вообще")


def test_schema_lists_allowed_reasons():
    assert "collective" in json.dumps(verdict_schema())


# Client


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _chat_reply(content: str = "{}", **extra) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": content},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
            **extra,
        },
    )


def test_cloud_models_are_refused_before_any_request():
    def handler(request):
        raise AssertionError("no request may be sent")

    async def scenario():
        client = OllamaClient("http://test", _transport(handler))
        try:
            with pytest.raises(CloudModelError):
                await client.chat("gpt-oss:20b-cloud", "s", "u", {})
        finally:
            await client.aclose()

    asyncio.run(scenario())
    with pytest.raises(CloudModelError):
        ensure_local("kimi:cloud")
    ensure_local("qwen3:14b")


def test_chat_sends_schema_and_fixed_options():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _chat_reply("ответ")

    async def scenario():
        client = OllamaClient("http://test", _transport(handler))
        try:
            return await client.chat("m", "system", "user", {"type": "object"}, think=False)
        finally:
            await client.aclose()

    result = asyncio.run(scenario())
    assert result.content == "ответ"
    assert seen["format"] == {"type": "object"}
    assert seen["think"] is False
    assert seen["stream"] is False
    assert seen["options"]["temperature"] == 0
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


def test_unavailable_server_raises_without_text():
    def handler(request):
        raise httpx.ConnectError("секретная фраза")

    async def scenario():
        client = OllamaClient("http://test", _transport(handler))
        try:
            with pytest.raises(OllamaUnavailableError) as error:
                await client.chat("m", "s", "секретная фраза", {})
            assert "секретная фраза" not in str(error.value)
        finally:
            await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("show", "expected"),
    [
        ({}, None),
        ({"thinking": {"values": [False]}}, False),
        ({"thinking": {"values": [True, False]}}, False),
        ({"thinking": {"values": ["low", "medium", "high"]}}, "low"),
    ],
)
def test_think_setting_asks_for_the_least_thinking(show, expected):
    assert think_setting(show) == expected


# Signals


def test_terms_match_in_any_case_form():
    matcher = TermMatcher(["блог", "личный кабинет", "/blog"])
    assert matcher.matches("Обнови блоге раздел")
    assert matcher.matches("Правки по личному кабинету")
    assert matcher.matches("Сломалось <URL_1>/blog/post")
    assert not matcher.matches("Ничего такого здесь нет")


@pytest.mark.parametrize(
    ("built", "signal"),
    [
        (thread(text="<ME>, посмотри"), "mention"),
        (thread(mentions_me=True), "mention"),
        (thread(kind=ChatKind.PRIVATE), "dm"),
        (thread(text="Сломался блог"), "section"),
        (thread(text="Вопрос к фронтенду"), "role"),
        (thread(text="Мы это обсудили"), "collective"),
        (thread(text="Посмотри, пожалуйста"), "request"),
    ],
)
def test_signals_are_detected(built, signal):
    assert SignalDetector(SCOPE).detect(built)[0] == signal


def test_reply_to_me_is_a_signal_but_reply_to_others_is_not():
    detector = SignalDetector(SCOPE)
    mine = Thread(
        chat_kind=ChatKind.GROUP,
        mentions_me=False,
        target=line(10, reply_to=9),
        reply_chain=[line(9, sender="<ME>")],
        previous=[],
    )
    other = mine.model_copy(update={"reply_chain": [line(9)]})
    assert "reply" in detector.detect(mine)
    assert "reply" not in detector.detect(other)


def test_plain_text_has_no_signal():
    assert SignalDetector(SCOPE).detect(thread(text="Хорошая погода сегодня")) == []


# Sampling


def _population() -> list[Example]:
    built = []
    next_id = 1
    for stratum, size in {"mention": 300, "dm": 500, "role": 4, "none": 400}.items():
        for _ in range(size):
            built.append(example(next_id, stratum))
            next_id += 1
    return built


def test_sample_is_reproducible():
    first = select_sample(_population(), 0)
    second = select_sample(list(reversed(_population())), 0)
    assert [e.message_id for e in first.examples] == [e.message_id for e in second.examples]


def test_sample_has_no_duplicates_and_expected_size():
    sample = select_sample(_population(), 0)
    ids = [e.message_id for e in sample.examples]
    assert len(ids) == len(set(ids))
    assert len(ids) == RANDOM_SIZE + sum(SIGNAL_QUOTAS.values())


def test_rare_group_is_included_and_unused_places_are_shared():
    sample = select_sample(_population(), 0)
    counters = sample_counters(sample)
    assert counters["sampled_role"] == 4
    assert sample.population == {"mention": 300, "dm": 500, "role": 4, "none": 400}


def test_small_population_is_taken_whole():
    population = [example(i, "mention") for i in range(1, 11)]
    assert len(select_sample(population, 0).examples) == 10


def test_counters_hold_no_text():
    population = [example(1, "mention", thread=thread(text="секретная фраза"))]
    assert "секретная фраза" not in json.dumps(sample_counters(select_sample(population, 3)))


# Reading from the database


@pytest.fixture
def engine(tmp_path, monkeypatch) -> AsyncEngine:
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return make_engine(path)


RULES = anonymizer_runner.current_rules([], ME, SCOPE, ANON)


async def _add(session: AsyncSession, msg_id: int, text: str, **extra) -> Message:
    anonymized = extra.pop("anonymized", f"обезличено {msg_id}")
    sender = extra.pop("sender", "<PERSON_1>")
    rules = extra.pop("rules", RULES)
    from task_manager.anonymizer.store import source_hash

    message = Message(
        chat_id=extra.pop("chat_id", 1),
        msg_id=msg_id,
        sent_at=extra.pop("sent_at", NOW + timedelta(minutes=msg_id)),
        text=text,
        status=extra.pop("status", MessageStatus.ARCHIVED),
        **extra,
    )
    session.add(message)
    await session.flush()
    if anonymized is not None:
        session.add(
            AnonymizedMessage(
                message_id=message.id,
                text=anonymized,
                sender=sender,
                source_hash=source_hash(text, []),
                rules_version=rules,
            )
        )
    await session.flush()
    return message


def _in_db(engine: AsyncEngine, scenario):
    async def wrapper():
        try:
            async with AsyncSession(engine, expire_on_commit=False) as session:
                session.add(Chat(chat_id=1, kind=ChatKind.GROUP))
                await session.flush()
                result = await scenario(session)
                await session.commit()
                return result
        finally:
            await engine.dispose()

    return asyncio.run(wrapper())


def test_thread_holds_only_anonymized_text(engine):
    async def scenario(session):
        await _add(session, 1, "Иван Настоящий раньше писал")
        await _add(session, 2, "Настоящая просьба", reply_to_msg_id=1)
        target = await _add(session, 3, "Настоящий текст", reply_to_msg_id=2)
        return await build_thread(session, target, RULES)

    built = _in_db(engine, scenario)
    dumped = built.model_dump_json()
    assert "Настоящ" not in dumped
    assert [item.msg_id for item in built.reply_chain] == [1, 2]
    assert built.previous == []


def test_thread_is_blocked_when_context_is_not_ready(engine):
    async def scenario(session):
        await _add(session, 1, "Настоящий контекст", anonymized=None)
        target = await _add(session, 2, "Настоящий текст")
        return await build_thread(session, target, RULES)

    assert _in_db(engine, scenario) is None


def test_thread_is_blocked_on_old_rules(engine):
    async def scenario(session):
        target = await _add(session, 1, "Настоящий текст", rules="old")
        return await build_thread(session, target, RULES)

    assert _in_db(engine, scenario) is None


def test_old_messages_are_not_context(engine):
    async def scenario(session):
        await _add(session, 1, "Старое", sent_at=NOW - timedelta(days=3))
        target = await _add(session, 2, "Новое", sent_at=NOW)
        return await build_thread(session, target, RULES)

    assert _in_db(engine, scenario).previous == []


def test_population_excludes_own_empty_pending_and_not_ready(engine):
    async def scenario(session):
        await _add(session, 1, "Подходит")
        await _add(session, 2, "Своё", is_outgoing=True, sender="<ME>")
        await _add(session, 3, "Очередь", status=MessageStatus.PENDING)
        await _add(session, 4, "", anonymized="")
        await session.commit()
        return await collect_examples(engine, [1], ME, SCOPE, ANON)

    examples, skipped = _in_db(engine, scenario)
    assert skipped == 0
    assert [e.thread.target.msg_id for e in examples] == [1]
    assert "Подходит" not in examples[0].model_dump_json()


# Labelling page


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.current: str | None = None
        self.csp = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script":
            self.current = attributes.get("id", "code")
        if attributes.get("http-equiv") == "Content-Security-Policy":
            self.csp = attributes["content"]

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None

    def handle_data(self, data):
        if self.current == "sample":
            self.scripts.append(data)


def _page_for(text: str) -> tuple[str, _Page]:
    sample = Sample(
        seed=1,
        population={"none": 1},
        skipped=0,
        examples=[example(1, thread=thread(text=text))],
    )
    html = render_labeling_page(sample)
    parser = _Page()
    parser.feed(html)
    return html, parser


def test_message_cannot_break_out_of_the_page():
    hostile = "</script><script>alert(1)</script><!-- <ME> &  "
    html, parser = _page_for(hostile)
    assert "alert(1)</script>" not in html
    payload = json.loads("".join(parser.scripts))
    assert payload["examples"][0]["target"]["text"] == hostile


def test_placeholders_survive_in_the_page_data():
    _, parser = _page_for("<ME>, посмотри <SECRET>")
    payload = json.loads("".join(parser.scripts))
    assert payload["examples"][0]["target"]["text"] == "<ME>, посмотри <SECRET>"


def test_page_uses_text_nodes_and_forbids_network():
    html, parser = _page_for("текст")
    assert "innerHTML" not in html
    assert "default-src 'none'" in parser.csp
    assert not re.search(r"https?://", html)


def test_page_does_not_reveal_the_sampling_group():
    _, parser = _page_for("текст")
    assert "stratum" not in "".join(parser.scripts)


def test_chat_outside_the_whitelist_is_excluded(engine):
    async def scenario(session):
        session.add(Chat(chat_id=2, kind=ChatKind.PRIVATE))
        await session.flush()
        await _add(session, 1, "Рабочее")
        await _add(session, 2, "Нерабочее", chat_id=2)
        await session.commit()
        return await collect_examples(engine, [1], ME, SCOPE, ANON)

    examples, _ = _in_db(engine, scenario)
    assert [e.thread.target.msg_id for e in examples] == [1]


def test_refresh_keeps_order_and_drops_what_is_no_longer_eligible():
    existing = Sample(
        seed=7, population={}, skipped=0, examples=[example(3), example(1), example(2)]
    )
    refreshed = refresh_sample(existing, [example(1, "dm"), example(3), example(9, "mention")], 2)
    assert [e.message_id for e in refreshed.examples] == [3, 1]
    assert refreshed.examples[1].stratum == "dm"
    assert refreshed.population == {"dm": 1, "none": 1, "mention": 1}
    assert refreshed.seed == 7


def _page_payload(labels) -> dict:
    sample = Sample(seed=1, population={}, skipped=0, examples=[example(1), example(2)])
    parser = _Page()
    parser.feed(render_labeling_page(sample, labels))
    return json.loads("".join(parser.scripts))


def test_page_opens_with_existing_labels_for_its_messages_only():
    assert _page_payload({1: 5, 99: 1})["labels"] == {"1": 5}


def test_corrected_labels_get_a_new_revision():
    assert _page_payload({1: 4})["revision"] != _page_payload({1: 1})["revision"]


def test_part_of_a_task_counts_neither_way():
    rows = [
        scored(1, verdict()),
        scored(metrics.LABEL_PART, verdict()),
        scored(metrics.LABEL_PART, None),
    ]
    result = metrics.score(rows, 0.5, {"none": 3})
    assert (result.precision.hits, result.precision.total) == (1, 1)
    assert (result.recall.hits, result.recall.total) == (1, 1)
    assert metrics.precision_by_reason(rows, 0.5)["mention"].total == 1


def test_labels_are_validated(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text('{"5": 1, "6": 5}', encoding="utf-8")
    assert read_labels(path) == {5: 1, 6: 5}
    path.write_text('{"5": 9}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown labels"):
        read_labels(path)


# Split


def test_messages_sharing_context_stay_together():
    shared = line(50)
    first = example(1, thread=thread(60, previous=[shared]))
    second = example(2, thread=thread(70, previous=[shared]))
    other_chat = example(3, chat_ref=1, thread=thread(80, previous=[line(50)]))
    groups = group_examples([first, second, other_chat])
    assert sorted(map(len, groups)) == [1, 2]

    parts = split_examples([first, second, other_chat], {1: 1, 2: 4, 3: 4})
    assert parts[1] == parts[2]


def test_split_is_about_seventy_thirty_and_keeps_labels_balanced():
    examples = [example(i, "mention" if i % 2 else "dm") for i in range(1, 201)]
    labels = {i: 1 if i % 5 == 0 else 4 for i in range(1, 201)}
    parts = split_examples(examples, labels)
    final = [i for i, part in parts.items() if part == "final"]
    assert 50 <= len(final) <= 70
    assert 8 <= sum(labels[i] == 1 for i in final) <= 16


def test_split_ignores_unlabelled_and_is_reproducible():
    examples = [example(i) for i in range(1, 21)]
    labels = {i: 4 for i in range(1, 11)}
    assert set(split_examples(examples, labels)) == set(labels)
    assert split_examples(examples, labels) == split_examples(examples, labels)


# Metrics


def test_precision_and_recall_on_a_known_example():
    rows = [
        scored(1, verdict()),  # found
        scored(1, verdict()),  # found
        scored(1, None),  # missed
        scored(3, verdict()),  # false draft
        scored(4, verdict(is_task=False, assigned_to_me=False)),  # correct rejection
    ]
    result = metrics.score(rows, 0.5, {"none": 5})
    assert result.precision.hits == 2
    assert result.precision.total == 3
    assert result.recall.hits == 2
    assert result.recall.total == 3


def test_maybe_label_counts_against_precision_but_not_recall():
    rows = [scored(1, verdict()), scored(2, verdict()), scored(2, None)]
    result = metrics.score(rows, 0.5, {"none": 3})
    assert (result.precision.hits, result.precision.total) == (1, 2)
    assert (result.recall.hits, result.recall.total) == (1, 1)
    assert result.maybe_label_destinations == {"draft": 1, "nothing": 1}


def test_threshold_removes_uncertain_drafts():
    rows = [scored(1, verdict(confidence=0.9)), scored(4, verdict(confidence=0.4))]
    assert metrics.score(rows, 0.8, {"none": 2}).precision.value == 1
    assert metrics.score(rows, 0.1, {"none": 2}).precision.value == 0.5


def test_collective_never_makes_a_draft():
    collective = verdict(assignment_reason=AssignmentReason.COLLECTIVE)
    assert not metrics.is_draft(collective, 0.0)
    assert metrics.destination(collective, 0.0) == "maybe"


def test_failed_answer_is_not_a_draft():
    assert metrics.destination(None, 0.0) == "nothing"


def test_weighting_scales_groups_back_to_population():
    # 10 of 10 "mention" messages sampled, and 1 of 100 from the rest.
    rows = [scored(1, verdict(), "mention") for _ in range(9)]
    rows.append(scored(4, None, "mention"))
    rows.append(scored(4, verdict(), "none"))
    result = metrics.score(rows, 0.5, {"mention": 10, "none": 100})
    assert result.precision.value == 0.9
    assert result.weighted_precision == pytest.approx(9 / 109)
    assert result.weighted_recall == 1


def test_precision_by_reason():
    rows = [
        scored(1, verdict(assignment_reason=AssignmentReason.DM)),
        scored(4, verdict(assignment_reason=AssignmentReason.DM)),
        scored(1, verdict(assignment_reason=AssignmentReason.ROLE)),
    ]
    rates = metrics.precision_by_reason(rows, 0.5)
    assert rates["dm"].value == 0.5
    assert rates["role"].value == 1


def test_interval_is_wide_on_few_examples():
    few = metrics.wilson(5, 6)
    many = metrics.wilson(500, 600)
    assert few.high - few.low > many.high - many.low
    assert few.low < 0.5
    assert metrics.wilson(0, 0).value is None


def test_description_checks():
    source = [line(1, "поправь пожалуйста большие отступы в шапке сайта")]
    assert metrics.copies_source("Нужно: поправь пожалуйста большие отступы в шапке", source)
    assert not metrics.copies_source("Исправить поля в верхней части страницы", source)
    assert has_placeholder("Спросить у <PERSON_3>")
    assert has_placeholder("Пароль <SECRET>")
    assert not has_placeholder("Поправить отступы")


# Prompt


@pytest.mark.parametrize("path", sorted(REPO_PROMPTS.glob("task_*.md")), ids=lambda path: path.stem)
def test_prompt_template_gets_scope_and_has_no_leftovers(path):
    system = render_system(load_template(path), SCOPE)
    assert "фронтенд, front-end" in system
    assert "- Статьи: блог, /blog" in system
    assert "$" not in system


def test_user_text_is_not_treated_as_a_template():
    built = thread(text="Заплати $role и $unknown <ME>")
    assert "Заплати $role и $unknown <ME>" in render_user(built)


def test_user_prompt_marks_the_target_last():
    built = thread(10, "цель", previous=[line(9, "до")], reply_chain=[line(8, "корень")])
    text = render_user(built)
    assert text.index("корень") < text.index("до") < text.index("Оцени только") < text.index("цель")


# Run


class _FakeClient:
    def __init__(self, contents: list[str], fail_after: int | None = None) -> None:
        self.contents = iter(contents)
        self.calls = 0
        self.fail_after = fail_after

    async def show(self, model):
        return {}

    async def chat(self, model, system, user, schema, think=None):
        from task_manager.analyzer.ollama import ChatResult

        if self.fail_after is not None and self.calls >= self.fail_after:
            raise OllamaUnavailableError("HTTP 500")
        self.calls += 1
        return ChatResult(next(self.contents), "stop", 10, 5)


def _prompt(tmp_path: Path) -> Path:
    path = tmp_path / "p.md"
    path.write_text("Роль $role, разделы $sections", encoding="utf-8")
    return path


def test_run_records_answers_and_schema_errors(tmp_path):
    examples = [example(1), example(2)]
    client = _FakeClient([verdict().model_dump_json(), "не json"])
    path = asyncio.run(
        runner.run_model(client, "m", _prompt(tmp_path), SCOPE, examples, "dev", tmp_path / "r")
    )
    answers = runner.read_answers(path)
    assert answers[1].error is None
    assert answers[2].error == "schema"


def test_run_continues_after_an_outage_without_recording_it(tmp_path):
    examples = [example(1), example(2)]
    prompt = _prompt(tmp_path)
    results = tmp_path / "r"
    good = verdict().model_dump_json()

    with pytest.raises(OllamaUnavailableError):
        asyncio.run(
            runner.run_model(
                _FakeClient([good, good], fail_after=1),
                "m",
                prompt,
                SCOPE,
                examples,
                "dev",
                results,
            )
        )
    second = _FakeClient([good])
    path = asyncio.run(runner.run_model(second, "m", prompt, SCOPE, examples, "dev", results))
    assert second.calls == 1
    assert set(runner.read_answers(path)) == {1, 2}


def test_edited_prompt_does_not_reuse_answers(tmp_path):
    prompt = _prompt(tmp_path)
    results = tmp_path / "r"
    good = verdict().model_dump_json()
    first = asyncio.run(
        runner.run_model(_FakeClient([good]), "m", prompt, SCOPE, [example(1)], "dev", results)
    )
    prompt.write_text("Другой промпт $role $sections", encoding="utf-8")
    second = asyncio.run(
        runner.run_model(_FakeClient([good]), "m", prompt, SCOPE, [example(1)], "dev", results)
    )
    assert first != second


def _report_sample() -> Sample:
    return Sample(
        seed=1,
        population={"none": 2},
        skipped=0,
        examples=[
            example(1, thread=thread(text="секретная фраза из чата")),
            example(2, thread=thread(text="вторая секретная фраза")),
        ],
    )


def test_report_prints_numbers_only():
    sample = _report_sample()
    answers = {
        1: runner.Answer(
            message_id=1,
            content=verdict(description="секретное описание").model_dump_json(),
            error=None,
            seconds=1.0,
            prompt_tokens=1,
            output_tokens=1,
            truncated_input=False,
        ),
        2: runner.Answer(
            message_id=2,
            content="секретный ответ",
            error="schema",
            seconds=1.0,
            prompt_tokens=1,
            output_tokens=1,
            truncated_input=False,
        ),
    }
    rows = runner.score_rows(answers, sample, {1: 1, 2: 4})
    report = runner.format_report("title", rows, sample, 0.8)
    assert "секрет" not in report
    assert "'schema': 1" in report


def test_review_file_shows_placeholders_in_code_blocks(tmp_path):
    sample = _report_sample()
    rows = [metrics.Scored(1, 1, "none", verdict(description="Поправить шапку"))]
    path = tmp_path / "review.md"
    assert runner.write_review(path, rows, sample, 0.5) == 1
    content = path.read_text("utf-8")
    assert "Поправить шапку" in content
    assert "```text" in content


def test_disagreements_list_extra_drafts_and_missed_tasks(tmp_path):
    sample = _report_sample()
    rows = [
        metrics.Scored(1, 4, "none", verdict()),
        metrics.Scored(2, 1, "none", None, error="schema"),
    ]
    path = tmp_path / "errors.md"
    assert runner.write_disagreements(path, rows, sample, 0.5) == (1, 1)
    content = path.read_text("utf-8")
    assert "(1) ваша метка: не задача" in content
    assert "Ошибка ответа: schema" in content


def test_answer_text_stays_out_of_logs(tmp_path):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter("test"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        client = _FakeClient(["секретный ответ"])
        asyncio.run(
            runner.run_model(client, "m", _prompt(tmp_path), SCOPE, [example(1)], "dev", tmp_path)
        )
    finally:
        root.removeHandler(handler)
    assert "секретный" not in stream.getvalue()
