"""Export to Jira against a fake server. Made-up data only."""

import ast
import asyncio
import io
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import yaml
from alembic import command
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.analyzer.runner import make_sessions
from task_manager.anonymizer.engine import Anonymizer, Span
from task_manager.collector import files as collector_files
from task_manager.collector.types import RateLimitedError
from task_manager.common.config import ConfigError, JiraConfig
from task_manager.common.log import SafeJsonFormatter
from task_manager.jira_export import content
from task_manager.jira_export.client import JiraClient
from task_manager.jira_export.exporter import Exporter, check_project, mask_secrets
from task_manager.review_bot.render import card_keyboard
from task_manager.review_bot.views import load_card
from task_manager.storage.attachments import goes_to_jira, upload_name
from task_manager.storage.db import make_engine
from task_manager.storage.models import (
    Attachment,
    AttachmentUpload,
    BotNotice,
    Chat,
    ChatKind,
    Draft,
    DraftKind,
    DraftMessage,
    DraftRole,
    DraftStatus,
    ExportAction,
    FileRequest,
    FileRequestStatus,
    Message,
    MessageStatus,
    Pseudonym,
)
from tests.test_review_bot import OWNER, FakeMessenger, _bot

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "task_manager"
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
CONFIG = JiraConfig(
    project_key="PROJ",
    issue_type="Задача",
    locked_statuses=["Тестирование", "Готово"],
    link_type="Relates",
    priorities={"medium": "Средний", "high": "Высокий"},
    auto_append_statuses=["Новая"],
)


@pytest.fixture
def engine(tmp_path, monkeypatch) -> AsyncEngine:
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    return make_engine(path)


class FakeJira:
    """Just enough of Jira, and a record of every request, in order."""

    def __init__(self) -> None:
        self.issues: dict[str, dict] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail: dict[tuple[str, str], httpx.Response] = {}
        self.links: list[dict] = []
        self.uploads: list[tuple[str, str, bytes]] = []
        self.next_number = 1

    def add_issue(self, key, status="В работе", description=None) -> None:
        self.issues[key] = {"status": status, "description": description}

    def writes(self) -> list[tuple[str, str]]:
        return [(m, p) for m, p, _ in self.calls if m != "GET" and "search" not in p]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = None
        if request.content and "attachments" not in path:
            body = json.loads(request.content)
        self.calls.append((request.method, path, body))
        forced = self.fail.get((request.method, path.rsplit("/", 1)[-1]))
        if forced is not None:
            return forced
        if path.endswith("/search/jql"):
            issues = [
                {"key": key, "fields": {"summary": issue.get("summary")}}
                for key, issue in self.issues.items()
            ]
            return httpx.Response(200, json={"issues": issues})
        if request.method == "POST" and path.endswith("/rest/api/3/issue"):
            key = f"PROJ-{self.next_number}"
            self.next_number += 1
            self.issues[key] = {"status": "Новая", **body["fields"]}
            return httpx.Response(201, json={"key": key})
        if path.endswith("/attachments"):
            key = path.split("/")[-2]
            assert request.headers["X-Atlassian-Token"] == "no-check"
            self.uploads.append((key, request.content.decode("latin-1"), request.content))
            return httpx.Response(200, json=[])
        if path.endswith("/issueLink"):
            self.links.append(body)
            return httpx.Response(201)
        if "/rest/api/3/issue/" in path:
            key = path.rsplit("/", 1)[-1]
            issue = self.issues.get(key)
            if issue is None:
                return httpx.Response(404, json={})
            if request.method == "PUT":
                issue["description"] = body["fields"]["description"]
                return httpx.Response(204)
            return httpx.Response(
                200,
                json={
                    "fields": {
                        "status": {"name": issue["status"]},
                        "description": issue.get("description"),
                    }
                },
            )
        raise AssertionError(f"unexpected request {request.method} {path}")


def _no_secrets(_text):
    return []


def _exporter(engine, jira: FakeJira, find=_no_secrets) -> Exporter:
    client = JiraClient(
        "https://x.atlassian.net", "me@example.com", "tok-en", httpx.MockTransport(jira)
    )
    return Exporter(make_sessions(engine), client, CONFIG, "acc-1", find)


def _run(engine, scenario):
    async def wrapper():
        try:
            return await scenario()
        finally:
            await engine.dispose()

    return asyncio.run(wrapper())


async def _draft(
    engine,
    title="Шапка",
    description="Поправить отступы",
    status=DraftStatus.APPROVED,
    action=ExportAction.CREATE,
    kind=DraftKind.TASK,
    parent=None,
    chat_kind=ChatKind.GROUP,
    chat_title="Фронт команда",
    text="поправь шапку",
    code=None,
    files=(),
    **fields,
) -> int:
    async with AsyncSession(engine) as session:
        chat = await session.get(Chat, 1)
        if chat is None:
            session.add(Chat(chat_id=1, kind=chat_kind, title=chat_title))
            await session.flush()
        count = len(list(await session.scalars(select(Message))))
        message = Message(
            chat_id=1,
            msg_id=count + 1,
            sent_at=NOW,
            text=text,
            status=MessageStatus.DONE,
            code_entities=code,
        )
        session.add(message)
        await session.flush()
        for kind_name, name in files:
            session.add(Attachment(message_id=message.id, kind=kind_name, file_name=name))
        draft = Draft(
            chat_id=1,
            status=status,
            kind=kind,
            parent_draft_id=parent,
            export_action=action,
            title=title,
            description=description,
            priority="medium",
            has_placeholders=False,
            model="m",
            prompt_id="p",
            last_message_at=NOW,
            created_at=NOW,
            updated_at=NOW,
            **fields,
        )
        session.add(draft)
        await session.flush()
        session.add(DraftMessage(draft_id=draft.id, message_id=message.id, role=DraftRole.TRIGGER))
        draft_id = draft.id
        await session.commit()
        return draft_id


async def _get(engine, draft_id) -> Draft:
    async with AsyncSession(engine) as session:
        return await session.get_one(Draft, draft_id)


async def _notices(engine) -> list[BotNotice]:
    async with AsyncSession(engine) as session:
        return list(await session.scalars(select(BotNotice).order_by(BotNotice.id)))


def _text_of(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False)


# Creating


def test_task_is_created_with_all_its_fields(engine):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(engine)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id)

    draft = _run(engine, scenario)
    fields = next(body for m, p, body in jira.calls if p.endswith("/issue"))["fields"]
    assert fields["project"] == {"key": "PROJ"}
    assert fields["issuetype"] == {"name": "Задача"}
    assert fields["summary"] == "Шапка"
    assert fields["assignee"] == {"accountId": "acc-1"}
    assert fields["priority"] == {"name": "Средний"}
    assert fields["description"]["type"] == "doc"
    assert draft.status is DraftStatus.EXPORTED
    assert draft.jira_key == "PROJ-1"
    assert draft.jira_link == "https://x.atlassian.net/browse/PROJ-1"


def test_placeholders_become_real_values_and_me_becomes_i(engine):
    jira = FakeJira()

    async def scenario():
        async with AsyncSession(engine) as session:
            session.add(Pseudonym(kind="PERSON", number=3, key="катя", value="Катя"))
            await session.commit()
        await _draft(engine, title="<ME> и <PERSON_3>", description="<ME> спрошу у <PERSON_3>")
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    fields = jira.issues["PROJ-1"]
    assert fields["summary"] == "я и Катя"
    assert "я спрошу у Катя" in _text_of(fields["description"])
    assert "PERSON" not in _text_of(fields["description"])
    assert "<ME>" not in _text_of(fields)


def test_source_links_never_reach_jira(engine):
    jira = FakeJira()

    async def scenario():
        await _draft(engine, text="поправь https://t.me/c/1/99")
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert "t.me" not in json.dumps(jira.calls, ensure_ascii=False)


def test_code_goes_as_blocks_with_secrets_masked(engine):
    jira = FakeJira()
    text = "вот: password=Qwerty123 name='Катя'"
    entities = json.dumps([{"offset": 5, "length": len(text) - 5}])

    def find(code):
        start = code.index("Qwerty123")
        return [Span(start, start + 9, "SECRET")]

    async def scenario():
        await _draft(engine, text=text, code=entities)
        await _exporter(engine, jira, find).tick(0.0)
        return await _notices(engine)

    notices = _run(engine, scenario)
    blocks = [
        n for n in jira.issues["PROJ-1"]["description"]["content"] if n["type"] == "codeBlock"
    ]
    code = blocks[0]["content"][0]["text"]
    assert "Qwerty123" not in code
    assert "<SECRET>" in code
    assert "name='Катя'" in code
    assert any("скрыты секреты" in notice.text for notice in notices)


def test_real_secret_finder_needs_no_language_model_and_keeps_names():
    engine = Anonymizer(None, ["Тестовый"], [], [], [], [], [], ["Qwerty123"])
    for code, secret in (
        ("password = 'Abc123xyz'", "Abc123xyz"),
        ("конфиг Qwerty123 тут", "Qwerty123"),
        ("карта 4111 1111 1111 1111", "4111"),
        ("curl https://example.com/a?token=abc123", "token=abc123"),
    ):
        masked, count = mask_secrets(code, engine.find_secrets)
        assert secret not in masked, code
        assert count
    kept, count = mask_secrets("const Иван = getUser('Тестовый')", engine.find_secrets)
    assert kept == "const Иван = getUser('Тестовый')"
    assert count == 0


@pytest.mark.parametrize(("by_owner", "expected"), [(True, "2026-12-01"), (False, None)])
def test_only_a_date_the_owner_set_fills_the_field(engine, by_owner, expected):
    jira = FakeJira()

    async def scenario():
        await _draft(engine, deadline="2026-12-01", deadline_by_owner=by_owner)
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.issues["PROJ-1"].get("duedate") == expected


# Nothing leaves by surprise


def test_draft_approved_before_export_stays_until_asked(engine):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(engine, action=None)
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        before = jira.writes()
        messenger = FakeMessenger()
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, "/export_old")
        offered = messenger.buttons()
        await controller.on_callback(OWNER, f"xo:{draft_id}")
        await exporter.tick(0.0)
        return before, offered, draft_id

    before, offered, draft_id = _run(engine, scenario)
    assert before == []
    assert f"xo:{draft_id}" in offered
    assert "PROJ-1" in jira.issues


@pytest.mark.parametrize("status", [DraftStatus.NEW, DraftStatus.SENT, DraftStatus.REJECTED])
def test_only_approved_drafts_are_exported(engine, status):
    jira = FakeJira()

    async def scenario():
        await _draft(engine, status=status)
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.issues == {}


# No duplicates


def test_retry_does_not_take_a_task_with_another_title_for_its_own(engine):
    jira = FakeJira()
    jira.issues["OTHER-1"] = {"status": "Новая", "summary": "Шапка сайта"}

    async def scenario():
        draft_id = await _draft(engine, export_started_at=datetime.now(UTC) - timedelta(hours=1))
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id)

    assert _run(engine, scenario).jira_key == "PROJ-1"


def test_task_created_before_a_crash_is_found_rather_than_created_again(engine):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(engine, export_started_at=NOW)
        jira.issues["PROJ-9"] = {"status": "Новая", "summary": "Шапка"}
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id)

    draft = _run(engine, scenario)
    assert draft.jira_key == "PROJ-9"
    assert not [p for m, p in jira.writes() if p.endswith("/issue")]


def test_recent_attempt_waits_for_search_to_catch_up(engine):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(engine, export_started_at=datetime.now(UTC))
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id)

    draft = _run(engine, scenario)
    assert draft.status is DraftStatus.APPROVED
    assert jira.issues == {}


def test_old_attempt_not_found_is_created(engine):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(engine, export_started_at=datetime.now(UTC) - timedelta(hours=1))
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id)

    assert _run(engine, scenario).jira_key == "PROJ-1"


def test_exported_task_is_not_sent_again(engine):
    jira = FakeJira()

    async def scenario():
        await _draft(engine)
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)

    _run(engine, scenario)
    assert len(jira.issues) == 1


# Change requests: the owner's choice, and locked statuses


async def _request(engine, parent_key="PROJ-5", parent_status="В работе", locked=False, **fields):
    parent = await _draft(
        engine,
        status=DraftStatus.EXPORTED,
        jira_key=parent_key,
        jira_link=f"https://x/browse/{parent_key}",
        jira_status=parent_status,
        jira_locked=locked,
        files_done=True,
    )
    request = await _draft(
        engine,
        title="Изменить",
        description="Ещё и для мобильной",
        kind=DraftKind.CHANGE_REQUEST,
        parent=parent,
        **fields,
    )
    return parent, request


def test_appending_keeps_what_is_in_jira(engine):
    jira = FakeJira()
    original = {"type": "doc", "version": 1, "content": [content.text_nodes("Правка в Jira")[0]]}
    jira.add_issue("PROJ-5", "В работе", original)

    async def scenario():
        _, request = await _request(engine, action=ExportAction.APPEND)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request)

    draft = _run(engine, scenario)
    text = _text_of(jira.issues["PROJ-5"]["description"])
    assert "Правка в Jira" in text
    assert "Ещё и для мобильной" in text
    assert "Изменение от 04.11" in text
    assert draft.status is DraftStatus.EXPORTED
    assert draft.jira_key == "PROJ-5"
    assert not [p for m, p in jira.writes() if p.endswith("/issue")]


@pytest.mark.parametrize("status", ["Новая", "В плане", "В работе", "На доработке", "На проверке"])
def test_every_open_status_can_be_extended_on_request(engine, status):
    jira = FakeJira()
    jira.add_issue("PROJ-5", status)

    async def scenario():
        _, request = await _request(engine, parent_status=status, action=ExportAction.APPEND)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request)

    assert _run(engine, scenario).status is DraftStatus.EXPORTED
    assert ("PUT", "/rest/api/3/issue/PROJ-5") in jira.writes()


@pytest.mark.parametrize("status", ["Тестирование", "Готово", "готово", "ТЕСТИРОВАНИЕ"])
def test_locked_task_is_never_written_even_if_it_moved_after_the_press(engine, status):
    jira = FakeJira()
    jira.add_issue("PROJ-5", status, {"type": "doc", "version": 1, "content": []})
    before = json.dumps(jira.issues["PROJ-5"], ensure_ascii=False)

    async def scenario():
        parent, request = await _request(engine, action=ExportAction.APPEND)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request), await _get(engine, parent), await _notices(engine)

    draft, parent, notices = _run(engine, scenario)
    assert jira.writes() == []
    assert json.dumps(jira.issues["PROJ-5"], ensure_ascii=False) == before
    assert draft.status is DraftStatus.SENT
    assert draft.export_action is None
    assert parent.jira_locked
    assert any("нельзя менять" in notice.text for notice in notices)


def test_task_deleted_in_jira_comes_back_to_the_owner(engine):
    jira = FakeJira()

    async def scenario():
        _, request = await _request(engine, action=ExportAction.APPEND)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request), await _notices(engine)

    draft, notices = _run(engine, scenario)
    assert draft.status is DraftStatus.SENT
    assert any("удалена" in notice.text for notice in notices)


def test_retry_after_a_crash_does_not_add_the_same_change_twice(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "В работе")

    async def scenario():
        _, request = await _request(engine, action=ExportAction.APPEND)
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Draft).where(Draft.id == request).values(status=DraftStatus.APPROVED)
            )
            await session.commit()
        await exporter.tick(1.0)

    _run(engine, scenario)
    assert _text_of(jira.issues["PROJ-5"]["description"]).count("Ещё и для мобильной") == 1


def test_only_one_place_can_write_to_an_existing_task():
    """The status check lives there, so the ban can't be bypassed by another code path."""
    callers = []
    for path in (SRC).rglob("*.py"):
        if path.name == "client.py":
            continue
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "update_description":
                callers.append(path.name)
    assert callers == ["exporter.py"]
    source = (SRC / "jira_export" / "exporter.py").read_text("utf-8")
    tree = ast.parse(source)
    holder = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and any(
            isinstance(inner, ast.Attribute) and inner.attr == "update_description"
            for inner in ast.walk(node)
        )
    ]
    assert holder == ["_write_to_existing"]


def test_new_task_from_a_request_links_to_an_open_original(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "В работе")

    async def scenario():
        await _request(engine, action=ExportAction.CREATE)
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.links == [
        {
            "type": {"name": "Relates"},
            "inwardIssue": {"key": "PROJ-1"},
            "outwardIssue": {"key": "PROJ-5"},
        }
    ]


@pytest.mark.parametrize("status", ["Тестирование", "Готово"])
def test_locked_original_gets_no_link_and_is_only_mentioned(engine, status):
    """A link shows up on the original too, and a locked task must not change at all."""
    jira = FakeJira()
    jira.add_issue("PROJ-5", status)
    before = json.dumps(jira.issues["PROJ-5"], ensure_ascii=False)

    async def scenario():
        await _request(engine, parent_status=status, locked=True, action=ExportAction.CREATE)
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.links == []
    assert json.dumps(jira.issues["PROJ-5"], ensure_ascii=False) == before
    assert "Продолжение задачи PROJ-5" in _text_of(jira.issues["PROJ-1"]["description"])


def test_request_waits_for_its_original_to_reach_jira(engine):
    jira = FakeJira()

    async def scenario():
        parent = await _draft(engine)
        await _draft(
            engine, kind=DraftKind.CHANGE_REQUEST, parent=parent, action=ExportAction.CREATE
        )
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Draft).where(Draft.id == parent).values(export_error="HTTP 400: summary")
            )
            await session.commit()
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.issues == {}


# The owner's buttons


def _buttons(card) -> list[str]:
    return [button.data for row in card_keyboard(card) for button in row]


@pytest.mark.parametrize(
    ("status", "locked", "has_append"),
    [("Новая", False, True), ("В работе", False, True), ("Готово", True, False)],
)
def test_request_card_offers_what_is_allowed(engine, status, locked, has_append):
    async def scenario():
        _, request = await _request(
            engine, parent_status=status, locked=locked, status=DraftStatus.SENT, action=None
        )
        async with AsyncSession(engine) as session:
            return await load_card(session, request), request

    card, request = _run(engine, scenario)
    buttons = _buttons(card)
    assert (f"ap:{request}" in buttons) is has_append
    assert f"ok:{request}" in buttons
    assert f"rj:{request}" in buttons


def test_unknown_status_offers_no_append(engine):
    async def scenario():
        _, request = await _request(
            engine, parent_status=None, status=DraftStatus.SENT, action=None
        )
        async with AsyncSession(engine) as session:
            return await load_card(session, request), request

    card, request = _run(engine, scenario)
    assert f"ap:{request}" not in _buttons(card)


def test_request_to_a_new_task_is_applied_by_itself(engine):
    jira = FakeJira()
    original = {"type": "doc", "version": 1, "content": content.text_nodes("Что было")}
    jira.add_issue("PROJ-5", "Новая", original)

    async def scenario():
        _, request = await _request(engine, status=DraftStatus.SENT, action=None)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request), await _notices(engine)

    draft, notices = _run(engine, scenario)
    text = _text_of(jira.issues["PROJ-5"]["description"])
    assert "Что было" in text
    assert "Ещё и для мобильной" in text
    assert draft.status is DraftStatus.EXPORTED
    assert draft.needs_refresh
    assert any("автоматически" in notice.text for notice in notices)


def test_request_with_a_placeholder_waits_for_the_owner(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "Новая")

    async def scenario():
        _, request = await _request(engine, status=DraftStatus.SENT, action=None)
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Draft).where(Draft.id == request).values(has_placeholders=True)
            )
            await session.commit()
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.writes() == []


def test_request_with_a_file_waits_for_the_owner(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "Новая")

    async def scenario():
        await _request(
            engine, status=DraftStatus.SENT, action=None, files=[("document", "личное.doc")]
        )
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.writes() == []


@pytest.mark.parametrize("status", ["В плане", "В работе", "На проверке", "Тестирование", "Готово"])
def test_nothing_is_applied_by_itself_outside_new(engine, status):
    jira = FakeJira()
    jira.add_issue("PROJ-5", status)

    async def scenario():
        await _request(engine, status=DraftStatus.SENT, action=None)
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.writes() == []


def test_owner_decision_is_not_applied_twice(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "Новая")

    async def scenario():
        _, request = await _request(engine, status=DraftStatus.SENT, action=None)
        controller, _ = _bot(engine, FakeMessenger())
        await controller.on_callback(OWNER, f"rjr:{request}:not_task")
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request)

    assert _run(engine, scenario).status is DraftStatus.REJECTED
    assert jira.writes() == []


def test_nothing_is_applied_without_a_press(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "В работе")

    async def scenario():
        await _request(engine, status=DraftStatus.SENT, action=None)
        await _exporter(engine, jira).tick(0.0)

    _run(engine, scenario)
    assert jira.writes() == []


def test_append_button_then_export(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "В работе")

    async def scenario():
        messenger = FakeMessenger()
        _, request = await _request(engine, status=DraftStatus.SENT, action=None)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ap:{request}")
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, request)

    draft = _run(engine, scenario)
    assert draft.export_action is ExportAction.APPEND
    assert "Ещё и для мобильной" in _text_of(jira.issues["PROJ-5"]["description"])


def test_forged_append_press_on_a_locked_task_is_refused(engine):
    async def scenario():
        messenger = FakeMessenger()
        _, request = await _request(
            engine, parent_status="Готово", locked=True, status=DraftStatus.SENT, action=None
        )
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ap:{request}")
        await controller.on_callback(OWNER, f"apc:{request}")
        return await _get(engine, request), messenger.last

    draft, text = _run(engine, scenario)
    assert draft.status is DraftStatus.SENT
    assert "нельзя" in text


def test_approving_records_create_so_export_picks_it_up(engine):
    async def scenario():
        draft_id = await _draft(engine, status=DraftStatus.SENT, action=None)
        controller, _ = _bot(engine, FakeMessenger())
        await controller.on_callback(OWNER, f"ok:{draft_id}")
        return await _get(engine, draft_id)

    assert _run(engine, scenario).export_action is ExportAction.CREATE


def test_owner_deadline_is_marked_as_the_owners(engine):
    async def scenario():
        draft_id = await _draft(engine, status=DraftStatus.SENT, action=None)
        controller, _ = _bot(engine, FakeMessenger())
        await controller.on_callback(OWNER, f"edf:{draft_id}:deadline")
        await controller.on_text(OWNER, "2026-12-01")
        return await _get(engine, draft_id)

    assert _run(engine, scenario).deadline_by_owner


def test_status_changes_reach_the_card(engine):
    jira = FakeJira()
    jira.add_issue("PROJ-5", "Готово")

    async def scenario():
        parent, request = await _request(engine, status=DraftStatus.SENT, action=None)
        await _exporter(engine, jira).tick(10_000.0)
        return await _get(engine, parent), await _get(engine, request)

    parent, request = _run(engine, scenario)
    assert parent.jira_status == "Готово"
    assert parent.jira_locked
    assert request.needs_refresh


def test_task_is_left_for_the_backlog_with_no_sprint_or_labels(engine):
    """Jira puts a task without a sprint in the backlog, and this issue type has no labels field."""
    jira = FakeJira()

    async def scenario():
        await _draft(engine)
        await _exporter(engine, jira).tick(10_000.0)

    _run(engine, scenario)
    fields = jira.issues["PROJ-1"]
    assert "labels" not in fields
    assert not [call for call in jira.calls if "/agile/" in call[1]]


# Service problems


@pytest.mark.parametrize("code", [401, 403])
def test_bad_token_gives_one_message_and_loses_nothing(engine, code):
    jira = FakeJira()
    jira.fail[("POST", "issue")] = httpx.Response(code)

    async def scenario():
        draft_id = await _draft(engine)
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        notices = await _notices(engine)
        state = (await _get(engine, draft_id)).status
        return notices, state

    notices, state = _run(engine, scenario)
    assert len([n for n in notices if "токен" in n.text]) == 1
    assert state is DraftStatus.APPROVED


def test_recovery_after_a_bad_token_is_reported(engine):
    jira = FakeJira()
    jira.fail[("POST", "issue")] = httpx.Response(401)

    async def scenario():
        draft_id = await _draft(engine)
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        jira.fail.clear()
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Draft)
                .where(Draft.id == draft_id)
                .values(export_started_at=datetime.now(UTC) - timedelta(hours=1))
            )
            await session.commit()
        await exporter.tick(1.0)
        return await _notices(engine), await _get(engine, draft_id)

    notices, draft = _run(engine, scenario)
    assert any("восстановлен" in notice.text for notice in notices)
    assert draft.jira_key == "PROJ-1"


def test_server_error_leaves_the_task_for_later(engine):
    jira = FakeJira()
    jira.fail[("POST", "issue")] = httpx.Response(503)

    async def scenario():
        draft_id = await _draft(engine)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id), await _notices(engine)

    draft, notices = _run(engine, scenario)
    assert draft.status is DraftStatus.APPROVED
    assert draft.export_error is None
    assert notices == []


def test_rate_limit_waits_as_told(engine, monkeypatch):
    jira = FakeJira()
    jira.fail[("POST", "issue")] = httpx.Response(429, headers={"Retry-After": "0.01"})

    async def scenario():
        draft_id = await _draft(engine)
        await _exporter(engine, jira).tick(0.0)
        return await _get(engine, draft_id)

    assert _run(engine, scenario).status is DraftStatus.APPROVED


def test_rejected_task_is_reported_and_retried_on_request(engine):
    jira = FakeJira()
    jira.fail[("POST", "issue")] = httpx.Response(
        400, json={"errors": {"summary": "текст задачи Секретная"}}
    )

    async def scenario():
        draft_id = await _draft(engine)
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        calls = len([p for m, p, _ in jira.calls if m == "POST" and p.endswith("/issue")])
        first = await _get(engine, draft_id)
        notices = await _notices(engine)
        jira.fail.clear()
        controller, _ = _bot(engine, FakeMessenger())
        await controller.on_callback(OWNER, f"xr:{draft_id}")
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Draft)
                .where(Draft.id == draft_id)
                .values(export_started_at=datetime.now(UTC) - timedelta(hours=1))
            )
            await session.commit()
        await exporter.tick(2.0)
        return calls, first, notices, await _get(engine, draft_id)

    calls, first, notices, after = _run(engine, scenario)
    assert calls == 1
    assert first.export_error == "HTTP 400: summary"
    assert "Секретная" not in notices[0].text
    assert after.jira_key == "PROJ-1"


def test_logs_hold_no_task_text_or_token(engine):
    jira = FakeJira()
    jira.fail[("POST", "issue")] = httpx.Response(400, json={"errors": {"summary": "x"}})
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter("test"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    async def scenario():
        await _draft(engine, title="Иннокентий", description="Иннокентий пишет")
        await _exporter(engine, jira).tick(0.0)

    try:
        _run(engine, scenario)
    finally:
        root.removeHandler(handler)
    logged = stream.getvalue()
    assert "jira_export_rejected" in logged
    assert "Иннокентий" not in logged
    assert "tok-en" not in logged


def test_project_check_catches_misspelt_statuses():
    statuses = {"Задача": ["Новая", "В работе", "Тестирование", "Готово"]}
    check_project(CONFIG, statuses)
    with pytest.raises(ConfigError, match="Готов"):
        check_project(CONFIG.model_copy(update={"locked_statuses": ["Готов"]}), statuses)
    with pytest.raises(ConfigError, match="issue type"):
        check_project(CONFIG.model_copy(update={"issue_type": "Баг"}), statuses)
    with pytest.raises(ConfigError, match="both locked and auto"):
        check_project(CONFIG.model_copy(update={"auto_append_statuses": ["Готово"]}), statuses)


def test_example_config_matches_the_settings():
    data = yaml.safe_load((ROOT / "config" / "jira.example.yaml").read_text("utf-8"))
    config = JiraConfig.model_validate(data)
    assert config.locked_statuses == ["Тестирование", "Готово"]
    assert config.auto_append_statuses == ["Новая"]


# Attachments


@pytest.mark.parametrize(
    ("kind", "excluded", "opted_in", "expected"),
    [
        ("photo", False, False, True),
        ("document", False, False, True),
        ("video", False, False, True),
        ("document", True, False, False),
        ("voice", False, True, False),
        ("audio", False, True, False),
        ("video_note", False, True, False),
        ("sticker", False, False, False),
        ("gif", False, False, False),
        ("sticker", False, True, True),
    ],
)
def test_which_files_go_to_jira(kind, excluded, opted_in, expected):
    assert goes_to_jira(kind, excluded, opted_in) is expected


def test_files_without_names_get_neutral_ones():
    assert upload_name("макет.fig", "document", 1) == "макет.fig"
    assert upload_name(None, "photo", 7) == "photo-7.jpg"


async def _ready(engine, tmp_path: Path) -> None:
    """Plays the collector: every waiting file is downloaded."""
    async with AsyncSession(engine) as session:
        for request in await session.scalars(
            select(FileRequest).where(FileRequest.status == FileRequestStatus.PENDING)
        ):
            path = tmp_path / f"request-{request.id}"
            path.write_bytes(b"data")
            request.status = FileRequestStatus.READY
            request.path = str(path)
        await session.commit()


def test_files_are_requested_uploaded_and_removed(engine, tmp_path):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(
            engine,
            files=[
                ("document", "заметки.md"),
                ("photo", None),
                ("voice", "голос.ogg"),
                ("sticker", None),
            ],
        )
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        await _ready(engine, tmp_path)
        await exporter.tick(2.0)
        await exporter.tick(3.0)
        return await _get(engine, draft_id)

    draft = _run(engine, scenario)
    bodies = [text for _, text, _ in jira.uploads]
    assert len(bodies) == 2
    assert any("заметки".encode().decode("latin-1") in body for body in bodies)
    assert not any("voice" in body or "sticker" in body for body in bodies)
    assert {key for key, _, _ in jira.uploads} == {"PROJ-1"}
    assert draft.files_done
    assert list(tmp_path.glob("request-*")) == []


def test_failed_file_keeps_the_task_names_the_file_and_can_be_retried(engine, tmp_path):
    jira = FakeJira()

    async def scenario():
        draft_id = await _draft(engine, files=[("document", "большой.zip")])
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        async with AsyncSession(engine) as session:
            await session.execute(
                update(FileRequest).values(status=FileRequestStatus.FAILED, error="message_gone")
            )
            await session.commit()
        await exporter.tick(2.0)
        first = await _get(engine, draft_id), await _notices(engine)
        controller, _ = _bot(engine, FakeMessenger())
        await controller.on_callback(OWNER, f"ur:{draft_id}")
        await exporter.tick(3.0)
        await exporter.tick(4.0)
        await _ready(engine, tmp_path)
        await exporter.tick(5.0)
        return first, jira.uploads

    (draft, notices), uploads = _run(engine, scenario)
    assert draft.jira_key == "PROJ-1"
    assert "большой.zip" in notices[-1].text
    assert len(uploads) == 1
    assert len([k for k in jira.issues if k.startswith("PROJ-")]) == 1


def test_oversized_file_rejected_by_jira_keeps_the_task(engine, tmp_path):
    jira = FakeJira()
    jira.fail[("POST", "attachments")] = httpx.Response(413)

    async def scenario():
        draft_id = await _draft(engine, files=[("video", "запись.mp4")])
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        await _ready(engine, tmp_path)
        await exporter.tick(2.0)
        async with AsyncSession(engine) as session:
            uploads = list(await session.scalars(select(AttachmentUpload)))
        return await _get(engine, draft_id), uploads, await _notices(engine)

    draft, uploads, notices = _run(engine, scenario)
    assert draft.jira_key == "PROJ-1"
    assert uploads[0].error == "jira_413"
    assert "запись.mp4" in notices[-1].text
    assert list(tmp_path.glob("request-*")) == []


def test_uploaded_file_is_not_uploaded_twice(engine, tmp_path):
    jira = FakeJira()

    async def scenario():
        await _draft(engine, files=[("document", "а.md")])
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        await _ready(engine, tmp_path)
        await exporter.tick(2.0)
        async with AsyncSession(engine) as session:
            await session.execute(update(Draft).values(files_done=False))
            await session.commit()
        await exporter.tick(3.0)

    _run(engine, scenario)
    assert len(jira.uploads) == 1


def test_excluded_file_is_not_requested(engine):
    jira = FakeJira()

    async def scenario():
        await _draft(engine, files=[("document", "личное.doc")])
        async with AsyncSession(engine) as session:
            await session.execute(update(Attachment).values(excluded=True))
            await session.commit()
        exporter = _exporter(engine, jira)
        await exporter.tick(0.0)
        await exporter.tick(1.0)
        async with AsyncSession(engine) as session:
            return list(await session.scalars(select(FileRequest)))

    assert _run(engine, scenario) == []


def test_audio_is_not_even_listed_for_the_owner(engine):
    async def scenario():
        draft_id = await _draft(
            engine,
            status=DraftStatus.SENT,
            action=None,
            files=[("voice", "г.ogg"), ("photo", None)],
        )
        async with AsyncSession(engine) as session:
            return await load_card(session, draft_id)

    assert [file.kind for file in _run(engine, scenario).files] == ["photo"]


def test_sticker_is_off_until_switched_on(engine):
    async def scenario():
        draft_id = await _draft(
            engine, status=DraftStatus.SENT, action=None, files=[("sticker", None)]
        )
        async with AsyncSession(engine) as session:
            before = (await load_card(session, draft_id)).files[0].excluded
            file_id = await session.scalar(select(Attachment.id))
        controller, _ = _bot(engine, FakeMessenger())
        await controller.on_callback(OWNER, f"atg:{draft_id}:{file_id}")
        async with AsyncSession(engine) as session:
            return before, (await load_card(session, draft_id)).files[0].excluded

    assert _run(engine, scenario) == (True, False)


# Viewing files in the bot


def test_viewing_a_file_sends_it_and_removes_it_from_disk(engine, tmp_path):
    async def scenario():
        messenger = FakeMessenger()
        sent = []

        async def send_file(path, name):
            sent.append((path.read_bytes(), name))
            return 500

        messenger.send_file = send_file
        draft_id = await _draft(
            engine, status=DraftStatus.SENT, action=None, files=[("document", "а.md")]
        )
        async with AsyncSession(engine) as session:
            file_id = await session.scalar(select(Attachment.id))
        controller, outbox = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"fv:{draft_id}:{file_id}")
        await _ready(engine, tmp_path)
        await outbox.flush()
        await outbox.flush()
        return sent

    assert _run(engine, scenario) == [(b"data", "а.md")]
    assert list(tmp_path.glob("request-*")) == []


def test_viewing_a_file_too_big_for_a_bot_explains_why(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(
            engine, status=DraftStatus.SENT, action=None, files=[("video", "запись.mp4")]
        )
        async with AsyncSession(engine) as session:
            await session.execute(update(Attachment).values(size_bytes=60 * 1024 * 1024))
            await session.commit()
            file_id = await session.scalar(select(Attachment.id))
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"fv:{draft_id}:{file_id}")
        async with AsyncSession(engine) as session:
            return messenger.last, list(await session.scalars(select(FileRequest)))

    text, requests = _run(engine, scenario)
    assert "50 МБ" in text
    assert requests == []


def test_notices_reach_the_owner_once(engine):
    async def scenario():
        messenger = FakeMessenger()
        async with AsyncSession(engine) as session:
            session.add(
                BotNotice(
                    text="Выберите",
                    keyboard=json.dumps([[["Да", "sp:1:0"]]]),
                    created_at=NOW,
                )
            )
            await session.commit()
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        await outbox.flush()
        return messenger

    messenger = _run(engine, scenario)
    assert len(messenger.sent) == 1
    assert messenger.buttons() == ["sp:1:0"]


# The collector's side


class FakeReader:
    def __init__(self, found=True, error=None) -> None:
        self.found = found
        self.error = error
        self.calls = []

    async def download(self, chat_id, msg_id, path):
        self.calls.append((chat_id, msg_id))
        if self.error:
            raise self.error
        if self.found:
            path.write_bytes(b"data")
        return self.found

    def is_connected(self):
        return True


async def _request_file(engine) -> int:
    draft_id = await _draft(engine, files=[("document", "а.md")])
    async with AsyncSession(engine) as session:
        file_id = await session.scalar(select(Attachment.id))
        request = FileRequest(
            attachment_id=file_id,
            draft_id=draft_id,
            purpose="upload",
            status=FileRequestStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        session.add(request)
        await session.flush()
        request_id = request.id
        await session.commit()
        return request_id


async def _file_request(engine, request_id) -> FileRequest:
    async with AsyncSession(engine) as session:
        return await session.get_one(FileRequest, request_id)


def test_collector_downloads_with_a_neutral_name(engine, tmp_path):
    async def scenario():
        request_id = await _request_file(engine)
        reader = FakeReader()
        await collector_files.process_requests(make_sessions(engine), reader, tmp_path)
        return request_id, await _file_request(engine, request_id), reader

    request_id, request, reader = _run(engine, scenario)
    assert request.status is FileRequestStatus.READY
    assert Path(request.path).name == f"request-{request_id}"
    assert Path(request.path).read_bytes() == b"data"
    assert reader.calls == [(1, 1)]
    assert list(tmp_path.glob("*.part")) == []


def test_deleted_message_fails_the_request(engine, tmp_path):
    async def scenario():
        request_id = await _request_file(engine)
        await collector_files.process_requests(
            make_sessions(engine), FakeReader(found=False), tmp_path
        )
        return await _file_request(engine, request_id)

    request = _run(engine, scenario)
    assert (request.status, request.error) == (FileRequestStatus.FAILED, "message_gone")


def test_failed_download_leaves_no_partial_file(engine, tmp_path):
    async def scenario():
        request_id = await _request_file(engine)
        await collector_files.process_requests(
            make_sessions(engine), FakeReader(error=OSError("disk")), tmp_path
        )
        return await _file_request(engine, request_id)

    assert _run(engine, scenario).error == "download_failed"
    assert list(tmp_path.glob("request-*")) == []


def test_rate_limit_keeps_the_request_waiting(engine, tmp_path):
    async def scenario():
        request_id = await _request_file(engine)
        with pytest.raises(RateLimitedError):
            await collector_files.process_requests(
                make_sessions(engine), FakeReader(error=RateLimitedError(5)), tmp_path
            )
        return await _file_request(engine, request_id)

    assert _run(engine, scenario).status is FileRequestStatus.PENDING


def test_forgotten_files_are_removed(tmp_path):
    old = tmp_path / "request-1"
    new = tmp_path / "request-2"
    other = tmp_path / "keep.txt"
    for path in (old, new, other):
        path.write_bytes(b"x")
    past = time.time() - 2 * collector_files.FILE_MAX_AGE_S
    os.utime(old, (past, past))
    assert collector_files.remove_forgotten_files(tmp_path) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt", "request-2"]


def test_unclaimed_ready_file_expires(engine):
    async def scenario():
        request_id = await _request_file(engine)
        async with AsyncSession(engine) as session:
            await session.execute(
                update(FileRequest).values(
                    status=FileRequestStatus.READY,
                    path="/gone",
                    created_at=datetime.now(UTC) - timedelta(hours=2),
                )
            )
            await session.commit()
        await collector_files.expire_requests(make_sessions(engine))
        return await _file_request(engine, request_id)

    request = _run(engine, scenario)
    assert (request.status, request.path) == (FileRequestStatus.FAILED, None)


# Document format


def test_empty_paragraphs_are_left_out():
    assert content.text_nodes("один\n\n\n\nдва") == [
        {"type": "paragraph", "content": [{"type": "text", "text": "один"}]},
        {"type": "paragraph", "content": [{"type": "text", "text": "два"}]},
    ]
    assert content.text_nodes("  \n ") == []
    assert content.text_nodes("а\nб")[0]["content"][1] == {"type": "hardBreak"}


def test_summary_is_one_line_within_the_limit():
    assert content.summary("а\nб") == "а б"
    assert len(content.summary("я" * 300)) == 255
