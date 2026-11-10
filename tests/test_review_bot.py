"""The review bot, without Telegram. Made-up data only."""

import asyncio
import io
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.analyzer.classifier import AnswerError
from task_manager.analyzer.commands import feedback_export
from task_manager.analyzer.ollama import OllamaUnavailableError
from task_manager.analyzer.runner import make_sessions
from task_manager.analyzer.verdict import DraftText, FollowUp
from task_manager.anonymizer.engine import Anonymizer, NerModel
from task_manager.anonymizer.runner import allowed_terms, current_rules, live_rules
from task_manager.anonymizer.store import prune_pseudonyms
from task_manager.common.log import SafeJsonFormatter
from task_manager.review_bot.chat_log import ChatLog
from task_manager.review_bot.controller import Controller
from task_manager.review_bot.messenger import COMMANDS, MessageGone, MessengerError
from task_manager.review_bot.outbox import Outbox
from task_manager.review_bot.render import (
    card_html,
    card_keyboard,
    extract_code,
    highlight,
    split_message,
)
from task_manager.review_bot.runner import check_owner
from task_manager.review_bot.views import load_card, markable, source_parts
from task_manager.storage.db import make_engine
from task_manager.storage.models import (
    Analysis,
    AnalysisOutcome,
    AnonymizerOverride,
    Attachment,
    Draft,
    DraftDecision,
    DraftKind,
    DraftMessage,
    DraftRole,
    DraftStatus,
    Message,
    MessageStatus,
    Person,
    Pseudonym,
)
from tests.test_analyzer import (
    ANON,
    ME,
    NOW,
    RULES,
    SCOPE,
    FakeClassifier,
    _decision,
    _drain,
    _links,
    _rows,
    _run,
    _seed,
    _status,
)

ROOT = Path(__file__).parent.parent
OWNER = 777


@pytest.fixture
def engine(tmp_path, monkeypatch) -> AsyncEngine:
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    return make_engine(path)


class FakeMessenger:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, object]] = []
        self.edited: list[tuple[int, str, object]] = []
        self.deleted: list[int] = []
        self.silent: list[bool] = []
        self.delete_error: Exception | None = None
        self.next_id = 100
        self.send_error: Exception | None = None
        self.edit_error: Exception | None = None

    async def send_file(self, path, name) -> int:
        self.next_id += 1
        self.sent.append((self.next_id, name, None))
        return self.next_id

    async def delete(self, message_id) -> None:
        if self.delete_error:
            raise self.delete_error
        self.deleted.append(message_id)

    async def send(self, text, keyboard=None, silent=False) -> int:
        if self.send_error:
            raise self.send_error
        self.silent.append(silent)
        self.next_id += 1
        self.sent.append((self.next_id, text, keyboard))
        return self.next_id

    async def edit(self, message_id, text, keyboard=None) -> None:
        if self.edit_error:
            raise self.edit_error
        self.edited.append((message_id, text, keyboard))

    @property
    def last(self) -> str:
        return self.sent[-1][1]

    def buttons(self) -> list[str]:
        keyboard = self.sent[-1][2] or []
        return [button.data for row in keyboard for button in row]


class FakeEditor:
    model = "fake"

    def __init__(self, answer=None, rewritten=None) -> None:
        self.answer = answer or DraftText(title="Новый", description="Новое описание")
        self.rewritten = rewritten or DraftText(title="Сделано вручную", description="Текст модели")
        self.calls: list[dict] = []
        self.rewrite_calls: list[list] = []

    async def rewrite(self, lines):
        self.rewrite_calls.append(list(lines))
        if isinstance(self.rewritten, Exception):
            raise self.rewritten
        return self.rewritten

    async def edit_text(self, lines, title, description, instruction):
        self.calls.append(
            {"lines": list(lines), "title": title, "description": description, "ask": instruction}
        )
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeNames:
    """Replaces one known name, the way the real anonymizer would."""

    async def anonymize(self, session, text):
        return text.replace("Катя", "<PERSON_3>")


def _bot(engine, messenger, editor=None, names=None):
    sessions = make_sessions(engine)
    log = ChatLog(sessions, messenger)
    outbox = Outbox(sessions, log, _ping_ok, 0, 60.0)

    async def rules(_session):
        return RULES

    controller = Controller(
        sessions,
        log,
        editor or FakeEditor(),
        names or FakeNames(),
        rules,
        outbox.flush,
        lambda: outbox.ollama_available,
        OWNER,
        0,
    )
    return controller, outbox


async def _ping_ok() -> None:
    return None


async def _draft(engine, title="Шапка", description="Поправить отступы", status=DraftStatus.SENT):
    """A draft already in the bot, built from one real queue message."""
    await _seed(engine, {"msg_id": 1})
    await _drain(engine, FakeClassifier({1: _decision(title=title, description=description)}))
    async with AsyncSession(engine) as session:
        await session.execute(
            update(Draft).values(status=status, bot_message_id=50, needs_refresh=False)
        )
        await session.commit()
        return await session.scalar(select(Draft.id))


async def _get(engine, draft_id) -> Draft:
    async with AsyncSession(engine) as session:
        return await session.get_one(Draft, draft_id)


async def _pseudonym(engine, kind, number, key, value) -> int:
    async with AsyncSession(engine, expire_on_commit=False) as session:
        row = Pseudonym(kind=kind, number=number, key=key, value=value)
        session.add(row)
        await session.commit()
        return row.id


# Outbox


def test_new_draft_is_sent_once_even_after_a_restart(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision(title="Шапка")}))
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        await outbox.flush()
        _, restarted = _bot(engine, messenger)
        await restarted.flush()
        return messenger, await _rows(engine, Draft)

    messenger, drafts = _run(engine, scenario)
    assert len(messenger.sent) == 1
    assert "Шапка" in messenger.last
    assert drafts[0].status is DraftStatus.SENT
    assert drafts[0].bot_message_id == messenger.sent[0][0]


def test_drafts_made_while_the_bot_was_down_all_arrive_at_start(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2, "chat_id": 2})
        await _drain(engine, FakeClassifier({1: _decision(), 2: _decision()}))
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        return messenger

    assert len(_run(engine, scenario).sent) == 2


def test_failed_sending_keeps_the_draft_waiting(engine):
    async def scenario():
        messenger = FakeMessenger()
        messenger.send_error = MessengerError("TelegramForbiddenError")
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision()}))
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        first = (await _rows(engine, Draft))[0].status
        messenger.send_error = None
        await outbox.flush()
        return first, (await _rows(engine, Draft))[0].status, len(messenger.sent)

    assert _run(engine, scenario) == (DraftStatus.NEW, DraftStatus.SENT, 1)


def test_added_message_moves_the_draft_down_silently(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision()}))
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FakeClassifier(
            {2: _decision()}, rewritten=DraftText(title="Вместе", description="О")
        )
        await _drain(engine, fake)
        await outbox.flush()
        await outbox.flush()
        return messenger

    messenger = _run(engine, scenario)
    assert len(messenger.sent) == 2
    assert messenger.silent == [False, True]
    assert messenger.deleted == [messenger.sent[0][0]]
    assert "Вместе" in messenger.last


def test_message_too_old_to_delete_stops_looking_current(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        async with AsyncSession(engine) as session:
            await session.execute(update(Draft).values(needs_refresh=True))
            await session.commit()
        messenger.delete_error = MessageGone("too old")
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        return messenger, await _get(engine, draft_id)

    messenger, draft = _run(engine, scenario)
    assert len(messenger.sent) == 1
    assert draft.bot_message_id == messenger.sent[0][0]
    assert not draft.needs_refresh
    assert messenger.edited[0][0] == 50
    assert "актуальная версия ниже" in messenger.edited[0][1]


def test_new_message_id_is_kept_even_if_the_draft_changed_meanwhile(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        async with AsyncSession(engine) as session:
            await session.execute(update(Draft).values(needs_refresh=True))
            await session.commit()
        _, outbox = _bot(engine, messenger)
        original = messenger.send

        async def send_and_change(*args, **kwargs):
            result = await original(*args, **kwargs)
            async with AsyncSession(engine) as session:
                await session.execute(
                    update(Draft).values(updated_at=NOW + timedelta(days=1), title="Новее")
                )
                await session.commit()
            return result

        messenger.send = send_and_change
        await outbox.flush()
        return await _get(engine, draft_id), messenger.sent[0][0]

    draft, sent_id = _run(engine, scenario)
    assert draft.bot_message_id == sent_id
    assert draft.needs_refresh


def test_rejecting_shows_the_result_at_the_bottom_and_clears_the_menu(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"rj:{draft_id}")
        menu = messenger.sent[-1][0]
        await controller.on_callback(OWNER, f"rjr:{draft_id}:not_task")
        return messenger, menu

    messenger, menu = _run(engine, scenario)
    assert "Отклонено" in messenger.last
    assert "Не задача" in messenger.last
    assert set(messenger.deleted) == {menu, 50}


def test_edit_moves_the_task_down_and_removes_the_old_card_and_prompts(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ed:{draft_id}")
        menu = messenger.sent[-1][0]
        await controller.on_callback(OWNER, f"edf:{draft_id}:title")
        prompt = messenger.sent[-1][0]
        await controller.on_text(OWNER, "Новый заголовок")
        return messenger, {menu, prompt}

    messenger, prompts = _run(engine, scenario)
    assert "Новый заголовок" in messenger.last
    assert prompts | {50} == set(messenger.deleted)


def test_source_lists_the_messages_of_a_draft_through_the_button(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1, "text": "поправь шапку"})
        await _drain(engine, FakeClassifier({1: _decision()}))
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, "src:1")
        return messenger.last

    text = _run(engine, scenario)
    assert "поправь шапку" in text
    assert "Сообщений нет" not in text


def test_failed_messages_are_reported_once(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1, "status": MessageStatus.FAILED})
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        await outbox.flush()
        return messenger

    messenger = _run(engine, scenario)
    assert len(messenger.sent) == 1
    assert messenger.buttons() == ["retry"]


def test_ollama_alert_waits_for_the_threshold_and_reports_recovery(engine, monkeypatch):
    monkeypatch.setattr("task_manager.review_bot.outbox.OLLAMA_CHECK_S", 0.0)
    down = [True]

    async def ping():
        if down[0]:
            raise OllamaUnavailableError("ConnectError")

    async def scenario():
        messenger = FakeMessenger()
        outbox = Outbox(make_sessions(engine), messenger, ping, 0, 0.05)
        await outbox.flush()
        before = len(messenger.sent)
        await asyncio.sleep(0.06)
        await outbox.flush()
        await outbox.flush()
        during = len(messenger.sent)
        down[0] = False
        await outbox.flush()
        await outbox.flush()
        return before, during, len(messenger.sent)

    assert _run(engine, scenario) == (0, 1, 2)


# Clearing the chat


def test_clear_empties_the_chat_and_keeps_every_status(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2, "chat_id": 2})
        await _drain(engine, FakeClassifier({1: _decision(), 2: _decision()}))
        controller, outbox = _bot(engine, messenger)
        await outbox.flush()
        await controller.on_callback(OWNER, "rjr:1:not_task")
        before = {row.id: row.status for row in await _rows(engine, Draft)}
        sent = [item[0] for item in messenger.sent]
        await controller.on_text(OWNER, "/clear", 900)
        return before, await _rows(engine, Draft), sent, messenger

    before, drafts, sent, messenger = _run(engine, scenario)
    assert {row.id: row.status for row in drafts} == before
    assert all(row.bot_message_id is None for row in drafts)
    # Its own command goes too. Earlier replaced cards had already been deleted once.
    assert 900 in messenger.deleted
    assert set(sent) <= set(messenger.deleted)
    assert "Ждут решения: 1" in messenger.last


def test_list_shows_a_task_again_after_clear(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision(title="Шапка")}))
        controller, outbox = _bot(engine, messenger)
        await outbox.flush()
        await controller.on_text(OWNER, "/clear")
        await controller.on_callback(OWNER, "open:1")
        return messenger, (await _rows(engine, Draft))[0]

    messenger, draft = _run(engine, scenario)
    assert "Шапка" in messenger.last
    assert draft.status is DraftStatus.SENT
    assert draft.bot_message_id == messenger.sent[-1][0]


def test_opening_a_task_twice_keeps_one_card(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision()}))
        controller, outbox = _bot(engine, messenger)
        await outbox.flush()
        await controller.on_callback(OWNER, "open:1")
        return messenger

    messenger = _run(engine, scenario)
    assert messenger.deleted == [messenger.sent[0][0]]


def test_clear_reports_messages_too_old_to_delete(engine):
    async def scenario():
        messenger = FakeMessenger()
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, "/status", 900)
        messenger.delete_error = MessageGone("too old")
        await controller.on_text(OWNER, "/clear")
        return messenger.last

    assert "старше 48 часов" in _run(engine, scenario)


def test_foreign_message_is_not_remembered(engine):
    async def scenario():
        messenger = FakeMessenger()
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER + 1, "привет", 900)
        await controller.on_text(OWNER, "/clear")
        return messenger.deleted

    assert 900 not in _run(engine, scenario)


@pytest.mark.parametrize("name", [name for name, _ in COMMANDS])
def test_every_command_in_the_menu_is_handled(engine, name):
    async def scenario():
        messenger = FakeMessenger()
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, f"/{name}")
        return messenger.last

    assert "Новые черновики приходят сами" not in _run(engine, scenario)


def test_menu_descriptions_fit_telegram_limits():
    for name, description in COMMANDS:
        assert name.islower() and 1 <= len(name) <= 32
        assert 1 <= len(description) <= 256


# Access


def test_foreign_user_gets_no_answer_and_changes_nothing(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER + 1, "/list")
        await controller.on_callback(OWNER + 1, f"ok:{draft_id}")
        return messenger.sent, (await _get(engine, draft_id)).status

    assert _run(engine, scenario) == ([], DraftStatus.SENT)


@pytest.mark.parametrize("user", [None, 0])
def test_unknown_user_is_not_the_owner(engine, user):
    async def scenario():
        controller, _ = _bot(engine, FakeMessenger())
        return controller.is_owner(user)

    assert not _run(engine, scenario)


def test_owner_mismatch_stops_the_service(engine):
    from task_manager.common.config import ConfigError

    async def scenario():
        async with AsyncSession(engine) as session:
            session.add(Person(user_id=OWNER, is_me=True))
            await session.commit()
        await check_owner(engine, OWNER)
        with pytest.raises(ConfigError):
            await check_owner(engine, OWNER + 1)

    _run(engine, scenario)


# Approving


def test_approving_a_draft_without_placeholders_is_immediate(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ok:{draft_id}")
        return await _get(engine, draft_id), await _rows(engine, DraftDecision), messenger

    draft, decisions, messenger = _run(engine, scenario)
    assert draft.status is DraftStatus.APPROVED
    assert [d.decision for d in decisions] == ["approved"]
    assert "ждёт отправки" in messenger.last
    assert messenger.deleted == [50]


def test_placeholders_are_shown_and_confirmed_before_approving(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine, description="Спросить у <PERSON_3>")
        await _pseudonym(engine, "PERSON", 3, "катя", "Катя")
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ok:{draft_id}")
        asked = (await _get(engine, draft_id)).status, messenger.last, messenger.buttons()
        await controller.on_callback(OWNER, f"okc:{draft_id}")
        return asked, (await _get(engine, draft_id)).status

    (status, text, buttons), final = _run(engine, scenario)
    assert status is DraftStatus.SENT
    assert "Катя" in text
    assert f"okc:{1}" in buttons
    assert final is DraftStatus.APPROVED


@pytest.mark.parametrize("mark", ["<SECRET>", "<CARD>"])
def test_irreversible_value_blocks_approving(engine, mark):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine, description=f"Пароль {mark}")
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ok:{draft_id}")
        return (await _get(engine, draft_id)).status, messenger.last

    status, text = _run(engine, scenario)
    assert status is DraftStatus.SENT
    assert "нельзя восстановить" in text


def test_old_button_changes_nothing(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"ok:{draft_id}")
        await controller.on_callback(OWNER, f"rjr:{draft_id}:not_task")
        await controller.on_callback(OWNER, f"okc:{draft_id}")
        return (await _get(engine, draft_id)).status, await _rows(engine, DraftDecision)

    status, decisions = _run(engine, scenario)
    assert status is DraftStatus.APPROVED
    assert len(decisions) == 1


def test_unknown_button_is_ignored(engine):
    async def scenario():
        messenger = FakeMessenger()
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, "nonsense:1")
        return messenger.sent

    assert _run(engine, scenario) == []


# Rejecting


def test_rejecting_needs_a_reason(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"rj:{draft_id}")
        asked = (await _get(engine, draft_id)).status, messenger.buttons()
        await controller.on_callback(OWNER, f"rjr:{draft_id}:not_mine")
        return asked, await _get(engine, draft_id), await _rows(engine, DraftDecision)

    (status, buttons), draft, decisions = _run(engine, scenario)
    assert status is DraftStatus.SENT
    assert {
        f"rjr:1:{code}" for code in ("not_task", "not_mine", "duplicate", "outdated", "other")
    } <= set(buttons)
    assert draft.status is DraftStatus.REJECTED
    assert [(d.decision, d.detail) for d in decisions] == [("rejected", "not_mine")]


def test_unknown_reason_does_not_reject(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"rjr:{draft_id}:whatever")
        return (await _get(engine, draft_id)).status

    assert _run(engine, scenario) is DraftStatus.SENT


def test_other_reason_takes_optional_anonymized_text(engine):
    async def scenario():
        messenger = FakeMessenger()
        first = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"rjr:{first}:other")
        waiting = (await _get(engine, first)).status
        await controller.on_text(OWNER, "Катя уже сделала")
        return waiting, await _get(engine, first), await _rows(engine, DraftDecision)

    waiting, draft, decisions = _run(engine, scenario)
    assert waiting is DraftStatus.SENT
    assert draft.status is DraftStatus.REJECTED
    assert decisions[0].note == "<PERSON_3> уже сделала"


def test_other_reason_can_be_skipped(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"rjr:{draft_id}:other")
        await controller.on_callback(OWNER, f"rjs:{draft_id}")
        return await _rows(engine, DraftDecision)

    decisions = _run(engine, scenario)
    assert [(d.detail, d.note) for d in decisions] == [("other", None)]


# Editing by hand


@pytest.mark.parametrize(
    ("field", "typed", "attribute", "expected"),
    [
        ("title", "Новый заголовок", "title", "Новый заголовок"),
        ("description", "Новое описание", "description", "Новое описание"),
        ("deadline", "2026-12-01", "deadline", "2026-12-01"),
        ("deadline", "-", "deadline", None),
    ],
)
def test_editing_a_field(engine, field, typed, attribute, expected):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"edf:{draft_id}:{field}")
        await controller.on_text(OWNER, typed)
        return await _get(engine, draft_id), await _rows(engine, DraftDecision)

    draft, decisions = _run(engine, scenario)
    assert getattr(draft, attribute) == expected
    assert draft.edited_by_owner
    assert [(d.decision, d.detail) for d in decisions] == [("edited", field)]


def test_priority_is_chosen_by_button(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"edf:{draft_id}:priority")
        await controller.on_callback(OWNER, f"edp:{draft_id}:high")
        return (await _get(engine, draft_id)).priority

    assert _run(engine, scenario) == "high"


def test_bad_date_keeps_waiting(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"edf:{draft_id}:deadline")
        await controller.on_text(OWNER, "завтра")
        first = (await _get(engine, draft_id)).deadline
        await controller.on_text(OWNER, "2026-12-01")
        return first, (await _get(engine, draft_id)).deadline

    assert _run(engine, scenario) == (None, "2026-12-01")


def test_cancel_stops_waiting_for_text(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"edf:{draft_id}:title")
        await controller.on_callback(OWNER, "cancel")
        await controller.on_text(OWNER, "Случайный текст")
        return (await _get(engine, draft_id)).title

    assert _run(engine, scenario) == "Шапка"


@pytest.mark.parametrize(
    ("answer", "expected"), [("yes", "Спросить <PERSON_3>"), ("no", "Спросить Катя")]
)
def test_typed_name_can_become_a_placeholder_or_stay(engine, answer, expected):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"edf:{draft_id}:description")
        await controller.on_text(OWNER, "Спросить Катя")
        before = (await _get(engine, draft_id)).description
        await controller.on_callback(OWNER, f"edn:{draft_id}:{answer}")
        return before, (await _get(engine, draft_id)).description

    before, description = _run(engine, scenario)
    assert before == "Поправить отступы"
    assert description == expected


def test_decided_draft_cannot_be_edited(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine, status=DraftStatus.APPROVED)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"edf:{draft_id}:title")
        await controller.on_text(OWNER, "Другой")
        return (await _get(engine, draft_id)).title, messenger.sent[0][1]

    title, first = _run(engine, scenario)
    assert title == "Шапка"
    assert "решение уже принято" in first


def test_html_in_typed_text_cannot_break_the_message(engine):
    async def scenario():
        draft_id = await _draft(engine, title="<b>жирно</b> & ещё")
        async with AsyncSession(engine) as session:
            card = await load_card(session, draft_id)
        return card_html(card, 0)

    text = _run(engine, scenario)
    assert "&lt;b&gt;жирно&lt;/b&gt; &amp; ещё" in text
    assert "<b>жирно</b>" not in text


# Editing through the model


def test_model_edit_changes_nothing_until_accepted(engine):
    async def scenario():
        messenger = FakeMessenger()
        editor = FakeEditor()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger, editor)
        await controller.on_callback(OWNER, f"ai:{draft_id}")
        await controller.on_text(OWNER, "короче")
        before = (await _get(engine, draft_id)).title, messenger.last, messenger.buttons()
        await controller.on_callback(OWNER, f"aia:{draft_id}")
        return before, await _get(engine, draft_id), editor, await _rows(engine, DraftDecision)

    (title, shown, buttons), draft, editor, decisions = _run(engine, scenario)
    assert title == "Шапка"
    assert "Новое описание" in shown
    assert "aia:1" in buttons
    assert (draft.title, draft.description) == ("Новый", "Новое описание")
    assert draft.edited_by_owner
    assert editor.calls[0]["title"] == "Шапка"
    assert [d.decision for d in decisions] == ["model_edited"]


def test_instruction_is_anonymized_before_the_model(engine):
    async def scenario():
        editor = FakeEditor()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, FakeMessenger(), editor)
        await controller.on_callback(OWNER, f"ai:{draft_id}")
        await controller.on_text(OWNER, "добавь, что спросить у Катя")
        return editor.calls[0]["ask"]

    assert _run(engine, scenario) == "добавь, что спросить у <PERSON_3>"


def test_model_only_gets_anonymized_messages(engine):
    async def scenario():
        editor = FakeEditor()
        await _seed(
            engine, {"msg_id": 1, "text": "Иван, поправь", "anonymized": "<PERSON_1>, поправь"}
        )
        await _drain(engine, FakeClassifier({1: _decision()}))
        controller, _ = _bot(engine, FakeMessenger(), editor)
        await controller.on_callback(OWNER, "ai:1")
        await controller.on_text(OWNER, "короче")
        return editor.calls[0]["lines"]

    lines = _run(engine, scenario)
    assert [line.text for line in lines] == ["<PERSON_1>, поправь"]


def test_second_edit_builds_on_the_first_proposal(engine):
    async def scenario():
        editor = FakeEditor()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, FakeMessenger(), editor)
        await controller.on_callback(OWNER, f"ai:{draft_id}")
        await controller.on_text(OWNER, "короче")
        await controller.on_callback(OWNER, f"ai:{draft_id}:more")
        await controller.on_text(OWNER, "ещё короче")
        await controller.on_callback(OWNER, f"aia:{draft_id}")
        return editor.calls, await _get(engine, draft_id)

    calls, draft = _run(engine, scenario)
    assert [call["title"] for call in calls] == ["Шапка", "Новый"]
    assert draft.title == "Новый"


def test_starting_a_new_model_edit_forgets_the_old_proposal(engine):
    async def scenario():
        editor = FakeEditor()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, FakeMessenger(), editor)
        await controller.on_callback(OWNER, f"ai:{draft_id}")
        await controller.on_text(OWNER, "короче")
        await controller.on_callback(OWNER, f"ai:{draft_id}")
        await controller.on_text(OWNER, "по-другому")
        return [call["title"] for call in editor.calls]

    assert _run(engine, scenario) == ["Шапка", "Шапка"]


@pytest.mark.parametrize(
    ("error", "phrase"),
    [
        (OllamaUnavailableError("ConnectError"), "недоступна"),
        (AnswerError("schema"), "понятного ответа"),
    ],
)
def test_model_problems_leave_the_draft_alone(engine, error, phrase):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger, FakeEditor(error))
        await controller.on_callback(OWNER, f"ai:{draft_id}")
        await controller.on_text(OWNER, "короче")
        return (await _get(engine, draft_id)).title, messenger.last

    title, text = _run(engine, scenario)
    assert title == "Шапка"
    assert phrase in text


# Looking


def test_card_shows_real_values_only_next_to_their_placeholders():
    text = highlight("Спросить у <PERSON_3> про <SECRET>", {"<PERSON_3>": "Катя", "<SECRET>": None})
    assert "<b>Катя</b> <code>&lt;PERSON_3&gt;</code>" in text
    assert "<b><code>&lt;SECRET&gt;</code></b>" in text


def test_me_stays_as_it_is_in_the_card():
    assert "&lt;ME&gt;" in highlight("<ME> сделает", {})


def test_source_shows_real_names_and_links(engine):
    async def scenario():
        await _seed(
            engine,
            {
                "msg_id": 1,
                "text": "Иван: <поправь> шапку",
                "anonymized": "<PERSON_1>: <поправь> шапку",
                "sender_id": 42,
                "link": "https://t.me/c/1/1",
            },
        )
        await _drain(engine, FakeClassifier({1: _decision()}))
        async with AsyncSession(engine) as session:
            # Added last: a new person changes the rules, and the message would wait for a rebuild.
            session.add(Person(user_id=42, first_name="Иван", last_name="Петров"))
            await session.commit()
            return await source_parts(session, 1, 3)

    parts = _run(engine, scenario)
    assert "Иван Петров" in parts[0]
    assert "&lt;поправь&gt;" in parts[0]
    assert 'href="https://t.me/c/1/1"' in parts[0]
    assert "15:" in parts[0]


def test_long_source_is_split_without_cutting_messages():
    parts = ["а" * 3000, "б" * 3000, "в"]
    messages = split_message(parts)
    assert len(messages) == 2
    assert messages[0] == "а" * 3000
    assert messages[1].endswith("в")


def test_code_offsets_are_counted_in_utf16():
    text = "😀 смотри: print(1)"
    offset = len("😀 смотри: ".encode("utf-16-le")) // 2
    entities = json.dumps([{"offset": offset, "length": len("print(1)")}])
    assert extract_code(text, entities) == ["print(1)"]
    assert extract_code("без кода", None) == []


def test_code_and_files_appear_on_the_card(engine):
    async def scenario():
        await _seed(
            engine,
            {
                "msg_id": 1,
                "text": "вот: a < b",
                "files": ["макет.png"],
                "code_entities": json.dumps([{"offset": 5, "length": 5}]),
            },
        )
        await _drain(engine, FakeClassifier({1: _decision()}))
        async with AsyncSession(engine) as session:
            card = await load_card(session, 1)
        return card_html(card, 0), [b.data for row in card_keyboard(card) for b in row]

    text, buttons = _run(engine, scenario)
    assert "<pre>a &lt; b</pre>" in text
    assert "Вложений: 1" in text
    assert "att:1" in buttons


def test_excluding_a_file_is_remembered(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1, "files": ["макет.png"]})
        await _drain(engine, FakeClassifier({1: _decision()}))
        controller, _ = _bot(engine, messenger)
        file_id = (await _rows(engine, Attachment))[0].id
        await controller.on_callback(OWNER, "att:1")
        await controller.on_callback(OWNER, f"atg:1:{file_id}")
        first = (await _rows(engine, Attachment))[0].excluded
        await controller.on_callback(OWNER, f"atg:1:{file_id}")
        return first, (await _rows(engine, Attachment))[0].excluded

    assert _run(engine, scenario) == (True, False)


def test_list_and_status_commands(engine):
    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, "/list")
        listed = messenger.last, messenger.buttons()
        await controller.on_text(OWNER, "/status")
        return listed, messenger.last, draft_id

    (text, buttons), status, draft_id = _run(engine, scenario)
    assert "Нерешённых черновиков: 1" in text
    assert f"open:{draft_id}" in buttons
    assert "ждут решения 1" in status


async def _rejected(engine, *msg_ids, drafted=()):
    """Messages the model turned down, and optionally some it made drafts of."""
    await _seed(engine, *({"msg_id": number} for number in sorted((*drafted, *msg_ids))))
    answers = {number: _decision(False) for number in msg_ids}
    answers |= {number: _decision() for number in drafted}
    await _drain(engine, FakeClassifier(answers))


async def _message_id(engine, msg_id) -> int:
    async with AsyncSession(engine) as session:
        return await session.scalar(select(Message.id).where(Message.msg_id == msg_id))


def test_recent_lists_turned_down_messages_newest_first(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _rejected(engine, 2, 3, drafted=(1,))
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, "/recent")
        ids = [await _message_id(engine, number) for number in (2, 3)]
        return messenger.last, messenger.buttons(), ids

    text, buttons, (older, newer) = _run(engine, scenario)
    # The drafted message has reached the owner already, so it isn't offered again.
    assert buttons == [f"mk:{newer}", f"mk:{older}"]
    assert "поправь шапку" in text


def test_recent_comes_in_pages(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _rejected(engine, *range(1, 11))
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, "/recent")
        first = messenger.buttons()
        await controller.on_callback(OWNER, "rc:8")
        return first, messenger.buttons()

    first, second = _run(engine, scenario)
    assert len(first) == 9 and first[-1] == "rc:8"
    assert len(second) == 2 and all(data.startswith("mk:") for data in second)


def test_make_task_turns_a_turned_down_message_into_a_draft(engine):
    async def scenario():
        messenger = FakeMessenger()
        editor = FakeEditor()
        await _seed(
            engine,
            {"msg_id": 1, "text": "а где макет?", "is_outgoing": True},
            {"msg_id": 2, "reply_to_msg_id": 1},
        )
        await _drain(engine, FakeClassifier({1: _decision(False), 2: _decision(False)}))
        controller, _ = _bot(engine, messenger, editor)
        message_id = await _message_id(engine, 2)
        await controller.on_callback(OWNER, f"mk:{message_id}")
        card = messenger.last
        await controller.on_callback(OWNER, f"mk:{message_id}")
        return (
            editor,
            card,
            messenger.last,
            await _rows(engine, Draft),
            await _rows(engine, DraftMessage),
            await _rows(engine, DraftDecision),
            message_id,
        )

    editor, card, again, drafts, links, decisions, message_id = _run(engine, scenario)
    # The reply chain goes along, so the text knows what the message answers.
    assert [line.msg_id for line in editor.rewrite_calls[0]] == [1, 2]
    assert len(editor.rewrite_calls) == 1
    assert len(drafts) == 1
    assert drafts[0].title == "Сделано вручную"
    assert drafts[0].status is DraftStatus.SENT
    assert [(link.message_id, link.role) for link in links] == [(message_id, DraftRole.TRIGGER)]
    assert [decision.decision for decision in decisions] == ["created_by_owner"]
    assert "Сделано вручную" in card
    assert "вы отметили сами" in card
    assert f"уже в задаче №{drafts[0].id}" in again


def test_make_task_creates_nothing_without_the_model(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _rejected(engine, 1)
        controller, _ = _bot(engine, messenger, FakeEditor(rewritten=OllamaUnavailableError()))
        await controller.on_callback(OWNER, f"mk:{await _message_id(engine, 1)}")
        return messenger.last, await _rows(engine, Draft)

    text, drafts = _run(engine, scenario)
    # Jira may get only the model's rephrasing, so a quote can't stand in for it.
    assert "Модель недоступна" in text
    assert drafts == []


def test_status_shows_the_last_day_of_analysis(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _rejected(engine, 2, 3, drafted=(1,))
        controller, _ = _bot(engine, messenger)
        await controller.on_text(OWNER, "/status")
        return messenger.last

    status = _run(engine, scenario)
    assert "Проанализировано за сутки: 3" in status
    assert "в черновик: 1" in status
    assert "отклонено моделью: 2" in status
    assert "/recent" in status


def test_retry_button_returns_failed_messages(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _seed(engine, {"msg_id": 1, "status": MessageStatus.FAILED})
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, "retry")
        return await _status(engine, 1)

    assert _run(engine, scenario) is MessageStatus.PENDING


def test_feedback_export_keeps_only_clear_decisions(engine, tmp_path):
    path = tmp_path / "labels.json"

    async def scenario():
        await _seed(engine, {"msg_id": 1}, {"msg_id": 2, "chat_id": 2}, {"msg_id": 3, "chat_id": 3})
        await _drain(
            engine,
            FakeClassifier({1: _decision(), 2: _decision(), 3: _decision()}),
            chat_ids=(1, 2, 3),
        )
        messenger = FakeMessenger()
        _, outbox = _bot(engine, messenger)
        await outbox.flush()
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, "ok:1")
        await controller.on_callback(OWNER, "rjr:2:not_task")
        await controller.on_callback(OWNER, "rjr:3:duplicate")
        return await feedback_export(engine, path)

    counters = _run(engine, scenario)
    assert counters == {"label_1": 1, "label_4": 1}
    assert sorted(json.loads(path.read_text("utf-8")).values()) == [1, 4]


# Anonymizer mistakes


async def _markable_setup(engine):
    await _seed(engine, {"msg_id": 1, "text": "Скролл сломан", "anonymized": "<PERSON_12> сломан"})
    await _drain(
        engine,
        FakeClassifier({1: _decision(title="Про <PERSON_12>", description="<PERSON_12> сломан")}),
    )
    return await _pseudonym(engine, "PERSON", 12, "скролл", "Скролл")


def test_mistaken_word_is_offered_and_marked_in_one_step(engine):
    async def scenario():
        messenger = FakeMessenger()
        pseudonym_id = await _markable_setup(engine)
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, "ph:1")
        offered = messenger.buttons()
        await controller.on_callback(OWNER, f"phx:{pseudonym_id}")
        asked = await _rows(engine, AnonymizerOverride)
        warning = messenger.last
        await controller.on_callback(OWNER, f"phy:{pseudonym_id}")
        return (
            offered,
            asked,
            warning,
            await _rows(engine, AnonymizerOverride),
            await _get(engine, 1),
        )

    offered, asked, warning, overrides, draft = _run(engine, scenario)
    assert "phx:1" in offered
    assert asked == []
    assert "читаемым" in warning
    assert [row.term for row in overrides] == ["Скролл"]
    assert draft.title == "Про Скролл"
    assert draft.description == "Скролл сломан"
    assert not draft.has_placeholders


@pytest.mark.parametrize(
    ("kind", "number", "key"),
    [("PERSON", 5, "user:42"), ("URL", 5, "url"), ("EMAIL", 5, "mail"), ("PHONE", 5, "phone")],
)
def test_real_people_and_contacts_are_not_offered(engine, kind, number, key):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "anonymized": f"<{kind}_{number}> поправь"})
        await _drain(engine, FakeClassifier({1: _decision()}))
        await _pseudonym(engine, kind, number, key, "значение")
        async with AsyncSession(engine) as session:
            return await markable(session, 1)

    assert _run(engine, scenario) == []


@pytest.mark.parametrize(
    ("kind", "key", "phrase"),
    [("PERSON", "user:42", "настоящий человек"), ("URL", "адрес", "не скрывается")],
)
def test_forged_button_cannot_mark_a_person_or_contact(engine, kind, key, phrase):
    async def scenario():
        messenger = FakeMessenger()
        row = await _pseudonym(engine, kind, 5, key, "значение")
        controller, _ = _bot(engine, messenger)
        await controller.on_callback(OWNER, f"phy:{row}")
        return await _rows(engine, AnonymizerOverride), messenger.last

    overrides, text = _run(engine, scenario)
    assert overrides == []
    assert phrase in text


def test_override_changes_the_rules_and_can_be_removed(engine):
    async def scenario():
        messenger = FakeMessenger()
        pseudonym_id = await _markable_setup(engine)
        controller, _ = _bot(engine, messenger)
        async with AsyncSession(engine) as session:
            before = await live_rules(session, ME, SCOPE, ANON)
        await controller.on_callback(OWNER, f"phy:{pseudonym_id}")
        async with AsyncSession(engine) as session:
            marked = await live_rules(session, ME, SCOPE, ANON)
        await controller.on_text(OWNER, "/overrides")
        buttons = messenger.buttons()
        await controller.on_callback(OWNER, buttons[0].replace("phx", "ovd"))
        async with AsyncSession(engine) as session:
            after = await live_rules(session, ME, SCOPE, ANON)
        return before, marked, after, await _rows(engine, AnonymizerOverride)

    before, marked, after, overrides = _run(engine, scenario)
    assert before == RULES
    assert marked != before
    assert after == before
    assert overrides == []


def test_no_overrides_keep_the_rules_version_of_existing_records():
    assert current_rules([], ME, SCOPE, ANON) == current_rules([], ME, SCOPE, ANON, [])


def test_cleanup_keeps_placeholders_drafts_still_use(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1, "anonymized": "без плейсхолдеров"})
        await _drain(engine, FakeClassifier({1: _decision(description="Спросить <PERSON_7>")}))
        await _pseudonym(engine, "PERSON", 7, "катя", "Катя")
        await _pseudonym(engine, "PERSON", 8, "никто", "Никто")
        async with AsyncSession(engine) as session:
            removed = await prune_pseudonyms(session)
            await session.commit()
        return removed, [row.number for row in await _rows(engine, Pseudonym)]

    assert _run(engine, scenario) == (1, [7])


@pytest.fixture(scope="module")
def ner() -> NerModel:
    return NerModel()


def _engine_with(ner: NerModel, overrides) -> Anonymizer:
    return Anonymizer(
        ner, ME.names, allowed_terms(SCOPE, ANON, overrides), ANON.names, [], [], [], ["Qwerty123"]
    )


def _spans_text(ner: NerModel, overrides, text: str) -> list[tuple[str, str]]:
    return [(s.kind, s.value) for s in _engine_with(ner, overrides).find_spans(text)]


def test_override_keeps_a_word_readable_in_any_form(ner):
    for text in ("Иннокентий сломан", "Спросить у Иннокентия"):
        assert not [
            span for span in _spans_text(ner, ["Иннокентий"], text) if span[1].startswith("Инн")
        ]


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("пароль Qwerty123", "Qwerty123"),
        ("почта Иннокентий@example.com", "Иннокентий@example.com"),
        ("телефон +7 999 123-45-67", "999 123-45-67"),
        ("https://Иннокентий.example.com/path?token=abc", "token=abc"),
    ],
)
def test_leak_override_never_unmasks_secrets_contacts_or_links(ner, text, secret):
    """Regexes stay ahead of allowed terms, so a marked word can't expose what surrounds it."""
    spans = _engine_with(ner, ["Иннокентий", "Qwerty123"]).find_spans(text)
    masked = text
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        masked = f"{masked[: span.start]}#{masked[span.end :]}"
    assert secret not in masked


def test_typed_placeholders_survive_the_real_anonymizer(engine, ner):
    from task_manager.review_bot.names import Names

    async def scenario():
        names = Names(ner, ME, SCOPE, ANON)
        async with AsyncSession(engine) as session:
            return await names.anonymize(session, "Оставь <PERSON_3> и <URL_1> как есть, <ME>")

    assert _run(engine, scenario) == "Оставь <PERSON_3> и <URL_1> как есть, <ME>"


# Logs


def test_logs_hold_no_message_text_or_reason_text(engine):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter("test"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    async def scenario():
        messenger = FakeMessenger()
        draft_id = await _draft(engine, title="Иннокентий", description="Иннокентий пишет")
        controller, outbox = _bot(engine, messenger)
        await outbox.flush()
        await controller.on_callback(OWNER, f"rjr:{draft_id}:other")
        await controller.on_text(OWNER, "Иннокентий передумал")
        messenger.send_error = MessengerError("TelegramNetworkError")
        await outbox.flush()
        await controller.on_text(OWNER + 1, "Иннокентий чужой")

    try:
        _run(engine, scenario)
    finally:
        root.removeHandler(handler)
    assert "Иннокентий" not in stream.getvalue()
    assert "bot_foreign_user" in stream.getvalue()


# Follow-ups in the analyzer


def _follow(needs: bool, title: str = "", description: str = "") -> FollowUp:
    return FollowUp(needs_ticket=needs, title=title, description=description)


class FollowUpClassifier(FakeClassifier):
    def __init__(self, answers=None, answer=None, **kwargs) -> None:
        super().__init__(answers, **kwargs)
        self.answer = answer or _follow(False)
        self.follow_calls: list[tuple[str, str | None]] = []

    async def follow_up(self, status, draft, rejection, context):
        self.follow_calls.append((status, rejection))
        return self.answer


async def _decided(engine, status: DraftStatus, reason=None) -> int:
    await _seed(engine, {"msg_id": 1})
    await _drain(engine, FakeClassifier({1: _decision(title="Шапка", description="Поправить")}))
    async with AsyncSession(engine) as session:
        await session.execute(update(Draft).values(status=status))
        if reason:
            session.add(
                DraftDecision(
                    draft_id=1, decision="rejected", detail=reason, decided_at=datetime.now(UTC)
                )
            )
        await session.commit()
    return 1


def test_sent_draft_is_extended_silently_without_asking_about_a_ticket(engine):
    async def scenario():
        await _decided(engine, DraftStatus.SENT)
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FollowUpClassifier({2: _decision()})
        await _drain(engine, fake)
        return fake.follow_calls, await _rows(engine, Draft)

    calls, drafts = _run(engine, scenario)
    assert calls == []
    assert len(drafts) == 1
    assert drafts[0].needs_refresh


def test_change_to_an_approved_task_opens_a_change_request(engine):
    async def scenario():
        await _decided(engine, DraftStatus.APPROVED)
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FollowUpClassifier(
            {2: _decision(False)}, answer=_follow(True, "Изменить", "Теперь и для мобильной")
        )
        await _drain(engine, fake)
        return fake.follow_calls, await _rows(engine, Draft), await _links(engine)

    calls, drafts, links = _run(engine, scenario)
    assert calls == [("approved", None)]
    original, ticket = sorted(drafts, key=lambda draft: draft.id)
    assert original.status is DraftStatus.APPROVED
    assert original.title == "Шапка"
    assert (ticket.kind, ticket.parent_draft_id) == (DraftKind.CHANGE_REQUEST, original.id)
    assert ticket.status is DraftStatus.NEW
    assert (ticket.title, ticket.description) == ("Изменить", "Теперь и для мобильной")
    assert links[2] is DraftRole.TRIGGER


def test_thanks_for_an_approved_task_opens_nothing(engine):
    async def scenario():
        await _decided(engine, DraftStatus.APPROVED)
        await _seed(engine, {"msg_id": 2, "text": "спасибо большое", "reply_to_msg_id": 1})
        fake = FollowUpClassifier({2: _decision(False)}, answer=_follow(False))
        await _drain(engine, fake)
        analyses = [row for row in await _rows(engine, Analysis) if row.detail]
        return await _rows(engine, Draft), analyses

    drafts, analyses = _run(engine, scenario)
    assert len(drafts) == 1
    assert [(row.outcome, row.detail) for row in analyses] == [
        (AnalysisOutcome.NO_DRAFT, "followup_ignored")
    ]


def test_returning_to_a_rejected_task_recreates_it_with_old_messages(engine):
    async def scenario():
        await _decided(engine, DraftStatus.REJECTED, "outdated")
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FollowUpClassifier(
            {2: _decision()},
            answer=_follow(True, "Снова", "Снова нужно"),
            rewritten=DraftText(title="Шапка снова", description="Вместе"),
        )
        await _drain(engine, fake)
        async with AsyncSession(engine) as session:
            links = (
                await session.execute(
                    select(DraftMessage.draft_id, Message.msg_id, DraftMessage.role).join(
                        Message, Message.id == DraftMessage.message_id
                    )
                )
            ).all()
        return fake, await _rows(engine, Draft), links

    fake, drafts, links = _run(engine, scenario)
    assert fake.follow_calls == [("rejected", "outdated")]
    assert [line.msg_id for line in fake.rewrite_calls[0]] == [1, 2]
    original, ticket = sorted(drafts, key=lambda draft: draft.id)
    assert original.status is DraftStatus.REJECTED
    assert (ticket.kind, ticket.parent_draft_id) == (DraftKind.TASK, original.id)
    assert ticket.title == "Шапка снова"
    assert {(row.msg_id, row.role) for row in links if row.draft_id == ticket.id} == {
        (1, DraftRole.EVIDENCE),
        (2, DraftRole.TRIGGER),
    }


def test_failed_follow_up_answer_still_opens_a_ticket_with_text(engine):
    async def scenario():
        await _decided(engine, DraftStatus.APPROVED)
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FollowUpClassifier({2: _decision(False)}, answer=_follow(True))
        await _drain(engine, fake)
        return sorted(await _rows(engine, Draft), key=lambda draft: draft.id)

    _, ticket = _run(engine, scenario)
    assert ticket.title == "Проверить: Шапка"
    assert ticket.description == "Поправить"


def test_more_messages_join_the_open_ticket_rather_than_making_another(engine):
    async def scenario():
        await _decided(engine, DraftStatus.APPROVED)
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FollowUpClassifier({2: _decision(False)}, answer=_follow(True, "И", "О"))
        await _drain(engine, fake)
        await _seed(engine, {"msg_id": 3, "reply_to_msg_id": 1})
        second = FollowUpClassifier({3: _decision(False)}, answer=_follow(True, "Лишний", "О"))
        await _drain(engine, second)
        return second.follow_calls, await _rows(engine, Draft)

    calls, drafts = _run(engine, scenario)
    assert calls == []
    assert len(drafts) == 2


def test_unrelated_message_does_not_touch_a_decided_draft(engine):
    async def scenario():
        await _decided(engine, DraftStatus.APPROVED)
        await _seed(engine, {"msg_id": 2, "text": "другое обсуждение"})
        fake = FollowUpClassifier({2: _decision(False)}, answer=_follow(True, "Т", "О"))
        await _drain(engine, fake)
        return fake.follow_calls, await _rows(engine, Draft)

    calls, drafts = _run(engine, scenario)
    assert calls == []
    assert len(drafts) == 1


def test_owner_wording_is_not_rewritten_by_a_new_message(engine):
    async def scenario():
        await _seed(engine, {"msg_id": 1})
        await _drain(engine, FakeClassifier({1: _decision(title="Шапка")}))
        async with AsyncSession(engine) as session:
            await session.execute(
                update(Draft).values(
                    status=DraftStatus.SENT, title="Моя формулировка", edited_by_owner=True
                )
            )
            await session.commit()
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FakeClassifier({2: _decision()})
        await _drain(engine, fake)
        return fake.rewrite_calls, (await _rows(engine, Draft))[0], await _links(engine)

    rewrites, draft, links = _run(engine, scenario)
    assert rewrites == []
    assert draft.title == "Моя формулировка"
    assert draft.text_is_stale
    assert set(links) == {1, 2}


def test_rejection_note_reaches_the_follow_up_prompt(engine):
    async def scenario():
        await _decided(engine, DraftStatus.REJECTED)
        async with AsyncSession(engine) as session:
            session.add(
                DraftDecision(
                    draft_id=1,
                    decision="rejected",
                    detail="other",
                    note="<PERSON_3> сделает",
                    decided_at=datetime.now(UTC),
                )
            )
            await session.commit()
        await _seed(engine, {"msg_id": 2, "reply_to_msg_id": 1})
        fake = FollowUpClassifier({2: _decision(False)})
        await _drain(engine, fake)
        return fake.follow_calls

    assert _run(engine, scenario) == [("rejected", "other: <PERSON_3> сделает")]
