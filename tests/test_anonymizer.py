"""Leak tests. All personal data here is made up: real data must never enter this file."""

import asyncio
import io
import logging
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from task_manager.anonymizer import runner, store
from task_manager.anonymizer.engine import (
    Anonymizer,
    KnownPerson,
    NerModel,
    looks_like_name,
    luhn_valid,
)
from task_manager.anonymizer.preview import write_preview
from task_manager.collector import store as collector_store
from task_manager.collector.types import IncomingMessage
from task_manager.common.config import (
    AnonymizerConfig,
    ConfigError,
    MeConfig,
    ScopeConfig,
    Section,
    load_anonymizer_config,
)
from task_manager.common.log import SafeJsonFormatter
from task_manager.storage.db import make_engine
from task_manager.storage.models import (
    AnonymizedMessage,
    Attachment,
    Chat,
    Message,
    MessageStatus,
    Pseudonym,
)

ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"
REPO_CONFIG = Path(__file__).parent.parent / "config"

ALLOWED = [
    "Telegram",
    "Jira",
    "фронтенд",
    "front-end",
    "SEO",
    "ЛК",
    "статьи",
    "блог",
    "/blog",
    "/site/shop",
    "/shop",
    "личный кабинет",
    "работа",
]
PEOPLE = [
    KnownPerson(1, "Ольга", "Смирнова", "olga_s"),
    KnownPerson(2, "Вася", None, None),
    KnownPerson(3, "Дмитрий", "Козлов", "dkozlov"),
    KnownPerson(4, "Анна", "Первая"),
    KnownPerson(5, "Анна", "Вторая"),
    KnownPerson(99, "Тест", "Владелец", "test_owner", is_me=True),
]


@pytest.fixture(scope="module")
def ner() -> NerModel:
    return NerModel()


@pytest.fixture
def anonymizer(ner: NerModel) -> Anonymizer:
    return Anonymizer(ner, ["Глеб"], ALLOWED, ["оленька"], PEOPLE)


def masked(anonymizer: Anonymizer, text: str) -> str:
    """The text with each span replaced by its kind, without the database."""
    result = []
    position = 0
    for span in anonymizer.find_spans(text):
        result += [text[position : span.start], f"<{span.kind}>"]
        position = span.end
    result.append(text[position:])
    return "".join(result)


@pytest.mark.parametrize(
    ("text", "leaks"),
    [
        # Unknown people, all case forms
        ("Иван Петров, посмотри задачу", ["Иван", "Петров"]),
        ("Передай Ивану Петрову, что макет готов", ["Иван", "Петров"]),
        ("Спросите у Ивана Петрова", ["Иван", "Петров"]),
        ("С Иваном Петровым согласовали", ["Иван", "Петров"]),
        ("Об Иване Петрове речь", ["Иван", "Петров"]),
        ("Спросите у Мадины Алиевой", ["Мадин", "Алиев"]),
        ("Сергей Иваныч просил поправить", ["Сергей", "Иваныч"]),
        ("Катя, Женя и Саша завтра на созвоне", ["Катя", "Женя", "Саша"]),
        ("Передай Кате и Диме", ["Кате", "Диме"]),
        ("Позови Диму", ["Диму"]),
        ("Настя просила Лену помочь", ["Настя", "Лену"]),
        # Unknown people, lowercase
        ("катя, глянь плиз", ["катя"]),
        ("саша сказал что готово", ["саша"]),
        ("коля в отпуске", ["коля"]),
        ("настя просила", ["настя"]),
        # Known people, any spelling
        ("ольга, глянь плиз баннер", ["ольга"]),
        ("передай ольге что готово", ["ольге"]),
        ("ОЛЬГА, срочно", ["ОЛЬГА"]),
        ("вася сказал что сломано", ["вася"]),
        ("смирнова в курсе", ["смирнова"]),
        ("спроси у смирновой", ["смирновой"]),
        ("козлову передай", ["козлову"]),
        ("Оленька просила", ["Оленька"]),
        ("оленька просила", ["оленька"]),
        # Usernames
        ("@ivan_petrov сделай ревью", ["ivan_petrov"]),
        ("@olga_s глянь", ["olga_s"]),
        # Contacts
        ("Звони +7 (900) 123-45-67", ["900", "123-45-67"]),
        ("Звони 89001234567", ["89001234567"]),
        ("Звони 8 900 123 45 67", ["900 123 45 67"]),
        ("Номер +79001234567", ["79001234567"]),
        ("Пиши на +380 44 123 45 67", ["380", "123 45 67"]),
        ("Пиши на ivan.petrov+work@example.com", ["ivan.petrov", "example.com"]),
        ("Почта test_1@почта.рф", ["test_1"]),
        # Links
        ("Ссылка https://secret.example.com/path?token=abc", ["secret.example", "abc"]),
        ("Смотри example.org/private", ["example.org", "private"]),
        ("Вход https://user:hunter2@example.com/admin", ["hunter2", "user", "example"]),
        ("Канал t.me/some_private_chat", ["some_private_chat"]),
        # Payment and documents
        ("Карта 4111 1111 1111 1111", ["4111"]),
        ("Карта 5500-0000-0000-0004", ["5500"]),
        ("ИНН 7707083893", ["7707083893"]),
        ("ИНН 500100732259", ["500100732259"]),
        ("Счёт DE89370400440532013000", ["DE89370400440532013000"]),
        ("Паспорт 4510 123456", ["4510", "123456"]),
        ("СНИЛС 112-233-445 95", ["112-233-445"]),
        # Secrets
        ("пароль: hunter2", ["hunter2"]),
        ("password=correct-horse", ["correct-horse"]),
        ("api_key: abcd1234", ["abcd1234"]),
        ("токен ghp_abcdefghijklmnopqrstuvwxyz0123", ["ghp_"]),
        ("бот 123456789:AAEhBOweik6ad9r_QXMENQjcrGbqqXhDqJk", ["AAEhBO"]),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl", ["eyJ"]),
        ("ключ ATATT3xFfGF0abcdefghijkl", ["ATATT"]),
        ("длинный a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8", ["a1b2c3d4"]),
        ("curl -s -u fakeuser:fakepass https://a.example.com", ["fakeuser", "fakepass"]),
        ("curl --user fakeuser:fakepass url", ["fakeuser", "fakepass"]),
        ("-u fakeuser:fakepass", ["fakeuser", "fakepass"]),
        ("логин: fakeuser / fakepass", ["fakeuser", "fakepass"]),
        ("логин/пароль: fakeuser / fakepass", ["fakeuser", "fakepass"]),
        ("login: fakeuser", ["fakeuser"]),
        ("Логин=fakeuser пароль=fakepass", ["fakeuser", "fakepass"]),
        # Network
        ("Сервер 192.168.1.15 упал", ["192.168.1.15"]),
        ("Адрес 2001:0db8:85a3:0000:0000:8a2e:0370:7334", ["2001:0db8"]),
        # Organizations and places
        ("Спросите в ООО Ромашка", ["Ромашка"]),
        ("Деплой в Москве упал", ["Москв"]),
    ],
)
def test_personal_data_does_not_survive(
    anonymizer: Anonymizer, text: str, leaks: list[str]
) -> None:
    result = masked(anonymizer, text)
    for leak in leaks:
        assert leak.lower() not in result.lower(), result


@pytest.mark.parametrize(
    "text",
    [
        "Глеб, посмотри",
        "глеб посмотри",
        "ГЛЕБ посмотри",
        "Передай Глебу",
        "передай глебу",
        "Спроси у Глеба",
        "С Глебом обсудили",
        "О Глебе говорили",
        "@test_owner глянь",
    ],
)
def test_the_user_becomes_me(anonymizer: Anonymizer, text: str) -> None:
    result = masked(anonymizer, text)
    assert "<ME>" in result, result
    assert "глеб" not in result.lower(), result
    assert "<PERSON>" not in result, result


@pytest.mark.parametrize("word", ["мы", "Мы", "нас", "Нас", "нам", "нами"])
def test_collective_pronouns_stay_readable(anonymizer: Anonymizer, word: str) -> None:
    text = f"{word} сделаем это завтра"
    assert masked(anonymizer, text) == text


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("Поправьте SEO в ЛК", ["SEO", "ЛК"]),
        ("Посмотри в Jira и в Telegram", ["Jira", "Telegram"]),
        ("Задача для фронтенда", ["фронтенда"]),
        ("Это на front-end", ["front-end"]),
        ("Раздел личного кабинета сломан", ["личного кабинета"]),
        ("Раздел Статьи и блоги", ["Статьи", "блоги"]),
        ("Страница /blog не открывается", ["/blog"]),
        ("Страница /site/shop", ["/site/shop"]),
        ("Отчёты и работа", ["работа"]),
    ],
)
def test_allowed_terms_stay_readable(anonymizer: Anonymizer, text: str, kept: list[str]) -> None:
    result = masked(anonymizer, text)
    for term in kept:
        assert term in result, result


@pytest.mark.parametrize(
    ("url", "path"),
    [
        ("https://site.example.com/blog", "/blog"),
        ("https://site.example.com/blog/post-1?x=2", "/blog"),
        ("https://site.example.com/site/shop/list", "/site/shop"),
        ("site.example.com/shop", "/shop"),
    ],
)
def test_allowed_path_survives_inside_a_masked_link(
    anonymizer: Anonymizer, url: str, path: str
) -> None:
    result = masked(anonymizer, f"Сломано тут {url} срочно")
    assert path in result, result
    assert "site.example" not in result, result
    assert "<URL>" in result, result
    assert "post-1" not in result, result


@pytest.mark.parametrize(
    ("text", "leaks"),
    [
        # An allowed word inside a value must not stop the whole value from being masked.
        ("Смотри https://private.example.com/работа/123", ["private.example", "123"]),
        ("Смотри https://private.example.com/статьи?id=5", ["private.example", "id=5"]),
        ("Смотри www.private.example.com/блог/secret", ["private.example", "secret"]),
        ("Смотри private.example.com/работа/secret", ["private.example", "secret"]),
        ("Смотри https://jira.private.example.com/browse/X-1", ["private.example", "X-1"]),
        ("Пиши на блог@private.example.com", ["private.example"]),
        ("Пиши на jira@private.example.com", ["private.example"]),
        ("Пароль: работа123", ["работа123"]),
        ("пароль: блог", ["блог"]),
    ],
)
def test_allowed_words_do_not_protect_values_around_them(
    anonymizer: Anonymizer, text: str, leaks: list[str]
) -> None:
    result = masked(anonymizer, text)
    for leak in leaks:
        assert leak.lower() not in result.lower(), result


@pytest.mark.parametrize(
    "text",
    [
        "Сделай это завтра",
        "Посмотри макет и поправь баннер",
        "Спасибо, сделаю завтра",
        "Любой вариант подойдёт",
        "Большой баннер на главной",
        "Новая задача",
        "Давай созвонимся в понедельник",
        "до 5 марта",
        "до 5 Марта",
        "Хорошая работа",
        "Написал роман и потерял веру",
        "Версия 1.2.3 готова",
        "Номер задачи 12345",
        "В 2026 году",
    ],
)
def test_ordinary_text_stays_readable(anonymizer: Anonymizer, text: str) -> None:
    assert masked(anonymizer, text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Пререлиз выкатили вчера",
        "ПРЕРЕЛИЗ выкатили вчера",
        "Ньюанс в том, что сломано",
        "Таймауты растут",
        "Подробнее в задаче",
        "Походу сломалось",
        "Прямой эфир отменили",
        "Новой версии нет",
        "Кросевое решение не подходит",
        "Посмотри версию на сервере",
        "картинки вставлены в макет",
        "паддинг поехал",
        "скроллбары появились",
        "это тригерит ошибку",
        "собери билдами отдельно",
        "тут варн в консоли",
        "блэт опять упало",
        "дизе поправили",
        "он отьехад надолго",
        "они зашорились совсем",
        "нову версию выкатили",
        "напиши аНгЛиЙсКиЙ текст",
        "нужен мини-попап на главной",
        "Мини-попап не закрывается",
        "тут Мини-попап не закрывается",
        "в разделе GALLERY пусто",
        "Открой через Zeplin",
        "тут — Пропадает картинка",
    ],
)
def test_chat_slang_stays_readable(anonymizer: Anonymizer, text: str) -> None:
    assert masked(anonymizer, text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Баги есть",
        "Эм, не знаю",
        "Перевод на EN и RU",
        "Упало в БД",
        "Brief готов",
        "Открой в Хроме",
        "Написал в Телеграм",
        "Скорее всего сломано",
        "Иду на созвон",
        "выбери пол в анкете",
        "Пол в анкете",
        "без указания пола",
        "Запрос через POSTman",
        "Запрос через Postman",
        "кука протухла",
        "Кука протухла",
        "сбросьте куки",
        "это бек ломает",
        "Бэк упал",
    ],
)
def test_repo_allow_list_keeps_listed_words(ner: NerModel, text: str) -> None:
    config = load_anonymizer_config(REPO_CONFIG)
    anonymizer = Anonymizer(ner, ["Глеб"], config.allow, config.names)
    assert masked(anonymizer, text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Доступ fakeuser / fakepass для стенда",
        "Вход fakeuser:fakepass на стенд",
        "curl -k fakeuser fakepass",
        "FAKEUSER и FakePass",
    ],
)
def test_listed_secrets_are_hidden_in_any_form(ner: NerModel, text: str) -> None:
    anonymizer = Anonymizer(ner, ["Глеб"], [], secrets=["fakeuser", "fakepass"])
    result = masked(anonymizer, text)
    assert "fakeuser" not in result.lower(), result
    assert "fakepass" not in result.lower(), result
    assert "<SECRET>" in result, result


def test_listed_secret_matches_whole_words_only(ner: NerModel) -> None:
    anonymizer = Anonymizer(ner, ["Глеб"], [], secrets=["part"])
    assert masked(anonymizer, "partner и department") == "partner и department"


def test_listed_secrets_never_reach_the_map(engine: AsyncEngine, ner: NerModel) -> None:
    anonymizer = Anonymizer(ner, ["Глеб"], [], secrets=["fakeuser", "fakepass"])

    async def scenario() -> None:
        message_id = await add_message(engine, 1, "curl -u fakeuser:fakepass и fakeuser / fakepass")
        await process(engine, anonymizer)
        record = await ready(engine, message_id)
        assert "fakeuser" not in record.text
        assert "fakepass" not in record.text
        async with AsyncSession(engine) as session:
            stored = " ".join((await session.scalars(select(Pseudonym.value))).all())
        assert "fakeuser" not in stored
        assert "fakepass" not in stored
        await engine.dispose()

    asyncio.run(scenario())


def test_rare_name_is_found_once_it_is_listed_or_known(ner: NerModel) -> None:
    # Without a list, "Роман" reads as "a novel" and stays: the known limit of the safety net.
    known = Anonymizer(ner, ["Глеб"], [], [], [KnownPerson(1, "Роман", "Тестов")])
    listed = Anonymizer(ner, ["Глеб"], [], ["Роман"])
    assert "Роман" not in masked(known, "Роман дочитал книгу")
    assert "Роман" not in masked(listed, "Роман дочитал книгу")


def test_short_form_of_a_known_person_shares_the_placeholder(ner: NerModel) -> None:
    anonymizer = Anonymizer(ner, ["Глеб"], [], [], [KnownPerson(3, "Дмитрий", "Козлов", "dkozlov")])
    keys = {
        span.key
        for text in (
            "Дмитрий, глянь",
            "дима, глянь",
            "Диме передай",
            "@dkozlov глянь",
            "Козлов тут",
        )
        for span in anonymizer.find_spans(text)
    }
    assert keys == {"user:3"}


def test_same_people_group_links_a_tag_and_a_name(ner: NerModel) -> None:
    anonymizer = Anonymizer(ner, ["Глеб"], [], same_people=[["@unknown_tag", "Зоя"]])
    keys = {
        span.key
        for text in ("@unknown_tag глянь", "Зоя, глянь", "передай Зое", "зоя тут")
        for span in anonymizer.find_spans(text)
    }
    assert len(keys) == 1


def test_same_people_group_joins_a_known_person(ner: NerModel) -> None:
    anonymizer = Anonymizer(
        ner,
        ["Глеб"],
        [],
        [],
        [KnownPerson(8, "Ян", None, "known_tag")],
        [],
        [["@known_tag", "Ваня"]],
    )
    keys = {
        span.key
        for text in ("@known_tag глянь", "Ваня, глянь")
        for span in anonymizer.find_spans(text)
    }
    assert keys == {"user:8"}


def test_named_organization_has_one_placeholder_in_every_spelling(ner: NerModel) -> None:
    organizations = [("Рога и Копыта", ["Рогакопыта", "RogaKopyta", "Тестбанк"])]
    anonymizer = Anonymizer(ner, ["Глеб"], ["работа"], organizations=organizations)
    spans = [
        span
        for text in (
            "Рога и Копыта наняли",
            "в Рогакопыта работа",
            "RogaKopyta наняли",
            "из тестбанка пишут",
            "Тестбанк пишет",
        )
        for span in anonymizer.find_spans(text)
    ]
    assert len(spans) == 5
    assert {(span.kind, span.key) for span in spans} == {("ORG", "org:рога и копыта")}


def test_rule_for_likely_names() -> None:
    for word in ("Катя", "Кате", "Диму", "вася", "ольга", "Петрову", "Хабибуллин"):
        assert looks_like_name(word), word
    for word in ("Любой", "Марта", "Привет", "роман", "вера", "баннер", "Ivan"):
        assert not looks_like_name(word), word
    # The morphology guesses these are names, but they aren't in its dictionary.
    for word in ("скроллбары", "билдами", "паддинг", "Пререлиз", "Ньюанс", "варн", "тригерит"):
        assert not looks_like_name(word), word


def test_card_checksum() -> None:
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")


def test_empty_and_symbol_only_text(anonymizer: Anonymizer) -> None:
    assert anonymizer.find_spans("") == []
    assert anonymizer.find_spans("   ") == []
    assert anonymizer.find_spans("👍🔥!!!") == []


# With the database


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return make_engine(path)


async def add_message(
    engine: AsyncEngine, msg_id: int, text: str, file_name: str | None = None, **fields
) -> int:
    from datetime import UTC, datetime

    async with AsyncSession(engine, expire_on_commit=False) as session:
        if await session.get(Chat, -100) is None:
            session.add(Chat(chat_id=-100, title="Чат"))
            await session.flush()
        message = Message(
            chat_id=-100, msg_id=msg_id, sent_at=datetime.now(UTC), text=text, **fields
        )
        if file_name:
            message.attachments.append(Attachment(kind="document", file_name=file_name))
        session.add(message)
        await session.commit()
        return message.id


async def process(engine: AsyncEngine, anonymizer: Anonymizer, rules: str = "rules-1") -> set[int]:
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    skip: set[int] = set()
    await runner.process_batch(sessions, anonymizer, rules, skip)
    return skip


async def ready(engine: AsyncEngine, message_id: int, rules: str = "rules-1"):
    async with AsyncSession(engine) as session:
        return await store.get_ready(session, message_id, rules)


def test_same_person_gets_the_same_placeholder(engine: AsyncEngine, anonymizer: Anonymizer) -> None:
    async def scenario() -> None:
        ids = [
            await add_message(engine, 1, "Ольга, глянь"),
            await add_message(engine, 2, "передай ольге"),
            await add_message(engine, 3, "смирнова в курсе"),
            await add_message(engine, 4, "@olga_s привет"),
            await add_message(engine, 5, "Иван просил"),
            await add_message(engine, 6, "Передай Ивану"),
        ]
        await process(engine, anonymizer)
        texts = [(await ready(engine, message_id)).text for message_id in ids]
        placeholder = re.search(r"<PERSON_\d+>", texts[0]).group()
        for text in texts[:4]:
            assert placeholder in text, text
        ivan = re.search(r"<PERSON_\d+>", texts[4]).group()
        assert ivan != placeholder
        assert ivan in texts[5]
        await engine.dispose()

    asyncio.run(scenario())


def test_namesakes_share_a_placeholder_without_a_leak(
    engine: AsyncEngine, anonymizer: Anonymizer
) -> None:
    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Анна просила")
        await process(engine, anonymizer)
        assert "Анна" not in (await ready(engine, message_id)).text
        await engine.dispose()

    asyncio.run(scenario())


def test_restore_returns_real_values(engine: AsyncEngine, anonymizer: Anonymizer) -> None:
    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Иван Петров, пиши на ivan@example.com")
        await process(engine, anonymizer)
        record = await ready(engine, message_id)
        async with AsyncSession(engine) as session:
            restored, unknown = await store.restore(session, record.text)
        assert restored == "Иван Петров, пиши на ivan@example.com"
        assert unknown == []
        await engine.dispose()

    asyncio.run(scenario())


def test_restore_reports_invented_placeholders(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        async with AsyncSession(engine) as session:
            restored, unknown = await store.restore(session, "Позови <PERSON_777>")
        assert restored == "Позови <PERSON_777>"
        assert unknown == ["<PERSON_777>"]
        await engine.dispose()

    asyncio.run(scenario())


def test_secrets_and_cards_never_reach_the_map(engine: AsyncEngine, anonymizer: Anonymizer) -> None:
    async def scenario() -> None:
        message_id = await add_message(
            engine,
            1,
            "пароль: hunter2 карта 4111 1111 1111 1111 "
            "https://u:topsecret@example.com/a?token=querysecret ghp_abcdefghijklmn",
        )
        await process(engine, anonymizer)
        record = await ready(engine, message_id)
        assert "<SECRET>" in record.text
        assert "<CARD>" in record.text
        async with AsyncSession(engine) as session:
            values = " ".join((await session.scalars(select(Pseudonym.value))).all())
            keys = " ".join((await session.scalars(select(Pseudonym.key))).all())
        for secret in ("hunter2", "4111", "topsecret", "querysecret", "ghp_"):
            assert secret not in values
            assert secret not in keys
        await engine.dispose()

    asyncio.run(scenario())


def test_file_names_and_sender_are_anonymized(engine: AsyncEngine, anonymizer: Anonymizer) -> None:
    async def scenario() -> None:
        first = await add_message(engine, 1, "", file_name="договор_Иванова_Ольга.pdf", sender_id=1)
        second = await add_message(engine, 2, "текст", is_outgoing=True, sender_id=99)
        await process(engine, anonymizer)

        record = await ready(engine, first)
        assert len(record.file_names) == 1
        assert "Иванов" not in record.file_names[0]
        assert "Ольга" not in record.file_names[0]
        assert record.file_names[0].endswith(".pdf")
        assert re.fullmatch(r"<PERSON_\d+>", record.sender)
        assert (await ready(engine, second)).sender == "<ME>"
        await engine.dispose()

    asyncio.run(scenario())


def test_message_without_a_record_is_not_available(engine: AsyncEngine) -> None:
    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Иван Петров")
        assert await ready(engine, message_id) is None
        await engine.dispose()

    asyncio.run(scenario())


def test_stale_rules_are_not_available_and_get_rebuilt(
    engine: AsyncEngine, anonymizer: Anonymizer
) -> None:
    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Иван Петров")
        await process(engine, anonymizer, "rules-1")
        assert await ready(engine, message_id, "rules-1") is not None
        assert await ready(engine, message_id, "rules-2") is None
        await process(engine, anonymizer, "rules-2")
        assert await ready(engine, message_id, "rules-2") is not None
        await engine.dispose()

    asyncio.run(scenario())


def test_direct_change_of_source_text_is_detected(
    engine: AsyncEngine, anonymizer: Anonymizer
) -> None:
    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Иван Петров")
        await process(engine, anonymizer)
        async with AsyncSession(engine) as session:
            (await session.get(Message, message_id)).text = "Новый текст"
            await session.commit()
        assert await ready(engine, message_id) is None
        await engine.dispose()

    asyncio.run(scenario())


def test_edit_gives_a_new_anonymized_version(engine: AsyncEngine, anonymizer: Anonymizer) -> None:
    from datetime import UTC, datetime

    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Иван Петров", status=MessageStatus.PENDING)
        await process(engine, anonymizer)
        assert "Новиков" not in (await ready(engine, message_id)).text

        async with AsyncSession(engine) as session:
            await collector_store.apply_edit(
                session,
                IncomingMessage(
                    chat_id=-100, msg_id=1, sent_at=datetime.now(UTC), text="Сергей Новиков"
                ),
            )
            await session.commit()
        assert await ready(engine, message_id) is None

        await process(engine, anonymizer)
        text = (await ready(engine, message_id)).text
        assert "Сергей" not in text
        assert "Новиков" not in text
        assert "<PERSON_" in text
        await engine.dispose()

    asyncio.run(scenario())


def test_failed_message_gets_no_record_and_others_continue(
    engine: AsyncEngine, anonymizer: Anonymizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "Иван Петров, карта 4111 1111 1111 1111"
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(SafeJsonFormatter("anonymizer"))
    root = logging.getLogger()
    root.addHandler(handler)

    async def scenario() -> None:
        bad = await add_message(engine, 1, secret)
        good = await add_message(engine, 2, "Иван просил")
        original = anonymizer.find_spans

        def failing(text: str):
            if text == secret:
                raise ValueError(text)
            return original(text)

        monkeypatch.setattr(anonymizer, "find_spans", failing)
        assert await process(engine, anonymizer) == {bad}
        assert await ready(engine, bad) is None
        assert await ready(engine, good) is not None
        async with AsyncSession(engine) as session:
            assert await session.get(AnonymizedMessage, bad) is None

        monkeypatch.setattr(anonymizer, "find_spans", original)
        assert await process(engine, anonymizer) == set()
        assert await ready(engine, bad) is not None
        await engine.dispose()

    try:
        asyncio.run(scenario())
    finally:
        root.removeHandler(handler)

    logs = output.getvalue()
    assert "message_anonymize_failed" in logs
    assert "Иван" not in logs
    assert "4111" not in logs


def test_new_person_rebuilds_old_messages(engine: AsyncEngine, ner: NerModel) -> None:
    async def scenario() -> None:
        message_id = await add_message(engine, 1, "зульфия просила")
        before = Anonymizer(ner, ["Глеб"], ALLOWED)
        rules_before = store.rules_version(["Глеб"], ALLOWED, [], [])
        await process(engine, before, rules_before)
        people = [KnownPerson(7, "Зульфия")]
        after = Anonymizer(ner, ["Глеб"], ALLOWED, [], people)
        rules_after = store.rules_version(["Глеб"], ALLOWED, [], people)
        assert rules_after != rules_before

        await process(engine, after, rules_after)
        assert "зульфия" not in (await ready(engine, message_id, rules_after)).text
        await engine.dispose()

    asyncio.run(scenario())


def test_service_processes_messages_and_stops(
    engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "IDLE_INTERVAL_S", 0.01)
    heartbeat = tmp_path / "anonymizer.heartbeat"
    scope = ScopeConfig(role=["фронтенд"], sections=[Section(name="ЛК", aliases=["SEO"])])

    async def scenario() -> None:
        message_id = await add_message(engine, 1, "Иван Петров, глянь SEO")
        stop = asyncio.Event()
        task = asyncio.create_task(
            runner.run_anonymizer(
                engine,
                MeConfig(names=["Глеб"]),
                scope,
                AnonymizerConfig(allow=[], names=[]),
                heartbeat,
                stop,
            )
        )
        record = None
        for _ in range(500):
            async with AsyncSession(engine) as session:
                record = await session.get(AnonymizedMessage, message_id)
            if record is not None:
                break
            await asyncio.sleep(0.02)
        assert record is not None
        assert "Иван" not in record.text
        assert "SEO" in record.text
        assert heartbeat.exists()
        stop.set()
        await asyncio.wait_for(task, timeout=10)
        await engine.dispose()

    asyncio.run(scenario())
    assert not heartbeat.exists()


def test_preview_keeps_text_in_the_file_and_returns_only_counters(
    engine: AsyncEngine, anonymizer: Anonymizer, tmp_path: Path
) -> None:
    path = tmp_path / "preview.txt"

    async def scenario() -> dict[str, int]:
        await add_message(engine, 1, "Иван Петров просил", is_outgoing=True)
        await add_message(engine, 2, "Сделаю как Просил")
        await process(engine, anonymizer)
        counters = await write_preview(engine, path)
        await engine.dispose()
        return counters

    counters = asyncio.run(scenario())
    assert counters["messages"] == 2
    assert counters["with_replacements"] == 1
    assert counters["placeholder_PERSON"] >= 1
    assert "Иван" not in str(counters)
    content = path.read_text(encoding="utf-8")
    assert "Иван Петров просил" in content
    assert "автор: <ME>" in content


def test_markdown_preview_keeps_placeholders_inside_code_blocks(
    engine: AsyncEngine, anonymizer: Anonymizer, tmp_path: Path
) -> None:
    """Outside a code block, Markdown viewers hide <ME> as if it were an HTML tag."""
    path = tmp_path / "preview.md"

    async def scenario() -> None:
        await add_message(engine, 1, "Глеб, Иван просил ``` код <b>жирный</b>", is_outgoing=True)
        await process(engine, anonymizer)
        await write_preview(engine, path)
        await engine.dispose()

    asyncio.run(scenario())
    content = path.read_text(encoding="utf-8")
    assert "<ME>" in content
    fence = None
    for line in content.splitlines():
        if fence:
            if line == fence:
                fence = None
        elif line.startswith("```"):
            fence = line.removesuffix("text")
        else:
            assert "<" not in line.replace("`<ME>`", ""), line
    assert fence is None
    # A longer fence, so the backticks in the message can't close the block.
    assert "````text" in content


@pytest.mark.parametrize("name", ["preview.html", "preview.HTM"])
def test_preview_refuses_formats_that_hide_placeholders(
    engine: AsyncEngine, tmp_path: Path, name: str
) -> None:
    async def scenario() -> None:
        try:
            await write_preview(engine, tmp_path / name)
        finally:
            await engine.dispose()

    with pytest.raises(ConfigError, match="txt"):
        asyncio.run(scenario())
    assert not (tmp_path / name).exists()


def test_stale_map_entries_are_removed_after_a_rebuild(engine: AsyncEngine, ner: NerModel) -> None:
    """A value once stored as a name must not linger after it became a known secret."""

    async def scenario() -> None:
        message_id = await add_message(
            engine, 1, "Зоя Фейкова и Иван Петров", file_name="Анна Сидорова.pdf", sender_id=7
        )
        await process(engine, Anonymizer(ner, ["Глеб"], []), "rules-1")
        async with AsyncSession(engine) as session:
            before = (await session.scalars(select(Pseudonym.value))).all()
        assert any("Зоя" in value for value in before)

        newer = Anonymizer(ner, ["Глеб"], [], secrets=["Зоя", "Фейкова"])
        await process(engine, newer, "rules-2")
        async with AsyncSession(engine) as session:
            assert await store.prune_pseudonyms(session) >= 1
            await session.commit()
        async with AsyncSession(engine) as session:
            values = " ".join((await session.scalars(select(Pseudonym.value))).all())
            restored, unknown = await store.restore(
                session, (await ready(engine, message_id, "rules-2")).text
            )
        assert "Зоя" not in values
        assert "Фейкова" not in values
        # Everything still in use survives and can be restored.
        assert "Иван Петров" in restored
        assert unknown == []
        assert (await ready(engine, message_id, "rules-2")).file_names
        await engine.dispose()

    asyncio.run(scenario())


def test_repo_configuration_is_valid() -> None:
    config = load_anonymizer_config(REPO_CONFIG)
    assert "Telegram" in config.allow
