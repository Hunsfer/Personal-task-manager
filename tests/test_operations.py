"""Encryption, secrets, backups and monitoring. Made-up data only."""

import asyncio
import importlib.util
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.analyzer.commands import make_replay_copy
from task_manager.analyzer.runner import make_sessions
from task_manager.common.config import ConfigError, load_me_config, load_settings
from task_manager.monitoring import record_state
from task_manager.review_bot.outbox import Outbox
from task_manager.storage.backup import (
    create_backup,
    restore_backup,
    rotate_backups,
    verify_backup,
)
from task_manager.storage.db import connect_raw, make_engine
from task_manager.storage.models import Chat, Message, MessageStatus, ServiceState
from tests.test_review_bot import FakeMessenger

ROOT = Path(__file__).parent.parent
ALEMBIC_INI = ROOT / "alembic.ini"
KEY = "ab" * 32
OTHER_KEY = "cd" * 32
NEEDLE = "НЕВИДИМАЯ-СТРОКА-12345"
SENT = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def encrypted(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    monkeypatch.setenv("DATABASE_KEY", KEY)
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return path


async def _write_message(path: Path, msg_id: int = 1) -> None:
    engine = make_engine(path)
    try:
        async with AsyncSession(engine) as session:
            if await session.get(Chat, 1) is None:
                session.add(Chat(chat_id=1, title="Чат"))
                await session.flush()
            session.add(Message(chat_id=1, msg_id=msg_id, sent_at=SENT, text=NEEDLE))
            await session.commit()
    finally:
        await engine.dispose()


async def _count(path: Path) -> int:
    engine = make_engine(path)
    try:
        async with AsyncSession(engine) as session:
            return len(list(await session.scalars(select(Message))))
    finally:
        await engine.dispose()


def _all_bytes(path: Path) -> bytes:
    found = b""
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            found += candidate.read_bytes()
    return found


# Encryption


def test_no_plain_text_on_disk_even_while_the_wal_is_open(encrypted):
    async def scenario():
        engine = make_engine(encrypted)
        try:
            async with AsyncSession(engine) as session:
                session.add(Chat(chat_id=1, title=NEEDLE))
                await session.commit()
                session.add(Message(chat_id=1, msg_id=1, sent_at=SENT, text=NEEDLE))
                await session.commit()
                return _all_bytes(encrypted)
        finally:
            await engine.dispose()

    during = asyncio.run(scenario())
    assert NEEDLE.encode() not in during
    assert b"SQLite format 3" not in during[:16]
    assert NEEDLE.encode() not in _all_bytes(encrypted)


def test_plain_sqlite_cannot_read_it(encrypted):
    asyncio.run(_write_message(encrypted))
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(encrypted).execute("SELECT * FROM messages").fetchall()


def test_wrong_key_gives_a_plain_explanation(encrypted, monkeypatch):
    asyncio.run(_write_message(encrypted))
    monkeypatch.setenv("DATABASE_KEY", OTHER_KEY)
    with pytest.raises(ConfigError, match="this key"):
        asyncio.run(_count(encrypted))


def test_key_error_does_not_show_the_key(monkeypatch, tmp_path):
    bad = "not-a-valid-key"
    monkeypatch.setenv("DATABASE_KEY", bad)
    with pytest.raises(ConfigError) as caught:
        make_engine(tmp_path / "x.db")
    assert bad not in str(caught.value)


def test_service_refuses_to_create_an_unencrypted_database(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_KEY", raising=False)
    monkeypatch.setenv("REQUIRE_ENCRYPTED_DB", "1")
    path = tmp_path / "app.db"
    with pytest.raises(ConfigError, match="tm start"):
        make_engine(path)
    assert not path.exists()


def test_migrations_and_data_work_encrypted(encrypted):
    command.check(Config(str(ALEMBIC_INI)))
    asyncio.run(_write_message(encrypted))
    assert asyncio.run(_count(encrypted)) == 1


def test_foreign_keys_and_wal_stay_on_when_encrypted(encrypted):
    async def scenario():
        engine = make_engine(encrypted)
        try:
            async with engine.connect() as connection:
                wal = (await connection.execute(text("PRAGMA journal_mode"))).scalar_one()
                keys = (await connection.execute(text("PRAGMA foreign_keys"))).scalar_one()
                return wal, keys
        finally:
            await engine.dispose()

    assert asyncio.run(scenario()) == ("wal", 1)


def test_replay_copy_stays_encrypted(encrypted, tmp_path):
    async def scenario():
        engine = make_engine(encrypted)
        async with AsyncSession(engine) as session:
            session.add(Chat(chat_id=1))
            await session.flush()
            session.add(
                Message(
                    chat_id=1, msg_id=1, sent_at=SENT, text=NEEDLE, status=MessageStatus.ARCHIVED
                )
            )
            await session.commit()
        await engine.dispose()

    asyncio.run(scenario())
    target = tmp_path / "copy.db"
    assert make_replay_copy(encrypted, target, [1]) == 1
    assert NEEDLE.encode() not in _all_bytes(target)


# Secrets


def test_secrets_come_from_memory_when_present(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "bot_token").write_text("from-memory\n", encoding="utf-8")
    (secrets / "database_key").write_text(KEY, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    settings = load_settings()
    assert settings.bot_token.get_secret_value() == "from-memory"
    assert settings.database_key.get_secret_value() == KEY
    assert "from-memory" not in repr(settings)
    assert KEY not in repr(settings)


def test_private_config_comes_from_memory_when_present(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "me.yaml").write_text("names: [Из памяти]\n", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    (config / "me.yaml").write_text("names: [С диска]\n", encoding="utf-8")
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    assert load_me_config(config).names == ["Из памяти"]
    monkeypatch.delenv("SECRETS_DIR")
    assert load_me_config(config).names == ["С диска"]


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))


def test_no_container_gets_secrets_from_files_or_the_environment():
    for name, service in _compose()["services"].items():
        for item in service.get("env_file", []):
            assert item["path"] == ".env", name
        for variable in service.get("environment", {}):
            assert "TOKEN" not in variable and "KEY" not in variable, (name, variable)


@pytest.mark.parametrize(
    ("service", "volume"),
    [
        ("collector", "secrets_collector"),
        ("review_bot", "secrets_bot"),
        ("jira_export", "secrets_jira"),
        ("anonymizer", "secrets_common"),
        ("analyzer", "secrets_common"),
        ("migrate", "secrets_common"),
        ("backup", "secrets_common"),
    ],
)
def test_each_container_sees_only_its_own_secrets(service, volume):
    mounts = [v for v in _compose()["services"][service]["volumes"] if v.startswith("secrets_")]
    assert mounts == [f"{volume}:/secrets:ro"]


def test_secrets_are_where_they_belong():
    data = json.loads((ROOT / "scripts" / "secret-map.json").read_text("utf-8"))
    volumes = data["volumes"]
    everywhere = set(data["shared"])
    assert {"database_key", "me.yaml", "chats.yaml", "scope.yaml"} == everywhere
    assert set(volumes["collector"]) == {"tg_api_id", "tg_api_hash", "tg_session"}
    assert set(volumes["bot"]) == {"bot_token", "bot_owner_id"}
    assert "jira_api_token" in volumes["jira"]
    assert volumes["common"] == []


def test_secret_volumes_and_files_stay_in_memory():
    volumes = _compose()["volumes"]
    for name in ("secrets_common", "secrets_collector", "secrets_bot", "secrets_jira"):
        assert volumes[name]["driver_opts"]["type"] == "tmpfs", name
    assert volumes["attachments_tmp"]["driver_opts"]["type"] == "tmpfs"
    assert "tg_session" not in volumes


def test_keeper_cannot_reach_the_network_and_no_service_opens_a_port():
    services = _compose()["services"]
    assert services["keeper"]["network_mode"] == "none"
    for name, service in services.items():
        assert "ports" not in service, name


def test_logs_are_limited():
    services = _compose()["services"]
    for name in ("collector", "anonymizer", "analyzer", "review_bot", "jira_export", "keeper"):
        assert services[name]["logging"]["options"]["max-size"] == "10m", name


def test_script_never_passes_secret_values_as_arguments():
    script = (ROOT / "scripts" / "tm.ps1").read_text("utf-8")
    assert "$Text | & $Docker compose exec -T keeper sh -c" in script
    assert "umask 077; cat > /secrets/" in script
    # An empty result never matches, so a stopped bot would go unnoticed without the quotes.
    assert '"$(& $Docker compose ps review_bot' in script
    # A value returned by the wrapper would end up inside captured secrets.
    assert "return $LASTEXITCODE" not in script
    assert "-e BOT_TOKEN" not in script
    assert "AsSecureString" in script


# Backups


def test_backup_is_encrypted_and_reads_back(encrypted, tmp_path):
    asyncio.run(_write_message(encrypted))
    path = create_backup(encrypted, tmp_path / "backups")
    assert NEEDLE.encode() not in path.read_bytes()
    assert verify_backup(path).messages == 1
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(path).execute("SELECT * FROM messages").fetchall()


def test_verifying_a_backup_leaves_nothing_beside_it(encrypted, tmp_path):
    asyncio.run(_write_message(encrypted))
    path = create_backup(encrypted, tmp_path / "backups")
    verify_backup(path)
    assert [item.name for item in path.parent.iterdir()] == [path.name]


def test_backup_cannot_be_read_with_another_key(encrypted, tmp_path, monkeypatch):
    asyncio.run(_write_message(encrypted))
    path = create_backup(encrypted, tmp_path / "backups")
    monkeypatch.setenv("DATABASE_KEY", OTHER_KEY)
    with pytest.raises(Exception):  # noqa: B017
        verify_backup(path)


def test_damaged_backup_is_noticed(encrypted, tmp_path):
    asyncio.run(_write_message(encrypted))
    path = create_backup(encrypted, tmp_path / "backups")
    data = bytearray(path.read_bytes())
    data[5000:5010] = b"xxxxxxxxxx"
    path.write_bytes(bytes(data))
    with pytest.raises(Exception):  # noqa: B017
        verify_backup(path)


def test_restore_brings_the_data_back(encrypted, tmp_path):
    asyncio.run(_write_message(encrypted, 1))
    path = create_backup(encrypted, tmp_path / "backups")
    asyncio.run(_write_message(encrypted, 2))
    assert asyncio.run(_count(encrypted)) == 2
    restore_backup(path, encrypted)
    assert asyncio.run(_count(encrypted)) == 1


def test_only_recent_backups_are_kept(tmp_path):
    for day in range(1, 21):
        (tmp_path / f"app-202611{day:02d}-120000.db").write_bytes(b"x")
    (tmp_path / "other.txt").write_bytes(b"x")
    assert rotate_backups(tmp_path) == 6
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names[0] == "app-20261107-120000.db"
    assert "other.txt" in names


def test_rotation_removes_what_older_backups_left_beside_them(tmp_path):
    for day in range(1, 16):
        (tmp_path / f"app-202611{day:02d}-120000.db").write_bytes(b"x")
    (tmp_path / "app-20261101-120000.db-wal").write_bytes(b"")
    (tmp_path / "app-20261101-120000.db-shm").write_bytes(b"x")
    assert rotate_backups(tmp_path) == 1
    assert not list(tmp_path.glob("*-wal")) + list(tmp_path.glob("*-shm"))


def test_watchdog_brings_the_system_back_but_gives_up_after_three_tries():
    script = (ROOT / "scripts" / "tm.ps1").read_text("utf-8")
    assert "$MaxRecoveries = 3" in script
    watchdog = script[script.index("function Invoke-Watchdog") :]
    # Recovery comes first, and a deliberate stop still holds.
    assert (
        watchdog.index("StoppedMark")
        < watchdog.index("Invoke-Recovery")
        < watchdog.index("Get-Problems")
    )
    recovery = script[
        script.index("function Invoke-Recovery") : script.index("function Get-Problems")
    ]
    assert "gave_up" in recovery
    assert "Test-StartRunning" in recovery
    assert "Больше" in recovery and "tm start" in recovery


def test_start_takes_a_lock_and_restarts_services_stuck_in_docker_backoff():
    script = (ROOT / "scripts" / "tm.ps1").read_text("utf-8")
    start = script[
        script.index("function Invoke-Start") : script.index("function Test-SystemHealthy")
    ]
    assert "Enter-StartLock" in start
    assert start.index("Push-Secrets") < start.index("Compose restart") < start.index("--wait")


def test_no_machine_paths_in_the_repository():
    for name in ("scripts/tm.ps1", "docker-compose.yml"):
        text = (ROOT / name).read_text("utf-8")
        assert not re.search(r"(?<![A-Za-z])[D-Z]:[/\\\\]", text), name


def test_scheduled_runs_never_show_a_console():
    script = (ROOT / "scripts" / "tm.ps1").read_text("utf-8")
    assert "--headless" in script
    assert "-WindowStyle Hidden" not in script


def test_compose_runs_the_light_health_check_with_one_blas_thread():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))
    for name, service in compose["services"].items():
        if "healthcheck" in service:
            command = service["healthcheck"]["test"][1:]
            assert command == ["python", "-m", "task_manager.health", name]
            assert service["environment"]["OPENBLAS_NUM_THREADS"] == "1"


def test_backup_while_a_writer_is_active_is_consistent(encrypted, tmp_path):
    async def scenario():
        await _write_message(encrypted, 1)
        engine = make_engine(encrypted)
        try:
            async with AsyncSession(engine) as session:
                session.add(Message(chat_id=1, msg_id=2, sent_at=SENT, text="ещё"))
                await session.flush()
                path = create_backup(encrypted, tmp_path / "backups")
                await session.commit()
        finally:
            await engine.dispose()
        return path

    assert verify_backup(asyncio.run(scenario())).messages == 1


# Evaluation archive


def _archive_module():
    spec = importlib.util.spec_from_file_location("archive", ROOT / "eval" / "archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_data_is_sealed_and_can_come_back(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_KEY", KEY)
    archive = _archive_module()
    root = tmp_path / "eval"
    (root / "dataset" / "own").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "dataset" / "labels.json").write_text(NEEDLE, encoding="utf-8")
    (root / "dataset" / "own" / "sample.json").write_text("разметка", encoding="utf-8")
    (root / "results" / "a.jsonl").write_bytes(b"\x00\x01")
    target = tmp_path / "archive.db"

    assert archive.seal(target, root) == 3
    assert not (root / "dataset" / "labels.json").exists()
    assert NEEDLE.encode() not in target.read_bytes()

    assert archive.open_archive(target, root) == 3
    assert (root / "dataset" / "labels.json").read_text("utf-8") == NEEDLE
    assert (root / "results" / "a.jsonl").read_bytes() == b"\x00\x01"


def test_sealing_with_another_key_later_fails_without_deleting(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_KEY", KEY)
    archive = _archive_module()
    root = tmp_path / "eval"
    (root / "dataset").mkdir(parents=True)
    (root / "dataset" / "labels.json").write_text("а", encoding="utf-8")
    target = tmp_path / "archive.db"
    archive.seal(target, root)
    (root / "dataset" / "labels.json").write_text("б", encoding="utf-8")
    monkeypatch.setenv("DATABASE_KEY", OTHER_KEY)
    with pytest.raises(Exception):  # noqa: B017
        archive.seal(target, root)
    assert (root / "dataset" / "labels.json").read_text("utf-8") == "б"


# Monitoring


@pytest.fixture
def engine(tmp_path, monkeypatch):
    path = tmp_path / "plain.db"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    monkeypatch.delenv("DATABASE_KEY", raising=False)
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    return make_engine(path)


async def _ping_ok() -> None:
    return None


def _outbox(engine, messenger, watched=("collector", "analyzer"), running_for=None) -> Outbox:
    """`running_for` makes the bot look as if it started that long ago."""
    outbox = Outbox(
        make_sessions(engine), messenger, _ping_ok, 0, 60.0, watched, announce_start=True
    )
    if running_for is not None:
        outbox._started_at -= running_for
    return outbox


def _run(engine, scenario):
    async def wrapper():
        try:
            return await scenario()
        finally:
            await engine.dispose()

    return asyncio.run(wrapper())


async def _state(engine, service, age_minutes=0, detail=None) -> None:
    moment = datetime.now(UTC) - timedelta(minutes=age_minutes)
    async with AsyncSession(engine) as session:
        await session.merge(ServiceState(service=service, last_seen=moment, detail=detail))
        await session.commit()


def test_start_is_reported_once_with_the_downtime(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _state(engine, "review_bot", age_minutes=125)
        outbox = _outbox(engine, messenger, ())
        await outbox.flush()
        await outbox.flush()
        return [item[1] for item in messenger.sent]

    sent = _run(engine, scenario)
    assert len(sent) == 1
    assert "Простой: 2 ч 5 мин" in sent[0]


def test_first_start_has_no_downtime(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _outbox(engine, messenger, ()).flush()
        return messenger.last

    text_sent = _run(engine, scenario)
    assert "запущена" in text_sent
    assert "Простой" not in text_sent


def test_silent_service_is_reported_once_and_recovery_too(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _state(engine, "collector", age_minutes=10)
        outbox = _outbox(engine, messenger, running_for=timedelta(minutes=10))
        await outbox.flush()
        await outbox.flush()
        alerts = [item[1] for item in messenger.sent if "не отвечает" in item[1]]
        await record_state(make_sessions(engine), "collector")
        await outbox.flush()
        await outbox.flush()
        return alerts, [item[1] for item in messenger.sent if "Восстановлено" in item[1]]

    alerts, recovered = _run(engine, scenario)
    assert len(alerts) == 1
    assert "collector" in alerts[0]
    assert len(recovered) == 1


def test_lost_telegram_connection_is_reported(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _state(engine, "collector", detail="telegram_disconnected")
        await _outbox(engine, messenger, running_for=timedelta(minutes=10)).flush()
        return [item[1] for item in messenger.sent]

    assert any("Нет связи с Telegram" in text_sent for text_sent in _run(engine, scenario))


def test_stuck_queue_is_reported_only_while_the_model_is_up(engine, monkeypatch):
    async def scenario():
        messenger = FakeMessenger()
        async with AsyncSession(engine) as session:
            session.add(Chat(chat_id=1))
            await session.flush()
            session.add(
                Message(
                    chat_id=1,
                    msg_id=1,
                    sent_at=datetime.now(UTC) - timedelta(hours=2),
                    text="поправь",
                )
            )
            await session.commit()
        outbox = _outbox(engine, messenger, (), running_for=timedelta(hours=3))
        outbox._down_since = 1.0
        await outbox.flush()
        while_down = [item[1] for item in messenger.sent if "Очередь" in item[1]]
        outbox._down_since = None
        await outbox.flush()
        return while_down, [item[1] for item in messenger.sent if "Очередь" in item[1]]

    while_down, while_up = _run(engine, scenario)
    assert while_down == []
    assert len(while_up) == 1


def test_just_started_bot_ignores_what_the_downtime_left_behind(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _state(engine, "collector", age_minutes=70, detail="telegram_disconnected")
        await _state(engine, "analyzer", age_minutes=70)
        async with AsyncSession(engine) as session:
            session.add(Chat(chat_id=1))
            await session.flush()
            session.add(
                Message(
                    chat_id=1, msg_id=1, sent_at=datetime.now(UTC) - timedelta(hours=2), text="x"
                )
            )
            await session.commit()
        outbox = _outbox(engine, messenger)
        await outbox.flush()
        at_start = [item[1] for item in messenger.sent if "⚠️" in item[1]]
        # A service still silent once the bot has settled is a real problem.
        outbox._started_at -= timedelta(minutes=4)
        await outbox.flush()
        return at_start, [item[1] for item in messenger.sent if "⚠️" in item[1]]

    at_start, settled = _run(engine, scenario)
    assert at_start == []
    assert any("collector не отвечает" in text_sent for text_sent in settled)
    # The queue only started with the bot, so it isn't stuck yet.
    assert not any("Очередь" in text_sent for text_sent in settled)


def test_healthy_services_raise_nothing(engine):
    async def scenario():
        messenger = FakeMessenger()
        await _state(engine, "collector")
        await _state(engine, "analyzer")
        outbox = _outbox(engine, messenger)
        await outbox.flush()
        await outbox.flush()
        return [item[1] for item in messenger.sent]

    assert all("запущена" in text_sent for text_sent in _run(engine, scenario))


def test_state_holds_no_text(engine):
    async def scenario():
        await record_state(make_sessions(engine), "collector", "telegram_disconnected")
        async with AsyncSession(engine) as session:
            await session.execute(update(ServiceState).values(detail="telegram_disconnected"))
            await session.commit()
            return (await session.get(ServiceState, "collector")).detail

    assert _run(engine, scenario) == "telegram_disconnected"


def test_raw_connection_has_the_key(encrypted):
    asyncio.run(_write_message(encrypted))
    connection = connect_raw(encrypted, read_only=True)
    assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
    connection.close()
