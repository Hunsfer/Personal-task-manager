import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from task_manager.__main__ import EXIT_CONFIG_ERROR, EXIT_FAILURE, EXIT_OK, main
from task_manager.collector.types import ChatInfo, NotAuthorizedError
from task_manager.health import HEARTBEAT_MAX_AGE_S, heartbeat_path, is_healthy
from task_manager.health import main as health_main
from task_manager.service import beat
from task_manager.storage.models import ChatKind


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Run away from the repository, so a real .env.telegram can't affect the tests.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TG_API_ID", raising=False)
    monkeypatch.delenv("TG_API_HASH", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "chats.yaml").write_text("chats: []\n", encoding="utf-8")
    (config_dir / "scope.yaml").write_text("role: []\nsections: []\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "app.db"))
    monkeypatch.setenv("HEARTBEAT_DIR", str(tmp_path))
    return tmp_path


def test_health_requires_a_recent_heartbeat(tmp_path: Path) -> None:
    path = tmp_path / "collector.heartbeat"
    assert not is_healthy(path)

    path.touch()
    assert is_healthy(path)

    stale = time.time() - HEARTBEAT_MAX_AGE_S - 1
    os.utime(path, (stale, stale))
    assert not is_healthy(path)


def test_health_command_exit_codes(environment: Path) -> None:
    assert main(["health", "collector"]) == EXIT_FAILURE
    heartbeat_path(environment, "collector").touch()
    assert main(["health", "collector"]) == EXIT_OK


def test_docker_health_check_exit_codes(environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert health_main(["collector"]) == EXIT_FAILURE
    heartbeat_path(environment, "collector").touch()
    assert health_main(["collector"]) == EXIT_OK
    assert health_main([]) == EXIT_FAILURE
    monkeypatch.delenv("HEARTBEAT_DIR")
    assert health_main(["collector"]) == EXIT_FAILURE


def test_docker_health_check_stays_light() -> None:
    # It starts in every container every 15 seconds, so a heavy import here costs the most CPU.
    heavy = ("sqlalchemy", "pydantic", "asyncio", "yaml", "numpy")
    probe = f"import sys, task_manager.health; print([m for m in {heavy} if m in sys.modules])"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_invalid_config_stops_startup_with_a_clear_reason(
    environment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (environment / "config" / "chats.yaml").unlink()
    assert main(["run", "collector"]) == EXIT_CONFIG_ERROR
    assert "chats.yaml" in capsys.readouterr().err


def test_review_bot_without_settings_explains_what_to_do(
    environment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "review_bot"]) == EXIT_CONFIG_ERROR
    assert "tm secrets" in capsys.readouterr().err


def test_unmigrated_database_fails_startup(
    environment: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = environment / "config"
    (config_dir / "me.yaml").write_text("names: [Тестовый]\n", encoding="utf-8")
    (config_dir / "anonymizer.yaml").write_text("allow: []\nnames: []\n", encoding="utf-8")
    prompts = config_dir / "prompts"
    prompts.mkdir()
    for path in (Path(__file__).parent.parent / "config" / "prompts").glob("*.md"):
        (prompts / path.name).write_bytes(path.read_bytes())
    monkeypatch.setenv("BOT_TOKEN", "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    monkeypatch.setenv("BOT_OWNER_ID", "1")
    assert main(["run", "review_bot"]) == EXIT_FAILURE
    assert not heartbeat_path(environment, "review_bot").exists()


def test_collector_without_credentials_explains_what_to_do(
    environment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "collector"]) == EXIT_CONFIG_ERROR
    assert "tm secrets" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["login", "list-chats"])
def test_telegram_commands_without_credentials_explain_what_to_do(
    environment: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    assert main([command]) == EXIT_CONFIG_ERROR
    assert "tm secrets" in capsys.readouterr().err


def test_collector_with_missing_session_suggests_login(
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def not_authorized(*_args: object) -> None:
        raise NotAuthorizedError("no valid Telegram session")

    monkeypatch.setenv("TG_API_ID", "1")
    monkeypatch.setenv("TG_API_HASH", "api-hash-value")
    monkeypatch.setattr("task_manager.__main__._collect", not_authorized)

    assert main(["run", "collector"]) == EXIT_CONFIG_ERROR
    output = capsys.readouterr().err
    assert "login" in output
    assert "api-hash-value" not in output


def test_anonymizer_without_me_config_explains_what_to_do(
    environment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "anonymizer"]) == EXIT_CONFIG_ERROR
    assert "me.example.yaml" in capsys.readouterr().err


def test_list_chats_prints_id_kind_and_title(
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Reader:
        async def connect(self) -> None: ...
        async def disconnect(self) -> None: ...
        async def is_authorized(self) -> bool:
            return True

        async def list_chats(self) -> list[ChatInfo]:
            return [ChatInfo(-100, "Рабочий чат", ChatKind.SUPERGROUP)]

    monkeypatch.setattr("task_manager.__main__._make_reader", lambda _settings: Reader())
    assert main(["list-chats"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == "-100\tsupergroup\tРабочий чат"


def test_heartbeat_runs_until_told_to_finish(tmp_path: Path) -> None:
    path = tmp_path / "service.heartbeat"

    async def scenario() -> bool:
        finished = asyncio.Event()
        task = asyncio.create_task(beat(path, finished))
        for _ in range(100):
            if path.exists():
                break
            await asyncio.sleep(0.01)
        beating = is_healthy(path)
        finished.set()
        await task
        return beating

    assert asyncio.run(scenario())
