from pathlib import Path

import pytest

from task_manager.common.config import (
    ConfigError,
    Settings,
    load_app_config,
    load_settings,
)

REPO_CONFIG = Path(__file__).parent.parent / "config"
VALID_SCOPE = "role: [фронтенд]\nsections:\n  - name: Статьи\n    aliases: [блог]\n"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "chats.yaml").write_text("chats:\n  - id: -100\n", encoding="utf-8")
    (tmp_path / "scope.yaml").write_text(VALID_SCOPE, encoding="utf-8")
    return tmp_path


def test_valid_config_loads_cyrillic(config_dir: Path) -> None:
    config = load_app_config(config_dir)
    assert config.chats.chats[0].id == -100
    assert config.scope.role == ["фронтенд"]
    assert config.scope.sections[0].name == "Статьи"


def test_missing_chats_file_points_to_example(config_dir: Path) -> None:
    (config_dir / "chats.yaml").unlink()
    with pytest.raises(ConfigError, match=r"chats\.example\.yaml"):
        load_app_config(config_dir)


def test_broken_yaml_names_the_file(config_dir: Path) -> None:
    (config_dir / "scope.yaml").write_text("role: [", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"scope\.yaml"):
        load_app_config(config_dir)


def test_unknown_key_is_rejected(config_dir: Path) -> None:
    (config_dir / "chats.yaml").write_text("chatz: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="chats"):
        load_app_config(config_dir)


def test_duplicate_chat_ids_are_rejected(config_dir: Path) -> None:
    (config_dir / "chats.yaml").write_text("chats:\n  - id: 1\n  - id: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_app_config(config_dir)


def test_section_without_aliases_is_rejected(config_dir: Path) -> None:
    scope = "role: []\nsections:\n  - name: Статьи\n    aliases: []\n"
    (config_dir / "scope.yaml").write_text(scope, encoding="utf-8")
    with pytest.raises(ConfigError, match="no aliases"):
        load_app_config(config_dir)


def test_settings_error_does_not_reveal_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "secret-value")
    with pytest.raises(ConfigError) as caught:
        load_settings()
    assert "log_level" in str(caught.value)
    assert "secret-value" not in str(caught.value)


def test_api_hash_is_hidden_from_representations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TG_API_ID", "1")
    monkeypatch.setenv("TG_API_HASH", "api-hash-value")
    settings = Settings(_env_file=None)
    assert "api-hash-value" not in repr(settings)
    assert settings.telegram_credentials() == (1, "api-hash-value")


def test_invalid_api_id_does_not_reveal_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TG_API_ID", "not-a-number-value")
    with pytest.raises(ConfigError) as caught:
        load_settings()
    assert "tg_api_id" in str(caught.value)
    assert "not-a-number-value" not in str(caught.value)


@pytest.mark.parametrize("variable", ["TG_API_ID", "TG_API_HASH"])
def test_missing_credentials_name_the_file_to_create(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv("TG_API_ID", "1")
    monkeypatch.setenv("TG_API_HASH", "api-hash-value")
    monkeypatch.delenv(variable)
    with pytest.raises(ConfigError, match=r"tm secrets"):
        Settings(_env_file=None).telegram_credentials()


def test_history_depth_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HISTORY_DEPTH_DAYS", "0")
    with pytest.raises(ConfigError, match="history_depth_days"):
        load_settings()


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", "/data/app.db")
    assert Settings().database_path == Path("/data/app.db")


def test_repo_examples_are_valid(tmp_path: Path) -> None:
    for name in ("chats", "scope"):
        (tmp_path / f"{name}.yaml").write_bytes((REPO_CONFIG / f"{name}.example.yaml").read_bytes())
    assert load_app_config(tmp_path).scope.sections


def test_missing_scope_file_points_to_example(config_dir: Path) -> None:
    (config_dir / "scope.yaml").unlink()
    with pytest.raises(ConfigError, match=r"scope\.example\.yaml"):
        load_app_config(config_dir)


def test_private_scope_replaces_the_file_in_config(
    config_dir: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets = tmp_path_factory.mktemp("secrets")
    (secrets / "scope.yaml").write_text("role: [Из памяти]\nsections: []\n", encoding="utf-8")
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    assert load_app_config(config_dir).scope.role == ["Из памяти"]
    monkeypatch.delenv("SECRETS_DIR")
    assert load_app_config(config_dir).scope.role == ["фронтенд"]
