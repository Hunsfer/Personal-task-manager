"""Application settings from the environment and validated YAML configuration."""

import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

Term = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConfigError(Exception):
    """Configuration is missing or invalid. The message is safe to show: it contains no secrets."""


class Settings(BaseSettings):
    # hide_input_in_errors keeps secret values out of validation error messages.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    database_path: Path = Path("data/app.db")
    database_key: SecretStr | None = None
    require_encrypted_db: bool = False
    config_dir: Path = Path("config")
    heartbeat_dir: Path = Path(tempfile.gettempdir())
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    tg_api_id: int | None = None
    tg_api_hash: SecretStr | None = None
    # The session itself, so there's no file granting access to the account on disk.
    tg_session: SecretStr | None = None
    session_path: Path = Path("data/telegram")
    history_depth_days: int = Field(default=30, ge=1)

    ollama_url: str = "http://127.0.0.1:11434"
    analyzer_model: str = "gemma4:12b"
    analyzer_prompt: str = "task_v4.md"

    bot_token: SecretStr | None = None
    bot_owner_id: int | None = None
    # Whole hours are enough for showing when a message was sent.
    bot_utc_offset_hours: int = Field(default=0, ge=-12, le=14)
    ollama_alert_minutes: int = Field(default=30, ge=1)

    jira_url: str | None = None
    jira_email: str | None = None
    jira_api_token: SecretStr | None = None
    attachments_dir: Path = Path("data/attachments")

    def telegram_credentials(self) -> tuple[int, str]:
        if self.tg_api_id is None or self.tg_api_hash is None:
            raise ConfigError(
                "Telegram credentials are missing: set tg_api_id and tg_api_hash "
                "with `tm secrets set`, then `tm start`"
            )
        return self.tg_api_id, self.tg_api_hash.get_secret_value()

    def bot_credentials(self) -> tuple[str, int]:
        if self.bot_token is None or self.bot_owner_id is None:
            raise ConfigError(
                "bot settings are missing: set bot_token and bot_owner_id "
                "with `tm secrets set`, then `tm start`"
            )
        return self.bot_token.get_secret_value(), self.bot_owner_id

    def jira_credentials(self) -> tuple[str, str, str]:
        if not self.jira_url or not self.jira_email or self.jira_api_token is None:
            raise ConfigError(
                "Jira settings are missing: set jira_url, jira_email and jira_api_token "
                "with `tm secrets set`, then `tm start`"
            )
        return self.jira_url, self.jira_email, self.jira_api_token.get_secret_value()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatEntry(_StrictModel):
    id: int
    title: Term | None = None


class ChatsConfig(_StrictModel):
    chats: list[ChatEntry]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "ChatsConfig":
        ids = [chat.id for chat in self.chats]
        duplicates = sorted({chat_id for chat_id in ids if ids.count(chat_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate chat ids: {duplicates}")
        return self


class Section(_StrictModel):
    name: Term
    aliases: list[Term]

    @model_validator(mode="after")
    def _has_aliases(self) -> "Section":
        if not self.aliases:
            raise ValueError(f"section {self.name!r} has no aliases")
        return self


class ScopeConfig(_StrictModel):
    role: list[Term]
    sections: list[Section]


class AppConfig(_StrictModel):
    chats: ChatsConfig
    scope: ScopeConfig


class Organization(_StrictModel):
    name: Term
    aliases: list[Term] = []


class MeConfig(_StrictModel):
    names: list[Term]
    organizations: list[Organization] = []
    same_people: list[list[Term]] = []
    secrets: list[Term] = []

    @model_validator(mode="after")
    def _has_names(self) -> "MeConfig":
        if not self.names:
            raise ValueError("list at least one name")
        return self


class AnonymizerConfig(_StrictModel):
    allow: list[Term]
    names: list[Term]


class JiraConfig(_StrictModel):
    project_key: Term
    issue_type: Term
    locked_statuses: list[Term]
    auto_append_statuses: list[Term] = []
    link_type: Term
    priorities: dict[str, Term] = {}


class AnalyzerConfig(_StrictModel):
    noise_phrases: list[Term]


def _describe(error: ValidationError) -> str:
    parts = []
    for item in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def _secrets_dir() -> Path | None:
    """Secrets live in memory inside the containers: see scripts/tm.ps1."""
    value = os.environ.get("SECRETS_DIR")
    return Path(value) if value else None


def _private_or_config(config_dir: Path, name: str) -> Path:
    """Files naming the employer or holding secrets come from memory when they're there.

    The repository is public, so such values can't live in a tracked file.
    """
    directory = _secrets_dir()
    if directory is not None and (directory / name).is_file():
        return directory / name
    return config_dir / name


def load_settings() -> Settings:
    directory = _secrets_dir()
    try:
        return Settings(_secrets_dir=directory) if directory else Settings()
    except ValidationError as error:
        raise ConfigError(f"invalid settings: {_describe(error)}") from None


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from None


def _load[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(_read_yaml(path))
    except ValidationError as error:
        raise ConfigError(f"{path}: {_describe(error)}") from None


def load_app_config(config_dir: Path) -> AppConfig:
    """Load and validate every YAML file. Raises ConfigError naming the file and the problem."""
    chats_path = _private_or_config(config_dir, "chats.yaml")
    if not chats_path.is_file():
        raise ConfigError(
            f"{chats_path} not found: list your chats with `tm secrets edit chats.yaml`, "
            "see chats.example.yaml"
        )

    scope_path = _private_or_config(config_dir, "scope.yaml")
    if not scope_path.is_file():
        raise ConfigError(
            f"{scope_path} not found: describe your role with `tm secrets edit scope.yaml`, "
            "see scope.example.yaml"
        )

    return AppConfig(
        chats=_load(chats_path, ChatsConfig),
        scope=_load(scope_path, ScopeConfig),
    )


def load_me_config(config_dir: Path) -> MeConfig:
    path = _private_or_config(config_dir, "me.yaml")
    if not path.is_file():
        raise ConfigError(
            f"{path} not found: list your names with `tm secrets edit me.yaml`, see me.example.yaml"
        )
    return _load(path, MeConfig)


def load_jira_config(config_dir: Path) -> JiraConfig:
    path = _private_or_config(config_dir, "jira.yaml")
    if not path.is_file():
        raise ConfigError(
            f"{path} not found: describe your project with `tm secrets edit jira.yaml`"
        )
    return _load(path, JiraConfig)


def load_analyzer_config(config_dir: Path) -> AnalyzerConfig:
    path = config_dir / "analyzer.yaml"
    if not path.is_file():
        raise ConfigError(f"{path} not found")
    return _load(path, AnalyzerConfig)


def load_anonymizer_config(config_dir: Path) -> AnonymizerConfig:
    path = config_dir / "anonymizer.yaml"
    if not path.is_file():
        raise ConfigError(f"{path} not found")
    return _load(path, AnonymizerConfig)
