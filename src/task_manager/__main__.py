"""Command line entry point: `python -m task_manager <command>`."""

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from task_manager.common.config import (
    AppConfig,
    ConfigError,
    Settings,
    load_analyzer_config,
    load_anonymizer_config,
    load_app_config,
    load_jira_config,
    load_me_config,
    load_settings,
)
from task_manager.common.log import configure_logging
from task_manager.health import heartbeat_path, is_healthy
from task_manager.service import SERVICES, stop_on_signals
from task_manager.storage.db import current_revision, make_engine

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2

LOGIN_HINT = "run `python -m task_manager login` first"


def _make_reader(settings: Settings):
    # Imported here so that other commands don't pay for loading Telethon.
    from task_manager.collector.telegram import TelegramReader

    api_id, api_hash = settings.telegram_credentials()
    session = settings.tg_session.get_secret_value() if settings.tg_session else None
    return TelegramReader(api_id, api_hash, settings.session_path, session)


async def _collect(settings: Settings, config: AppConfig) -> None:
    from task_manager.collector.runner import run_collector

    stop = asyncio.Event()
    stop_on_signals(stop)
    engine = make_engine(settings.database_path)
    try:
        await run_collector(
            reader=_make_reader(settings),
            engine=engine,
            chat_ids=[chat.id for chat in config.chats.chats],
            history_depth_days=settings.history_depth_days,
            heartbeat=heartbeat_path(settings.heartbeat_dir, "collector"),
            stop=stop,
            attachments_dir=settings.attachments_dir,
        )
    finally:
        await engine.dispose()


async def _anonymize(settings: Settings, config: AppConfig) -> None:
    from task_manager.anonymizer.runner import run_anonymizer

    stop = asyncio.Event()
    stop_on_signals(stop)
    engine = make_engine(settings.database_path)
    try:
        await run_anonymizer(
            engine=engine,
            me=load_me_config(settings.config_dir),
            scope=config.scope,
            config=load_anonymizer_config(settings.config_dir),
            heartbeat=heartbeat_path(settings.heartbeat_dir, "anonymizer"),
            stop=stop,
        )
    finally:
        await engine.dispose()


def _make_worker(settings: Settings, config: AppConfig, engine, client):
    from task_manager.analyzer.classifier import Classifier
    from task_manager.analyzer.prefilter import Prefilter
    from task_manager.analyzer.runner import QueueWorker, RulesSource, make_sessions

    return QueueWorker(
        make_sessions(engine),
        Classifier(
            client,
            settings.analyzer_model,
            settings.config_dir / "prompts",
            settings.analyzer_prompt,
            config.scope,
        ),
        Prefilter(load_analyzer_config(settings.config_dir).noise_phrases),
        RulesSource(
            load_me_config(settings.config_dir),
            config.scope,
            load_anonymizer_config(settings.config_dir),
        ),
        [chat.id for chat in config.chats.chats],
    )


async def _analyze(settings: Settings, config: AppConfig) -> None:
    from task_manager.analyzer.ollama import OllamaClient
    from task_manager.analyzer.runner import run_analyzer

    stop = asyncio.Event()
    stop_on_signals(stop)
    engine = make_engine(settings.database_path)
    client = OllamaClient(settings.ollama_url)
    try:
        await run_analyzer(
            _make_worker(settings, config, engine, client),
            heartbeat_path(settings.heartbeat_dir, "analyzer"),
            stop,
        )
    finally:
        await client.aclose()
        await engine.dispose()


async def _jira_export(settings: Settings, url: str, email: str, api_token: str) -> None:
    from task_manager.analyzer.runner import make_sessions
    from task_manager.anonymizer.engine import Anonymizer
    from task_manager.jira_export.client import JiraClient
    from task_manager.jira_export.exporter import Exporter, check_project, run_exporter

    stop = asyncio.Event()
    stop_on_signals(stop)
    jira = load_jira_config(settings.config_dir)
    me = load_me_config(settings.config_dir)
    # Only the regular expressions are used, on code going to Jira, so no language model is loaded.
    secrets = Anonymizer(None, me.names, [], [], [], [], [], me.secrets)
    engine = make_engine(settings.database_path)
    client = JiraClient(url, email, api_token)
    try:
        await current_revision(engine)
        account_id = await client.myself()
        statuses = await client.project_statuses(jira.project_key)
        check_project(jira, statuses)
        await run_exporter(
            Exporter(
                make_sessions(engine),
                client,
                jira,
                account_id,
                secrets.find_secrets,
            ),
            heartbeat_path(settings.heartbeat_dir, "jira_export"),
            stop,
        )
    finally:
        await client.aclose()
        await engine.dispose()


async def _review_bot(settings: Settings, config: AppConfig, token: str, owner_id: int) -> None:
    from task_manager.analyzer.classifier import Classifier
    from task_manager.analyzer.ollama import OllamaClient
    from task_manager.analyzer.runner import make_sessions
    from task_manager.anonymizer.engine import NerModel
    from task_manager.anonymizer.runner import live_rules
    from task_manager.review_bot.chat_log import ChatLog
    from task_manager.review_bot.controller import Controller
    from task_manager.review_bot.names import Names
    from task_manager.review_bot.outbox import Outbox
    from task_manager.review_bot.runner import run_until_stopped
    from task_manager.review_bot.telegram_api import TelegramMessenger

    stop = asyncio.Event()
    stop_on_signals(stop)
    me = load_me_config(settings.config_dir)
    anonymizer = load_anonymizer_config(settings.config_dir)
    engine = make_engine(settings.database_path)
    client = OllamaClient(settings.ollama_url)
    telegram = TelegramMessenger(token, owner_id)
    sessions = make_sessions(engine)
    messenger = ChatLog(sessions, telegram)
    outbox = Outbox(
        sessions,
        messenger,
        client.ping,
        settings.bot_utc_offset_hours,
        settings.ollama_alert_minutes * 60,
        SERVICES,
        announce_start=True,
    )

    async def rules(session) -> str:
        return await live_rules(session, me, config.scope, anonymizer)

    controller = Controller(
        sessions,
        messenger,
        Classifier(
            client,
            settings.analyzer_model,
            settings.config_dir / "prompts",
            settings.analyzer_prompt,
            config.scope,
        ),
        Names(NerModel(), me, config.scope, anonymizer),
        rules,
        outbox.flush,
        lambda: outbox.ollama_available,
        owner_id,
        settings.bot_utc_offset_hours,
    )
    try:
        await current_revision(engine)
        await run_until_stopped(
            engine,
            sessions,
            owner_id,
            telegram,
            controller,
            outbox,
            heartbeat_path(settings.heartbeat_dir, "review_bot"),
            stop,
        )
    finally:
        await telegram.close()
        await client.aclose()
        await engine.dispose()


def _run(service: str) -> int:
    from task_manager.collector.types import NotAuthorizedError

    try:
        settings = load_settings()
        config = load_app_config(settings.config_dir)
        if service == "collector":
            settings.telegram_credentials()
        if service == "review_bot":
            token, owner_id = settings.bot_credentials()
        if service == "jira_export":
            jira_url, jira_email, jira_token = settings.jira_credentials()
            load_jira_config(settings.config_dir)
            load_me_config(settings.config_dir)
        if service in ("anonymizer", "analyzer", "review_bot"):
            load_me_config(settings.config_dir)
            load_anonymizer_config(settings.config_dir)
        if service == "analyzer":
            load_analyzer_config(settings.config_dir)
    except ConfigError as error:
        # Printed as is rather than logged: the log formatter drops message details by design.
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    configure_logging(service, settings.log_level)
    try:
        if service == "collector":
            asyncio.run(_collect(settings, config))
        elif service == "anonymizer":
            asyncio.run(_anonymize(settings, config))
        elif service == "analyzer":
            asyncio.run(_analyze(settings, config))
        elif service == "jira_export":
            asyncio.run(_jira_export(settings, jira_url, jira_email, jira_token))
        else:
            asyncio.run(_review_bot(settings, config, token, owner_id))
    except NotAuthorizedError as error:
        print(f"Configuration error: {error}: {LOGIN_HINT}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception:
        logging.getLogger(__name__).exception("service_failed")
        return EXIT_FAILURE
    return EXIT_OK


def _health(service: str) -> int:
    try:
        settings = load_settings()
    except ConfigError:
        return EXIT_FAILURE
    path = heartbeat_path(settings.heartbeat_dir, service)
    return EXIT_OK if is_healthy(path) else EXIT_FAILURE


async def _login(settings: Settings, save_to: Path | None = None) -> None:
    from task_manager.collector.telegram import TelegramReader

    api_id, api_hash = settings.telegram_credentials()
    reader = TelegramReader(api_id, api_hash, settings.session_path, "" if save_to else None)
    try:
        await reader.login()
        if save_to:
            save_to.write_text(reader.export_session(), encoding="utf-8")
    finally:
        await reader.disconnect()
    print("Telegram session saved. Keep it private: it gives full access to the account.")


async def _list_chats(settings: Settings) -> None:
    from task_manager.collector.types import NotAuthorizedError

    reader = _make_reader(settings)
    try:
        await reader.connect()
        if not await reader.is_authorized():
            raise NotAuthorizedError("no valid Telegram session")
        chats = await reader.list_chats()
    finally:
        await reader.disconnect()
    for chat in chats:
        print(f"{chat.id}\t{chat.kind.value}\t{chat.title or ''}")


def _telegram_command(command) -> int:
    from task_manager.collector.types import NotAuthorizedError, RateLimitedError

    try:
        asyncio.run(command(load_settings()))
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except NotAuthorizedError as error:
        print(f"Configuration error: {error}: {LOGIN_HINT}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except RateLimitedError as error:
        print(f"Telegram asks to wait {error.seconds} seconds. Try again later.", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_OK


async def _preview(settings: Settings, path: Path) -> None:
    from task_manager.anonymizer.preview import write_preview

    engine = make_engine(settings.database_path)
    try:
        counters = await write_preview(engine, path)
    finally:
        await engine.dispose()
    # Counters only: the sample itself stays in the file.
    for name, count in counters.items():
        print(f"{name}: {count}")
    print(f"Sample written to {path}. It holds real messages: delete it after reading.")


def _preview_command(path: Path) -> int:
    try:
        asyncio.run(_preview(load_settings(), path))
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    return EXIT_OK


async def _eval_sample(settings: Settings, directory: Path, refresh: bool) -> None:
    from task_manager.evaluation import labeling, sampling

    config = load_app_config(settings.config_dir)
    engine = make_engine(settings.database_path)
    try:
        examples, skipped = await sampling.collect_examples(
            engine,
            [chat.id for chat in config.chats.chats],
            load_me_config(settings.config_dir),
            config.scope,
            load_anonymizer_config(settings.config_dir),
        )
    finally:
        await engine.dispose()
    sample = (
        sampling.refresh_sample(sampling.read_sample(directory), examples, skipped)
        if refresh
        else sampling.select_sample(examples, skipped)
    )
    sampling.write_sample(sample, directory)
    labels_path = directory / labeling.LABELS_FILE
    labels = labeling.read_labels(labels_path) if labels_path.is_file() else {}
    page = labeling.write_labeling_page(sample, directory, labels)
    # Counters only: the messages stay in the files.
    for name, count in sampling.sample_counters(sample).items():
        print(f"{name}: {count}")
    print(f"Labelling page written to {page}. It holds anonymized messages: keep it out of git.")


def _eval_sample_command(directory: Path, refresh: bool) -> int:
    return _config_command(lambda: _eval_sample(load_settings(), directory, refresh))


async def _eval_sample_own(settings: Settings, directory: Path) -> None:
    from task_manager.analyzer.prefilter import Prefilter
    from task_manager.evaluation import labeling, sampling

    config = load_app_config(settings.config_dir)
    prefilter = Prefilter(load_analyzer_config(settings.config_dir).noise_phrases)
    engine = make_engine(settings.database_path)
    try:
        examples, skipped = await sampling.collect_examples(
            engine,
            [chat.id for chat in config.chats.chats],
            load_me_config(settings.config_dir),
            config.scope,
            load_anonymizer_config(settings.config_dir),
            own=True,
        )
    finally:
        await engine.dispose()
    # A reply such as "ок" never reaches the model, so labelling it would be wasted time.
    examples = [e for e in examples if prefilter.noise_reason(e.thread.target) is None]
    directory.mkdir(parents=True, exist_ok=True)
    sample = sampling.select_sample(
        examples, skipped, sampling.OWN_SEED, sampling.OWN_RANDOM_SIZE, sampling.OWN_QUOTAS
    )
    sampling.write_sample(sample, directory)
    page = labeling.write_labeling_page(
        sample, directory, None, labeling.OWN_LABELS, labeling.OWN_LABELS_HINT
    )
    for name, count in sampling.sample_counters(sample).items():
        print(f"{name}: {count}")
    print(f"Labelling page written to {page}. It holds anonymized messages: keep it out of git.")


def _config_command(make_coroutine) -> int:
    """Runs a command against the database and reports bad configuration in plain words."""
    try:
        asyncio.run(make_coroutine())
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    return EXIT_OK


async def _analyze_retry(settings: Settings) -> None:
    from task_manager.analyzer.commands import retry_failed

    engine = make_engine(settings.database_path)
    try:
        print(f"returned_to_queue: {await retry_failed(engine)}")
    finally:
        await engine.dispose()


async def _archive_unlisted(settings: Settings) -> None:
    from task_manager.analyzer.commands import archive_unlisted

    chats = load_app_config(settings.config_dir).chats.chats
    engine = make_engine(settings.database_path)
    try:
        count = await archive_unlisted(engine, [chat.id for chat in chats])
    finally:
        await engine.dispose()
    print(f"archived: {count}")


async def _current_rules(settings: Settings, engine) -> str:
    from sqlalchemy.ext.asyncio import AsyncSession

    from task_manager.anonymizer.runner import live_rules

    async with AsyncSession(engine) as session:
        return await live_rules(
            session,
            load_me_config(settings.config_dir),
            load_app_config(settings.config_dir).scope,
            load_anonymizer_config(settings.config_dir),
        )


async def _write_preview_and_counters(settings: Settings, engine, path: Path) -> None:
    from task_manager.analyzer.commands import draft_counters, write_drafts_preview

    await write_drafts_preview(engine, path, await _current_rules(settings, engine))
    # Counters only: the drafts stay in the file.
    for name, count in (await draft_counters(engine)).items():
        print(f"{name}: {count}")
    print(f"Drafts written to {path}. It holds anonymized messages: delete it after reading.")


async def _drafts_preview(settings: Settings, path: Path) -> None:
    engine = make_engine(settings.database_path)
    try:
        await _write_preview_and_counters(settings, engine, path)
    finally:
        await engine.dispose()


async def _feedback_export(settings: Settings, path: Path) -> None:
    from task_manager.analyzer.commands import feedback_export

    engine = make_engine(settings.database_path)
    try:
        counters = await feedback_export(engine, path)
    finally:
        await engine.dispose()
    for name, count in counters.items():
        print(f"{name}: {count}")
    print(f"Labels written to {path}.")


async def _analyze_replay(settings: Settings, path: Path, days: int | None) -> None:
    from task_manager.analyzer.commands import (
        make_replay_copy,
        remove_replay_copy,
        run_replay,
        status_counts,
    )
    from task_manager.analyzer.ollama import OllamaClient

    config = load_app_config(settings.config_dir)
    real = make_engine(settings.database_path)
    before = await status_counts(real)
    target = Path(tempfile.gettempdir()) / "replay.db"
    remove_replay_copy(target)
    client = OllamaClient(settings.ollama_url)
    try:
        chat_ids = [chat.id for chat in config.chats.chats]
        print(
            f"replay_messages: {make_replay_copy(settings.database_path, target, chat_ids, days)}"
        )
        engine = make_engine(target)
        try:
            worker = _make_worker(settings, config, engine, client)
            started = time.monotonic()
            counters = await run_replay(worker)
            print(f"replay_minutes: {(time.monotonic() - started) / 60:.1f}")
            for name, count in sorted(counters.items()):
                print(f"replay_{name}: {count}")
            await _write_preview_and_counters(settings, engine, path)
        finally:
            await engine.dispose()
        after = await status_counts(real)
        print(f"real_statuses_before: {before}")
        print(f"real_statuses_after: {after}")
    finally:
        await client.aclose()
        await real.dispose()
        remove_replay_copy(target)


def _backup_command(directory: Path) -> int:
    from task_manager.storage.backup import create_backup, rotate_backups, verify_backup

    try:
        settings = load_settings()
        path = create_backup(settings.database_path, directory)
        check = verify_backup(path)
        removed = rotate_backups(directory)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(f"backup: {path.name}, messages {check.messages}, drafts {check.drafts}, old {removed}")
    return EXIT_OK


def _verify_command(path: Path) -> int:
    from task_manager.storage.backup import verify_backup

    try:
        load_settings()
        check = verify_backup(path)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as error:
        print(f"The backup can't be read: {type(error).__name__}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"ok: messages {check.messages}, drafts {check.drafts}")
    return EXIT_OK


def _restore_command(path: Path) -> int:
    from task_manager.storage.backup import restore_backup

    try:
        restore_backup(path, load_settings().database_path)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print("restored")
    return EXIT_OK


def _replay_command(path: Path, days: int | None) -> int:
    from task_manager.analyzer.ollama import OllamaUnavailableError

    try:
        return _config_command(lambda: _analyze_replay(load_settings(), path, days))
    except OllamaUnavailableError as error:
        print(f"Ollama is unavailable: {error}. The replay was not finished.", file=sys.stderr)
        return EXIT_FAILURE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="task_manager")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "health"):
        commands.add_parser(name).add_argument("service", choices=SERVICES)
    login = commands.add_parser("login", help="log in to Telegram interactively")
    login.add_argument("--save-to", type=Path, help="write the session to this file")
    backup = commands.add_parser("backup", help="write an encrypted copy of the database")
    backup.add_argument("directory", type=Path)
    verify = commands.add_parser("verify-backup", help="open a backup and check that it reads")
    verify.add_argument("path", type=Path)
    restore = commands.add_parser("restore", help="replace the database: stop the services first")
    restore.add_argument("path", type=Path)
    commands.add_parser("list-chats", help="print the ID, kind and title of your chats")
    preview = commands.add_parser(
        "anonymize-preview", help="write a before/after sample of real messages to a file"
    )
    preview.add_argument("path", type=Path)
    sample = commands.add_parser(
        "eval-sample", help="write the messages to label and a page for labelling them"
    )
    sample.add_argument("directory", type=Path)
    sample.add_argument(
        "--refresh",
        action="store_true",
        help="keep the existing messages and their labels, with current text and whitelist",
    )
    commands.add_parser(
        "eval-sample-own", help="write the owner's own messages to label for promises"
    ).add_argument("directory", type=Path)
    commands.add_parser("analyze-retry", help="return failed messages to the analysis queue")
    commands.add_parser(
        "archive-unlisted", help="take waiting messages of unlisted chats out of the queue"
    )
    commands.add_parser(
        "drafts-preview", help="write the drafts next to their messages to a file"
    ).add_argument("path", type=Path)
    commands.add_parser(
        "feedback-export", help="write the owner's decisions as labels for a later evaluation"
    ).add_argument("path", type=Path)
    replay = commands.add_parser(
        "analyze-replay", help="run the analyzer over history on a temporary copy of the database"
    )
    replay.add_argument("path", type=Path)
    replay.add_argument("--days", type=int, help="only this many most recent days")
    args = parser.parse_args(argv)

    match args.command:
        case "run":
            return _run(args.service)
        case "health":
            return _health(args.service)
        case "login":
            return _telegram_command(lambda settings: _login(settings, args.save_to))
        case "backup":
            return _backup_command(args.directory)
        case "verify-backup":
            return _verify_command(args.path)
        case "restore":
            return _restore_command(args.path)
        case "anonymize-preview":
            return _preview_command(args.path)
        case "eval-sample":
            return _eval_sample_command(args.directory, args.refresh)
        case "eval-sample-own":
            return _config_command(lambda: _eval_sample_own(load_settings(), args.directory))
        case "analyze-retry":
            return _config_command(lambda: _analyze_retry(load_settings()))
        case "archive-unlisted":
            return _config_command(lambda: _archive_unlisted(load_settings()))
        case "drafts-preview":
            return _config_command(lambda: _drafts_preview(load_settings(), args.path))
        case "feedback-export":
            return _config_command(lambda: _feedback_export(load_settings(), args.path))
        case "analyze-replay":
            return _replay_command(args.path, args.days)
        case _:
            return _telegram_command(_list_chats)


if __name__ == "__main__":
    sys.exit(main())
