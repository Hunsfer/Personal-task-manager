# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
uv sync                                 # install dependencies
uv run pytest                           # all tests
uv run pytest tests/test_config.py::test_unknown_key_is_rejected   # one test
uv run ruff check . && uv run ruff format .

pwsh scripts/tm.ps1 start               # the only way to start: it loads secrets into memory first
pwsh scripts/tm.ps1 stop                # also: restart, status, logs <service>, backup, eval seal|open
pwsh scripts/tm.ps1 secrets list        # names only; set <name>, edit <file.yaml>
pwsh scripts/tm.ps1 telegram-login      # interactive
docker compose run --rm --no-deps collector python -m task_manager list-chats       # needs tm start first

uv run alembic upgrade head             # local database at data/app.db, needed before autogenerate
uv run alembic revision --autogenerate -m "describe the change"

# Before/after sample of real messages. Prints counters only. Don't read the file: it's the owner's.
docker compose run --rm --no-deps -v "${PWD}/temp:/out" anonymizer python -m task_manager anonymize-preview /out/anonymizer_preview.md

# Model evaluation. Prints counters and rates only. Don't read eval/dataset or eval/results: they're the owner's.
# They are sealed in eval/archive.db between uses: tm eval open, then tm eval seal.
docker compose run --rm --no-deps -v "${PWD}/eval/dataset:/out" anonymizer python -m task_manager eval-sample /out
uv run python eval/run_eval.py split
uv run python eval/run_eval.py run --model gemma4:12b --prompt config/prompts/task_v4.md
uv run python eval/run_eval.py report
uv run python eval/run_eval.py prefilter          # what the prefilter drops, per label
docker compose run --rm --no-deps -v "${PWD}/eval/dataset/own:/out" anonymizer python -m task_manager eval-sample-own /out
uv run python eval/run_eval.py own-run && uv run python eval/run_eval.py own-report

# Drafts. Print counters only. Don't read the files: they're the owner's.
docker compose run --rm --no-deps -v "${PWD}/temp:/out" analyzer python -m task_manager drafts-preview /out/drafts.md
docker compose run --rm --no-deps -v "${PWD}/temp:/out" analyzer python -m task_manager analyze-replay /out/replay.md
docker compose run --rm --no-deps analyzer python -m task_manager analyze-retry
docker compose run --rm --no-deps collector python -m task_manager archive-unlisted
```

Commits follow Conventional Commits, for example `feat(storage): ...`.

## Project status

Configuration, database, logging, containers, `collector`, `anonymizer` and `analyzer` exist. `collector` is covered by tests with a fake client and has been verified against a real Telegram account. `analyzer` has run on real messages and writes drafts to the database. `review_bot` has been verified against real Telegram. `jira_export` has created a real task, but file upload, code blocks, appending and the locked-status check have only run against a fake server, so check them first if something looks wrong. Descriptions of those stages below are the intended design, so check what actually exists before relying on them.

## What this is

A self-hosted app that reads the user's Telegram messages, uses a local LLM to find tasks assigned to them personally, and prepares Jira task drafts with descriptions and attachments from the source messages. A human approves each draft before it reaches Jira.

Planned stack: Python 3.12, Telethon, Pydantic and pydantic-settings, SQLAlchemy with SQLite, Ollama, Natasha, aiogram, Docker Compose.

Expected volume is low: 20 chats, about 100 messages a day. Avoid designing for throughput.

## Architecture

The pipeline runs in this order, with each stage planned as a package under `src/task_manager/`:

`collector` → `storage` → `anonymizer` → `analyzer` → `review_bot` → `jira_export`

- **collector**: Telethon user client (MTProto), because a bot cannot see the user's personal chats or other people's groups. Reads only chats whitelisted in `config/chats.yaml`. Direct messages are whitelisted like any other chat.
  - A chat's first read stores `HISTORY_DEPTH_DAYS` of history as `archived`, which never enters the analysis queue. It sets `history_loaded_at` when finished, and an interrupted load starts over. Anything newer than the chat's latest message at the start of the load is left for catch-up, so it isn't archived by mistake.
  - Catch-up runs at startup, on reconnect and every 5 minutes, reading newer than the `last_msg_id` cursor. Only catch-up moves the cursor, in the same transaction as the message. Realtime messages must not move it, or they would hide a gap before them.
  - Edits update the stored message, and changed content returns it to `pending`, unless it is `archived`. Telegram also reports reactions as edits, so unchanged content isn't requeued. Edits made during downtime and deletions aren't tracked.
  - `mentions_me` must be saved on arrival, because a mention by name without `@` can't be recovered from the text later.
  - Link previews aren't attachments, although Telethon's `message.photo` and `message.file` expose them as if they were.
  - Don't cancel a task that may be opening a database connection: aiosqlite's non-daemon worker thread would lose its owner and could keep the process from exiting. That's why the heartbeat loop ends on an event.
- **anonymizer**: runs as its own container and keeps one `anonymized_messages` record per message, including `archived` ones, without touching `messages.status`. `store.get_ready` is the only way to read text for the LLM: it returns nothing unless the record matches both the current message and the current rules, so a failed message has nothing to read. An edit deletes the record in the same transaction.
  - `engine.py` works in layers, each unable to touch what an earlier one claimed: regexes for contacts, links and secrets; protection of allowed terms; `<ME>`; known people; Natasha's NER; a morphological safety net.
  - Regexes must stay ahead of term protection. An allowed word inside a link or password would otherwise keep the whole value readable.
  - Natasha alone isn't enough: it misses lowercase names and some capitalized ones. Its `MorphVocab` doesn't work on Python 3.12, so use `pymorphy3`, and check every parse of a word: the first reading of "Кате" is "кат".
  - Neither tool can be trusted as is on chat text. `pymorphy3` guesses a name tag for words outside its dictionary (`скроллбары`, `билдами`), so check `word_is_known`. Natasha often takes a sentence's first word for a name or organization, so its hits go through `_entity_is_plausible`.
  - An unfamiliar name at the start of a sentence, or a name that is also a common word ("Роман"), can still slip through unless the person is in `people` or listed. That's the accepted trade-off after real over-masking.
  - Organizations, spellings of one person, and known secrets are listed in `config/me.yaml`, which isn't in git because it names the employer and holds secrets. Generic terms go in `config/anonymizer.yaml`, which is, and the owner's own words go in the `anonymizer_overrides` table. Nothing from `me.yaml` may appear in tests or commits.
  - Hyphenated words aren't treated as names: compound names are very rare here, and words such as "мини-попап" are common.
  - A lone Latin word isn't treated as an organization: in these chats it's nearly always a tool.
  - After a rebuild, `prune_pseudonyms` removes map entries nothing refers to, so a value older rules stored as a name doesn't outlive being reclassified as a secret.
  - Pseudonyms are reversible and global, so one person has one placeholder in every chat. `<SECRET>` and `<CARD>` are irreversible and must never reach the `pseudonyms` table, and neither may URL query strings or logins.
  - The user becomes `<ME>`, matched by the `is_me` row in `people`, `@username`, and the names in `config/me.yaml`. Without this token the model cannot tell that a task is addressed to the user.
  - `collector` fills `people` at startup and as messages arrive. Known people are part of the rules version, so learning a name rebuilds older messages. Raise `ENGINE_VERSION` in `anonymizer/store.py` when detection logic changes.
  - Terms from `config/scope.yaml` and `allow` in `config/anonymizer.yaml` stay readable, including URL paths such as `/blog`. Collective words ("мы", "нас") stay readable too: they don't prove the user is addressed, so they must not become `<ME>`.
  - Test with made-up data only. When checking real data, print counters or text shapes, never the messages or names. Viewers hide `<ME>` and `<SECRET>` as HTML tags, so `anonymize-preview` puts messages in code blocks for Markdown and refuses HTML.
- **analyzer**: `QueueWorker` takes `pending` messages of whitelisted chats in the order they were sent, because history may have been loaded newest first. The decision is binary: a task or nothing. There is no confidence threshold and no "maybe" list, as the model's confidence barely varies; `confidence` is only recorded. `collective` and `none` never create a draft.
  - The prefilter (`config/analyzer.yaml`) drops only certain noise, since a missed task is worse than asking the model about small talk. `eval/run_eval.py prefilter` checks it against the labels.
  - Incoming messages use `gemma4:12b` with `task_v4.md`. `verdict_schema()` and that prompt were measured on the final part, so changing either invalidates those numbers. The first attempt must keep the evaluation's seed. Retries change it, because the same seed repeats the same bad answer.
  - The owner's own messages use `promise_v3.md` with its own schema and count only as promises. `promise` isn't in `AssignmentReason`, so the incoming schema can't produce it.
  - A message stays `pending` until its result is stored, and `processing` isn't used, so a crash loses nothing. `_store` writes first to take the lock, then compares `source_hash`: an edit during a model call discards the answer instead of being overwritten by `done`. Ollama being down raises `OllamaUnavailableError`, leaves statuses alone and retries with growing pauses. A bad answer ends in `failed`, which `analyze-retry` requeues.
  - The heartbeat has its own loop, since one model call can far outlast its 30 seconds. It ends on an event and touches only a file.
  - Merging happens within one chat and 24 hours of conversation time (`last_message_at`, not the clock, so replays group the same way): replies and shared `evidence` first, then `merge_v2.md`, which compares the messages because titles of fragments of one conversation look unrelated. A non-task reply to a draft's message joins as `context`. Merging only adds `draft_messages` rows, so it can be seen and undone. Duplicates across chats aren't merged; embeddings were dropped because there's no data to set a similarity threshold.
  - `evidence` is limited to numbers the model was actually shown, so an invented one can't attach another message's files. `deadline` is the model's guess and its date arithmetic is unreliable, so nothing may use it automatically. Times in prompts are UTC.
  - `analyze-replay` works on a copy in `/tmp` and never touches the real database.
- **review_bot**: aiogram bot with its own token and long polling, answering only `BOT_OWNER_ID`, which must match the `is_me` person or the service stops. `telegram_api.py` is the only module importing aiogram, and `controller.py` and `outbox.py` work against the `Messenger` protocol so tests need no Telegram. `tests/test_telegram.py` exempts this package from the Telethon write check, on the condition that it can't import Telethon or `collector`.
  - Draft status goes `new` → `sent` → `approved` or `rejected`. `new` is the queue that survives the bot being down. Any change sets `needs_refresh`, and `outbox.py` then sends the card silently at the bottom of the chat and deletes the old one. Nothing edits a card in place.
  - Text in `drafts` is always anonymized. Real values are substituted only when showing it (`views.py`, `render.py`), and everything from chats or the model must go through `escape` or `highlight`, since messages use HTML.
  - Anything the owner types passes through `names.py` before it can reach the model or a draft, and rejection notes reach the follow-up prompt, so they're anonymized too. Don't log any of it.
  - Messages about a decided draft never change it: `follow_up` decides whether to open a linked ticket, and unsure or failed answers mean yes. `edited_by_owner` stops rebuilding the text from messages, so the owner's wording survives; the draft shows `text_is_stale` instead.
  - Words marked "not a name" live in `anonymizer_overrides`, because `config/` is read-only and that file is in git. They're part of the rules version, so `anonymizer` rebuilds records by itself. Marking replaces the placeholder in drafts and notes in the same transaction, since cleanup may remove it from the map afterwards. `prune_pseudonyms` treats drafts as references for the same reason. Regexes stay ahead of allowed terms, so a marked word can't expose a secret.
  - Telegram can't list a bot's messages, so `chat_log.py` records their IDs for `/clear`. Telegram refuses to delete messages older than 48 hours.
  - Code formatting is saved in `messages.code_entities`, with offsets in UTF-16 code units.
  - Services without a bot token reach the owner through `bot_notices`, which `outbox.py` delivers.
  - `pydantic` is held below 2.13 because aiogram requires it. `verdict_schema()` was compared byte for byte across that downgrade; check it again when changing the version, since the schema is part of what was evaluated.
- **jira_export**: its own container, and the only one with the Jira token. It targets a single project on Jira Cloud, reachable from home without VPN: REST API v3, Basic auth with email and API token, `description` in ADF. Tasks are stored on Atlassian's servers, so the text is the LLM's rephrasing, with no quotes. Links to source messages never go to Jira. Text becomes real only here, right before leaving, so never log it; `client.py` drops Jira's error messages because they can quote the task.

  - Names may reach Jira only as placeholders such as `<PERSON_3>` left in the text and shown to the owner, resolved, before they approve. `<SECRET>` and `<CARD>` block approving because they can't be restored. `<ME>` becomes "я". Code blocks are the exception to "no quotes": they go verbatim except what `Anonymizer.find_secrets` masks, which needs no NER.
  - **Tasks in `locked_statuses` ("Готово", "Тестирование") must never change.** `_write_to_existing` is the only caller of `update_description` and reads the status immediately before writing, because a card's status can be stale. `tests/test_jira.py` enforces that it's the only caller. A link counts as a change, since it also appears on the original, so a new task only mentions a locked one in its text. `check_project` stops the service if a locked status isn't in the project, because a misspelt one would silently stop protecting tasks.
  - Change requests go to the owner, who chooses `append` or `create`. The one exception is a task in `auto_append_statuses` ("Новая"), which nobody has started: the owner already approved the task itself, so the change applies automatically, unless the request holds placeholders or files, which the owner must see before they leave. Appending adds a section and keeps the description, since it may have been edited in Jira. `export_action` is None on drafts approved before export existed, and those go only through `/export_old`.
  - The issue type in use has no labels field, and Jira rejects a field the type doesn't have, so tasks carry no labels. A task without a sprint lands in the backlog, which is not a sprint, so nothing handles sprints. Duplicate protection therefore searches the owner's tasks of the last 3 days for the exact title after a crash (`export_started_at`), waiting 2 minutes for Jira's index. The old `/rest/api/3/search` is gone; `search/jql` returns no fields unless asked.

  - Attachments aren't anonymized, so the review bot lists them and the owner can exclude any. `storage/attachments.py` decides what goes: audio never, stickers and GIFs only when switched on. Only `collector` holds the session, so it downloads through `file_requests` into the `attachments_tmp` volume under a neutral name. Whoever uses a file deletes it, and anything left is removed after an hour. `Reader.download` must stay read-only. A failed upload keeps the task, reports the file, and retries only files, recorded per file in `attachment_uploads`.

A task counts as the user's if it has an explicit tag or name, arrives in a DM, replies to the user's message, is addressed to front-end, or concerns a section listed in `config/scope.yaml`. That file must stay extendable without code changes. The LLM reports which rule applied in `assignment_reason`.

Everything runs on one Windows PC with a GPU. Machine-specific paths come from the Windows user variables `TM_BACKUP_DIR` and `TM_OLLAMA_APP`, not from tracked files. Docker Desktop only runs after sign-in, so the system starts at sign-in through `tm autostart`. Auto-login is deliberately not used: it would hand the database key to anyone who powers the PC on. Analysis must tolerate Ollama being unavailable: messages wait in the queue and collection continues.

`scripts/tm.ps1` is the way in. `tm start` starts `keeper`, copies secrets into in-memory volumes, then starts the services. Anything that isn't in memory, such as environment variables and Compose secrets, lands on the unencrypted Docker disk. `scripts/secret-map.json` says which container sees what, and `tests/test_operations.py` checks it against the compose file. Docker restarting clears the secrets, and services then exit with code 2 until `tm start`. After `tm stop` nothing brings the system back, so a deliberate stop survives reboots and raises no alarm.

The bot can't report its own failure, so two layers watch: services write to `service_states` and `outbox.py` watches them, and `tm watchdog` runs from Task Scheduler for what the bot can't see. Neither sees a powered-off PC; the bot reports the downtime on the next start.

`collector`, `anonymizer`, `analyzer`, `review_bot` and `jira_export` run as separate containers from one image, started by `python -m task_manager run <service>` after the `migrate` job finishes. Docker's health check is `python -m task_manager.health <service>`, which checks the age of a heartbeat file, so a real service must keep refreshing it. It starts in every container every 15 seconds, so that module may import only the standard library: importing the application there once made up most of the system's idle CPU use. `OPENBLAS_NUM_THREADS=1` is set for a similar reason, since numpy's default of a thread per core burned over 30 times the CPU on Natasha's small matrices and finished later. Invalid configuration exits with code 2 and a plain message on stderr.

Containers run as non-root on a read-only filesystem, so only `/data`, `/tmp` and `/attachments` are writable. `config/` is mounted read-only, so anything saved at runtime lives in the database. `me.yaml`, `chats.yaml`, `scope.yaml` and `jira.yaml` name the employer, reveal its business or hold secrets, so they exist only in the encrypted store and reach containers in memory; `_private_or_config` prefers that copy.

The database is SQLite in the named volume `app_data`, not a bind mount, because file locking is unreliable on Windows bind mounts. Several containers write to it, so always create engines through `storage/db.py`, which sets the key, WAL and a busy timeout. The whole file is encrypted with SQLCipher when `DATABASE_KEY` is set, and `REQUIRE_ENCRYPTED_DB` stops a container from creating a plain database. Tests run plain unless they set a key. Use `connect_raw` rather than `sqlite3` for whole-file operations, or the copy won't open. `messages.status` is the work queue. Timestamps are stored in UTC, and SQLite returns them without a timezone. `tests/test_storage.py` fails if models and migrations drift apart.

Migrations run with foreign keys off, and `env.py` then runs `foreign_key_check`. A batch migration drops and recreates a table, and with foreign keys on that drop cascades and silently empties every table pointing at it, as once happened to `draft_messages`. `tests/test_storage.py` covers this on a database with rows, because an empty one can't show it.

Ollama runs natively on Windows with `OLLAMA_NO_CLOUD=1`, listening only on `127.0.0.1`; containers reach it through `host.docker.internal`. A cloud tag can still be pulled as a stub even with cloud disabled, which is why `analyzer/ollama.py` also refuses such names.

`eval/run_eval.py` reports precision and recall per model and prompt, reading only `eval/dataset/`, never the database. Precision matters more than recall, because false positives flood Jira, so a draft on a "maybe mine" label counts as an error while missing one doesn't. A message that's only part of a conversation leading to a task has its own label and counts neither way: the model can't see what comes next, and such messages are meant to merge into one draft. The sample is enriched, so reports also show numbers weighted back to the whole population, plus intervals, because tasks are rare. Examples sharing a message stay in one part of the split, and the final part is meant to run once. Prompts in `config/prompts/` hold made-up examples only. The chosen combination scored 78.9% precision (15/19, interval 57–91%) and 83.3% recall (15/18) on the final part. That doesn't confirm the 85% precision target; the owner accepted it because every draft is reviewed by hand.

## Invariants

These are easy to break by accident:

- **Anonymize before the LLM, even though the LLM is local.** This protects logs, eval datasets and Jira descriptions. Placeholders are substituted back into the draft only after the model responds.
- **Never log raw message text.** `common/log.py` writes only the event name and fields in `ALLOWED_FIELDS`, and drops `%` arguments and exception messages. Log a fixed event name and pass data through `extra`. An f-string event name bypasses this protection.
- **The Telegram session is read-only.** It must never send messages from the user's account (Telegram ToS) or mark anything as read. Sending belongs to the review bot, which uses its own bot token. Only `collector/telegram.py` may import Telethon, and `tests/test_telegram.py` rejects write operations in the source, so don't add one to get a test passing.
- **Handle `FloodWaitError`** in all Telethon calls that hit rate limits. The wrapper translates it to `RateLimitedError`, and `Collector._with_rate_limit` waits and retries, which is safe because every write is idempotent.
- **Nothing reaches Jira without human approval.** A new task always needs it. The only automatic write is a change to a task the owner already approved, while it's in `auto_append_statuses`.
- **No exposed ports.** Services run in one `docker compose` setup.
- **Nothing sensitive at rest in the clear.** The Telegram session grants full access to the account, and the pseudonym map reverses anonymization. The session is a string in the DPAPI store, not a file; the map is in the encrypted database. Never add an environment variable or file holding a secret: the disks aren't encrypted, and this only protects against a stolen PC or leaked files, not against someone using the signed-in account. Backups need the key, which `tm secrets export-recovery` shows once for safekeeping.
- **The repository is public.** Tests, prompts, comments and commit messages may hold only made-up examples: no real names, employer terms, its URL paths or plans, no chat fragments and no machine paths. A word found in real chats goes in `anonymizer_overrides` or the private `scope.yaml`. Database files never go in git.
- **Anonymizer changes need leak tests:** a set of strings containing personal data, with assertions that none of it survives processing.

## Language

Chats are in Russian, so handle Russian case forms in name matching and in the prefilter. Technical terms may still appear in Latin script (`front-end`), so list both spellings as aliases in `config/scope.yaml`.
