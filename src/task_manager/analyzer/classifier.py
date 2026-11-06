"""Asks the model about messages and drafts. Everything it receives is already anonymized."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from task_manager.analyzer.ollama import SEED, OllamaClient, think_setting
from task_manager.analyzer.prompt import (
    format_line,
    load_template,
    render_system,
    render_user,
    template_id,
)
from task_manager.analyzer.thread import Thread, ThreadLine
from task_manager.analyzer.verdict import (
    DraftText,
    FollowUp,
    InvalidVerdictError,
    PromiseVerdict,
    SameTask,
    creates_draft,
    parse_answer,
    parse_verdict,
    verdict_schema,
)
from task_manager.common.config import ScopeConfig

PROMISE_PROMPT = "promise_v3.md"
MERGE_PROMPT = "merge_v2.md"
REWRITE_PROMPT = "rewrite_v1.md"
FOLLOWUP_PROMPT = "followup_v1.md"
EDIT_PROMPT = "edit_v1.md"
PROMISE_REASON = "promise"
# The first attempt uses the plain seed, so it gets the same answer the evaluation measured.
ATTEMPTS = 3
# Enough to recognize a conversation, without letting a long draft crowd out the others.
MERGE_LINES = 6


class AnswerError(Exception):
    """No usable answer after retrying. Its message is a fixed code, never the model's text."""


@dataclass(frozen=True)
class Decision:
    creates_draft: bool
    is_task: bool
    assigned_to_me: bool
    reason: str
    confidence: float
    title: str
    description: str
    deadline: str | None
    priority: str
    evidence: tuple[int, ...]
    prompt_id: str


@dataclass(frozen=True)
class DraftSummary:
    id: int
    title: str
    description: str
    lines: tuple[ThreadLine, ...] = ()


def _format_lines(lines: Iterable[ThreadLine]) -> str:
    return "\n".join(format_line(line) for line in lines)


def _require_text(creates: bool, title: str, description: str) -> None:
    if creates and not (title.strip() and description.strip()):
        raise InvalidVerdictError("empty draft")


class Classifier:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        prompts_dir: Path,
        task_prompt: str,
        scope: ScopeConfig,
    ) -> None:
        self.model = model
        self._client = client
        self._think: bool | str | None = None
        self._think_known = False
        self._task = self._load(prompts_dir / task_prompt, scope)
        self._promise = self._load(prompts_dir / PROMISE_PROMPT, scope)
        self._merge = self._load(prompts_dir / MERGE_PROMPT, scope)
        self._rewrite = self._load(prompts_dir / REWRITE_PROMPT, scope)
        self._followup = self._load(prompts_dir / FOLLOWUP_PROMPT, scope)
        self._edit = self._load(prompts_dir / EDIT_PROMPT, scope)

    @staticmethod
    def _load(path: Path, scope: ScopeConfig) -> tuple[str, str]:
        template = load_template(path)
        return render_system(template, scope), template_id(path, template)

    @property
    def promise_prompt_id(self) -> str:
        return self._promise[1]

    async def _ask[AnswerT](
        self, system: str, user: str, schema: dict[str, Any], parse: Callable[[str], AnswerT]
    ) -> AnswerT:
        """Retries a bad answer. The same seed would only repeat it, so each attempt changes it.

        OllamaUnavailableError passes through: an outage says nothing about the message.
        """
        if not self._think_known:
            self._think = think_setting(await self._client.show(self.model))
            self._think_known = True
        failure = "schema"
        for attempt in range(ATTEMPTS):
            result = await self._client.chat(
                self.model, system, user, schema, self._think, seed=SEED + attempt
            )
            if result.truncated_input:
                # The start of the prompt was dropped, and a retry wouldn't make it fit.
                raise AnswerError("input_too_long")
            if result.truncated_output:
                failure = "output_too_long"
                continue
            try:
                return parse(result.content)
            except InvalidVerdictError:
                failure = "schema"
        raise AnswerError(failure)

    async def classify(self, thread: Thread, own: bool) -> Decision:
        return await (self._classify_own(thread) if own else self._classify_incoming(thread))

    async def _classify_incoming(self, thread: Thread) -> Decision:
        system, prompt_id = self._task

        def parse(content: str) -> Decision:
            verdict = parse_verdict(content)
            creates = creates_draft(verdict)
            _require_text(creates, verdict.title, verdict.description)
            return Decision(
                creates_draft=creates,
                is_task=verdict.is_task,
                assigned_to_me=verdict.assigned_to_me,
                reason=verdict.assignment_reason.value,
                confidence=verdict.confidence,
                title=verdict.title,
                description=verdict.description,
                deadline=verdict.deadline,
                priority=verdict.priority.value,
                evidence=tuple(verdict.evidence),
                prompt_id=prompt_id,
            )

        return await self._ask(system, render_user(thread), verdict_schema(), parse)

    async def _classify_own(self, thread: Thread) -> Decision:
        system, prompt_id = self._promise

        def parse(content: str) -> Decision:
            verdict = parse_answer(PromiseVerdict, content)
            _require_text(verdict.is_promise, verdict.title, verdict.description)
            return Decision(
                creates_draft=verdict.is_promise,
                is_task=verdict.is_promise,
                assigned_to_me=verdict.is_promise,
                reason=PROMISE_REASON,
                confidence=verdict.confidence,
                title=verdict.title,
                description=verdict.description,
                deadline=verdict.deadline,
                priority=verdict.priority.value,
                evidence=tuple(verdict.evidence),
                prompt_id=prompt_id,
            )

        return await self._ask(
            system, render_user(thread), PromiseVerdict.model_json_schema(), parse
        )

    async def same_task(
        self,
        title: str,
        description: str,
        drafts: Iterable[DraftSummary],
        context: Iterable[ThreadLine] = (),
    ) -> int | None:
        """Which open draft this is part of. Unsure means None: two drafts beat a lost task.

        The messages decide: titles of two fragments of one conversation often look unrelated.
        """
        system, _ = self._merge
        candidates = list(drafts)
        known = {draft.id for draft in candidates}
        listed = "\n\n".join(
            f"Черновик {draft.id}\nНазвание: {draft.title}\nОписание: {draft.description}\n"
            f"Сообщения:\n{_format_lines(draft.lines[-MERGE_LINES:])}"
            for draft in candidates
        )
        user = f"Открытые черновики:\n\n{listed}\n\nНовая задача\nНазвание: {title}\n"
        user += f"Описание: {description}\nСообщения:\n{_format_lines(context)}"
        try:
            answer = await self._ask(
                system,
                user,
                SameTask.model_json_schema(),
                lambda content: parse_answer(SameTask, content),
            )
        except AnswerError:
            return None
        return answer.same_draft if answer.same_draft in known else None

    async def follow_up(
        self,
        status: str,
        draft: DraftSummary,
        rejection: str | None,
        context: Iterable[ThreadLine],
    ) -> FollowUp:
        """Does a message about a decided draft need a new ticket? Unsure means yes.

        A missing task is worse than a ticket the owner rejects, so a failed answer means yes too,
        with empty text for the caller to fill in.
        """
        system, _ = self._followup

        def parse(content: str) -> FollowUp:
            answer = parse_answer(FollowUp, content)
            _require_text(answer.needs_ticket, answer.title, answer.description)
            return answer

        names = {"approved": "принята", "exported": "принята", "rejected": "отклонена"}
        parts = [f"Статус задачи: {names[status]}"]
        if rejection:
            parts.append(f"Причина отклонения: {rejection}")
        parts += [
            f"Название: {draft.title}",
            f"Описание: {draft.description}",
            f"Сообщения задачи:\n{_format_lines(draft.lines[-MERGE_LINES:])}",
            "",
            f"Новое сообщение с контекстом (оно указано последним):\n{_format_lines(context)}",
        ]
        try:
            return await self._ask(system, "\n".join(parts), FollowUp.model_json_schema(), parse)
        except AnswerError:
            return FollowUp(needs_ticket=True, title="", description="")

    async def edit_text(
        self, lines: Iterable[ThreadLine], title: str, description: str, instruction: str
    ) -> DraftText:
        """The owner's change to a draft's text. All of the input must already be anonymized."""
        system, _ = self._edit

        def parse(content: str) -> DraftText:
            text = parse_answer(DraftText, content)
            _require_text(True, text.title, text.description)
            return text

        user = (
            f"Сообщения задачи:\n{_format_lines(lines)}\n\n"
            f"Текущее название: {title}\nТекущее описание: {description}\n\n"
            f"Просьба владельца: {instruction}"
        )
        return await self._ask(system, user, DraftText.model_json_schema(), parse)

    async def rewrite(self, lines: Iterable[ThreadLine]) -> DraftText | None:
        """One title and description for all of a draft's messages, or None if none came back."""
        system, _ = self._rewrite

        def parse(content: str) -> DraftText:
            text = parse_answer(DraftText, content)
            _require_text(True, text.title, text.description)
            return text

        user = "Сообщения задачи:\n\n" + "\n\n".join(format_line(line) for line in lines)
        try:
            return await self._ask(system, user, DraftText.model_json_schema(), parse)
        except AnswerError:
            return None
