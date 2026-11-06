"""The model's answer about one message: the strict schema and how it is checked."""

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from task_manager.anonymizer.store import PLACEHOLDER

DateString = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class AssignmentReason(StrEnum):
    MENTION = "mention"
    DM = "dm"
    REPLY = "reply"
    ROLE = "role"
    SECTION = "section"
    # "We" or "us": doesn't prove the owner is addressed, so it never makes a draft on its own.
    COLLECTIVE = "collective"
    NONE = "none"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_task: bool
    assigned_to_me: bool
    assignment_reason: AssignmentReason
    confidence: float = Field(ge=0, le=1)
    title: str
    description: str
    deadline: DateString | None
    priority: Priority
    evidence: list[int]


class PromiseVerdict(BaseModel):
    """The model's answer about the owner's own message."""

    model_config = ConfigDict(extra="forbid")

    is_promise: bool
    confidence: float = Field(ge=0, le=1)
    title: str
    description: str
    deadline: DateString | None
    priority: Priority
    evidence: list[int]


class SameTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    same_draft: int | None


class DraftText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str


class FollowUp(BaseModel):
    """Whether a message about an already decided draft needs a new ticket."""

    model_config = ConfigDict(extra="forbid")

    needs_ticket: bool
    title: str
    description: str


def creates_draft(verdict: TaskVerdict) -> bool:
    """Binary: a task for the owner or nothing. "We" alone doesn't prove the owner is meant."""
    return (
        verdict.is_task
        and verdict.assigned_to_me
        and verdict.assignment_reason not in (AssignmentReason.COLLECTIVE, AssignmentReason.NONE)
    )


def has_placeholder(text: str) -> bool:
    """A person or secret left in a draft's text, which must be fixed before it can leave."""
    return bool(PLACEHOLDER.search(text)) or "<SECRET>" in text or "<CARD>" in text


class InvalidVerdictError(Exception):
    """The answer doesn't match the schema. Its message never contains the model's text."""


def verdict_schema() -> dict[str, Any]:
    """JSON schema for Ollama's `format` parameter."""
    return TaskVerdict.model_json_schema()


def parse_answer[AnswerT: BaseModel](model: type[AnswerT], content: str) -> AnswerT:
    """Parse strictly: a wrong answer is an error rather than something to repair silently."""
    try:
        return model.model_validate_json(content)
    except ValidationError as error:
        fields = sorted({str(item["loc"][0]) for item in error.errors() if item["loc"]})
        raise InvalidVerdictError(f"invalid fields: {fields}") from None


def parse_verdict(content: str) -> TaskVerdict:
    return parse_answer(TaskVerdict, content)
