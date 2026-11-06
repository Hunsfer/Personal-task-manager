"""Cheap signs that a message may be addressed to the owner. They only choose what to look at."""

import re
from collections.abc import Iterable

from task_manager.analyzer.thread import Thread
from task_manager.anonymizer.engine import is_imperative, lemmas
from task_manager.common.config import ScopeConfig
from task_manager.storage.models import ChatKind

ME_TOKEN = "<ME>"
NO_SIGNAL = "none"
# The first matching sign names the group a message is sampled from, so strongest come first.
SIGNAL_ORDER = ("mention", "reply", "dm", "section", "role", "collective", "request")

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_COLLECTIVE = frozenset({"мы", "нас", "нам", "нами"})
_REQUEST_WORDS = re.compile(r"\b(нужно|надо|необходимо|просьба|прошу|дедлайн|срочно)\b", re.I)


class TermMatcher:
    """Finds terms in any case form. Paths such as /blog match as plain text."""

    def __init__(self, terms: Iterable[str]) -> None:
        self._paths: list[str] = []
        self._phrases: list[list[frozenset[str]]] = []
        for term in terms:
            if term.startswith("/"):
                self._paths.append(term.lower())
            else:
                self._phrases.append([lemmas(word) for word in _WORD.findall(term)])

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        if any(path in lowered for path in self._paths):
            return True
        words = [lemmas(word) for word in _WORD.findall(text)]
        for phrase in self._phrases:
            size = len(phrase)
            for start in range(len(words) - size + 1):
                window = words[start : start + size]
                if size and all(a & b for a, b in zip(phrase, window, strict=True)):
                    return True
        return False


class SignalDetector:
    def __init__(self, scope: ScopeConfig) -> None:
        self._role = TermMatcher(scope.role)
        self._sections = TermMatcher(
            term for section in scope.sections for term in (section.name, *section.aliases)
        )

    def detect(self, thread: Thread) -> list[str]:
        """Which signs the message shows, strongest first."""
        text = thread.target.text
        words = [word.lower() for word in _WORD.findall(text)]
        found = {
            "mention": thread.mentions_me or ME_TOKEN in text,
            "reply": any(
                line.msg_id == thread.target.reply_to and line.sender == ME_TOKEN
                for line in thread.reply_chain
            ),
            "dm": thread.chat_kind == ChatKind.PRIVATE,
            "section": self._sections.matches(text),
            "role": self._role.matches(text),
            "collective": any(word in _COLLECTIVE for word in words),
            "request": bool(_REQUEST_WORDS.search(text)) or any(map(is_imperative, words)),
        }
        return [name for name in SIGNAL_ORDER if found[name]]
