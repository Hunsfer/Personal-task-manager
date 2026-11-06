"""Removes what can't be a task. Missing a task is worse than asking the model about small talk."""

from collections.abc import Iterable

from task_manager.analyzer.thread import ThreadLine

NO_TEXT = "no_text"
NO_WORDS = "no_words"
PHRASE = "phrase"


def _normalize(text: str) -> str:
    """Lowercase words only: "Ок!" and "ок 👍" are the same reply."""
    kept = "".join(char if char.isalnum() else " " for char in text.lower())
    return " ".join(kept.split())


class Prefilter:
    def __init__(self, noise_phrases: Iterable[str]) -> None:
        self._phrases = frozenset(_normalize(phrase) for phrase in noise_phrases)

    def noise_reason(self, line: ThreadLine) -> str | None:
        """Why the model needn't see this message, or None if it should."""
        if not line.text.strip():
            return NO_TEXT
        if line.files:
            return None
        normalized = _normalize(line.text)
        if not normalized:
            return NO_WORDS
        return PHRASE if normalized in self._phrases else None
