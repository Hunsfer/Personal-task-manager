"""Anonymizes what the owner types, since it may reach the model or a draft."""

import re

from sqlalchemy.ext.asyncio import AsyncSession

from task_manager.anonymizer import store
from task_manager.anonymizer.engine import Anonymizer, NerModel
from task_manager.anonymizer.runner import allowed_terms, current_rules
from task_manager.common.config import AnonymizerConfig, MeConfig, ScopeConfig

# A placeholder the owner typed is already safe, and the engine must not take it for a name.
_KEEP = re.compile(r"(<(?:ME|SECRET|CARD|(?:PERSON|ORG|LOC|PHONE|EMAIL|URL|IP|ID)_\d+)>)")


class Names:
    def __init__(
        self, ner: NerModel, me: MeConfig, scope: ScopeConfig, config: AnonymizerConfig
    ) -> None:
        self._ner = ner
        self._me = me
        self._scope = scope
        self._config = config
        self._rules = ""
        self._engine: Anonymizer | None = None

    async def _current(self, session: AsyncSession) -> Anonymizer:
        people = await store.load_people(session)
        overrides = await store.load_overrides(session)
        rules = current_rules(people, self._me, self._scope, self._config, overrides)
        if self._engine is None or rules != self._rules:
            self._rules = rules
            self._engine = Anonymizer(
                self._ner,
                self._me.names,
                allowed_terms(self._scope, self._config, overrides),
                self._config.names,
                people,
                [(item.name, item.aliases) for item in self._me.organizations],
                self._me.same_people,
                self._me.secrets,
            )
        return self._engine

    async def anonymize(self, session: AsyncSession, text: str) -> str:
        """The caller commits: a name seen for the first time gets a new placeholder."""
        engine = await self._current(session)
        pieces = _KEEP.split(text)
        for index in range(0, len(pieces), 2):
            pieces[index] = await store.render(
                session, pieces[index], engine.find_spans(pieces[index])
            )
        return "".join(pieces)
