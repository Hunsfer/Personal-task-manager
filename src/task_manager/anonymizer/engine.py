"""Find personal data in text. Pure logic: no database, and no access to the pseudonym map.

Layers run in order, and a later layer can't touch what an earlier one claimed:

1. Regular expressions mask contacts, links and secrets. This comes first because an allowed word
   inside a link or a password must not keep the rest of that value readable.
2. Allowed terms are protected from the name and organization detection below.
3. The user's own names become <ME>.
4. Names of known people are found by every possible reading of each word, so lowercase names
   and tricky case forms ("Кате" is first read as "кат") still match.
5. The NER model finds unknown people, organizations and places.
6. A morphological safety net catches names that the model misses.

When in doubt, a word is masked: a stray placeholder costs less than a leaked name.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache

import pymorphy3
from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter

NAME_TAGS = ("Name", "Surn", "Patr")
PLACE_OR_ORG_TAGS = ("Geox", "Orgn")
# Keeps "Кате" (0.25) and "Петрову" (0.20), but not "Баги" (0.08) or "Марта" (0.02).
# "Роман" (0.07) and "Слава" (0.05) fall below it too, so they need a list or the NER model.
CAPITALIZED_NAME_MIN_SCORE = 0.1
# The morphology guesses a tag for words outside its dictionary, usually a name. Only a surname
# with a very typical ending is trusted: "Хабибуллин" (1.0) but not "Ньюанс" or "Рамштайн".
GUESSED_SURNAME_MIN_SCORE = 0.7
MIN_LOWERCASE_NAME_LENGTH = 3

# Common short forms, so "Дима" in a message matches a person listed as "Дмитрий".
DIMINUTIVES = {
    "александр": ("саша", "саня", "шура"),
    "александра": ("саша", "саня", "шура"),
    "алексей": ("алеша", "леша"),
    "анастасия": ("настя",),
    "анна": ("аня",),
    "антон": ("тоша",),
    "валентина": ("валя",),
    "василий": ("вася",),
    "виктор": ("витя",),
    "виктория": ("вика",),
    "владимир": ("вова", "володя"),
    "дарья": ("даша",),
    "денис": ("дэн",),
    "дмитрий": ("дима", "митя"),
    "евгений": ("женя",),
    "евгения": ("женя",),
    "екатерина": ("катя",),
    "елена": ("лена",),
    "иван": ("ваня",),
    "ирина": ("ира",),
    "константин": ("костя",),
    "ксения": ("ксюша",),
    "мария": ("маша",),
    "максим": ("макс",),
    "михаил": ("миша",),
    "наталья": ("наташа",),
    "николай": ("коля",),
    "ольга": ("оля",),
    "павел": ("паша",),
    "светлана": ("света",),
    "сергей": ("сережа",),
    "татьяна": ("таня",),
    "юлия": ("юля",),
    "юрий": ("юра",),
}

ME = "ME"
PERSON = "PERSON"
ORG = "ORG"
LOC = "LOC"
PHONE = "PHONE"
EMAIL = "EMAIL"
URL = "URL"
IP = "IP"
ID = "ID"
SECRET = "SECRET"
CARD = "CARD"
# These have no entry in the pseudonym map: <ME> is known, and secrets must not be stored at all.
FIXED_KINDS = frozenset({ME, SECRET, CARD})

_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_CYRILLIC = re.compile(r"^[а-яёА-ЯЁ-]+$")
_LATIN = re.compile(r"^[A-Za-z-]+$")
_URL_CHARS = r"[^\s<>\"']"
_SCHEME_URL = re.compile(rf"\b(?:https?|ftp)://{_URL_CHARS}+|\bwww\.{_URL_CHARS}+", re.I)
_BARE_URL = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:ru|com|net|org|io|рф|su|dev|app|me|info|biz|ua|by|kz|co|ai)\b"
    rf"(?:/{_URL_CHARS}*)?",
    re.I,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?:\bpass(?:word)?|\bpwd|\btoken|\bapi[_-]?key|\bsecret|пароль|токен|ключ|секрет)\w*"
    r"\s*[:=]\s*(\S+)",
    re.I,
)
# curl -u login:password, and "логин: login / password".
_CREDENTIALS = (
    re.compile(r"(?:(?<=\s)|^)(?:-u|--user)[ =]+(\S+:\S+)"),
    re.compile(
        r"(?:логин|login)(?:\s*/\s*(?:пароль|password|pass))?\s*[:=]\s*(\S+?)(?:\s*[/|:]\s*(\S+))?(?=\s|$)",
        re.I,
    ),
)
_TOKENS = (
    re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]*"),
    re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{30,}\b"),
    re.compile(r"\b(?:sk|pk)[-_][-_A-Za-z0-9]{8,}\b"),
    re.compile(r"\b(?:ghp|gho|github_pat|glpat|xox[abpr]|ATATT|AKIA)[-_A-Za-z0-9]{8,}\b"),
)
_LONG_TOKEN = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")
_MENTION = re.compile(r"(?<![\w@/])@([A-Za-z][A-Za-z0-9_]{3,31})\b")
_PHONES = (
    re.compile(r"(?<![\d+])(?:\+7|8|7)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"),
    re.compile(r"(?<![\d+])\+\d{1,3}[\s\-()]*\d[\d\s\-()]{7,}\d(?!\d)"),
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6 = re.compile(r"\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b", re.I)
_IDS = (
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[ -]\d{2}(?!\d)"),
    re.compile(r"(?<!\d)\d{4} \d{6}(?!\d)"),
    re.compile(r"(?<!\d)\d{10,}(?!\d)"),
)

_analyzer = pymorphy3.MorphAnalyzer()


def _normalize(word: str) -> str:
    return word.lower().replace("ё", "е")


@lru_cache(maxsize=50_000)
def _parses(word: str) -> tuple:
    return tuple(_analyzer.parse(word))


@lru_cache(maxsize=50_000)
def lemmas(word: str) -> frozenset[str]:
    """Every possible base form of a word, plus the word itself."""
    forms = {_normalize(parse.normal_form) for parse in _parses(word)}
    forms.add(_normalize(word))
    return frozenset(forms)


def _is_name(parse) -> bool:
    return any(tag in parse.tag for tag in NAME_TAGS)


def name_lemma(word: str) -> str:
    """The base form as a name if the word can be one, so "Ивану" gives "иван"."""
    parses = _parses(word)
    best = next((parse for parse in parses if _is_name(parse)), parses[0])
    return _normalize(best.normal_form)


def looks_like_name(word: str) -> bool:
    """The safety net's rule. Trusts the morphology's dictionary, not its guesses.

    Chat slang such as "скроллбары" isn't in the dictionary, and the morphology then tends to guess
    that it's a name. A lowercase word counts only if a name is its likeliest reading.
    """
    if not _CYRILLIC.match(word) or len(word) < 2:
        return False
    # Compound names are very rare here, and compounds such as "мини-попап" are common.
    if "-" in word:
        return False
    parses = _parses(word)
    if not _analyzer.word_is_known(word):
        return (
            word[0].isupper()
            and not word.isupper()
            and "Surn" in parses[0].tag
            and parses[0].score >= GUESSED_SURNAME_MIN_SCORE
        )
    if word[0].isupper():
        return any(
            _is_name(parse) and parse.score >= CAPITALIZED_NAME_MIN_SCORE for parse in parses
        )
    return len(word) >= MIN_LOWERCASE_NAME_LENGTH and _is_name(parses[0])


def _plausible_person_word(word: str, starts_sentence: bool) -> bool:
    """Check the NER model's guess: it often takes a sentence's first word for a name."""
    if looks_like_name(word):
        return True
    if word[0].islower() or (len(word) > 1 and word.isupper()) or "-" in word:
        return False
    if _analyzer.word_is_known(word):
        return any(_is_name(parse) for parse in _parses(word))
    # An unfamiliar capitalized word is only telling in the middle of a sentence.
    return not starts_sentence


def _plausible_place_or_org(word: str, starts_sentence: bool) -> bool:
    # All-caps words are abbreviations or constants, and a lone Latin word in a developers' chat
    # is nearly always a tool. Real organizations with such names go in the owner's list.
    if word[0].islower() or word.isupper() or _LATIN.match(word):
        return False
    if not _analyzer.word_is_known(word):
        return not starts_sentence
    parses = _parses(word)
    if any(tag in parse.tag for parse in parses for tag in PLACE_OR_ORG_TAGS):
        return True
    return parses[0].tag.POS == "NOUN" and not starts_sentence


def is_plain_verb(word: str) -> bool:
    """The NER model sometimes pulls a verb into a name, as in "Передай Глебу".

    Such verbs state the task itself, so they must stay readable for the LLM.
    """
    parses = _parses(word)
    return parses[0].tag.POS in ("VERB", "INFN") and not any(_is_name(p) for p in parses)


def is_imperative(word: str) -> bool:
    return "impr" in _parses(word)[0].tag


def luhn_valid(number: str) -> bool:
    digits = [int(char) for char in number if char.isdigit()]
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


@dataclass(frozen=True)
class KnownPerson:
    user_id: int
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None
    is_me: bool = False

    @property
    def display_name(self) -> str:
        parts = [part for part in (self.first_name, self.last_name) if part]
        return " ".join(parts) or self.username or f"user:{self.user_id}"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str
    key: str = ""
    value: str = ""


class NerModel:
    """Natasha's NER. Only this part is used: its morphology doesn't work on Python 3.12."""

    def __init__(self) -> None:
        self._segmenter = Segmenter()
        self._tagger = NewsNERTagger(NewsEmbedding())

    def find(self, text: str) -> list[tuple[int, int, str]]:
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._tagger)
        return [(span.start, span.stop, span.type) for span in doc.spans]


class _Claims:
    """Character ranges that are already decided, either masked or protected."""

    def __init__(self, size: int) -> None:
        self._taken = bytearray(size)
        self.spans: list[Span] = []

    def is_free(self, start: int, end: int) -> bool:
        return not any(self._taken[start:end])

    def protect(self, start: int, end: int) -> None:
        self._taken[start:end] = b"\x01" * (end - start)

    def add(self, span: Span) -> bool:
        if not self.is_free(span.start, span.end):
            return False
        self.protect(span.start, span.end)
        self.spans.append(span)
        return True


class Anonymizer:
    def __init__(
        self,
        ner: NerModel,
        me_names: Iterable[str],
        allowed_terms: Iterable[str],
        extra_names: Iterable[str] = (),
        people: Iterable[KnownPerson] = (),
        organizations: Iterable[tuple[str, Iterable[str]]] = (),
        same_people: Iterable[Iterable[str]] = (),
        secrets: Iterable[str] = (),
    ) -> None:
        self._ner = ner
        self._secret_words = [
            re.compile(rf"(?<![\w]){re.escape(secret)}(?![\w])", re.I) for secret in secrets
        ]
        self._organizations = [
            (name, _WORD.findall(alias))
            for name, aliases in organizations
            for alias in (name, *aliases)
        ]
        self._me_lemmas = {_normalize(_parses(name)[0].normal_form) for name in me_names}
        self._me_lemmas |= {_normalize(name) for name in me_names}
        self._extra_names = {_normalize(_parses(name)[0].normal_form) for name in extra_names}
        self._extra_names |= {_normalize(name) for name in extra_names}

        self._paths: list[str] = []
        self._terms: list[list[str]] = []
        for term in allowed_terms:
            if term.startswith("/"):
                self._paths.append(term)
            else:
                self._terms.append([match.group() for match in _WORD.finditer(term)])
        self._paths.sort(key=len, reverse=True)

        self._me_usernames: set[str] = set()
        self._usernames: dict[str, KnownPerson] = {}
        self._name_owners: dict[str, list[KnownPerson]] = {}
        for person in people:
            if person.username:
                if person.is_me:
                    self._me_usernames.add(person.username.lower())
                else:
                    self._usernames[person.username.lower()] = person
            if person.is_me:
                continue
            for part in (person.first_name, person.last_name):
                for match in _WORD.finditer(part or ""):
                    lemma = name_lemma(match.group())
                    for form in (lemma, *DIMINUTIVES.get(lemma, ())):
                        owners = self._name_owners.setdefault(form, [])
                        if person not in owners:
                            owners.append(person)

        # Spellings the owner says belong to one person: "@tag" and how people write the name.
        self._group_usernames: dict[str, tuple[str, str]] = {}
        self._group_names: dict[str, tuple[str, str]] = {}
        for group in same_people:
            items = list(group)
            tags = [item[1:].lower() for item in items if item.startswith("@")]
            spoken = [item for item in items if not item.startswith("@")]
            known = next((self._usernames[tag] for tag in tags if tag in self._usernames), None)
            identity = (
                (f"user:{known.user_id}", known.display_name)
                if known
                else (f"group:{items[0].lower()}", spoken[0] if spoken else items[0])
            )
            self._group_usernames.update(dict.fromkeys(tags, identity))
            for name in spoken:
                self._group_names[name_lemma(name)] = identity
                self._group_names[_normalize(name)] = identity

    def find_secrets(self, text: str) -> list[Span]:
        """Passwords, card numbers and URL query strings, for text that otherwise stays as it is.

        Used on code going to Jira: masking names there would break it. This needs no NER.
        """
        claims = _Claims(len(text))
        self._mask_patterns(text, claims)
        found = []
        for span in claims.spans:
            if span.kind in (SECRET, CARD):
                found.append(span)
            elif "?" in text[span.start : span.end]:
                # A link in a chat keeps its readable part. Its query may hold a token.
                found.append(Span(text.index("?", span.start, span.end), span.end, SECRET))
        return found

    def find_spans(self, text: str) -> list[Span]:
        """Return non-overlapping spans to replace, in text order."""
        claims = _Claims(len(text))
        words = [(m.start(), m.end(), m.group()) for m in _WORD.finditer(text)]

        self._mask_patterns(text, claims)
        self._mask_organizations(words, claims)
        self._protect_terms(text, words, claims)
        self._mask_me(words, claims)
        self._mask_known_people(words, claims)
        self._mask_ner(text, words, claims)
        self._mask_likely_names(words, claims)
        return sorted(claims.spans, key=lambda span: span.start)

    @staticmethod
    def _occurrences(
        words: list[tuple[int, int, str]], term: list[str]
    ) -> Iterator[tuple[int, int]]:
        """Where a possibly multi-word term occurs, in any case form."""
        term_lemmas = [lemmas(word) for word in term]
        for index in range(len(words) - len(term) + 1):
            window = words[index : index + len(term)]
            if term and all(
                lemmas(word[2]) & term_lemma
                for word, term_lemma in zip(window, term_lemmas, strict=True)
            ):
                yield window[0][0], window[-1][1]

    def _mask_organizations(self, words: list[tuple[int, int, str]], claims: _Claims) -> None:
        """Organizations named by the owner: every spelling gets the same placeholder."""
        for name, alias in self._organizations:
            for start, end in self._occurrences(words, alias):
                claims.add(Span(start, end, ORG, f"org:{_normalize(name)}", name))

    def _protect_terms(self, text: str, words: list[tuple[int, int, str]], claims: _Claims) -> None:
        for path in self._paths:
            for match in re.finditer(re.escape(path) + r"(?=$|[/?#\s.,;:!)])", text, re.I):
                claims.protect(match.start(), match.end())

        for term in self._terms:
            for start, end in self._occurrences(words, term):
                claims.protect(start, end)

    def _mask_patterns(self, text: str, claims: _Claims) -> None:
        for match in _SCHEME_URL.finditer(text):
            self._mask_url(match, claims)
        for match in _EMAIL.finditer(text):
            claims.add(Span(*match.span(), EMAIL, match.group().lower(), match.group()))
        for match in _BARE_URL.finditer(text):
            self._mask_url(match, claims)

        for pattern in self._secret_words:
            for match in pattern.finditer(text):
                claims.add(Span(*match.span(), SECRET))
        for pattern in _CREDENTIALS:
            for match in pattern.finditer(text):
                for group in range(1, pattern.groups + 1):
                    if match.group(group):
                        claims.add(Span(*match.span(group), SECRET))
        for match in _SECRET_ASSIGNMENT.finditer(text):
            claims.add(Span(*match.span(1), SECRET))
        for pattern in _TOKENS:
            for match in pattern.finditer(text):
                claims.add(Span(*match.span(), SECRET))
        for match in _LONG_TOKEN.finditer(text):
            token = match.group()
            if any(char.isdigit() for char in token) and any(char.isalpha() for char in token):
                claims.add(Span(*match.span(), SECRET))

        for match in _MENTION.finditer(text):
            self._mask_mention(match, claims)
        for pattern in _PHONES:
            for match in pattern.finditer(text):
                digits = re.sub(r"\D", "", match.group())
                if len(digits) == 11 and digits[0] == "8":
                    digits = "7" + digits[1:]
                claims.add(Span(*match.span(), PHONE, digits, match.group()))
        for match in _CARD.finditer(text):
            if luhn_valid(match.group()):
                claims.add(Span(*match.span(), CARD))
        for pattern in (_IPV4, _IPV6):
            for match in pattern.finditer(text):
                claims.add(Span(*match.span(), IP, match.group().lower(), match.group()))
        for pattern in _IDS:
            for match in pattern.finditer(text):
                claims.add(Span(*match.span(), ID, re.sub(r"\D", "", match.group()), match.group()))

    def _mask_url(self, match: re.Match[str], claims: _Claims) -> None:
        """Mask a link, keeping an allowed path such as /blog readable."""
        raw = match.group().rstrip(".,;:!?)»")
        start = match.start()
        for path in self._paths:
            found = re.search(re.escape(path) + r"(?=$|[/?#])", raw, re.I)
            if found and "." in raw[: found.start()]:
                self._add_url_part(start, raw[: found.start()], claims)
                claims.protect(start + found.start(), start + found.end())
                rest = raw[found.end() :]
                if rest:
                    self._add_url_part(start + found.end(), rest, claims)
                return
        self._add_url_part(start, raw, claims)

    @staticmethod
    def _add_url_part(start: int, part: str, claims: _Claims) -> None:
        # Query strings and logins can hold secrets, so they never reach the pseudonym map.
        value = re.sub(r"(?<=://)[^/@]*@", "", re.split(r"[?#]", part, maxsplit=1)[0])
        claims.add(Span(start, start + len(part), URL, value.lower() or part.lower(), value))

    def _mask_mention(self, match: re.Match[str], claims: _Claims) -> None:
        username = match.group(1).lower()
        if username in self._me_usernames:
            claims.add(Span(*match.span(), ME))
        elif username in self._group_usernames:
            key, value = self._group_usernames[username]
            claims.add(Span(*match.span(), PERSON, key, value))
        elif username in self._usernames:
            person = self._usernames[username]
            key = f"user:{person.user_id}"
            claims.add(Span(*match.span(), PERSON, key, person.display_name))
        else:
            claims.add(Span(*match.span(), PERSON, f"username:{username}", match.group()))

    def _mask_me(self, words: list[tuple[int, int, str]], claims: _Claims) -> None:
        for start, end, word in words:
            if claims.is_free(start, end) and lemmas(word) & self._me_lemmas:
                claims.add(Span(start, end, ME))

    def _mask_known_people(self, words: list[tuple[int, int, str]], claims: _Claims) -> None:
        for start, end, word in words:
            if not claims.is_free(start, end):
                continue
            if word[0].islower() and len(word) < MIN_LOWERCASE_NAME_LENGTH:
                continue
            word_lemmas = lemmas(word)
            group = next(
                (self._group_names[lemma] for lemma in word_lemmas if lemma in self._group_names),
                None,
            )
            if group:
                claims.add(Span(start, end, PERSON, *group))
                continue
            owners = {
                person.user_id: person
                for lemma in word_lemmas
                for person in self._name_owners.get(lemma, ())
            }
            if len(owners) == 1:
                person = next(iter(owners.values()))
                key = f"user:{person.user_id}"
                claims.add(Span(start, end, PERSON, key, person.display_name))
            elif owners or word_lemmas & self._extra_names:
                lemma = name_lemma(word)
                claims.add(Span(start, end, PERSON, f"name:{lemma}", word))

    def _mask_ner(self, text: str, words: list[tuple[int, int, str]], claims: _Claims) -> None:
        kinds = {"PER": PERSON, "ORG": ORG, "LOC": LOC}
        for start, stop, entity in self._ner.find(text):
            kind = kinds.get(entity)
            if kind is None:
                continue
            for group in self._free_groups(start, stop, words, claims):
                if not self._entity_is_plausible(text, group, kind):
                    continue
                first, last = group[0], group[-1]
                key = " ".join(name_lemma(word[2]) for word in group)
                prefix = "name" if kind == PERSON else kind.lower()
                claims.add(
                    Span(first[0], last[1], kind, f"{prefix}:{key}", text[first[0] : last[1]])
                )

    @staticmethod
    def _entity_is_plausible(text: str, group: list[tuple[int, int, str]], kind: str) -> bool:
        """The NER model has many false hits on chat text, so check them before masking."""
        words = [word[2] for word in group]
        if len(words) > 1:
            if kind == PERSON:
                return any(_plausible_person_word(word, False) for word in words)
            return any(word[0].isupper() for word in words)
        before = text[: group[0][0]].rstrip()
        # After a dash, bullet, quote or emoji a capital letter says nothing about a name either.
        starts_sentence = not before or not (before[-1].isalnum() or before[-1] == ",")
        if kind == PERSON:
            return _plausible_person_word(words[0], starts_sentence)
        return _plausible_place_or_org(words[0], starts_sentence)

    @staticmethod
    def _free_groups(
        start: int, stop: int, words: list[tuple[int, int, str]], claims: _Claims
    ) -> Iterator[list[tuple[int, int, str]]]:
        """Runs of still-free words inside an entity, so a protected word in it stays readable."""
        group: list[tuple[int, int, str]] = []
        for word in words:
            if word[0] >= stop or word[1] <= start:
                continue
            if claims.is_free(word[0], word[1]) and not is_plain_verb(word[2]):
                group.append(word)
            elif group:
                yield group
                group = []
        if group:
            yield group

    def _mask_likely_names(self, words: list[tuple[int, int, str]], claims: _Claims) -> None:
        for start, end, word in words:
            if claims.is_free(start, end) and looks_like_name(word):
                claims.add(Span(start, end, PERSON, f"name:{name_lemma(word)}", word))
