"""Language detection for user queries: ``"en"``, ``"es"`` or ``"other"``.

The lexical index carries an English porter stemmer, which hurts Spanish queries:
it folds Spanish tokens onto English stems (``las``/``la``) and drops accents, so
a Spanish question matches pages that have nothing to do with it.  Retrieval
therefore needs a cheap, dependency-free guess of the query language.

The heuristic only looks at:

* Spanish-only characters: ``¿``, ``¡``, ``ñ``/``Ñ`` and accented vowels;
* short lists of function words in Spanish and English (question words,
  articles, prepositions, auxiliaries).

Technical tokens are ignored on purpose: a token that looks like an identifier
(contains ``_`` or a digit) or that is written in all caps (``VACUUM``,
``COPY``, ``pg_dump``) says nothing about the language, so ``pg_dump`` or
``VACUUM`` alone classify as ``"other"``.  Balanced ties (a genuinely mixed
query) also classify as ``"other"``.
"""

from __future__ import annotations

import re

SPANISH_CHARS = "¿¡ñÑ"
ACCENTED_VOWELS = "áéíóúüÁÉÍÓÚÜ"

#: Strong Spanish-only punctuation/diacritics.  Each occurrence is worth this
#: many function-word hits, so ``"¿cómo ...?"`` does not depend on a word list.
STRONG_CHAR_SCORE = 3

#: Ambiguous words are deliberately absent: ``a`` and ``no`` exist in both
#: languages, so a query made only of those stays ``"other"``.
SPANISH_WORDS = frozenset(
    """
    al algo con cuál cuáles cuándo cuando cuánto cuántos de del dónde donde el en
    entre es esa ese eso esta está están este esto estos hay la las le les lo los
    más mi mis muy o para pero por qué que se ser si sí sin sobre son su sus
    también una unas uno unos y ya cómo como
    """.split()
)

ENGLISH_WORDS = frozenset(
    """
    about an and are as at be by can could do does for from how i in is it my of
    on or should that the their there these this those to was we were what when
    where which who why with would you
    """.split()
)

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _is_identifier(token: str) -> bool:
    """True for SQL/CLI tokens (``pg_dump``, ``VACUUM``, ``btree_gin2d``)."""
    if "_" in token or any(char.isdigit() for char in token):
        return True
    return token.isupper() and len(token) > 1


def detect_lang(query: str) -> str:
    """Return ``"en"``, ``"es"`` or ``"other"`` for ``query``.

    Every signal adds points to one side; the side with more points wins and a
    tie (including "no signal at all") means ``"other"``.
    """
    spanish = 0
    english = 0

    for char in query:
        if char in SPANISH_CHARS:
            spanish += STRONG_CHAR_SCORE

    for token in WORD_RE.findall(query):
        if _is_identifier(token):
            continue
        lowered = token.lower()
        if any(char in ACCENTED_VOWELS for char in token):
            spanish += 1
        if lowered in SPANISH_WORDS:
            spanish += 1
        elif lowered in ENGLISH_WORDS:
            english += 1

    if spanish > english:
        return "es"
    if english > spanish:
        return "en"
    return "other"


def strip_spanish_stopwords(query: str) -> str:
    """Drop Spanish function words before a lexical (FTS5) query.

    Only ``\\w+`` tokens survive, so the signs ``¿``, ``?``, ``¡`` and ``!`` go
    away with the punctuation.  Returns the remaining terms joined by single
    spaces, and ``""`` when every term was a function word (an empty lexical
    query, not an error).
    """
    terms = [token for token in WORD_RE.findall(query) if token.lower() not in SPANISH_WORDS]
    return " ".join(terms)
