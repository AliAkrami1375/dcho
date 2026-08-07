"""The Persian text-to-phoneme frontend: lexicon, rules, ezafe and stress.

Three stages, in order of authority:

1. `lexicon.py` knows the word. Nothing beats a listed pronunciation.
2. Morphology. Persian is agglutinative at the edges - plurals, enclitic
   possessives, verb person endings, comparatives - and a word built out of
   a known stem and a known affix is as good as listed. This is also where
   stress placement is decided, because stress in Persian is a function of
   the stem/affix split: `خریدم` is /xaˈridam/, stem-final, not final.
3. `rules.py` guesses, badly, and only for words the first two missed.

On top of that sit two sentence-level decisions that no word-at-a-time
process can make: the ezafe linker (`ezafe.py`) and, for a handful of
words, whether the token is stressed at all.

Stress conventions follow `tests/golden/golden.tsv`: exactly one `1` per
phonological word, immediately before the stressed vowel; final syllable by
default; verbal prefixes می/بـ/نـ pull the stress onto themselves; enclitics
and the listed monosyllabic function words get no mark of their own.
"""

from __future__ import annotations

from . import rules
from .ezafe import (
    BREAKS,
    GIVEN_NAMES,
    NUMBER_HEADS,
    TIME_OF_DAY,
    predict_ezafe,
    split_ezafe_marker,
)
from .lexicon import DEFAULT, STRESS_INDEX, VERB_FORMS, VERB_STEMS, Lexicon, normalize_key
from .numbers import expand_numbers

__all__ = [
    "word_to_phonemes",
    "apply_stress",
    "text_to_phonemes",
    "tokenize_text",
    "UNSTRESSED_WORDS",
]

_VOWELS = "Aiueao"

# The monosyllabic function words that golden.tsv writes without a stress
# mark. They are clitics prosodically: they lean on the next content word.
UNSTRESSED_WORDS = frozenset(
    normalize_key(w) for w in ("را", "به", "از", "در", "با", "بر", "تا", "که", "و", "هر")
)


def _count_vowels(phonemes: str) -> int:
    return sum(1 for ch in phonemes if ch in _VOWELS)


# --------------------------------------------------------------------------
# Word analysis
# --------------------------------------------------------------------------

# (orthographic suffix, reading after a consonant, reading after a vowel,
#  whether the suffix is an enclitic and therefore never stressed)
_SUFFIXES = [
    ("شان", "eshAn", "shAn", True),
    ("تان", "etAn", "tAn", True),
    ("مان", "emAn", "mAn", True),
    ("ترین", "tarin", "tarin", False),
    ("های", "hAy", "hAy", False),
    ("یان", "yAn", "yAn", False),
    ("ها", "hA", "hA", False),
    ("ان", "An", "An", False),
    ("تر", "tar", "tar", False),
    ("یم", "im", "yam", True),
    ("ید", "id", "yid", True),
    ("ند", "and", "yand", True),
    ("ام", "am", "'am", True),
    ("م", "am", "yam", True),
    ("ت", "at", "yat", True),
    ("ش", "ash", "yash", True),
    ("ی", "i", "yi", True),
]

_VERB_PREFIXES = (("نمی", "nemi"), ("می", "mi"), ("ن", "na"), ("ب", "be"))

_MAX_DEPTH = 4


def _analyze(word: str, lexicon: Lexicon, depth: int = 0):
    """Return `(phonemes, stress_vowel_index_or_None)` or None.

    A `None` index means "no special claim": the caller applies the default
    final-syllable rule.
    """
    key = normalize_key(word)
    if not key:
        return None

    readings = lexicon.lookup(key)
    if readings:
        return readings[0], STRESS_INDEX.get(key)

    if depth >= _MAX_DEPTH:
        return None

    # Verbal prefixes take the stress off the stem and onto themselves.
    for prefix_orth, prefix_phon in _VERB_PREFIXES:
        if key.startswith(prefix_orth) and len(key) > len(prefix_orth):
            rest = key[len(prefix_orth):]
            if rest in VERB_FORMS or rest in VERB_STEMS:
                sub = _analyze(rest, lexicon, depth + 1)
                if sub is not None and sub[0]:
                    return prefix_phon + sub[0], 0

    for suffix, after_consonant, after_vowel, enclitic in _SUFFIXES:
        if not key.endswith(suffix) or len(key) <= len(suffix):
            continue
        base = key[: -len(suffix)]
        sub = _analyze(base, lexicon, depth + 1)
        if sub is None or not sub[0]:
            continue
        base_phonemes, base_index = sub
        tail = after_vowel if base_phonemes[-1] in _VOWELS else after_consonant
        phonemes = base_phonemes + tail
        if not enclitic:
            return phonemes, None
        if base_index is None:
            base_index = _count_vowels(base_phonemes) - 1
        return phonemes, base_index

    return None


def word_to_phonemes(word: str, lexicon: Lexicon | None = None) -> str:
    """Phonemize one word, without stress and without the ezafe linker."""
    lex = lexicon if lexicon is not None else DEFAULT
    analysis = _analyze(word, lex)
    if analysis is not None:
        return analysis[0]
    return rules.grapheme_to_phoneme(word)


# --------------------------------------------------------------------------
# Ezafe realisation
# --------------------------------------------------------------------------
# Persian sometimes writes the ezafe linker and sometimes does not. Both
# have to be handled, and they have to be handled together, because the
# written form changes what the stem is:
#
#     کتاب من      unwritten    ketAb  + e   ->  ketAb-
#     خانهٔ من      hamza        xAne   + ye  ->  xAney-
#     خانه‌ی من     zwnj + ye    xAne   + ye  ->  xAney-
#     صدای موسیقی  alef + ye    sedA   + ye  ->  sedAy-
#
# Treating that final ye as part of the stem yields xAneyi and sedAyi -
# the indefinite marker instead of the linker, which is a different word.

ZWNJ = "‌"


def ezafe_stem(word: str, lexicon: Lexicon | None = None) -> str:
    """Strip an explicitly written ezafe linker to recover the stem.

    The hamza and zero-width-non-joiner forms are unambiguous. The bare
    alef+ye form is not - روی is a word in its own right - so it is only
    stripped when doing so leaves something the lexicon recognises.
    """
    if word.endswith(ZWNJ + "ی"):
        return word[: -len(ZWNJ) - 1]
    if word.endswith("ای") and len(word) > 2:
        lex = lexicon if lexicon is not None else DEFAULT
        stem = word[:-1]
        if stem in lex:
            return stem
    return word


def apply_ezafe(phonemes: str) -> str:
    """Attach the ezafe linker to an already-stressed phoneme string.

    After a consonant the linker is a bare /e/, carried by the marker
    itself. After a vowel it needs the /y/ glide: xAne takes xAne-ye, not
    xAne-e, and the glide is a real segment the acoustic model has to see.
    """
    if phonemes and phonemes[-1] in _VOWELS:
        phonemes += "y"
    return phonemes + "-"


# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------


def apply_stress(
    phonemes: str,
    word: str,
    *,
    stressed: bool | None = None,
    vowel_index: int | None = None,
    lexicon: Lexicon | None = None,
) -> str:
    """Insert the single `1` mark into `phonemes`.

    `stressed` and `vowel_index` let a caller that has sentence context
    override the word-level decision - the vocative shift and the
    preposition/noun split on در are the two cases that need it.
    """
    if not phonemes:
        return phonemes
    positions = [i for i, ch in enumerate(phonemes) if ch in _VOWELS]
    if not positions:
        return phonemes

    key = normalize_key(word)
    if vowel_index is None:
        if stressed is False:
            return phonemes
        if stressed is not True and key in UNSTRESSED_WORDS:
            return phonemes
        lex = lexicon if lexicon is not None else DEFAULT
        analysis = _analyze(word, lex)
        index = analysis[1] if analysis is not None else None
        if index is None:
            index = len(positions) - 1
    else:
        index = vowel_index

    index = max(0, min(index, len(positions) - 1))
    cut = positions[index]
    return phonemes[:cut] + "1" + phonemes[cut:]


# --------------------------------------------------------------------------
# Homographs
# --------------------------------------------------------------------------
#
# Words with more than one reading are resolved from a handful of local
# context cues. This is the crudest part of the frontend and it is meant to
# be: real homograph resolution needs the syntactic parse the neural stage
# will learn, and a cue table at least fails predictably.

_KARDAN = frozenset(
    normalize_key(w)
    for w in ("کرد", "کردند", "کردم", "کردیم", "کردید", "کردی", "کرده", "میکند", "کند", "کنند")
)


def _clause_final(index: int, keys: list) -> bool:
    return index == len(keys) - 1 or keys[index + 1] in BREAKS


def _choose_reading(key: str, readings: list, index: int, keys: list, flags: list) -> str:
    if len(readings) == 1:
        return readings[0]
    following = keys[index + 1] if index + 1 < len(keys) else ""
    preceding = keys[index - 1] if index > 0 else ""

    if key == normalize_key("مرد"):
        # را marks it as a direct object, so a noun; clause-final, a verb.
        if following == normalize_key("را"):
            return "mard"
        return "mord" if _clause_final(index, keys) else "mard"
    if key == normalize_key("کشت"):
        return "kosht" if _clause_final(index, keys) else "kesht"
    if key == normalize_key("مهر"):
        return "mohr" if following in _KARDAN else "mehr"
    if key == normalize_key("پر"):
        if following == normalize_key("از"):
            return "por"
        return "par" if flags[index] else "por"
    if key == normalize_key("تنگ"):
        return "tong" if preceding == normalize_key("در") else "tang"
    if key == normalize_key("نه"):
        if preceding in NUMBER_HEADS or following in TIME_OF_DAY:
            return "noh"
        return "na"
    if key == normalize_key("کرم"):
        if following in (normalize_key("او"), normalize_key("من"), normalize_key("شما")):
            return "karam"
        if following == normalize_key("پوست"):
            return "kerem"
        return "kerm"
    return readings[0]


# --------------------------------------------------------------------------
# Sentence assembly
# --------------------------------------------------------------------------

_PUNCTUATION = {
    "،": "|",
    ",": "|",
    "؛": "|",
    ";": "|",
    ":": "|",
    "…": "|",
    "(": "|",
    ")": "|",
    "[": "|",
    "]": "|",
    "-": "|",
    "—": "|",
    ".": "||",
    "؟": "?",
    "?": "?",
    "!": "!",
}

_DROPPED = "«»\"'“”‘’"


def tokenize_text(text: str) -> list[str]:
    """Split raw Persian text into word and break tokens.

    A convenience for callers and tests that have a sentence rather than a
    token list; the project normalizer is the real entry point and does
    considerably more than this.
    """
    expanded = expand_numbers(text)
    tokens: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for char in expanded:
        if char in _DROPPED:
            flush()
            continue
        if char in _PUNCTUATION:
            flush()
            tokens.append(_PUNCTUATION[char])
            continue
        if char.isspace():
            flush()
            continue
        buffer.append(char)
    flush()
    return tokens


def text_to_phonemes(
    tokens: list[str],
    ezafe_flags: list[bool] | None = None,
    lexicon: Lexicon | None = None,
) -> str:
    """Phonemize a tokenised sentence, with stress, ezafe and breaks."""
    lex = lexicon if lexicon is not None else DEFAULT
    if not tokens:
        return ""

    normalised = [_PUNCTUATION.get(t, t) for t in tokens]
    flags = list(ezafe_flags) if ezafe_flags is not None else predict_ezafe(normalised)
    if len(flags) < len(normalised):
        flags += [False] * (len(normalised) - len(flags))

    bare: list[str] = []
    keys: list[str] = []
    for token in normalised:
        if token in BREAKS:
            bare.append(token)
            keys.append(token)
            continue
        stripped, _ = split_ezafe_marker(token)
        bare.append(stripped)
        keys.append(normalize_key(stripped))

    out: list[str] = []
    for index, token in enumerate(normalised):
        if token in BREAKS:
            out.append(token)
            continue

        word = bare[index]
        key = keys[index]
        readings = lex.lookup(key)
        if readings:
            phonemes = _choose_reading(key, readings, index, keys, flags)
        else:
            phonemes = word_to_phonemes(word, lex)
        if not phonemes:
            continue

        stressed = None
        vowel_index = None
        if key == normalize_key("در"):
            # در is a preposition (unstressed) unless it is the noun "door",
            # which it is when it carries a linker or governs را.
            following = keys[index + 1] if index + 1 < len(keys) else ""
            if flags[index] or following == normalize_key("را"):
                stressed = True
        if (
            index == 0
            and key in GIVEN_NAMES
            and index + 1 < len(normalised)
            and normalised[index + 1] in BREAKS
        ):
            # Vocative: stress moves to the first syllable.
            vowel_index = 0

        phonemes = apply_stress(
            phonemes,
            word,
            stressed=stressed,
            vowel_index=vowel_index,
            lexicon=lex,
        )

        if flags[index]:
            if phonemes[-1] in _VOWELS:
                phonemes += "y"
            phonemes += "-"
        out.append(phonemes)

    return " ".join(out)
