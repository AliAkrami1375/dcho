"""Persian text normalisation.

Everything here happens before phonemisation and exists to collapse the
many ways the same Persian utterance can be written down into one. Persian
text in the wild mixes Arabic and Persian letter forms, three digit sets,
optional diacritics, and a zero-width non-joiner that is frequently typed
as a space, omitted entirely, or replaced by a plain hyphen.

Leaving that variation in place would force the acoustic model to spend
capacity learning that four spellings sound identical, using training data
that would be better spent on prosody.
"""

from __future__ import annotations

import re
import unicodedata

from .numbers import expand_numbers, normalize_digits

ZWNJ = "‌"
ZWJ = "‍"

# --------------------------------------------------------------------------
# Character-level unification
# --------------------------------------------------------------------------
# Arabic letter forms that Persian keyboards and older documents produce,
# mapped onto their Persian equivalents.
CHAR_MAP = {
    "ي": "ی",  # ARABIC YEH            -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA          -> FARSI YEH
    "ے": "ی",  # YEH BARREE            -> FARSI YEH
    "ك": "ک",  # ARABIC KAF            -> KEHEH
    "ڪ": "ک",  # SWASH KAF             -> KEHEH
    "ة": "ه",  # TEH MARBUTA           -> HEH
    "ؤ": "و",  # WAW WITH HAMZA        -> WAW
    "ئ": "ی",  # YEH WITH HAMZA        -> FARSI YEH
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "ٲ": "ا",
    "ٵ": "ا",
    "ھ": "ه",
    "ۀ": "ه",  # HEH WITH YEH ABOVE    -> HEH
    "ۍ": "ی",
    "ې": "ی",
    "﻿": "",        # BOM
    "‎": "",        # LRM
    "‏": "",        # RLM
    " ": " ",       # NBSP
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "٫": "٫",  # decimal separator, handled downstream
}

# Short vowels and other diacritics. Persian text is normally written
# without them; where they do appear they are more often decorative than
# reliable, so they are stripped rather than trusted.
DIACRITICS = re.compile("[ً-ْٓ-ٰٕۖ-ۭـ]")

# Tanwin is the exception: it changes pronunciation and is preserved by
# rewriting it into letters before the strip above runs.
TANWIN_FATHA = "ً"

PUNCT_MAP = {
    "،": "،",   # Arabic comma (already the Persian one, normalised for clarity)
    "؛": "؛",
    "؟": "؟",
    "٪": "%",
    "٬": ",",
    ",": "،",
    ";": "؛",
    "?": "؟",
    "…": "...",
}

# Punctuation the frontend maps to prosodic breaks. Everything else is
# dropped, because an unmapped symbol reaching the phonemiser becomes UNK
# and UNK in the middle of a word damages that word's whole realisation.
KEEP_PUNCT = "،؛؟!.:()«»\"'-"

_SPACES = re.compile(r"[ \t -​]+")
_REPEATED_PUNCT = re.compile(r"([،؛؟!.])\1+")
_LATIN = re.compile(r"[A-Za-z]")
_PERSIAN_LETTER = re.compile(r"[ء-ی]")

# --------------------------------------------------------------------------
# ZWNJ handling
# --------------------------------------------------------------------------
# Prefixes and suffixes that are written with a ZWNJ in correct Persian
# orthography but are routinely typed with a space or nothing at all.
ZWNJ_PREFIXES = ("می", "نمی", "بی")
ZWNJ_SUFFIXES = ("ها", "های", "هایی", "هایم", "هایت", "هایش", "تر", "تری", "ترین", "ام", "ات", "اش")

_MI_SPACE = re.compile(r"\b(ن?می)\s+(?=[ء-ی])")
_HA_SPACE = re.compile(r"(?<=[ء-ی])\s+(ها(?:ی(?:ی|م|ت|ش)?)?|تر(?:ی|ین)?)\b")


def unify_characters(text: str) -> str:
    """Map Arabic letter forms and stray Unicode onto Persian equivalents."""
    text = unicodedata.normalize("NFC", text)
    return "".join(CHAR_MAP.get(ch, ch) for ch in text)


def expand_tanwin(text: str) -> str:
    """Rewrite tanwin as letters so it survives diacritic stripping.

    مثلاً is pronounced /masalan/: the tanwin is a real /an/, unlike the
    other diacritics which merely restate what the letters already say.
    """
    text = text.replace("ا" + TANWIN_FATHA, "ان")
    text = text.replace(TANWIN_FATHA + "ا", "ان")
    return text.replace(TANWIN_FATHA, "ن")


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", text)


def normalize_zwnj(text: str) -> str:
    """Put the zero-width non-joiner where orthography wants it.

    Both directions have to be handled: a ZWNJ typed as a space, and a
    ZWNJ omitted so that two words fused. Only the first is safely
    reversible, so that is all this does.
    """
    text = text.replace(ZWJ, "")
    text = text.replace("​", "")
    text = _MI_SPACE.sub(r"\1" + ZWNJ, text)
    text = _HA_SPACE.sub(ZWNJ + r"\1", text)
    # A ZWNJ that ended up adjacent to a space is redundant.
    text = re.sub(r"\s*" + ZWNJ + r"\s*", ZWNJ, text)
    return text


def normalize_punctuation(text: str) -> str:
    text = "".join(PUNCT_MAP.get(ch, ch) for ch in text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    # Ellipsis reads as a pause, not as three sentence ends.
    text = text.replace("...", "…")
    return text


def drop_unsupported(text: str) -> str:
    """Remove anything that is neither Persian, kept punctuation, nor space."""
    out = []
    for ch in text:
        if _PERSIAN_LETTER.match(ch) or ch in KEEP_PUNCT or ch.isspace() or ch in (ZWNJ, "…"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\s*([،؛؟!.:])", r"\1", text)
    text = re.sub(r"([،؛؟!.:])(?=[^\s])", r"\1 ", text)
    return text.strip()


def normalize(text: str, expand_numeric: bool = True) -> str:
    """Full normalisation pipeline.

    Order matters. Digits are unified before number expansion so the
    expander sees one digit set; numbers are expanded before unsupported
    characters are dropped, or the digits would be deleted first; tanwin is
    rewritten before diacritics are stripped, or it would be lost.
    """
    text = unify_characters(text)
    text = normalize_digits(text)
    text = expand_tanwin(text)
    text = strip_diacritics(text)
    if expand_numeric:
        text = expand_numbers(text)
    text = normalize_punctuation(text)
    text = normalize_zwnj(text)
    text = drop_unsupported(text)
    return collapse_whitespace(text)


# --------------------------------------------------------------------------
# Inspection helpers, used by the data audit job
# --------------------------------------------------------------------------

def analyse(text: str) -> dict:
    """Report what a piece of raw text contains.

    The audit job runs this over the whole corpus. The point is to find out
    what is actually in the data before deciding what the frontend has to
    cope with, rather than assuming.
    """
    unified = unify_characters(normalize_digits(text))
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "has_latin": bool(_LATIN.search(text)),
        "has_digits": any(ch.isdigit() for ch in unified),
        "has_zwnj": ZWNJ in text,
        "has_diacritics": bool(DIACRITICS.search(unified)),
        "has_tanwin": TANWIN_FATHA in unified,
        "n_persian_letters": len(_PERSIAN_LETTER.findall(unified)),
    }


def unsupported_characters(text: str) -> set[str]:
    """Characters that `normalize` would discard. Feeds the audit report."""
    unified = unify_characters(normalize_digits(strip_diacritics(expand_tanwin(text))))
    unified = normalize_punctuation(unified)
    bad = set()
    for ch in unified:
        if _PERSIAN_LETTER.match(ch) or ch in KEEP_PUNCT or ch.isspace() or ch in (ZWNJ, "…"):
            continue
        if ch.isdigit():
            continue
        bad.add(ch)
    return bad
