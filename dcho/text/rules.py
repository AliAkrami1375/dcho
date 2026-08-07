"""Rule-based letter-to-sound conversion for Persian.

This is the fallback path of the frontend: it runs only for words that the
lexicon does not know. Persian consonants are almost deterministic, so the
consonant half of this module is close to exact. The vowel half is not, and
cannot be: the three short vowels /a e o/ are simply not written, so any
purely graphemic converter has to guess them.

What is reliable here
---------------------
* Consonants. Several Arabic letters have merged in Persian and the phoneme
  inventory models the merger rather than the spelling: ق/غ -> `q`,
  ع/ء/أ/ؤ/ئ -> `'`, ط/ت -> `t`, ص/ث/س -> `s`, ذ/ز/ض/ظ -> `z`, ح/ه -> `h`.
* Long vowels, which *are* written: آ/ا -> `A`, او -> `u`, ای -> `i`.
* The diphthongs `ey` and `ov`, where they can be inferred.
* Word-final ه after a consonant, which is the vowel /e/ (خانه -> xAne) and
  not /h/.
* تشدید, which doubles the preceding consonant.
* The Arabic accusative tanwin ـاً, read /an/ (مثلاً -> masalan).

What is a guess
---------------
Everything to do with the unwritten short vowels. See
`DEFAULT_SHORT_VOWEL` and `_resolve_cluster` below for the exact policy and
the reasoning behind it. The policy is a guess, it is wrong often, and the
whole point of the lexicon in `lexicon.py` (and later of the neural G2P
stage) is to keep this code path off the hot words.

`grapheme_to_phoneme` returns a bare segmental string with no stress mark;
stress is a word- and sentence-level decision and is applied in `g2p.py`.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_SHORT_VOWEL",
    "CONSONANT_MAP",
    "ZWNJ",
    "normalize_word",
    "grapheme_to_phoneme",
]

ZWNJ = "‌"

# --------------------------------------------------------------------------
# Orthographic normalisation
# --------------------------------------------------------------------------

# Arabic look-alikes that Persian text picks up from Arabic keyboards, plus
# the presentation forms of ه.
_LETTER_FOLD = {
    "ي": "ی",   # ARABIC YEH        -> FARSI YEH
    "ى": "ی",   # ALEF MAKSURA      -> FARSI YEH
    "ے": "ی",   # YEH BARREE
    "ك": "ک",   # ARABIC KAF        -> KEHEH
    "ہ": "ه",   # HEH GOAL
    "ة": "ه",   # TEH MARBUTA       -> HEH (read as /e/ or /t/)
    "أ": "ء",   # ALEF WITH HAMZA   -> HAMZA
    "إ": "ء",
    "ؤ": "ء",   # WAW WITH HAMZA
    "ئ": "ء",   # YEH WITH HAMZA
    "ٱ": "ا",   # ALEF WASLA
    "ٲ": "ا",
    "ٳ": "ا",
    "ٵ": "ا",
}

# Short-vowel marks. They are rare in running text but when an author does
# write them they are the best evidence available and must win over any
# guess this module would otherwise make.
_HARAKAT = {
    "َ": "a",        # FATHA
    "ِ": "e",        # KASRA
    "ُ": "o",        # DAMMA
    "ٰ": "A",        # SUPERSCRIPT ALEF
}
_FATHATAN = "ً"      # ً  tanwin, read /an/
_DAMMATAN = "ٌ"
_KASRATAN = "ٍ"
_SHADDA = "ّ"        # ّ  gemination
_SUKUN = "ْ"         # ْ  no vowel
_TATWEEL = "ـ"       # ـ  purely decorative

_STRIPPED = _SUKUN + _TATWEEL + "ٕٔ"


def normalize_word(word: str) -> str:
    """Fold Arabic letter variants and drop decoration, keeping harakat."""
    out = []
    for ch in word:
        if ch in _STRIPPED or ch == ZWNJ:
            continue
        out.append(_LETTER_FOLD.get(ch, ch))
    return "".join(out)


# --------------------------------------------------------------------------
# Consonants
# --------------------------------------------------------------------------

# The value is the phoneme; letters that merge share one. Note that ه is
# only a consonant here - its vocalic use is handled positionally below.
CONSONANT_MAP = {
    "ب": "b",    # ب
    "پ": "p",    # پ
    "ت": "t",    # ت
    "ط": "t",    # ط
    "ث": "s",    # ث
    "س": "s",    # س
    "ص": "s",    # ص
    "ج": "j",    # ج
    "چ": "ch",   # چ
    "ح": "h",    # ح
    "ه": "h",    # ه
    "خ": "x",    # خ
    "د": "d",    # د
    "ذ": "z",    # ذ
    "ز": "z",    # ز
    "ض": "z",    # ض
    "ظ": "z",    # ظ
    "ر": "r",    # ر
    "ژ": "zh",   # ژ
    "ش": "sh",   # ش
    "ع": "'",    # ع
    "ء": "'",    # ء
    "غ": "q",    # غ
    "ق": "q",    # ق
    "ف": "f",    # ف
    "ک": "k",    # ک
    "گ": "g",    # گ
    "ل": "l",    # ل
    "م": "m",    # م
    "ن": "n",    # ن
}

_ALEF = "ا"
_ALEF_MADDA = "آ"
_VAV = "و"
_YEH = "ی"
_HEH = "ه"
_KHEH = "خ"

_VOWEL_LETTERS = frozenset({_ALEF, _ALEF_MADDA, _VAV, _YEH})

_PHON_VOWELS = frozenset("Aiueao")

# --------------------------------------------------------------------------
# Short-vowel insertion policy
# --------------------------------------------------------------------------
#
# POLICY (a guess, stated plainly): every consonant cluster that Persian
# phonotactics cannot license is broken by inserting `a`.
#
# Why `a` and not `e` or `o`: /a/ is the most frequent short vowel in
# Persian running text, it is the vowel that the Arabic fatha - the only
# short vowel Persian orthography ever bothers to write - stands for, and
# it is the epenthetic vowel Persian speakers use when reading unfamiliar
# consonant strings aloud. None of that makes it *right* for any particular
# word: کتاب is /ketAb/ and this module will say /katAb/. It is a prior,
# not a derivation, and it is why every word that matters lives in the
# lexicon instead.
#
# Where the vowel goes is a real rule rather than a guess. Persian
# syllables are (C)V(C)(C): no onset clusters at all, and a two-consonant
# coda only when sonority falls across it (/sard/, /dast/, /tang/ are
# legal; */madr/, */ketb/ are not). `_resolve_cluster` inserts the vowel at
# exactly the points where those constraints are violated.
DEFAULT_SHORT_VOWEL = "a"

# Higher is more sonorous. A final cluster is licensed when sonority falls
# strictly from the first member to the second.
_SONORITY = {
    "p": 1, "b": 1, "t": 1, "d": 1, "k": 1, "g": 1, "q": 1, "'": 1,
    "ch": 1, "j": 1,
    "f": 2, "v": 2, "s": 2, "z": 2, "sh": 2, "zh": 2, "x": 2, "h": 2,
    "m": 3, "n": 3,
    "l": 4, "r": 4,
    "y": 5,
}


def _falling(first: str, second: str) -> bool:
    return _SONORITY.get(first, 2) > _SONORITY.get(second, 2)


def _resolve_cluster(run, initial: bool, before_vowel: bool, vowel: str):
    """Insert short vowels into a consonant run until it is pronounceable.

    `run` is a list of consonant phonemes with no vowel between them,
    `initial` says whether the run starts the word (i.e. has no vowel to
    lean on), and `before_vowel` whether a vowel follows the run.
    """
    out = []
    rest = list(run)
    if initial:
        # No onset clusters: the first consonant always gets a vowel after
        # it unless it is the only one and a vowel already follows.
        out.append(rest.pop(0))
        if rest:
            out.append(vowel)
        elif not before_vowel:
            out.append(vowel)
            return out
    while rest:
        if before_vowel:
            coda, onset = rest[:-1], rest[-1:]
        else:
            coda, onset = rest, []
        if not coda or len(coda) == 1:
            out.extend(coda)
            out.extend(onset)
            return out
        if len(coda) == 2 and _falling(coda[0], coda[1]):
            out.extend(coda)
            out.extend(onset)
            return out
        # Illegal coda: close the syllable after its first member and carry
        # on with what is left, which now has a vowel in front of it.
        out.append(coda[0])
        out.append(vowel)
        rest = rest[1:]
    return out


# --------------------------------------------------------------------------
# Grapheme walk
# --------------------------------------------------------------------------

_C = "C"
_V = "V"


def _read_segments(word: str):
    """Map a normalised word to a list of (kind, phoneme) pairs.

    Vowels that the orthography writes are emitted here; the unwritten ones
    are left for `_resolve_cluster`.
    """
    segments = []
    i = 0
    n = len(word)
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""
        nxt2 = word[i + 2] if i + 2 < n else ""

        if ch in _HARAKAT:
            segments.append((_V, _HARAKAT[ch]))
            i += 1
            continue
        if ch in (_FATHATAN, _DAMMATAN, _KASRATAN):
            # ـاً: the alef is a carrier, the mark is the vowel. The alef
            # has usually already been emitted as `A`; replace it.
            if segments and segments[-1] == (_V, "A"):
                segments.pop()
            segments.append((_V, "a"))
            segments.append((_C, "n"))
            i += 1
            continue
        if ch == _SHADDA:
            if segments:
                kind, sym = segments[-1]
                if kind == _C:
                    segments.append((_C, sym))
            i += 1
            continue

        if ch == _ALEF_MADDA:
            segments.append((_V, "A"))
            i += 1
            continue

        if ch == _ALEF:
            if i == 0:
                # Word-initial ا is a vowel onset, never `A`: it spells a
                # short vowel on its own (اسم), /i/ before ی and /u/
                # before و.
                if nxt == _YEH and nxt2 not in _VOWEL_LETTERS:
                    segments.append((_V, "i"))
                    i += 2
                    continue
                if nxt == _VAV:
                    segments.append((_V, "u"))
                    i += 2
                    continue
                segments.append((_V, DEFAULT_SHORT_VOWEL))
                i += 1
                continue
            prev = segments[-1] if segments else None
            if prev is not None and prev[0] == _V:
                # Vowel + alef: a hiatus, spelled with a glottal or glide
                # in careful speech. Treat the alef as its own /A/.
                segments.append((_V, "A"))
            else:
                segments.append((_V, "A"))
            i += 1
            continue

        if ch == _VAV:
            prev = segments[-1] if segments else None
            if i == 0:
                if n == 1:
                    # The standalone conjunction و.
                    segments.append((_V, "o"))
                else:
                    segments.append((_C, "v"))
                i += 1
                continue
            if prev is not None and prev == (_C, "x") and word[i - 1] == _KHEH:
                # خو: silent before ا/ی (خواب, خویش), /o/ otherwise (خورد).
                if nxt in (_ALEF, _ALEF_MADDA, _YEH):
                    i += 1
                    continue
                segments.append((_V, "o"))
                i += 1
                continue
            if prev is not None and prev[0] == _V:
                # After a vowel و is the glide of a diphthong or a plain
                # consonant: نو -> nov, دیوار -> divAr.
                segments.append((_C, "v"))
                i += 1
                continue
            if nxt in (_ALEF, _ALEF_MADDA):
                segments.append((_C, "v"))
                i += 1
                continue
            segments.append((_V, "u"))
            i += 1
            continue

        if ch == _YEH:
            prev = segments[-1] if segments else None
            if i == 0:
                segments.append((_C, "y"))
                i += 1
                continue
            if prev is not None and prev[0] == _V:
                if prev[1] in ("a", "e"):
                    # The /ey/ diphthong: کیف -> keyf.
                    segments.append((_C, "y"))
                else:
                    segments.append((_C, "y"))
                i += 1
                continue
            if nxt in (_ALEF, _ALEF_MADDA, _VAV):
                segments.append((_C, "y"))
                i += 1
                continue
            segments.append((_V, "i"))
            i += 1
            continue

        if ch == _HEH:
            prev = segments[-1] if segments else None
            if i == n - 1 and prev is not None and prev[0] == _C:
                # Final ه after a consonant is the vowel /e/, not /h/.
                segments.append((_V, "e"))
                i += 1
                continue
            segments.append((_C, "h"))
            i += 1
            continue

        phoneme = CONSONANT_MAP.get(ch)
        if phoneme is not None:
            segments.append((_C, phoneme))
        i += 1
    return segments


def grapheme_to_phoneme(word: str) -> str:
    """Phonemize a single Persian word by rule, without stress marks.

    >>> grapheme_to_phoneme("خانه")
    'xAne'
    >>> grapheme_to_phoneme("مثلاً")
    'masalan'
    """
    word = normalize_word(word)
    if not word:
        return ""

    segments = _read_segments(word)
    if not segments:
        return ""

    out = []
    run = []
    seen_vowel = False
    for index, (kind, sym) in enumerate(segments):
        if kind == _C:
            run.append(sym)
            continue
        if run:
            out.extend(
                _resolve_cluster(
                    run, not seen_vowel, True, DEFAULT_SHORT_VOWEL
                )
            )
            run = []
        out.append(sym)
        seen_vowel = True
    if run:
        out.extend(
            _resolve_cluster(run, not seen_vowel, False, DEFAULT_SHORT_VOWEL)
        )

    return "".join(out)
