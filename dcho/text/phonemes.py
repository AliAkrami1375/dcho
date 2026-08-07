"""Persian phoneme inventory for dcho.

The inventory is deliberately small. A compact symbol set keeps the text
encoder embedding table small and lets every symbol be seen often enough
during training to be learned well.

Symbols are IPA-based but written as ASCII-safe multi-character tokens so
that the whole pipeline can be inspected, diffed and unit-tested as plain
text. `PHONEME_TO_IPA` maps back to real IPA for display purposes.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Consonants
# --------------------------------------------------------------------------
# Persian has 23 consonants. Note that <غ> and <ق> have merged in modern
# standard Tehrani Persian: both are realised as a voiced uvular fricative
# or stop in free variation. We model them as a single phoneme `q`, which
# reflects actual pronunciation and avoids forcing the model to learn a
# distinction that is not present in the audio.
CONSONANTS = [
    "p", "b",           # bilabial stops
    "t", "d",           # alveolar stops
    "k", "g",           # velar stops
    "q",                # uvular (ق/غ merged)
    "'",                # glottal stop (ء/ع)
    "f", "v",           # labiodental fricatives
    "s", "z",           # alveolar fricatives
    "sh", "zh",         # postalveolar fricatives (ش/ژ)
    "x",                # voiceless uvular fricative (خ)
    "h",                # glottal fricative (ه/ح)
    "ch", "j",          # affricates (چ/ج)
    "m", "n",           # nasals
    "l",                # lateral
    "r",                # alveolar tap (ر)
    "y",                # palatal approximant (ی as consonant)
]

# --------------------------------------------------------------------------
# Vowels
# --------------------------------------------------------------------------
# Persian has a 6-vowel system. The three "long" vowels are written in the
# script; the three "short" ones are not, which is the central difficulty
# this project has to solve (see docs/DESIGN.fa.md section 5).
LONG_VOWELS = [
    "A",    # /ɒː/  آ / ا      as in آب
    "i",    # /iː/  ای         as in بید
    "u",    # /uː/  او         as in بود
]

SHORT_VOWELS = [
    "a",    # /æ/   fatha      as in بد
    "e",    # /e/   kasra      as in بِرد
    "o",    # /o/   zamma      as in بُرد
]

# Diphthongs are modelled as vowel + glide sequences rather than as unit
# phonemes, and are written with this module's own symbols:
#
#     ey   as in کیف  ->  keyf     (IPA /ej/)
#     ov   as in نو   ->  nov      (IPA /ow/)
#
# Note the glides are `y` and `v`, the same symbols used for the
# consonants: that sharing is the point, since the acoustic realisation is
# the same and the model should not have to learn it twice. Do not write
# these as `ej` or `ow` - `j` is the affricate ج and `w` is not a symbol in
# this inventory at all.

VOWELS = LONG_VOWELS + SHORT_VOWELS

# --------------------------------------------------------------------------
# Suprasegmentals and structural markers
# --------------------------------------------------------------------------
# These carry prosody. They are real inputs to the model, not decoration:
# the text encoder attends over them and the duration predictor uses them
# to place pauses.
STRESS = "1"        # primary stress, placed BEFORE the stressed vowel
WORD_SEP = " "      # word boundary
EZAFE = "-"         # ezafe linker; realised as /e/ or /ye/, kept distinct
                    # from a plain /e/ so the model can learn its specific
                    # (shorter, unstressed, tightly bound) realisation

# Prosodic breaks, ordered by increasing strength.
BREAK_MINOR = "|"   # comma, clause boundary      -> short pause
BREAK_MAJOR = "||"  # sentence end                -> long pause
BREAK_QUEST = "?"   # question intonation
BREAK_EXCL = "!"    # exclamation intonation

MARKERS = [STRESS, WORD_SEP, EZAFE, BREAK_MINOR, BREAK_MAJOR, BREAK_QUEST, BREAK_EXCL]

# --------------------------------------------------------------------------
# Special tokens
# --------------------------------------------------------------------------
PAD = "_"           # index 0, must stay first
BOS = "^"
EOS = "$"
UNK = "#"

SPECIALS = [PAD, BOS, EOS, UNK]

# --------------------------------------------------------------------------
# The symbol table
# --------------------------------------------------------------------------
# Order matters and must never change once a model is trained against it:
# it defines the embedding row for every symbol.
SYMBOLS: list[str] = SPECIALS + CONSONANTS + VOWELS + MARKERS

SYMBOL_TO_ID: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL: dict[int, str] = {i: s for i, s in enumerate(SYMBOLS)}

N_SYMBOLS = len(SYMBOLS)

PAD_ID = SYMBOL_TO_ID[PAD]
BOS_ID = SYMBOL_TO_ID[BOS]
EOS_ID = SYMBOL_TO_ID[EOS]
UNK_ID = SYMBOL_TO_ID[UNK]

# Longest symbol first so that greedy tokenisation matches "sh" before "s".
_SORTED_SYMBOLS = sorted(SYMBOLS, key=len, reverse=True)

# --------------------------------------------------------------------------
# Display mapping
# --------------------------------------------------------------------------
PHONEME_TO_IPA = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "ɡ",
    "q": "ɢ", "'": "ʔ", "f": "f", "v": "v", "s": "s", "z": "z",
    "sh": "ʃ", "zh": "ʒ", "x": "χ", "h": "h", "ch": "t͡ʃ", "j": "d͡ʒ",
    "m": "m", "n": "n", "l": "l", "r": "ɾ", "y": "j",
    "A": "ɒː", "i": "iː", "u": "uː", "a": "æ", "e": "e", "o": "o",
    STRESS: "ˈ", EZAFE: "‿e", WORD_SEP: " ",
    BREAK_MINOR: "|", BREAK_MAJOR: "‖", BREAK_QUEST: "?", BREAK_EXCL: "!",
}


def tokenize(phoneme_string: str) -> list[str]:
    """Split a phoneme string into symbols using longest-match-first.

    Unknown characters become UNK rather than raising, so that a bad
    lexicon entry degrades one word instead of killing a training batch.
    """
    out: list[str] = []
    i = 0
    n = len(phoneme_string)
    while i < n:
        for sym in _SORTED_SYMBOLS:
            if phoneme_string.startswith(sym, i):
                out.append(sym)
                i += len(sym)
                break
        else:
            out.append(UNK)
            i += 1
    return out


def to_ids(phoneme_string: str, add_bos_eos: bool = True) -> list[int]:
    """Convert a phoneme string to model input ids."""
    ids = [SYMBOL_TO_ID.get(s, UNK_ID) for s in tokenize(phoneme_string)]
    if add_bos_eos:
        ids = [BOS_ID] + ids + [EOS_ID]
    return ids


def from_ids(ids: list[int]) -> str:
    """Inverse of `to_ids`, for debugging and test output."""
    return "".join(
        ID_TO_SYMBOL.get(i, UNK)
        for i in ids
        if i not in (PAD_ID, BOS_ID, EOS_ID)
    )


def to_ipa(phoneme_string: str) -> str:
    """Render a phoneme string as IPA, for documentation and error reports."""
    return "".join(PHONEME_TO_IPA.get(s, s) for s in tokenize(phoneme_string))


def is_vowel(symbol: str) -> bool:
    return symbol in VOWELS


def is_consonant(symbol: str) -> bool:
    return symbol in CONSONANTS
