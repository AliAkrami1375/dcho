"""The text frontend: raw Persian text in, model input ids out.

This module owns the order of operations and nothing else; each stage
lives in its own module. The order is not arbitrary and changing it breaks
things quietly rather than loudly, so it is worth stating why it is what
it is:

  normalise      digits must be unified before numbers are expanded, and
                 numbers must be expanded before unsupported characters are
                 dropped, or the digits are deleted before anyone reads them
  tokenise       needs normalised text, since word boundaries depend on the
                 ZWNJ having been put where orthography wants it
  ezafe          needs whole words, and needs to see its neighbours, so it
                 cannot run per-word inside the phonemiser
  phonemise      needs the ezafe decision, because the linker changes the
                 word's final segment
  encode         phoneme string to embedding indices

A blank symbol is interleaved between phonemes by default. That is not
cosmetic: it gives the monotonic alignment search somewhere to put the
silence and coarticulation between phones, instead of forcing it to
stretch a real phone across material that is not that phone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import phonemes as ph
from .normalizer import normalize

# Punctuation to prosodic break. Anything not listed here has already been
# dropped by the normaliser.
BREAK_MAP = {
    "،": ph.BREAK_MINOR,
    "؛": ph.BREAK_MINOR,
    ":": ph.BREAK_MINOR,
    "…": ph.BREAK_MINOR,
    "(": ph.BREAK_MINOR,
    ")": ph.BREAK_MINOR,
    "«": "",
    "»": "",
    '"': "",
    ".": ph.BREAK_MAJOR,
    "؟": ph.BREAK_QUEST,
    "!": ph.BREAK_EXCL,
}

_TOKEN = re.compile(r"[ء-ی‌]+|[،؛؟!.:()«»\"…]")


@dataclass
class FrontendOutput:
    text: str
    normalized: str
    tokens: list[str]
    ezafe: list[bool]
    phonemes: str
    ids: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        return self.phonemes


def tokenize_text(text: str) -> list[str]:
    """Split normalised text into words and punctuation marks."""
    return _TOKEN.findall(text)


def _is_punct(token: str) -> bool:
    return token in BREAK_MAP


class Frontend:
    """Callable text frontend.

    `lexicon` and the phonemiser are injected rather than imported at module
    scope so that a future learned grapheme-to-phoneme model can be dropped
    in without touching this file.
    """

    def __init__(self, lexicon=None, add_blank: bool = True, predict_ezafe: bool = True):
        self.add_blank = add_blank
        self.predict_ezafe = predict_ezafe
        self._lexicon = lexicon
        self._g2p = None
        self._ezafe = None

    # Imported lazily: the rule tables are large and a caller that only
    # wants `normalize` should not pay to build them.
    def _load(self):
        if self._g2p is None:
            from . import g2p as _g2p

            self._g2p = _g2p
            if self._lexicon is None:
                from .lexicon import DEFAULT

                self._lexicon = DEFAULT
        if self._ezafe is None:
            from . import ezafe as _ezafe

            self._ezafe = _ezafe

    def __call__(self, text: str) -> FrontendOutput:
        self._load()

        normalized = normalize(text)
        tokens = tokenize_text(normalized)

        words = [t for t in tokens if not _is_punct(t)]
        if self.predict_ezafe and words:
            word_flags = self._ezafe.predict_ezafe(tokens)
        else:
            word_flags = [False] * len(tokens)

        pieces: list[str] = []
        for token, flag in zip(tokens, word_flags):
            if _is_punct(token):
                mark = BREAK_MAP[token]
                if mark:
                    pieces.append(mark)
                continue
            # When ezafe is written out, the linker has to be removed before
            # phonemising or it is mistaken for the indefinite marker.
            stem = self._g2p.ezafe_stem(token, self._lexicon) if flag else token
            p = self._g2p.word_to_phonemes(stem, self._lexicon)
            p = self._g2p.apply_stress(p, stem)
            if flag:
                p = self._g2p.apply_ezafe(p)
            pieces.append(p)

        phoneme_string = " ".join(x for x in pieces if x)
        # A break marker should hug the preceding word rather than float.
        phoneme_string = re.sub(r"\s+([|?!])", r" \1", phoneme_string)

        ids = self.encode(phoneme_string)
        return FrontendOutput(
            text=text, normalized=normalized, tokens=tokens,
            ezafe=word_flags, phonemes=phoneme_string, ids=ids,
        )

    def encode(self, phoneme_string: str) -> list[int]:
        """Phoneme string to model input ids, with optional blank interleaving."""
        symbols = ph.tokenize(phoneme_string)
        ids = [ph.SYMBOL_TO_ID.get(s, ph.UNK_ID) for s in symbols]
        if self.add_blank:
            from ..model.commons import intersperse

            ids = intersperse(ids, ph.PAD_ID)
        return [ph.BOS_ID] + ids + [ph.EOS_ID]

    def phonemize(self, text: str) -> str:
        return self(text).phonemes

    def to_ipa(self, text: str) -> str:
        return ph.to_ipa(self(text).phonemes)


_DEFAULT: Frontend | None = None


def default_frontend() -> Frontend:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Frontend()
    return _DEFAULT


def phonemize(text: str) -> str:
    return default_frontend().phonemize(text)


def text_to_ids(text: str) -> list[int]:
    return default_frontend()(text).ids
