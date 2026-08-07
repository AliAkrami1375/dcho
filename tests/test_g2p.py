"""Frontend accuracy against the golden set.

This is the project's cheapest and most important quality gate. It runs on
a CPU in a couple of seconds and costs nothing, and it catches the class of
error that would otherwise only surface after a full training run: a
mispronunciation baked into every utterance the model ever learns.

The rule-based frontend is not expected to score 100%. Persian omits short
vowels, so a large share of words simply cannot be resolved without either
a lexicon entry or a model that has learned the language's statistics. The
floor below is set from the measured value, and its purpose is to detect
regressions, not to certify correctness.

Measured at the time of writing: see FLOOR constants, which sit a few
points under the observed figures.
"""

from __future__ import annotations

import unittest
from collections import defaultdict
from pathlib import Path

from dcho.text.frontend import Frontend
from dcho.text.phonemes import UNK, tokenize

GOLDEN = Path(__file__).resolve().parent / "golden" / "golden.tsv"

# Regression floors, set a little below the measured values. Raise them
# when the frontend genuinely improves; never lower them to make a change
# pass - a lowered floor is a silently accepted regression.
#
# Measured 2026-08-07, rule-based frontend with a 5,862-entry lexicon:
#   overall exact match  84.0%
#   ezafe  precision 92.3%  recall 97.8%  f1 95.0%
# Weakest categories are arabic (45%) and homograph (68%). Both are
# expected to stay weak until the learned grapheme-to-phoneme model lands:
# homographs need sentence context, which no rule set can supply.
OVERALL_FLOOR = 0.80
EZAFE_RECALL_FLOOR = 0.90


def load_golden() -> list[tuple[str, str, str, str]]:
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        rows.append(tuple(p.strip() for p in parts))
    return rows


class TestGoldenSet(unittest.TestCase):
    """Structural checks on the reference data itself."""

    @classmethod
    def setUpClass(cls):
        cls.rows = load_golden()

    def test_golden_set_is_populated(self):
        self.assertGreaterEqual(len(self.rows), 200)

    def test_ids_are_unique(self):
        ids = [r[0] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reference_phonemes_use_only_known_symbols(self):
        """A typo in the reference is worse than a bug in the code.

        It would send someone hunting through the frontend for a fault that
        does not exist, so the reference is validated before it is trusted.
        """
        bad = []
        for rid, _, _, phon in self.rows:
            if UNK in tokenize(phon):
                bad.append((rid, phon))
        self.assertEqual(bad, [], f"unknown symbols in reference rows: {bad[:5]}")


class TestFrontendAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_golden()
        cls.frontend = Frontend()
        cls.results = []
        for rid, category, text, expected in cls.rows:
            try:
                got = cls.frontend(text).phonemes
            except Exception as exc:  # a crash counts as a miss, not an error
                got = f"<error: {exc}>"
            cls.results.append((rid, category, text, expected, got))

    def test_frontend_never_emits_unknown_symbols(self):
        """Output must be encodable. An UNK reaching the model is a silent
        corruption of that word, so this is a hard failure regardless of
        whether the phonemisation was otherwise correct."""
        bad = [(r[0], r[4]) for r in self.results if UNK in tokenize(r[4])]
        self.assertEqual(bad, [], f"UNK in frontend output: {bad[:5]}")

    def test_exact_match_accuracy_above_floor(self):
        exact = sum(1 for r in self.results if r[3] == r[4])
        accuracy = exact / len(self.results)

        per_cat: dict[str, list[int]] = defaultdict(list)
        for _, cat, _, exp, got in self.results:
            per_cat[cat].append(int(exp == got))

        print(f"\n  golden exact match: {exact}/{len(self.results)} = {accuracy:.1%}")
        for cat in sorted(per_cat):
            hits = per_cat[cat]
            print(f"    {cat:14s} {sum(hits):3d}/{len(hits):3d}  {sum(hits)/len(hits):6.1%}")

        self.assertGreaterEqual(
            accuracy, OVERALL_FLOOR,
            f"exact-match accuracy {accuracy:.1%} fell below the {OVERALL_FLOOR:.0%} floor",
        )

    def test_ezafe_prediction_quality(self):
        """Ezafe is scored on its own because it is the single prediction
        that matters most, and overall accuracy hides it."""
        tp = fp = fn = 0
        for _, _, text, expected, _got in self.results:
            out = self.frontend(text)
            predicted = sum(1 for f in out.ezafe if f)
            reference = expected.count("-")
            tp += min(predicted, reference)
            fp += max(0, predicted - reference)
            fn += max(0, reference - predicted)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"\n  ezafe: precision {precision:.1%}  recall {recall:.1%}  f1 {f1:.1%}"
              f"  (tp={tp} fp={fp} fn={fn})")

        self.assertGreaterEqual(
            recall, EZAFE_RECALL_FLOOR,
            f"ezafe recall {recall:.1%} fell below the {EZAFE_RECALL_FLOOR:.0%} floor",
        )

    def test_report_worst_categories(self):
        """Not an assertion. Prints the most common mismatches so the next
        lexicon or rule change can be aimed at what actually costs the most."""
        misses = [(r[1], r[2], r[3], r[4]) for r in self.results if r[3] != r[4]]
        print(f"\n  {len(misses)} mismatches; first 8:")
        for cat, text, exp, got in misses[:8]:
            print(f"    [{cat}] {text}")
            print(f"       want {exp}")
            print(f"       got  {got}")


class TestLexicon(unittest.TestCase):
    def test_round_trip_through_disk(self):
        import tempfile

        from dcho.text.lexicon import DEFAULT, Lexicon

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lex.tsv"
            DEFAULT.save(path)
            reloaded = Lexicon.load(path)
        self.assertEqual(len(reloaded), len(DEFAULT))
        for word in list(DEFAULT._entries)[:50]:
            self.assertEqual(reloaded.lookup(word), DEFAULT.lookup(word))

    def test_lexicon_entries_are_valid_phoneme_strings(self):
        from dcho.text.lexicon import SEED

        bad = []
        for word, readings in SEED.items():
            for r in readings:
                if UNK in tokenize(r):
                    bad.append((word, r))
        self.assertEqual(bad, [], f"invalid phonemes in lexicon: {bad[:5]}")

    def test_lexicon_covers_the_most_common_words(self):
        from dcho.text.lexicon import DEFAULT

        for word in ("و", "در", "به", "از", "که", "را", "این", "است", "می‌شود", "کرد"):
            self.assertIn(word, DEFAULT, f"lexicon is missing the frequent word {word!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
