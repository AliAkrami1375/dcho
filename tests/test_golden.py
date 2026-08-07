"""Validate the golden test set for the Persian text-to-phoneme frontend.

This test does not exercise the frontend itself; it guards the ground truth
the frontend is measured against. A golden file with a typo in it is worse
than no golden file at all, because it makes correct code look broken.
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import Counter

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dcho.text.phonemes import UNK, tokenize  # noqa: E402

GOLDEN_PATH = os.path.join(_TESTS_DIR, "golden", "golden.tsv")

CATEGORIES = {
    "ezafe",
    "homograph",
    "stress",
    "numbers",
    "zwnj",
    "arabic",
    "foreign",
    "punctuation",
    "names",
    "general",
}


def load_lines():
    """Yield (line_number, raw_line) for every non-comment, non-blank line."""
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            yield lineno, line


class GoldenFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = list(load_lines())

    def test_file_exists_and_is_not_empty(self):
        self.assertTrue(os.path.isfile(GOLDEN_PATH), GOLDEN_PATH)
        self.assertGreaterEqual(
            len(self.lines), 220, "golden set must hold at least 220 rows"
        )

    def test_every_row_has_four_columns(self):
        bad = []
        for lineno, line in self.lines:
            cols = line.split("\t")
            if len(cols) != 4 or any(c.strip() == "" for c in cols):
                bad.append((lineno, len(cols), line[:60]))
        self.assertEqual(bad, [], "rows without exactly 4 non-empty columns")

    def test_categories_are_known(self):
        unknown = sorted(
            {line.split("\t")[1] for _, line in self.lines} - CATEGORIES
        )
        self.assertEqual(unknown, [], "unknown category slugs")

    def test_ids_are_unique(self):
        seen = {}
        dupes = []
        for lineno, line in self.lines:
            rid = line.split("\t")[0].lstrip("?")
            if rid in seen:
                dupes.append((rid, seen[rid], lineno))
            else:
                seen[rid] = lineno
        self.assertEqual(dupes, [], "duplicate row ids (id, first_line, dup_line)")

    def test_phonemes_tokenize_without_unknown_symbols(self):
        failures = []
        for lineno, line in self.lines:
            cols = line.split("\t")
            if len(cols) != 4:
                continue  # reported by test_every_row_has_four_columns
            rid, phonemes = cols[0], cols[3]
            symbols = tokenize(phonemes)
            if UNK in symbols:
                # Walk the string in step with the symbols so that the
                # reported character is the one that actually failed.
                offenders, pos = set(), 0
                for sym in symbols:
                    if sym == UNK:
                        offenders.add(phonemes[pos])
                        pos += 1
                    else:
                        pos += len(sym)
                failures.append((lineno, rid, sorted(offenders), phonemes))
        self.assertEqual(
            failures,
            [],
            "phoneme strings containing symbols outside the inventory "
            "(line, id, offending characters, string)",
        )

    def test_phoneme_column_is_normalised(self):
        """No leading/trailing or doubled spaces, which would tokenize oddly."""
        bad = []
        for lineno, line in self.lines:
            cols = line.split("\t")
            if len(cols) != 4:
                continue
            ph = cols[3]
            if ph != ph.strip() or "  " in ph:
                bad.append((lineno, cols[0], repr(ph)))
        self.assertEqual(bad, [], "phoneme strings with stray whitespace")

    def test_report_counts(self):
        """Reporting only: never fails, prints the shape of the set."""
        per_category = Counter(line.split("\t")[1] for _, line in self.lines)
        flagged = [
            line.split("\t")[0]
            for _, line in self.lines
            if line.startswith("?")
        ]
        total = len(self.lines)
        print("\ngolden.tsv: %d rows" % total)
        for cat in sorted(per_category):
            print("  %-12s %3d" % (cat, per_category[cat]))
        print(
            "  %-12s %3d (%.1f%%) - flagged for human review"
            % ("?-prefixed", len(flagged), 100.0 * len(flagged) / max(total, 1))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
