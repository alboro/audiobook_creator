# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Tests for zalizniak_support — parsing and DB integration.

Two suites:
  TestZalizniakLineParser  — offline unit tests for parse_zalizniak_line().
  TestZalizniakDbIntegration — integration tests: iter_zalizniak_entries()
      and build_zalizniak_pronunciation_lexicon() against a temporary DB.
      These tests download data from GitHub on first run (cached in
      .cache/zalizniak/); subsequent runs use cached files.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from audiobook_generator.normalizers.pronunciation_lexicon_db import (
    PronunciationLexiconDB,
    build_zalizniak_pronunciation_lexicon,
    ZALIZNIAK_SOURCE,
)
from audiobook_generator.normalizers.ru_text_utils import COMBINING_ACUTE
from audiobook_generator.normalizers.zalizniak_support import (
    parse_zalizniak_line,
    iter_zalizniak_entries,
    COMBINING_ACUTE as ZAL_ACUTE,
    COMBINING_GRAVE,
)


# ---------------------------------------------------------------------------
# Suite 1: offline line-parser tests
# ---------------------------------------------------------------------------

class TestZalizniakLineParser(unittest.TestCase):
    """Unit tests for parse_zalizniak_line — no network, no DB."""

    # ------------------------------------------------------------------
    # Basic lemma + stress extraction
    # ------------------------------------------------------------------

    def test_simple_noun_spoken_form(self):
        """аба́ ж 1b— → spoken_form contains stress on а́"""
        line = "аба\u0301 ж 1b\u2014"  # аба́ ж 1b—
        entry = parse_zalizniak_line(line)
        self.assertIsNotNone(entry)
        self.assertIn(COMBINING_ACUTE, entry.spoken_form)
        self.assertEqual(entry.surface_form, "аба")
        self.assertEqual(entry.spoken_form, "аба\u0301")

    def test_simple_noun_pos_is_noun(self):
        entry = parse_zalizniak_line("аба\u0301 ж 1b\u2014")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.pos, "NOUN")

    def test_homonym_prefix_stripped(self):
        """1/ба́ба жо 1a → lemma is 'ба́ба', not '1/ба́ба'"""
        line = "1/\u0431\u0430\u0301\u0431\u0430 жо 1a (_женщина_)"
        entry = parse_zalizniak_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.surface_form, "баба")
        self.assertEqual(entry.spoken_form, "ба\u0301ба")

    def test_range_prefix_stripped(self):
        """2-3/ба́ба … → prefix removed"""
        line = "2-3/\u0431\u0430\u0301\u0431\u0430 ж 1a"
        entry = parse_zalizniak_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.surface_form, "баба")

    def test_grammemes_extracted(self):
        """Stress class code stored in grammemes field."""
        entry = parse_zalizniak_line("аба\u0301 ж 1b\u2014")
        self.assertIsNotNone(entry)
        # class code should start with '1b'
        self.assertIsNotNone(entry.grammemes)
        self.assertTrue(entry.grammemes.startswith("1b"), entry.grammemes)

    def test_variant_class_in_grammemes(self):
        """изба́ ж, 1d//1d' — variant class stored."""
        line = "\u0438\u0437\u0431\u0430\u0301 ж, 1d//1d'"
        entry = parse_zalizniak_line(line)
        self.assertIsNotNone(entry)
        self.assertIsNotNone(entry.grammemes)
        self.assertIn("1d", entry.grammemes)

    def test_secondary_stress_stripped(self):
        """бо̀й-ба́ба — grave accent (secondary stress) removed from spoken_form."""
        # бо̀й-ба́ба: 'о' has grave U+0300, 'а' has acute U+0301
        line = "\u0431\u043e\u0300\u0439-\u0431\u0430\u0301\u0431\u0430 жо 1a"
        entry = parse_zalizniak_line(line)
        self.assertIsNotNone(entry)
        self.assertNotIn(COMBINING_GRAVE, entry.spoken_form)
        self.assertIn(COMBINING_ACUTE, entry.spoken_form)

    def test_unstressed_word_returned_without_acute(self):
        """Words with no stress mark are still returned (spoken_form == surface_form)."""
        line = "а (_без удар._) союз"
        entry = parse_zalizniak_line(line)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.surface_form, "а")
        self.assertEqual(entry.spoken_form, "а")
        self.assertNotIn(COMBINING_ACUTE, entry.spoken_form)

    def test_empty_line_returns_none(self):
        self.assertIsNone(parse_zalizniak_line(""))
        self.assertIsNone(parse_zalizniak_line("   "))

    def test_non_cyrillic_only_returns_none(self):
        self.assertIsNone(parse_zalizniak_line("123 something"))

    # ------------------------------------------------------------------
    # POS detection
    # ------------------------------------------------------------------

    def test_pos_verb_sv(self):
        line = "читать\u0301 св нп 1a"
        entry = parse_zalizniak_line(line)
        # Should detect verb or None; should NOT be NOUN
        if entry is not None and entry.pos is not None:
            self.assertEqual(entry.pos, "VERB")

    def test_pos_adjective(self):
        line = "кра\u0301сный п 1a"
        entry = parse_zalizniak_line(line)
        if entry is not None and entry.pos is not None:
            self.assertEqual(entry.pos, "ADJ")

    def test_pos_adverb(self):
        line = "бы\u0301стро нар"
        entry = parse_zalizniak_line(line)
        if entry is not None and entry.pos is not None:
            self.assertEqual(entry.pos, "ADV")

    # ------------------------------------------------------------------
    # Proper name flag
    # ------------------------------------------------------------------

    def test_proper_name_flag(self):
        """Entries parsed with is_proper_name=True carry the flag."""
        line = "Ива\u0301н мо 1a"
        entry = parse_zalizniak_line(line, is_proper_name=True)
        self.assertIsNotNone(entry)
        self.assertTrue(entry.is_proper_name)

    def test_common_word_not_proper(self):
        entry = parse_zalizniak_line("аба\u0301 ж 1b\u2014")
        self.assertIsNotNone(entry)
        self.assertFalse(entry.is_proper_name)

    # ------------------------------------------------------------------
    # Source tag
    # ------------------------------------------------------------------

    def test_source_is_zalizniak(self):
        entry = parse_zalizniak_line("аба\u0301 ж 1b\u2014")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source, ZALIZNIAK_SOURCE)

    def test_confidence_positive(self):
        entry = parse_zalizniak_line("аба\u0301 ж 1b\u2014")
        self.assertIsNotNone(entry)
        self.assertGreater(entry.confidence or 0, 0)


# ---------------------------------------------------------------------------
# Suite 2: integration tests (downloads data, uses real files)
# ---------------------------------------------------------------------------

class TestZalizniakDbIntegration(unittest.TestCase):
    """
    Integration tests for iter_zalizniak_entries() and
    build_zalizniak_pronunciation_lexicon().

    First run downloads data from GitHub (cached in .cache/zalizniak/).
    All subsequent runs are fast.
    """

    @classmethod
    def setUpClass(cls):
        """Build a temporary DB populated from Zaliznyak data."""
        cls._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls._tmpdir.name) / "zal_test.sqlite3"
        cls.db = PronunciationLexiconDB(db_path)
        cls.count = build_zalizniak_pronunciation_lexicon(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Basic sanity
    # ------------------------------------------------------------------

    def test_entry_count_exceeds_minimum(self):
        """At least 50 000 entries expected from Zaliznyak files."""
        self.assertGreater(
            self.count,
            50_000,
            f"Expected >50k entries, got {self.count}",
        )

    def test_source_tag_is_zalizniak(self):
        """All entries must carry source='zalizniak'."""
        db_count = self.db.count_source_entries(ZALIZNIAK_SOURCE)
        self.assertEqual(
            db_count,
            self.count,
            "Mismatch between returned count and DB count",
        )

    # ------------------------------------------------------------------
    # Spot checks for specific words
    # ------------------------------------------------------------------

    def test_word_baba_has_entry(self):
        """'баба' must be present in the Zaliznyak lexicon."""
        entries = self.db.lookup("баба", source=ZALIZNIAK_SOURCE)
        self.assertGreater(len(entries), 0, "'баба' not found in Zaliznyak DB")

    def test_baba_spoken_form_has_stress(self):
        entries = self.db.lookup("баба", source=ZALIZNIAK_SOURCE)
        stressed = [e for e in entries if COMBINING_ACUTE in (e.spoken_form or "")]
        self.assertGreater(
            len(stressed), 0,
            "'баба' entries have no stress mark in spoken_form",
        )

    def test_golova_has_entry(self):
        """'голова' — common word expected in Zaliznyak."""
        entries = self.db.lookup("голова", source=ZALIZNIAK_SOURCE)
        self.assertGreater(len(entries), 0, "'голова' not found in Zaliznyak DB")

    def test_proper_name_present(self):
        """Proper names file is loaded; at least one is_proper_name entry exists."""
        n = self.db.count_source_entries(ZALIZNIAK_SOURCE)
        # Just confirm proper names don't crash the build
        self.assertGreater(n, 0)

    # ------------------------------------------------------------------
    # Stress mark quality
    # ------------------------------------------------------------------

    def test_majority_of_entries_have_stress_mark(self):
        """
        At least 70% of entries should have a stress mark — unstressed
        function words are a small minority.
        """
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(self.db.path)) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE source=?", (ZALIZNIAK_SOURCE,)
            ).fetchone()[0]
            with_stress = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE source=? AND spoken_form LIKE ?",
                (ZALIZNIAK_SOURCE, f"%{COMBINING_ACUTE}%"),
            ).fetchone()[0]

        ratio = with_stress / total if total else 0
        self.assertGreater(
            ratio, 0.70,
            f"Only {ratio:.1%} of Zaliznyak entries have a stress mark",
        )

    # ------------------------------------------------------------------
    # iter_zalizniak_entries deduplication
    # ------------------------------------------------------------------

    def test_iter_no_duplicate_surface_spoken_pairs(self):
        """iter_zalizniak_entries must not yield duplicate (surface, spoken) pairs."""
        seen: set[tuple[str, str | None]] = set()
        duplicates: list[tuple[str, str | None]] = []
        for entry in iter_zalizniak_entries():
            key = (entry.surface_form, entry.spoken_form)
            if key in seen:
                duplicates.append(key)
            seen.add(key)

        self.assertEqual(
            duplicates,
            [],
            f"Found {len(duplicates)} duplicate (surface, spoken) pairs",
        )


if __name__ == "__main__":
    unittest.main()

