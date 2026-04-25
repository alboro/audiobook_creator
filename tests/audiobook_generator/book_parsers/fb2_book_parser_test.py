# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""
Tests for Fb2BookParser.
Analogous to epub_book_parser_test.py.
"""
import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from audiobook_generator.book_parsers.base_book_parser import get_book_parser
from audiobook_generator.book_parsers.fb2_book_parser import Fb2BookParser
from audiobook_generator.config.general_config import GeneralConfig

# ---------------------------------------------------------------------------
# Minimal sample FB2 document
# ---------------------------------------------------------------------------

# 1×1 red pixel PNG (base64) used as a fake cover image
_COVER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)

SAMPLE_FB2 = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <description>
    <title-info>
      <book-title>Тестовая книга</book-title>
      <author>
        <first-name>Иван</first-name>
        <middle-name>Иванович</middle-name>
        <last-name>Петров</last-name>
      </author>
      <coverpage>
        <image l:href="#cover.png"/>
      </coverpage>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Глава первая</p></title>
      <p>Первый абзац главы один.</p>
      <empty-line/>
      <p>Второй абзац главы один.</p>
    </section>
    <section>
      <title><p>Часть вторая</p></title>
      <section>
        <title><p>Глава 2.1</p></title>
        <p>Текст подглавы 2.1.</p>
      </section>
      <section>
        <title><p>Глава 2.2</p></title>
        <p>Текст подглавы 2.2.</p>
      </section>
    </section>
  </body>
  <body name="notes">
    <section>
      <p>Эта сноска должна быть пропущена.</p>
    </section>
  </body>
  <binary id="cover.png" content-type="image/png">{cover}</binary>
</FictionBook>
""".format(cover=_COVER_PNG_B64)

# Same file but without a cover image (no <coverpage>)
SAMPLE_FB2_NO_COVER = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <book-title>Книга без обложки</book-title>
      <author>
        <first-name>Анна</first-name>
        <last-name>Иванова</last-name>
      </author>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Единственная глава</p></title>
      <p>Текст единственной главы.</p>
    </section>
  </body>
</FictionBook>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fb2_config(path: str, **kwargs) -> GeneralConfig:
    """Return a GeneralConfig pointing at *path* with sensible defaults."""
    defaults = dict(
        input_file=path,
        output_folder="output",
        output_text=False,
        title_mode="auto",
        log="INFO",
        newline_mode="double",
        chapter_start=1,
        chapter_end=-1,
        search_and_replace_file=None,
        chapter_mode="documents",
    )
    defaults.update(kwargs)
    return GeneralConfig(MagicMock(**defaults))


def _write_tmp_fb2(content: str) -> str:
    """Write *content* to a temp .fb2 file and return its path."""
    with tempfile.NamedTemporaryFile(
        suffix=".fb2", mode="w", encoding="utf-8", delete=False
    ) as fh:
        fh.write(content)
        return fh.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFb2BookParserFactory(unittest.TestCase):
    """Tests for the get_book_parser factory function."""

    def setUp(self):
        self._path = _write_tmp_fb2(SAMPLE_FB2)
        self._config = _make_fb2_config(self._path)

    def tearDown(self):
        Path(self._path).unlink(missing_ok=True)

    def test_get_book_parser_returns_fb2_parser(self):
        parser = get_book_parser(self._config)
        self.assertIsInstance(parser, Fb2BookParser)

    def test_unsupported_extension_raises(self):
        config = MagicMock(input_file="book.unsupported")
        with self.assertRaises(NotImplementedError):
            get_book_parser(config)

    def test_wrong_extension_in_fb2_parser_raises(self):
        with self.assertRaises(ValueError):
            Fb2BookParser(MagicMock(input_file="book.epub"))


class TestFb2BookParserMetadata(unittest.TestCase):
    """Tests for title / author / cover extraction."""

    def setUp(self):
        self._path = _write_tmp_fb2(SAMPLE_FB2)
        config = _make_fb2_config(self._path)
        self.parser = Fb2BookParser(config)

    def tearDown(self):
        Path(self._path).unlink(missing_ok=True)

    def test_get_book_title(self):
        self.assertEqual(self.parser.get_book_title(), "Тестовая книга")

    def test_get_book_author_full_name(self):
        self.assertEqual(self.parser.get_book_author(), "Иван Иванович Петров")

    def test_get_book_cover_returns_bytes_and_media_type(self):
        result = self.parser.get_book_cover()
        self.assertIsNotNone(result)
        data, media_type = result
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
        self.assertEqual(media_type, "image/png")

    def test_get_book_cover_content_matches_embedded_binary(self):
        result = self.parser.get_book_cover()
        self.assertIsNotNone(result)
        data, _ = result
        expected = base64.b64decode(_COVER_PNG_B64)
        self.assertEqual(data, expected)

    def test_get_book_cover_returns_none_when_absent(self):
        path = _write_tmp_fb2(SAMPLE_FB2_NO_COVER)
        try:
            parser = Fb2BookParser(_make_fb2_config(path))
            self.assertIsNone(parser.get_book_cover())
        finally:
            Path(path).unlink(missing_ok=True)

    def test_untitled_book_fallback(self):
        fb2 = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description><title-info></title-info></description>
  <body><section><p>Text.</p></section></body>
</FictionBook>"""
        path = _write_tmp_fb2(fb2)
        try:
            parser = Fb2BookParser(_make_fb2_config(path))
            self.assertEqual(parser.get_book_title(), "Untitled")
            self.assertEqual(parser.get_book_author(), "Unknown")
        finally:
            Path(path).unlink(missing_ok=True)


class TestFb2BookParserChapters(unittest.TestCase):
    """Tests for chapter / section extraction."""

    def setUp(self):
        self._path = _write_tmp_fb2(SAMPLE_FB2)
        config = _make_fb2_config(self._path)
        self.parser = Fb2BookParser(config)

    def tearDown(self):
        Path(self._path).unlink(missing_ok=True)

    def test_chapter_count(self):
        # Leaf sections: Chapter 1 (flat) + Sub 2.1 + Sub 2.2 = 3
        chapters = self.parser.get_chapters("   ")
        self.assertEqual(len(chapters), 3)

    def test_notes_body_is_skipped(self):
        chapters = self.parser.get_chapters("   ")
        all_text = " ".join(text for _, text in chapters)
        self.assertNotIn("сноска", all_text.lower())

    def test_chapter_titles_are_sanitized(self):
        chapters = self.parser.get_chapters("   ")
        titles = [t for t, _ in chapters]
        self.assertEqual(titles[0], "Глава_первая")
        self.assertEqual(titles[1], "Глава_21")
        self.assertEqual(titles[2], "Глава_22")

    def test_chapter_text_contains_paragraphs(self):
        chapters = self.parser.get_chapters("   ")
        ch1_text = chapters[0][1]
        self.assertIn("Первый абзац главы один", ch1_text)
        self.assertIn("Второй абзац главы один", ch1_text)

    def test_title_not_duplicated_in_body_text(self):
        # The <title> element should not appear in the body text
        chapters = self.parser.get_chapters("   ")
        ch1_text = chapters[0][1]
        self.assertNotIn("Глава первая", ch1_text)

    def test_empty_line_produces_break_string(self):
        # The <empty-line/> between paras in Chapter 1 should become break_string
        chapters = self.parser.get_chapters("BREAK")
        ch1_text = chapters[0][1]
        self.assertIn("BREAK", ch1_text)

    def test_newline_mode_none_collapses_paragraphs(self):
        path = _write_tmp_fb2(SAMPLE_FB2)
        try:
            config = _make_fb2_config(path, newline_mode="none")
            parser = Fb2BookParser(config)
            chapters = parser.get_chapters("   ")
            ch1_text = chapters[0][1]
            self.assertNotIn("\n", ch1_text)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_nested_sections_become_separate_chapters(self):
        chapters = self.parser.get_chapters("   ")
        ch2_text = chapters[1][1]
        ch3_text = chapters[2][1]
        self.assertIn("2.1", ch2_text)
        self.assertIn("2.2", ch3_text)

    def test_empty_sections_are_skipped(self):
        fb2 = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description><title-info><book-title>T</book-title></title-info></description>
  <body>
    <section><title><p>Empty</p></title></section>
    <section><title><p>Real</p></title><p>Has content.</p></section>
  </body>
</FictionBook>"""
        path = _write_tmp_fb2(fb2)
        try:
            parser = Fb2BookParser(_make_fb2_config(path))
            chapters = parser.get_chapters("   ")
            self.assertEqual(len(chapters), 1)
            self.assertIn("Has content", chapters[0][1])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_remove_endnotes(self):
        """The remove_endnotes normalizer strips inline footnote numbers."""
        from audiobook_generator.normalizers.remove_endnotes_normalizer import RemoveEndnotesNormalizer
        config = _make_fb2_config(":memory:")
        normalizer = RemoveEndnotesNormalizer(config)
        text = "Текст предложения.1 Продолжение."
        result = normalizer.normalize(text)
        self.assertNotIn("1", result.split("предложения")[1][:2])

    def test_poem_verses_are_extracted(self):
        fb2 = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description><title-info><book-title>T</book-title></title-info></description>
  <body>
    <section>
      <poem>
        <stanza>
          <v>Строка первая.</v>
          <v>Строка вторая.</v>
        </stanza>
      </poem>
    </section>
  </body>
</FictionBook>"""
        path = _write_tmp_fb2(fb2)
        try:
            parser = Fb2BookParser(_make_fb2_config(path))
            chapters = parser.get_chapters("   ")
            self.assertEqual(len(chapters), 1)
            text = chapters[0][1]
            self.assertIn("Строка первая", text)
            self.assertIn("Строка вторая", text)
        finally:
            Path(path).unlink(missing_ok=True)


class TestFb2BookParserStaticHelpers(unittest.TestCase):
    def test_sanitize_title_removes_punctuation(self):
        self.assertEqual(
            Fb2BookParser._sanitize_title("Глава: первая!", "   "),
            "Глава_первая",
        )

    def test_sanitize_title_replaces_break_string(self):
        self.assertEqual(
            Fb2BookParser._sanitize_title("До   после", "   "),
            "До_после",
        )

    def test_sanitize_title_unicode_letters_preserved(self):
        result = Fb2BookParser._sanitize_title("Пролог — вступление", "   ")
        self.assertIn("Пролог", result)
        self.assertIn("вступление", result)


if __name__ == "__main__":
    unittest.main()

