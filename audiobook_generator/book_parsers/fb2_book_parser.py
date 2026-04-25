# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

import base64
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

from audiobook_generator.book_parsers.base_book_parser import BaseBookParser
from audiobook_generator.config.general_config import GeneralConfig

logger = logging.getLogger(__name__)

# FictionBook 2 standard namespace
_FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
_XLINK_NS = "http://www.w3.org/1999/xlink"


class Fb2BookParser(BaseBookParser):
    """Parser for FictionBook 2 (.fb2) files."""

    def __init__(self, config: GeneralConfig):
        super().__init__(config)
        self._tree = ET.parse(self.config.input_file)
        self._root = self._tree.getroot()
        # Detect the actual namespace used in this file (handles both standard
        # and namespace-less FB2 files).
        root_tag = self._root.tag
        if root_tag.startswith("{"):
            self._fb = root_tag[: root_tag.index("}") + 1]
        else:
            self._fb = ""
        # XLink namespace for image href attributes
        self._xl = f"{{{_XLINK_NS}}}"

    def __str__(self) -> str:
        return super().__str__()

    # ------------------------------------------------------------------
    # BaseBookParser interface
    # ------------------------------------------------------------------

    def validate_config(self):
        if self.config.input_file is None:
            raise ValueError("FB2 Parser: Input file cannot be empty")
        if not self.config.input_file.lower().endswith(".fb2"):
            raise ValueError(
                f"FB2 Parser: Unsupported file format: {self.config.input_file}"
            )

    def get_book(self):
        return self._tree

    def get_book_title(self) -> str:
        el = self._root.find(f".//{self._fb}book-title")
        if el is not None and el.text:
            return el.text.strip()
        return "Untitled"

    def get_book_author(self) -> str:
        author_el = self._root.find(f".//{self._fb}author")
        if author_el is None:
            return "Unknown"
        parts = []
        for sub in ("first-name", "middle-name", "last-name"):
            part = author_el.find(f"{self._fb}{sub}")
            if part is not None and part.text:
                parts.append(part.text.strip())
        return " ".join(filter(None, parts)) or "Unknown"

    def get_book_cover(self):
        """Return (bytes, media_type) for the cover image or None."""
        coverpage = self._root.find(f".//{self._fb}coverpage")
        if coverpage is None:
            return None
        image_el = coverpage.find(f"{self._fb}image")
        if image_el is None:
            return None
        # Try both XLink href and plain href attribute
        href = image_el.get(f"{self._xl}href") or image_el.get("href", "")
        if href.startswith("#"):
            binary_id = href[1:]
            for binary in self._root.iter(f"{self._fb}binary"):
                if binary.get("id") == binary_id:
                    content_type = binary.get("content-type", "image/jpeg")
                    raw = (binary.text or "").strip()
                    try:
                        data = base64.b64decode(raw)
                        return data, content_type
                    except Exception:
                        logger.warning("FB2 Parser: failed to decode cover image")
                        return None
        return None

    def get_chapters(self, break_string) -> List[Tuple[str, str]]:
        chapters: List[Tuple[str, str]] = []
        search_and_replaces = self.get_search_and_replaces()

        chapter_mode = getattr(self.config, "chapter_mode", "documents") or "documents"
        if chapter_mode == "toc_sections":
            logger.warning(
                "FB2 Parser: 'toc_sections' chapter mode is not supported for FB2 files; "
                "falling back to per-section chapters."
            )

        for body in self._root.iter(f"{self._fb}body"):
            # Skip auxiliary bodies (notes, footnotes, comments, …)
            body_name = body.get("name", "")
            if body_name:
                logger.debug("FB2 Parser: skipping auxiliary body name=%r", body_name)
                continue
            top_sections = body.findall(f"{self._fb}section")
            self._collect_chapters(
                top_sections, chapters, break_string, search_and_replaces
            )

        return chapters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_chapters(
        self, sections, chapters, break_string, search_and_replaces
    ):
        """Recursively walk sections; leaf sections become chapters."""
        for section in sections:
            subsections = section.findall(f"{self._fb}section")
            if subsections:
                # Non-leaf: recurse into children
                self._collect_chapters(
                    subsections, chapters, break_string, search_and_replaces
                )
            else:
                # Leaf section: emit as a chapter
                title = self._extract_section_title(section)
                raw_text = self._extract_section_raw_text(section)
                cleaned_text = self._clean_text(
                    raw_text, break_string, search_and_replaces
                )
                if not cleaned_text.strip():
                    logger.debug(
                        "FB2 Parser: skipping empty section with title=%r", title
                    )
                    continue
                sanitized_title = self._sanitize_title(
                    title or cleaned_text[:60], break_string
                )
                chapters.append((sanitized_title, cleaned_text))

    def _extract_section_title(self, section_el) -> str:
        """Return the plain-text title of a <section> element (may be empty)."""
        title_el = section_el.find(f"{self._fb}title")
        if title_el is None:
            return ""
        return " ".join(t.strip() for t in title_el.itertext() if t.strip())

    def _extract_section_raw_text(self, section_el) -> str:
        """
        Collect paragraph text from a leaf section.
        Paragraphs are separated by a single newline; empty-line elements
        produce a blank line (double newline) so that newline_mode='double'
        can recognise paragraph groups.
        """
        lines: List[str] = []

        for child in section_el:
            local = _local(child.tag)

            if local == "title":
                # Handled separately; do not duplicate in body text
                continue

            elif local in ("p", "subtitle"):
                lines.append(_element_text(child))

            elif local == "empty-line":
                # FB2 explicit paragraph break → blank separator line
                lines.append("")

            elif local == "poem":
                for stanza in child.findall(f"{self._fb}stanza"):
                    for v in stanza.findall(f"{self._fb}v"):
                        lines.append(_element_text(v))
                    lines.append("")  # stanza break

            elif local in ("cite", "epigraph"):
                for p in child.iter(f"{self._fb}p"):
                    lines.append(_element_text(p))

            elif local == "section":
                # Should not happen for a leaf, but guard defensively
                pass

        return "\n".join(lines)

    def _clean_text(self, raw: str, break_string: str, search_and_replaces) -> str:
        """Mirror the EPUB parser's _clean_document_text logic for FB2 text."""
        newline_mode = self.config.newline_mode or "double"

        if newline_mode == "single":
            cleaned = re.sub(r"\n+", break_string, raw.strip())
        elif newline_mode == "double":
            cleaned = re.sub(r"\n{2,}", break_string, raw.strip())
        elif newline_mode == "none":
            cleaned = re.sub(r"\n+", " ", raw.strip())
        else:
            raise ValueError(f"FB2 Parser: Invalid newline mode: {newline_mode}")

        # Collapse stray multiple spaces
        cleaned = re.sub(r" +", " ", cleaned)


        for sar in search_and_replaces:
            cleaned = re.sub(sar["search"], sar["replace"], cleaned)

        return cleaned

    def get_search_and_replaces(self):
        """Read optional search-and-replace file (same format as EPUB parser)."""
        search_and_replaces = []
        if self.config.search_and_replace_file:
            with open(self.config.search_and_replace_file) as fp:
                for line in fp.readlines():
                    if (
                        "==" in line
                        and not line.startswith("==")
                        and not line.endswith("==")
                        and not line.startswith("#")
                    ):
                        search_and_replaces.append(
                            {
                                "search": r"{}".format(line.split("==")[0]),
                                "replace": r"{}".format(line.split("==")[1][:-1]),
                            }
                        )
        return search_and_replaces

    @staticmethod
    def _sanitize_title(title: str, break_string: str) -> str:
        title = title.replace(break_string, " ")
        sanitized = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
        sanitized = re.sub(r"\s+", "_", sanitized.strip())
        return sanitized


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _local(tag: str) -> str:
    """Strip the Clark-notation namespace from an element tag."""
    return tag.split("}")[-1] if "}" in tag else tag


def _element_text(el) -> str:
    """Concatenate all text nodes inside an element (including inline children)."""
    return "".join(el.itertext()).strip()

