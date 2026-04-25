import logging
import re
from pathlib import PurePosixPath
from typing import Iterable, List, Tuple
from urllib.parse import unquote

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from audiobook_generator.book_parsers.base_book_parser import BaseBookParser
from audiobook_generator.config.general_config import GeneralConfig

logger = logging.getLogger(__name__)


class EpubBookParser(BaseBookParser):
    def __init__(self, config: GeneralConfig):
        super().__init__(config)
        self.book = epub.read_epub(self.config.input_file, {"ignore_ncx": True})

    def __str__(self) -> str:
        return super().__str__()

    def validate_config(self):
        if self.config.input_file is None:
            raise ValueError("Epub Parser: Input file cannot be empty")
        if not self.config.input_file.endswith(".epub"):
            raise ValueError(f"Epub Parser: Unsupported file format: {self.config.input_file}")

    def get_book(self):
        return self.book

    def get_book_title(self) -> str:
        if self.book.get_metadata("DC", "title"):
            return self.book.get_metadata("DC", "title")[0][0]
        return "Untitled"

    def get_book_author(self) -> str:
        if self.book.get_metadata("DC", "creator"):
            return self.book.get_metadata("DC", "creator")[0][0]
        return "Unknown"

    def get_book_cover(self):
        cover_items = list(self.book.get_items_of_type(ebooklib.ITEM_COVER))
        if cover_items:
            cover = cover_items[0]
            return cover.get_content(), cover.media_type
        return None

    def get_chapters(self, break_string) -> List[Tuple[str, str]]:
        chapter_mode = getattr(self.config, "chapter_mode", "documents") or "documents"
        if chapter_mode == "toc_sections":
            chapters = self._get_toc_section_chapters(break_string)
            if chapters:
                return chapters
            logger.warning(
                "EPUB TOC section grouping was requested but no usable TOC ranges were found; "
                "falling back to per-document chapters."
            )
        return self._get_document_chapters(break_string)

    def _get_document_chapters(self, break_string) -> List[Tuple[str, str]]:
        chapters = []
        for document_info in self._build_document_infos(break_string):
            chapters.append((document_info["title"], document_info["text"]))
        return chapters

    def _iter_document_items_in_reading_order(self):
        document_items = list(self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        document_map = {item.id: item for item in document_items}
        yielded_ids = set()

        for spine_entry in self.book.spine:
            item_id = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
            item = document_map.get(item_id)
            if item is None or item.id in yielded_ids:
                continue
            yielded_ids.add(item.id)
            yield item

        for item in document_items:
            if item.id in yielded_ids:
                continue
            yield item

    def _build_document_infos(self, break_string):
        search_and_replaces = self.get_search_and_replaces()
        document_infos = []
        for item in self._iter_document_items_in_reading_order():
            content = item.get_content()
            soup = BeautifulSoup(content, "lxml-xml")
            raw = soup.get_text(strip=False)
            logger.debug("Raw text: <%s>", raw[:100])

            cleaned_text = self._clean_document_text(
                raw=raw,
                break_string=break_string,
                search_and_replaces=search_and_replaces,
            )
            title = self._sanitize_title(
                self._extract_document_title(soup, cleaned_text),
                break_string,
            )
            item_name = self._get_item_name(item)
            document_infos.append(
                {
                    "item": item,
                    "item_name": item_name,
                    "href_candidates": self._normalize_href_candidates(item_name),
                    "title": title,
                    "text": cleaned_text,
                }
            )
            soup.decompose()

        return document_infos

    def _clean_document_text(self, *, raw, break_string, search_and_replaces):
        if self.config.newline_mode == "single":
            cleaned_text = re.sub(r"[\n]+", break_string, raw.strip())
        elif self.config.newline_mode == "double":
            cleaned_text = re.sub(r"[\n]{2,}", break_string, raw.strip())
        elif self.config.newline_mode == "none":
            cleaned_text = re.sub(r"[\n]+", " ", raw.strip())
        else:
            raise ValueError(f"Invalid newline mode: {self.config.newline_mode}")

        cleaned_text = re.sub(r"\s+", " ", cleaned_text)


        for search_and_replace in search_and_replaces:
            cleaned_text = re.sub(
                search_and_replace["search"],
                search_and_replace["replace"],
                cleaned_text,
            )

        return cleaned_text

    def _extract_document_title(self, soup, cleaned_text):
        if self.config.title_mode == "auto":
            title = self._extract_title_from_tags(soup)
            if title.strip() == "" or re.match(r"^\d{1,3}$", title) is not None:
                title = cleaned_text[:60]
            return title

        if self.config.title_mode == "tag_text":
            title = self._extract_title_from_tags(soup)
            return title if title.strip() else "<blank>"

        if self.config.title_mode == "first_few":
            return cleaned_text[:60]

        raise ValueError("Unsupported title_mode")

    def _extract_title_from_tags(self, soup):
        for level in ["title", "h1", "h2", "h3"]:
            found = soup.find(level)
            if found:
                return found.text
        return ""

    def _get_toc_section_chapters(self, break_string) -> List[Tuple[str, str]]:
        document_infos = self._build_document_infos(break_string)
        if not document_infos or not self.book.toc:
            return []

        href_to_index = {}
        for idx, info in enumerate(document_infos):
            for candidate in info["href_candidates"]:
                href_to_index.setdefault(candidate, idx)

        raw_ranges = []
        for entry in self.book.toc:
            title = self._get_toc_entry_title(entry)
            start_index = self._resolve_toc_entry_start_index(entry, href_to_index)
            if start_index is None:
                continue
            raw_ranges.append({"title": title, "start_index": start_index})

        if not raw_ranges:
            return []

        raw_ranges.sort(key=lambda item: item["start_index"])
        ranges = []
        for entry in raw_ranges:
            if ranges and entry["start_index"] == ranges[-1]["start_index"]:
                if not ranges[-1]["title"] and entry["title"]:
                    ranges[-1]["title"] = entry["title"]
                continue
            ranges.append(entry)

        chapters = []
        for idx, entry in enumerate(ranges):
            start_index = entry["start_index"]
            next_start_index = ranges[idx + 1]["start_index"] if idx + 1 < len(ranges) else len(document_infos)
            chunk_infos = document_infos[start_index:next_start_index]
            if not chunk_infos:
                continue

            combined_text = self._join_document_texts(
                [chunk_info["text"] for chunk_info in chunk_infos],
                break_string,
            )
            if not combined_text.strip():
                continue

            raw_title = entry["title"] or chunk_infos[0]["title"]
            title = self._sanitize_title(raw_title, break_string)
            chapters.append((title, combined_text))

        return chapters

    def _resolve_toc_entry_start_index(self, entry, href_to_index):
        indexes = []
        for href in self._iter_toc_entry_hrefs(entry):
            resolved_index = self._lookup_href_index(
                self._normalize_href_candidates(href),
                href_to_index,
            )
            if resolved_index is not None:
                indexes.append(resolved_index)
        if not indexes:
            return None
        return min(indexes)

    def _lookup_href_index(self, candidates, href_to_index):
        for candidate in candidates:
            if candidate in href_to_index:
                return href_to_index[candidate]
        return None

    def _iter_toc_entry_hrefs(self, entry) -> Iterable[str]:
        if isinstance(entry, tuple):
            section, children = entry
            section_href = getattr(section, "href", None)
            if section_href:
                yield section_href
            for child in children:
                yield from self._iter_toc_entry_hrefs(child)
            return

        href = getattr(entry, "href", None)
        if href:
            yield href

    def _get_toc_entry_title(self, entry) -> str:
        if isinstance(entry, tuple):
            section, children = entry
            section_title = (getattr(section, "title", "") or "").strip()
            if section_title:
                return section_title
            for child in children:
                child_title = self._get_toc_entry_title(child)
                if child_title:
                    return child_title
            return ""

        return (getattr(entry, "title", "") or "").strip()

    def _join_document_texts(self, texts, break_string):
        cleaned_parts = [text.strip() for text in texts if text and text.strip()]
        if not cleaned_parts:
            return ""
        joiner = f"{break_string}{break_string}" if break_string else "\n\n"
        return joiner.join(cleaned_parts).strip()

    def get_search_and_replaces(self):
        search_and_replaces = []
        if self.config.search_and_replace_file:
            with open(self.config.search_and_replace_file) as fp:
                search_and_replace_content = fp.readlines()
                for search_and_replace in search_and_replace_content:
                    if '==' in search_and_replace and not search_and_replace.startswith('==') and not search_and_replace.endswith('==') and not search_and_replace.startswith('#'):
                        search_and_replaces = search_and_replaces + [ {'search': r"{}".format(search_and_replace.split('==')[0]), 'replace': r"{}".format(search_and_replace.split('==')[1][:-1])} ]
        return search_and_replaces

    @staticmethod
    def _get_item_name(item):
        if hasattr(item, "get_name"):
            return item.get_name()
        return getattr(item, "file_name", "") or ""

    @staticmethod
    def _normalize_href_candidates(href):
        if not href:
            return []

        normalized = unquote(str(href)).split("#", 1)[0].replace("\\", "/").strip().lower()
        if not normalized:
            return []

        candidates = [normalized]
        basename = PurePosixPath(normalized).name
        if basename and basename != normalized:
            candidates.append(basename)
        return candidates

    @staticmethod
    def _sanitize_title(title, break_string) -> str:
        title = title.replace(break_string, " ")
        sanitized_title = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
        sanitized_title = re.sub(r"\s+", "_", sanitized_title.strip())
        return sanitized_title
