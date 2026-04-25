# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from audiobook_generator.normalizers.ru_text_utils import strip_combining_acute
from audiobook_generator.normalizers.tsnorm_support import load_tsnorm_dictionary_data

SCHEMA_VERSION = 1
TSNORM_SOURCE = "tsnorm"
ZALIZNIAK_SOURCE = "zalizniak"
PROPER_NAME_POS_TAGS = {"PNOUN", "CHARACTER"}


@dataclass(frozen=True)
class PronunciationLexiconEntry:
    surface_form: str
    spoken_form: str | None
    lemma: str | None
    pos: str | None
    grammemes: str | None
    is_proper_name: bool
    source: str
    confidence: float | None


class PronunciationLexiconDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def lookup(self, surface_form: str, *, source: str | None = None) -> list[PronunciationLexiconEntry]:
        normalized_form = strip_combining_acute(surface_form).lower()
        query = """
            SELECT surface_form, spoken_form, lemma, pos, grammemes, is_proper_name, source, confidence
            FROM entries
            WHERE surface_form = ?
        """
        params: list[object] = [normalized_form]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY source, spoken_form, lemma, grammemes"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def lookup_spoken_forms(
        self,
        surface_form: str,
        *,
        source: str | None = None,
        only_ambiguous: bool = False,
    ) -> tuple[str, ...]:
        entries = self.lookup(surface_form, source=source)
        spoken_forms = tuple(
            sorted(
                {
                    entry.spoken_form
                    for entry in entries
                    if entry.spoken_form
                }
            )
        )
        if only_ambiguous and len(spoken_forms) < 2:
            return ()
        return spoken_forms

    def lookup_ambiguous_entries(
        self,
        surface_form: str,
        *,
        source: str | None = None,
    ) -> tuple[PronunciationLexiconEntry, ...]:
        entries = tuple(self.lookup(surface_form, source=source))
        spoken_forms = {
            entry.spoken_form
            for entry in entries
            if entry.spoken_form
        }
        if len(spoken_forms) < 2:
            return ()
        return entries

    def replace_source_entries(
        self,
        *,
        source: str,
        entries: Iterable[PronunciationLexiconEntry],
    ) -> int:
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM entries WHERE source = ?", (source,))
            batch: list[tuple[object, ...]] = []
            for entry in entries:
                batch.append(
                    (
                        entry.surface_form,
                        entry.spoken_form,
                        entry.lemma,
                        entry.pos,
                        entry.grammemes,
                        int(entry.is_proper_name),
                        entry.source,
                        entry.confidence,
                    )
                )
                if len(batch) >= 5000:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO entries (
                            surface_form,
                            spoken_form,
                            lemma,
                            pos,
                            grammemes,
                            is_proper_name,
                            source,
                            confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO entries (
                        surface_form,
                        spoken_form,
                        lemma,
                        pos,
                        grammemes,
                        is_proper_name,
                        source,
                        confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
            count = connection.execute(
                "SELECT COUNT(*) FROM entries WHERE source = ?",
                (source,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (f"source:{source}:entry_count", str(count)),
            )
            connection.commit()
        return count

    def set_metadata(self, key: str, value: str):
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )
            connection.commit()

    def get_metadata(self, key: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def get_stats(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            total_entries = connection.execute(
                "SELECT COUNT(*) FROM entries"
            ).fetchone()[0]
            ambiguous_surface_forms = connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT surface_form
                    FROM entries
                    WHERE spoken_form IS NOT NULL
                    GROUP BY surface_form
                    HAVING COUNT(DISTINCT spoken_form) > 1
                )
                """
            ).fetchone()[0]
            proper_name_entries = connection.execute(
                "SELECT COUNT(*) FROM entries WHERE is_proper_name = 1"
            ).fetchone()[0]
        return {
            "total_entries": total_entries,
            "ambiguous_surface_forms": ambiguous_surface_forms,
            "proper_name_entries": proper_name_entries,
        }

    def count_source_entries(self, source: str) -> int:
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM entries WHERE source = ?",
                (source,),
            ).fetchone()[0]

    def _initialize(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    surface_form TEXT NOT NULL,
                    spoken_form TEXT,
                    lemma TEXT,
                    pos TEXT,
                    grammemes TEXT,
                    is_proper_name INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL,
                    confidence REAL
                );

                CREATE INDEX IF NOT EXISTS idx_entries_surface_form
                ON entries(surface_form);

                CREATE INDEX IF NOT EXISTS idx_entries_source_surface
                ON entries(source, surface_form);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_unique
                ON entries(
                    surface_form,
                    IFNULL(spoken_form, ''),
                    IFNULL(lemma, ''),
                    IFNULL(pos, ''),
                    IFNULL(grammemes, ''),
                    is_proper_name,
                    source
                );

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            connection.commit()
        schema_version = self.get_metadata("schema_version")
        if schema_version != str(SCHEMA_VERSION):
            self.set_metadata("schema_version", str(SCHEMA_VERSION))

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> PronunciationLexiconEntry:
        return PronunciationLexiconEntry(
            surface_form=row["surface_form"],
            spoken_form=row["spoken_form"],
            lemma=row["lemma"],
            pos=row["pos"],
            grammemes=row["grammemes"],
            is_proper_name=bool(row["is_proper_name"]),
            source=row["source"],
            confidence=row["confidence"],
        )


def get_default_pronunciation_lexicon_db_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".cache" / "ru_pronunciation_lexicon.sqlite3"


def ensure_pronunciation_lexicon_db(
    path: str | Path | None = None,
    *,
    include_zalizniak: bool = False,
) -> PronunciationLexiconDB:
    """
    Open (or create) the pronunciation lexicon DB and ensure all requested
    sources are built.

    Parameters
    ----------
    path:
        Path to the SQLite file.  Defaults to
        ``.cache/ru_pronunciation_lexicon.sqlite3`` in the project root.
    include_zalizniak:
        When True, also populate the ``zalizniak`` source (lemma-level stress
        from gramdict/zalizniak-2010).  Requires internet access on first run;
        data is then cached in ``.cache/zalizniak/``.
    """
    database = PronunciationLexiconDB(path or get_default_pronunciation_lexicon_db_path())
    built_sources = json.loads(database.get_metadata("built_sources") or "[]")

    if TSNORM_SOURCE not in built_sources:
        if database.count_source_entries(TSNORM_SOURCE) == 0:
            build_tsnorm_pronunciation_lexicon(database)
        built_sources = sorted(set(built_sources + [TSNORM_SOURCE]))
        database.set_metadata("built_sources", json.dumps(built_sources, ensure_ascii=False))

    if include_zalizniak and ZALIZNIAK_SOURCE not in built_sources:
        if database.count_source_entries(ZALIZNIAK_SOURCE) == 0:
            build_zalizniak_pronunciation_lexicon(database)
        built_sources = sorted(set(built_sources + [ZALIZNIAK_SOURCE]))
        database.set_metadata("built_sources", json.dumps(built_sources, ensure_ascii=False))

    return database


def build_zalizniak_pronunciation_lexicon(
    database: PronunciationLexiconDB,
    *,
    force_refresh: bool = False,
) -> int:
    """
    Populate the DB with lemma-level stress entries sourced from the
    gramdict/zalizniak-2010 repository (CC BY-NC).

    On first call the text files are downloaded from GitHub and cached in
    ``.cache/zalizniak/``.  Subsequent calls reuse the cached files unless
    ``force_refresh=True``.

    Returns the number of entries stored.
    """
    from audiobook_generator.normalizers.zalizniak_support import iter_zalizniak_entries

    entries = iter_zalizniak_entries(force_refresh=force_refresh)
    count = database.replace_source_entries(source=ZALIZNIAK_SOURCE, entries=entries)
    database.set_metadata(
        "source:zalizniak:stats",
        json.dumps(database.get_stats(), ensure_ascii=False),
    )
    return count


def build_tsnorm_pronunciation_lexicon(
    database: PronunciationLexiconDB,
    *,
    word_forms: dict | None = None,
    lemmas: dict | None = None,
) -> int:
    if word_forms is None or lemmas is None:
        word_forms, lemmas = load_tsnorm_dictionary_data()
    entries = iter_tsnorm_lexicon_entries(word_forms=word_forms, lemmas=lemmas)
    count = database.replace_source_entries(source=TSNORM_SOURCE, entries=entries)
    database.set_metadata("source:tsnorm:stats", json.dumps(database.get_stats(), ensure_ascii=False))
    return count


def iter_tsnorm_lexicon_entries(
    *,
    word_forms: dict,
    lemmas: dict,
) -> Iterable[PronunciationLexiconEntry]:
    for interpretations in word_forms.values():
        for interpretation in interpretations:
            word_form = interpretation.get("word_form")
            if not word_form:
                continue
            surface_form = strip_combining_acute(word_form).lower()
            lemma = interpretation.get("lemma")
            lemma_info = lemmas.get(lemma or "", {})
            pos_list = tuple(sorted(set(lemma_info.get("pos") or ())))
            pos = "|".join(pos_list) if pos_list else None
            grammemes = interpretation.get("form_tags") or None
            is_proper_name = any(tag in PROPER_NAME_POS_TAGS for tag in pos_list)
            spoken_form = _apply_stress_positions(
                word_form,
                interpretation.get("stress_pos") or (),
            )
            if strip_combining_acute(spoken_form) == surface_form and spoken_form == word_form:
                spoken_form = word_form
            yield PronunciationLexiconEntry(
                surface_form=surface_form,
                spoken_form=spoken_form,
                lemma=lemma,
                pos=pos,
                grammemes=grammemes,
                is_proper_name=is_proper_name,
                source=TSNORM_SOURCE,
                confidence=1.0,
            )


def _apply_stress_positions(word_form: str, stress_positions: Iterable[int]) -> str:
    result = list(word_form)
    inserts = 0
    for position in stress_positions:
        result.insert(position + 1 + inserts, "\u0301")
        inserts += 1
    return "".join(result)
