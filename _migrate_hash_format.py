#!/usr/bin/env python3
# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

"""Migrate sentence_hash from voice|model format to simple sha256(text).

This updates:
1. DB: sentence_hash column
2. DB: audio_path column
3. Filesystem: rename old_hash.wav -> new_hash.wav
"""

import hashlib
import os
import sqlite3
import sys
from pathlib import Path


def compute_hash(text: str) -> str:
    """Compute new hash format: sha256(text)[:16]."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def migrate_audio_chunks(db_path: str, dry_run: bool = True):
    """Migrate audio_chunks table and rename files."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get all records
    rows = conn.execute("SELECT id, chapter_key, sentence_pos, sentence_hash, sentence_text, audio_path FROM audio_chunks").fetchall()

    print(f"Found {len(rows)} records in audio_chunks")

    updates = []
    for row in rows:
        old_hash = row["sentence_hash"]
        new_hash = compute_hash(row["sentence_text"])

        if old_hash == new_hash:
            continue  # Already migrated

        old_path = Path(row["audio_path"]) if row["audio_path"] else None
        new_path = None

        if old_path and old_path.exists():
            # Compute new path
            ext = old_path.suffix
            new_path = old_path.parent / f"{new_hash}{ext}"
            print(f"  {row['chapter_key']}/{row['sentence_pos']}: {old_hash} -> {new_hash}")

        updates.append({
            "id": row["id"],
            "old_hash": old_hash,
            "new_hash": new_hash,
            "old_path": old_path,
            "new_path": new_path,
        })

    if not updates:
        print("No records need migration")
        conn.close()
        return

    print(f"\n{len(updates)} records need migration (showing first 10):")
    for u in updates[:10]:
        print(f"  {u['old_hash']} -> {u['new_hash']}")
    if len(updates) > 10:
        print(f"  ... and {len(updates) - 10} more")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run with --apply to apply changes.")
        conn.close()
        return

    # Apply changes
    print("\nApplying changes...")

    for u in updates:
        # Update DB
        new_audio_path = str(u["new_path"]) if u["new_path"] else None
        conn.execute(
            "UPDATE audio_chunks SET sentence_hash = ?, audio_path = ? WHERE id = ?",
            (u["new_hash"], new_audio_path, u["id"])
        )

        # Rename file
        if u["new_path"] and u["old_path"].exists() and not u["new_path"].exists():
            u["old_path"].rename(u["new_path"])
            print(f"  Renamed: {u['old_path'].name} -> {u['new_path'].name}")

    conn.commit()
    conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    dry_run = "--apply" not in sys.argv

    if len(sys.argv) < 2:
        print("Usage: python _migrate_hash_format.py <db_path> [--apply]")
        print("  db_path: Path to audio_chunks.sqlite3")
        print("  --apply: Actually apply changes (default is dry-run)")
        sys.exit(1)

    db_path = sys.argv[1]
    if not Path(db_path).exists():
        print(f"Error: DB not found: {db_path}")
        sys.exit(1)

    print(f"Migrating: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY'}")

    migrate_audio_chunks(db_path, dry_run=dry_run)
