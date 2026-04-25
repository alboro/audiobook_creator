# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_python(project_root: Path) -> list[str]:
    checked: list[str] = []

    explicit = os.environ.get("EPUB_TO_AUDIOBOOK_PYTHON")
    if explicit:
        candidate = Path(explicit).expanduser()
        checked.append(str(candidate))
        if candidate.is_file() and is_usable_python([str(candidate.resolve())]):
            return [str(candidate.resolve())]
        raise FileNotFoundError(
            f"EPUB_TO_AUDIOBOOK_PYTHON is not a usable Python executable: {candidate}"
        )

    candidates = (
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file() and is_usable_python([str(candidate.resolve())]):
            return [str(candidate.resolve())]

    current = Path(sys.executable).resolve()
    checked.append(str(current))
    if is_usable_python([str(current)]):
        return [str(current)]

    if os.name == "nt":
        for command in (["py", "-3.12"], ["py", "-3.11"], ["py", "-3"]):
            checked.append(" ".join(command))
            if is_usable_python(command):
                return command

    raise FileNotFoundError(
        "A usable Python for epub_to_audiobook was not found. Checked: "
        + ", ".join(checked)
        + ". You can set EPUB_TO_AUDIOBOOK_PYTHON to an explicit Python executable."
    )


def is_usable_python(command: list[str]) -> bool:
    try:
        completed = subprocess.run(
            [*command, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    python_command = resolve_python(PROJECT_ROOT)
    main_py = PROJECT_ROOT / "main.py"
    command = [*python_command, str(main_py), *(argv if argv is not None else sys.argv[1:])]
    try:
        return subprocess.call(command, cwd=PROJECT_ROOT, env=os.environ.copy())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
