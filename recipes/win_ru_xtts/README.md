# win_ru_xtts

This recipe is now a thin Windows-friendly launcher for `main.py`.

It does not define XTTS, polling, normalizer, language, voice, chapter, or packaging defaults. Pass the same arguments you would pass to:

```bash
.venv/bin/python main.py ...
```

On Windows, the wrapper only adapts execution to the local project layout:

- finds a usable project Python from `.venv/Scripts/python.exe` or `.venv/bin/python`
- optionally uses `EPUB_TO_AUDIOBOOK_PYTHON`
- falls back to the Python currently running this wrapper
- falls back to Windows `py -3.12`, `py -3.11`, then `py -3` if no working venv Python is found
- runs `main.py` from the repository root
- forwards every CLI argument unchanged

Example:

```powershell
python recipes/win_ru_xtts/run_book.py `
  "G:\path\to\book.epub" `
  "G:\path\to\out" `
  --tts openai `
  --language ru-RU `
  --no_prompt
```

For machine-specific defaults, keep a local wrapper outside the repo or under an ignored folder such as `.local/`, and let that wrapper call this script with the desired `main.py` arguments.
