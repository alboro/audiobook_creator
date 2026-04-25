# Recipes

This folder contains reusable launch recipes for specific deployment setups.

Principles:

- recipes are commit-friendly
- recipes should avoid hardcoding personal secrets or machine-specific paths
- local convenience wrappers can live outside the repo or under ignored folders such as `.local/`

Current recipe:

- `win_ru_xtts`
  A thin Windows-friendly launcher for `main.py`. It finds a usable local Python and forwards
  all arguments unchanged, so deployment-specific defaults should live in local wrappers.
