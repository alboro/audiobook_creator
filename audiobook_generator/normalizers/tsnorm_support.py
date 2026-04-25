# Copyright (c) 2026 alboro <alboro@users.noreply.github.com>
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from __future__ import annotations

import importlib
import importlib.util
import io
import pickle
import sys
import types
from importlib import resources
from pathlib import Path


def install_pkg_resources_compat():
    try:
        import pkg_resources  # noqa: F401
        return
    except ImportError:
        pass

    compat_module = types.ModuleType("pkg_resources")

    def resource_stream(package_or_requirement, resource_name):
        package_name = _resolve_package_name(package_or_requirement)
        data = resources.files(package_name).joinpath(resource_name).read_bytes()
        return io.BytesIO(data)

    compat_module.resource_stream = resource_stream
    sys.modules["pkg_resources"] = compat_module


def load_tsnorm_backend():
    install_pkg_resources_compat()
    from tsnorm import Normalizer as TSNormBackend

    return TSNormBackend


def create_tsnorm_backend(
    *,
    stress_mark: str,
    stress_mark_pos: str,
    stress_monosyllabic: bool,
    stress_yo: bool,
    min_word_len: int,
):
    backend_class = load_tsnorm_backend()
    return backend_class(
        stress_mark=stress_mark,
        stress_mark_pos=stress_mark_pos,
        stress_yo=stress_yo,
        stress_monosyllabic=stress_monosyllabic,
        min_word_len=min_word_len,
    )


def load_tsnorm_dictionary_data():
    package_dir = _resolve_package_dir("tsnorm")
    dictionary_dir = package_dir / "dictionary"
    with (dictionary_dir / "wordforms.dat").open("rb") as stream:
        word_forms = pickle.load(stream)
    with (dictionary_dir / "lemmas.dat").open("rb") as stream:
        lemmas = pickle.load(stream)
    return word_forms, lemmas


def _resolve_package_name(package_or_requirement):
    if hasattr(package_or_requirement, "__spec__"):
        package_name = package_or_requirement.__spec__.name
    else:
        package_name = str(package_or_requirement)

    module = importlib.import_module(package_name)
    if getattr(module, "__spec__", None) and module.__spec__.submodule_search_locations is None:
        return module.__package__ or package_name.rpartition(".")[0]
    return package_name


def _resolve_package_dir(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if not spec or not spec.submodule_search_locations:
        raise ImportError(f"Unable to locate package directory for {package_name!r}")
    return Path(next(iter(spec.submodule_search_locations)))
