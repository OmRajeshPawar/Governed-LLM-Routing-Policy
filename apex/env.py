from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _candidate_dirs(starts: Iterable[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for start in starts:
        current = start if start.is_dir() else start.parent
        try:
            current = current.resolve()
        except OSError:
            continue
        for directory in (current, *current.parents):
            if directory in seen:
                continue
            seen.add(directory)
            dirs.append(directory)
    return dirs


def _fallback_load(path: Path, *, override: bool = False) -> bool:
    loaded = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded = True
    return loaded


def load_project_env(*starts: str | Path, override: bool = False) -> list[Path]:
    start_paths = [Path.cwd()]
    start_paths.extend(Path(s) for s in starts if s)
    loaded: list[Path] = []

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    for directory in _candidate_dirs(start_paths):
        env_path = directory / ".env"
        if not env_path.is_file():
            continue
        if load_dotenv is not None:
            did_load = bool(load_dotenv(env_path, override=override))
        else:
            did_load = _fallback_load(env_path, override=override)
        if did_load:
            loaded.append(env_path)
    return loaded
