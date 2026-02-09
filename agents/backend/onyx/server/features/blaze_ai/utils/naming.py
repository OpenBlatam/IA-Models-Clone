from __future__ import annotations

from typing import Iterable


def is_descriptive(name: str) -> bool:
    return len(name) >= 3 and name.isidentifier() and not name.lower() in {"x", "y", "z", "n", "i", "j", "k"}


def enforce_descriptive_names(names: Iterable[str]) -> list[str]:
    return [n for n in names if is_descriptive(n)]


