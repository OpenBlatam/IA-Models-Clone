from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def _inner(x: Any) -> Any:
        for f in funcs:
            x = f(x)
        return x
    return _inner


def map_fn(fn: Callable[[Any], Any], it: Iterable[Any]) -> Iterator[Any]:
    for x in it:
        yield fn(x)


def filter_fn(pred: Callable[[Any], bool], it: Iterable[Any]) -> Iterator[Any]:
    for x in it:
        if pred(x):
            yield x


_WS = re.compile(r"\s+")


def normalize_whitespace(s: str) -> str:
    return _WS.sub(" ", s).strip()


def to_lower(s: str) -> str:
    return s.lower()


def drop_empty(s: str) -> bool:
    return bool(s and s.strip())


def filter_min_length(n: int) -> Callable[[str], bool]:
    def _pred(s: str) -> bool:
        return len(s) >= n
    return _pred


def map_labels(classes: Sequence[str]) -> Callable[[str], int]:
    lut = {c: i for i, c in enumerate(classes)}
    def _map(c: str) -> int:
        return int(lut[c])
    return _map


def preprocess_texts(texts: Sequence[str]) -> List[str]:
    pipeline = compose(normalize_whitespace, to_lower)
    return list(map_fn(pipeline, filter_fn(drop_empty, texts)))


def build_text_classification_dataset(
    texts: Sequence[str],
    labels: Sequence[str] | Sequence[int],
    tokenizer,
    classes: Sequence[str] | None = None,
    max_length: int = 128,
):
    from ..datasets.text_dataset import LabeledTextDataset

    if classes is not None and isinstance(labels[0], str):  # type: ignore[index]
        label_mapper = map_labels(classes)
        labels = [label_mapper(c) for c in labels]  # type: ignore[assignment]
    return LabeledTextDataset(texts, labels, tokenizer, max_length=max_length)


