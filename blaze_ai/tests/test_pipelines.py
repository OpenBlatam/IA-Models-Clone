from __future__ import annotations

from blaze_ai.pipelines import preprocess_texts, compose, normalize_whitespace, to_lower, drop_empty, filter_min_length


def test_preprocess_texts_basic() -> None:
    xs = ["  Hello  WORLD  ", "", "  ", "Test"]
    out = preprocess_texts(xs)
    assert out == ["hello world", "test"]


def test_compose_and_filters() -> None:
    f = compose(normalize_whitespace, to_lower)
    s = f("  A   B  C  ")
    assert s == "a b c"
    assert drop_empty(" ") is False
    assert filter_min_length(3)("ab") is False


