from __future__ import annotations

from blaze_ai.utils.naming import is_descriptive, enforce_descriptive_names


def test_is_descriptive() -> None:
    assert is_descriptive("num_epochs")
    assert is_descriptive("learning_rate")
    assert not is_descriptive("x")
    assert not is_descriptive("i")


def test_enforce_descriptive_names() -> None:
    out = enforce_descriptive_names(["x", "y", "num_layers", "model", "n"])
    assert out == ["num_layers", "model"]


