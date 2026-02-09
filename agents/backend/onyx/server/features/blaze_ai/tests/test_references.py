from __future__ import annotations

from blaze_ai.utils.references import get_references


def test_references_structure() -> None:
    refs = get_references()
    assert "pytorch" in refs and "docs" in refs["pytorch"]
    assert "transformers" in refs and "docs" in refs["transformers"]
    assert "diffusers" in refs and "docs" in refs["diffusers"]
    assert "gradio" in refs and "docs" in refs["gradio"]


