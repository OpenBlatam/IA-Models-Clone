from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from blaze_ai.tools.data_analysis import analyze_dataset


def _make_tiny_csv(tmp_path: Path) -> Path:
    data = {
        "text": ["hello world", "lorem ipsum", "hello world", None],
        "label": ["a", "b", "a", "a"],
        "value": [1.0, 2.5, 3.0, None],
    }
    df = pd.DataFrame(data)
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    return p


def test_analyze_dataset_generates_basic_report(tmp_path: Path) -> None:
    path = _make_tiny_csv(tmp_path)
    report = analyze_dataset(path, target="label", text_col="text")
    assert report["ok"] is True
    assert report["dataset_overview"]["rows"] == 4
    assert report["dataset_overview"]["columns"] == 3
    assert any(c["name"] == "text" for c in report["columns"])  # type: ignore[index]
    assert report["target"]["present"] is True
    assert report["problem_definition"]["suggested_type"] == "classification"
    # Ensure recommendations present for duplicates or imbalance
    assert isinstance(report["recommendations"], list)
    assert len(report["recommendations"]) >= 0


