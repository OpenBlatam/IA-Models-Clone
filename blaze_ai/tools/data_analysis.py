from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    inferred_type: str
    missing_count: int
    missing_pct: float
    nunique: int
    sample_values: List[Any]
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None


def _infer_semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    return "categorical" if series.nunique(dropna=True) <= max(20, int(0.05 * len(series))) else "text"


def _numeric_stats(series: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        ser = pd.to_numeric(series, errors="coerce")
        if ser.notna().sum() == 0:
            return (None, None, None, None, None, None, None)
        q = ser.quantile([0.25, 0.5, 0.75])
        return (
            float(ser.min()),
            float(ser.max()),
            float(ser.mean()),
            float(ser.std() if ser.notna().sum() > 1 else 0.0),
            float(q.loc[0.25]),
            float(q.loc[0.5]),
            float(q.loc[0.75]),
        )
    except Exception:
        return (None, None, None, None, None, None, None)


def summarize_columns(df: pd.DataFrame, sample_values_per_col: int = 5) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        inferred = _infer_semantic_type(series)
        missing = int(series.isna().sum())
        missing_pct = float(100.0 * missing / max(1, len(series)))
        nunique = int(series.nunique(dropna=True))
        sample = series.dropna().head(sample_values_per_col).tolist()
        min_v = max_v = mean_v = std_v = q25 = med = q75 = None
        if inferred == "numeric":
            min_v, max_v, mean_v, std_v, q25, med, q75 = _numeric_stats(series)
        item = ColumnSummary(
            name=str(col),
            dtype=str(series.dtype),
            inferred_type=inferred,
            missing_count=missing,
            missing_pct=missing_pct,
            nunique=nunique,
            sample_values=sample,
            min=min_v,
            max=max_v,
            mean=mean_v,
            std=std_v,
            q25=q25,
            median=med,
            q75=q75,
        )
        summaries.append(asdict(item))
    return summaries


def analyze_target(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        return {"present": False, "reason": "target_not_in_columns"}
    ser = df[target]
    stype = _infer_semantic_type(ser)
    result: Dict[str, Any] = {"present": True, "inferred_type": stype}
    if stype in ("categorical", "bool"):
        counts = ser.value_counts(dropna=True)
        num_classes = int(counts.shape[0])
        major = int(counts.iloc[0]) if num_classes > 0 else 0
        total = int(counts.sum())
        imbalance = float(major / max(1, total)) if total else 0.0
        result.update(
            {
                "num_classes": num_classes,
                "class_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
                "majority_fraction": imbalance,
            }
        )
    elif stype == "numeric":
        mn, mx, mean, std, q25, med, q75 = _numeric_stats(ser)
        result.update({"min": mn, "max": mx, "mean": mean, "std": std, "q25": q25, "median": med, "q75": q75})
    return result


def analyze_text(df: pd.DataFrame, text_col: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text_col or text_col not in df.columns:
        return None
    ser = df[text_col].astype(str)
    lengths = ser.str.len()
    mn, mx, mean, std, q25, med, q75 = _numeric_stats(lengths)
    return {
        "column": text_col,
        "len_min": mn,
        "len_max": mx,
        "len_mean": mean,
        "len_std": std,
        "len_q25": q25,
        "len_median": med,
        "len_q75": q75,
    }


def _detect_problem_type(target_info: Dict[str, Any]) -> Optional[str]:
    if not target_info.get("present"):
        return None
    if target_info.get("inferred_type") in ("categorical", "bool"):
        return "classification"
    if target_info.get("inferred_type") == "numeric":
        return "regression"
    return None


def build_recommendations(df: pd.DataFrame, target_info: Dict[str, Any], text_info: Optional[Dict[str, Any]]) -> List[str]:
    recs: List[str] = []
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        recs.append(f"Remove {dup_rows} duplicate rows")
    missing_cells = int(df.isna().sum().sum())
    if missing_cells > 0:
        recs.append("Handle missing values (impute or drop)")
    if target_info.get("present") and target_info.get("inferred_type") in ("categorical", "bool"):
        maj = float(target_info.get("majority_fraction", 0.0))
        if maj >= 0.8:
            recs.append("Severe class imbalance detected; consider stratified split, class weights or resampling")
    if text_info is not None and (text_info.get("len_mean") or 0) > 2000:
        recs.append("Long text sequences; enable truncation/gradient_accumulation and efficient tokenization")
    return recs


def analyze_dataset(
    input_path: str | Path,
    target: Optional[str] = None,
    text_col: Optional[str] = None,
    sample_rows: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    # Load
    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json"}:
        df = pd.read_json(path, lines=False)
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type. Use CSV/JSON/Parquet")
    if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)
    # Overview
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    columns_meta = summarize_columns(df)
    target_info = analyze_target(df, target) if target else {"present": False}
    text_info = analyze_text(df, text_col)
    problem_type = _detect_problem_type(target_info)
    recommendations = build_recommendations(df, target_info, text_info)
    return {
        "ok": True,
        "dataset_overview": {
            "path": str(path),
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "memory_mb": round(memory_mb, 3),
            "column_names": df.columns.tolist(),
        },
        "columns": columns_meta,
        "target": target_info,
        "text": text_info,
        "problem_definition": {
            "suggested_type": problem_type,
            "has_text": bool(text_info is not None),
        },
        "quality": {
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_cells": int(df.isna().sum().sum()),
        },
        "recommendations": recommendations,
    }


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset analysis and problem definition")
    parser.add_argument("input", help="Path to CSV/JSON/Parquet")
    parser.add_argument("--target", help="Target column name", default=None)
    parser.add_argument("--text-col", help="Text column name (optional)", default=None)
    parser.add_argument("--sample-rows", type=int, default=None, help="Sample N rows for quick EDA")
    parser.add_argument("--out", help="Output JSON file (optional)", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    report = analyze_dataset(args.input, target=args.target, text_col=args.text_col, sample_rows=args.sample_rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


