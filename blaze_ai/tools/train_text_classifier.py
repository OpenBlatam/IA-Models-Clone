from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..utils.config import load_yaml_config, deep_merge_dict
from ..utils.seed import seed_all
from ..utils.performance import make_dataloader
from ..datasets.text_dataset import build_collate_fn
from ..pipelines.functional import preprocess_texts, build_text_classification_dataset
from ..models.text_classifier import TransformerTextClassifier
from ..trainers.text_classifier_trainer import TextClassifierTrainer, TrainerConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train text classifier (TransformerEncoder)")
    p.add_argument("data", help="CSV file with columns: text,label")
    p.add_argument("--config", help="YAML config path", default=None)
    p.add_argument("--model-vocab", help="HF model for tokenizer vocab (e.g., gpt2)", default="gpt2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="./outputs/blaze_ai/train_text_classifier")
    return p.parse_args()


def _load_data(csv_path: str | Path) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    return texts, labels


def _stratified_split(texts: Sequence[str], labels: Sequence[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=val_ratio, random_state=seed, stratify=labels)
        return list(X_train), list(X_val), list(y_train), list(y_val)
    except Exception:
        # Fallback: simple split
        n = len(texts)
        k = int(n * (1 - val_ratio))
        return list(texts[:k]), list(texts[k:]), list(labels[:k]), list(labels[k:])


def _load_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("transformers is required to run this training CLI") from exc
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def main() -> int:
    args = _parse_args()
    seed_all(42)
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_yaml_config(args.config)
    # Data
    texts, labels = _load_data(args.data)
    texts = preprocess_texts(texts)
    classes = sorted({c for c in labels})
    X_tr, X_va, y_tr, y_va = _stratified_split(texts, labels, val_ratio=0.2, seed=42)
    # Tokenizer
    tokenizer = _load_tokenizer(args.model_vocab)
    # Datasets
    train_ds = build_text_classification_dataset(X_tr, y_tr, tokenizer, classes=classes, max_length=128)
    val_ds = build_text_classification_dataset(X_va, y_va, tokenizer, classes=classes, max_length=128)
    # DataLoaders
    collate = build_collate_fn(pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0)
    train_loader = make_dataloader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False, collate_fn=collate)
    val_loader = make_dataloader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False, collate_fn=collate)
    # Model
    vocab_size = int(getattr(tokenizer, "vocab_size", 50257))
    model = TransformerTextClassifier(vocab_size=vocab_size, num_classes=len(classes), embedding_dim=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1, padding_idx=getattr(tokenizer, "pad_token_id", 0) or 0)
    # Trainer
    base_trainer_cfg = {
        "epochs": int(args.epochs),
        "device": "cuda" if False else "cpu",  # force CPU in generic env
        "use_fp16": False,
        "output_dir": str(args.out),
        "log_every_steps": 10,
        "reduce_on_plateau": True,
        "save_best_only": True,
        "save_last": True,
    }
    merged = deep_merge_dict(base_trainer_cfg, cfg.get("trainer"))
    trainer_cfg = TrainerConfig(**merged)  # type: ignore[arg-type]
    trainer = TextClassifierTrainer(model, trainer_cfg)
    result = trainer.train(train_loader, val_loader)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


