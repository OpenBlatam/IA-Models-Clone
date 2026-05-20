from __future__ import annotations

from typing import Any, Optional


def prepare_tokenizer(
    tokenizer: Optional[Any] = None,
    model_name_or_path: Optional[str] = None,
    *,
    padding_side: str = "left",
):
    if tokenizer is None:
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required when tokenizer is None")
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError("transformers is required to load tokenizer") from exc
        tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
    # Ensure PAD is set for batched generation/training
    try:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id  # type: ignore[attr-defined]
    except Exception:
        pass
    # Normalize padding side
    try:
        tokenizer.padding_side = padding_side  # type: ignore[attr-defined]
    except Exception:
        pass
    return tokenizer


