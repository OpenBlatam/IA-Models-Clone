from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..datasets.text_dataset import LabeledTextDataset, build_collate_fn
from ..evaluation import evaluate_classification
from ..trainers.text_classifier_trainer import TextClassifierTrainer, TrainerConfig
from ..utils.performance import make_dataloader
from .. import create_modular_ai
from ..core import SystemMode


def example_dataloader(texts: Sequence[str], labels: Sequence[int], tokenizer, batch_size: int = 8) -> DataLoader:
    ds = LabeledTextDataset(texts, labels, tokenizer, max_length=64)
    collate = build_collate_fn(pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0)
    return make_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=False,
        collate_fn=collate,
    )


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(50257, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.fc(x)


def example_train_and_eval(train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
    model = TinyClassifier(num_classes=2)
    trainer = TextClassifierTrainer(model, TrainerConfig(epochs=1, log_every_steps=10, use_fp16=False))
    result = trainer.train(train_loader, val_loader)
    return result


async def example_llm_generate(prompt: str) -> Dict[str, Any]:
    ai = await create_modular_ai(system_mode=SystemMode.PRODUCTION)
    return await ai.process({"_engine": "llm.generate", "prompt": prompt, "overrides": {"max_new_tokens": 24}})


async def example_diffusion_generate(prompt: str) -> Dict[str, Any]:
    ai = await create_modular_ai(system_mode=SystemMode.PRODUCTION)
    return await ai.process({
        "_engine": "diffusion.generate",
        "prompt": prompt,
        "overrides": {"width": 512, "height": 512, "num_inference_steps": 10},
    })


def _demo() -> None:  # pragma: no cover
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        print("Install transformers to run the dataloader/training demo")
        return
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    texts = ["great product", "bad service", "awesome", "terrible"] * 4
    labels = [1, 0, 1, 0] * 4
    train_loader = example_dataloader(texts, labels, tok, batch_size=8)
    val_loader = example_dataloader(texts, labels, tok, batch_size=8)
    out = example_train_and_eval(train_loader, val_loader)
    print(out)
    print(asyncio.run(example_llm_generate("Say hi")))


if __name__ == "__main__":  # pragma: no cover
    _demo()


