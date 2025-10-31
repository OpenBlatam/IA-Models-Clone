from typing import List, Dict, Any
import torch

from factories.registry import Registry

COLLATE = Registry()


@COLLATE.register("lm")
def build_lm_collate(tokenizer, max_length: int):
    def collate_fn(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        tokens = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn


@COLLATE.register("cv")
def build_cv_collate():
    def collate_fn(samples):
        return samples
    return collate_fn



