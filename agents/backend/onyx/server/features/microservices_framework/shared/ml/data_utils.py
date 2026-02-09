"""
Shared ML Data Utilities
Common utilities for data loading, preprocessing, and augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Generic text dataset for language models."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding="max_length" if self.padding else False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with common settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split dataset into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )
    
    return train_dataset, val_dataset, test_dataset


def collate_fn_padding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function with padding for variable-length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item.get("attention_mask", torch.ones_like(ids)) for item, ids in zip(batch, input_ids)]
    
    # Pad sequences
    max_length = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        pad_length = max_length - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)]))
        padded_attention_masks.append(torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)]))
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
    }


def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Convert to lowercase (optional, depending on use case)
    # text = text.lower()
    return text


def create_vocab_from_texts(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    """Create vocabulary from texts."""
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    
    # Filter by minimum frequency
    vocab = {word: idx + 1 for idx, (word, count) in enumerate(
        (word for word, count in word_counts.items() if count >= min_freq)
    )}
    
    # Add special tokens
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)
    
    return vocab



