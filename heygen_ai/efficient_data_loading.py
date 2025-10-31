from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
from typing import List, Tuple, Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Efficient Data Loading for Diffusion Model Training
Implements a robust PyTorch Dataset and DataLoader for image-text pairs.
"""


class ImageTextDataset(Dataset):
    """
    Custom Dataset for loading image-text pairs for diffusion model training.
    Expects a CSV file with columns: 'image_path', 'prompt'.
    """
    def __init__(self, csv_file: str, image_root: str, transform: Optional[Callable] = None, tokenizer: Optional[Callable] = None, max_length: int = 77):
        
    """__init__ function."""
self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # For diffusion models
        ])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> Any:
        return len(self.data)

    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, row['image_path'])
        prompt = row['prompt']

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        image = self.transform(image)

        # Tokenize prompt if tokenizer is provided
        if self.tokenizer:
            prompt_ids = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.squeeze(0)
        else:
            prompt_ids = prompt

        return {
            'image': image,
            'prompt': prompt,
            'prompt_ids': prompt_ids
        }

def create_dataloader(
    csv_file: str,
    image_root: str,
    tokenizer: Optional[Callable] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    max_length: int = 77
) -> DataLoader:
    """
    Create a PyTorch DataLoader for efficient batching and shuffling.
    """
    dataset = ImageTextDataset(
        csv_file=csv_file,
        image_root=image_root,
        tokenizer=tokenizer,
        max_length=max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    return dataloader

def split_dataset_csv(
    csv_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split a CSV file into train/val/test CSVs and return their paths.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    df = pd.read_csv(csv_file)
    if shuffle:
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    train_csv = csv_file.replace('.csv', '_train.csv')
    val_csv = csv_file.replace('.csv', '_val.csv')
    test_csv = csv_file.replace('.csv', '_test.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_csv, val_csv, test_csv

def create_kfold_splits(
    csv_file: str,
    k: int = 5,
    shuffle: bool = True,
    random_seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Create k-fold train/val splits from a CSV file. Returns list of (train_csv, val_csv) paths.
    """
    df = pd.read_csv(csv_file)
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_seed)
    splits = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        train_csv = csv_file.replace('.csv', f'_train_fold{fold+1}.csv')
        val_csv = csv_file.replace('.csv', f'_val_fold{fold+1}.csv')
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        splits.append((train_csv, val_csv))
    return splits

def demonstrate_data_loading():
    """
    Demonstrate efficient data loading for diffusion model training, including splits and cross-validation.
    """
    # Example usage (replace with your actual paths and tokenizer)
    csv_file = 'data/train.csv'  # CSV with columns: image_path,prompt
    image_root = 'data/images/'
    batch_size = 4

    # Split into train/val/test
    train_csv, val_csv, test_csv = split_dataset_csv(csv_file)
    print(f"Train CSV: {train_csv}\nVal CSV: {val_csv}\nTest CSV: {test_csv}")

    # Dummy tokenizer for demonstration
    class DummyTokenizer:
        def __call__(self, text, padding, max_length, truncation, return_tensors) -> Any:
            # Simulate token IDs as random integers
            return type('obj', (object,), {'input_ids': torch.randint(0, 10000, (1, max_length))})()

    tokenizer = DummyTokenizer()

    # Create DataLoaders for each split
    train_loader = create_dataloader(
        csv_file=train_csv,
        image_root=image_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        max_length=77
    )
    val_loader = create_dataloader(
        csv_file=val_csv,
        image_root=image_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        max_length=77
    )
    test_loader = create_dataloader(
        csv_file=test_csv,
        image_root=image_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        max_length=77
    )

    print("\nIterating through a train batch:")
    for i, batch in enumerate(train_loader):
        print(f"Train Batch {i+1}")
        print(f"  Images shape: {batch['image'].shape}")
        print(f"  Prompts: {batch['prompt']}")
        print(f"  Prompt IDs shape: {batch['prompt_ids'].shape}")
        if i >= 1:
            break

    # Demonstrate k-fold cross-validation splits
    print("\nCreating 3-fold cross-validation splits:")
    kfold_splits = create_kfold_splits(csv_file, k=3)
    for fold, (train_csv, val_csv) in enumerate(kfold_splits):
        print(f"Fold {fold+1}: Train CSV: {train_csv}, Val CSV: {val_csv}")

match __name__:
    case "__main__":
    demonstrate_data_loading() 