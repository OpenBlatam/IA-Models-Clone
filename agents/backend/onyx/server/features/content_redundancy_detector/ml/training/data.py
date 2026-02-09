"""
Data Loading Module
Modular data loading and preprocessing utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Callable, Any
import numpy as np
from PIL import Image
import io


class ImageDataset(Dataset):
    """
    Image Dataset for MobileNet training
    Proper data loading with augmentation support
    """
    
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            labels: List of labels
            transform: Optional transform function for images
            target_transform: Optional transform function for labels
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        
        if len(images) != len(labels):
            raise ValueError(f"Images ({len(images)}) and labels ({len(labels)}) must have same length")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            elif len(image.shape) == 2:
                image = Image.fromarray(image, mode='L').convert('RGB')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Convert PIL Image to tensor
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
            # Convert to tensor
            image_array = np.array(image).transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image_array).float()
        
        # Ensure proper shape (C, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class ImageBytesDataset(Dataset):
    """
    Dataset for images stored as bytes
    Useful for loading from files or API responses
    """
    
    def __init__(
        self,
        image_bytes_list: List[bytes],
        labels: List[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset from bytes
        
        Args:
            image_bytes_list: List of image bytes
            labels: List of labels
            transform: Optional transform function
            target_transform: Optional target transform
        """
        self.image_bytes_list = image_bytes_list
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.image_bytes_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_bytes = self.image_bytes_list[idx]
        label = self.labels[idx]
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to tensor
        image_array = np.array(image).transpose(2, 0, 1) / 255.0
        image = torch.from_numpy(image_array).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create DataLoader with standard settings
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test
    
    Args:
        dataset: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import random_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    return train_dataset, val_dataset, test_dataset



