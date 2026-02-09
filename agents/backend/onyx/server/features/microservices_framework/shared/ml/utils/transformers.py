"""
Data Transformers
Functional transformations for data processing.
"""

from typing import Callable, List, Any, Optional
import torch
import numpy as np
from functools import partial


class DataTransformer:
    """Base transformer class."""
    
    def __call__(self, data: Any) -> Any:
        """Transform data."""
        raise NotImplementedError


class ComposeTransformer(DataTransformer):
    """Compose multiple transformers."""
    
    def __init__(self, transformers: List[DataTransformer]):
        self.transformers = transformers
    
    def __call__(self, data: Any) -> Any:
        for transformer in self.transformers:
            data = transformer(data)
        return data


class NormalizeTransformer(DataTransformer):
    """Normalize tensor data."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std


class ToTensorTransformer(DataTransformer):
    """Convert to tensor."""
    
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
    
    def __call__(self, data: Any) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.dtype)
        elif isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=self.dtype)
        else:
            return torch.tensor(data, dtype=self.dtype)


class PadTransformer(DataTransformer):
    """Pad tensor to specified length."""
    
    def __init__(self, max_length: int, pad_value: float = 0.0):
        self.max_length = max_length
        self.pad_value = pad_value
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if len(data) >= self.max_length:
            return data[:self.max_length]
        
        padding = torch.full(
            (self.max_length - len(data),),
            self.pad_value,
            dtype=data.dtype,
            device=data.device
        )
        return torch.cat([data, padding])


class TruncateTransformer(DataTransformer):
    """Truncate tensor to specified length."""
    
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:self.max_length]


class LambdaTransformer(DataTransformer):
    """Apply lambda function transformation."""
    
    def __init__(self, func: Callable):
        self.func = func
    
    def __call__(self, data: Any) -> Any:
        return self.func(data)


def create_text_transformer_pipeline(
    tokenizer: Any,
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True,
) -> DataTransformer:
    """
    Create text transformation pipeline.
    
    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        padding: Whether to pad
        truncation: Whether to truncate
        
    Returns:
        Composed transformer
    """
    def tokenize_func(text: str) -> torch.Tensor:
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze()
    
    transformers = [
        LambdaTransformer(tokenize_func),
    ]
    
    return ComposeTransformer(transformers)


def create_image_transformer_pipeline(
    size: tuple = (224, 224),
    normalize: bool = True,
) -> DataTransformer:
    """
    Create image transformation pipeline.
    
    Args:
        size: Target size (height, width)
        normalize: Whether to normalize
        
    Returns:
        Composed transformer
    """
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform_list = [
        transforms.Resize(size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    
    def transform_func(image: Image.Image) -> torch.Tensor:
        return transform(image)
    
    return LambdaTransformer(transform_func)



