"""
Data Loading Utilities for AI Services

Provides efficient data loading and preprocessing for:
- Batch processing
- Text preprocessing
- Tokenization
- Data augmentation
"""

import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TextSample:
    """Data class for text samples"""
    text: str
    metadata: Optional[Dict[str, Any]] = None


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data
    
    Provides efficient batching and preprocessing for text data.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize text dataset
        
        Args:
            texts: List of text strings
            tokenizer: Optional tokenizer for preprocessing
            max_length: Maximum sequence length
            metadata: Optional metadata for each sample
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata = metadata or [{}] * len(texts)
        
        if len(self.metadata) != len(self.texts):
            raise ValueError("Metadata length must match texts length")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        meta = self.metadata[idx]
        
        result = {
            "text": text,
            "metadata": meta
        }
        
        if self.tokenizer:
            # Tokenize text
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["input_ids"] = encoded["input_ids"].squeeze(0)
            if "attention_mask" in encoded:
                result["attention_mask"] = encoded["attention_mask"].squeeze(0)
        
        return result


class BatchProcessor:
    """
    Utility class for batch processing of text data
    
    Handles efficient batching, preprocessing, and GPU transfer.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize batch processor
        
        Args:
            batch_size: Batch size for processing
            device: Device to move batches to
            tokenizer: Optional tokenizer for preprocessing
        """
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.tokenizer = tokenizer
    
    def create_dataloader(
        self,
        texts: List[str],
        shuffle: bool = False,
        num_workers: int = 0,
        max_length: int = 512,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> DataLoader:
        """
        Create a DataLoader for text data
        
        Args:
            texts: List of texts
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            max_length: Maximum sequence length
            metadata: Optional metadata
            
        Returns:
            DataLoader instance
        """
        dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=max_length,
            metadata=metadata
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda"
        )
    
    def process_batches(
        self,
        texts: List[str],
        process_fn: callable,
        **kwargs
    ) -> List[Any]:
        """
        Process texts in batches
        
        Args:
            texts: List of texts to process
            process_fn: Function to process each batch
            **kwargs: Additional arguments for process_fn
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = process_fn(batch, **kwargs)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch to device
        
        Args:
            batch: Batch dictionary
            
        Returns:
            Batch on the correct device
        """
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved[key] = self.move_to_device(value)
            else:
                moved[key] = value
        
        return moved


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_extra_whitespace: bool = True,
    max_length: Optional[int] = None
) -> str:
    """
    Preprocess text for model input
    
    Args:
        text: Input text
        lowercase: Whether to lowercase
        remove_extra_whitespace: Whether to remove extra whitespace
        max_length: Maximum length (truncate if longer)
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    if lowercase:
        text = text.lower()
    
    if remove_extra_whitespace:
        import re
        text = re.sub(r'\s+', ' ', text).strip()
    
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def batch_texts(texts: List[str], batch_size: int) -> Iterator[List[str]]:
    """
    Generator that yields batches of texts
    
    Args:
        texts: List of texts
        batch_size: Size of each batch
        
    Yields:
        Batches of texts
    """
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def collate_texts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for text batches
    
    Args:
        batch: List of batch items
        
    Returns:
        Collated batch
    """
    texts = [item["text"] for item in batch]
    metadata = [item.get("metadata", {}) for item in batch]
    
    result = {
        "texts": texts,
        "metadata": metadata
    }
    
    # If tokenized, stack tensors
    if "input_ids" in batch[0]:
        result["input_ids"] = torch.stack([item["input_ids"] for item in batch])
        if "attention_mask" in batch[0]:
            result["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    
    return result















