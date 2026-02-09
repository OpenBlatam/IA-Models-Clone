"""
Data Loading Module
===================
Efficient data loading and preprocessing for psychological analysis
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import structlog
import numpy as np

logger = structlog.get_logger()


@dataclass
class PsychologicalDataPoint:
    """Single data point for psychological analysis"""
    text: str
    personality_labels: Optional[Dict[str, float]] = None
    sentiment_label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PsychologicalDataset(Dataset):
    """
    Dataset for psychological analysis
    Implements efficient data loading and preprocessing
    """
    
    def __init__(
        self,
        data_points: List[PsychologicalDataPoint],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        task: str = "personality"  # "personality", "sentiment", "both"
    ):
        """
        Initialize dataset
        
        Args:
            data_points: List of data points
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            task: Task type
        """
        self.data_points = data_points
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
        logger.info(
            "Dataset initialized",
            size=len(data_points),
            task=task,
            max_length=max_length
        )
    
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item with proper tokenization and error handling
        
        Args:
            idx: Index
            
        Returns:
            Processed data point
        """
        try:
            data_point = self.data_points[idx]
            
            # Tokenize text
            encoding = self.tokenizer(
                data_point.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            result = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "text": data_point.text
            }
            
            # Add labels based on task
            if self.task in ["personality", "both"] and data_point.personality_labels:
                traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
                result["personality_labels"] = torch.tensor([
                    data_point.personality_labels.get(trait, 0.5)
                    for trait in traits
                ], dtype=torch.float32)
            
            if self.task in ["sentiment", "both"] and data_point.sentiment_label:
                sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}
                result["sentiment_label"] = torch.tensor(
                    sentiment_map.get(data_point.sentiment_label, 1),
                    dtype=torch.long
                )
            
            if data_point.metadata:
                result["metadata"] = data_point.metadata
            
            return result
            
        except Exception as e:
            logger.error("Error processing data point", idx=idx, error=str(e))
            # Return empty/default item
            return self._get_default_item()
    
    def _get_default_item(self) -> Dict[str, Any]:
        """Return default item on error"""
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "text": "",
            "personality_labels": torch.zeros(5, dtype=torch.float32),
            "sentiment_label": torch.tensor(1, dtype=torch.long)
        }


class DataLoaderFactory:
    """Factory for creating data loaders with proper configuration"""
    
    @staticmethod
    def create_data_loader(
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False
    ) -> DataLoader:
        """
        Create optimized DataLoader
        
        Args:
            dataset: Dataset
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Prefetch factor
            drop_last: Drop last incomplete batch
            
        Returns:
            Configured DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            persistent_workers=num_workers > 0
        )
    
    @staticmethod
    def split_dataset(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/val/test
        
        Args:
            dataset: Full dataset
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio
            
        Returns:
            Tuple of (train, val, test) datasets
        """
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(
            "Dataset split",
            train_size=len(train_dataset),
            val_size=len(val_dataset),
            test_size=len(test_dataset)
        )
        
        return train_dataset, val_dataset, test_dataset


class DataPreprocessor:
    """Preprocessor for psychological data"""
    
    def __init__(self, tokenizer_name: str = "distilbert-base-uncased"):
        """
        Initialize preprocessor
        
        Args:
            tokenizer_name: Name of tokenizer
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error("Error loading tokenizer", error=str(e))
            self.tokenizer = None
    
    def preprocess_texts(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess batch of texts
        
        Args:
            texts: List of texts
            max_length: Maximum length
            
        Returns:
            Tokenized texts
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        try:
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"]
            }
        except Exception as e:
            logger.error("Error preprocessing texts", error=str(e))
            raise
    
    def create_data_points(
        self,
        texts: List[str],
        personality_labels: Optional[List[Dict[str, float]]] = None,
        sentiment_labels: Optional[List[str]] = None
    ) -> List[PsychologicalDataPoint]:
        """
        Create data points from raw data
        
        Args:
            texts: List of texts
            personality_labels: Optional personality labels
            sentiment_labels: Optional sentiment labels
            
        Returns:
            List of data points
        """
        data_points = []
        
        for i, text in enumerate(texts):
            data_point = PsychologicalDataPoint(
                text=text,
                personality_labels=personality_labels[i] if personality_labels else None,
                sentiment_label=sentiment_labels[i] if sentiment_labels else None
            )
            data_points.append(data_point)
        
        return data_points




