"""
Refactored Data Manager for HeyGen AI

This module provides clean, efficient data handling following deep learning best practices
with proper validation, error handling, and optimized data loading.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass, field
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image
import pandas as pd

from transformers import (
    AutoTokenizer, 
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    # File paths
    train_file: str = "data/train.json"
    validation_file: str = "data/validation.json"
    test_file: str = "data/test.json"
    
    # Text processing
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    
    # DataLoader settings
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    
    # Validation
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42


class TextDataset(Dataset):
    """Custom dataset for text data with proper error handling."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        truncation: bool = True,
        padding: str = "max_length"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        
        # Validate inputs
        if not texts:
            raise ValueError("Texts list cannot be empty")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        
        logger.info(f"Initialized TextDataset with {len(texts)} samples")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            text = self.texts[idx]
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            return {k: v.squeeze(0) for k, v in encoding.items()}
            
        except Exception as e:
            logger.error(f"Error processing text at index {idx}: {e}")
            # Return a safe fallback
            return self._get_fallback_item()
    
    def _get_fallback_item(self) -> Dict[str, torch.Tensor]:
        """Return a safe fallback item in case of error."""
        fallback_text = "Error in text processing"
        encoding = self.tokenizer(
            fallback_text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


class ImageDataset(Dataset):
    """Custom dataset for image data with proper error handling."""
    
    def __init__(
        self,
        image_paths: List[str],
        transform=None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.target_size = target_size
        
        # Validate inputs
        if not image_paths:
            raise ValueError("Image paths list cannot be empty")
        if len(target_size) != 2 or any(s <= 0 for s in target_size):
            raise ValueError("target_size must be a tuple of positive integers")
        
        logger.info(f"Initialized ImageDataset with {len(image_paths)} samples")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            image_path = self.image_paths[idx]
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Apply transforms if available
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {e}")
            # Return a safe fallback
            return self._get_fallback_item()
    
    def _get_fallback_item(self) -> torch.Tensor:
        """Return a safe fallback item in case of error."""
        # Create a blank image
        fallback_image = Image.new('RGB', self.target_size, color='black')
        if self.transform:
            fallback_image = self.transform(fallback_image)
        return fallback_image


class DataManager:
    """Refactored data manager with best practices."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logger
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # Setup data directories
        self._setup_directories()
        
        logger.info("DataManager initialized successfully")
    
    def _setup_directories(self) -> None:
        """Create necessary directories for data handling."""
        try:
            # Create data directories
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Create cache directory
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            self.logger.info("Data directories created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise
    
    def load_tokenizer(self, model_name: str = "gpt2") -> PreTrainedTokenizer:
        """Load and configure tokenizer with error handling."""
        try:
            self.logger.info(f"Loading tokenizer: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Validate tokenizer
            if not hasattr(self.tokenizer, 'encode'):
                raise ValueError("Invalid tokenizer: missing encode method")
            
            self.logger.info(f"Tokenizer loaded successfully: {model_name}")
            return self.tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def load_text_data(self, file_path: str) -> List[str]:
        """Load text data from file with proper error handling."""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                return []
            
            # Determine file type and load accordingly
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    texts = [str(item) if isinstance(item, str) else str(item.get('text', '')) for item in data]
                elif isinstance(data, dict):
                    texts = [str(data.get('text', ''))]
                else:
                    texts = [str(data)]
                    
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                    
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
                # Assume first column contains text
                texts = df.iloc[:, 0].astype(str).tolist()
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Filter out empty texts
            texts = [text for text in texts if text.strip()]
            
            self.logger.info(f"Loaded {len(texts)} text samples from {file_path}")
            return texts
            
        except Exception as e:
            self.logger.error(f"Error loading text data from {file_path}: {e}")
            return []
    
    def create_text_datasets(self, model_name: str = "gpt2") -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/validation/test datasets for text data."""
        try:
            # Load tokenizer if not already loaded
            if self.tokenizer is None:
                self.load_tokenizer(model_name)
            
            # Load data
            train_texts = self.load_text_data(self.config.train_file)
            val_texts = self.load_text_data(self.config.validation_file)
            test_texts = self.load_text_data(self.config.test_file)
            
            # If validation/test files are empty, split from training data
            if not val_texts and train_texts:
                split_idx = int(len(train_texts) * (1 - self.config.validation_split - self.config.test_split))
                val_split_idx = int(len(train_texts) * (1 - self.config.test_split))
                
                train_texts, val_texts, test_texts = (
                    train_texts[:split_idx],
                    train_texts[split_idx:val_split_idx],
                    train_texts[val_split_idx:]
                )
            
            # Create datasets
            self.train_dataset = TextDataset(
                train_texts, 
                self.tokenizer,
                self.config.max_length,
                self.config.truncation,
                self.config.padding
            )
            
            self.val_dataset = TextDataset(
                val_texts, 
                self.tokenizer,
                self.config.max_length,
                self.config.truncation,
                self.config.padding
            )
            
            self.test_dataset = TextDataset(
                test_texts, 
                self.tokenizer,
                self.config.max_length,
                self.config.truncation,
                self.config.padding
            )
            
            self.logger.info(f"Created datasets - Train: {len(self.train_dataset)}, "
                           f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
            
            return self.train_dataset, self.val_dataset, self.test_dataset
            
        except Exception as e:
            self.logger.error(f"Error creating text datasets: {e}")
            raise
    
    def create_dataloaders(
        self,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for all datasets."""
        try:
            if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
                raise ValueError("Datasets not created. Call create_text_datasets() first.")
            
            # Use config values if not specified
            batch_size = batch_size or self.config.batch_size
            shuffle = shuffle if shuffle is not None else self.config.shuffle
            
            # Create data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt"
            )
            
            # Create DataLoaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last,
                persistent_workers=self.config.persistent_workers,
                collate_fn=data_collator
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                persistent_workers=self.config.persistent_workers,
                collate_fn=data_collator
            )
            
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                persistent_workers=self.config.persistent_workers,
                collate_fn=data_collator
            )
            
            self.logger.info("DataLoaders created successfully")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            self.logger.error(f"Error creating DataLoaders: {e}")
            raise
    
    def get_batch_sample(self, dataloader: DataLoader, num_samples: int = 1) -> List[Dict[str, torch.Tensor]]:
        """Get sample batches from a DataLoader for inspection."""
        try:
            samples = []
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                samples.append(batch)
            
            self.logger.info(f"Retrieved {len(samples)} batch samples")
            return samples
            
        except Exception as e:
            self.logger.error(f"Error getting batch samples: {e}")
            return []
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and provide statistics."""
        try:
            if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
                return {"error": "Datasets not created"}
            
            stats = {
                "train_samples": len(self.train_dataset),
                "val_samples": len(self.val_dataset),
                "test_samples": len(self.test_dataset),
                "total_samples": len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset),
                "train_ratio": len(self.train_dataset) / (len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)),
                "val_ratio": len(self.val_dataset) / (len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)),
                "test_ratio": len(self.test_dataset) / (len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset))
            }
            
            # Validate ratios
            total_ratio = stats["train_ratio"] + stats["val_ratio"] + stats["test_ratio"]
            if abs(total_ratio - 1.0) > 1e-6:
                self.logger.warning(f"Dataset ratios don't sum to 1.0: {total_ratio}")
            
            self.logger.info("Data quality validation completed")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return {"error": str(e)}
    
    def save_data_info(self, file_path: str = "data/data_info.json") -> None:
        """Save data information and statistics to file."""
        try:
            data_info = {
                "config": self.config.__dict__,
                "quality_stats": self.validate_data_quality(),
                "tokenizer_info": {
                    "model_name": self.tokenizer.name_or_path if self.tokenizer else None,
                    "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
                    "max_length": self.config.max_length
                } if self.tokenizer else None
            }
            
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_info, f, indent=2, default=str)
            
            self.logger.info(f"Data info saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data info: {e}")
    
    def cleanup(self) -> None:
        """Cleanup resources and memory."""
        try:
            # Clear datasets
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            
            # Clear tokenizer
            self.tokenizer = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("DataManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Factory function for easy usage
def create_data_manager(config: Optional[DataConfig] = None) -> DataManager:
    """Create a data manager instance."""
    if config is None:
        config = DataConfig()
    
    return DataManager(config)


# Example usage
if __name__ == "__main__":
    # Create data manager
    config = DataConfig(
        train_file="data/sample_train.json",
        validation_file="data/sample_val.json",
        test_file="data/sample_test.json",
        batch_size=4,
        max_length=256
    )
    
    data_manager = create_data_manager(config)
    
    try:
        # Create datasets
        train_ds, val_ds, test_ds = data_manager.create_text_datasets("gpt2")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = data_manager.create_dataloaders()
        
        # Get sample batch
        sample_batch = data_manager.get_batch_sample(train_loader, 1)
        
        # Validate data quality
        quality_stats = data_manager.validate_data_quality()
        print(f"Data quality stats: {quality_stats}")
        
        # Save data info
        data_manager.save_data_info()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        data_manager.cleanup()

