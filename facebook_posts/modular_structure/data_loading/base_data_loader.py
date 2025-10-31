"""
📊 Base Data Loader Class

Abstract base class for all data loading operations.
Provides common interface for data management and preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    This class provides a common interface for:
    - Data loading and preprocessing
    - Dataset creation and management
    - Data augmentation and validation
    - Batch generation and iteration
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "base_data_loader"):
        """
        Initialize the base data loader.
        
        Args:
            config: Data loader configuration dictionary
            name: Data loader name for identification
        """
        self.config = config
        self.name = name
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Data statistics
        self.data_info = {
            "train_size": 0,
            "val_size": 0,
            "test_size": 0,
            "input_shape": None,
            "output_shape": None,
            "num_classes": None,
            "data_types": []
        }
        
        # Initialize data loader
        self._setup_data_loader()
        self._log_data_info()
    
    @abstractmethod
    def _setup_data_loader(self) -> None:
        """
        Setup the data loader and create datasets.
        Must be implemented by subclasses.
        """
        pass
    
    def _log_data_info(self) -> None:
        """Log data information and statistics."""
        logger.info(f"Data Loader {self.name} initialized:")
        logger.info(f"  Train size: {self.data_info['train_size']:,}")
        logger.info(f"  Validation size: {self.data_info['val_size']:,}")
        logger.info(f"  Test size: {self.data_info['test_size']:,}")
        logger.info(f"  Input shape: {self.data_info['input_shape']}")
        logger.info(f"  Output shape: {self.data_info['output_shape']}")
        if self.data_info['num_classes']:
            logger.info(f"  Number of classes: {self.data_info['num_classes']}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive data information."""
        return self.data_info.copy()
    
    def get_dataset_sizes(self) -> Tuple[int, int, int]:
        """
        Get train, validation, and test dataset sizes.
        
        Returns:
            Tuple of (train_size, val_size, test_size)
        """
        return (
            self.data_info["train_size"],
            self.data_info["val_size"], 
            self.data_info["test_size"]
        )
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        return self.test_loader
    
    def get_sample_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch from the specified split.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            
        Returns:
            Tuple of (inputs, targets)
        """
        if split == "train" and self.train_loader:
            loader = self.train_loader
        elif split == "val" and self.val_loader:
            loader = self.val_loader
        elif split == "test" and self.test_loader:
            loader = self.test_loader
        else:
            raise ValueError(f"Invalid split: {split}")
        
        batch = next(iter(loader))
        return batch
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Calculate and return data statistics.
        
        Returns:
            Dictionary containing data statistics
        """
        stats = {}
        
        # Get sample batch for statistics
        try:
            sample_inputs, sample_targets = self.get_sample_batch("train")
            
            # Input statistics
            if isinstance(sample_inputs, torch.Tensor):
                stats["input_mean"] = sample_inputs.mean().item()
                stats["input_std"] = sample_inputs.std().item()
                stats["input_min"] = sample_inputs.min().item()
                stats["input_max"] = sample_inputs.max().item()
                stats["input_shape"] = list(sample_inputs.shape)
            
            # Target statistics
            if isinstance(sample_targets, torch.Tensor):
                if sample_targets.dtype in [torch.long, torch.int]:
                    # Classification task
                    unique_targets = torch.unique(sample_targets)
                    stats["num_classes"] = len(unique_targets)
                    stats["class_distribution"] = {
                        int(target.item()): int((sample_targets == target).sum().item())
                        for target in unique_targets
                    }
                else:
                    # Regression task
                    stats["target_mean"] = sample_targets.mean().item()
                    stats["target_std"] = sample_targets.std().item()
                    stats["target_min"] = sample_targets.min().item()
                    stats["target_max"] = sample_targets.max().item()
            
        except Exception as e:
            logger.warning(f"Could not calculate data statistics: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def save_data_info(self, path: Union[str, Path]) -> None:
        """
        Save data information to JSON file.
        
        Args:
            path: Path to save the data info
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine data info and statistics
        save_dict = {
            "data_info": self.data_info,
            "data_statistics": self.get_data_statistics(),
            "config": self.config,
            "name": self.name
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2, default=str)
        
        logger.info(f"Data info saved to: {path}")
    
    def load_data_info(self, path: Union[str, Path]) -> None:
        """
        Load data information from JSON file.
        
        Args:
            path: Path to load the data info from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data info file not found: {path}")
        
        with open(path, 'r') as f:
            data_dict = json.load(f)
        
        # Update data info
        if "data_info" in data_dict:
            self.data_info.update(data_dict["data_info"])
        if "config" in data_dict:
            self.config.update(data_dict["config"])
        
        logger.info(f"Data info loaded from: {path}")
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate data integrity and quality.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": 0,
            "total_checks": 0
        }
        
        # Check if data loaders exist
        validation_results["total_checks"] += 1
        if self.train_loader is not None:
            validation_results["checks_passed"] += 1
        else:
            validation_results["errors"].append("Training data loader is None")
            validation_results["is_valid"] = False
        
        validation_results["total_checks"] += 1
        if self.val_loader is not None:
            validation_results["checks_passed"] += 1
        else:
            validation_results["warnings"].append("Validation data loader is None")
        
        validation_results["total_checks"] += 1
        if self.test_loader is not None:
            validation_results["checks_passed"] += 1
        else:
            validation_results["warnings"].append("Test data loader is None")
        
        # Check data sizes
        validation_results["total_checks"] += 1
        if self.data_info["train_size"] > 0:
            validation_results["checks_passed"] += 1
        else:
            validation_results["errors"].append("Training dataset is empty")
            validation_results["is_valid"] = False
        
        # Check if we can iterate through data
        try:
            validation_results["total_checks"] += 1
            sample_batch = self.get_sample_batch("train")
            if sample_batch is not None:
                validation_results["checks_passed"] += 1
            else:
                validation_results["errors"].append("Cannot get sample batch from training data")
                validation_results["is_valid"] = False
        except Exception as e:
            validation_results["errors"].append(f"Error getting sample batch: {e}")
            validation_results["is_valid"] = False
        
        logger.info(f"Data validation completed: {validation_results['checks_passed']}/{validation_results['total_checks']} checks passed")
        
        return validation_results
    
    def get_data_summary(self) -> str:
        """
        Generate a summary of the data.
        
        Returns:
            String summary of the data
        """
        train_size, val_size, test_size = self.get_dataset_sizes()
        total_size = train_size + val_size + test_size
        
        summary = f"""
Data Summary: {self.name}
{'=' * 50}
Total Samples: {total_size:,}
  - Training: {train_size:,} ({train_size/total_size*100:.1f}%)
  - Validation: {val_size:,} ({val_size/total_size*100:.1f}%)
  - Test: {test_size:,} ({test_size/total_size*100:.1f}%)

Input Shape: {self.data_info['input_shape']}
Output Shape: {self.data_info['output_shape']}
Number of Classes: {self.data_info['num_classes'] or 'N/A'}
Data Types: {', '.join(self.data_info['data_types']) if self.data_info['data_types'] else 'N/A'}
        """.strip()
        
        return summary
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return sum(self.get_dataset_sizes())
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate through all data."""
        if self.train_loader:
            for batch in self.train_loader:
                yield batch
        if self.val_loader:
            for batch in self.val_loader:
                yield batch
        if self.test_loader:
            for batch in self.test_loader:
                yield batch






