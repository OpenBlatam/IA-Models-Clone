"""
Data Validation

Validates datasets and data loaders.
"""

import logging
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates datasets and data loaders."""
    
    @staticmethod
    def validate_dataset(
        dataset: Dataset,
        expected_keys: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate dataset.
        
        Args:
            dataset: Dataset to validate
            expected_keys: Expected keys in dataset items
            
        Returns:
            (is_valid, error_message)
        """
        if len(dataset) == 0:
            return False, "Dataset is empty"
        
        # Check first sample
        try:
            sample = dataset[0]
        except Exception as e:
            return False, f"Error accessing dataset sample: {e}"
        
        if expected_keys:
            if isinstance(sample, dict):
                for key in expected_keys:
                    if key not in sample:
                        return False, f"Missing key in dataset: {key}"
            else:
                return False, "Dataset samples must be dictionaries"
        
        return True, None
    
    @staticmethod
    def validate_dataloader(
        dataloader: DataLoader,
        num_samples: int = 5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate data loader.
        
        Args:
            dataloader: Data loader to validate
            num_samples: Number of samples to check
            
        Returns:
            (is_valid, error_message)
        """
        try:
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                # Check batch structure
                if not isinstance(batch, (dict, tuple, torch.Tensor)):
                    return False, f"Invalid batch type: {type(batch)}"
                
                # Check for NaN/Inf
                if isinstance(batch, dict):
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                                return False, f"NaN/Inf found in batch[{key}]"
                elif isinstance(batch, torch.Tensor):
                    if torch.any(torch.isnan(batch)) or torch.any(torch.isinf(batch)):
                        return False, "NaN/Inf found in batch"
        
        except Exception as e:
            return False, f"Error iterating data loader: {e}"
        
        return True, None


def validate_dataset(dataset: Dataset, **kwargs) -> Tuple[bool, Optional[str]]:
    """Convenience function for dataset validation."""
    return DataValidator.validate_dataset(dataset, **kwargs)



