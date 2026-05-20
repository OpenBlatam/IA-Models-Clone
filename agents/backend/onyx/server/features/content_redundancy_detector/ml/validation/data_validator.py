"""
Data Validator
Advanced data validation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Advanced data validation
    """
    
    @staticmethod
    def validate_dataset_distribution(
        dataset: Dataset,
        num_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Validate dataset class distribution
        
        Args:
            dataset: Dataset to validate
            num_samples: Number of samples to check
            
        Returns:
            Dictionary with distribution info
        """
        import numpy as np
        from collections import Counter
        
        labels = []
        for i in range(min(num_samples, len(dataset))):
            try:
                _, label = dataset[i]
                labels.append(int(label))
            except Exception as e:
                logger.warning(f"Error getting label from sample {i}: {e}")
        
        if not labels:
            return {'error': 'No valid labels found'}
        
        counter = Counter(labels)
        total = len(labels)
        
        distribution = {
            label: count / total for label, count in counter.items()
        }
        
        return {
            'distribution': distribution,
            'class_counts': dict(counter),
            'num_classes': len(counter),
            'imbalance_ratio': max(counter.values()) / min(counter.values()) if counter else 1.0,
        }
    
    @staticmethod
    def validate_data_quality(
        dataloader: DataLoader,
        num_batches: int = 10,
    ) -> Dict[str, Any]:
        """
        Validate data quality
        
        Args:
            dataloader: DataLoader to validate
            num_batches: Number of batches to check
            
        Returns:
            Dictionary with quality metrics
        """
        issues = []
        stats = {
            'nan_count': 0,
            'inf_count': 0,
            'zero_count': 0,
            'mean': 0.0,
            'std': 0.0,
        }
        
        all_values = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            inputs, _ = batch
            
            # Check for issues
            if torch.isnan(inputs).any():
                stats['nan_count'] += 1
                issues.append(f"NaN found in batch {i}")
            
            if torch.isinf(inputs).any():
                stats['inf_count'] += 1
                issues.append(f"Inf found in batch {i}")
            
            if (inputs == 0).all():
                stats['zero_count'] += 1
                issues.append(f"All zeros in batch {i}")
            
            all_values.append(inputs.flatten())
        
        if all_values:
            all_tensor = torch.cat(all_values)
            stats['mean'] = float(all_tensor.mean().item())
            stats['std'] = float(all_tensor.std().item())
        
        return {
            'stats': stats,
            'issues': issues,
            'quality_score': 1.0 - (len(issues) / num_batches) if num_batches > 0 else 0.0,
        }



