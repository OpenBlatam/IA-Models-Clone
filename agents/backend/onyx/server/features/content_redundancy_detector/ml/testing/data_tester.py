"""
Data Testing Utilities
Testing utilities for datasets and data loaders
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DataTester:
    """
    Testing utilities for data
    """
    
    @staticmethod
    def test_dataset(
        dataset: Dataset,
        num_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Test dataset
        
        Args:
            dataset: Dataset to test
            num_samples: Number of samples to test
            
        Returns:
            Test results
        """
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
            'sample_shapes': [],
        }
        
        try:
            # Test length
            dataset_len = len(dataset)
            if dataset_len == 0:
                results['errors'].append("Dataset is empty")
                return results
            
            # Test samples
            for i in range(min(num_samples, dataset_len)):
                try:
                    sample = dataset[i]
                    if isinstance(sample, tuple):
                        inputs, targets = sample
                        results['sample_shapes'].append({
                            'input': list(inputs.shape) if hasattr(inputs, 'shape') else str(type(inputs)),
                            'target': list(targets.shape) if hasattr(targets, 'shape') else str(type(targets)),
                        })
                    else:
                        results['sample_shapes'].append({
                            'sample': list(sample.shape) if hasattr(sample, 'shape') else str(type(sample)),
                        })
                except Exception as e:
                    results['errors'].append(f"Error loading sample {i}: {str(e)}")
            
            results['success'] = len(results['errors']) == 0
            results['dataset_length'] = dataset_len
        
        except Exception as e:
            results['errors'].append(f"Dataset test failed: {str(e)}")
            logger.error(f"Dataset test failed: {e}", exc_info=True)
        
        return results
    
    @staticmethod
    def test_dataloader(
        dataloader: DataLoader,
        num_batches: int = 5,
    ) -> Dict[str, Any]:
        """
        Test data loader
        
        Args:
            dataloader: DataLoader to test
            num_batches: Number of batches to test
            
        Returns:
            Test results
        """
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
            'batch_shapes': [],
            'batch_times': [],
        }
        
        try:
            import time
            
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                start_time = time.time()
                
                if isinstance(batch, tuple):
                    inputs, targets = batch
                    results['batch_shapes'].append({
                        'input': list(inputs.shape),
                        'target': list(targets.shape),
                    })
                    
                    # Check for issues
                    if torch.isnan(inputs).any():
                        results['warnings'].append(f"NaN in batch {i} inputs")
                    
                    if torch.isinf(inputs).any():
                        results['warnings'].append(f"Inf in batch {i} inputs")
                
                elapsed = time.time() - start_time
                results['batch_times'].append(elapsed)
            
            results['success'] = len(results['errors']) == 0
            
            if results['batch_times']:
                results['avg_batch_time'] = sum(results['batch_times']) / len(results['batch_times'])
        
        except Exception as e:
            results['errors'].append(f"DataLoader test failed: {str(e)}")
            logger.error(f"DataLoader test failed: {e}", exc_info=True)
        
        return results
    
    @staticmethod
    def validate_data_consistency(
        dataset: Dataset,
        num_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Validate data consistency
        
        Args:
            dataset: Dataset to validate
            num_samples: Number of samples to check
            
        Returns:
            Validation results
        """
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
            'statistics': {},
        }
        
        try:
            import numpy as np
            
            all_inputs = []
            all_targets = []
            
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                if isinstance(sample, tuple):
                    inputs, targets = sample
                    if isinstance(inputs, torch.Tensor):
                        all_inputs.append(inputs.numpy())
                    if isinstance(targets, torch.Tensor):
                        all_targets.append(targets.numpy())
            
            if all_inputs:
                inputs_array = np.array(all_inputs)
                results['statistics']['inputs'] = {
                    'mean': float(np.mean(inputs_array)),
                    'std': float(np.std(inputs_array)),
                    'min': float(np.min(inputs_array)),
                    'max': float(np.max(inputs_array)),
                }
            
            if all_targets:
                targets_array = np.array(all_targets)
                results['statistics']['targets'] = {
                    'mean': float(np.mean(targets_array)),
                    'std': float(np.std(targets_array)),
                    'min': float(np.min(targets_array)),
                    'max': float(np.max(targets_array)),
                }
            
            results['success'] = True
        
        except Exception as e:
            results['errors'].append(f"Consistency validation failed: {str(e)}")
            logger.error(f"Consistency validation failed: {e}", exc_info=True)
        
        return results



