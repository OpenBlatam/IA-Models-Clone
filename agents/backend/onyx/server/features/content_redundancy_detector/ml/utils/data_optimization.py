"""
Data Optimization Utilities
Optimize data loading and processing
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataOptimizer:
    """
    Optimize data loading
    """
    
    @staticmethod
    def optimize_dataloader(
        dataloader: DataLoader,
        pin_memory: bool = True,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """
        Optimize DataLoader settings
        
        Args:
            dataloader: DataLoader to optimize
            pin_memory: Pin memory for faster GPU transfer
            num_workers: Number of workers (None = auto)
            prefetch_factor: Prefetch factor
            
        Returns:
            Optimized DataLoader
        """
        if num_workers is None:
            num_workers = min(4, torch.get_num_threads())
        
        # Create new DataLoader with optimized settings
        optimized = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        
        logger.info(f"Optimized DataLoader: {num_workers} workers, pin_memory={pin_memory}")
        return optimized
    
    @staticmethod
    def get_dataloader_stats(dataloader: DataLoader) -> Dict[str, Any]:
        """
        Get DataLoader statistics
        
        Args:
            dataloader: DataLoader to analyze
            
        Returns:
            Dictionary with statistics
        """
        return {
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'dataset_size': len(dataloader.dataset),
            'num_batches': len(dataloader),
        }
    
    @staticmethod
    def profile_dataloader(
        dataloader: DataLoader,
        num_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Profile DataLoader performance
        
        Args:
            dataloader: DataLoader to profile
            num_batches: Number of batches to profile
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        times = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            start = time.time()
            _ = batch  # Load batch
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            'mean_time': sum(times) / len(times) if times else 0.0,
            'min_time': min(times) if times else 0.0,
            'max_time': max(times) if times else 0.0,
            'total_time': sum(times),
            'batches_per_second': len(times) / sum(times) if times else 0.0,
        }



