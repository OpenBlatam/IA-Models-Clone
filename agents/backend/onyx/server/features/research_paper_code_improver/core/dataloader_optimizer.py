"""
Data Loader Optimizer - Optimizador de DataLoaders
===================================================
"""

import logging
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuración optimizada de DataLoader"""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    shuffle: bool = True


class DataLoaderOptimizer:
    """Optimizador de DataLoaders"""
    
    def __init__(self):
        self.performance_stats: Dict[str, Dict[str, float]] = {}
    
    def create_optimized_loader(
        self,
        dataset: Dataset,
        config: DataLoaderConfig
    ) -> DataLoader:
        """Crea un DataLoader optimizado"""
        # Ajustar num_workers según disponibilidad
        if not torch.cuda.is_available():
            config.num_workers = min(config.num_workers, 2)
            config.pin_memory = False
        
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and torch.cuda.is_available(),
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else 2,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            drop_last=config.drop_last
        )
        
        logger.info(f"DataLoader optimizado creado: batch_size={config.batch_size}, workers={config.num_workers}")
        return loader
    
    def profile_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """Profila un DataLoader"""
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            # Simular procesamiento mínimo
            if isinstance(batch, dict):
                _ = {k: v for k, v in batch.items()}
            else:
                _ = batch
            batch_times.append(time.time() - batch_start)
        
        total_time = time.time() - start_time
        
        stats = {
            "total_time": total_time,
            "avg_batch_time": sum(batch_times) / len(batch_times) if batch_times else 0,
            "batches_per_second": num_batches / total_time if total_time > 0 else 0,
            "throughput_samples_per_sec": (num_batches * dataloader.batch_size) / total_time if total_time > 0 else 0
        }
        
        return stats
    
    def find_optimal_batch_size(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        device: str = "cuda",
        start_batch_size: int = 1,
        max_batch_size: int = 128
    ) -> int:
        """Encuentra el batch size óptimo"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        optimal_batch_size = start_batch_size
        
        for batch_size in [start_batch_size * (2 ** i) for i in range(10)]:
            if batch_size > max_batch_size:
                break
            
            try:
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                batch = next(iter(loader))
                
                # Mover a device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
                else:
                    batch = batch.to(device)
                
                # Forward pass
                with torch.no_grad():
                    if isinstance(batch, dict):
                        _ = model(**batch)
                    else:
                        _ = model(batch)
                
                optimal_batch_size = batch_size
                logger.info(f"Batch size {batch_size} funciona")
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM con batch_size {batch_size}")
                    break
                else:
                    raise
        
        return optimal_batch_size
    
    def optimize_num_workers(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        test_workers: List[int] = None
    ) -> int:
        """Encuentra el número óptimo de workers"""
        if test_workers is None:
            import os
            cpu_count = os.cpu_count() or 4
            test_workers = [0, 2, 4, 6, 8, cpu_count]
        
        best_workers = 0
        best_throughput = 0
        
        for num_workers in test_workers:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False
            )
            
            stats = self.profile_dataloader(loader, num_batches=20)
            throughput = stats["throughput_samples_per_sec"]
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_workers = num_workers
            
            logger.info(f"Workers {num_workers}: {throughput:.2f} samples/sec")
        
        logger.info(f"Óptimo: {best_workers} workers con {best_throughput:.2f} samples/sec")
        return best_workers




