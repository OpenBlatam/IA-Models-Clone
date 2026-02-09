"""
Async Training

Utilities for asynchronous training operations.
"""

import logging
import asyncio
import torch
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncTraining:
    """Asynchronous training handler."""
    
    def __init__(
        self,
        max_workers: int = 2
    ):
        """
        Initialize async training.
        
        Args:
            max_workers: Maximum worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def train_step(
        self,
        model: torch.nn.Module,
        batch: dict,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """
        Async training step.
        
        Args:
            model: Model to train
            batch: Training batch
            loss_fn: Loss function
            optimizer: Optimizer
            
        Returns:
            Training step results
        """
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self.executor,
            self._train_step_sync,
            model,
            batch,
            loss_fn,
            optimizer
        )
        
        return result
    
    def _train_step_sync(
        self,
        model: torch.nn.Module,
        batch: dict,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer
    ) -> dict:
        """Synchronous training step."""
        optimizer.zero_grad()
        
        inputs = batch.get('input', batch[0] if isinstance(batch, tuple) else None)
        targets = batch.get('target', batch[1] if isinstance(batch, tuple) and len(batch) > 1 else None)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'outputs': outputs.detach()
        }
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


async def async_train_step(
    model: torch.nn.Module,
    batch: dict,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer
) -> dict:
    """Async training step convenience function."""
    trainer = AsyncTraining()
    return await trainer.train_step(model, batch, loss_fn, optimizer)



