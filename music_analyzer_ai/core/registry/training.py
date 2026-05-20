"""
Training Component Registry Module

Training component registration (losses, optimizers, schedulers).
"""

from typing import Dict, Type, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingComponentRegistry:
    """Training component registration mixin."""
    
    def __init__(self):
        self._losses: Dict[str, Type] = {}
        self._optimizers: Dict[str, Callable] = {}
        self._schedulers: Dict[str, Callable] = {}
    
    # Loss registration
    def register_loss(self, name: str, loss_class: Type):
        """
        Register a loss class.
        
        Args:
            name: Loss name.
            loss_class: Loss class.
        """
        if name in self._losses:
            logger.warning(f"Loss {name} already registered, overwriting")
        self._losses[name] = loss_class
        logger.info(f"Registered loss: {name}")
    
    def get_loss(self, name: str) -> Optional[Type]:
        """
        Get loss class by name.
        
        Args:
            name: Loss name.
        
        Returns:
            Loss class or None.
        """
        return self._losses.get(name)
    
    # Optimizer registration
    def register_optimizer(self, name: str, optimizer_factory: Callable):
        """
        Register an optimizer factory.
        
        Args:
            name: Optimizer name.
            optimizer_factory: Optimizer factory function.
        """
        if name in self._optimizers:
            logger.warning(f"Optimizer {name} already registered, overwriting")
        self._optimizers[name] = optimizer_factory
        logger.info(f"Registered optimizer: {name}")
    
    def get_optimizer_factory(self, name: str) -> Optional[Callable]:
        """
        Get optimizer factory by name.
        
        Args:
            name: Optimizer name.
        
        Returns:
            Optimizer factory or None.
        """
        return self._optimizers.get(name)
    
    # Scheduler registration
    def register_scheduler(self, name: str, scheduler_factory: Callable):
        """
        Register a scheduler factory.
        
        Args:
            name: Scheduler name.
            scheduler_factory: Scheduler factory function.
        """
        if name in self._schedulers:
            logger.warning(f"Scheduler {name} already registered, overwriting")
        self._schedulers[name] = scheduler_factory
        logger.info(f"Registered scheduler: {name}")
    
    def get_scheduler_factory(self, name: str) -> Optional[Callable]:
        """
        Get scheduler factory by name.
        
        Args:
            name: Scheduler name.
        
        Returns:
            Scheduler factory or None.
        """
        return self._schedulers.get(name)



