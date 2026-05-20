"""
Builder Pattern - Fluent Interface for Model Building
=====================================================

Provides builder pattern for constructing complex models and training configurations.
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from ..core.base import BaseComponent
from ..models import create_model, BaseModel

logger = logging.getLogger(__name__)


class ModelBuilder(BaseComponent):
    """
    Builder for constructing models with fluent interface.
    
    Example:
        model = (ModelBuilder()
                .with_type('transformer')
                .with_vocab_size(10000)
                .with_d_model(512)
                .with_num_heads(8)
                .build())
    """
    
    def _initialize(self) -> None:
        """Initialize builder."""
        self.model_type: Optional[str] = None
        self.model_config: Dict[str, Any] = {}
    
    def with_type(self, model_type: str) -> 'ModelBuilder':
        """Set model type."""
        self.model_type = model_type
        return self
    
    def with_vocab_size(self, vocab_size: int) -> 'ModelBuilder':
        """Set vocabulary size."""
        self.model_config['vocab_size'] = vocab_size
        return self
    
    def with_d_model(self, d_model: int) -> 'ModelBuilder':
        """Set model dimension."""
        self.model_config['d_model'] = d_model
        return self
    
    def with_num_heads(self, num_heads: int) -> 'ModelBuilder':
        """Set number of attention heads."""
        self.model_config['num_heads'] = num_heads
        return self
    
    def with_num_layers(self, num_layers: int) -> 'ModelBuilder':
        """Set number of layers."""
        self.model_config['num_layers'] = num_layers
        return self
    
    def with_dropout(self, dropout: float) -> 'ModelBuilder':
        """Set dropout rate."""
        self.model_config['dropout'] = dropout
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'ModelBuilder':
        """Set full configuration."""
        self.model_config.update(config)
        return self
    
    def build(self) -> BaseModel:
        """
        Build model.
        
        Returns:
            Constructed model
            
        Raises:
            ValueError: If model type not set
        """
        if self.model_type is None:
            raise ValueError("Model type must be set before building")
        
        model = create_model(self.model_type, self.model_config)
        logger.info(f"Model built: {self.model_type} with {model.get_num_parameters():,} parameters")
        return model


class TrainingBuilder(BaseComponent):
    """
    Builder for constructing training configurations.
    
    Example:
        config = (TrainingBuilder()
                 .with_epochs(10)
                 .with_batch_size(32)
                 .with_learning_rate(1e-4)
                 .with_mixed_precision()
                 .build())
    """
    
    def _initialize(self) -> None:
        """Initialize builder."""
        self.config: Dict[str, Any] = {}
    
    def with_epochs(self, num_epochs: int) -> 'TrainingBuilder':
        """Set number of epochs."""
        self.config['num_epochs'] = num_epochs
        return self
    
    def with_batch_size(self, batch_size: int) -> 'TrainingBuilder':
        """Set batch size."""
        self.config['batch_size'] = batch_size
        return self
    
    def with_learning_rate(self, lr: float) -> 'TrainingBuilder':
        """Set learning rate."""
        self.config['learning_rate'] = lr
        return self
    
    def with_optimizer(self, optimizer_type: str) -> 'TrainingBuilder':
        """Set optimizer type."""
        self.config['optimizer'] = optimizer_type
        return self
    
    def with_scheduler(self, scheduler_type: str) -> 'TrainingBuilder':
        """Set scheduler type."""
        self.config['scheduler'] = scheduler_type
        return self
    
    def with_mixed_precision(self) -> 'TrainingBuilder':
        """Enable mixed precision."""
        self.config['use_mixed_precision'] = True
        return self
    
    def with_gradient_accumulation(self, steps: int) -> 'TrainingBuilder':
        """Set gradient accumulation steps."""
        self.config['gradient_accumulation_steps'] = steps
        return self
    
    def with_early_stopping(self, patience: int) -> 'TrainingBuilder':
        """Enable early stopping."""
        self.config['use_early_stopping'] = True
        self.config['early_stopping_patience'] = patience
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build training configuration."""
        return self.config.copy()



