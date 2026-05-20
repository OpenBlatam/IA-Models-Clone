"""
Trainer Factory Module
======================

Factory pattern for creating trainers with different configurations.
Provides a centralized way to create and configure trainers.

Author: BUL System
Date: 2024
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..trainer import CustomLLMTrainer
from .base_trainer import BaseLLMTrainer

logger = logging.getLogger(__name__)


class TrainerFactory:
    """
    Factory for creating CustomLLMTrainer instances.
    
    Provides convenient methods to create trainers with common configurations
    and presets.
    
    Example:
        >>> factory = TrainerFactory()
        >>> trainer = factory.create_basic_trainer(
        ...     model_name="gpt2",
        ...     dataset_path="data.json"
        ... )
    """
    
    @staticmethod
    def create_basic_trainer(
        model_name: str,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "./checkpoints",
        **kwargs
    ) -> CustomLLMTrainer:
        """
        Create a basic trainer with default settings.
        
        Args:
            model_name: Name of the model
            dataset_path: Path to dataset
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            CustomLLMTrainer instance
        """
        return CustomLLMTrainer(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            learning_rate=3e-5,
            num_train_epochs=3,
            batch_size=8,
            **kwargs
        )
    
    @staticmethod
    def create_advanced_trainer(
        model_name: str,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "./checkpoints",
        learning_rate: float = 3e-5,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        enable_early_stopping: bool = True,
        enable_evaluation: bool = True,
        **kwargs
    ) -> CustomLLMTrainer:
        """
        Create an advanced trainer with evaluation and early stopping.
        
        Args:
            model_name: Name of the model
            dataset_path: Path to dataset
            output_dir: Output directory
            learning_rate: Learning rate
            num_train_epochs: Number of epochs
            batch_size: Batch size
            enable_early_stopping: Enable early stopping
            enable_evaluation: Enable evaluation
            **kwargs: Additional arguments
            
        Returns:
            CustomLLMTrainer instance
        """
        eval_config = {
            "evaluation_strategy": "steps" if enable_evaluation else "no",
            "eval_steps": 100 if enable_evaluation else None,
            "load_best_model_at_end": enable_evaluation,
        }
        
        if enable_early_stopping and enable_evaluation:
            eval_config["early_stopping_patience"] = 3
        
        return CustomLLMTrainer(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            **eval_config,
            **kwargs
        )
    
    @staticmethod
    def create_memory_efficient_trainer(
        model_name: str,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path] = "./checkpoints",
        **kwargs
    ) -> CustomLLMTrainer:
        """
        Create a memory-efficient trainer with gradient checkpointing.
        
        Args:
            model_name: Name of the model
            dataset_path: Path to dataset
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            CustomLLMTrainer instance
        """
        return CustomLLMTrainer(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            learning_rate=3e-5,
            num_train_epochs=3,
            batch_size=4,  # Smaller batch size
            gradient_accumulation_steps=2,  # Compensate with accumulation
            gradient_checkpointing=True,
            fp16=True,
            **kwargs
        )
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> CustomLLMTrainer:
        """
        Create trainer from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            CustomLLMTrainer instance
        """
        required_keys = ["model_name", "dataset_path"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        return CustomLLMTrainer(**config)

