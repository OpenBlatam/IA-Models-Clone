"""
Training factory for the ads training system.

This module provides a unified factory interface for creating and managing
different types of trainers, consolidating all scattered training implementations.
"""

from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass
import logging
from enum import Enum

from .base_trainer import BaseTrainer, TrainingConfig
from .pytorch_trainer import PyTorchTrainer, PyTorchModelConfig, PyTorchDataConfig
from .diffusion_trainer import DiffusionTrainer, DiffusionModelConfig, DiffusionTrainingConfig
from .multi_gpu_trainer import MultiGPUTrainer, GPUConfig, MultiGPUTrainingConfig

logger = logging.getLogger(__name__)

class TrainerType(Enum):
    """Types of available trainers."""
    PYTORCH = "pytorch"
    DIFFUSION = "diffusion"
    MULTI_GPU = "multi_gpu"
    HYBRID = "hybrid"  # Combination of multiple approaches

@dataclass
class TrainerConfig:
    """Configuration for creating trainers."""
    trainer_type: TrainerType
    base_config: TrainingConfig
    specific_configs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.specific_configs is None:
            self.specific_configs = {}

class TrainingFactory:
    """
    Factory for creating and managing different types of trainers.
    
    This factory consolidates all training implementations and provides
    a unified interface for creating trainers based on configuration.
    """
    
    def __init__(self):
        """Initialize the training factory."""
        self._registered_trainers: Dict[str, Type[BaseTrainer]] = {}
        self._trainer_instances: Dict[str, BaseTrainer] = {}
        self._trainer_configs: Dict[str, TrainerConfig] = {}
        
        # Register default trainers
        self._register_default_trainers()
        
        logger.info("Training factory initialized with default trainers")
    
    def _register_default_trainers(self):
        """Register the default trainer implementations."""
        self.register_trainer("pytorch", PyTorchTrainer)
        self.register_trainer("diffusion", DiffusionTrainer)
        self.register_trainer("multi_gpu", MultiGPUTrainer)
        
        logger.info("Default trainers registered: pytorch, diffusion, multi_gpu")
    
    def register_trainer(self, name: str, trainer_class: Type[BaseTrainer]):
        """Register a new trainer type."""
        if not issubclass(trainer_class, BaseTrainer):
            raise ValueError(f"Trainer class must inherit from BaseTrainer: {trainer_class}")
        
        self._registered_trainers[name] = trainer_class
        logger.info(f"Trainer registered: {name} -> {trainer_class.__name__}")
    
    def create_trainer(self, config: TrainerConfig) -> BaseTrainer:
        """Create a trainer based on configuration."""
        trainer_type = config.trainer_type.value
        
        if trainer_type not in self._registered_trainers:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        
        trainer_class = self._registered_trainers[trainer_type]
        
        try:
            if trainer_type == "pytorch":
                trainer = self._create_pytorch_trainer(config)
            elif trainer_type == "diffusion":
                trainer = self._create_diffusion_trainer(config)
            elif trainer_type == "multi_gpu":
                trainer = self._create_multi_gpu_trainer(config)
            elif trainer_type == "hybrid":
                trainer = self._create_hybrid_trainer(config)
            else:
                raise ValueError(f"Unsupported trainer type: {trainer_type}")
            
            # Store instance and config
            instance_id = f"{trainer_type}_{id(trainer)}"
            self._trainer_instances[instance_id] = trainer
            self._trainer_configs[instance_id] = config
            
            logger.info(f"Trainer created successfully: {instance_id}")
            return trainer
            
        except Exception as e:
            logger.error(f"Failed to create trainer {trainer_type}: {e}")
            raise
    
    def _create_pytorch_trainer(self, config: TrainerConfig) -> PyTorchTrainer:
        """Create a PyTorch trainer."""
        # Extract specific configs
        model_config = config.specific_configs.get("model_config")
        data_config = config.specific_configs.get("data_config")
        
        # Create configs if not provided
        if model_config is None:
            model_config = PyTorchModelConfig()
        
        if data_config is None:
            data_config = PyTorchDataConfig()
        
        # Create trainer
        trainer = PyTorchTrainer(
            config=config.base_config,
            model_config=model_config,
            data_config=data_config
        )
        
        return trainer
    
    def _create_diffusion_trainer(self, config: TrainerConfig) -> DiffusionTrainer:
        """Create a diffusion trainer."""
        # Extract specific configs
        diffusion_config = config.specific_configs.get("diffusion_config")
        training_config = config.specific_configs.get("training_config")
        
        # Create configs if not provided
        if diffusion_config is None:
            diffusion_config = DiffusionModelConfig()
        
        if training_config is None:
            training_config = DiffusionTrainingConfig()
        
        # Create trainer
        trainer = DiffusionTrainer(
            config=config.base_config,
            diffusion_config=diffusion_config,
            training_config=training_config
        )
        
        return trainer
    
    def _create_multi_gpu_trainer(self, config: TrainerConfig) -> MultiGPUTrainer:
        """Create a multi-GPU trainer."""
        # Extract specific configs
        gpu_config = config.specific_configs.get("gpu_config")
        multi_gpu_config = config.specific_configs.get("multi_gpu_config")
        
        # Create configs if not provided
        if gpu_config is None:
            gpu_config = GPUConfig()
        
        if multi_gpu_config is None:
            multi_gpu_config = MultiGPUTrainingConfig()
        
        # Create trainer
        trainer = MultiGPUTrainer(
            config=config.base_config,
            gpu_config=gpu_config,
            multi_gpu_config=multi_gpu_config
        )
        
        return trainer
    
    def _create_hybrid_trainer(self, config: TrainerConfig) -> BaseTrainer:
        """Create a hybrid trainer combining multiple approaches."""
        # This is a placeholder for future hybrid implementations
        # For now, create a PyTorch trainer as the base
        logger.info("Creating hybrid trainer (using PyTorch as base)")
        return self._create_pytorch_trainer(config)
    
    def get_trainer(self, instance_id: str) -> Optional[BaseTrainer]:
        """Get a trainer instance by ID."""
        return self._trainer_instances.get(instance_id)
    
    def get_trainer_config(self, instance_id: str) -> Optional[TrainerConfig]:
        """Get a trainer configuration by ID."""
        return self._trainer_configs.get(instance_id)
    
    def list_trainers(self) -> List[str]:
        """List all registered trainer types."""
        return list(self._registered_trainers.keys())
    
    def list_instances(self) -> List[str]:
        """List all trainer instances."""
        return list(self._trainer_instances.keys())
    
    def get_trainer_info(self, instance_id: str) -> Dict[str, Any]:
        """Get information about a trainer instance."""
        trainer = self._trainer_instances.get(instance_id)
        config = self._trainer_configs.get(instance_id)
        
        if not trainer or not config:
            return {"error": "Trainer not found"}
        
        info = {
            "instance_id": instance_id,
            "trainer_type": config.trainer_type.value,
            "trainer_class": trainer.__class__.__name__,
            "status": trainer.get_status(),
            "config": config.base_config.to_dict()
        }
        
        # Add trainer-specific info
        if hasattr(trainer, 'get_model_info'):
            info["model_info"] = trainer.get_model_info()
        
        return info
    
    def cleanup_trainer(self, instance_id: str) -> bool:
        """Clean up a trainer instance."""
        if instance_id not in self._trainer_instances:
            return False
        
        trainer = self._trainer_instances[instance_id]
        
        # Call cleanup method if available
        if hasattr(trainer, 'cleanup'):
            try:
                trainer.cleanup()
            except Exception as e:
                logger.warning(f"Error during trainer cleanup: {e}")
        
        # Remove from tracking
        del self._trainer_instances[instance_id]
        del self._trainer_configs[instance_id]
        
        logger.info(f"Trainer cleaned up: {instance_id}")
        return True
    
    def cleanup_all(self):
        """Clean up all trainer instances."""
        instance_ids = list(self._trainer_instances.keys())
        
        for instance_id in instance_ids:
            self.cleanup_trainer(instance_id)
        
        logger.info("All trainers cleaned up")
    
    def get_optimal_trainer(self, requirements: Dict[str, Any]) -> TrainerType:
        """Determine the optimal trainer type based on requirements."""
        # Simple heuristic for trainer selection
        # In production, this could be more sophisticated
        
        if requirements.get("multi_gpu", False):
            return TrainerType.MULTI_GPU
        
        if requirements.get("diffusion_model", False):
            return TrainerType.DIFFUSION
        
        if requirements.get("hybrid", False):
            return TrainerType.HYBRID
        
        # Default to PyTorch
        return TrainerType.PYTORCH
    
    def create_optimal_trainer(self, base_config: TrainingConfig, 
                              requirements: Dict[str, Any]) -> BaseTrainer:
        """Create the optimal trainer based on requirements."""
        optimal_type = self.get_optimal_trainer(requirements)
        
        trainer_config = TrainerConfig(
            trainer_type=optimal_type,
            base_config=base_config,
            specific_configs=requirements
        )
        
        return self.create_trainer(trainer_config)

# Global factory instance
_global_factory: Optional[TrainingFactory] = None

def get_training_factory() -> TrainingFactory:
    """Get the global training factory instance."""
    global _global_factory
    
    if _global_factory is None:
        _global_factory = TrainingFactory()
    
    return _global_factory

def create_trainer(trainer_type: str, base_config: TrainingConfig, 
                   specific_configs: Optional[Dict[str, Any]] = None) -> BaseTrainer:
    """Convenience function to create a trainer."""
    factory = get_training_factory()
    
    config = TrainerConfig(
        trainer_type=TrainerType(trainer_type),
        base_config=base_config,
        specific_configs=specific_configs or {}
    )
    
    return factory.create_trainer(config)

def create_optimal_trainer(base_config: TrainingConfig, 
                          requirements: Dict[str, Any]) -> BaseTrainer:
    """Convenience function to create the optimal trainer."""
    factory = get_training_factory()
    return factory.create_optimal_trainer(base_config, requirements)
