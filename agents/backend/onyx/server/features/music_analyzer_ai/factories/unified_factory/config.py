"""
Config Factory Module

Factory creation from configuration.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

from ...config.model_config import ModelConfig


class ConfigFactoryMixin:
    """Config factory mixin."""
    
    def create_from_config(
        self,
        config: ModelConfig
    ) -> Dict[str, Any]:
        """
        Create complete training setup from configuration.
        
        Args:
            config: ModelConfig instance
        
        Returns:
            Dictionary with model, optimizer, scheduler, loss, training_loop, callbacks
        """
        # Create model
        model = self.create_model(
            model_type=config.architecture.model_type,
            config={
                "input_dim": config.architecture.input_dim,
                "embed_dim": config.architecture.embed_dim,
                "num_heads": config.architecture.num_heads,
                "num_layers": config.architecture.num_layers,
                "ff_dim": config.architecture.ff_dim,
                "dropout": config.architecture.dropout
            },
            device=config.device
        )
        
        # Create optimizer
        optimizer = self.create_optimizer(
            optimizer_type=config.training.optimizer,
            parameters=model.parameters(),
            config={
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay
            }
        )
        
        # Create scheduler
        scheduler = self.create_scheduler(
            scheduler_type=config.training.scheduler_type,
            optimizer=optimizer,
            config={
                "T_max": config.training.epochs,
                "warmup_steps": config.training.warmup_steps
            }
        )
        
        # Create loss (simplified - would need task-specific config)
        loss_fn = self.create_loss("classification", {})
        
        # Create training loop
        training_loop = self.create_training_loop(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config={
                "device": config.device,
                "use_mixed_precision": config.training.use_mixed_precision,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "max_grad_norm": config.training.max_grad_norm
            }
        )
        
        # Create callbacks
        callbacks = self.create_callbacks({
            "early_stopping": {
                "patience": config.training.early_stopping_patience
            },
            "checkpoint": {
                "checkpoint_dir": config.training.checkpoint_dir,
                "save_best": config.training.save_best_only
            },
            "metrics": {}
        })
        
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss": loss_fn,
            "training_loop": training_loop,
            "callbacks": callbacks
        }



