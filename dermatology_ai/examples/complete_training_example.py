"""
Complete Training Example
Demonstrates all improvements and best practices
"""

import logging
from pathlib import Path

from ml.training.pipeline import TrainingPipeline
from ml.models.factories import SkinAnalysisModelFactory
from ml.common import (
    set_seed,
    setup_logging,
    TrainingLogger,
    log_model_info,
    log_training_summary,
    ConfigManager,
    CheckpointManager,
    plot_training_history
)

logger = logging.getLogger(__name__)


def main():
    """Complete training example with all improvements"""
    
    # 1. Setup logging
    setup_logging(
        log_level="INFO",
        log_file=Path("logs/training.log")
    )
    
    # 2. Set seed for reproducibility
    set_seed(42)
    
    # 3. Load configuration with ConfigManager
    config_manager = ConfigManager("config/model_config.yaml")
    config = config_manager.config
    
    # Validate configuration
    if not config_manager.validate():
        logger.error("Invalid configuration")
        return
    
    # 4. Create model from factory
    model = SkinAnalysisModelFactory.create(
        "vit_skin",
        config=config.get('model', {})
    )
    
    # 5. Log model information
    log_model_info(model, logger)
    
    # 6. Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path("checkpoints"),
        max_checkpoints=5,
        keep_best=True
    )
    
    # 7. Training with context manager
    with TrainingLogger(logger, config.get('experiment_name', 'experiment_001')):
        # Create pipeline
        pipeline = TrainingPipeline.from_config(
            model=model,
            config=config,
            train_images=train_images,
            val_images=val_images,
            train_labels=train_labels,
            val_labels=val_labels
        )
        
        # Add checkpoint callback
        from ml.training.callbacks import ModelCheckpointCallback
        checkpoint_callback = ModelCheckpointCallback(
            checkpoint_dir="checkpoints",
            save_best=True,
            monitor="val_loss"
        )
        pipeline.trainer.add_callback(checkpoint_callback)
        
        # Train
        results = pipeline.train()
    
    # 8. Log training summary
    log_training_summary(
        results['training_history'],
        logger,
        config.get('training', {}).get('num_epochs', 100)
    )
    
    # 9. Plot training history
    plot_training_history(
        results['training_history'],
        save_path=Path("plots/training_history.png"),
        show=False
    )
    
    # 10. Load best checkpoint
    best_checkpoint = checkpoint_manager.load(
        load_best=True,
        model=model
    )
    
    logger.info(f"Best checkpoint loaded from epoch {best_checkpoint['epoch']}")
    logger.info(f"Best metrics: {best_checkpoint['metrics']}")
    
    return results


if __name__ == "__main__":
    main()








