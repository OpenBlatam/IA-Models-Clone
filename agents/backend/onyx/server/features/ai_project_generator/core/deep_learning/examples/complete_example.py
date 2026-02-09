"""
Complete Example - Using the Modular Deep Learning Framework
===========================================================

This example demonstrates the complete workflow:
1. Model creation
2. Data loading
3. Training with best practices
4. Evaluation
5. Inference with Gradio
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn

# Import modular components
from ..models import TransformerModel, create_model
from ..data import TextDataset, create_dataloader, train_val_test_split
from ..training import Trainer, TrainingConfig, EarlyStopping, create_optimizer, create_scheduler
from ..evaluation import evaluate_model, compute_classification_metrics
from ..inference import InferenceEngine, create_text_classification_app
from ..config import ConfigManager
from ..utils import get_device, set_seed, ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Complete training and inference example."""
    
    # 1. Set random seed for reproducibility
    set_seed(42)
    
    # 2. Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # 3. Load configuration
    config_manager = ConfigManager()
    config = {
        'model': {
            'vocab_size': 10000,
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1
        },
        'training': {
            'num_epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 1,
            'use_mixed_precision': True
        }
    }
    config_manager.config = config
    
    # 4. Create model
    model = TransformerModel(**config['model'])
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    
    # 5. Create dataset
    # In practice, load your actual data here
    texts = ["Sample text 1", "Sample text 2"] * 100
    labels = [0, 1] * 100
    
    dataset = TextDataset(
        texts=texts,
        labels=labels,
        max_length=config['model']['max_seq_len']
    )
    
    # 6. Split dataset
    train_dataset, val_dataset, test_dataset = train_val_test_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 7. Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 8. Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type='cosine',
        num_epochs=config['training']['num_epochs']
    )
    
    # 9. Setup experiment tracking
    tracker = ExperimentTracker(
        experiment_name="transformer_example",
        log_dir=Path("logs/transformer_example"),
        use_tensorboard=True,
        use_wandb=False
    )
    
    # 10. Create training config
    training_config = TrainingConfig(
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        use_mixed_precision=config['training']['use_mixed_precision'],
        device=device,
        save_dir=Path("checkpoints"),
        early_stopping=EarlyStopping(patience=5, mode='min')
    )
    
    # 11. Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # 12. Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    logger.info("Training completed!")
    
    # 13. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics, test_info = evaluate_model(
        model,
        test_loader,
        device,
        task_type='classification',
        num_classes=2
    )
    logger.info(f"Test metrics: {test_metrics.to_dict()}")
    
    # 14. Create inference engine
    inference_engine = InferenceEngine(model, device=device)
    
    # 15. Run inference
    sample_input = {'input_ids': torch.randint(0, 10000, (1, 512))}
    predictions = inference_engine.predict(sample_input, return_probabilities=True)
    logger.info(f"Sample predictions: {predictions}")
    
    # 16. Create Gradio app (optional)
    # Uncomment to launch interactive demo
    # app = create_text_classification_app(model, class_names=['Class 0', 'Class 1'])
    # app.launch(share=True)
    
    # 17. Close tracking
    tracker.close()
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()



