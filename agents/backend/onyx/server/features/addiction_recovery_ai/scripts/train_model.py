#!/usr/bin/env python3
"""
Training Script
Complete training script with all features
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from addiction_recovery_ai import (
    get_config,
    ModelFactory,
    TrainerFactory,
    DataLoaderFactory,
    create_recovery_dataset,
    create_tracker,
    create_checkpoint_manager,
    create_evaluator,
    optimize_model_memory,
    split_data
)


def main():
    parser = argparse.ArgumentParser(description="Train recovery model")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--model", type=str, required=True,
                       help="Model type to train")
    parser.add_argument("--data", type=str, required=True,
                       help="Training data path")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--output", type=str, default="checkpoints",
                       help="Output directory")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Enable TensorBoard")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable WandB")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with CLI args
    if args.epochs:
        config.set("training.num_epochs", args.epochs)
    if args.batch_size:
        config.set("training.batch_size", args.batch_size)
        config.set("data.batch_size", args.batch_size)
    
    # Create model
    print(f"Creating {args.model} model...")
    model_config = config.get_model_config(args.model)
    model = ModelFactory.create(args.model, model_config)
    
    # Load data
    print(f"Loading data from {args.data}...")
    # TODO: Implement data loading from file
    # For now, using mock data
    import json
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    # Create datasets
    feature_keys = model_config.get("feature_keys", ["days_sober", "cravings", "stress", "mood"])
    target_key = model_config.get("target_key", "progress")
    
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    train_dataset = create_recovery_dataset(train_data, feature_keys, target_key)
    val_dataset = create_recovery_dataset(val_data, feature_keys, target_key)
    
    # Create data loaders
    data_config = config.get_data_config()
    train_loader = DataLoaderFactory.create(train_dataset, data_config, split="train")
    val_loader = DataLoaderFactory.create(val_dataset, data_config, split="val")
    
    # Create trainer
    training_config = config.get_training_config()
    trainer = TrainerFactory.create(
        "RecoveryModelTrainer",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Setup tracking
    experiment_name = args.experiment or f"{args.model}_{Path(args.data).stem}"
    tracker = create_tracker(
        experiment_name=experiment_name,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb
    )
    
    # Setup checkpointing
    checkpoint_manager = create_checkpoint_manager(args.output)
    
    # Setup evaluator
    evaluator = create_evaluator()
    
    # Training
    optimizer_config = training_config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.001)
    )
    
    criterion = nn.BCELoss()
    
    num_epochs = config.get("training.num_epochs", 10)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        train_metrics = trainer.train_epoch(optimizer, criterion, epoch)
        val_metrics = trainer.validate(criterion)
        
        tracker.log_metrics(train_metrics, epoch, prefix="train")
        tracker.log_metrics(val_metrics, epoch, prefix="val")
        
        is_best = val_metrics.get("loss", float('inf')) < checkpoint_manager.best_metric
        checkpoint_manager.save(
            model=model,
            epoch=epoch,
            metrics={**train_metrics, **val_metrics},
            optimizer=optimizer,
            is_best=is_best
        )
        
        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train Loss={train_metrics.get('loss', 0):.4f}, "
              f"Val Loss={val_metrics.get('loss', 0):.4f}")
    
    # Final evaluation
    final_metrics = evaluator.evaluate_regression(model, val_loader, criterion)
    print(f"Final Metrics: {final_metrics}")
    
    # Optimize for inference
    optimize_model_memory(model)
    
    tracker.close()
    print("Training completed!")


if __name__ == "__main__":
    main()

