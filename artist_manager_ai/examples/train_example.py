#!/usr/bin/env python3
"""
Complete Training Example
========================

Complete example of training a model using the modular architecture.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.factories import ModelFactory, TrainerFactory, DataFactory
from ml.utils import CheckpointManager, MetricsTracker
from ml.utils.visualization import TrainingVisualizer
from ml.utils.debugging import Debugger


def main():
    """Main training function."""
    print("🚀 Starting Training Example")
    
    # 1. Create factories
    model_factory = ModelFactory()
    data_factory = DataFactory()
    trainer_factory = TrainerFactory()
    
    # 2. Create model
    print("📦 Creating model...")
    model = model_factory.create(
        "event_duration",
        config={
            "input_dim": 32,
            "hidden_dims": [128, 64, 32],
            "dropout_rate": 0.2,
            "use_batch_norm": True
        }
    )
    print(f"✅ Model created: {type(model).__name__}")
    
    # 3. Create dataset (using dummy data for example)
    print("📊 Creating dataset...")
    from datetime import datetime, timedelta
    
    events = [
        {
            "type": "concert",
            "start_time": (datetime.now() + timedelta(days=i)).isoformat(),
            "end_time": (datetime.now() + timedelta(days=i, hours=3)).isoformat(),
            "location": f"Venue {i % 5}",
            "description": f"Concert {i}"
        }
        for i in range(100)
    ]
    
    dataset = data_factory.create_dataset("event", events)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # 4. Create dataloaders
    print("🔄 Creating dataloaders...")
    train_loader, val_loader, test_loader = data_factory.create_dataloaders(
        dataset,
        config={
            "batch_size": 16,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1
        }
    )
    print(f"✅ Dataloaders created")
    
    # 5. Create trainer
    print("🏋️ Creating trainer...")
    trainer = trainer_factory.create(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.MSELoss(),
        config={
            "learning_rate": 0.001,
            "use_mixed_precision": True,
            "grad_clip": 1.0,
            "early_stopping_patience": 5
        },
        optimizer_type="adam"
    )
    print("✅ Trainer created")
    
    # 6. Setup utilities
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
    metrics_tracker = MetricsTracker()
    visualizer = TrainingVisualizer()
    debugger = Debugger(enable_anomaly_detection=False)
    
    # 7. Training loop
    print("🎯 Starting training...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        with debugger.detect_anomaly():
            train_metrics = trainer.train_epoch()
            metrics_tracker.log_batch(train_metrics, step=epoch)
        
        # Validate
        val_metrics = trainer.validate()
        metrics_tracker.log_batch(val_metrics, step=epoch)
        
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        is_best = (
            metrics_tracker.get_best("val_loss") is None or
            val_metrics["loss"] < metrics_tracker.get_best("val_loss")
        )
        
        checkpoint_manager.save(
            model,
            trainer.optimizer,
            epoch=epoch,
            metrics=val_metrics,
            is_best=is_best
        )
    
    # 8. Visualize results
    print("\n📈 Generating visualizations...")
    history = metrics_tracker.get_all_metrics()
    visualizer.plot_training_history(history)
    print("✅ Visualizations saved")
    
    # 9. Print summary
    print("\n📊 Training Summary:")
    print(metrics_tracker.summary())
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()




