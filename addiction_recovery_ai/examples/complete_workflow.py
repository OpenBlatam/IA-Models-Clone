"""
Complete Workflow Example
Demonstrates the full pipeline from data to deployment
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import all components
from addiction_recovery_ai import (
    # Configuration
    get_config,
    
    # Models
    ModelFactory,
    ModelBuilder,
    
    # Data
    create_recovery_dataset,
    DataLoaderFactory,
    
    # Training
    TrainerFactory,
    create_tracker,
    create_checkpoint_manager,
    create_evaluator,
    
    # Inference
    create_ultra_fast_inference,
    create_embedding_cache,
    
    # Optimization
    optimize_model_memory,
    get_memory_stats
)


def main():
    """Complete workflow example"""
    
    # 1. Load Configuration
    print("📋 Loading configuration...")
    config = get_config("config/model_config.yaml")
    
    # 2. Create Model
    print("🤖 Creating model...")
    model_config = config.get_model_config("progress_predictor")
    model = ModelFactory.create("RecoveryProgressPredictor", model_config)
    
    # 3. Create Datasets
    print("📊 Creating datasets...")
    # Mock data
    train_data = [
        {"days_sober": 30, "cravings": 3, "stress": 4, "mood": 7, "progress": 0.75}
        for _ in range(1000)
    ]
    val_data = [
        {"days_sober": 30, "cravings": 3, "stress": 4, "mood": 7, "progress": 0.75}
        for _ in range(200)
    ]
    
    train_dataset = create_recovery_dataset(
        train_data,
        feature_keys=["days_sober", "cravings", "stress", "mood"],
        target_key="progress"
    )
    val_dataset = create_recovery_dataset(
        val_data,
        feature_keys=["days_sober", "cravings", "stress", "mood"],
        target_key="progress"
    )
    
    # 4. Create Data Loaders
    print("🔄 Creating data loaders...")
    data_config = config.get_data_config()
    train_loader = DataLoaderFactory.create(train_dataset, data_config, split="train")
    val_loader = DataLoaderFactory.create(val_dataset, data_config, split="val")
    
    # 5. Create Trainer
    print("🏋️ Creating trainer...")
    training_config = config.get_training_config()
    trainer = TrainerFactory.create(
        "RecoveryModelTrainer",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # 6. Setup Experiment Tracking
    print("📈 Setting up experiment tracking...")
    tracker = create_tracker(
        experiment_name="recovery_model_v1",
        use_tensorboard=True,
        use_wandb=False
    )
    
    # 7. Setup Checkpointing
    print("💾 Setting up checkpointing...")
    checkpoint_manager = create_checkpoint_manager("checkpoints")
    
    # 8. Setup Evaluator
    print("📊 Setting up evaluator...")
    evaluator = create_evaluator()
    
    # 9. Training
    print("🚀 Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    num_epochs = config.get("training.num_epochs", 10)
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(optimizer, criterion, epoch)
        
        # Validate
        val_metrics = trainer.validate(criterion)
        
        # Log metrics
        tracker.log_metrics(train_metrics, epoch, prefix="train")
        tracker.log_metrics(val_metrics, epoch, prefix="val")
        
        # Checkpoint
        is_best = val_metrics.get("loss", float('inf')) < best_loss
        if is_best:
            best_loss = val_metrics.get("loss", float('inf'))
        
        checkpoint_manager.save(
            model=model,
            epoch=epoch,
            metrics={**train_metrics, **val_metrics},
            optimizer=optimizer,
            is_best=is_best
        )
        
        print(f"Epoch {epoch}: Train Loss={train_metrics.get('loss', 0):.4f}, "
              f"Val Loss={val_metrics.get('loss', 0):.4f}")
    
    # 10. Final Evaluation
    print("📊 Final evaluation...")
    final_metrics = evaluator.evaluate_regression(model, val_loader, criterion)
    print(f"Final Metrics: {final_metrics}")
    
    # 11. Optimize for Inference
    print("⚡ Optimizing for inference...")
    optimize_model_memory(model)
    fast_engine = create_ultra_fast_inference(model)
    
    # 12. Test Inference
    print("🧪 Testing inference...")
    test_input = torch.tensor([[30/365, 0.3, 0.4, 0.7]], dtype=torch.float32)
    output = fast_engine.predict(test_input)
    print(f"Test prediction: {output.item():.4f}")
    
    # 13. Memory Stats
    print("💾 Memory statistics...")
    stats = get_memory_stats()
    print(f"GPU Memory: {stats.get('allocated_gb', 0):.2f} GB")
    
    # 14. Cleanup
    tracker.close()
    print("✅ Complete workflow finished!")


if __name__ == "__main__":
    main()













