"""
Complete Training Example
Demonstrates best practices for training models
"""

import torch
from ml import (
    ViTSkinAnalyzer,
    Trainer,
    SkinDataset,
    get_train_transforms,
    get_val_transforms,
    MultiTaskLoss,
    get_optimizer,
    get_scheduler,
    create_data_loaders
)
from ml.experiments import ExperimentTracker, ExperimentConfig
from utils.advanced_optimization import enable_all_optimizations
from config import load_config

def main():
    """Complete training pipeline"""
    
    # 1. Load configuration
    config = load_config("config/model_config.yaml")
    
    # 2. Enable optimizations
    enable_all_optimizations()
    
    # 3. Create model
    model = ViTSkinAnalyzer(
        num_conditions=config['model']['num_conditions'],
        num_metrics=config['model']['num_metrics'],
        use_pretrained=config['model']['parameters']['use_pretrained']
    )
    
    # 4. Prepare data
    # (Assuming you have train_images, train_labels, etc.)
    train_dataset = SkinDataset(
        images=train_images,
        labels={
            'conditions': train_conditions,
            'metrics': train_metrics
        },
        transform=get_train_transforms(
            target_size=tuple(config['model']['input_size']),
            augmentation_strength=config['data']['augmentation']['strength']
        ),
        cache_images=True
    )
    
    val_dataset = SkinDataset(
        images=val_images,
        labels={
            'conditions': val_conditions,
            'metrics': val_metrics
        },
        transform=get_val_transforms(
            target_size=tuple(config['model']['input_size'])
        )
    )
    
    # 5. Create data loaders
    loaders = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # 6. Setup experiment tracking
    tracker = ExperimentTracker(
        use_wandb=config['experiment_tracking']['use_wandb'],
        use_tensorboard=config['experiment_tracking']['use_tensorboard'],
        wandb_project=config['experiment_tracking']['wandb_project']
    )
    
    exp_config = ExperimentConfig(
        experiment_id="exp_001",
        name="ViT Skin Analysis",
        description="Training Vision Transformer for skin analysis",
        model_type=config['model']['type'],
        hyperparameters=config['training']
    )
    tracker.create_experiment(exp_config)
    
    # 7. Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        device=config['device']['type'],
        use_mixed_precision=config['training']['use_mixed_precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        experiment_tracker=tracker
    )
    
    # 8. Setup loss, optimizer, scheduler
    loss_fn = MultiTaskLoss(
        condition_weight=config['training']['loss']['condition_weight'],
        metric_weight=config['training']['loss']['metric_weight']
    )
    
    optimizer = get_optimizer(
        model,
        optimizer_name=config['training']['optimizer']['name'],
        learning_rate=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=config['training']['scheduler']['name'],
        num_epochs=config['training']['num_epochs']
    )
    
    # 9. Train
    trainer.fit(
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        scheduler=scheduler,
        criterion=loss_fn,
        checkpoint_dir=config['checkpointing']['save_dir']
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()













