"""
Training Example - Addition Removal AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from addition_removal_ai import (
    ModelTrainer,
    create_lora_model,
    Config,
    create_default_config_file
)
from transformers import AutoModel, AutoTokenizer


class ContentDataset(Dataset):
    """Example dataset"""
    
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.texts[idx], self.labels[idx]
        return self.texts[idx]


def train_example():
    """Example training"""
    print("=== Training Example ===\n")
    
    # Create default config
    create_default_config_file("my_training_config.yaml")
    config = Config.from_yaml("my_training_config.yaml")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(config.model.model_name).to(device)
    
    # Apply LoRA if enabled
    if config.model.use_lora:
        print("Applying LoRA...")
        model = create_lora_model(
            model,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            device=device
        )
    
    # Create dummy dataset
    train_texts = [f"Sample text {i}" for i in range(100)]
    train_labels = [i % 3 for i in range(100)]
    train_dataset = ContentDataset(train_texts, train_labels)
    
    val_texts = [f"Validation text {i}" for i in range(20)]
    val_labels = [i % 3 for i in range(20)]
    val_dataset = ContentDataset(val_texts, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_mixed_precision=config.training.use_mixed_precision,
        log_dir=config.experiment.log_dir,
        use_wandb=config.experiment.use_wandb
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config.scheduler.num_training_steps
    )
    
    # Train
    trainer.train(
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.training.num_epochs,
        scheduler=scheduler,
        clip_grad_norm=config.training.clip_grad_norm,
        save_dir=config.experiment.save_dir
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    train_example()

