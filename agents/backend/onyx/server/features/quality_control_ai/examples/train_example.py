"""
Example: Training Quality Control AI Models
"""

import torch
from quality_control_ai import (
    create_autoencoder,
    create_defect_classifier,
    Config,
    ModelTrainer,
    create_default_config_file
)
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim


class QualityControlDataset(Dataset):
    """Example dataset for quality control"""
    
    def __init__(self, num_samples=1000, image_size=(224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Replace with actual image loading
        image = torch.randn(3, *self.image_size)
        return image


def train_autoencoder_example():
    """Example: Train autoencoder for anomaly detection"""
    print("Training Autoencoder for Anomaly Detection")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_autoencoder(
        input_channels=3,
        latent_dim=128,
        input_size=(224, 224),
        device=device
    )
    
    # Create dataset
    train_dataset = QualityControlDataset(1000, (224, 224))
    val_dataset = QualityControlDataset(200, (224, 224))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_mixed_precision=True,
        log_dir="./logs/autoencoder"
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Train
    trainer.train(
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=10,
        save_dir="./checkpoints/autoencoder"
    )
    
    print("Training completed!")


def train_classifier_example():
    """Example: Train ViT classifier for defect classification"""
    print("Training ViT Classifier for Defect Classification")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_defect_classifier(
        num_classes=10,
        model_name="google/vit-base-patch16-224",
        pretrained=True,
        device=device
    )
    
    # Create dataset (with labels)
    class LabeledDataset(Dataset):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 10, (1,)).item()
            return image, label
    
    train_dataset = LabeledDataset(1000)
    val_dataset = LabeledDataset(200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_mixed_precision=True,
        log_dir="./logs/classifier"
    )
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    trainer.train(
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=10,
        save_dir="./checkpoints/classifier"
    )
    
    print("Training completed!")


def config_example():
    """Example: Using YAML configuration"""
    print("Creating default configuration...")
    
    # Create default config
    create_default_config_file("my_config.yaml")
    
    # Load config
    config = Config.from_yaml("my_config.yaml")
    
    print(f"Model type: {config.model.model_type}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    
    # Modify and save
    config.training.batch_size = 64
    config.to_yaml("my_config_modified.yaml")
    
    print("Configuration saved!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "autoencoder":
            train_autoencoder_example()
        elif sys.argv[1] == "classifier":
            train_classifier_example()
        elif sys.argv[1] == "config":
            config_example()
    else:
        print("Usage:")
        print("  python train_example.py autoencoder  # Train autoencoder")
        print("  python train_example.py classifier  # Train classifier")
        print("  python train_example.py config      # Config example")

