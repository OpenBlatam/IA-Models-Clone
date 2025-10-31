from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import logging
from custom_model_architectures import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Usage Examples for Custom PyTorch Model Architectures
Practical examples demonstrating how to use each custom model
"""


# Import custom architectures
    CustomTransformer, CNNLSTMHybrid, TransformerCNN, MultiTaskModel,
    HierarchicalAttentionNetwork, DeepResidualCNN, ModelFactory,
    MODEL_CONFIGS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelUsageExamples:
    """Comprehensive examples for using custom model architectures"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
    
    def example_custom_transformer(self) -> None:
        """Example: Custom Transformer for text classification"""
        logger.info("=== Custom Transformer Example ===")
        
        # Configuration
        config = {
            "vocab_size": 10000,
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,
            "activation": "gelu",
            "use_relative_pos": True
        }
        
        # Create model
        model = CustomTransformer(**config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample data
        batch_size = 4
        seq_len = 50
        x = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(self.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Output shape: {output.shape}")
        
        # Training example
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Dummy labels
        labels = torch.randint(0, 5, (batch_size,)).to(self.device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.mean(dim=1), labels)  # Global average pooling
        loss.backward()
        optimizer.step()
        
        logger.info(f"Training loss: {loss.item():.4f}")
    
    def example_cnn_lstm_hybrid(self) -> None:
        """Example: CNN-LSTM Hybrid for sentiment analysis"""
        logger.info("=== CNN-LSTM Hybrid Example ===")
        
        # Configuration
        config = {
            "vocab_size": 15000,
            "embed_dim": 300,
            "hidden_dims": [128, 256, 512],
            "kernel_sizes": [3, 4, 5],
            "lstm_hidden_size": 256,
            "num_classes": 3,  # Positive, Negative, Neutral
            "dropout": 0.2,
            "bidirectional": True
        }
        
        # Create model
        model = CNNLSTMHybrid(**config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample data with variable lengths
        batch_size = 8
        max_seq_len = 100
        x = torch.randint(0, config["vocab_size"], (batch_size, max_seq_len)).to(self.device)
        lengths = torch.randint(20, max_seq_len, (batch_size,)).to(self.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, lengths)
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Output shape: {output.shape}")
        
        # Training example with focal loss
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=2e-4)
        
        # Focal loss for imbalanced classes
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2) -> Any:
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, inputs, targets) -> Any:
                ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss()
        labels = torch.randint(0, config["num_classes"], (batch_size,)).to(self.device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x, lengths)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        logger.info(f"Training loss: {loss.item():.4f}")
    
    def example_transformer_cnn(self) -> None:
        """Example: Transformer-CNN Hybrid for topic classification"""
        logger.info("=== Transformer-CNN Hybrid Example ===")
        
        # Configuration
        config = {
            "vocab_size": 20000,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 1024,
            "cnn_hidden_dims": [128, 256, 512],
            "cnn_kernel_sizes": [3, 4, 5],
            "num_classes": 10,  # Different topics
            "dropout": 0.15
        }
        
        # Create model
        model = TransformerCNN(**config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample data with attention mask
        batch_size = 6
        seq_len = 80
        x = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(self.device)
        mask = torch.ones(batch_size, seq_len, seq_len).to(self.device)
        mask[:, :, 60:] = 0  # Mask last 20 positions
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x, mask)
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Output shape: {output.shape}")
        
        # Training example with label smoothing
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        labels = torch.randint(0, config["num_classes"], (batch_size,)).to(self.device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x, mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        logger.info(f"Training loss: {loss.item():.4f}")
    
    def example_multi_task_model(self) -> None:
        """Example: Multi-task learning model"""
        logger.info("=== Multi-task Model Example ===")
        
        # Configuration
        config = {
            "vocab_size": 12000,
            "d_model": 384,
            "n_layers": 3,
            "n_heads": 6,
            "d_ff": 1536,
            "task_configs": {
                "sentiment": {"num_classes": 3, "type": "classification"},
                "topic": {"num_classes": 5, "type": "classification"},
                "readability": {"num_classes": 1, "type": "regression"},
                "quality": {"num_classes": 1, "type": "regression"}
            },
            "dropout": 0.1
        }
        
        # Create model
        model = MultiTaskModel(**config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample data
        batch_size = 4
        seq_len = 60
        x = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(self.device)
        
        # Test each task
        model.eval()
        with torch.no_grad():
            for task_name in config["task_configs"].keys():
                output = model(x, task_name)
                expected_classes = config["task_configs"][task_name]["num_classes"]
                logger.info(f"{task_name} output shape: {output.shape}")
                assert output.shape == (batch_size, expected_classes)
        
        # Training example with task-specific losses
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Task-specific criteria
        criteria = {
            "sentiment": nn.CrossEntropyLoss(),
            "topic": nn.CrossEntropyLoss(),
            "readability": nn.MSELoss(),
            "quality": nn.MSELoss()
        }
        
        # Dummy labels for each task
        labels = {
            "sentiment": torch.randint(0, 3, (batch_size,)).to(self.device),
            "topic": torch.randint(0, 5, (batch_size,)).to(self.device),
            "readability": torch.rand(batch_size, 1).to(self.device),
            "quality": torch.rand(batch_size, 1).to(self.device)
        }
        
        # Training step for all tasks
        total_loss = 0
        optimizer.zero_grad()
        
        for task_name in config["task_configs"].keys():
            output = model(x, task_name)
            loss = criteria[task_name](output, labels[task_name])
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        
        logger.info(f"Total training loss: {total_loss.item():.4f}")
    
    def example_hierarchical_attention(self) -> None:
        """Example: Hierarchical Attention Network for document classification"""
        logger.info("=== Hierarchical Attention Network Example ===")
        
        # Configuration
        config = {
            "vocab_size": 8000,
            "embed_dim": 200,
            "hidden_size": 128,
            "num_classes": 4,  # Document categories
            "num_sentences": 25,
            "dropout": 0.2
        }
        
        # Create model
        model = HierarchicalAttentionNetwork(**config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample data (documents with sentences)
        batch_size = 3
        sentence_length = 20
        x = torch.randint(0, config["vocab_size"], 
                         (batch_size, config["num_sentences"], sentence_length)).to(self.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Output shape: {output.shape}")
        
        # Training example
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        labels = torch.randint(0, config["num_classes"], (batch_size,)).to(self.device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        logger.info(f"Training loss: {loss.item():.4f}")
    
    def example_deep_residual_cnn(self) -> None:
        """Example: Deep Residual CNN for text processing"""
        logger.info("=== Deep Residual CNN Example ===")
        
        # Configuration
        config = {
            "input_dim": 300,  # Word embeddings
            "hidden_dims": [64, 128, 256, 512],
            "num_classes": 6,
            "num_residual_blocks": 3,
            "dropout": 0.15
        }
        
        # Create model
        model = DeepResidualCNN(**config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create sample data
        batch_size = 5
        seq_len = 40
        x = torch.randn(batch_size, seq_len, config["input_dim"]).to(self.device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Output shape: {output.shape}")
        
        # Training example with learning rate scheduling
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        labels = torch.randint(0, config["num_classes"], (batch_size,)).to(self.device)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        logger.info(f"Training loss: {loss.item():.4f}")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    def example_model_factory(self) -> None:
        """Example: Using ModelFactory for easy model creation"""
        logger.info("=== Model Factory Example ===")
        
        # Create different models using factory
        models = {}
        
        for model_type, config in MODEL_CONFIGS.items():
            try:
                model = ModelFactory.create_model(model_type, config)
                models[model_type] = model
                logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
            except Exception as e:
                logger.warning(f"Failed to create {model_type} model: {e}")
        
        # Test model inference
        batch_size = 2
        seq_len = 30
        
        for model_type, model in models.items():
            model = model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                if "vocab_size" in MODEL_CONFIGS[model_type]:
                    vocab_size = MODEL_CONFIGS[model_type]["vocab_size"]
                    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
                else:
                    # For models that don't use vocab_size
                    input_dim = MODEL_CONFIGS[model_type].get("input_dim", 300)
                    x = torch.randn(batch_size, seq_len, input_dim).to(self.device)
                
                output = model(x)
                logger.info(f"{model_type} output shape: {output.shape}")
    
    def example_training_pipeline(self) -> None:
        """Example: Complete training pipeline with custom model"""
        logger.info("=== Complete Training Pipeline Example ===")
        
        # Create model
        model = CustomTransformer(
            vocab_size=5000,
            d_model=256,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.1
        ).to(self.device)
        
        # Create synthetic dataset
        num_samples = 1000
        seq_len = 50
        num_classes = 5
        
        # Generate random data
        x_data = torch.randint(0, 5000, (num_samples, seq_len))
        y_data = torch.randint(0, num_classes, (num_samples,))
        
        # Create data loader
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * 10)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(3):
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.mean(dim=1), y)  # Global average pooling
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")
    
    def run_all_examples(self) -> None:
        """Run all usage examples"""
        logger.info("Starting all custom architecture examples...")
        
        examples = [
            self.example_custom_transformer,
            self.example_cnn_lstm_hybrid,
            self.example_transformer_cnn,
            self.example_multi_task_model,
            self.example_hierarchical_attention,
            self.example_deep_residual_cnn,
            self.example_model_factory,
            self.example_training_pipeline
        ]
        
        for example in examples:
            try:
                example()
                logger.info("Example completed successfully\n")
            except Exception as e:
                logger.error(f"Example failed: {e}\n")


def main():
    """Main function to run usage examples"""
    # Create usage examples
    examples = ModelUsageExamples()
    
    # Run all examples
    examples.run_all_examples()
    
    logger.info("All custom architecture examples completed!")


match __name__:
    case "__main__":
    main() 