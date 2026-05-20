from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Tuple
import random
import string
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import sys
import os
from deep_learning_models import (
from typing import Any, List, Dict, Optional
"""
🧠 Deep Learning Demo - Facebook Posts Processing
================================================

This demo showcases advanced deep learning models with proper weight initialization,
normalization techniques, loss functions, and optimization algorithms for Facebook Posts analysis.

Features Demonstrated:
- Multiple model architectures (Transformer, LSTM, CNN)
- Advanced weight initialization techniques
- Various normalization layers
- Comprehensive loss functions
- Optimized training pipelines
- GPU acceleration and mixed precision
"""


# Import our deep learning models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ModelConfig,
    WeightInitializer,
    NormalizationLayers,
    LossFunctions,
    OptimizerFactory,
    SchedulerFactory,
    FacebookPostsTransformer,
    FacebookPostsLSTM,
    FacebookPostsCNN,
    FacebookPostsDataset,
    FacebookPostsTrainer,
    create_facebook_posts_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


def generate_sample_data(num_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Generate sample Facebook posts data for demonstration."""
    sample_texts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏",
        "New features coming soon. Stay tuned! 🔥",
        "Working on exciting new projects. Can't wait to share! 💡",
        "Team building event was a huge success! 🎉",
        "Product update: Improved performance by 50%! 📈",
        "Customer satisfaction at all-time high! 🏆",
        "Innovation never stops. Always pushing boundaries! 🌟",
        "Thank you to our amazing community! You rock! 🤘",
        "New partnership announcement coming soon! 🤝",
        "Behind the scenes: Our development process 🛠️",
        "User experience improvements live now! ✨",
        "Data shows incredible growth this quarter! 📊",
        "Team collaboration leads to amazing results! 🎯",
        "Customer support team doing fantastic work! 👏",
        "Product roadmap update: Exciting features ahead! 🗺️",
        "Community feedback drives our decisions! 💬",
        "Innovation in action: Real-time analytics! 📱",
        "Success metrics exceeded expectations! 🎊"
    ]
    
    # Generate random posts
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # Randomly select and modify sample texts
        base_text = random.choice(sample_texts)
        
        # Add random variations
        variations = [
            f"Just {base_text.lower()}",
            f"Update: {base_text}",
            f"Breaking news: {base_text}",
            f"Excited to announce: {base_text}",
            f"Proud to share: {base_text}",
            base_text,
            f"🚀 {base_text}",
            f"💡 {base_text}",
            f"🎉 {base_text}",
            f"🔥 {base_text}"
        ]
        
        text = random.choice(variations)
        texts.append(text)
        
        # Generate random labels (0-4 for different categories)
        label = random.randint(0, 4)
        labels.append(label)
    
    return texts, labels


def create_tokenizer():
    """Create a simple tokenizer for demonstration."""
    # For demo purposes, we'll create a simple tokenizer
    # In production, you'd use a proper tokenizer like BERT's
    
    class SimpleTokenizer:
        def __init__(self) -> Any:
            self.vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
            self.vocab_size = 4
            self.max_length = 512
        
        def encode(self, text: str) -> List[int]:
            """Simple encoding for demonstration."""
            # Convert text to lowercase and split
            words = text.lower().split()
            
            # Convert words to token IDs
            token_ids = [self.vocab["<CLS>"]]
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1
                token_ids.append(self.vocab[word])
            
            token_ids.append(self.vocab["<SEP>"])
            
            # Pad or truncate
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids.extend([self.vocab["<PAD>"]] * (self.max_length - len(token_ids)))
            
            return token_ids
        
        def __call__(self, text, **kwargs) -> Any:
            """Compatible interface with transformers tokenizer."""
            token_ids = self.encode(text)
            attention_mask = [1 if token_id != self.vocab["<PAD>"] else 0 for token_id in token_ids]
            
            return {
                'input_ids': torch.tensor([token_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
    
    return SimpleTokenizer()


async def demo_weight_initialization():
    """Demonstrate different weight initialization techniques."""
    logger.info("🔧 Weight Initialization Demo")
    logger.info("=" * 50)
    
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    initialization_methods = [
        ("Xavier Uniform", WeightInitializer.xavier_uniform_init),
        ("Xavier Normal", WeightInitializer.xavier_normal_init),
        ("Kaiming Uniform", WeightInitializer.kaiming_uniform_init),
        ("Kaiming Normal", WeightInitializer.kaiming_normal_init),
        ("Orthogonal", WeightInitializer.orthogonal_init),
        ("Sparse", WeightInitializer.sparse_init)
    ]
    
    for name, init_method in initialization_methods:
        # Reset model weights
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.zeros_(module.bias)
        
        # Apply initialization method
        init_method(model)
        
        # Analyze weight statistics
        total_params = 0
        weight_norm = 0.0
        
        for param in model.parameters():
            total_params += param.numel()
            weight_norm += torch.norm(param).item()
        
        avg_weight_norm = weight_norm / total_params
        
        logger.info(f"{name:15} | Params: {total_params:6,} | Avg Norm: {avg_weight_norm:.4f}")
    
    logger.info("")


async def demo_normalization_layers():
    """Demonstrate different normalization techniques."""
    logger.info("📊 Normalization Layers Demo")
    logger.info("=" * 50)
    
    # Create sample data
    batch_size, seq_len, hidden_dim = 32, 100, 256
    sample_data = torch.randn(batch_size, seq_len, hidden_dim)
    
    normalization_methods = [
        ("Layer Norm", NormalizationLayers.layer_norm(hidden_dim)),
        ("Batch Norm", NormalizationLayers.batch_norm(hidden_dim)),
        ("Instance Norm", NormalizationLayers.instance_norm(hidden_dim)),
        ("Group Norm", NormalizationLayers.group_norm(8, hidden_dim)),
        ("Adaptive Layer Norm", NormalizationLayers.adaptive_layer_norm(hidden_dim))
    ]
    
    for name, norm_layer in normalization_methods:
        # Apply normalization
        if name == "Batch Norm":
            # Batch norm expects (N, C, L) format
            normalized = norm_layer(sample_data.transpose(1, 2)).transpose(1, 2)
        elif name == "Instance Norm":
            # Instance norm expects (N, C, L) format
            normalized = norm_layer(sample_data.transpose(1, 2)).transpose(1, 2)
        elif name == "Group Norm":
            # Group norm expects (N, C, L) format
            normalized = norm_layer(sample_data.transpose(1, 2)).transpose(1, 2)
        else:
            normalized = norm_layer(sample_data)
        
        # Calculate statistics
        mean = normalized.mean().item()
        std = normalized.std().item()
        var = normalized.var().item()
        
        logger.info(f"{name:20} | Mean: {mean:8.4f} | Std: {std:8.4f} | Var: {var:8.4f}")
    
    logger.info("")


async def demo_loss_functions():
    """Demonstrate different loss functions."""
    logger.info("📉 Loss Functions Demo")
    logger.info("=" * 50)
    
    # Create sample predictions and targets
    batch_size, num_classes = 32, 5
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # For regression tasks
    reg_predictions = torch.randn(batch_size, 1)
    reg_targets = torch.randn(batch_size, 1)
    
    # For embedding tasks
    embed_dim = 128
    anchor = torch.randn(batch_size, embed_dim)
    positive = torch.randn(batch_size, embed_dim)
    negative = torch.randn(batch_size, embed_dim)
    
    loss_functions = [
        ("Cross Entropy", lambda: LossFunctions.cross_entropy_loss(predictions, targets)),
        ("Focal Loss", lambda: LossFunctions.focal_loss(predictions, targets)),
        ("Huber Loss", lambda: LossFunctions.huber_loss(reg_predictions, reg_targets)),
        ("Cosine Embedding", lambda: LossFunctions.cosine_embedding_loss(anchor, positive)),
        ("Triplet Loss", lambda: LossFunctions.triplet_loss(anchor, positive, negative))
    ]
    
    for name, loss_fn in loss_functions:
        try:
            loss_value = loss_fn().item()
            logger.info(f"{name:20} | Loss: {loss_value:10.6f}")
        except Exception as e:
            logger.info(f"{name:20} | Error: {str(e)}")
    
    logger.info("")


async def demo_model_architectures():
    """Demonstrate different model architectures."""
    logger.info("🏗️ Model Architectures Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = ModelConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        vocab_size=1000,
        num_classes=5,
        max_seq_length=100
    )
    
    model_types = ["transformer", "lstm", "cnn"]
    
    for model_type in model_types:
        try:
            # Create model
            model = create_facebook_posts_model(model_type, config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test forward pass
            batch_size = 8
            input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_length))
            
            if model_type == "transformer":
                attention_mask = torch.ones(batch_size, config.max_seq_length)
                output = model(input_ids, attention_mask)
            else:
                output = model(input_ids)
            
            logger.info(f"{model_type.upper():12} | Params: {total_params:8,} | "
                       f"Trainable: {trainable_params:8,} | Output Shape: {output.shape}")
            
        except Exception as e:
            logger.info(f"{model_type.upper():12} | Error: {str(e)}")
    
    logger.info("")


async def demo_optimization_algorithms():
    """Demonstrate different optimization algorithms."""
    logger.info("⚡ Optimization Algorithms Demo")
    logger.info("=" * 50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create sample data
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    
    # Test different optimizers
    optimizers = [
        ("Adam", optim.Adam(model.parameters(), lr=1e-3)),
        ("AdamW", optim.AdamW(model.parameters(), lr=1e-3)),
        ("SGD", optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)),
        ("RMSprop", optim.RMSprop(model.parameters(), lr=1e-3)),
        ("Adagrad", optim.Adagrad(model.parameters(), lr=1e-2))
    ]
    
    for name, optimizer in optimizers:
        # Reset model
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Training loop
        losses = []
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        loss_reduction = losses[0] - losses[-1]
        
        logger.info(f"{name:10} | Final Loss: {final_loss:8.4f} | "
                   f"Reduction: {loss_reduction:8.4f}")
    
    logger.info("")


async def demo_training_pipeline():
    """Demonstrate complete training pipeline."""
    logger.info("🚀 Training Pipeline Demo")
    logger.info("=" * 50)
    
    # Generate sample data
    texts, labels = generate_sample_data(500)
    tokenizer = create_tokenizer()
    
    # Create dataset
    dataset = FacebookPostsDataset(texts, labels, tokenizer, max_length=100)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create configuration
    config = ModelConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=16,
        max_epochs=5,
        patience=3,
        gradient_clip=1.0,
        use_mixed_precision=False,  # Disable for demo
        vocab_size=tokenizer.vocab_size,
        num_classes=5,
        max_seq_length=100,
        task_type="classification",
        use_focal_loss=False,
        optimizer_type="adam",
        scheduler_type="step",
        step_size=2,
        gamma=0.5
    )
    
    # Create model and trainer
    model = create_facebook_posts_model("lstm", config)
    trainer = FacebookPostsTrainer(model, config)
    
    logger.info(f"Training on {len(train_dataset)} samples")
    logger.info(f"Validating on {len(val_dataset)} samples")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    start_time = time.time()
    train_losses, val_losses, train_accs, val_accs = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final train accuracy: {train_accs[-1]:.4f}")
    logger.info(f"Final validation accuracy: {val_accs[-1]:.4f}")
    
    logger.info("")


async def demo_mixed_precision_training():
    """Demonstrate mixed precision training."""
    logger.info("🎯 Mixed Precision Training Demo")
    logger.info("=" * 50)
    
    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping mixed precision demo")
        return
    
    # Create configuration
    config = ModelConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        max_epochs=3,
        patience=2,
        gradient_clip=1.0,
        use_mixed_precision=True,
        vocab_size=1000,
        num_classes=5,
        max_seq_length=100,
        task_type="classification",
        use_focal_loss=False,
        optimizer_type="adam",
        scheduler_type="step",
        step_size=1,
        gamma=0.5
    )
    
    # Create model
    model = create_facebook_posts_model("transformer", config)
    
    # Create sample data
    batch_size = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_length)).cuda()
    attention_mask = torch.ones(batch_size, config.max_seq_length).cuda()
    labels = torch.randint(0, config.num_classes, (batch_size,)).cuda()
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info("Starting mixed precision training...")
    start_time = time.time()
    
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        
        for step in range(10):  # 10 steps per epoch
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 10
        logger.info(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"Mixed precision training completed in {training_time:.2f} seconds")
    
    logger.info("")


async def demo_model_comparison():
    """Compare different model architectures."""
    logger.info("📊 Model Comparison Demo")
    logger.info("=" * 50)
    
    # Generate sample data
    texts, labels = generate_sample_data(200)
    tokenizer = create_tokenizer()
    
    # Create dataset
    dataset = FacebookPostsDataset(texts, labels, tokenizer, max_length=100)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Configuration
    config = ModelConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=16,
        max_epochs=3,
        patience=2,
        gradient_clip=1.0,
        use_mixed_precision=False,
        vocab_size=tokenizer.vocab_size,
        num_classes=5,
        max_seq_length=100,
        task_type="classification",
        use_focal_loss=False,
        optimizer_type="adam",
        scheduler_type="step",
        step_size=1,
        gamma=0.5
    )
    
    model_types = ["transformer", "lstm", "cnn"]
    results = {}
    
    for model_type in model_types:
        logger.info(f"Training {model_type.upper()} model...")
        
        # Create model and trainer
        model = create_facebook_posts_model(model_type, config)
        trainer = FacebookPostsTrainer(model, config)
        
        # Train model
        start_time = time.time()
        train_losses, val_losses, train_accs, val_accs = trainer.train(train_loader, val_loader)
        training_time = time.time() - start_time
        
        # Store results
        results[model_type] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'training_time': training_time,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    # Display comparison
    logger.info("\nModel Comparison Results:")
    logger.info("-" * 80)
    logger.info(f"{'Model':12} | {'Params':>8} | {'Time(s)':>8} | {'Train Acc':>10} | {'Val Acc':>8} | {'Train Loss':>10} | {'Val Loss':>8}")
    logger.info("-" * 80)
    
    for model_type, result in results.items():
        logger.info(f"{model_type.upper():12} | {result['parameters']:8,} | "
                   f"{result['training_time']:8.2f} | {result['final_train_acc']:10.4f} | "
                   f"{result['final_val_acc']:8.4f} | {result['final_train_loss']:10.4f} | "
                   f"{result['final_val_loss']:8.4f}")
    
    logger.info("")


async def run_deep_learning_demo():
    """Run the complete deep learning demonstration."""
    logger.info("🧠 DEEP LEARNING DEMO - Facebook Posts Processing")
    logger.info("=" * 60)
    logger.info("Demonstrating advanced deep learning techniques with proper")
    logger.info("weight initialization, normalization, loss functions, and optimization")
    logger.info("=" * 60)
    
    # Run all demos
    await demo_weight_initialization()
    await demo_normalization_layers()
    await demo_loss_functions()
    await demo_model_architectures()
    await demo_optimization_algorithms()
    await demo_training_pipeline()
    await demo_mixed_precision_training()
    await demo_model_comparison()
    
    logger.info("🎉 Deep Learning Demo Completed Successfully!")
    logger.info("All techniques demonstrated:")
    logger.info("✅ Weight initialization (Xavier, Kaiming, Orthogonal, Sparse)")
    logger.info("✅ Normalization layers (Layer, Batch, Instance, Group)")
    logger.info("✅ Loss functions (Cross-entropy, Focal, Huber, Triplet)")
    logger.info("✅ Model architectures (Transformer, LSTM, CNN)")
    logger.info("✅ Optimization algorithms (Adam, AdamW, SGD, RMSprop)")
    logger.info("✅ Training pipeline with validation")
    logger.info("✅ Mixed precision training")
    logger.info("✅ Model comparison and benchmarking")


async def quick_deep_learning_demo():
    """Quick demonstration of key features."""
    logger.info("⚡ QUICK DEEP LEARNING DEMO")
    logger.info("=" * 40)
    
    # Quick weight initialization demo
    logger.info("Weight Initialization:")
    model = nn.Linear(100, 50)
    WeightInitializer.xavier_uniform_init(model)
    logger.info(f"Xavier uniform initialized - Weight norm: {torch.norm(model.weight):.4f}")
    
    # Quick normalization demo
    logger.info("\nNormalization:")
    data = torch.randn(32, 100)
    layer_norm = NormalizationLayers.layer_norm(100)
    normalized = layer_norm(data)
    logger.info(f"Layer norm applied - Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
    
    # Quick model creation
    logger.info("\nModel Creation:")
    config = ModelConfig(
        input_dim=256,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        vocab_size=1000,
        num_classes=5,
        max_seq_length=50
    )
    
    model = create_facebook_posts_model("lstm", config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LSTM model created - Parameters: {total_params:,}")
    
    logger.info("\n✅ Quick demo completed!")


if __name__ == "__main__":
    # Run the complete demo
    asyncio.run(run_deep_learning_demo())
    
    # Uncomment for quick demo
    # asyncio.run(quick_deep_learning_demo()) 