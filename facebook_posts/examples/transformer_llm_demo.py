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

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import random
import string
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from transformer_llm_models import (
from typing import Any, List, Dict, Optional
"""
🧠 Transformer & LLM Demo - Facebook Posts Processing
====================================================

This demo showcases advanced transformer architectures and Large Language Models
for Facebook Posts analysis, generation, and processing.

Features Demonstrated:
- Multi-head attention mechanisms
- Position encoding and embeddings
- Advanced transformer architectures
- LLM training and fine-tuning
- Attention visualization
- Text generation
- Model compression techniques
"""


# Import our transformer and LLM models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    TransformerConfig,
    FacebookPostsTransformer,
    FacebookPostsLLM,
    FacebookPostsLLMTrainer,
    AttentionVisualizer,
    ModelCompressor,
    create_transformer_model,
    create_llm_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


def generate_facebook_posts_data(num_samples: int = 1000) -> List[str]:
    """Generate sample Facebook posts data for demonstration."""
    sample_posts = [
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
        "Success metrics exceeded expectations! 🎊",
        "Breaking news: Major milestone achieved! 🎯",
        "Excited to announce our latest breakthrough! 🚀",
        "Customer success stories inspire us daily! 💪",
        "Technology is transforming our industry! ⚡",
        "Collaboration across teams delivers results! 🤝",
        "Innovation culture drives our success! 💡",
        "Data-driven decisions lead to growth! 📊",
        "User-centric design wins every time! 🎨",
        "Agile methodology accelerates development! ⚡",
        "Continuous improvement is our mantra! 🔄",
        "Customer-first approach never fails! ❤️"
    ]
    
    # Generate variations
    posts = []
    for _ in range(num_samples):
        base_post = random.choice(sample_posts)
        
        # Add random variations
        variations = [
            f"Just {base_post.lower()}",
            f"Update: {base_post}",
            f"Breaking news: {base_post}",
            f"Excited to announce: {base_post}",
            f"Proud to share: {base_post}",
            base_post,
            f"🚀 {base_post}",
            f"💡 {base_post}",
            f"🎉 {base_post}",
            f"🔥 {base_post}",
            f"✨ {base_post}",
            f"📈 {base_post}",
            f"🎯 {base_post}",
            f"💪 {base_post}",
            f"⚡ {base_post}"
        ]
        
        posts.append(random.choice(variations))
    
    return posts


def create_simple_tokenizer():
    """Create a simple tokenizer for demonstration."""
    class SimpleTokenizer:
        def __init__(self) -> Any:
            self.vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}
            self.vocab_size = 5
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
        
        def decode(self, token_ids: List[int]) -> str:
            """Simple decoding for demonstration."""
            # Create reverse vocabulary
            reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            # Convert token IDs back to words
            words = []
            for token_id in token_ids:
                if token_id in reverse_vocab:
                    word = reverse_vocab[token_id]
                    if word not in ["<PAD>", "<CLS>", "<SEP>"]:
                        words.append(word)
            
            return " ".join(words)
    
    return SimpleTokenizer()


class FacebookPostsDataset(torch.utils.data.Dataset):
    """Custom dataset for Facebook Posts."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        
    """__init__ function."""
self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        text = self.texts[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        
        # Create labels for language modeling (shifted by 1)
        labels = token_ids[1:] + [self.tokenizer.vocab["<PAD>"]]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if tid != self.tokenizer.vocab["<PAD>"] else 0 for tid in token_ids]),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


async def demo_transformer_architecture():
    """Demonstrate transformer architecture components."""
    logger.info("🏗️ Transformer Architecture Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_length=128,
        d_model=256,
        d_ff=1024,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_rope=True,
        use_relative_position=True
    )
    
    # Create transformer model
    model = create_transformer_model(config)
    model.to(DEVICE)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
    attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    logger.info(f"Transformer model created successfully!")
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with attention weights
    outputs_with_attention = model(input_ids, attention_mask, return_attention_weights=True)
    logger.info(f"Output with attention weights shape: {outputs_with_attention[0].shape}")
    
    logger.info("")


async def demo_llm_model():
    """Demonstrate LLM model functionality."""
    logger.info("🧠 LLM Model Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_length=128,
        d_model=256,
        d_ff=1024,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_rope=True,
        use_relative_position=True
    )
    
    # Create LLM model
    model = create_llm_model(config)
    model.to(DEVICE)
    
    # Test forward pass with labels
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
    attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels)
    
    logger.info(f"LLM model created successfully!")
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Output logits shape: {outputs['logits'].shape}")
    logger.info(f"Hidden states shape: {outputs['hidden_states'].shape}")
    logger.info(f"Loss: {outputs['loss'].item():.4f}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("")


async def demo_attention_visualization():
    """Demonstrate attention weight visualization."""
    logger.info("👁️ Attention Visualization Demo")
    logger.info("=" * 50)
    
    # Create a simple model for visualization
    config = TransformerConfig(
        vocab_size=50,
        max_seq_length=16,
        d_model=64,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_rope=False,
        use_relative_position=False
    )
    
    model = create_transformer_model(config)
    model.to(DEVICE)
    
    # Create sample input
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len)).to(DEVICE)
    attention_mask = torch.ones(1, seq_len).to(DEVICE)
    
    # Sample tokens for visualization
    sample_tokens = ["<CLS>", "amazing", "product", "launch", "today", "innovation", "tech", "<SEP>"]
    
    # Get attention weights (simplified)
    with torch.no_grad():
        # Create dummy attention weights for demonstration
        attention_weights = torch.randn(seq_len, seq_len)
        attention_weights = F.softmax(attention_weights, dim=-1)
    
    # Visualize attention
    try:
        AttentionVisualizer.visualize_attention(
            attention_weights,
            sample_tokens[:seq_len],
            save_path="attention_visualization.png"
        )
        logger.info("Attention visualization created successfully!")
    except Exception as e:
        logger.info(f"Attention visualization failed: {e}")
    
    logger.info("")


async def demo_text_generation():
    """Demonstrate text generation capabilities."""
    logger.info("📝 Text Generation Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_length=64,
        d_model=128,
        d_ff=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_rope=True,
        use_relative_position=True
    )
    
    # Create model
    model = create_llm_model(config)
    model.to(DEVICE)
    
    # Simple text generation function
    def generate_text(model, tokenizer, prompt: str, max_length: int = 20, temperature: float = 1.0):
        
    """generate_text function."""
model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(DEVICE)
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = model(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to generated tokens
                generated_tokens.append(next_token.item())
                
                # Update input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we reach max length or generate special token
                if len(generated_tokens) >= max_length or next_token.item() == tokenizer.vocab["<SEP>"]:
                    break
        
        # Decode generated text
        full_sequence = input_ids[0].tolist()
        generated_text = tokenizer.decode(full_sequence)
        
        return generated_text
    
    # Test text generation
    tokenizer = create_simple_tokenizer()
    prompts = [
        "amazing product",
        "great meeting",
        "customer feedback",
        "new features"
    ]
    
    logger.info("Text Generation Examples:")
    for prompt in prompts:
        try:
            generated_text = generate_text(model, tokenizer, prompt, max_length=10, temperature=0.8)
            logger.info(f"Prompt: '{prompt}'")
            logger.info(f"Generated: '{generated_text}'")
            logger.info("-" * 30)
        except Exception as e:
            logger.info(f"Generation failed for '{prompt}': {e}")
    
    logger.info("")


async def demo_model_compression():
    """Demonstrate model compression techniques."""
    logger.info("🗜️ Model Compression Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_length=64,
        d_model=128,
        d_ff=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Create model
    model = create_transformer_model(config)
    model.to(DEVICE)
    
    # Original model size
    original_params = sum(p.numel() for p in model.parameters())
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    logger.info(f"Original model parameters: {original_params:,}")
    logger.info(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    
    # Test quantization
    try:
        quantized_model = ModelCompressor.quantize_model(model, 'int8')
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        logger.info(f"Quantized model parameters: {quantized_params:,}")
        logger.info(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
        logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")
    except Exception as e:
        logger.info(f"Quantization failed: {e}")
    
    # Test pruning
    try:
        pruned_model = ModelCompressor.prune_model(model, pruning_ratio=0.3)
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        
        logger.info(f"Pruned model parameters: {pruned_params:,}")
        logger.info(f"Pruning reduction: {(original_params - pruned_params) / original_params * 100:.1f}%")
    except Exception as e:
        logger.info(f"Pruning failed: {e}")
    
    logger.info("")


async def demo_training_pipeline():
    """Demonstrate LLM training pipeline."""
    logger.info("🚀 LLM Training Pipeline Demo")
    logger.info("=" * 50)
    
    # Generate sample data
    posts = generate_facebook_posts_data(100)
    tokenizer = create_simple_tokenizer()
    
    # Create dataset
    dataset = FacebookPostsDataset(posts, tokenizer, max_length=64)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=64,
        d_model=128,
        d_ff=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        learning_rate=1e-3,
        warmup_steps=100,
        max_steps=500,
        batch_size=4,
        use_rope=True,
        use_relative_position=True,
        use_mixed_precision=False  # Disable for demo
    )
    
    # Create model and trainer
    model = create_llm_model(config)
    trainer = FacebookPostsLLMTrainer(model, config)
    
    logger.info(f"Training on {len(train_dataset)} samples")
    logger.info(f"Validating on {len(val_dataset)} samples")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model (limited steps for demo)
    start_time = time.time()
    training_results = trainer.train(train_loader, val_loader, max_steps=50)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final train loss: {training_results['train_losses'][-1]:.4f}")
    if training_results['val_losses']:
        logger.info(f"Final validation loss: {training_results['val_losses'][-1]:.4f}")
    
    # Save model
    trainer.save_model("llm_checkpoint.pth")
    
    logger.info("")


async def demo_model_comparison():
    """Compare different transformer configurations."""
    logger.info("📊 Model Comparison Demo")
    logger.info("=" * 50)
    
    configurations = [
        ("Small", TransformerConfig(
            vocab_size=1000,
            max_seq_length=64,
            d_model=64,
            d_ff=256,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )),
        ("Medium", TransformerConfig(
            vocab_size=1000,
            max_seq_length=64,
            d_model=128,
            d_ff=512,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )),
        ("Large", TransformerConfig(
            vocab_size=1000,
            max_seq_length=64,
            d_model=256,
            d_ff=1024,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        ))
    ]
    
    results = {}
    
    for name, config in configurations:
        logger.info(f"Testing {name} model...")
        
        # Create model
        model = create_transformer_model(config)
        model.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Test inference speed
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
        attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids, attention_mask)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids, attention_mask)
        inference_time = time.time() - start_time
        
        results[name] = {
            'parameters': total_params,
            'inference_time': inference_time / 10,  # Average time per inference
            'config': config
        }
        
        logger.info(f"{name} model: {total_params:,} parameters, {inference_time/10*1000:.2f}ms per inference")
    
    # Display comparison
    logger.info("\nModel Comparison Results:")
    logger.info("-" * 60)
    logger.info(f"{'Model':8} | {'Parameters':>12} | {'Inference Time':>15}")
    logger.info("-" * 60)
    
    for name, result in results.items():
        logger.info(f"{name:8} | {result['parameters']:12,} | {result['inference_time']*1000:15.2f}ms")
    
    logger.info("")


async def run_transformer_llm_demo():
    """Run the complete transformer and LLM demonstration."""
    logger.info("🧠 TRANSFORMER & LLM DEMO - Facebook Posts Processing")
    logger.info("=" * 60)
    logger.info("Demonstrating advanced transformer architectures and Large Language Models")
    logger.info("for Facebook Posts analysis, generation, and processing")
    logger.info("=" * 60)
    
    # Run all demos
    await demo_transformer_architecture()
    await demo_llm_model()
    await demo_attention_visualization()
    await demo_text_generation()
    await demo_model_compression()
    await demo_training_pipeline()
    await demo_model_comparison()
    
    logger.info("🎉 Transformer & LLM Demo Completed Successfully!")
    logger.info("All features demonstrated:")
    logger.info("✅ Transformer architecture with multi-head attention")
    logger.info("✅ LLM model with language modeling capabilities")
    logger.info("✅ Attention weight visualization")
    logger.info("✅ Text generation with temperature sampling")
    logger.info("✅ Model compression (quantization and pruning)")
    logger.info("✅ Complete training pipeline with validation")
    logger.info("✅ Model comparison and benchmarking")


async def quick_transformer_demo():
    """Quick demonstration of key transformer features."""
    logger.info("⚡ QUICK TRANSFORMER DEMO")
    logger.info("=" * 40)
    
    # Quick transformer creation
    config = TransformerConfig(
        vocab_size=100,
        max_seq_length=32,
        d_model=64,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_rope=True
    )
    
    model = create_transformer_model(config)
    model.to(DEVICE)
    
    # Quick inference test
    input_ids = torch.randint(0, config.vocab_size, (1, 16)).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    logger.info(f"Quick transformer test completed!")
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("\n✅ Quick demo completed!")


if __name__ == "__main__":
    # Run the complete demo
    asyncio.run(run_transformer_llm_demo())
    
    # Uncomment for quick demo
    # asyncio.run(quick_transformer_demo()) 