#!/usr/bin/env python3
"""
Test script for gradient clipping and NaN/Inf handling in the ultra-optimized SEO model.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import sys
import math

# Add the current directory to the path to import the SEO modules
sys.path.append(str(Path(__file__).parent))

from evaluation_metrics_ultra_optimized import (
    UltraOptimizedConfig, 
    UltraOptimizedSEOMetricsModule, 
    UltraOptimizedSEOTrainer,
    SEOTokenizer,
    SEODataset
)
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(num_samples: int = 100) -> tuple:
    """Create test data for training."""
    # Sample SEO texts
    sample_texts = [
        "This is a sample SEO text for testing purposes with keywords",
        "Another example of SEO content optimization and best practices",
        "Third sample text for evaluation metrics and analysis",
        "SEO optimization techniques for better search engine rankings",
        "Content marketing strategies for improved organic traffic"
    ] * (num_samples // 5)
    
    # Create labels (binary classification)
    labels = torch.randint(0, 2, (len(sample_texts),))
    
    return sample_texts, labels

def test_gradient_clipping():
    """Test gradient clipping functionality."""
    logger.info("Testing gradient clipping functionality...")
    
    # Create configuration with gradient clipping enabled
    config = UltraOptimizedConfig(
        use_multi_gpu=False,  # Disable for testing
        use_amp=True,
        use_lora=True,
        use_diffusion=False,  # Disable for simpler testing
        batch_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,  # Enable gradient clipping
        patience=3
    )
    
    # Initialize model and trainer
    model = UltraOptimizedSEOMetricsModule(config)
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Test gradient clipping method
    logger.info("Testing _clip_gradients method...")
    
    # Create dummy gradients with some extreme values
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Create some extreme gradients to test clipping
            param.grad = torch.randn_like(param) * 1000  # Large gradients
            break
    
    # Test gradient clipping
    grad_norm_before = trainer._get_gradient_norm()
    logger.info(f"Gradient norm before clipping: {grad_norm_before:.6f}")
    
    trainer._clip_gradients()
    
    grad_norm_after = trainer._get_gradient_norm()
    logger.info(f"Gradient norm after clipping: {grad_norm_after:.6f}")
    
    # Verify that gradients were clipped
    assert grad_norm_after <= config.max_grad_norm, "Gradients were not properly clipped!"
    logger.info("✓ Gradient clipping test passed!")

def test_nan_inf_handling():
    """Test NaN/Inf value handling."""
    logger.info("Testing NaN/Inf value handling...")
    
    config = UltraOptimizedConfig(
        use_multi_gpu=False,
        use_amp=True,
        use_lora=True,
        use_diffusion=False,
        batch_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        patience=3
    )
    
    model = UltraOptimizedSEOMetricsModule(config)
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Test NaN detection
    logger.info("Testing NaN detection...")
    normal_tensor = torch.randn(3, 3)
    nan_tensor = torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]])
    inf_tensor = torch.tensor([[1.0, float('inf'), 3.0], [4.0, 5.0, 6.0]])
    
    assert not trainer._check_nan_inf(normal_tensor, "normal tensor"), "Normal tensor incorrectly flagged as NaN/Inf"
    assert trainer._check_nan_inf(nan_tensor, "nan tensor"), "NaN tensor not detected"
    assert trainer._check_nan_inf(inf_tensor, "inf tensor"), "Inf tensor not detected"
    logger.info("✓ NaN/Inf detection test passed!")
    
    # Test NaN/Inf handling
    logger.info("Testing NaN/Inf handling...")
    cleaned_tensor = trainer._handle_nan_inf(nan_tensor, replacement_value=0.0)
    assert not torch.isnan(cleaned_tensor).any(), "NaN values not properly cleaned"
    assert not torch.isinf(cleaned_tensor).any(), "Inf values not properly cleaned"
    logger.info("✓ NaN/Inf handling test passed!")

def test_training_stability_monitoring():
    """Test training stability monitoring."""
    logger.info("Testing training stability monitoring...")
    
    config = UltraOptimizedConfig(
        use_multi_gpu=False,
        use_amp=True,
        use_lora=True,
        use_diffusion=False,
        batch_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        patience=3
    )
    
    model = UltraOptimizedSEOMetricsModule(config)
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Test training health monitoring
    logger.info("Testing training health monitoring...")
    health_status = trainer.monitor_training_health()
    logger.info(f"Training health status: {health_status}")
    
    # Test training statistics
    stats = trainer.get_training_stats()
    logger.info(f"Training statistics: {stats}")
    
    logger.info("✓ Training stability monitoring test passed!")

def test_checkpoint_safety():
    """Test checkpoint saving/loading with NaN/Inf safety."""
    logger.info("Testing checkpoint safety features...")
    
    config = UltraOptimizedConfig(
        use_multi_gpu=False,
        use_amp=True,
        use_lora=True,
        use_diffusion=False,
        batch_size=8,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        patience=3
    )
    
    model = UltraOptimizedSEOMetricsModule(config)
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Create some training history
    trainer.train_history = [{'epoch': 1, 'loss': 0.5, 'accuracy': 0.8, 'f1_score': 0.75}]
    trainer.val_history = [{'epoch': 1, 'loss': 0.6, 'accuracy': 0.75, 'f1_score': 0.7}]
    
    # Test checkpoint saving
    checkpoint_path = "./test_checkpoint.pth"
    trainer.save_checkpoint(checkpoint_path)
    
    # Test checkpoint loading
    new_trainer = UltraOptimizedSEOTrainer(model, config)
    success = new_trainer.load_checkpoint(checkpoint_path)
    
    assert success, "Checkpoint loading failed"
    assert len(new_trainer.train_history) == 1, "Training history not loaded correctly"
    assert len(new_trainer.val_history) == 1, "Validation history not loaded correctly"
    
    logger.info("✓ Checkpoint safety test passed!")
    
    # Clean up test file
    Path(checkpoint_path).unlink(missing_ok=True)

def test_full_training_cycle():
    """Test a full training cycle with gradient clipping and NaN/Inf handling."""
    logger.info("Testing full training cycle...")
    
    config = UltraOptimizedConfig(
        use_multi_gpu=False,
        use_amp=True,
        use_lora=True,
        use_diffusion=False,
        batch_size=4,  # Small batch size for testing
        learning_rate=1e-3,
        max_grad_norm=1.0,
        patience=2,
        num_epochs=2
    )
    
    # Create test data
    texts, labels = create_test_data(num_samples=20)
    
    # Initialize model and trainer
    model = UltraOptimizedSEOMetricsModule(config)
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Create data loaders
    train_size = int(0.7 * len(texts))
    val_size = int(0.15 * len(texts))
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    
    # Create datasets
    train_dataset = SEODataset(train_texts, train_labels, model.seo_tokenizer)
    val_dataset = SEODataset(val_texts, val_labels, model.seo_tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info("Starting training cycle...")
    
    try:
        # Train for a few epochs
        for epoch in range(2):
            logger.info(f"Training epoch {epoch + 1}/2")
            
            # Training
            trainer.model.train()
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(trainer.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Training step
                metrics = trainer.train_step(batch)
                logger.info(f"Training metrics: {metrics}")
                
                # Check for NaN/Inf in metrics
                if math.isnan(metrics['loss']) or math.isinf(metrics['loss']):
                    logger.warning("NaN/Inf detected in training loss")
                    break
            
            # Validation
            val_metrics = trainer.validate(val_loader)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Check training health
            health_status = trainer.monitor_training_health()
            logger.info(f"Training health: {health_status}")
            
            # Early stopping check
            if not health_status['overall_healthy']:
                logger.warning("Training health issues detected, stopping early")
                break
        
        logger.info("✓ Full training cycle test completed!")
        
    except Exception as e:
        logger.error(f"Training cycle test failed: {e}")
        raise

def main():
    """Run all tests."""
    logger.info("Starting comprehensive tests for gradient clipping and NaN/Inf handling...")
    
    try:
        # Test individual components
        test_gradient_clipping()
        test_nan_inf_handling()
        test_training_stability_monitoring()
        test_checkpoint_safety()
        
        # Test full training cycle
        test_full_training_cycle()
        
        logger.info("🎉 All tests passed successfully!")
        logger.info("The gradient clipping and NaN/Inf handling features are working correctly.")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise

if __name__ == "__main__":
    main()
