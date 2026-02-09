#!/usr/bin/env python3
"""
Test script for Early Stopping and Learning Rate Scheduling in the Advanced LLM SEO Engine
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_llm_seo_engine import (
    SEOConfig, DataLoaderConfig, SEODataset, 
    DataLoaderManager, AdvancedLLMSEOEngine, EarlyStopping
)

def test_early_stopping():
    """Test EarlyStopping class functionality."""
    print("🧪 Testing EarlyStopping class...")
    
    # Test early stopping with patience=3, monitoring val_loss (min mode)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001, monitor="val_loss", mode="min")
    
    # Simulate training with improving then stagnating loss
    val_losses = [0.5, 0.4, 0.35, 0.35, 0.35, 0.35, 0.35]
    
    for i, loss in enumerate(val_losses):
        should_stop = early_stopping(loss)
        print(f"Epoch {i+1}: Val Loss: {loss:.3f}, Should Stop: {should_stop}")
        
        if should_stop:
            print(f"✅ Early stopping triggered at epoch {i+1}")
            break
    
    # Test reset functionality
    early_stopping.reset()
    print("✅ Early stopping reset successfully")
    
    # Test with accuracy monitoring (max mode)
    acc_early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitor="val_accuracy", mode="max")
    accuracies = [0.7, 0.75, 0.75, 0.75]
    
    for i, acc in enumerate(accuracies):
        should_stop = acc_early_stopping(acc)
        print(f"Epoch {i+1}: Val Accuracy: {acc:.3f}, Should Stop: {should_stop}")
        
        if should_stop:
            print(f"✅ Early stopping triggered at epoch {i+1}")
            break

def test_learning_rate_scheduling():
    """Test learning rate scheduling functionality."""
    print("\n🧪 Testing Learning Rate Scheduling...")
    
    try:
        config = SEOConfig(
            batch_size=8,
            learning_rate=1e-4,
            lr_scheduler="cosine",
            warmup_steps=10,
            use_mixed_precision=False,
            use_diffusion=False
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Test different scheduler types
        schedulers_to_test = ["cosine", "linear", "exponential", "step", "plateau"]
        
        for scheduler_type in schedulers_to_test:
            print(f"\nTesting {scheduler_type} scheduler...")
            config.lr_scheduler = scheduler_type
            engine.config = config
            
            # Reinitialize scheduler
            engine._initialize_optimizer()
            engine._initialize_scheduler()
            
            lr_info = engine.get_learning_rate_info()
            print(f"  Scheduler: {lr_info['scheduler_type']}")
            print(f"  Current LR: {lr_info['current_lr']:.2e}")
            print(f"  Scheduler State: {lr_info['scheduler_state']}")
        
        print("✅ Learning rate scheduling test passed!")
        
    except Exception as e:
        print(f"❌ Learning rate scheduling test failed: {e}")
        import traceback
        traceback.print_exc()

def test_training_with_early_stopping():
    """Test training with early stopping."""
    print("\n🧪 Testing Training with Early Stopping...")
    
    try:
        config = SEOConfig(
            batch_size=4,
            learning_rate=1e-3,
            lr_scheduler="cosine",
            warmup_steps=5,
            early_stopping_patience=2,
            early_stopping_min_delta=0.001,
            early_stopping_monitor="val_loss",
            early_stopping_mode="min",
            use_mixed_precision=False,
            use_diffusion=False
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Create simple dataset
        texts = ["Sample text " + str(i) for i in range(20)]
        labels = [i % 2 for i in range(20)]
        
        # Create dataloaders
        train_loader, val_loader = engine.create_training_dataloaders(texts, labels, "test_training", 0.3)
        print(f"✅ DataLoaders created: train={len(train_loader)}, val={len(val_loader)}")
        
        # Test training with early stopping
        results = engine.train_with_early_stopping(train_loader, val_loader, max_epochs=10)
        
        print("✅ Training with early stopping completed!")
        print(f"  Epochs completed: {results['epochs_completed']}")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        print(f"  Best epoch: {results['best_epoch']}")
        print(f"  Early stopping triggered: {results['early_stopping_triggered']}")
        
    except Exception as e:
        print(f"❌ Training with early stopping test failed: {e}")
        import traceback
        traceback.print_exc()

def test_checkpoint_functionality():
    """Test checkpoint saving and loading."""
    print("\n🧪 Testing Checkpoint Functionality...")
    
    try:
        config = SEOConfig(
            batch_size=4,
            learning_rate=1e-3,
            save_checkpoints=True,
            checkpoint_dir="./test_checkpoints",
            use_mixed_precision=False,
            use_diffusion=False
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Create simple dataset and train for one epoch
        texts = ["Checkpoint test text " + str(i) for i in range(10)]
        labels = [i % 2 for i in range(10)]
        
        train_loader, val_loader = engine.create_training_dataloaders(texts, labels, "checkpoint_test", 0.3)
        
        # Train one epoch
        engine.train_epoch(train_loader, val_loader)
        
        # Save checkpoint
        engine.save_checkpoint("test_checkpoint.pt")
        print("✅ Checkpoint saved successfully")
        
        # Load checkpoint
        engine.load_checkpoint("./test_checkpoints/test_checkpoint.pt")
        print("✅ Checkpoint loaded successfully")
        
        # Clean up
        import shutil
        if os.path.exists("./test_checkpoints"):
            shutil.rmtree("./test_checkpoints")
        print("✅ Test checkpoints cleaned up")
        
    except Exception as e:
        print(f"❌ Checkpoint functionality test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🚀 Starting Early Stopping and Learning Rate Scheduling tests...\n")
    
    try:
        test_early_stopping()
        test_learning_rate_scheduling()
        test_training_with_early_stopping()
        test_checkpoint_functionality()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 