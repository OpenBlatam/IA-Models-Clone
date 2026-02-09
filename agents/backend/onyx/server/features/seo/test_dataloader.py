#!/usr/bin/env python3
"""
Test script for DataLoader implementation in the Advanced LLM SEO Engine
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to the path to import the engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_llm_seo_engine import (
    SEOConfig, DataLoaderConfig, SEODataset, 
    DataLoaderManager, AdvancedLLMSEOEngine
)

def test_dataloader_basic():
    """Test basic DataLoader functionality."""
    print("🧪 Testing basic DataLoader functionality...")
    
    # Create configuration
    config = DataLoaderConfig(
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        pin_memory=False,  # Disable for testing
        persistent_workers=False
    )
    
    # Create manager
    manager = DataLoaderManager(config)
    
    # Create sample data
    texts = [
        "This is a sample SEO text for testing.",
        "Another sample text with different content.",
        "Third sample text to test batching.",
        "Fourth sample text for validation.",
        "Fifth sample text for additional testing.",
        "Sixth sample text to ensure proper splitting."
    ]
    
    labels = [1, 0, 1, 0, 1, 0]
    
    # Create dataset
    dataset = manager.create_dataset("test", texts, labels)
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = manager.create_dataloader("test", dataset)
    print(f"✅ DataLoader created with batch size {dataloader.batch_size}")
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch['input_ids'].shape}, labels: {batch['labels']}")
    
    print("✅ Basic DataLoader test passed!")

def test_dataloader_split():
    """Test train/validation split functionality."""
    print("\n🧪 Testing train/validation split...")
    
    config = DataLoaderConfig(batch_size=2, num_workers=0)
    manager = DataLoaderManager(config)
    
    texts = ["Text " + str(i) for i in range(10)]
    labels = [i % 2 for i in range(10)]
    
    dataset = manager.create_dataset("split_test", texts, labels)
    train_loader, val_loader = manager.create_train_val_split("split_test", dataset, val_split=0.3)
    
    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Validation loader: {len(val_loader)} batches")
    
    # Test train loader
    for i, batch in enumerate(train_loader):
        print(f"Train batch {i}: {batch['input_ids'].shape}")
    
    # Test val loader
    for i, batch in enumerate(val_loader):
        print(f"Val batch {i}: {batch['input_ids'].shape}")
    
    print("✅ Train/validation split test passed!")

def test_dataloader_benchmark():
    """Test DataLoader benchmarking functionality."""
    print("\n🧪 Testing DataLoader benchmarking...")
    
    config = DataLoaderConfig(batch_size=4, num_workers=0)
    manager = DataLoaderManager(config)
    
    texts = ["Benchmark text " + str(i) for i in range(20)]
    labels = [i % 2 for i in range(20)]
    
    dataset = manager.create_dataset("benchmark_test", texts, labels)
    dataloader = manager.create_dataloader("benchmark_test", dataset)
    
    # Run benchmark
    benchmark_results = manager.benchmark_dataloader(dataloader, num_batches=3)
    
    print("✅ Benchmark results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")
    
    print("✅ Benchmark test passed!")

def test_engine_integration():
    """Test DataLoader integration with the SEO engine."""
    print("\n🧪 Testing engine integration...")
    
    try:
        # Create engine config
        config = SEOConfig(
            batch_size=8,
            dataloader_num_workers=0,  # Disable for testing
            use_mixed_precision=False,  # Disable for testing
            use_diffusion=False  # Disable for testing
        )
        
        # Create engine
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Test data loading methods
        texts = ["Engine test text " + str(i) for i in range(10)]
        labels = [i % 2 for i in range(10)]
        
        # Create dataset
        dataset = engine.create_training_dataset(texts, labels, "engine_test")
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Create dataloaders
        train_loader, val_loader = engine.create_training_dataloaders(
            texts, labels, "engine_test", 0.2
        )
        print(f"✅ DataLoaders created: train={len(train_loader)}, val={len(val_loader)}")
        
        # Test stats
        stats = engine.get_data_loading_stats()
        print(f"✅ Stats retrieved: {len(stats)} items")
        
        print("✅ Engine integration test passed!")
        
    except Exception as e:
        print(f"❌ Engine integration test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("🚀 Starting DataLoader tests...\n")
    
    try:
        test_dataloader_basic()
        test_dataloader_split()
        test_dataloader_benchmark()
        test_engine_integration()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())






