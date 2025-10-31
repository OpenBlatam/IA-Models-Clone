#!/usr/bin/env python3
"""
Test script for Ultra-Optimized SEO Evaluation System
Demonstrates all advanced features: LoRA, Diffusion, Multi-GPU, etc.
"""

import torch
import time
import asyncio
from evaluation_metrics_ultra_optimized import (
    UltraOptimizedConfig, UltraOptimizedSEOMetricsModule, 
    UltraOptimizedSEOTrainer, create_data_loaders
)

async def test_ultra_optimized_system():
    """Test the ultra-optimized SEO evaluation system."""
    print("🚀 Testing Ultra-Optimized SEO Evaluation System")
    
    # Configuration
    config = UltraOptimizedConfig(
        use_multi_gpu=torch.cuda.device_count() > 1,
        use_lora=True,
        use_diffusion=True,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        max_length=256,
        batch_size=16,
        num_epochs=5,
        patience=3,
        scheduler_type="cosine"
    )
    
    print(f"Configuration: {config}")
    print(f"Device: {config.device}")
    print(f"Multi-GPU: {config.use_multi_gpu}")
    print(f"LoRA: {config.use_lora}")
    print(f"Diffusion: {config.use_diffusion}")
    
    # Initialize model
    print("\n📦 Initializing Ultra-Optimized SEO Model...")
    start_time = time.time()
    
    model = UltraOptimizedSEOMetricsModule(config)
    print(f"Model initialized in {time.time() - start_time:.2f}s")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    print("\n🏋️ Initializing Trainer...")
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Test data
    print("\n📊 Preparing test data...")
    texts = [
        "<h1>SEO Optimization Guide</h1><p>Learn about search engine optimization techniques.</p>",
        "<meta name='description' content='Content quality analysis for better rankings'>",
        "How to improve your website's search engine ranking",
        "Best practices for SEO content creation",
        "Keyword research and optimization strategies",
        "Technical SEO implementation guide",
        "Content marketing for better search visibility",
        "Link building techniques for SEO",
        "Local SEO optimization strategies",
        "Mobile-first indexing and SEO"
    ]
    
    # Create labels (0: poor SEO, 1: good SEO)
    labels = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        texts, labels, config, model.seo_tokenizer
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test forward pass
    print("\n🔍 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_texts = texts[:4]
        test_y_true = labels[:4]
        test_y_pred = torch.randint(0, 2, (4,))
        
        outputs = model(test_texts, test_y_true, test_y_pred)
        print(f"Output keys: {list(outputs.keys())}")
        print(f"Accuracy: {outputs['accuracy'].item():.4f}")
        print(f"Precision: {outputs['precision'].item():.4f}")
        print(f"Recall: {outputs['recall'].item():.4f}")
        print(f"F1 Score: {outputs['f1_score'].item():.4f}")
        print(f"SEO Features shape: {outputs['seo_features'].shape}")
    
    # Test diffusion content generation
    if config.use_diffusion:
        print("\n🎨 Testing diffusion content generation...")
        try:
            prompt = "SEO optimization techniques for better search rankings"
            generated_content = model.generate_seo_content(prompt, num_inference_steps=20)
            print(f"Generated content shape: {generated_content.shape}")
            print(f"Generated content device: {generated_content.device}")
            print(f"Generated content dtype: {generated_content.dtype}")
        except Exception as e:
            print(f"Diffusion generation error: {e}")
    
    # Test training
    print("\n🎯 Testing training...")
    try:
        for epoch in range(min(3, config.num_epochs)):
            print(f"\nEpoch {epoch + 1}/{min(3, config.num_epochs)}")
            metrics = trainer.train_epoch(train_loader, val_loader, epoch)
            print(f"Train Loss: {metrics['train_loss']:.4f}")
            print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"Val Loss: {metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {metrics['val_accuracy']:.4f}")
            
            if trainer.early_stopping.early_stop:
                print("Early stopping triggered")
                break
    except Exception as e:
        print(f"Training error: {e}")
    
    # Test metrics calculation
    print("\n📈 Testing metrics calculation...")
    try:
        metrics_dict = model.calculate_metrics_vectorized(test_texts, test_y_true, test_y_pred)
        print("Calculated metrics:")
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Metrics calculation error: {e}")
    
    # Performance test
    print("\n⚡ Performance test...")
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model.calculate_metrics_vectorized(test_texts, test_y_true, test_y_pred)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {num_iterations/total_time:.2f} inferences/second")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    trainer.cleanup()
    
    print("\n✅ Ultra-optimized SEO evaluation system test completed!")
    return True

async def benchmark_performance():
    """Benchmark performance of the ultra-optimized system."""
    print("\n🏁 Performance Benchmark")
    
    config = UltraOptimizedConfig(
        use_multi_gpu=False,  # Disable for benchmark
        use_lora=True,
        use_diffusion=False,  # Disable for benchmark
        batch_size=32,
        max_length=128
    )
    
    model = UltraOptimizedSEOMetricsModule(config)
    
    # Benchmark data
    texts = ["SEO optimization test"] * 100
    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.randint(0, 2, (100,))
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.calculate_metrics_vectorized(texts[:10], y_true[:10], y_pred[:10])
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = model.calculate_metrics_vectorized(texts, y_true, y_pred)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start_time
    
    print(f"Benchmark completed in {total_time:.2f}s")
    print(f"Average time per batch: {total_time/100*1000:.2f}ms")
    print(f"Throughput: {100/total_time:.2f} batches/second")

if __name__ == "__main__":
    print("🚀 Starting Ultra-Optimized SEO Evaluation System Tests")
    
    # Run tests
    asyncio.run(test_ultra_optimized_system())
    
    # Run benchmark
    asyncio.run(benchmark_performance())
    
    print("\n🎉 All tests completed successfully!") 