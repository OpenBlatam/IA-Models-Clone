"""
Ultra Fast Training and Inference Example
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from addition_removal_ai import (
    create_fast_trainer,
    create_fast_dataloader,
    create_fast_inference_model,
    compile_model
)
import time


class SimpleDataset(Dataset):
    """Simple dataset for example"""
    
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 3, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    """Demonstrate ultra-fast training and inference"""
    
    print("=== Ultra Fast Training & Inference ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 3)
    ).to(device)
    
    # Create dataset and fast dataloader
    print("1. Creating Fast DataLoader...")
    dataset = SimpleDataset(1000)
    train_loader = create_fast_dataloader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    print(f"   DataLoader created with {train_loader.num_workers} workers")
    
    # Compile model for faster training
    print("\n2. Compiling Model...")
    if hasattr(torch, 'compile'):
        model = compile_model(model, mode="reduce-overhead")
        print("   Model compiled with torch.compile")
    else:
        print("   torch.compile not available")
    
    # Create fast trainer
    print("\n3. Creating Fast Trainer...")
    trainer = create_fast_trainer(
        model=model,
        train_loader=train_loader,
        use_torch_compile=True,
        use_mixed_precision=True,
        device=device
    )
    print("   Fast trainer created")
    
    # Training benchmark
    print("\n4. Training Benchmark...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    trainer.train_epoch(optimizer, criterion, epoch=1)
    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.3f}s")
    print(f"   Throughput: {len(dataset)/train_time:.1f} samples/sec")
    
    # Inference optimization
    print("\n5. Optimizing for Inference...")
    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Optimize model
    fast_model = create_fast_inference_model(
        model,
        example_input=example_input,
        use_compile=True,
        use_cache=False
    )
    print("   Model optimized for inference")
    
    # Inference benchmark
    print("\n6. Inference Benchmark...")
    num_runs = 100
    warmup = 10
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = fast_model(example_input)
    
    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = fast_model(example_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    inference_time = (time.time() - start_time) / num_runs
    fps = 1.0 / inference_time
    
    print(f"   Average inference time: {inference_time*1000:.2f}ms")
    print(f"   FPS: {fps:.1f}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()

