"""
Extreme Speed Example - All Optimizations Combined
"""

import torch
import torch.nn as nn
import time
from addition_removal_ai import (
    create_ultra_fast_engine,
    prune_model,
    quantize_model_advanced,
    compile_model,
    export_model_to_onnx,
    ONNXInference,
    create_embedding_cache,
    optimize_memory_usage
)


def main():
    """Demonstrate extreme speed optimizations"""
    
    print("=== Extreme Speed Optimizations ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(100, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    model.eval()
    example_input = torch.randn(1, 100).to(device)
    
    print("1. Baseline (FP32):")
    print("-" * 50)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.time() - start) / 100
    print(f"   Time: {baseline_time*1000:.2f}ms")
    
    print("\n2. Pruned Model (30%):")
    print("-" * 50)
    pruned = prune_model(model, pruning_ratio=0.3)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = pruned(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pruned_time = (time.time() - start) / 100
    print(f"   Time: {pruned_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/pruned_time:.2f}x")
    
    print("\n3. Quantized Model (INT8):")
    print("-" * 50)
    quantized = quantize_model_advanced(pruned, method="dynamic")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = quantized(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        quantized_time = (time.time() - start) / 100
    print(f"   Time: {quantized_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/quantized_time:.2f}x")
    
    print("\n4. Compiled Model:")
    print("-" * 50)
    compiled = compile_model(quantized, mode="reduce-overhead")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = compiled(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compiled_time = (time.time() - start) / 100
    print(f"   Time: {compiled_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/compiled_time:.2f}x")
    
    print("\n5. Memory Optimized:")
    print("-" * 50)
    optimize_memory_usage(compiled)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = compiled(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        optimized_time = (time.time() - start) / 100
    print(f"   Time: {optimized_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/optimized_time:.2f}x")
    
    print("\n6. ONNX Inference:")
    print("-" * 50)
    try:
        export_model_to_onnx(
            compiled,
            example_input.cpu(),
            "extreme_model.onnx",
            optimize=True
        )
        
        onnx_inference = ONNXInference("extreme_model.onnx", use_gpu=False)
        input_np = example_input.cpu().numpy()
        
        start = time.time()
        for _ in range(100):
            _ = onnx_inference(input_np)
        onnx_time = (time.time() - start) / 100
        print(f"   Time: {onnx_time*1000:.2f}ms")
        print(f"   Speedup: {baseline_time/onnx_time:.2f}x")
    except Exception as e:
        print(f"   ONNX not available: {e}")
        onnx_time = baseline_time
    
    print("\n7. Performance Summary:")
    print("-" * 50)
    print(f"Baseline:           {baseline_time*1000:.2f}ms (1.00x)")
    print(f"Pruned:             {pruned_time*1000:.2f}ms ({baseline_time/pruned_time:.2f}x)")
    print(f"Quantized:          {quantized_time*1000:.2f}ms ({baseline_time/quantized_time:.2f}x)")
    print(f"Compiled:           {compiled_time*1000:.2f}ms ({baseline_time/compiled_time:.2f}x)")
    print(f"Memory Optimized:   {optimized_time*1000:.2f}ms ({baseline_time/optimized_time:.2f}x)")
    print(f"ONNX:               {onnx_time*1000:.2f}ms ({baseline_time/onnx_time:.2f}x)")
    
    total_speedup = baseline_time / min(pruned_time, quantized_time, compiled_time, onnx_time)
    print(f"\nMaximum Speedup: {total_speedup:.2f}x")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()

