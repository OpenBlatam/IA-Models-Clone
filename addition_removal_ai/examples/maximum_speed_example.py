"""
Maximum Speed Example - All Optimizations Combined
"""

import torch
import torch.nn as nn
import time
from addition_removal_ai import (
    quantize_model_advanced,
    export_model_to_onnx,
    ONNXInference,
    create_async_engine,
    compile_model,
    create_fast_inference_model
)
import asyncio


def main():
    """Demonstrate maximum speed optimizations"""
    
    print("=== Maximum Speed Optimizations ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    model.eval()
    example_input = torch.randn(1, 100).to(device)
    
    print("1. PyTorch FP32 Baseline:")
    print("-" * 50)
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.time() - start) / 100
    print(f"   Time: {baseline_time*1000:.2f}ms")
    
    print("\n2. Compiled Model (torch.compile):")
    print("-" * 50)
    compiled_model = compile_model(model, mode="reduce-overhead")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = compiled_model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compiled_time = (time.time() - start) / 100
    print(f"   Time: {compiled_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/compiled_time:.2f}x")
    
    print("\n3. Quantized Model (INT8):")
    print("-" * 50)
    quantized = quantize_model_advanced(model, method="dynamic")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = quantized(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        quantized_time = (time.time() - start) / 100
    print(f"   Time: {quantized_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/quantized_time:.2f}x")
    
    print("\n4. ONNX Inference:")
    print("-" * 50)
    try:
        # Export to ONNX
        onnx_path = "model.onnx"
        export_model_to_onnx(
            model,
            example_input.cpu(),
            onnx_path,
            optimize=True
        )
        
        # ONNX inference
        onnx_inference = ONNXInference(onnx_path, use_gpu=False)
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
    
    print("\n5. Combined Optimizations:")
    print("-" * 50)
    # Quantized + Compiled
    quantized_compiled = compile_model(quantized, mode="reduce-overhead")
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = quantized_compiled(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        combined_time = (time.time() - start) / 100
    print(f"   Time: {combined_time*1000:.2f}ms")
    print(f"   Speedup: {baseline_time/combined_time:.2f}x")
    
    print("\n6. Performance Summary:")
    print("-" * 50)
    print(f"Baseline (FP32):     {baseline_time*1000:.2f}ms (1.00x)")
    print(f"Compiled:            {compiled_time*1000:.2f}ms ({baseline_time/compiled_time:.2f}x)")
    print(f"Quantized:           {quantized_time*1000:.2f}ms ({baseline_time/quantized_time:.2f}x)")
    print(f"ONNX:                {onnx_time*1000:.2f}ms ({baseline_time/onnx_time:.2f}x)")
    print(f"Combined:            {combined_time*1000:.2f}ms ({baseline_time/combined_time:.2f}x)")
    
    print("\n=== Example Complete ===")


async def async_example():
    """Demonstrate async inference"""
    print("\n=== Async Inference Example ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    model.eval()
    
    # Create async engine
    async_engine = create_async_engine(model, max_workers=4)
    
    # Prepare inputs
    inputs = [torch.randn(1, 100).to(device) for _ in range(10)]
    
    # Async batch inference
    start = time.time()
    results = await async_engine.infer_batch_async(inputs)
    async_time = time.time() - start
    
    print(f"Async batch inference (10 items): {async_time*1000:.2f}ms")
    print(f"Average per item: {async_time*10:.2f}ms")
    
    async_engine.shutdown()


if __name__ == "__main__":
    main()
    asyncio.run(async_example())

