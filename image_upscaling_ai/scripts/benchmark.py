"""
Benchmark Script
================

Benchmark different upscaling methods.
"""

import asyncio
import time
import statistics
from pathlib import Path
from PIL import Image
import numpy as np


async def benchmark_method(method_name, upscale_func, image, scale_factor, iterations=5):
    """Benchmark a single method."""
    times = []
    qualities = []
    
    for i in range(iterations):
        start = time.time()
        result = await upscale_func(image, scale_factor)
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Calculate quality (simplified)
        if hasattr(result, 'size'):
            # Simple quality metric
            quality = 0.8  # Placeholder
            qualities.append(quality)
    
    return {
        "method": method_name,
        "avg_time": statistics.mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
        "avg_quality": statistics.mean(qualities) if qualities else 0.0,
        "iterations": iterations
    }


async def run_benchmark():
    """Run comprehensive benchmark."""
    print("="*60)
    print("Upscaling Methods Benchmark")
    print("="*60)
    
    # Create test image
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    scale_factor = 4.0
    
    print(f"\nTest Image: {test_image.size}")
    print(f"Scale Factor: {scale_factor}x")
    print(f"Target Size: {tuple(int(s * scale_factor) for s in test_image.size)}")
    print("\n" + "-"*60)
    
    results = []
    
    # Benchmark Lanczos
    try:
        from image_upscaling_ai.models.advanced_upscaling import AdvancedUpscaling
        
        async def lanczos_upscale(img, scale):
            return AdvancedUpscaling.upscale_lanczos(img, scale)
        
        result = await benchmark_method("Lanczos", lanczos_upscale, test_image, scale_factor)
        results.append(result)
        print(f"✅ {result['method']}: {result['avg_time']:.3f}s")
    except Exception as e:
        print(f"❌ Lanczos failed: {e}")
    
    # Benchmark OpenCV
    try:
        async def opencv_upscale(img, scale):
            return AdvancedUpscaling.upscale_opencv_edsr(img, scale)
        
        result = await benchmark_method("OpenCV EDSR", opencv_upscale, test_image, scale_factor)
        results.append(result)
        print(f"✅ {result['method']}: {result['avg_time']:.3f}s")
    except Exception as e:
        print(f"❌ OpenCV failed: {e}")
    
    # Benchmark Real-ESRGAN
    try:
        from image_upscaling_ai.models import RealESRGANModelManager
        
        manager = RealESRGANModelManager()
        
        async def realesrgan_upscale(img, scale):
            return await manager.upscale_async(img, scale)
        
        result = await benchmark_method("Real-ESRGAN", realesrgan_upscale, test_image, scale_factor, iterations=3)
        results.append(result)
        print(f"✅ {result['method']}: {result['avg_time']:.3f}s")
    except Exception as e:
        print(f"⚠️  Real-ESRGAN not available: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    if results:
        # Sort by time
        results.sort(key=lambda x: x['avg_time'])
        
        print(f"\n{'Method':<20} {'Avg Time':<12} {'Min':<10} {'Max':<10} {'Quality':<10}")
        print("-"*60)
        
        for r in results:
            print(f"{r['method']:<20} {r['avg_time']:<12.3f} {r['min_time']:<10.3f} "
                  f"{r['max_time']:<10.3f} {r['avg_quality']:<10.2f}")
        
        fastest = results[0]
        print(f"\n🏆 Fastest: {fastest['method']} ({fastest['avg_time']:.3f}s)")
    else:
        print("No results available")


if __name__ == "__main__":
    asyncio.run(run_benchmark())


