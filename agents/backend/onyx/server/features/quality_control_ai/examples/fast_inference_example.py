"""
Fast Inference Example
"""

import torch
import numpy as np
from quality_control_ai import (
    FastQualityInspector,
    CameraConfig,
    DetectionConfig,
    create_fast_autoencoder,
    PerformanceBenchmark
)


def main():
    """Example of fast inference"""
    print("Fast Quality Control AI Inference")
    
    # Create fast inspector
    inspector = FastQualityInspector(
        camera_config=CameraConfig(),
        detection_config=DetectionConfig(),
        use_fast_models=True,
        batch_size=8
    )
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Fast inspection
    print("\nFast Inspection:")
    result = inspector.inspect_frame_fast(dummy_image)
    print(f"Quality Score: {result['quality_score']}")
    print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
    print(f"FPS: {1000/result['inference_time_ms']:.2f}")
    
    # Benchmark
    print("\nBenchmarking Model:")
    model = create_fast_autoencoder(device='cuda' if torch.cuda.is_available() else 'cpu')
    benchmark_results = PerformanceBenchmark.benchmark_inference(
        model,
        input_shape=(1, 3, 224, 224),
        num_iterations=100
    )
    
    print(f"Average Time: {benchmark_results['avg_time_ms']:.2f}ms")
    print(f"FPS: {benchmark_results['fps']:.2f}")
    print(f"Throughput: {benchmark_results['throughput']:.2f} images/sec")
    
    # Batch processing
    print("\nBatch Processing:")
    batch_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    batch_results = inspector.inspect_batch_fast(batch_images)
    print(f"Processed {len(batch_results)} images")
    
    inspector.release()


if __name__ == "__main__":
    main()

