"""
Code Profiling Examples for Video-OpusClip

Comprehensive examples demonstrating code profiling implementation
to identify and optimize bottlenecks in the Video-OpusClip system.
"""

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import structlog
from pathlib import Path
import json
import cv2
import random
from typing import Dict, List, Tuple, Optional, Any

# Import our profiling modules
from code_profiler import (
    VideoOpusClipProfiler,
    ProfilerConfig,
    create_profiler_config,
    profile_function,
    profile_class
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC FUNCTION PROFILING
# =============================================================================

def example_basic_function_profiling():
    """Basic function profiling example."""
    print("=== Example 1: Basic Function Profiling ===")
    
    # Create profiler
    config = create_profiler_config("basic")
    profiler = VideoOpusClipProfiler(config)
    
    # Define functions to profile
    @profiler.profile_function
    def fast_function():
        """Fast function that should be optimized."""
        return sum(range(1000))
    
    @profiler.profile_function
    def slow_function():
        """Slow function that needs optimization."""
        time.sleep(0.1)  # Simulate slow operation
        return sum(range(10000))
    
    @profiler.profile_function
    def memory_intensive_function():
        """Memory intensive function."""
        large_array = np.random.randn(1000, 1000)
        result = np.linalg.eig(large_array)
        return result
    
    # Start profiling
    profiler.start_profiling()
    
    # Run functions multiple times
    for i in range(10):
        fast_function()
        slow_function()
        if i % 3 == 0:  # Run memory intensive function less frequently
            memory_intensive_function()
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Function Profiling Results:")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    print(f"Slow functions: {len(bottlenecks['slow_functions'])}")
    print(f"Memory intensive functions: {len(bottlenecks['memory_intensive_functions'])}")
    
    # Print detailed results
    for func_name, profile_data in report['performance']['profiles'].items():
        print(f"\n{func_name}:")
        print(f"  Execution time: {profile_data['execution_time']:.4f}s")
        print(f"  Memory delta: {profile_data['memory_delta'] / 1024 / 1024:.1f}MB")
    
    return profiler

# =============================================================================
# EXAMPLE 2: CLASS PROFILING
# =============================================================================

def example_class_profiling():
    """Class profiling example."""
    print("\n=== Example 2: Class Profiling ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Define a class to profile
    @profiler.profile_class
    class VideoProcessor:
        def __init__(self):
            self.model = nn.Sequential(
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 100)
            )
            self.cache = {}
        
        def process_video_frame(self, frame):
            """Process a single video frame."""
            # Simulate frame processing
            time.sleep(0.01)
            
            # Convert frame to tensor
            tensor = torch.from_numpy(frame).float()
            
            # Process through model
            with torch.no_grad():
                result = self.model(tensor)
            
            return result
        
        def process_video_batch(self, frames):
            """Process a batch of video frames."""
            results = []
            for frame in frames:
                result = self.process_video_frame(frame)
                results.append(result)
            return torch.stack(results)
        
        def cache_result(self, key, value):
            """Cache a result."""
            self.cache[key] = value
        
        def get_cached_result(self, key):
            """Get a cached result."""
            return self.cache.get(key)
    
    # Start profiling
    profiler.start_profiling()
    
    # Create processor and test methods
    processor = VideoProcessor()
    
    # Test individual frame processing
    for i in range(5):
        frame = np.random.randn(1000)
        result = processor.process_video_frame(frame)
        processor.cache_result(f"frame_{i}", result)
    
    # Test batch processing
    frames = [np.random.randn(1000) for _ in range(10)]
    batch_result = processor.process_video_batch(frames)
    
    # Test cache operations
    for i in range(5):
        cached = processor.get_cached_result(f"frame_{i}")
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Class Profiling Results:")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    print(f"Slow functions: {len(bottlenecks['slow_functions'])}")
    
    # Print method performance
    for func_name, profile_data in report['performance']['profiles'].items():
        if 'VideoProcessor' in func_name:
            print(f"\n{func_name}:")
            print(f"  Execution time: {profile_data['execution_time']:.4f}s")
            print(f"  Memory delta: {profile_data['memory_delta'] / 1024 / 1024:.1f}MB")
    
    return profiler

# =============================================================================
# EXAMPLE 3: DATA LOADING PROFILING
# =============================================================================

def example_data_loading_profiling():
    """Data loading profiling example."""
    print("\n=== Example 3: Data Loading Profiling ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Create synthetic dataset
    class SyntheticVideoDataset:
        def __init__(self, num_samples=1000, frame_size=(224, 224)):
            self.num_samples = num_samples
            self.frame_size = frame_size
            self.data = []
            
            # Generate synthetic data
            for i in range(num_samples):
                # Simulate video frame loading
                frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
                label = random.randint(0, 9)
                self.data.append((frame, label))
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, index):
            # Simulate slow loading for some indices
            if index % 10 == 0:
                time.sleep(0.1)  # Simulate slow loading
            
            frame, label = self.data[index]
            
            # Simulate preprocessing
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
            
            return torch.from_numpy(frame), label
    
    # Create dataset and profile it
    dataset = SyntheticVideoDataset(num_samples=100)
    profiled_dataset = profiler.profile_dataset(dataset)
    
    # Create data loader and profile it
    train_loader = DataLoader(
        profiled_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Single worker for profiling
    )
    profiled_loader = profiler.profile_data_loader(train_loader)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test data loading
    print("Testing data loading...")
    for batch_idx, (frames, labels) in enumerate(profiled_loader):
        if batch_idx >= 5:  # Test first 5 batches
            break
        
        # Simulate processing
        time.sleep(0.01)
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    dataset_stats = profiled_dataset.get_stats()
    loader_stats = profiled_loader.get_stats()
    data_report = profiler.data_loading_profiler.get_data_loading_report()
    
    print("Data Loading Profiling Results:")
    print(f"\nDataset Statistics:")
    print(f"  Total accesses: {dataset_stats['total_accesses']}")
    print(f"  Average load time: {dataset_stats['avg_load_time']:.4f}s")
    print(f"  Slow loads: {dataset_stats['slow_loads']}")
    print(f"  Average memory delta: {dataset_stats['avg_memory_delta'] / 1024 / 1024:.1f}MB")
    
    print(f"\nDataLoader Statistics:")
    print(f"  Average batch time: {loader_stats['avg_batch_time']:.4f}s")
    print(f"  Max batch time: {loader_stats['max_batch_time']:.4f}s")
    print(f"  Total batches: {loader_stats['total_batches']}")
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    print(f"\nData Loading Bottlenecks:")
    for item in bottlenecks['slow_data_loading']:
        print(f"  - {item}")
    
    return profiler

# =============================================================================
# EXAMPLE 4: MEMORY PROFILING
# =============================================================================

def example_memory_profiling():
    """Memory profiling example."""
    print("\n=== Example 4: Memory Profiling ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test memory usage patterns
    with profiler.memory_profiler.memory_context("data_creation"):
        # Create large dataset
        large_data = np.random.randn(1000, 1000)
        print(f"Created large data: {large_data.nbytes / 1024 / 1024:.1f}MB")
    
    with profiler.memory_profiler.memory_context("data_processing"):
        # Process data
        processed_data = large_data * 2 + 1
        print(f"Processed data: {processed_data.nbytes / 1024 / 1024:.1f}MB")
    
    with profiler.memory_profiler.memory_context("model_creation"):
        # Create model
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    with profiler.memory_profiler.memory_context("model_inference"):
        # Run model inference
        with torch.no_grad():
            input_tensor = torch.randn(100, 1000)
            output = model(input_tensor)
        print(f"Model inference completed")
    
    # Test memory intensive function
    @profiler.memory_profiler.profile_memory_usage
    def memory_intensive_operation():
        # Create multiple large arrays
        arrays = []
        for i in range(5):
            array = np.random.randn(500, 500)
            arrays.append(array)
        
        # Process arrays
        result = sum(np.sum(arr) for arr in arrays)
        
        # Clean up
        del arrays
        
        return result
    
    result = memory_intensive_operation()
    print(f"Memory intensive operation result: {result}")
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get memory report
    memory_report = profiler.memory_profiler.get_memory_report()
    
    print("Memory Profiling Results:")
    print(f"Total traces: {memory_report['total_traces']}")
    
    stats = memory_report['memory_statistics']
    print(f"\nMemory Statistics:")
    print(f"  Total memory delta: {stats['total_memory_delta'] / 1024 / 1024:.1f}MB")
    print(f"  Average memory delta: {stats['avg_memory_delta'] / 1024 / 1024:.1f}MB")
    print(f"  Max memory delta: {stats['max_memory_delta'] / 1024 / 1024:.1f}MB")
    print(f"  Min memory delta: {stats['min_memory_delta'] / 1024 / 1024:.1f}MB")
    
    gc_stats = memory_report['gc_statistics']
    print(f"\nGarbage Collection Statistics:")
    print(f"  Total collections: {gc_stats['total_collections']}")
    print(f"  Average collections: {gc_stats['avg_collections']:.1f}")
    
    print(f"\nMemory Intensive Contexts:")
    for context in memory_report['memory_intensive_contexts']:
        print(f"  - {context['context']}: {context['memory_delta'] / 1024 / 1024:.1f}MB")
    
    return profiler

# =============================================================================
# EXAMPLE 5: GPU PROFILING
# =============================================================================

def example_gpu_profiling():
    """GPU profiling example."""
    print("\n=== Example 5: GPU Profiling ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU profiling example")
        return None
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Start profiling
    profiler.start_profiling()
    
    # Test GPU operations
    with profiler.gpu_profiler.cuda_context("model_creation"):
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        ).cuda()
        print(f"Created GPU model with {sum(p.numel() for p in model.parameters())} parameters")
    
    with profiler.gpu_profiler.cuda_context("data_transfer"):
        # Transfer data to GPU
        input_data = torch.randn(100, 1000)
        gpu_data = input_data.cuda()
        print(f"Transferred {input_data.numel()} elements to GPU")
    
    with profiler.gpu_profiler.cuda_context("model_inference"):
        # Run model inference
        with torch.no_grad():
            output = model(gpu_data)
        print(f"Model inference completed")
    
    # Test GPU intensive function
    @profiler.gpu_profiler.profile_cuda_operation
    def gpu_intensive_operation():
        # Create large tensors on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform matrix multiplication
        result = torch.mm(x, y)
        
        # Additional operations
        result = torch.relu(result)
        result = torch.sum(result)
        
        return result.cpu().item()
    
    result = gpu_intensive_operation()
    print(f"GPU intensive operation result: {result}")
    
    # Test multiple GPU operations
    with profiler.gpu_profiler.cuda_context("batch_processing"):
        batch_size = 32
        inputs = torch.randn(batch_size, 1000).cuda()
        
        for i in range(10):
            with torch.no_grad():
                outputs = model(inputs)
        
        print(f"Processed {batch_size * 10} samples")
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get GPU report
    gpu_report = profiler.gpu_profiler.get_gpu_report()
    
    print("GPU Profiling Results:")
    print(f"Total events: {gpu_report['total_events']}")
    
    stats = gpu_report['gpu_statistics']
    print(f"\nGPU Statistics:")
    print(f"  Total time: {stats['total_time_ms']:.2f}ms")
    print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
    print(f"  Max time: {stats['max_time_ms']:.2f}ms")
    print(f"  Min time: {stats['min_time_ms']:.2f}ms")
    
    memory_stats = gpu_report['memory_statistics']
    print(f"\nGPU Memory Statistics:")
    print(f"  Total memory delta: {memory_stats['total_memory_delta'] / 1024 / 1024:.1f}MB")
    print(f"  Average memory delta: {memory_stats['avg_memory_delta'] / 1024 / 1024:.1f}MB")
    print(f"  Max memory delta: {memory_stats['max_memory_delta'] / 1024 / 1024:.1f}MB")
    
    print(f"\nSlow GPU Operations:")
    for op in gpu_report['slow_gpu_operations']:
        print(f"  - {op['context']}: {op['elapsed_time_ms']:.2f}ms")
    
    print(f"\nMemory Intensive GPU Operations:")
    for op in gpu_report['memory_intensive_operations']:
        print(f"  - {op['context']}: {op['memory_delta'] / 1024 / 1024:.1f}MB")
    
    return profiler

# =============================================================================
# EXAMPLE 6: PREPROCESSING PROFILING
# =============================================================================

def example_preprocessing_profiling():
    """Preprocessing profiling example."""
    print("\n=== Example 6: Preprocessing Profiling ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Define preprocessing functions
    @profiler.profile_preprocessing
    def resize_image(image):
        """Resize image to target size."""
        return cv2.resize(image, (224, 224))
    
    @profiler.profile_preprocessing
    def normalize_image(image):
        """Normalize image values."""
        return image.astype(np.float32) / 255.0
    
    @profiler.profile_preprocessing
    def apply_augmentation(image):
        """Apply data augmentation."""
        # Random rotation
        angle = random.uniform(-15, 15)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random flip
        if random.random() > 0.5:
            rotated = cv2.flip(rotated, 1)
        
        return rotated
    
    @profiler.profile_preprocessing
    def convert_to_tensor(image):
        """Convert image to tensor."""
        # Convert to CHW format
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
    
    # Start profiling
    profiler.start_profiling()
    
    # Test preprocessing pipeline
    print("Testing preprocessing pipeline...")
    
    for i in range(20):
        # Create synthetic image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Apply preprocessing
        resized = resize_image(image)
        normalized = normalize_image(resized)
        augmented = apply_augmentation(normalized)
        tensor = convert_to_tensor(augmented)
        
        # Simulate some slow operations
        if i % 5 == 0:
            time.sleep(0.05)  # Simulate occasional slow processing
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get preprocessing report
    data_report = profiler.data_loading_profiler.get_data_loading_report()
    
    print("Preprocessing Profiling Results:")
    
    if 'preprocessing' in data_report:
        pre_stats = data_report['preprocessing']
        print(f"\nPreprocessing Statistics:")
        print(f"  Total operations: {pre_stats['total_operations']}")
        print(f"  Average time: {pre_stats['avg_preprocess_time']:.4f}s")
        print(f"  Max time: {pre_stats['max_preprocess_time']:.4f}s")
        print(f"  Slow operations: {pre_stats['slow_operations']}")
        
        print(f"\nFunction Breakdown:")
        for func_name, func_stats in pre_stats['function_breakdown'].items():
            print(f"  {func_name}:")
            print(f"    Count: {func_stats['count']}")
            print(f"    Average time: {func_stats['avg_time']:.4f}s")
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    print(f"\nPreprocessing Bottlenecks:")
    for item in bottlenecks['slow_preprocessing']:
        print(f"  - {item}")
    
    return profiler

# =============================================================================
# EXAMPLE 7: COMPREHENSIVE PROFILING
# =============================================================================

def example_comprehensive_profiling():
    """Comprehensive profiling example."""
    print("\n=== Example 7: Comprehensive Profiling ===")
    
    # Create comprehensive profiler
    config = create_profiler_config("comprehensive")
    profiler = VideoOpusClipProfiler(config)
    
    # Create synthetic video processing pipeline
    class VideoProcessingPipeline:
        def __init__(self):
            self.model = nn.Sequential(
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 100)
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.cache = {}
        
        @profiler.profile_function
        def load_video_frame(self, frame_id):
            """Load a video frame."""
            # Simulate frame loading
            time.sleep(0.01)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return frame
        
        @profiler.profile_preprocessing
        def preprocess_frame(self, frame):
            """Preprocess a video frame."""
            # Resize
            resized = cv2.resize(frame, (224, 224))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert to tensor
            tensor = torch.from_numpy(normalized).float()
            if len(tensor.shape) == 3:
                tensor = tensor.permute(2, 0, 1)  # HWC to CHW
            
            return tensor
        
        @profiler.profile_function
        def process_frame(self, frame_tensor):
            """Process frame through model."""
            if torch.cuda.is_available():
                frame_tensor = frame_tensor.cuda()
            
            with torch.no_grad():
                result = self.model(frame_tensor)
            
            return result.cpu()
        
        @profiler.profile_function
        def cache_result(self, frame_id, result):
            """Cache processing result."""
            self.cache[frame_id] = result
        
        @profiler.profile_function
        def get_cached_result(self, frame_id):
            """Get cached result."""
            return self.cache.get(frame_id)
    
    # Create pipeline
    pipeline = VideoProcessingPipeline()
    
    # Start comprehensive profiling
    profiler.start_profiling()
    
    # Test complete pipeline
    print("Testing complete video processing pipeline...")
    
    with profiler.profiling_context("video_processing_session"):
        for frame_id in range(10):
            with profiler.profiling_context(f"frame_{frame_id}"):
                # Load frame
                frame = pipeline.load_video_frame(frame_id)
                
                # Check cache
                cached_result = pipeline.get_cached_result(frame_id)
                
                if cached_result is None:
                    # Preprocess frame
                    frame_tensor = pipeline.preprocess_frame(frame)
                    
                    # Process frame
                    result = pipeline.process_frame(frame_tensor)
                    
                    # Cache result
                    pipeline.cache_result(frame_id, result)
                else:
                    result = cached_result
                
                # Simulate additional processing
                time.sleep(0.005)
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get comprehensive report
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Comprehensive Profiling Results:")
    print(f"Session duration: {report['session_info']['duration']:.2f}s")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    
    print(f"\nPerformance Summary:")
    print(f"  Slow functions: {len(bottlenecks['slow_functions'])}")
    print(f"  Memory intensive functions: {len(bottlenecks['memory_intensive_functions'])}")
    print(f"  Slow data loading: {len(bottlenecks['slow_data_loading'])}")
    print(f"  Slow preprocessing: {len(bottlenecks['slow_preprocessing'])}")
    print(f"  Slow GPU operations: {len(bottlenecks['slow_gpu_operations'])}")
    
    print(f"\nTop Slow Functions:")
    for func in bottlenecks['slow_functions'][:5]:
        print(f"  - {func}")
    
    print(f"\nTop Memory Intensive Functions:")
    for func in bottlenecks['memory_intensive_functions'][:5]:
        print(f"  - {func}")
    
    print(f"\nRecommendations:")
    for recommendation in bottlenecks['recommendations']:
        print(f"  - {recommendation}")
    
    # Save comprehensive report
    report_path = profiler.save_comprehensive_report("comprehensive_profile.json")
    print(f"\nComprehensive report saved to: {report_path}")
    
    return profiler

# =============================================================================
# EXAMPLE 8: BOTTLENECK OPTIMIZATION
# =============================================================================

def example_bottleneck_optimization():
    """Bottleneck optimization example."""
    print("\n=== Example 8: Bottleneck Optimization ===")
    
    # Create profiler
    config = create_profiler_config("detailed")
    profiler = VideoOpusClipProfiler(config)
    
    # Original slow implementation
    @profiler.profile_function
    def slow_data_processing(data_list):
        """Slow data processing implementation."""
        results = []
        for item in data_list:
            # Simulate slow processing
            time.sleep(0.01)
            result = item * 2 + 1
            results.append(result)
        return results
    
    # Optimized implementation
    @profiler.profile_function
    def fast_data_processing(data_list):
        """Fast data processing implementation."""
        # Vectorized operation
        data_array = np.array(data_list)
        results = data_array * 2 + 1
        return results.tolist()
    
    # Memory inefficient implementation
    @profiler.profile_function
    def memory_inefficient_processing():
        """Memory inefficient processing."""
        large_arrays = []
        for i in range(10):
            array = np.random.randn(1000, 1000)
            large_arrays.append(array)
        
        # Process arrays
        result = sum(np.sum(arr) for arr in large_arrays)
        
        # Don't clean up (memory leak simulation)
        return result
    
    # Memory efficient implementation
    @profiler.profile_function
    def memory_efficient_processing():
        """Memory efficient processing."""
        result = 0
        for i in range(10):
            array = np.random.randn(1000, 1000)
            result += np.sum(array)
            del array  # Explicit cleanup
        
        return result
    
    # Start profiling
    profiler.start_profiling()
    
    # Test original implementations
    print("Testing original implementations...")
    data_list = list(range(100))
    
    slow_result = slow_data_processing(data_list)
    memory_inefficient_result = memory_inefficient_processing()
    
    # Test optimized implementations
    print("Testing optimized implementations...")
    fast_result = fast_data_processing(data_list)
    memory_efficient_result = memory_efficient_processing()
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get results
    report = profiler.get_comprehensive_report()
    bottlenecks = profiler.identify_bottlenecks()
    
    print("Bottleneck Optimization Results:")
    print(f"Total functions profiled: {report['performance']['total_functions_profiled']}")
    
    # Compare performance
    profiles = report['performance']['profiles']
    
    print(f"\nPerformance Comparison:")
    if 'slow_data_processing' in profiles and 'fast_data_processing' in profiles:
        slow_time = profiles['slow_data_processing']['execution_time']
        fast_time = profiles['fast_data_processing']['execution_time']
        speedup = slow_time / fast_time if fast_time > 0 else float('inf')
        
        print(f"  Slow processing: {slow_time:.4f}s")
        print(f"  Fast processing: {fast_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    if 'memory_inefficient_processing' in profiles and 'memory_efficient_processing' in profiles:
        inefficient_memory = profiles['memory_inefficient_processing']['memory_delta']
        efficient_memory = profiles['memory_efficient_processing']['memory_delta']
        
        print(f"  Memory inefficient: {inefficient_memory / 1024 / 1024:.1f}MB")
        print(f"  Memory efficient: {efficient_memory / 1024 / 1024:.1f}MB")
        print(f"  Memory improvement: {abs(inefficient_memory - efficient_memory) / 1024 / 1024:.1f}MB")
    
    print(f"\nOptimization Recommendations:")
    for recommendation in bottlenecks['recommendations']:
        print(f"  - {recommendation}")
    
    return profiler

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_profiling_examples():
    """Run all profiling examples."""
    print("Code Profiling Examples for Video-OpusClip")
    print("=" * 60)
    
    # Check system information
    print(f"System Information:")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print()
    
    try:
        # Run examples
        example_basic_function_profiling()
        example_class_profiling()
        example_data_loading_profiling()
        example_memory_profiling()
        example_gpu_profiling()
        example_preprocessing_profiling()
        example_comprehensive_profiling()
        example_bottleneck_optimization()
        
        print("\n" + "=" * 60)
        print("All profiling examples completed successfully!")
        
        # Summary
        print("\nSummary:")
        print("- Basic function profiling: Identify slow functions")
        print("- Class profiling: Profile object-oriented code")
        print("- Data loading profiling: Optimize data pipelines")
        print("- Memory profiling: Detect memory issues")
        print("- GPU profiling: Optimize CUDA operations")
        print("- Preprocessing profiling: Improve data preparation")
        print("- Comprehensive profiling: Full system analysis")
        print("- Bottleneck optimization: Performance improvements")
        
    except Exception as e:
        print(f"\nError running profiling examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_profiling_examples() 