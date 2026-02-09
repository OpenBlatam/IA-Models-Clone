#!/usr/bin/env python3
"""
NumPy Examples for Video-OpusClip

Comprehensive examples demonstrating NumPy library usage
in the Video-OpusClip system for high-performance numerical computing.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# BASIC ARRAY EXAMPLES
# =============================================================================

def example_array_creation():
    """Example 1: Various ways to create NumPy arrays."""
    
    print("🔢 Example 1: Array Creation")
    print("=" * 50)
    
    # From Python lists
    list_data = [1, 2, 3, 4, 5]
    arr1 = np.array(list_data)
    print(f"From list: {arr1}")
    
    # From nested lists (2D array)
    nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    arr2 = np.array(nested_list)
    print(f"2D array:\n{arr2}")
    
    # Special arrays
    zeros = np.zeros((3, 4))
    ones = np.ones((2, 3))
    eye = np.eye(3)
    random_arr = np.random.rand(3, 3)
    
    print(f"Zeros:\n{zeros}")
    print(f"Ones:\n{ones}")
    print(f"Identity:\n{eye}")
    print(f"Random:\n{random_arr}")
    
    # Range arrays
    range_arr = np.arange(0, 10, 2)
    linspace_arr = np.linspace(0, 1, 5)
    logspace_arr = np.logspace(0, 2, 5)
    
    print(f"Range: {range_arr}")
    print(f"Linspace: {linspace_arr}")
    print(f"Logspace: {logspace_arr}")
    
    # Different data types
    float32_arr = np.array([1, 2, 3], dtype=np.float32)
    uint8_arr = np.array([255, 128, 64], dtype=np.uint8)
    complex_arr = np.array([1+2j, 3+4j, 5+6j])
    
    print(f"Float32: {float32_arr} (dtype: {float32_arr.dtype})")
    print(f"Uint8: {uint8_arr} (dtype: {uint8_arr.dtype})")
    print(f"Complex: {complex_arr} (dtype: {complex_arr.dtype})")
    
    return arr1, arr2

def example_array_operations():
    """Example 2: Basic array operations."""
    
    print("\n🔢 Example 2: Array Operations")
    print("=" * 50)
    
    # Create test arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])
    
    print(f"Array a: {a}")
    print(f"Array b: {b}")
    
    # Basic arithmetic
    print(f"\nBasic arithmetic:")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a ** 2 = {a ** 2}")
    
    # Broadcasting
    print(f"\nBroadcasting:")
    print(f"a + 10 = {a + 10}")
    print(f"a * 2 = {a * 2}")
    print(f"a > 3 = {a > 3}")
    
    # Statistical operations
    print(f"\nStatistical operations:")
    print(f"Mean: {np.mean(a)}")
    print(f"Median: {np.median(a)}")
    print(f"Standard deviation: {np.std(a)}")
    print(f"Variance: {np.var(a)}")
    print(f"Min: {np.min(a)}")
    print(f"Max: {np.max(a)}")
    print(f"Sum: {np.sum(a)}")
    print(f"Product: {np.prod(a)}")
    
    # Array methods
    print(f"\nArray methods:")
    print(f"a.mean() = {a.mean()}")
    print(f"a.std() = {a.std()}")
    print(f"a.var() = {a.var()}")
    print(f"a.min() = {a.min()}")
    print(f"a.max() = {a.max()}")
    print(f"a.sum() = {a.sum()}")
    
    return a, b

def example_indexing_slicing():
    """Example 3: Advanced indexing and slicing."""
    
    print("\n🔢 Example 3: Indexing and Slicing")
    print("=" * 50)
    
    # Create a 3D array
    arr = np.random.randint(0, 100, (4, 5, 3))
    print(f"3D array shape: {arr.shape}")
    print(f"3D array:\n{arr}")
    
    # Basic indexing
    print(f"\nBasic indexing:")
    print(f"arr[0, 0, 0] = {arr[0, 0, 0]}")
    print(f"arr[1, 2, 1] = {arr[1, 2, 1]}")
    
    # Slicing
    print(f"\nSlicing:")
    print(f"arr[0:2, 1:3, :] (first 2 rows, columns 1-2, all channels):\n{arr[0:2, 1:3, :]}")
    print(f"arr[:, 2, :] (all rows, column 2, all channels):\n{arr[:, 2, :]}")
    print(f"arr[1, :, 0] (row 1, all columns, channel 0): {arr[1, :, 0]}")
    
    # Boolean indexing
    print(f"\nBoolean indexing:")
    mask = arr > 50
    print(f"Mask (arr > 50) shape: {mask.shape}")
    print(f"Values where arr > 50: {arr[mask]}")
    print(f"Count of values > 50: {np.sum(mask)}")
    
    # Fancy indexing
    print(f"\nFancy indexing:")
    row_indices = [0, 2]
    col_indices = [1, 3]
    print(f"arr[row_indices, col_indices, :]:\n{arr[row_indices, col_indices, :]}")
    
    # Negative indexing
    print(f"\nNegative indexing:")
    print(f"arr[-1, :, :] (last row):\n{arr[-1, :, :]}")
    print(f"arr[:, -1, :] (last column):\n{arr[:, -1, :]}")
    print(f"arr[:, :, -1] (last channel):\n{arr[:, :, -1]}")
    
    # Step indexing
    print(f"\nStep indexing:")
    print(f"arr[::2, ::2, :] (every 2nd row and column):\n{arr[::2, ::2, :]}")
    
    return arr

# =============================================================================
# VIDEO PROCESSING EXAMPLES
# =============================================================================

def example_video_frame_processing():
    """Example 4: Video frame processing with NumPy."""
    
    print("\n🎬 Example 4: Video Frame Processing")
    print("=" * 50)
    
    # Simulate video frames
    num_frames = 5
    height, width, channels = 480, 640, 3
    
    # Create video array (frames, height, width, channels)
    video_frames = np.random.randint(0, 255, (num_frames, height, width, channels), dtype=np.uint8)
    print(f"Video shape: {video_frames.shape}")
    print(f"Video memory usage: {video_frames.nbytes / (1024*1024):.2f} MB")
    
    # Basic frame operations
    print(f"\nBasic frame operations:")
    
    # Convert to float for processing
    video_float = video_frames.astype(np.float32) / 255.0
    print(f"Converted to float range [0, 1]")
    
    # Apply gamma correction
    gamma = 0.8
    video_gamma = np.power(video_float, gamma)
    print(f"Applied gamma correction (gamma={gamma})")
    
    # Apply contrast enhancement
    mean_vals = np.mean(video_gamma, axis=(1, 2, 3), keepdims=True)
    video_contrast = (video_gamma - mean_vals) * 1.2 + mean_vals
    print(f"Applied contrast enhancement")
    
    # Clip values
    video_clipped = np.clip(video_contrast, 0.0, 1.0)
    print(f"Clipped values to [0, 1]")
    
    # Convert back to uint8
    video_processed = (video_clipped * 255.0).astype(np.uint8)
    print(f"Converted back to uint8")
    
    # Compute statistics
    print(f"\nVideo statistics:")
    print(f"Original mean brightness: {np.mean(video_frames):.2f}")
    print(f"Processed mean brightness: {np.mean(video_processed):.2f}")
    print(f"Original std brightness: {np.std(video_frames):.2f}")
    print(f"Processed std brightness: {np.std(video_processed):.2f}")
    
    # Frame-by-frame analysis
    print(f"\nFrame-by-frame analysis:")
    for i in range(num_frames):
        frame_mean = np.mean(video_frames[i])
        frame_processed_mean = np.mean(video_processed[i])
        frame_std = np.std(video_frames[i])
        frame_processed_std = np.std(video_processed[i])
        
        print(f"Frame {i+1}: mean {frame_mean:.1f}->{frame_processed_mean:.1f}, "
              f"std {frame_std:.1f}->{frame_processed_std:.1f}")
    
    return video_frames, video_processed

def example_batch_video_processing():
    """Example 5: Batch video processing with NumPy."""
    
    print("\n🎬 Example 5: Batch Video Processing")
    print("=" * 50)
    
    # Create multiple video batches
    batch_size = 3
    num_frames = 10
    height, width, channels = 240, 320, 3
    
    # Create batch of videos
    video_batch = np.random.randint(0, 255, (batch_size, num_frames, height, width, channels), dtype=np.uint8)
    print(f"Video batch shape: {video_batch.shape}")
    print(f"Total memory usage: {video_batch.nbytes / (1024*1024):.2f} MB")
    
    # Batch processing function
    def process_video_batch(videos: np.ndarray) -> np.ndarray:
        """Process a batch of videos efficiently."""
        
        # Normalize to [0, 1]
        videos_normalized = videos.astype(np.float32) / 255.0
        
        # Apply batch operations
        # Gamma correction
        videos_gamma = np.power(videos_normalized, 0.8)
        
        # Contrast enhancement
        mean_vals = np.mean(videos_gamma, axis=(1, 2, 3, 4), keepdims=True)
        videos_contrast = (videos_gamma - mean_vals) * 1.2 + mean_vals
        
        # Clip values
        videos_clipped = np.clip(videos_contrast, 0.0, 1.0)
        
        # Convert back to uint8
        return (videos_clipped * 255.0).astype(np.uint8)
    
    # Process batch
    start_time = time.time()
    processed_batch = process_video_batch(video_batch)
    processing_time = time.time() - start_time
    
    print(f"Batch processing completed in {processing_time:.4f} seconds")
    
    # Compute batch statistics
    print(f"\nBatch statistics:")
    print(f"Original batch mean: {np.mean(video_batch):.2f}")
    print(f"Processed batch mean: {np.mean(processed_batch):.2f}")
    print(f"Original batch std: {np.std(video_batch):.2f}")
    print(f"Processed batch std: {np.std(processed_batch):.2f}")
    
    # Per-video statistics
    print(f"\nPer-video statistics:")
    for i in range(batch_size):
        original_mean = np.mean(video_batch[i])
        processed_mean = np.mean(processed_batch[i])
        print(f"Video {i+1}: {original_mean:.1f} -> {processed_mean:.1f}")
    
    return video_batch, processed_batch

def example_video_filters():
    """Example 6: Video filtering with NumPy."""
    
    print("\n🎬 Example 6: Video Filters")
    print("=" * 50)
    
    # Create test video
    num_frames = 5
    height, width = 100, 100
    video = np.random.rand(num_frames, height, width).astype(np.float32)
    
    print(f"Video shape: {video.shape}")
    
    # Define filters
    def gaussian_filter_2d(size=5, sigma=1.0):
        """Create a 2D Gaussian filter."""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)
    
    def apply_filter_2d(frames: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D filter to all frames."""
        from scipy.ndimage import convolve
        
        filtered_frames = np.zeros_like(frames)
        for i in range(frames.shape[0]):
            filtered_frames[i] = convolve(frames[i], kernel, mode='reflect')
        
        return filtered_frames
    
    # Apply different filters
    print(f"\nApplying filters:")
    
    # Gaussian filter
    gaussian_kernel = gaussian_filter_2d(size=5, sigma=1.0)
    video_gaussian = apply_filter_2d(video, gaussian_kernel)
    print(f"Applied Gaussian filter")
    
    # Sharpening filter
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    video_sharpened = apply_filter_2d(video, sharpening_kernel)
    print(f"Applied sharpening filter")
    
    # Edge detection filter
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    video_edges = apply_filter_2d(video, edge_kernel)
    print(f"Applied edge detection filter")
    
    # Compute filter statistics
    print(f"\nFilter statistics:")
    print(f"Original std: {np.std(video):.4f}")
    print(f"Gaussian filtered std: {np.std(video_gaussian):.4f}")
    print(f"Sharpened std: {np.std(video_sharpened):.4f}")
    print(f"Edge detection std: {np.std(video_edges):.4f}")
    
    return video, video_gaussian, video_sharpened, video_edges

# =============================================================================
# PERFORMANCE OPTIMIZATION EXAMPLES
# =============================================================================

def example_performance_comparison():
    """Example 7: Performance comparison between different approaches."""
    
    print("\n⚡ Example 7: Performance Comparison")
    print("=" * 50)
    
    # Create large array
    size = 1000
    arr = np.random.rand(size, size)
    
    print(f"Array size: {size}x{size}")
    print(f"Memory usage: {arr.nbytes / (1024*1024):.2f} MB")
    
    # Method 1: Loops (slow)
    print(f"\nMethod 1: Loops")
    start_time = time.time()
    
    result_loop = np.zeros_like(arr)
    for i in range(size):
        for j in range(size):
            result_loop[i, j] = arr[i, j] ** 2 + np.sin(arr[i, j])
    
    loop_time = time.time() - start_time
    print(f"Loop time: {loop_time:.4f} seconds")
    
    # Method 2: Vectorized (fast)
    print(f"\nMethod 2: Vectorized")
    start_time = time.time()
    
    result_vectorized = arr ** 2 + np.sin(arr)
    
    vectorized_time = time.time() - start_time
    print(f"Vectorized time: {vectorized_time:.4f} seconds")
    
    # Method 3: Numba JIT (if available)
    try:
        from numba import jit
        
        @jit(nopython=True)
        def numba_function(arr):
            result = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    result[i, j] = arr[i, j] ** 2 + np.sin(arr[i, j])
            return result
        
        print(f"\nMethod 3: Numba JIT")
        start_time = time.time()
        
        result_numba = numba_function(arr)
        
        numba_time = time.time() - start_time
        print(f"Numba time: {numba_time:.4f} seconds")
        
    except ImportError:
        print(f"\nMethod 3: Numba not available")
        numba_time = float('inf')
    
    # Performance comparison
    print(f"\nPerformance comparison:")
    speedup_vectorized = loop_time / vectorized_time
    print(f"Vectorized speedup: {speedup_vectorized:.1f}x faster")
    
    if numba_time != float('inf'):
        speedup_numba = loop_time / numba_time
        print(f"Numba speedup: {speedup_numba:.1f}x faster")
    
    # Memory efficiency
    print(f"\nMemory efficiency:")
    print(f"Original array memory: {arr.nbytes / (1024*1024):.2f} MB")
    print(f"Result array memory: {result_vectorized.nbytes / (1024*1024):.2f} MB")
    
    return arr, result_vectorized

def example_memory_optimization():
    """Example 8: Memory optimization techniques."""
    
    print("\n⚡ Example 8: Memory Optimization")
    print("=" * 50)
    
    # Memory pool for efficient array reuse
    class NumPyMemoryPool:
        """Memory pool for efficient NumPy array reuse."""
        
        def __init__(self):
            self.pools = {}
        
        def get_array(self, shape: Tuple, dtype=np.float32) -> np.ndarray:
            """Get an array from the pool or create a new one."""
            key = (shape, dtype)
            
            if key in self.pools and self.pools[key]:
                return self.pools[key].pop()
            else:
                return np.empty(shape, dtype=dtype)
        
        def return_array(self, arr: np.ndarray):
            """Return an array to the pool."""
            key = (arr.shape, arr.dtype)
            
            if key not in self.pools:
                self.pools[key] = []
            
            # Clear the array
            arr.fill(0)
            self.pools[key].append(arr)
        
        def clear_pool(self):
            """Clear all pools."""
            self.pools.clear()
    
    # Test memory pool
    memory_pool = NumPyMemoryPool()
    
    # Create arrays using pool
    print("Creating arrays using memory pool:")
    
    for i in range(5):
        # Get array from pool
        arr = memory_pool.get_array((100, 100))
        
        # Use array
        arr.fill(i + 1)
        print(f"Array {i+1}: sum = {np.sum(arr)}")
        
        # Return array to pool
        memory_pool.return_array(arr)
    
    # Memory-efficient processing
    print(f"\nMemory-efficient processing:")
    
    def memory_efficient_processing(large_array: np.ndarray, chunk_size: int = 100):
        """Process large array in chunks to reduce memory usage."""
        
        total_size = large_array.shape[0]
        result = np.empty_like(large_array)
        
        for i in range(0, total_size, chunk_size):
            end_idx = min(i + chunk_size, total_size)
            chunk = large_array[i:end_idx]
            
            # Process chunk
            processed_chunk = chunk * 2 + 1
            result[i:end_idx] = processed_chunk
            
            # Force garbage collection
            del chunk, processed_chunk
            gc.collect()
        
        return result
    
    # Test memory-efficient processing
    large_array = np.random.rand(1000, 1000)
    print(f"Large array shape: {large_array.shape}")
    print(f"Memory usage: {large_array.nbytes / (1024*1024):.2f} MB")
    
    start_time = time.time()
    result = memory_efficient_processing(large_array, chunk_size=100)
    processing_time = time.time() - start_time
    
    print(f"Memory-efficient processing completed in {processing_time:.4f} seconds")
    
    return large_array, result

# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

def example_video_opusclip_integration():
    """Example 9: Integration with Video-OpusClip components."""
    
    print("\n🔗 Example 9: Video-OpusClip Integration")
    print("=" * 50)
    
    # Import Video-OpusClip components
    try:
        from optimized_config import get_config
        config = get_config()
        print("✅ Optimized config imported")
    except ImportError:
        config = {}
        print("⚠️ Optimized config not available")
    
    try:
        from performance_monitor import PerformanceMonitor
        performance_monitor = PerformanceMonitor(config)
        print("✅ Performance monitor imported")
    except ImportError:
        performance_monitor = None
        print("⚠️ Performance monitor not available")
    
    # Create integrated NumPy processor
    class NumPyVideoProcessor:
        """NumPy-based video processor for Video-OpusClip."""
        
        def __init__(self):
            self.config = config
            self.performance_monitor = performance_monitor
            self.setup_components()
        
        def setup_components(self):
            """Setup integration components."""
            print("✅ Integration components setup complete")
        
        def process_video_frames(self, frames: np.ndarray) -> Dict[str, Any]:
            """Process video frames using NumPy operations."""
            
            start_time = time.time()
            
            try:
                # Start performance monitoring
                if self.performance_monitor:
                    self.performance_monitor.start_timing()
                
                # Normalize frames
                frames_normalized = frames.astype(np.float32) / 255.0
                
                # Apply processing pipeline
                # Gamma correction
                gamma = self.config.get('gamma', 0.8)
                frames_gamma = np.power(frames_normalized, gamma)
                
                # Contrast enhancement
                contrast_factor = self.config.get('contrast_factor', 1.2)
                mean_vals = np.mean(frames_gamma, axis=(1, 2, 3), keepdims=True)
                frames_contrast = (frames_gamma - mean_vals) * contrast_factor + mean_vals
                
                # Brightness adjustment
                brightness_offset = self.config.get('brightness_offset', 0.1)
                frames_brightness = frames_contrast + brightness_offset
                
                # Clip values
                frames_clipped = np.clip(frames_brightness, 0.0, 1.0)
                
                # Convert back to uint8
                processed_frames = (frames_clipped * 255.0).astype(np.uint8)
                
                # Get performance metrics
                processing_time = time.time() - start_time
                metrics = {}
                
                if self.performance_monitor:
                    metrics = self.performance_monitor.get_metrics()
                    self.performance_monitor.end_timing()
                
                # Compute statistics
                stats = {
                    'original_mean': float(np.mean(frames)),
                    'processed_mean': float(np.mean(processed_frames)),
                    'original_std': float(np.std(frames)),
                    'processed_std': float(np.std(processed_frames)),
                    'processing_time': processing_time,
                    'frames_processed': frames.shape[0]
                }
                
                return {
                    "processed_frames": processed_frames,
                    "statistics": stats,
                    "metrics": metrics,
                    "config": {
                        "array_shape": frames.shape,
                        "data_type": str(frames.dtype),
                        "memory_usage_mb": frames.nbytes / (1024 * 1024),
                        "processing_parameters": {
                            "gamma": gamma,
                            "contrast_factor": contrast_factor,
                            "brightness_offset": brightness_offset
                        }
                    }
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "processing_time": time.time() - start_time
                }
    
    # Test integration
    processor = NumPyVideoProcessor()
    
    # Create test video frames
    test_frames = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)
    
    # Process frames
    result = processor.process_video_frames(test_frames)
    
    if "error" not in result:
        print("✅ Integration test successful")
        print(f"Processing time: {result['statistics']['processing_time']:.4f} seconds")
        print(f"Frames processed: {result['statistics']['frames_processed']}")
        print(f"Original mean: {result['statistics']['original_mean']:.2f}")
        print(f"Processed mean: {result['statistics']['processed_mean']:.2f}")
        print(f"Config: {result['config']}")
    else:
        print(f"❌ Integration test failed: {result['error']}")
    
    return test_frames, result

# =============================================================================
# ADVANCED EXAMPLES
# =============================================================================

def example_advanced_operations():
    """Example 10: Advanced NumPy operations."""
    
    print("\n🔢 Example 10: Advanced Operations")
    print("=" * 50)
    
    # Linear algebra operations
    print("Linear algebra operations:")
    
    A = np.random.rand(3, 3)
    b = np.random.rand(3)
    
    print(f"Matrix A:\n{A}")
    print(f"Vector b: {b}")
    
    # Solve linear system: Ax = b
    x = np.linalg.solve(A, b)
    print(f"Solution x: {x}")
    
    # Verify solution
    Ax = A @ x
    print(f"A @ x: {Ax}")
    print(f"b: {b}")
    print(f"Error: {np.linalg.norm(Ax - b):.2e}")
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(A)
    print(f"Singular values: {S}")
    
    # Broadcasting with complex operations
    print(f"\nComplex broadcasting:")
    
    # Create arrays for broadcasting
    arr_3d = np.random.rand(3, 4, 5)
    arr_2d = np.random.rand(4, 5)
    arr_1d = np.random.rand(5)
    
    print(f"3D array shape: {arr_3d.shape}")
    print(f"2D array shape: {arr_2d.shape}")
    print(f"1D array shape: {arr_1d.shape}")
    
    # Broadcasting operations
    result_1 = arr_3d + arr_2d
    result_2 = arr_3d * arr_1d
    result_3 = arr_2d + arr_1d
    
    print(f"3D + 2D result shape: {result_1.shape}")
    print(f"3D * 1D result shape: {result_2.shape}")
    print(f"2D + 1D result shape: {result_3.shape}")
    
    # Advanced indexing
    print(f"\nAdvanced indexing:")
    
    # Create test array
    arr = np.random.randint(0, 100, (5, 5))
    print(f"Original array:\n{arr}")
    
    # Boolean indexing with conditions
    mask = (arr > 50) & (arr < 80)
    filtered_values = arr[mask]
    print(f"Values between 50 and 80: {filtered_values}")
    
    # Fancy indexing
    row_indices = [0, 2, 4]
    col_indices = [1, 3]
    selected = arr[row_indices][:, col_indices]
    print(f"Selected elements:\n{selected}")
    
    # Structured arrays
    print(f"\nStructured arrays:")
    
    # Define data type
    dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]
    
    # Create structured array
    people = np.array([
        ('Alice', 25, 1.75),
        ('Bob', 30, 1.80),
        ('Charlie', 35, 1.70)
    ], dtype=dtype)
    
    print(f"Structured array:\n{people}")
    print(f"Names: {people['name']}")
    print(f"Ages: {people['age']}")
    print(f"Heights: {people['height']}")
    
    # Sort by age
    sorted_people = np.sort(people, order='age')
    print(f"Sorted by age:\n{sorted_people}")
    
    return A, b, x, people

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_examples():
    """Run all NumPy examples."""
    
    print("🚀 NumPy Examples for Video-OpusClip")
    print("=" * 60)
    
    examples = {
        "1": ("Array Creation", example_array_creation),
        "2": ("Array Operations", example_array_operations),
        "3": ("Indexing and Slicing", example_indexing_slicing),
        "4": ("Video Frame Processing", example_video_frame_processing),
        "5": ("Batch Video Processing", example_batch_video_processing),
        "6": ("Video Filters", example_video_filters),
        "7": ("Performance Comparison", example_performance_comparison),
        "8": ("Memory Optimization", example_memory_optimization),
        "9": ("Video-OpusClip Integration", example_video_opusclip_integration),
        "10": ("Advanced Operations", example_advanced_operations)
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\n0. Exit")
    
    while True:
        choice = input("\nEnter your choice (0-10): ").strip()
        
        if choice == "0":
            print("👋 Exiting...")
            break
        
        if choice in examples:
            name, func = examples[choice]
            print(f"\n🔢 Running: {name}")
            print("=" * 50)
            
            try:
                result = func()
                print(f"✅ {name} completed successfully")
                
            except Exception as e:
                print(f"❌ Error running {name}: {e}")
        
        else:
            print("❌ Invalid choice. Please enter a number between 0-10.")

if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    print("\n🔧 Next Steps:")
    print("1. Explore the NumPy documentation")
    print("2. Read the NUMPY_GUIDE.md for detailed usage")
    print("3. Run quick_start_numpy.py for basic setup")
    print("4. Integrate with your Video-OpusClip workflow") 