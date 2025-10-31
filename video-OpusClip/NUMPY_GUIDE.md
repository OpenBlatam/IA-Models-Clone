# NumPy Guide for Video-OpusClip

Complete guide to using the NumPy library in your Video-OpusClip system for high-performance numerical computing, array operations, and video processing.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Array Operations](#array-operations)
5. [Video Processing with NumPy](#video-processing-with-numpy)
6. [Performance Optimization](#performance-optimization)
7. [Integration with Video-OpusClip](#integration-with-video-opusclip)
8. [Advanced Features](#advanced-features)
9. [Memory Management](#memory-management)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

## Overview

NumPy is the fundamental package for scientific computing in Python. In your Video-OpusClip system, NumPy provides:

- **High-performance Array Operations**: Fast numerical computations for video frames
- **Memory-efficient Data Structures**: Optimized arrays for large video datasets
- **Mathematical Functions**: Advanced mathematical operations for video processing
- **Integration with Other Libraries**: Seamless integration with PyTorch, OpenCV, and other tools
- **Vectorized Operations**: Efficient processing of multiple video frames simultaneously

## Installation & Setup

### Current Dependencies

Your Video-OpusClip system already includes NumPy in the requirements:

```txt
# From requirements_complete.txt
numpy>=1.24.0
```

### Installation Commands

```bash
# Install basic NumPy
pip install numpy

# Install with optimized BLAS/LAPACK
pip install numpy[all]

# Install from your requirements
pip install -r requirements_complete.txt

# Install with specific optimizations
pip install numpy --no-binary numpy
```

### Verify Installation

```python
import numpy as np
print(f"NumPy version: {np.__version__}")

# Test basic functionality
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
print("✅ NumPy installation successful!")
```

### Check Optimizations

```python
import numpy as np

# Check BLAS/LAPACK
print(f"BLAS info: {np.__config__.show()}")
print(f"NumPy configuration: {np.__config__.get_info()}")

# Check if optimized
print(f"Using optimized BLAS: {'blas' in np.__config__.get_info()}")
```

## Core Concepts

### 1. Array Creation

```python
import numpy as np

# Basic arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Special arrays
zeros = np.zeros((3, 4))  # 3x4 array of zeros
ones = np.ones((2, 3))    # 2x3 array of ones
eye = np.eye(3)           # 3x3 identity matrix
random_arr = np.random.rand(5, 5)  # 5x5 random array

# Array with specific data type
float_arr = np.array([1, 2, 3], dtype=np.float32)
uint8_arr = np.array([255, 128, 64], dtype=np.uint8)
```

### 2. Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Shape: {arr.shape}")           # (3, 3)
print(f"Data type: {arr.dtype}")       # int64
print(f"Size: {arr.size}")             # 9
print(f"Number of dimensions: {arr.ndim}")  # 2
print(f"Memory usage: {arr.nbytes} bytes")  # 72
```

### 3. Array Indexing and Slicing

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Basic indexing
print(arr[0, 0])      # 1
print(arr[1, 2])      # 7

# Slicing
print(arr[0:2, 1:3])  # [[2, 3], [6, 7]]
print(arr[:, 2])      # [3, 7, 11] - all rows, column 2
print(arr[1, :])      # [5, 6, 7, 8] - row 1, all columns

# Boolean indexing
mask = arr > 5
print(arr[mask])      # [6, 7, 8, 9, 10, 11, 12]
```

## Array Operations

### 1. Mathematical Operations

```python
import numpy as np

# Basic arithmetic
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)    # [6, 8, 10, 12]
print(a - b)    # [-4, -4, -4, -4]
print(a * b)    # [5, 12, 21, 32]
print(a / b)    # [0.2, 0.333, 0.429, 0.5]
print(a ** 2)   # [1, 4, 9, 16]

# Statistical operations
print(np.mean(a))      # 2.5
print(np.std(a))       # 1.118
print(np.var(a))       # 1.25
print(np.median(a))    # 2.5
print(np.min(a))       # 1
print(np.max(a))       # 4
```

### 2. Broadcasting

```python
# Broadcasting allows operations between arrays of different shapes
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 2

# Broadcasting scalar to array
print(arr + scalar)    # [[3, 4, 5], [6, 7, 8]]
print(arr * scalar)    # [[2, 4, 6], [8, 10, 12]]

# Broadcasting arrays
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])

print(a + b)    # [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
```

### 3. Linear Algebra

```python
import numpy as np
from numpy import linalg as LA

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
C = A @ B  # Alternative syntax

# Matrix properties
print(f"Determinant: {LA.det(A)}")
print(f"Eigenvalues: {LA.eigvals(A)}")
print(f"Inverse: {LA.inv(A)}")

# Solving linear equations: Ax = b
b = np.array([1, 2])
x = LA.solve(A, b)
```

## Video Processing with NumPy

### 1. Image/Frame Representation

```python
import numpy as np
import cv2

# Load image as NumPy array
image = cv2.imread('image.jpg')
print(f"Image shape: {image.shape}")  # (height, width, channels)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"Grayscale shape: {gray.shape}")  # (height, width)

# Video frame processing
def process_frame(frame):
    """Process a single video frame."""
    # Convert to float for processing
    frame_float = frame.astype(np.float32) / 255.0
    
    # Apply gamma correction
    frame_processed = np.power(frame_float, 0.8)
    
    # Apply contrast enhancement
    mean_val = np.mean(frame_processed)
    frame_processed = (frame_processed - mean_val) * 1.2 + mean_val
    
    # Clip values
    frame_processed = np.clip(frame_processed, 0.0, 1.0)
    
    # Convert back to uint8
    return (frame_processed * 255.0).astype(np.uint8)
```

### 2. Batch Frame Processing

```python
import numpy as np
from typing import List

def process_frames_batch(frames: List[np.ndarray]) -> np.ndarray:
    """Process multiple frames efficiently."""
    
    # Convert to NumPy array for vectorized operations
    frames_array = np.array(frames)
    print(f"Batch shape: {frames_array.shape}")  # (num_frames, height, width, channels)
    
    # Normalize to [0, 1]
    frames_normalized = frames_array.astype(np.float32) / 255.0
    
    # Apply batch operations
    # Gamma correction
    frames_gamma = np.power(frames_normalized, 0.8)
    
    # Contrast enhancement
    mean_vals = np.mean(frames_gamma, axis=(1, 2, 3), keepdims=True)
    frames_contrast = (frames_gamma - mean_vals) * 1.2 + mean_vals
    
    # Clip values
    frames_clipped = np.clip(frames_contrast, 0.0, 1.0)
    
    # Convert back to uint8
    return (frames_clipped * 255.0).astype(np.uint8)

# Usage
frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
processed_frames = process_frames_batch(frames)
```

### 3. Video Array Operations

```python
import numpy as np

class VideoArrayProcessor:
    """High-performance video array processing."""
    
    def __init__(self, target_shape=(480, 640, 3)):
        self.target_shape = target_shape
    
    def resize_frames(self, frames: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize all frames in a batch."""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (
            target_size[0] / frames.shape[1],
            target_size[1] / frames.shape[2],
            1  # Keep channels unchanged
        )
        
        # Apply zoom to all frames
        resized_frames = zoom(frames, (1, *zoom_factors), order=1)
        return resized_frames
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize frames to [0, 1] range."""
        return frames.astype(np.float32) / 255.0
    
    def denormalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Convert frames back to [0, 255] range."""
        return np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    
    def apply_filter(self, frames: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D filter to all frames."""
        from scipy.ndimage import convolve
        
        # Apply convolution to each frame
        filtered_frames = np.zeros_like(frames)
        for i in range(frames.shape[0]):
            for c in range(frames.shape[3]):
                filtered_frames[i, :, :, c] = convolve(
                    frames[i, :, :, c], kernel, mode='reflect'
                )
        
        return filtered_frames
    
    def compute_optical_flow(self, frames: np.ndarray) -> np.ndarray:
        """Compute optical flow between consecutive frames."""
        import cv2
        
        flows = []
        for i in range(1, frames.shape[0]):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
        
        return np.array(flows)
```

## Performance Optimization

### 1. Memory Layout Optimization

```python
import numpy as np

# Ensure contiguous memory layout
def optimize_memory_layout(arr: np.ndarray) -> np.ndarray:
    """Optimize array memory layout for better performance."""
    
    # Check if array is contiguous
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    
    # Align memory for better performance
    if arr.dtype.itemsize % 8 == 0:
        # Ensure 8-byte alignment
        arr = np.asarray(arr, dtype=arr.dtype, order='C')
    
    return arr

# Example usage
large_array = np.random.rand(1000, 1000, 3)
optimized_array = optimize_memory_layout(large_array)
```

### 2. Vectorized Operations

```python
import numpy as np
import time

# Non-vectorized (slow)
def slow_operation(arr):
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = arr[i, j] ** 2 + np.sin(arr[i, j])
    return result

# Vectorized (fast)
def fast_operation(arr):
    return arr ** 2 + np.sin(arr)

# Performance comparison
arr = np.random.rand(1000, 1000)

start_time = time.time()
result_slow = slow_operation(arr)
slow_time = time.time() - start_time

start_time = time.time()
result_fast = fast_operation(arr)
fast_time = time.time() - start_time

print(f"Slow operation: {slow_time:.4f}s")
print(f"Fast operation: {fast_time:.4f}s")
print(f"Speedup: {slow_time / fast_time:.1f}x")
```

### 3. Numba JIT Compilation

```python
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def numba_optimized_processing(frames: np.ndarray) -> np.ndarray:
    """Numba-optimized frame processing."""
    
    num_frames, height, width, channels = frames.shape
    processed_frames = np.empty_like(frames)
    
    for i in prange(num_frames):
        for h in range(height):
            for w in range(width):
                for c in range(channels):
                    # Apply processing
                    pixel = frames[i, h, w, c]
                    processed_pixel = pixel * 1.2 + 10
                    processed_frames[i, h, w, c] = np.clip(processed_pixel, 0, 255)
    
    return processed_frames

# Usage
frames = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
processed = numba_optimized_processing(frames)
```

### 4. Memory Pooling

```python
import numpy as np
from typing import Dict, Tuple

class NumPyMemoryPool:
    """Memory pool for efficient NumPy array reuse."""
    
    def __init__(self):
        self.pools: Dict[Tuple, list] = {}
    
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

# Usage
memory_pool = NumPyMemoryPool()

# Get arrays from pool
arr1 = memory_pool.get_array((100, 100))
arr2 = memory_pool.get_array((100, 100))

# Use arrays
arr1.fill(1.0)
arr2.fill(2.0)

# Return arrays to pool
memory_pool.return_array(arr1)
memory_pool.return_array(arr2)
```

## Integration with Video-OpusClip

### 1. Integration with Existing Components

```python
import numpy as np
from optimized_libraries import OptimizedVideoDiffusionPipeline
from enhanced_error_handling import safe_load_ai_model

class NumPyVideoProcessor:
    """NumPy-based video processing for Video-OpusClip."""
    
    def __init__(self):
        self.video_generator = OptimizedVideoDiffusionPipeline()
        self.memory_pool = NumPyMemoryPool()
    
    def process_video_frames(self, frames: np.ndarray) -> np.ndarray:
        """Process video frames using NumPy operations."""
        
        # Get optimized array from pool
        processed_frames = self.memory_pool.get_array(frames.shape, frames.dtype)
        
        try:
            # Normalize frames
            frames_normalized = frames.astype(np.float32) / 255.0
            
            # Apply processing
            # Gamma correction
            frames_gamma = np.power(frames_normalized, 0.8)
            
            # Contrast enhancement
            mean_vals = np.mean(frames_gamma, axis=(1, 2, 3), keepdims=True)
            frames_contrast = (frames_gamma - mean_vals) * 1.2 + mean_vals
            
            # Clip values
            frames_clipped = np.clip(frames_contrast, 0.0, 1.0)
            
            # Convert back to uint8
            processed_frames[:] = (frames_clipped * 255.0).astype(np.uint8)
            
            return processed_frames.copy()
            
        finally:
            # Return arrays to pool
            self.memory_pool.return_array(processed_frames)
    
    def generate_video_arrays(self, prompt: str, num_frames: int = 30) -> np.ndarray:
        """Generate video frames as NumPy arrays."""
        
        # Generate frames using AI pipeline
        frames = self.video_generator.generate_video_frames(
            prompt=prompt,
            num_frames=num_frames
        )
        
        # Convert to NumPy array
        frames_array = np.array(frames)
        
        return frames_array
```

### 2. Performance Monitoring Integration

```python
import numpy as np
from performance_monitor import PerformanceMonitor

class NumPyPerformanceMonitor:
    """Monitor NumPy operations performance."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.operation_times = {}
    
    def monitor_operation(self, operation_name: str):
        """Decorator to monitor NumPy operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = self.monitor.start_timing()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = self.monitor.end_timing(start_time)
                    
                    if operation_name not in self.operation_times:
                        self.operation_times[operation_name] = []
                    
                    self.operation_times[operation_name].append(execution_time)
            
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            stats[operation] = {
                'count': len(times),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        return stats

# Usage
numpy_monitor = NumPyPerformanceMonitor()

@numpy_monitor.monitor_operation("frame_processing")
def process_frames(frames: np.ndarray) -> np.ndarray:
    """Process frames with monitoring."""
    return frames * 1.2 + 10

# Get performance stats
stats = numpy_monitor.get_performance_stats()
print(f"Performance stats: {stats}")
```

## Advanced Features

### 1. Custom NumPy UFuncs

```python
import numpy as np
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='parallel')
def custom_video_filter(pixel, factor):
    """Custom vectorized video filter."""
    return pixel * factor + np.sin(pixel * 0.1)

# Usage
frames = np.random.rand(100, 480, 640, 3).astype(np.float32)
filtered_frames = custom_video_filter(frames, 1.2)
```

### 2. Advanced Broadcasting

```python
import numpy as np

def advanced_video_processing(frames: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Advanced video processing with broadcasting."""
    
    # frames: (num_frames, height, width, channels)
    # filters: (num_filters, height, width, channels)
    
    # Add frame dimension to filters for broadcasting
    filters_expanded = filters[:, np.newaxis, :, :, :]
    
    # Apply all filters to all frames
    # Result: (num_filters, num_frames, height, width, channels)
    filtered_frames = frames * filters_expanded
    
    # Average across filters
    result = np.mean(filtered_frames, axis=0)
    
    return result

# Usage
frames = np.random.rand(10, 480, 640, 3)
filters = np.random.rand(5, 480, 640, 3)
result = advanced_video_processing(frames, filters)
```

### 3. Memory-mapped Arrays

```python
import numpy as np
import tempfile
import os

def create_memory_mapped_video(filename: str, shape: tuple) -> np.ndarray:
    """Create a memory-mapped video array."""
    
    # Create memory-mapped array
    mmap_array = np.memmap(
        filename,
        dtype=np.uint8,
        mode='w+',
        shape=shape
    )
    
    return mmap_array

def load_memory_mapped_video(filename: str, shape: tuple) -> np.ndarray:
    """Load a memory-mapped video array."""
    
    # Load memory-mapped array
    mmap_array = np.memmap(
        filename,
        dtype=np.uint8,
        mode='r',
        shape=shape
    )
    
    return mmap_array

# Usage
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
filename = temp_file.name
temp_file.close()

# Create memory-mapped video
video_shape = (100, 480, 640, 3)
video_array = create_memory_mapped_video(filename, video_shape)

# Fill with data
video_array[:] = np.random.randint(0, 255, video_shape, dtype=np.uint8)

# Load memory-mapped video
loaded_video = load_memory_mapped_video(filename, video_shape)

# Clean up
os.unlink(filename)
```

## Memory Management

### 1. Memory-efficient Operations

```python
import numpy as np
import gc

def memory_efficient_processing(frames: np.ndarray) -> np.ndarray:
    """Memory-efficient frame processing."""
    
    # Process in chunks to reduce memory usage
    chunk_size = 10
    num_frames = frames.shape[0]
    processed_frames = np.empty_like(frames)
    
    for i in range(0, num_frames, chunk_size):
        end_idx = min(i + chunk_size, num_frames)
        chunk = frames[i:end_idx]
        
        # Process chunk
        processed_chunk = chunk * 1.2 + 10
        processed_frames[i:end_idx] = processed_chunk
        
        # Force garbage collection
        del chunk, processed_chunk
        gc.collect()
    
    return processed_frames

# Usage
large_frames = np.random.rand(1000, 480, 640, 3)
processed = memory_efficient_processing(large_frames)
```

### 2. Memory Pooling with Context Manager

```python
import numpy as np
from contextlib import contextmanager

class NumPyMemoryManager:
    """Context manager for NumPy memory management."""
    
    def __init__(self):
        self.allocated_arrays = []
    
    @contextmanager
    def temporary_array(self, shape: tuple, dtype=np.float32):
        """Context manager for temporary arrays."""
        arr = np.empty(shape, dtype=dtype)
        self.allocated_arrays.append(arr)
        
        try:
            yield arr
        finally:
            self.allocated_arrays.remove(arr)
            del arr
            gc.collect()
    
    def clear_all(self):
        """Clear all allocated arrays."""
        for arr in self.allocated_arrays:
            del arr
        self.allocated_arrays.clear()
        gc.collect()

# Usage
memory_manager = NumPyMemoryManager()

with memory_manager.temporary_array((100, 100)) as temp_arr:
    temp_arr.fill(1.0)
    result = temp_arr * 2
    print(f"Result sum: {np.sum(result)}")

# Array is automatically cleaned up
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Solution: Use memory-efficient operations
   import numpy as np
   
   # Instead of loading all frames at once
   frames = np.load('large_video.npy')  # Memory error
   
   # Use memory mapping
   frames = np.load('large_video.npy', mmap_mode='r')
   ```

2. **Performance Issues**
   ```python
   # Solution: Use vectorized operations
   # Slow: loop-based operations
   result = np.zeros_like(arr)
   for i in range(arr.shape[0]):
       for j in range(arr.shape[1]):
           result[i, j] = arr[i, j] ** 2
   
   # Fast: vectorized operations
   result = arr ** 2
   ```

3. **Data Type Issues**
   ```python
   # Solution: Ensure correct data types
   # Check data type
   print(f"Array dtype: {arr.dtype}")
   
   # Convert if needed
   if arr.dtype != np.float32:
       arr = arr.astype(np.float32)
   ```

4. **Shape Mismatch**
   ```python
   # Solution: Reshape arrays properly
   # Check shapes
   print(f"Array shapes: {arr1.shape}, {arr2.shape}")
   
   # Reshape if needed
   if arr1.shape != arr2.shape:
       arr2 = arr2.reshape(arr1.shape)
   ```

### Debug Mode

```python
# Enable NumPy debugging
import numpy as np

# Set error handling
np.seterr(all='raise')

# Check array properties
def debug_array(arr):
    print(f"Shape: {arr.shape}")
    print(f"Data type: {arr.dtype}")
    print(f"Memory layout: {arr.flags}")
    print(f"Contiguous: {arr.flags['C_CONTIGUOUS']}")
    print(f"Memory usage: {arr.nbytes} bytes")
```

## Examples

### Complete Video Processing Pipeline

```python
import numpy as np
import cv2
from typing import List, Tuple
import time

class NumPyVideoPipeline:
    """Complete NumPy-based video processing pipeline."""
    
    def __init__(self, target_size: Tuple[int, int] = (480, 640)):
        self.target_size = target_size
        self.memory_pool = NumPyMemoryPool()
    
    def load_video_frames(self, video_path: str, max_frames: int = None) -> np.ndarray:
        """Load video frames as NumPy array."""
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
        finally:
            cap.release()
        
        return np.array(frames)
    
    def resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize all frames to target size."""
        
        num_frames, height, width, channels = frames.shape
        resized_frames = np.empty((num_frames, *self.target_size, channels), dtype=frames.dtype)
        
        for i in range(num_frames):
            resized_frames[i] = cv2.resize(frames[i], self.target_size)
        
        return resized_frames
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize frames to [0, 1] range."""
        return frames.astype(np.float32) / 255.0
    
    def apply_filters(self, frames: np.ndarray) -> np.ndarray:
        """Apply various filters to frames."""
        
        # Get temporary array from pool
        filtered_frames = self.memory_pool.get_array(frames.shape, frames.dtype)
        
        try:
            # Gamma correction
            gamma = 0.8
            filtered_frames[:] = np.power(frames, gamma)
            
            # Contrast enhancement
            mean_vals = np.mean(filtered_frames, axis=(1, 2, 3), keepdims=True)
            filtered_frames[:] = (filtered_frames - mean_vals) * 1.2 + mean_vals
            
            # Clip values
            filtered_frames[:] = np.clip(filtered_frames, 0.0, 1.0)
            
            return filtered_frames.copy()
            
        finally:
            self.memory_pool.return_array(filtered_frames)
    
    def compute_statistics(self, frames: np.ndarray) -> dict:
        """Compute video statistics."""
        
        stats = {
            'num_frames': frames.shape[0],
            'resolution': frames.shape[1:3],
            'channels': frames.shape[3],
            'mean_brightness': np.mean(frames),
            'std_brightness': np.std(frames),
            'min_brightness': np.min(frames),
            'max_brightness': np.max(frames),
            'total_memory_mb': frames.nbytes / (1024 * 1024)
        }
        
        return stats
    
    def save_video(self, frames: np.ndarray, output_path: str, fps: int = 30):
        """Save frames as video."""
        
        # Convert back to uint8 if needed
        if frames.dtype != np.uint8:
            frames = (frames * 255.0).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frames_bgr = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames])
        
        # Get video properties
        height, width = frames_bgr.shape[1:3]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame in frames_bgr:
                out.write(frame)
        finally:
            out.release()
    
    def process_video(self, video_path: str, output_path: str) -> dict:
        """Complete video processing pipeline."""
        
        start_time = time.time()
        
        # Load video
        print("Loading video...")
        frames = self.load_video_frames(video_path)
        
        # Compute initial statistics
        initial_stats = self.compute_statistics(frames)
        print(f"Initial stats: {initial_stats}")
        
        # Resize frames
        print("Resizing frames...")
        frames = self.resize_frames(frames)
        
        # Normalize frames
        print("Normalizing frames...")
        frames_normalized = self.normalize_frames(frames)
        
        # Apply filters
        print("Applying filters...")
        frames_filtered = self.apply_filters(frames_normalized)
        
        # Save processed video
        print("Saving video...")
        self.save_video(frames_filtered, output_path)
        
        # Compute final statistics
        final_stats = self.compute_statistics(frames_filtered)
        
        # Performance metrics
        processing_time = time.time() - start_time
        
        return {
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'processing_time': processing_time,
            'output_path': output_path
        }

# Usage
pipeline = NumPyVideoPipeline(target_size=(480, 640))

# Process video
result = pipeline.process_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4'
)

print(f"Processing completed in {result['processing_time']:.2f} seconds")
print(f"Output saved to: {result['output_path']}")
```

This comprehensive guide covers all aspects of using NumPy in your Video-OpusClip system. NumPy provides the foundation for high-performance numerical computing and array operations that are essential for video processing and AI model integration.

The integration with your existing components ensures seamless operation with your optimized libraries, error handling, and performance monitoring systems. 