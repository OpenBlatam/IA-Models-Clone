#!/usr/bin/env python3
"""
Quick Start NumPy for Video-OpusClip

This script demonstrates how to quickly get started with NumPy
in the Video-OpusClip system for high-performance numerical computing.
"""

import sys
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_numpy_installation():
    """Check if NumPy is properly installed."""
    
    print("🔍 Checking NumPy Installation")
    print("=" * 50)
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        
        # Test basic imports
        from numpy import array, zeros, ones, random, linalg
        print("✅ Core components imported successfully")
        
        from numpy import ndarray, dtype
        print("✅ Array types imported successfully")
        
        # Check configuration
        config_info = np.__config__.get_info()
        print("✅ NumPy configuration loaded")
        
        # Check optimizations
        blas_available = 'blas' in config_info
        lapack_available = 'lapack' in config_info
        
        print(f"✅ BLAS available: {blas_available}")
        print(f"✅ LAPACK available: {lapack_available}")
        
        return True
        
    except ImportError as e:
        print(f"❌ NumPy import error: {e}")
        print("💡 Install with: pip install numpy")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def quick_start_basic_operations():
    """Basic NumPy operations demonstration."""
    
    print("\n🔢 Quick Start: Basic Operations")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Create arrays
        print("Creating arrays...")
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([[1, 2, 3], [4, 5, 6]])
        
        print(f"1D Array: {arr1}")
        print(f"2D Array:\n{arr2}")
        
        # Basic operations
        print("\nBasic operations:")
        print(f"Sum: {np.sum(arr1)}")
        print(f"Mean: {np.mean(arr1)}")
        print(f"Standard deviation: {np.std(arr1)}")
        
        # Array properties
        print(f"\nArray properties:")
        print(f"Shape: {arr2.shape}")
        print(f"Data type: {arr2.dtype}")
        print(f"Size: {arr2.size}")
        print(f"Memory usage: {arr2.nbytes} bytes")
        
        # Mathematical operations
        print(f"\nMathematical operations:")
        print(f"Array + 2: {arr1 + 2}")
        print(f"Array * 2: {arr1 * 2}")
        print(f"Array squared: {arr1 ** 2}")
        
        print("✅ Basic operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic operations error: {e}")
        return False

def quick_start_array_creation():
    """NumPy array creation demonstration."""
    
    print("\n🔢 Quick Start: Array Creation")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Different ways to create arrays
        print("Creating arrays with different methods:")
        
        # From lists
        list_array = np.array([1, 2, 3, 4, 5])
        print(f"From list: {list_array}")
        
        # Zeros and ones
        zeros_array = np.zeros((3, 4))
        ones_array = np.ones((2, 3))
        print(f"Zeros array:\n{zeros_array}")
        print(f"Ones array:\n{ones_array}")
        
        # Identity matrix
        eye_array = np.eye(3)
        print(f"Identity matrix:\n{eye_array}")
        
        # Random arrays
        random_array = np.random.rand(3, 3)
        print(f"Random array:\n{random_array}")
        
        # Range arrays
        range_array = np.arange(0, 10, 2)
        linspace_array = np.linspace(0, 1, 5)
        print(f"Range array: {range_array}")
        print(f"Linspace array: {linspace_array}")
        
        # Different data types
        float_array = np.array([1, 2, 3], dtype=np.float32)
        uint8_array = np.array([255, 128, 64], dtype=np.uint8)
        print(f"Float32 array: {float_array} (dtype: {float_array.dtype})")
        print(f"Uint8 array: {uint8_array} (dtype: {uint8_array.dtype})")
        
        print("✅ Array creation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Array creation error: {e}")
        return False

def quick_start_indexing_slicing():
    """NumPy indexing and slicing demonstration."""
    
    print("\n🔢 Quick Start: Indexing and Slicing")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Create a 2D array
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        print(f"Original array:\n{arr}")
        
        # Basic indexing
        print(f"\nBasic indexing:")
        print(f"arr[0, 0] = {arr[0, 0]}")
        print(f"arr[1, 2] = {arr[1, 2]}")
        
        # Slicing
        print(f"\nSlicing:")
        print(f"arr[0:2, 1:3] (first 2 rows, columns 1-2):\n{arr[0:2, 1:3]}")
        print(f"arr[:, 2] (all rows, column 2): {arr[:, 2]}")
        print(f"arr[1, :] (row 1, all columns): {arr[1, :]}")
        
        # Boolean indexing
        print(f"\nBoolean indexing:")
        mask = arr > 5
        print(f"Mask (arr > 5):\n{mask}")
        print(f"Values where arr > 5: {arr[mask]}")
        
        # Fancy indexing
        print(f"\nFancy indexing:")
        indices = [0, 2]
        print(f"arr[indices, :] (rows 0 and 2):\n{arr[indices, :]}")
        
        # Negative indexing
        print(f"\nNegative indexing:")
        print(f"arr[-1, :] (last row): {arr[-1, :]}")
        print(f"arr[:, -1] (last column): {arr[:, -1]}")
        
        print("✅ Indexing and slicing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Indexing and slicing error: {e}")
        return False

def quick_start_mathematical_operations():
    """NumPy mathematical operations demonstration."""
    
    print("\n🔢 Quick Start: Mathematical Operations")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Create test arrays
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([5, 4, 3, 2, 1])
        matrix = np.array([[1, 2], [3, 4]])
        
        print(f"Array a: {a}")
        print(f"Array b: {b}")
        print(f"Matrix:\n{matrix}")
        
        # Basic arithmetic
        print(f"\nBasic arithmetic:")
        print(f"a + b = {a + b}")
        print(f"a - b = {a - b}")
        print(f"a * b = {a * b}")
        print(f"a / b = {a / b}")
        print(f"a ** 2 = {a ** 2}")
        
        # Statistical operations
        print(f"\nStatistical operations:")
        print(f"Mean of a: {np.mean(a)}")
        print(f"Median of a: {np.median(a)}")
        print(f"Standard deviation of a: {np.std(a)}")
        print(f"Variance of a: {np.var(a)}")
        print(f"Min of a: {np.min(a)}")
        print(f"Max of a: {np.max(a)}")
        print(f"Sum of a: {np.sum(a)}")
        
        # Broadcasting
        print(f"\nBroadcasting:")
        print(f"a + 10 = {a + 10}")
        print(f"a * 2 = {a * 2}")
        
        # Linear algebra
        print(f"\nLinear algebra:")
        print(f"Matrix determinant: {np.linalg.det(matrix)}")
        print(f"Matrix eigenvalues: {np.linalg.eigvals(matrix)}")
        print(f"Matrix inverse:\n{np.linalg.inv(matrix)}")
        
        # Trigonometric functions
        print(f"\nTrigonometric functions:")
        angles = np.array([0, np.pi/4, np.pi/2])
        print(f"Angles: {angles}")
        print(f"Sin: {np.sin(angles)}")
        print(f"Cos: {np.cos(angles)}")
        print(f"Tan: {np.tan(angles)}")
        
        print("✅ Mathematical operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical operations error: {e}")
        return False

def quick_start_video_processing():
    """NumPy video processing demonstration."""
    
    print("\n🎬 Quick Start: Video Processing")
    print("=" * 50)
    
    try:
        import numpy as np
        
        # Simulate video frames
        print("Creating simulated video frames...")
        num_frames = 5
        height, width, channels = 480, 640, 3
        
        # Create video array (frames, height, width, channels)
        video_frames = np.random.randint(0, 255, (num_frames, height, width, channels), dtype=np.uint8)
        print(f"Video shape: {video_frames.shape}")
        print(f"Video memory usage: {video_frames.nbytes / (1024*1024):.2f} MB")
        
        # Basic video operations
        print(f"\nBasic video operations:")
        
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
        
        # Frame-by-frame processing
        print(f"\nFrame-by-frame processing:")
        for i in range(num_frames):
            frame_mean = np.mean(video_frames[i])
            frame_processed_mean = np.mean(video_processed[i])
            print(f"Frame {i+1}: {frame_mean:.1f} -> {frame_processed_mean:.1f}")
        
        print("✅ Video processing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        return False

def quick_start_performance_demo():
    """NumPy performance demonstration."""
    
    print("\n⚡ Quick Start: Performance Demo")
    print("=" * 50)
    
    try:
        import numpy as np
        import time
        
        # Performance comparison: loops vs vectorized operations
        print("Performance comparison: loops vs vectorized operations")
        
        # Create large array
        size = 1000
        arr = np.random.rand(size, size)
        
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
        
        # Performance comparison
        speedup = loop_time / vectorized_time
        print(f"\nPerformance comparison:")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"Time saved: {loop_time - vectorized_time:.4f} seconds")
        
        # Memory efficiency
        print(f"\nMemory efficiency:")
        print(f"Array size: {arr.shape}")
        print(f"Memory usage: {arr.nbytes / (1024*1024):.2f} MB")
        print(f"Data type: {arr.dtype}")
        
        # Broadcasting performance
        print(f"\nBroadcasting performance:")
        start_time = time.time()
        
        # Broadcasting operation
        result_broadcast = arr + np.array([1, 2, 3])[:, np.newaxis]
        
        broadcast_time = time.time() - start_time
        print(f"Broadcasting time: {broadcast_time:.4f} seconds")
        
        print("✅ Performance demo completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Performance demo error: {e}")
        return False

def quick_start_integration_demo():
    """Demonstrate integration with Video-OpusClip components."""
    
    print("\n🔗 Quick Start: Video-OpusClip Integration")
    print("=" * 50)
    
    try:
        import numpy as np
        
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
            
            def process_video_frames(self, frames: np.ndarray) -> np.ndarray:
                """Process video frames using NumPy operations."""
                
                start_time = time.time()
                
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
                    processed_frames = (frames_clipped * 255.0).astype(np.uint8)
                    
                    # Get performance metrics if available
                    metrics = {}
                    if self.performance_monitor:
                        metrics = self.performance_monitor.get_metrics()
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "processed_frames": processed_frames,
                        "processing_time": processing_time,
                        "metrics": metrics,
                        "config": {
                            "array_shape": frames.shape,
                            "data_type": str(frames.dtype),
                            "memory_usage_mb": frames.nbytes / (1024 * 1024)
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
            print(f"Processing time: {result['processing_time']:.4f} seconds")
            print(f"Config: {result['config']}")
        else:
            print(f"❌ Integration test failed: {result['error']}")
        
        return "error" not in result
        
    except Exception as e:
        print(f"❌ Integration demo error: {e}")
        return False

def run_all_quick_starts():
    """Run all NumPy quick start demonstrations."""
    
    print("🚀 NumPy Quick Start for Video-OpusClip")
    print("=" * 60)
    
    results = {}
    
    # Check installation
    results['installation'] = check_numpy_installation()
    
    if results['installation']:
        print("\n🎯 Choose a demo to run:")
        print("1. Basic Operations")
        print("2. Array Creation")
        print("3. Indexing and Slicing")
        print("4. Mathematical Operations")
        print("5. Video Processing")
        print("6. Performance Demo")
        print("7. Integration Demo")
        print("8. Run all demos")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == "1":
            results['basic'] = quick_start_basic_operations()
        elif choice == "2":
            results['creation'] = quick_start_array_creation()
        elif choice == "3":
            results['indexing'] = quick_start_indexing_slicing()
        elif choice == "4":
            results['math'] = quick_start_mathematical_operations()
        elif choice == "5":
            results['video'] = quick_start_video_processing()
        elif choice == "6":
            results['performance'] = quick_start_performance_demo()
        elif choice == "7":
            results['integration'] = quick_start_integration_demo()
        elif choice == "8":
            print("\n🔄 Running all demos...")
            results['basic'] = quick_start_basic_operations()
            results['creation'] = quick_start_array_creation()
            results['indexing'] = quick_start_indexing_slicing()
            results['math'] = quick_start_mathematical_operations()
            results['video'] = quick_start_video_processing()
            results['performance'] = quick_start_performance_demo()
            results['integration'] = quick_start_integration_demo()
        elif choice == "0":
            print("👋 Exiting...")
            return
        else:
            print("❌ Invalid choice")
            return
    
    # Summary
    print("\n📊 Quick Start Summary")
    print("=" * 60)
    
    if results.get('installation'):
        print("✅ Installation: Successful")
        
        if results.get('basic'):
            print("✅ Basic Operations: Completed")
        
        if results.get('creation'):
            print("✅ Array Creation: Completed")
        
        if results.get('indexing'):
            print("✅ Indexing and Slicing: Completed")
        
        if results.get('math'):
            print("✅ Mathematical Operations: Completed")
        
        if results.get('video'):
            print("✅ Video Processing: Completed")
        
        if results.get('performance'):
            print("✅ Performance Demo: Completed")
        
        if results.get('integration'):
            print("✅ Integration Demo: Completed")
        
        print("\n🎉 NumPy quick starts completed successfully!")
        
    else:
        print("❌ Installation failed - please check your setup")
    
    return results

if __name__ == "__main__":
    # Run all quick starts
    results = run_all_quick_starts()
    
    print("\n🔧 Next Steps:")
    print("1. Explore the NumPy documentation")
    print("2. Read the NUMPY_GUIDE.md for detailed usage")
    print("3. Check numpy_examples.py for more examples")
    print("4. Integrate with your Video-OpusClip workflow") 