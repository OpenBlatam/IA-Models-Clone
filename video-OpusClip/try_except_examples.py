#!/usr/bin/env python3
"""
Try-Except Examples for Video-OpusClip

Practical examples demonstrating comprehensive error handling patterns
for data loading, model inference, and other error-prone operations.
"""

import sys
import os
import time
import traceback
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def setup_example_logging():
    """Setup logging for examples."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('try_except_examples.log')
        ]
    )
    return logging.getLogger(__name__)

# =============================================================================
# EXAMPLE 1: BASIC TRY-EXCEPT PATTERNS
# =============================================================================

def example_basic_try_except():
    """Demonstrate basic try-except patterns."""
    logger = setup_example_logging()
    
    print("\n🔍 Example 1: Basic Try-Except Patterns")
    print("=" * 50)
    
    # Pattern 1: Simple error handling
    def safe_division(a, b):
        try:
            result = a / b
            logger.info(f"✅ Division successful: {a} / {b} = {result}")
            return result
        except ZeroDivisionError as e:
            logger.error(f"❌ Division by zero: {e}")
            return None
        except TypeError as e:
            logger.error(f"❌ Invalid types: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    # Test cases
    test_cases = [
        (10, 2),      # Normal case
        (10, 0),      # Division by zero
        (10, "2"),    # Type error
        (10, None),   # None value
    ]
    
    for a, b in test_cases:
        print(f"Testing: {a} / {b}")
        result = safe_division(a, b)
        print(f"Result: {result}\n")

# =============================================================================
# EXAMPLE 2: DATA LOADING WITH ERROR HANDLING
# =============================================================================

def example_data_loading():
    """Demonstrate data loading with comprehensive error handling."""
    logger = setup_example_logging()
    
    print("\n📊 Example 2: Data Loading with Error Handling")
    print("=" * 50)
    
    def safe_load_file(file_path: str) -> Optional[Dict[str, Any]]:
        """Safely load file with comprehensive error handling."""
        logger.info(f"🔄 Attempting to load: {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Cannot read file: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("File is empty")
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                logger.warning(f"⚠️ Large file: {file_size / (1024**2):.2f}MB")
            
            # Load based on file type
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = {'content': f.read(), 'type': 'text'}
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            logger.info(f"✅ File loaded successfully: {file_path}")
            return data
            
        except FileNotFoundError as e:
            logger.error(f"❌ File not found: {e}")
            return None
        except PermissionError as e:
            logger.error(f"❌ Permission denied: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON: {e}")
            return None
        except UnicodeDecodeError as e:
            logger.error(f"❌ Encoding error: {e}")
            return None
        except MemoryError as e:
            logger.error(f"❌ Memory error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error loading {file_path}: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return None
    
    # Test with various scenarios
    test_files = [
        "config.json",           # Should work if exists
        "nonexistent.json",      # File not found
        "large_file.txt",        # Large file warning
        "invalid.json",          # Invalid JSON
    ]
    
    for file_path in test_files:
        print(f"\nTesting: {file_path}")
        result = safe_load_file(file_path)
        if result:
            print(f"✅ Loaded: {type(result)} with {len(str(result))} characters")
        else:
            print("❌ Failed to load")

# =============================================================================
# EXAMPLE 3: MODEL INFERENCE WITH ERROR HANDLING
# =============================================================================

def example_model_inference():
    """Demonstrate model inference with comprehensive error handling."""
    logger = setup_example_logging()
    
    print("\n🤖 Example 3: Model Inference with Error Handling")
    print("=" * 50)
    
    def safe_model_inference(model_info: Dict[str, Any], input_data: Any) -> Optional[Any]:
        """Safely perform model inference with error handling."""
        logger.info(f"🔄 Starting model inference")
        
        try:
            # Validate model info
            if not model_info or 'model_type' not in model_info:
                raise ValueError("Invalid model information")
            
            # Validate input data
            if input_data is None:
                raise ValueError("Input data is None")
            
            if isinstance(input_data, str) and len(input_data.strip()) == 0:
                raise ValueError("Input data is empty")
            
            # Simulate model inference with potential errors
            model_type = model_info['model_type']
            
            if model_type == 'text_generation':
                # Simulate text generation
                if len(input_data) > 1000:
                    raise ValueError("Input too long for text generation")
                
                result = f"Generated text for: {input_data[:50]}..."
                
            elif model_type == 'video_generation':
                # Simulate video generation
                if len(input_data) > 500:
                    raise ValueError("Input too long for video generation")
                
                # Simulate memory error
                if 'memory_error' in input_data.lower():
                    raise MemoryError("Simulated memory error during video generation")
                
                # Simulate GPU error
                if 'gpu_error' in input_data.lower():
                    raise RuntimeError("CUDA out of memory")
                
                result = f"Generated video for: {input_data[:50]}..."
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"✅ Model inference completed successfully")
            return result
            
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            return None
        except MemoryError as e:
            logger.error(f"❌ Memory error: {e}")
            # Try fallback to CPU
            logger.info("🔄 Attempting CPU fallback...")
            try:
                result = f"CPU fallback result for: {input_data[:50]}..."
                logger.info("✅ CPU fallback successful")
                return result
            except Exception as e2:
                logger.error(f"❌ CPU fallback also failed: {e2}")
                return None
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"❌ GPU error: {e}")
                # Try fallback to CPU
                logger.info("🔄 Attempting CPU fallback...")
                try:
                    result = f"CPU fallback result for: {input_data[:50]}..."
                    logger.info("✅ CPU fallback successful")
                    return result
                except Exception as e2:
                    logger.error(f"❌ CPU fallback also failed: {e2}")
                    return None
            else:
                logger.error(f"❌ Runtime error: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during inference: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return None
    
    # Test with various scenarios
    test_cases = [
        {
            'model_info': {'model_type': 'text_generation'},
            'input_data': 'Normal text input'
        },
        {
            'model_info': {'model_type': 'text_generation'},
            'input_data': 'A' * 2000  # Too long
        },
        {
            'model_info': {'model_type': 'video_generation'},
            'input_data': 'Normal video prompt'
        },
        {
            'model_info': {'model_type': 'video_generation'},
            'input_data': 'memory_error test'  # Simulate memory error
        },
        {
            'model_info': {'model_type': 'video_generation'},
            'input_data': 'gpu_error test'  # Simulate GPU error
        },
        {
            'model_info': {'model_type': 'unsupported'},
            'input_data': 'test'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['model_info']['model_type']} - '{test_case['input_data']}'")
        result = safe_model_inference(test_case['model_info'], test_case['input_data'])
        if result:
            print(f"✅ Success: {result}")
        else:
            print("❌ Failed")

# =============================================================================
# EXAMPLE 4: RETRY PATTERN WITH EXPONENTIAL BACKOFF
# =============================================================================

def example_retry_pattern():
    """Demonstrate retry pattern with exponential backoff."""
    logger = setup_example_logging()
    
    print("\n🔄 Example 4: Retry Pattern with Exponential Backoff")
    print("=" * 50)
    
    def retry_operation(operation_func, max_retries=3, base_delay=1.0):
        """Retry operation with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                result = operation_func()
                if attempt > 0:
                    logger.info(f"✅ Operation succeeded on attempt {attempt + 1}")
                return result
                
            except (ConnectionError, TimeoutError) as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"🔄 Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Operation failed after {max_retries + 1} attempts")
                    raise
            except Exception as e:
                logger.error(f"❌ Non-retryable error: {e}")
                raise
    
    # Simulate unreliable operation
    def unreliable_api_call():
        """Simulate an unreliable API call."""
        import random
        error_type = random.choice(['success', 'connection_error', 'timeout', 'permanent_error'])
        
        if error_type == 'success':
            return "API call successful"
        elif error_type == 'connection_error':
            raise ConnectionError("Connection failed")
        elif error_type == 'timeout':
            raise TimeoutError("Request timed out")
        else:
            raise ValueError("Permanent error")
    
    # Test retry pattern
    print("Testing retry pattern with unreliable API call...")
    for i in range(5):
        print(f"\nTest {i + 1}:")
        try:
            result = retry_operation(unreliable_api_call, max_retries=2, base_delay=0.5)
            print(f"✅ Final result: {result}")
        except Exception as e:
            print(f"❌ All retries failed: {e}")

# =============================================================================
# EXAMPLE 5: RESOURCE MANAGEMENT
# =============================================================================

def example_resource_management():
    """Demonstrate resource management with try-finally."""
    logger = setup_example_logging()
    
    print("\n🔧 Example 5: Resource Management")
    print("=" * 50)
    
    class SimulatedResource:
        """Simulate a resource that needs proper cleanup."""
        
        def __init__(self, name: str):
            self.name = name
            self.is_open = True
            logger.info(f"🔓 Resource '{name}' opened")
        
        def use(self):
            """Use the resource."""
            if not self.is_open:
                raise RuntimeError(f"Resource '{self.name}' is closed")
            logger.info(f"🔧 Using resource '{self.name}'")
            return f"Data from {self.name}"
        
        def close(self):
            """Close the resource."""
            if self.is_open:
                self.is_open = False
                logger.info(f"🔒 Resource '{self.name}' closed")
    
    def safe_resource_operation(resource_name: str):
        """Safely use a resource with proper cleanup."""
        resource = None
        try:
            # Acquire resource
            resource = SimulatedResource(resource_name)
            
            # Use resource
            result = resource.use()
            
            # Simulate potential error
            if 'error' in resource_name.lower():
                raise RuntimeError(f"Simulated error with resource '{resource_name}'")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error using resource '{resource_name}': {e}")
            raise
        finally:
            # Always clean up resource
            if resource:
                try:
                    resource.close()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to close resource '{resource_name}': {e}")
    
    # Test resource management
    test_resources = [
        "database_connection",
        "file_handle",
        "error_resource",  # Will cause error
        "network_connection"
    ]
    
    for resource_name in test_resources:
        print(f"\nTesting resource: {resource_name}")
        try:
            result = safe_resource_operation(resource_name)
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Failed: {e}")

# =============================================================================
# EXAMPLE 6: CONTEXT MANAGERS
# =============================================================================

def example_context_managers():
    """Demonstrate context managers for error handling."""
    logger = setup_example_logging()
    
    print("\n📦 Example 6: Context Managers")
    print("=" * 50)
    
    class SafeOperationContext:
        """Context manager for safe operations."""
        
        def __init__(self, operation_name: str):
            self.operation_name = operation_name
            self.start_time = None
            self.success = False
        
        def __enter__(self):
            self.start_time = time.time()
            logger.info(f"🚀 Starting {self.operation_name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            
            if exc_type is None:
                self.success = True
                logger.info(f"✅ {self.operation_name} completed successfully in {duration:.2f}s")
            else:
                logger.error(f"❌ {self.operation_name} failed after {duration:.2f}s: {exc_val}")
            
            return False  # Don't suppress exceptions
    
    class MemoryMonitor:
        """Context manager for memory monitoring."""
        
        def __init__(self, threshold_mb: int = 100):
            self.threshold_mb = threshold_mb
            self.initial_memory = None
        
        def __enter__(self):
            import psutil
            process = psutil.Process()
            self.initial_memory = process.memory_info().rss
            logger.debug(f"Memory at start: {self.initial_memory / (1024**2):.2f}MB")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            import psutil
            process = psutil.Process()
            final_memory = process.memory_info().rss
            memory_used = final_memory - self.initial_memory
            
            logger.debug(f"Memory at end: {final_memory / (1024**2):.2f}MB (+{memory_used / (1024**2):.2f}MB)")
            
            if memory_used > self.threshold_mb * 1024 * 1024:
                logger.warning(f"⚠️ High memory usage: {memory_used / (1024**2):.2f}MB")
            
            return False
    
    def process_data_with_context(data: str):
        """Process data using context managers."""
        with SafeOperationContext("data_processing"):
            with MemoryMonitor(threshold_mb=50):
                # Simulate data processing
                if 'error' in data.lower():
                    raise ValueError("Simulated processing error")
                
                # Simulate memory usage
                large_list = [i for i in range(1000000)]  # Use some memory
                
                result = f"Processed: {data}"
                return result
    
    # Test context managers
    test_data = [
        "normal data",
        "data with error",
        "large data processing"
    ]
    
    for data in test_data:
        print(f"\nProcessing: {data}")
        try:
            result = process_data_with_context(data)
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")

# =============================================================================
# EXAMPLE 7: ERROR RECOVERY STRATEGIES
# =============================================================================

def example_error_recovery():
    """Demonstrate error recovery strategies."""
    logger = setup_example_logging()
    
    print("\n🛡️ Example 7: Error Recovery Strategies")
    print("=" * 50)
    
    def robust_video_processing(video_path: str, prompt: str):
        """Robust video processing with multiple recovery strategies."""
        
        strategies = [
            # Strategy 1: High quality processing
            lambda: f"High quality video generated from {video_path} with prompt: {prompt}",
            
            # Strategy 2: Medium quality processing
            lambda: f"Medium quality video generated from {video_path} with prompt: {prompt}",
            
            # Strategy 3: Low quality processing
            lambda: f"Low quality video generated from {video_path} with prompt: {prompt}",
            
            # Strategy 4: Fallback processing
            lambda: f"Fallback video generated from {video_path} with prompt: {prompt}"
        ]
        
        strategy_names = [
            "High Quality",
            "Medium Quality", 
            "Low Quality",
            "Fallback"
        ]
        
        for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
            try:
                logger.info(f"🔄 Trying strategy {i + 1}: {name}")
                
                # Simulate potential errors
                if 'error' in prompt.lower() and i < 2:
                    raise RuntimeError(f"Simulated error in {name} strategy")
                
                result = strategy()
                logger.info(f"✅ Strategy {i + 1} succeeded: {name}")
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ Strategy {i + 1} failed ({name}): {e}")
                if i == len(strategies) - 1:
                    logger.error("❌ All strategies failed")
                    raise
        
        return None
    
    # Test error recovery
    test_cases = [
        ("video1.mp4", "Normal prompt"),
        ("video2.mp4", "error prompt"),  # Will cause errors in first strategies
        ("video3.mp4", "Another normal prompt")
    ]
    
    for video_path, prompt in test_cases:
        print(f"\nProcessing: {video_path} with prompt: '{prompt}'")
        try:
            result = robust_video_processing(video_path, prompt)
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ All strategies failed: {e}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run all try-except examples."""
    print("🐛 Try-Except Examples for Video-OpusClip")
    print("=" * 60)
    print("This script demonstrates comprehensive error handling patterns")
    print("for data loading, model inference, and other operations.\n")
    
    # Run all examples
    examples = [
        example_basic_try_except,
        example_data_loading,
        example_model_inference,
        example_retry_pattern,
        example_resource_management,
        example_context_managers,
        example_error_recovery
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print("\n" + "="*60 + "\n")
    
    print("🎉 All examples completed!")
    print("\n📋 Summary:")
    print("• Basic try-except patterns for error handling")
    print("• Data loading with validation and error recovery")
    print("• Model inference with GPU/CPU fallbacks")
    print("• Retry patterns with exponential backoff")
    print("• Resource management with proper cleanup")
    print("• Context managers for safe operations")
    print("• Error recovery strategies for robust processing")

if __name__ == "__main__":
    main() 