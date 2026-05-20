from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import numpy as np
from PIL import Image
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
from typing import Any, List, Dict, Optional
import asyncio
"""
🛡️ Error Handling Examples with Try-Except Blocks
=================================================

This file demonstrates comprehensive error handling with try-except blocks
for data loading and model inference operations in the Gradio app.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustDataLoader:
    """Enhanced data loader with comprehensive error handling."""
    
    def __init__(self, max_workers: int = 4, timeout: int = 30):
        
    """__init__ function."""
self.max_workers = max_workers
        self.timeout = timeout
        self.error_count = 0
        self.success_count = 0
    
    def load_image_safe(self, image_path: str) -> Optional[Image.Image]:
        """Safely load an image with comprehensive error handling."""
        try:
            # Validate input
            if not image_path or not isinstance(image_path, str):
                logger.warning(f"Invalid image path: {image_path}")
                return None
            
            # Load image with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(Image.open, image_path)
                try:
                    image = future.result(timeout=self.timeout)
                except TimeoutError:
                    logger.error(f"Timeout loading image: {image_path}")
                    return None
            
            # Validate loaded image
            if not image:
                logger.warning(f"Failed to load image: {image_path}")
                return None
            
            # Check image dimensions
            if image.size[0] == 0 or image.size[1] == 0:
                logger.warning(f"Image has zero dimensions: {image_path}")
                return None
            
            # Convert to RGB if needed
            try:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Failed to convert image to RGB: {image_path}, error: {e}")
                return None
            
            self.success_count += 1
            return image
            
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
        except PermissionError:
            logger.error(f"Permission denied accessing image: {image_path}")
        except Exception as e:
            logger.error(f"Unexpected error loading image {image_path}: {e}")
        
        self.error_count += 1
        return None
    
    def load_images_batch(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple images with parallel processing and error handling."""
        if not image_paths:
            logger.warning("No image paths provided")
            return []
        
        loaded_images = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all image loading tasks
                future_to_path = {
                    executor.submit(self.load_image_safe, path): path 
                    for path in image_paths
                }
                
                # Collect results with error handling
                for future in future_to_path:
                    path = future_to_path[future]
                    try:
                        image = future.result(timeout=self.timeout)
                        if image is not None:
                            loaded_images.append(image)
                        else:
                            logger.warning(f"Failed to load image: {path}")
                    except TimeoutError:
                        logger.error(f"Timeout loading image: {path}")
                    except Exception as e:
                        logger.error(f"Error loading image {path}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to create thread pool for image loading: {e}")
            # Fallback to sequential loading
            for path in image_paths:
                try:
                    image = self.load_image_safe(path)
                    if image is not None:
                        loaded_images.append(image)
                except Exception as e:
                    logger.error(f"Sequential loading failed for {path}: {e}")
        
        logger.info(f"Successfully loaded {len(loaded_images)}/{len(image_paths)} images")
        return loaded_images

class RobustModelInference:
    """Enhanced model inference with comprehensive error handling."""
    
    def __init__(self, model, device: str = 'auto'):
        
    """__init__ function."""
self.model = model
        self.device = self._setup_device(device)
        self.error_count = 0
        self.success_count = 0
    
    def _setup_device(self, device: str) -> torch.device:
        """Safely setup device with error handling."""
        try:
            if device == 'auto':
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                else:
                    device = torch.device('cpu')
                    logger.info("Using CPU device")
            else:
                device = torch.device(device)
            
            return device
        except Exception as e:
            logger.error(f"Failed to setup device {device}: {e}")
            return torch.device('cpu')
    
    def preprocess_input_safe(self, input_data: Any) -> Optional[torch.Tensor]:
        """Safely preprocess input data with comprehensive error handling."""
        try:
            # Handle different input types
            if isinstance(input_data, torch.Tensor):
                tensor = input_data
            elif isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, list):
                # Handle list of images or tensors
                if all(isinstance(item, torch.Tensor) for item in input_data):
                    tensor = torch.stack(input_data)
                elif all(isinstance(item, np.ndarray) for item in input_data):
                    tensor = torch.stack([torch.from_numpy(item) for item in input_data])
                else:
                    raise ValueError("List contains mixed or unsupported types")
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Validate tensor
            if torch.isnan(tensor).any():
                raise ValueError("Input contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError("Input contains Inf values")
            
            # Move to device
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Input preprocessing failed: {e}")
            return None
    
    def inference_safe(self, input_data: Any, **kwargs) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Safely perform model inference with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Check system resources before inference
            self._check_system_resources()
            
            # Preprocess input
            processed_input = self.preprocess_input_safe(input_data)
            if processed_input is None:
                return None, "Input preprocessing failed"
            
            # Set model to evaluation mode
            try:
                self.model.eval()
            except Exception as e:
                logger.warning(f"Failed to set model to eval mode: {e}")
            
            # Perform inference with gradient disabled
            with torch.no_grad():
                try:
                    # Check for CUDA memory before inference
                    if self.device.type == 'cuda':
                        memory_before = torch.cuda.memory_allocated()
                        if memory_before > torch.cuda.max_memory_allocated() * 0.9:
                            logger.warning("High GPU memory usage before inference")
                    
                    # Perform inference
                    output = self.model(processed_input, **kwargs)
                    
                    # Validate output
                    if output is None:
                        raise ValueError("Model returned None output")
                    
                    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                        raise ValueError("Model output contains NaN values")
                    
                    inference_time = time.time() - start_time
                    logger.info(f"Inference completed in {inference_time:.2f}s")
                    
                    self.success_count += 1
                    return output, None
                    
                except torch.cuda.OutOfMemoryError as e:
                    error_msg = f"GPU out of memory during inference: {e}"
                    logger.error(error_msg)
                    self._handle_memory_error()
                    return None, error_msg
                    
                except Exception as e:
                    error_msg = f"Inference failed: {e}"
                    logger.error(error_msg)
                    return None, error_msg
                    
        except Exception as e:
            error_msg = f"Unexpected error during inference: {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, error_msg
        
        finally:
            self.error_count += 1
    
    def _check_system_resources(self) -> Any:
        """Check system resources before inference."""
        try:
            # Check CPU memory
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"High CPU memory usage: {memory_percent}%")
            
            # Check GPU memory if available
            if self.device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if gpu_memory > 0.9:
                    logger.warning(f"High GPU memory usage: {gpu_memory:.1%}")
                    
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
    
    def _handle_memory_error(self) -> Any:
        """Handle memory errors by clearing cache."""
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache after memory error")
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {e}")

class RobustPipeline:
    """Complete pipeline with comprehensive error handling."""
    
    def __init__(self, model, data_loader: RobustDataLoader, inference: RobustModelInference):
        
    """__init__ function."""
self.model = model
        self.data_loader = data_loader
        self.inference = inference
        self.pipeline_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'errors': []
        }
    
    def process_batch(self, image_paths: List[str], **kwargs) -> Dict[str, Any]:
        """Process a batch of images with comprehensive error handling."""
        self.pipeline_stats['total_operations'] += 1
        start_time = time.time()
        
        try:
            # Load images
            logger.info(f"Loading {len(image_paths)} images...")
            images = self.data_loader.load_images_batch(image_paths)
            
            if not images:
                error_msg = "No images were successfully loaded"
                self._record_error(error_msg)
                return self._create_error_response(error_msg)
            
            # Perform inference
            logger.info(f"Performing inference on {len(images)} images...")
            output, error = self.inference.inference_safe(images, **kwargs)
            
            if output is None:
                self._record_error(error)
                return self._create_error_response(error)
            
            # Process results
            processing_time = time.time() - start_time
            self.pipeline_stats['successful_operations'] += 1
            
            return {
                'status': 'success',
                'output': output,
                'processing_time': processing_time,
                'images_processed': len(images),
                'total_images_requested': len(image_paths),
                'success_rate': len(images) / len(image_paths)
            }
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._record_error(error_msg)
            return self._create_error_response(error_msg)
    
    def _record_error(self, error_msg: str):
        """Record error for statistics."""
        self.pipeline_stats['failed_operations'] += 1
        self.pipeline_stats['errors'].append({
            'timestamp': time.time(),
            'error': error_msg
        })
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'status': 'error',
            'error': error_msg,
            'output': None,
            'processing_time': 0,
            'images_processed': 0,
            'total_images_requested': 0,
            'success_rate': 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = self.pipeline_stats['total_operations']
        success = self.pipeline_stats['successful_operations']
        failed = self.pipeline_stats['failed_operations']
        
        return {
            'total_operations': total,
            'successful_operations': success,
            'failed_operations': failed,
            'success_rate': success / total if total > 0 else 0,
            'error_rate': failed / total if total > 0 else 0,
            'recent_errors': self.pipeline_stats['errors'][-5:],  # Last 5 errors
            'data_loader_stats': {
                'success_count': self.data_loader.success_count,
                'error_count': self.data_loader.error_count
            },
            'inference_stats': {
                'success_count': self.inference.success_count,
                'error_count': self.inference.error_count
            }
        }

def demonstrate_error_handling():
    """Demonstrate the error handling capabilities."""
    print("🚀 Demonstrating Comprehensive Error Handling")
    print("=" * 50)
    
    # Create components
    data_loader = RobustDataLoader(max_workers=2, timeout=10)
    inference = RobustModelInference(model=None, device='cpu')  # Dummy model
    pipeline = RobustPipeline(model=None, data_loader=data_loader, inference=inference)
    
    # Test data loading with various scenarios
    print("\n📁 Testing Data Loading Error Handling:")
    
    # Valid image paths (these won't exist, but will test error handling)
    test_paths = [
        "valid_image.jpg",
        "nonexistent_image.png",
        "permission_denied.jpg",
        "corrupted_image.bmp",
        "valid_image2.jpg"
    ]
    
    # Test image loading
    images = data_loader.load_images_batch(test_paths)
    print(f"   Loaded {len(images)}/{len(test_paths)} images")
    print(f"   Success count: {data_loader.success_count}")
    print(f"   Error count: {data_loader.error_count}")
    
    # Test input preprocessing
    print("\n🔧 Testing Input Preprocessing Error Handling:")
    
    test_inputs = [
        torch.randn(3, 224, 224),  # Valid tensor
        np.random.randn(3, 224, 224),  # Valid numpy array
        [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],  # Valid list
        None,  # Invalid input
        "invalid_input",  # Invalid input
        torch.tensor([float('nan')]),  # NaN values
        torch.tensor([float('inf')])  # Inf values
    ]
    
    for i, test_input in enumerate(test_inputs):
        try:
            result = inference.preprocess_input_safe(test_input)
            if result is not None:
                print(f"   Input {i}: ✅ Success")
            else:
                print(f"   Input {i}: ❌ Failed")
        except Exception as e:
            print(f"   Input {i}: ❌ Error - {e}")
    
    # Test pipeline processing
    print("\n🔄 Testing Pipeline Error Handling:")
    
    # Test with various scenarios
    test_scenarios = [
        [],  # Empty list
        ["nonexistent1.jpg", "nonexistent2.jpg"],  # All invalid
        ["valid1.jpg", "nonexistent.jpg", "valid2.jpg"],  # Mixed
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n   Scenario {i + 1}: {len(scenario)} images")
        result = pipeline.process_batch(scenario)
        print(f"   Status: {result['status']}")
        if result['status'] == 'error':
            print(f"   Error: {result['error']}")
        else:
            print(f"   Success rate: {result['success_rate']:.1%}")
    
    # Show final statistics
    print("\n📊 Final Statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Error handling demonstration completed!")

match __name__:
    case "__main__":
    demonstrate_error_handling() 