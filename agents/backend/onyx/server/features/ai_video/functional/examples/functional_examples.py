from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Callable, List, Dict, Any, Optional, Tuple
from functools import reduce, partial
import torch
import numpy as np
import logging
        import torch.nn.functional as F
    from typing import Union, Tuple
            import time
        import time
        import time
from typing import Any, List, Dict, Optional
import asyncio
"""
Functional Programming Examples for AI Video
===========================================

Comprehensive examples demonstrating functional programming patterns.
"""


logger = logging.getLogger(__name__)

def compose(*functions) -> Any:
    """Compose multiple functions: compose(f, g, h)(x) = f(g(h(x)))."""
    def inner(arg) -> Any:
        return reduce(lambda acc, f: f(acc), reversed(functions), arg)
    return inner

# ============================================================================
# Example 1: Data Processing Pipeline
# ============================================================================

def example_data_pipeline():
    """Example of functional data processing pipeline."""
    
    # Pure functions for data processing
    def load_video(path: str) -> torch.Tensor:
        return torch.randn(16, 3, 512, 512)
    
    def resize_frames(frames: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(frames, size=size, mode='bilinear')
    
    def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
        return frames / 255.0
    
    def add_noise(frames: torch.Tensor, noise_level: float) -> torch.Tensor:
        noise = torch.randn_like(frames) * noise_level
        return frames + noise
    
    # Create pipeline using composition
    pipeline = compose(
        lambda x: resize_frames(x, (256, 256)),
        normalize_frames,
        partial(add_noise, noise_level=0.1)
    )
    
    # Process data
    video_data = load_video("example.mp4")
    processed_data = pipeline(video_data)
    
    print(f"Original shape: {video_data.shape}")
    print(f"Processed shape: {processed_data.shape}")
    
    return processed_data

# ============================================================================
# Example 2: Model Training with Functional Patterns
# ============================================================================

def example_functional_training():
    """Example of functional training approach."""
    
    # Pure functions for training
    def create_model() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=lr)
    
    def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)
    
    def training_step(model: torch.nn.Module, 
                     optimizer: torch.optim.Optimizer,
                     batch: torch.Tensor, 
                     target: torch.Tensor) -> float:
        optimizer.zero_grad()
        pred = model(batch)
        loss = mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Create training pipeline
    def create_training_pipeline(model: torch.nn.Module, 
                               optimizer: torch.optim.Optimizer) -> Callable:
        return partial(training_step, model, optimizer)
    
    # Setup
    model = create_model()
    optimizer = create_optimizer(model, lr=1e-4)
    training_fn = create_training_pipeline(model, optimizer)
    
    # Generate dummy data
    batch = torch.randn(4, 3, 256, 256)
    target = torch.randn(4, 3, 256, 256)
    
    # Training loop
    for epoch in range(5):
        loss = training_fn(batch, target)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    return model

# ============================================================================
# Example 3: API Response Building
# ============================================================================

def example_api_responses():
    """Example of functional API response building."""
    
    # Pure functions for response building
    def create_success_response(data: Any) -> Dict[str, Any]:
        return {
            'success': True,
            'data': data,
            'timestamp': '2024-01-01T00:00:00Z'
        }
    
    def create_error_response(error: str, code: int) -> Dict[str, Any]:
        return {
            'success': False,
            'error': error,
            'code': code,
            'timestamp': '2024-01-01T00:00:00Z'
        }
    
    def add_metadata(response: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {**response, 'metadata': metadata}
    
    def add_pagination(response: Dict[str, Any], page: int, total: int) -> Dict[str, Any]:
        return {**response, 'pagination': {'page': page, 'total': total}}
    
    # Compose response builders
    success_with_metadata = compose(
        partial(add_metadata, metadata={'version': '1.0.0'}),
        create_success_response
    )
    
    success_with_pagination = compose(
        partial(add_pagination, page=1, total=100),
        create_success_response
    )
    
    # Example usage
    video_data = {'id': '123', 'url': 'video.mp4'}
    
    response1 = success_with_metadata(video_data)
    response2 = success_with_pagination(video_data)
    error_response = create_error_response("Video not found", 404)
    
    print("Response 1:", response1)
    print("Response 2:", response2)
    print("Error Response:", error_response)
    
    return response1, response2, error_response

# ============================================================================
# Example 4: Configuration Management
# ============================================================================

def example_config_management():
    """Example of functional configuration management."""
    
    # Pure functions for config processing
    def load_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        return config_dict.copy()
    
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = ['model_name', 'batch_size', 'learning_rate']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return config
    
    def set_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            'device': 'cuda',
            'num_epochs': 100,
            'save_interval': 10
        }
        return {**defaults, **config}
    
    def override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate environment variable overrides
        env_overrides = {
            'batch_size': 32,  # Would come from os.environ
            'device': 'cpu'    # Would come from os.environ
        }
        return {**config, **env_overrides}
    
    # Create config pipeline
    config_pipeline = compose(
        override_with_env,
        set_defaults,
        validate_config,
        load_config
    )
    
    # Example config
    raw_config = {
        'model_name': 'video_diffusion',
        'batch_size': 16,
        'learning_rate': 1e-4
    }
    
    processed_config = config_pipeline(raw_config)
    print("Processed Config:", processed_config)
    
    return processed_config

# ============================================================================
# Example 5: Error Handling with Functional Patterns
# ============================================================================

def example_error_handling():
    """Example of functional error handling."""
    
    
    Result = Union[Tuple[None, Exception], Tuple[Any, None]]
    
    def safe_divide(a: float, b: float) -> Result:
        try:
            return a / b, None
        except Exception as e:
            return None, e
    
    def safe_sqrt(x: float) -> Result:
        try:
            return np.sqrt(x), None
        except Exception as e:
            return None, e
    
    def map_result(func: Callable[[Any], Any], result: Result) -> Result:
        if result[1] is not None:
            return None, result[1]
        try:
            return func(result[0]), None
        except Exception as e:
            return None, e
    
    def bind_result(func: Callable[[Any], Result], result: Result) -> Result:
        if result[1] is not None:
            return None, result[1]
        return func(result[0])
    
    # Example usage
    result1 = safe_divide(10, 2)
    result2 = map_result(lambda x: x * 2, result1)
    result3 = bind_result(safe_sqrt, result2)
    
    print("Result 1:", result1)
    print("Result 2:", result2)
    print("Result 3:", result3)
    
    # Error case
    error_result = safe_divide(10, 0)
    print("Error Result:", error_result)
    
    return result1, result2, result3, error_result

# ============================================================================
# Example 6: Caching and Memoization
# ============================================================================

def example_caching():
    """Example of functional caching patterns."""
    
    def memoize(func: Callable) -> Callable:
        cache = {}
        
        def wrapper(*args, **kwargs) -> Any:
            key = str(args) + str(sorted(kwargs.items()))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        
        return wrapper
    
    def cache_with_ttl(ttl_seconds: int):
        
    """cache_with_ttl function."""
def decorator(func: Callable) -> Callable:
            cache = {}
            
            def wrapper(*args, **kwargs) -> Any:
                key = str(args) + str(sorted(kwargs.items()))
                now = time.time()
                
                if key in cache:
                    result, timestamp = cache[key]
                    if now - timestamp < ttl_seconds:
                        return result
                
                result = func(*args, **kwargs)
                cache[key] = (result, now)
                return result
            
            return wrapper
        return decorator
    
    # Example expensive function
    @memoize
    def expensive_computation(x: int) -> int:
        time.sleep(0.1)  # Simulate expensive computation
        return x * x
    
    @cache_with_ttl(5)  # 5 second TTL
    async def expensive_api_call(user_id: str) -> Dict[str, Any]:
        time.sleep(0.1)  # Simulate API call
        return {'user_id': user_id, 'data': 'some_data'}
    
    # Test memoization
    print("First call (slow):", expensive_computation(5))
    print("Second call (fast):", expensive_computation(5))
    
    # Test TTL caching
    print("First API call (slow):", expensive_api_call("user123"))
    print("Second API call (fast):", expensive_api_call("user123"))
    
    return expensive_computation, expensive_api_call

# ============================================================================
# Example 7: Data Transformation Pipelines
# ============================================================================

def example_data_transformation():
    """Example of functional data transformation pipelines."""
    
    # Pure transformation functions
    def filter_videos(videos: List[Dict], min_duration: float) -> List[Dict]:
        return [v for v in videos if v.get('duration', 0) >= min_duration]
    
    def sort_videos(videos: List[Dict], key: str, reverse: bool = False) -> List[Dict]:
        return sorted(videos, key=lambda x: x.get(key, 0), reverse=reverse)
    
    def map_video_metadata(videos: List[Dict]) -> List[Dict]:
        return [{
            'id': v.get('id'),
            'title': v.get('title', 'Untitled'),
            'duration': v.get('duration', 0),
            'size_mb': v.get('size', 0) / (1024 * 1024)
        } for v in videos]
    
    def group_by_category(videos: List[Dict]) -> Dict[str, List[Dict]]:
        grouped = {}
        for video in videos:
            category = video.get('category', 'unknown')
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(video)
        return grouped
    
    # Create transformation pipeline
    video_processing_pipeline = compose(
        partial(filter_videos, min_duration=60.0),
        partial(sort_videos, key='duration', reverse=True),
        map_video_metadata
    )
    
    # Example data
    videos = [
        {'id': '1', 'title': 'Video 1', 'duration': 120, 'size': 50*1024*1024, 'category': 'tutorial'},
        {'id': '2', 'title': 'Video 2', 'duration': 30, 'size': 20*1024*1024, 'category': 'short'},
        {'id': '3', 'title': 'Video 3', 'duration': 180, 'size': 80*1024*1024, 'category': 'tutorial'},
    ]
    
    # Process data
    processed_videos = video_processing_pipeline(videos)
    grouped_videos = group_by_category(videos)
    
    print("Processed Videos:", processed_videos)
    print("Grouped Videos:", grouped_videos)
    
    return processed_videos, grouped_videos

# ============================================================================
# Example 8: Testing with Functional Patterns
# ============================================================================

def example_functional_testing():
    """Example of functional testing patterns."""
    
    def test_function(func: Callable, test_cases: List[Tuple[Any, Any]]) -> List[bool]:
        """Test function with multiple test cases."""
        results = []
        for input_data, expected_output in test_cases:
            try:
                actual_output = func(input_data)
                result = actual_output == expected_output
                results.append(result)
                print(f"Test: {func.__name__}({input_data}) = {actual_output} (Expected: {expected_output}) -> {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results.append(False)
                print(f"Test: {func.__name__}({input_data}) -> ERROR: {e}")
        return results
    
    def property_based_test(func: Callable, property_func: Callable, test_data: List[Any]) -> List[bool]:
        """Property-based testing."""
        results = []
        for data in test_data:
            try:
                result = func(data)
                property_holds = property_func(result)
                results.append(property_holds)
                print(f"Property test: {property_func.__name__}({func.__name__}({data})) = {property_holds}")
            except Exception as e:
                results.append(False)
                print(f"Property test: {func.__name__}({data}) -> ERROR: {e}")
        return results
    
    # Test functions
    def add_one(x: int) -> int:
        return x + 1
    
    def is_positive(x: int) -> bool:
        return x > 0
    
    # Test cases
    test_cases = [(1, 2), (5, 6), (-1, 0)]
    test_data = [1, 5, -1, 0]
    
    # Run tests
    unit_test_results = test_function(add_one, test_cases)
    property_test_results = property_based_test(add_one, is_positive, test_data)
    
    print(f"Unit tests passed: {sum(unit_test_results)}/{len(unit_test_results)}")
    print(f"Property tests passed: {sum(property_test_results)}/{len(property_test_results)}")
    
    return unit_test_results, property_test_results

# ============================================================================
# Main Example Runner
# ============================================================================

def run_all_examples():
    """Run all functional programming examples."""
    print("=" * 60)
    print("FUNCTIONAL PROGRAMMING EXAMPLES")
    print("=" * 60)
    
    print("\n1. Data Processing Pipeline:")
    example_data_pipeline()
    
    print("\n2. Functional Training:")
    example_functional_training()
    
    print("\n3. API Response Building:")
    example_api_responses()
    
    print("\n4. Configuration Management:")
    example_config_management()
    
    print("\n5. Error Handling:")
    example_error_handling()
    
    print("\n6. Caching and Memoization:")
    example_caching()
    
    print("\n7. Data Transformation:")
    example_data_transformation()
    
    print("\n8. Functional Testing:")
    example_functional_testing()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)

match __name__:
    case "__main__":
    run_all_examples() 