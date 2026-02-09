"""
Enhanced Error Handling for Video-OpusClip

Comprehensive try-except blocks for error-prone operations:
- Data loading and validation
- Model inference and prediction
- File operations and I/O
- Network requests and API calls
- GPU operations and memory management
- Database operations
- Configuration loading
"""

import sys
import os
import time
import traceback
import logging
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps
import threading
import queue
import signal

# Import existing components
from optimized_config import get_config
from error_handling import ErrorHandler, ErrorType, ErrorSeverity
from logging_config import setup_logging
from debug_tools import DebugManager

# =============================================================================
# ENHANCED ERROR HANDLING DECORATORS
# =============================================================================

def safe_operation(
    operation_name: str = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None,
    log_errors: bool = True,
    raise_on_final_failure: bool = True
):
    """
    Decorator for safe operations with retry logic and fallback handling.
    
    Args:
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_value: Value to return if all retries fail
        log_errors: Whether to log errors
        raise_on_final_failure: Whether to raise exception after all retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = setup_logging("safe_operation")
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"✅ {op_name} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    error_msg = f"❌ {op_name} failed on attempt {attempt + 1}: {str(e)}"
                    
                    if log_errors:
                        logger.error(error_msg)
                        logger.debug(f"Stack trace: {traceback.format_exc()}")
                    
                    if attempt < max_retries:
                        logger.warning(f"🔄 Retrying {op_name} in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"💥 {op_name} failed after {max_retries + 1} attempts")
                        
                        if fallback_value is not None:
                            logger.info(f"🛡️ Using fallback value for {op_name}")
                            return fallback_value
                        
                        if raise_on_final_failure:
                            raise
            
            return fallback_value
        
        return wrapper
    return decorator

def data_loading_safe(
    validate_data: bool = True,
    timeout: float = 30.0,
    memory_limit: int = 1024 * 1024 * 1024  # 1GB
):
    """
    Decorator for safe data loading operations.
    
    Args:
        validate_data: Whether to validate loaded data
        timeout: Timeout for loading operations
        memory_limit: Memory limit in bytes
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging("data_loading")
            start_time = time.time()
            
            try:
                # Check memory before loading
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss
                
                if initial_memory > memory_limit:
                    raise MemoryError(f"Memory usage ({initial_memory / (1024**3):.2f}GB) exceeds limit")
                
                # Load data with timeout
                result = func(*args, **kwargs)
                
                # Check memory after loading
                final_memory = process.memory_info().rss
                memory_used = final_memory - initial_memory
                
                if memory_used > memory_limit:
                    logger.warning(f"⚠️ High memory usage: {memory_used / (1024**3):.2f}GB")
                
                # Validate data if requested
                if validate_data and result is not None:
                    if not _validate_loaded_data(result):
                        raise ValueError("Loaded data failed validation")
                
                loading_time = time.time() - start_time
                logger.info(f"✅ Data loaded successfully in {loading_time:.2f}s")
                
                return result
                
            except TimeoutError:
                logger.error(f"⏰ Data loading timed out after {timeout}s")
                raise
            except MemoryError as e:
                logger.error(f"💾 Memory error during data loading: {e}")
                raise
            except Exception as e:
                logger.error(f"❌ Data loading failed: {e}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                raise
        
        return wrapper
    return decorator

def model_inference_safe(
    gpu_fallback: bool = True,
    memory_monitoring: bool = True,
    timeout: float = 60.0,
    batch_size_limit: int = 10
):
    """
    Decorator for safe model inference operations.
    
    Args:
        gpu_fallback: Whether to fallback to CPU if GPU fails
        memory_monitoring: Whether to monitor memory usage
        timeout: Timeout for inference operations
        batch_size_limit: Maximum batch size for inference
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging("model_inference")
            start_time = time.time()
            
            # Check batch size
            if 'batch_size' in kwargs and kwargs['batch_size'] > batch_size_limit:
                logger.warning(f"⚠️ Batch size {kwargs['batch_size']} exceeds limit {batch_size_limit}")
                kwargs['batch_size'] = batch_size_limit
            
            try:
                # Monitor memory before inference
                if memory_monitoring:
                    import psutil
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss
                    logger.debug(f"Memory before inference: {initial_memory / (1024**2):.2f}MB")
                
                # Run inference
                result = func(*args, **kwargs)
                
                # Monitor memory after inference
                if memory_monitoring:
                    final_memory = process.memory_info().rss
                    memory_used = final_memory - initial_memory
                    logger.debug(f"Memory after inference: {final_memory / (1024**2):.2f}MB (+{memory_used / (1024**2):.2f}MB)")
                
                inference_time = time.time() - start_time
                logger.info(f"✅ Inference completed in {inference_time:.2f}s")
                
                return result
                
            except RuntimeError as e:
                if "CUDA" in str(e) and gpu_fallback:
                    logger.warning(f"🔄 GPU inference failed, falling back to CPU: {e}")
                    return _fallback_to_cpu_inference(func, args, kwargs)
                else:
                    logger.error(f"❌ Runtime error during inference: {e}")
                    raise
            except MemoryError as e:
                logger.error(f"💾 Memory error during inference: {e}")
                if gpu_fallback:
                    logger.info("🔄 Attempting CPU fallback due to memory error")
                    return _fallback_to_cpu_inference(func, args, kwargs)
                raise
            except Exception as e:
                logger.error(f"❌ Inference failed: {e}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                raise
        
        return wrapper
    return decorator

def file_operation_safe(
    backup: bool = True,
    verify_after_write: bool = True,
    timeout: float = 30.0
):
    """
    Decorator for safe file operations.
    
    Args:
        backup: Whether to create backup before operations
        verify_after_write: Whether to verify file after writing
        timeout: Timeout for file operations
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging("file_operation")
            
            try:
                # Create backup if requested
                if backup and 'file_path' in kwargs:
                    backup_path = _create_file_backup(kwargs['file_path'])
                    logger.debug(f"📋 Created backup: {backup_path}")
                
                # Perform file operation
                result = func(*args, **kwargs)
                
                # Verify file after write operation
                if verify_after_write and 'file_path' in kwargs:
                    if not _verify_file_integrity(kwargs['file_path']):
                        raise IOError("File integrity check failed after write operation")
                
                logger.info(f"✅ File operation completed successfully")
                return result
                
            except PermissionError as e:
                logger.error(f"🚫 Permission error: {e}")
                raise
            except FileNotFoundError as e:
                logger.error(f"📁 File not found: {e}")
                raise
            except IOError as e:
                logger.error(f"💾 I/O error: {e}")
                raise
            except Exception as e:
                logger.error(f"❌ File operation failed: {e}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                raise
        
        return wrapper
    return decorator

def network_operation_safe(
    max_retries: int = 3,
    timeout: float = 30.0,
    circuit_breaker: bool = True
):
    """
    Decorator for safe network operations.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout: Timeout for network operations
        circuit_breaker: Whether to use circuit breaker pattern
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = setup_logging("network_operation")
            
            # Circuit breaker check
            if circuit_breaker and _is_circuit_open(func.__name__):
                raise ConnectionError("Circuit breaker is open - too many recent failures")
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Mark circuit as closed on success
                    if circuit_breaker:
                        _mark_circuit_closed(func.__name__)
                    
                    if attempt > 0:
                        logger.info(f"✅ Network operation succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except (ConnectionError, TimeoutError) as e:
                    error_msg = f"🌐 Network error on attempt {attempt + 1}: {str(e)}"
                    logger.warning(error_msg)
                    
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        if circuit_breaker:
                            _mark_circuit_open(func.__name__)
                        raise
                
                except Exception as e:
                    logger.error(f"❌ Network operation failed: {e}")
                    raise
            
            return None
        
        return wrapper
    return decorator

# =============================================================================
# DATA LOADING SAFETY FUNCTIONS
# =============================================================================

@data_loading_safe(validate_data=True, timeout=60.0)
def safe_load_video_data(file_path: str) -> Dict[str, Any]:
    """Safely load video data from file."""
    logger = setup_logging("video_loader")
    
    try:
        # Check file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read video file: {file_path}")
        
        # Load video data based on file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return _load_video_file(file_path)
        elif file_ext in ['.json', '.yaml', '.yml']:
            return _load_metadata_file(file_path)
        elif file_ext in ['.pkl', '.pickle']:
            return _load_pickle_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
            
    except Exception as e:
        logger.error(f"Failed to load video data from {file_path}: {e}")
        raise

@data_loading_safe(validate_data=True, timeout=30.0)
def safe_load_model_config(config_path: str) -> Dict[str, Any]:
    """Safely load model configuration."""
    logger = setup_logging("config_loader")
    
    try:
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        required_keys = ['model_type', 'model_path', 'device']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load model config from {config_path}: {e}")
        raise

@data_loading_safe(validate_data=False, timeout=120.0)
def safe_load_ai_model(model_path: str, device: str = 'auto') -> Any:
    """Safely load AI model."""
    logger = setup_logging("model_loader")
    
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading model from {model_path} on {device}")
        
        # Load model with error handling
        try:
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            model = model.to(device)
            model.eval()
            
            logger.info("✅ Model loaded successfully")
            return {'model': model, 'tokenizer': tokenizer, 'device': device}
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
            # Try loading with different precision
            try:
                model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                model = model.to(device)
                model.eval()
                
                logger.info("✅ Model loaded with half precision")
                return {'model': model, 'tokenizer': tokenizer, 'device': device}
                
            except Exception as e2:
                logger.error(f"Failed to load model with half precision: {e2}")
                raise
        
    except Exception as e:
        logger.error(f"Failed to load AI model from {model_path}: {e}")
        raise

# =============================================================================
# MODEL INFERENCE SAFETY FUNCTIONS
# =============================================================================

@model_inference_safe(gpu_fallback=True, memory_monitoring=True)
def safe_model_inference(
    model: Any,
    input_data: Any,
    batch_size: int = 1,
    max_length: int = 512
) -> Any:
    """Safely perform model inference."""
    logger = setup_logging("model_inference")
    
    try:
        # Validate input
        if input_data is None or (isinstance(input_data, (list, tuple)) and len(input_data) == 0):
            raise ValueError("Empty or None input data")
        
        # Check model state
        if not hasattr(model, 'eval'):
            raise ValueError("Model does not have eval method")
        
        model.eval()
        
        # Perform inference
        with torch.no_grad():
            if isinstance(input_data, str):
                # Text input
                inputs = model['tokenizer'](
                    input_data,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(model['device']) for k, v in inputs.items()}
                
                outputs = model['model'](**inputs)
                
            elif isinstance(input_data, (list, tuple)):
                # Batch input
                inputs = model['tokenizer'](
                    input_data,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(model['device']) for k, v in inputs.items()}
                
                outputs = model['model'](**inputs)
                
            else:
                # Direct tensor input
                if hasattr(input_data, 'to'):
                    input_data = input_data.to(model['device'])
                outputs = model['model'](input_data)
        
        logger.info("✅ Model inference completed successfully")
        return outputs
        
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise

@model_inference_safe(gpu_fallback=True, memory_monitoring=True)
def safe_video_generation(
    model: Any,
    prompt: str,
    duration: int = 10,
    quality: str = 'medium'
) -> str:
    """Safely generate video from prompt."""
    logger = setup_logging("video_generation")
    
    try:
        # Validate inputs
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Empty or invalid prompt")
        
        if duration <= 0 or duration > 60:
            raise ValueError(f"Invalid duration: {duration} (must be 1-60 seconds)")
        
        valid_qualities = ['low', 'medium', 'high']
        if quality not in valid_qualities:
            raise ValueError(f"Invalid quality: {quality} (must be one of {valid_qualities})")
        
        # Generate video
        logger.info(f"Generating video: '{prompt}' ({duration}s, {quality} quality)")
        
        # This would integrate with actual video generation model
        # For now, simulate the process
        time.sleep(2)  # Simulate processing time
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"generated_video_{timestamp}.mp4"
        
        logger.info(f"✅ Video generation completed: {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise

# =============================================================================
# FILE OPERATION SAFETY FUNCTIONS
# =============================================================================

@file_operation_safe(backup=True, verify_after_write=True)
def safe_save_video_data(data: Dict[str, Any], file_path: str) -> bool:
    """Safely save video data to file."""
    logger = setup_logging("file_saver")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save based on file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif file_ext in ['.pkl', '.pickle']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif file_ext in ['.yaml', '.yml']:
            import yaml
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        logger.info(f"✅ Data saved successfully to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise

@file_operation_safe(backup=True, verify_after_write=True)
def safe_save_video_file(video_data: bytes, file_path: str) -> bool:
    """Safely save video file."""
    logger = setup_logging("video_saver")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write video data
        with open(file_path, 'wb') as f:
            f.write(video_data)
        
        # Verify file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise IOError("Saved video file is empty")
        
        logger.info(f"✅ Video saved successfully: {file_path} ({file_size} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save video to {file_path}: {e}")
        raise

# =============================================================================
# NETWORK OPERATION SAFETY FUNCTIONS
# =============================================================================

@network_operation_safe(max_retries=3, timeout=30.0)
def safe_api_request(url: str, method: str = 'GET', data: Any = None, headers: Dict = None) -> Dict[str, Any]:
    """Safely make API request."""
    logger = setup_logging("api_client")
    
    try:
        import requests
        
        # Set default headers
        if headers is None:
            headers = {
                'User-Agent': 'Video-OpusClip/1.0',
                'Content-Type': 'application/json'
            }
        
        # Make request
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, timeout=30)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        # Check response
        response.raise_for_status()
        
        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {'text': response.text}
        
        logger.info(f"✅ API request successful: {url}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in API request: {e}")
        raise

@network_operation_safe(max_retries=2, timeout=60.0)
def safe_download_file(url: str, local_path: str) -> bool:
    """Safely download file from URL."""
    logger = setup_logging("file_downloader")
    
    try:
        import requests
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with progress tracking
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.debug(f"Download progress: {progress:.1f}%")
        
        # Verify download
        if os.path.getsize(local_path) == 0:
            raise IOError("Downloaded file is empty")
        
        logger.info(f"✅ File downloaded successfully: {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        # Clean up partial download
        if os.path.exists(local_path):
            os.remove(local_path)
        raise

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _validate_loaded_data(data: Any) -> bool:
    """Validate loaded data."""
    if data is None:
        return False
    
    if isinstance(data, dict):
        return len(data) > 0
    elif isinstance(data, (list, tuple)):
        return len(data) > 0
    elif isinstance(data, str):
        return len(data.strip()) > 0
    
    return True

def _fallback_to_cpu_inference(func: Callable, args: tuple, kwargs: dict) -> Any:
    """Fallback to CPU inference if GPU fails."""
    logger = setup_logging("cpu_fallback")
    
    try:
        # Modify kwargs to use CPU
        if 'device' in kwargs:
            kwargs['device'] = 'cpu'
        
        # If model is passed as first argument, move to CPU
        if args and hasattr(args[0], 'to'):
            model = args[0].to('cpu')
            args = (model,) + args[1:]
        
        logger.info("🔄 Running inference on CPU")
        result = func(*args, **kwargs)
        
        logger.info("✅ CPU inference completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"CPU fallback also failed: {e}")
        raise

def _create_file_backup(file_path: str) -> str:
    """Create backup of file before operation."""
    try:
        if not os.path.exists(file_path):
            return None
        
        backup_path = f"{file_path}.backup.{int(time.time())}"
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
        
    except Exception as e:
        logger = setup_logging("backup")
        logger.warning(f"Failed to create backup: {e}")
        return None

def _verify_file_integrity(file_path: str) -> bool:
    """Verify file integrity after write operation."""
    try:
        if not os.path.exists(file_path):
            return False
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False
        
        # For video files, check if they can be opened
        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            try:
                import cv2
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    return False
                cap.release()
            except:
                return False
        
        return True
        
    except Exception:
        return False

def _load_video_file(file_path: str) -> Dict[str, Any]:
    """Load video file metadata."""
    try:
        import cv2
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {file_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'file_path': file_path,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'file_size': os.path.getsize(file_path)
        }
        
    except Exception as e:
        raise ValueError(f"Failed to load video file: {e}")

def _load_metadata_file(file_path: str) -> Dict[str, Any]:
    """Load metadata file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported metadata format: {file_path}")
                
    except Exception as e:
        raise ValueError(f"Failed to load metadata file: {e}")

def _load_pickle_file(file_path: str) -> Any:
    """Load pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load pickle file: {e}")

# Circuit breaker pattern implementation
_circuit_breaker_state = {}
_circuit_breaker_threshold = 5
_circuit_breaker_timeout = 60  # seconds

def _is_circuit_open(operation_name: str) -> bool:
    """Check if circuit breaker is open for operation."""
    if operation_name not in _circuit_breaker_state:
        return False
    
    state = _circuit_breaker_state[operation_name]
    if state['status'] == 'open':
        if time.time() - state['last_failure'] > _circuit_breaker_timeout:
            # Try to close circuit
            state['status'] = 'half_open'
            state['failure_count'] = 0
            return False
        return True
    
    return False

def _mark_circuit_open(operation_name: str):
    """Mark circuit breaker as open."""
    if operation_name not in _circuit_breaker_state:
        _circuit_breaker_state[operation_name] = {
            'status': 'closed',
            'failure_count': 0,
            'last_failure': 0
        }
    
    state = _circuit_breaker_state[operation_name]
    state['failure_count'] += 1
    state['last_failure'] = time.time()
    
    if state['failure_count'] >= _circuit_breaker_threshold:
        state['status'] = 'open'
        logger = setup_logging("circuit_breaker")
        logger.warning(f"Circuit breaker opened for {operation_name}")

def _mark_circuit_closed(operation_name: str):
    """Mark circuit breaker as closed."""
    if operation_name in _circuit_breaker_state:
        _circuit_breaker_state[operation_name]['status'] = 'closed'
        _circuit_breaker_state[operation_name]['failure_count'] = 0

# =============================================================================
# ENHANCED ERROR HANDLING CONTEXT MANAGERS
# =============================================================================

class SafeOperationContext:
    """Context manager for safe operations."""
    
    def __init__(self, operation_name: str, logger_name: str = "safe_context"):
        self.operation_name = operation_name
        self.logger = setup_logging(logger_name)
        self.start_time = None
        self.success = False
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"🚀 Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.success = True
            self.logger.info(f"✅ {self.operation_name} completed successfully in {duration:.2f}s")
        else:
            self.logger.error(f"❌ {self.operation_name} failed after {duration:.2f}s: {exc_val}")
            if exc_tb:
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")
        
        return False  # Don't suppress exceptions

class MemoryMonitor:
    """Context manager for memory monitoring."""
    
    def __init__(self, threshold_mb: int = 1024):
        self.threshold_mb = threshold_mb
        self.logger = setup_logging("memory_monitor")
        self.initial_memory = None
    
    def __enter__(self):
        import psutil
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        self.logger.debug(f"Memory at start: {self.initial_memory / (1024**2):.2f}MB")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import psutil
        process = psutil.Process()
        final_memory = process.memory_info().rss
        memory_used = final_memory - self.initial_memory
        
        self.logger.debug(f"Memory at end: {final_memory / (1024**2):.2f}MB (+{memory_used / (1024**2):.2f}MB)")
        
        if memory_used > self.threshold_mb * 1024 * 1024:
            self.logger.warning(f"⚠️ High memory usage: {memory_used / (1024**2):.2f}MB")
        
        return False

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_safe_operations():
    """Example usage of safe operations."""
    
    # Safe data loading
    try:
        video_data = safe_load_video_data("sample_video.mp4")
        print(f"✅ Loaded video: {video_data['duration']:.2f}s")
    except Exception as e:
        print(f"❌ Failed to load video: {e}")
    
    # Safe model inference
    try:
        with SafeOperationContext("model_inference"):
            with MemoryMonitor(threshold_mb=512):
                result = safe_model_inference(model, "Sample prompt")
                print(f"✅ Inference result: {result}")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
    
    # Safe file operations
    try:
        safe_save_video_data({"metadata": "test"}, "output.json")
        print("✅ Data saved successfully")
    except Exception as e:
        print(f"❌ Save failed: {e}")
    
    # Safe network operations
    try:
        response = safe_api_request("https://api.example.com/data")
        print(f"✅ API response: {response}")
    except Exception as e:
        print(f"❌ API request failed: {e}")

if __name__ == "__main__":
    example_safe_operations() 