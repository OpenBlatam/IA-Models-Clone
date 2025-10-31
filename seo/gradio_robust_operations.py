#!/usr/bin/env python3
"""
Robust Operations with Comprehensive Try-Except Blocks
Enhanced error handling for data loading, model inference, and other error-prone operations
"""

import torch
import numpy as np
import pandas as pd
import logging
import traceback
import time
import gc
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import json
import warnings
from contextlib import contextmanager
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustDataLoader:
    """Robust data loading with comprehensive error handling."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_log = []
        self.success_count = 0
        self.failure_count = 0
    
    def safe_load_file(self, file_path: str, file_type: str = "auto") -> Tuple[Any, Optional[str]]:
        """Safely load a file with comprehensive error handling."""
        for attempt in range(self.max_retries):
            try:
                if file_type == "auto":
                    file_type = self._detect_file_type(file_path)
                
                if file_type == "csv":
                    data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                elif file_type == "json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif file_type == "txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                elif file_type == "pickle":
                    data = pd.read_pickle(file_path)
                elif file_type == "excel":
                    data = pd.read_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                self.success_count += 1
                logger.info(f"Successfully loaded {file_path} (attempt {attempt + 1})")
                return data, None
                
            except FileNotFoundError as e:
                error_msg = f"File not found: {file_path}"
                self._log_error("FileNotFoundError", error_msg, attempt, file_path)
                if attempt == self.max_retries - 1:
                    return None, error_msg
                    
            except PermissionError as e:
                error_msg = f"Permission denied accessing file: {file_path}"
                self._log_error("PermissionError", error_msg, attempt, file_path)
                if attempt == self.max_retries - 1:
                    return None, error_msg
                    
            except UnicodeDecodeError as e:
                error_msg = f"Encoding error in file {file_path}: {e}"
                self._log_error("UnicodeDecodeError", error_msg, attempt, file_path)
                # Try different encodings
                if attempt < self.max_retries - 1:
                    try:
                        if file_type == "csv":
                            data = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
                        elif file_type == "txt":
                            with open(file_path, 'r', encoding='latin-1') as f:
                                data = f.read()
                        else:
                            raise e
                        self.success_count += 1
                        logger.info(f"Successfully loaded {file_path} with latin-1 encoding")
                        return data, None
                    except:
                        pass
                        
            except pd.errors.EmptyDataError as e:
                error_msg = f"File is empty: {file_path}"
                self._log_error("EmptyDataError", error_msg, attempt, file_path)
                if attempt == self.max_retries - 1:
                    return None, error_msg
                    
            except pd.errors.ParserError as e:
                error_msg = f"Parsing error in {file_path}: {e}"
                self._log_error("ParserError", error_msg, attempt, file_path)
                if attempt == self.max_retries - 1:
                    return None, error_msg
                    
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error in {file_path}: {e}"
                self._log_error("JSONDecodeError", error_msg, attempt, file_path)
                if attempt == self.max_retries - 1:
                    return None, error_msg
                    
            except Exception as e:
                error_msg = f"Unexpected error loading {file_path}: {e}"
                self._log_error("UnexpectedError", error_msg, attempt, file_path)
                if attempt == self.max_retries - 1:
                    return None, error_msg
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        self.failure_count += 1
        return None, f"Failed to load {file_path} after {self.max_retries} attempts"
    
    def safe_load_batch(self, file_paths: List[str], file_types: Optional[List[str]] = None) -> Tuple[List[Any], List[str]]:
        """Safely load multiple files with error handling."""
        if file_types is None:
            file_types = ["auto"] * len(file_paths)
        
        results = []
        errors = []
        
        for file_path, file_type in zip(file_paths, file_types):
            data, error = self.safe_load_file(file_path, file_type)
            if error:
                errors.append(f"{file_path}: {error}")
                results.append(None)
            else:
                results.append(data)
        
        return results, errors
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.json':
            return 'json'
        elif ext == '.txt':
            return 'txt'
        elif ext == '.pkl':
            return 'pickle'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        else:
            return 'txt'  # Default to text
    
    def _log_error(self, error_type: str, error_msg: str, attempt: int, file_path: str):
        """Log error information."""
        self.error_log.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_msg,
            'attempt': attempt + 1,
            'file_path': file_path
        })
        logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0,
            'total_errors': len(self.error_log),
            'error_types': list(set([e['error_type'] for e in self.error_log]))
        }

class RobustModelInference:
    """Robust model inference with comprehensive error handling."""
    
    def __init__(self, model: Optional[torch.nn.Module] = None, device: str = "auto"):
        self.model = model
        self.device = self._setup_device(device)
        self.error_log = []
        self.inference_count = 0
        self.success_count = 0
        self.failure_count = 0
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for model inference."""
        try:
            if device == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                else:
                    device = torch.device("cpu")
                    logger.info("Using CPU device")
            else:
                device = torch.device(device)
            
            return device
        except Exception as e:
            logger.warning(f"Device setup failed: {e}, falling back to CPU")
            return torch.device("cpu")
    
    def safe_inference(self, input_data: Any, **kwargs) -> Tuple[Any, Optional[str]]:
        """Safely perform model inference with comprehensive error handling."""
        self.inference_count += 1
        
        try:
            # Validate input data
            if input_data is None:
                raise ValueError("Input data cannot be None")
            
            # Check model availability
            if self.model is None:
                raise RuntimeError("Model is not initialized")
            
            # Move model to device if needed
            if next(self.model.parameters()).device != self.device:
                self.model = self.model.to(self.device)
            
            # Prepare input data
            prepared_input = self._prepare_input(input_data)
            
            # Perform inference
            with torch.no_grad():
                if self.device.type == "cuda":
                    # GPU inference with memory management
                    result = self._gpu_inference(prepared_input, **kwargs)
                else:
                    # CPU inference
                    result = self._cpu_inference(prepared_input, **kwargs)
            
            self.success_count += 1
            logger.info(f"Inference successful (attempt {self.inference_count})")
            return result, None
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU out of memory during inference: {e}"
            self._log_error("CUDAOutOfMemoryError", error_msg, input_data)
            self._handle_memory_error()
            return None, error_msg
            
        except torch.cuda.CudaError as e:
            error_msg = f"CUDA error during inference: {e}"
            self._log_error("CudaError", error_msg, input_data)
            return None, error_msg
            
        except RuntimeError as e:
            error_msg = f"Runtime error during inference: {e}"
            self._log_error("RuntimeError", error_msg, input_data)
            return None, error_msg
            
        except ValueError as e:
            error_msg = f"Value error during inference: {e}"
            self._log_error("ValueError", error_msg, input_data)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during inference: {e}"
            self._log_error("UnexpectedError", error_msg, input_data)
            return None, error_msg
    
    def _prepare_input(self, input_data: Any) -> torch.Tensor:
        """Prepare input data for model inference."""
        try:
            if isinstance(input_data, torch.Tensor):
                return input_data.to(self.device)
            elif isinstance(input_data, np.ndarray):
                return torch.from_numpy(input_data).to(self.device)
            elif isinstance(input_data, (list, tuple)):
                return torch.tensor(input_data, dtype=torch.float32).to(self.device)
            elif isinstance(input_data, (int, float)):
                return torch.tensor([input_data], dtype=torch.float32).to(self.device)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
        except Exception as e:
            raise ValueError(f"Input preparation failed: {e}")
    
    def _gpu_inference(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform GPU inference with memory management."""
        try:
            # Check available GPU memory
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                required_memory = input_data.element_size() * input_data.nelement() * 10  # Rough estimate
                
                if free_memory < required_memory:
                    # Clear cache and try again
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if torch.cuda.memory_allocated() > required_memory:
                        raise torch.cuda.OutOfMemoryError("Insufficient GPU memory even after cleanup")
            
            # Perform inference
            result = self.model(input_data, **kwargs)
            
            # Clear intermediate tensors
            if hasattr(result, 'detach'):
                result = result.detach()
            
            return result
            
        except Exception as e:
            # Cleanup on error
            torch.cuda.empty_cache()
            gc.collect()
            raise e
    
    def _cpu_inference(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform CPU inference."""
        try:
            result = self.model(input_data, **kwargs)
            
            if hasattr(result, 'detach'):
                result = result.detach()
            
            return result
            
        except Exception as e:
            gc.collect()
            raise e
    
    def _handle_memory_error(self):
        """Handle GPU memory errors."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("GPU memory cleared after error")
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")
    
    def _log_error(self, error_type: str, error_msg: str, input_data: Any):
        """Log inference error information."""
        self.error_log.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_msg,
            'input_shape': str(getattr(input_data, 'shape', 'N/A')),
            'device': str(self.device),
            'model_loaded': self.model is not None
        })
        self.failure_count += 1
        logger.error(f"Inference error: {error_msg}")
    
    def batch_inference(self, input_batch: List[Any], batch_size: int = 32, **kwargs) -> Tuple[List[Any], List[str]]:
        """Perform batch inference with error handling."""
        results = []
        errors = []
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            
            try:
                # Process batch
                for item in batch:
                    result, error = self.safe_inference(item, **kwargs)
                    if error:
                        errors.append(f"Item {i}: {error}")
                        results.append(None)
                    else:
                        results.append(result)
                        
            except Exception as e:
                error_msg = f"Batch processing error at index {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                results.extend([None] * len(batch))
        
        return results, errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            'total_inferences': self.inference_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / self.inference_count if self.inference_count > 0 else 0,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'total_errors': len(self.error_log),
            'error_types': list(set([e['error_type'] for e in self.error_log]))
        }

class RobustDataProcessor:
    """Robust data processing with comprehensive error handling."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.error_log = []
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def safe_process_data(self, data: Any, processor_func: Callable, **kwargs) -> Tuple[Any, Optional[str]]:
        """Safely process data with comprehensive error handling."""
        self.processed_count += 1
        
        try:
            # Validate input
            if data is None:
                raise ValueError("Input data cannot be None")
            
            if not callable(processor_func):
                raise ValueError("Processor function must be callable")
            
            # Process data
            result = processor_func(data, **kwargs)
            
            # Validate output
            if result is None:
                raise ValueError("Processor function returned None")
            
            self.success_count += 1
            logger.info(f"Data processing successful (item {self.processed_count})")
            return result, None
            
        except MemoryError as e:
            error_msg = f"Memory error during processing: {e}"
            self._log_error("MemoryError", error_msg, data)
            self._handle_memory_error()
            return None, error_msg
            
        except ValueError as e:
            error_msg = f"Value error during processing: {e}"
            self._log_error("ValueError", error_msg, data)
            return None, error_msg
            
        except TypeError as e:
            error_msg = f"Type error during processing: {e}"
            self._log_error("TypeError", error_msg, data)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during processing: {e}"
            self._log_error("UnexpectedError", error_msg, data)
            return None, error_msg
    
    def parallel_process(self, data_list: List[Any], processor_func: Callable, **kwargs) -> Tuple[List[Any], List[str]]:
        """Process data in parallel with error handling."""
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_data = {
                executor.submit(self.safe_process_data, data, processor_func, **kwargs): i 
                for i, data in enumerate(data_list)
            }
            
            # Collect results
            for future in as_completed(future_to_data):
                data_index = future_to_data[future]
                try:
                    result, error = future.result()
                    if error:
                        errors.append(f"Item {data_index}: {error}")
                        results.append(None)
                    else:
                        results.append(result)
                except Exception as e:
                    error_msg = f"Item {data_index}: Future execution error: {e}"
                    errors.append(error_msg)
                    results.append(None)
        
        return results, errors
    
    def _handle_memory_error(self):
        """Handle memory errors."""
        try:
            gc.collect()
            logger.info("Memory cleared after error")
        except Exception as e:
            logger.warning(f"Failed to clear memory: {e}")
    
    def _log_error(self, error_type: str, error_msg: str, data: Any):
        """Log processing error information."""
        self.error_log.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_msg,
            'data_type': str(type(data)),
            'data_shape': str(getattr(data, 'shape', 'N/A')) if hasattr(data, 'shape') else 'N/A'
        })
        self.failure_count += 1
        logger.error(f"Processing error: {error_msg}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_processed': self.processed_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / self.processed_count if self.processed_count > 0 else 0,
            'total_errors': len(self.error_log),
            'error_types': list(set([e['error_type'] for e in self.error_log]))
        }

class RobustOperationManager:
    """Manager for robust operations with comprehensive error handling."""
    
    def __init__(self):
        self.data_loader = RobustDataLoader()
        self.model_inference = RobustModelInference()
        self.data_processor = RobustDataProcessor()
        self.operation_log = []
    
    def safe_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Tuple[Any, Optional[str]]:
        """Safely execute any operation with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Log success
            execution_time = time.time() - start_time
            self._log_operation(operation_name, True, None, execution_time, args, kwargs)
            
            logger.info(f"Operation '{operation_name}' completed successfully in {execution_time:.4f}s")
            return result, None
            
        except Exception as e:
            # Log failure
            execution_time = time.time() - start_time
            error_msg = f"Operation '{operation_name}' failed: {e}"
            self._log_operation(operation_name, False, str(e), execution_time, args, kwargs)
            
            logger.error(error_msg)
            return None, error_msg
    
    def safe_pipeline(self, pipeline_steps: List[Tuple[str, Callable]], initial_data: Any) -> Tuple[Any, List[str]]:
        """Execute a pipeline of operations with error handling."""
        current_data = initial_data
        errors = []
        
        for step_name, step_func in pipeline_steps:
            try:
                logger.info(f"Executing pipeline step: {step_name}")
                current_data, error = self.safe_operation(step_name, step_func, current_data)
                
                if error:
                    errors.append(f"Step '{step_name}': {error}")
                    break
                    
            except Exception as e:
                error_msg = f"Pipeline step '{step_name}' failed: {e}"
                errors.append(error_msg)
                break
        
        return current_data, errors
    
    def _log_operation(self, operation_name: str, success: bool, error_msg: Optional[str], 
                       execution_time: float, args: tuple, kwargs: dict):
        """Log operation information."""
        self.operation_log.append({
            'timestamp': time.time(),
            'operation_name': operation_name,
            'success': success,
            'error_message': error_msg,
            'execution_time': execution_time,
            'args_count': len(args),
            'kwargs_count': len(kwargs)
        })
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            'data_loading': self.data_loader.get_statistics(),
            'model_inference': self.model_inference.get_statistics(),
            'data_processing': self.data_processor.get_statistics(),
            'operations': {
                'total_operations': len(self.operation_log),
                'successful_operations': sum(1 for op in self.operation_log if op['success']),
                'failed_operations': sum(1 for op in self.operation_log if not op['success']),
                'average_execution_time': np.mean([op['execution_time'] for op in self.operation_log]) if self.operation_log else 0
            }
        }
    
    def export_error_report(self, filepath: str) -> bool:
        """Export comprehensive error report."""
        try:
            report = {
                'timestamp': time.time(),
                'statistics': self.get_comprehensive_statistics(),
                'error_logs': {
                    'data_loading': self.data_loader.error_log,
                    'model_inference': self.model_inference.error_log,
                    'data_processing': self.data_processor.error_log,
                    'operations': self.operation_log
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Error report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export error report: {e}")
            return False

# Utility decorators for robust operations
def robust_operation(max_retries: int = 3, retry_delay: float = 1.0):
    """Decorator for robust operations with retry logic."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Operation '{func.__name__}' failed after {max_retries} attempts: {e}")
                        raise
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for '{func.__name__}': {e}")
                        time.sleep(retry_delay * (attempt + 1))
            return None
        return wrapper
    return decorator

def safe_execution(error_handler: Optional[Callable] = None):
    """Decorator for safe execution with custom error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler:
                    return error_handler(e, func.__name__, args, kwargs)
                else:
                    logger.error(f"Function '{func.__name__}' failed: {e}")
                    return None
        return wrapper
    return decorator

# Context managers for robust operations
@contextmanager
def robust_file_operation(file_path: str, mode: str = 'r', encoding: str = 'utf-8'):
    """Context manager for robust file operations."""
    file_handle = None
    try:
        file_handle = open(file_path, mode, encoding=encoding)
        yield file_handle
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        raise
    except UnicodeDecodeError:
        logger.error(f"Encoding error: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with file {file_path}: {e}")
        raise
    finally:
        if file_handle:
            file_handle.close()

@contextmanager
def robust_model_inference(model: torch.nn.Module, device: str = "auto"):
    """Context manager for robust model inference."""
    original_device = next(model.parameters()).device
    target_device = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Move model to target device
        model = model.to(target_device)
        yield model
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise
    finally:
        # Move model back to original device
        try:
            model = model.to(original_device)
        except:
            pass

# Example usage functions
def example_data_loading():
    """Example of robust data loading."""
    loader = RobustDataLoader(max_retries=3)
    
    # Load single file
    data, error = loader.safe_load_file("example.csv", "csv")
    if error:
        print(f"Failed to load file: {error}")
    else:
        print("File loaded successfully")
    
    # Load multiple files
    files = ["data1.csv", "data2.json", "data3.txt"]
    results, errors = loader.safe_load_batch(files)
    
    # Print statistics
    stats = loader.get_statistics()
    print(f"Loading statistics: {stats}")

def example_model_inference():
    """Example of robust model inference."""
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    inference = RobustModelInference(model, device="auto")
    
    # Test inference
    input_data = torch.randn(5, 10)
    result, error = inference.safe_inference(input_data)
    
    if error:
        print(f"Inference failed: {error}")
    else:
        print("Inference successful")
    
    # Print statistics
    stats = inference.get_statistics()
    print(f"Inference statistics: {stats}")

def example_data_processing():
    """Example of robust data processing."""
    processor = RobustDataProcessor(max_workers=2)
    
    # Define processing function
    def process_item(item):
        return item * 2
    
    # Process single item
    data = [1, 2, 3, 4, 5]
    result, error = processor.safe_process_data(data, process_item)
    
    if error:
        print(f"Processing failed: {error}")
    else:
        print("Processing successful")
    
    # Process multiple items in parallel
    data_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    results, errors = processor.parallel_process(data_list, process_item)
    
    # Print statistics
    stats = processor.get_statistics()
    print(f"Processing statistics: {stats}")

if __name__ == "__main__":
    # Run examples
    print("=== Robust Data Loading Example ===")
    example_data_loading()
    
    print("\n=== Robust Model Inference Example ===")
    example_model_inference()
    
    print("\n=== Robust Data Processing Example ===")
    example_data_processing()
