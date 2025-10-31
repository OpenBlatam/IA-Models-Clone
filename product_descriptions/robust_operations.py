from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import pickle
import gzip
import hashlib
import threading
from contextlib import contextmanager
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import structlog
from error_handling_debugging import ErrorHandlingDebuggingSystem, ErrorSeverity, ErrorCategory
            import ipaddress
from typing import Any, List, Dict, Optional
"""
Robust Operations with Comprehensive Try-Except Blocks

This module provides robust error handling for error-prone operations in cybersecurity ML:
- Data loading and preprocessing with comprehensive error handling
- Model inference with fallback mechanisms
- File operations with retry logic
- Network operations with timeout and retry
- Memory management with cleanup
- Security-focused error handling
"""




# Configure logging
logger = structlog.get_logger(__name__)


class OperationType(Enum):
    """Types of operations that need error handling."""
    DATA_LOADING = "data_loading"
    MODEL_INFERENCE = "model_inference"
    FILE_OPERATION = "file_operation"
    NETWORK_OPERATION = "network_operation"
    MEMORY_OPERATION = "memory_operation"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"


@dataclass
class OperationResult:
    """Result of an operation with error handling."""
    success: bool
    data: Any = None
    error: Optional[Exception] = None
    error_message: str = ""
    operation_type: OperationType = OperationType.DATA_LOADING
    retry_count: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class RobustDataLoader:
    """Robust data loading with comprehensive error handling."""
    
    def __init__(self, error_system: ErrorHandlingDebuggingSystem):
        
    """__init__ function."""
self.error_system = error_system
        self.loaded_data_cache = {}
        self.data_validation_rules = {}
        
    def load_csv_data(
        self, 
        file_path: str, 
        encoding: str = 'utf-8',
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> OperationResult:
        """Load CSV data with comprehensive error handling."""
        start_time = time.time()
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.error_system.error_context("csv_data_loading", ErrorSeverity.ERROR):
                    # Validate file path
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"CSV file not found: {file_path}")
                    
                    # Check file size
                    file_size = Path(file_path).stat().st_size
                    if file_size > 100 * 1024 * 1024:  # 100MB limit
                        raise ValueError(f"File too large: {file_size / 1024 / 1024:.2f}MB")
                    
                    # Load CSV with error handling
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        # Try different encodings
                        for enc in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                df = pd.read_csv(file_path, encoding=enc)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            raise ValueError(f"Unable to decode file with any encoding: {file_path}")
                    
                    # Validate data
                    if df.empty:
                        raise ValueError("CSV file is empty")
                    
                    # Check for required columns (example for cybersecurity data)
                    required_columns = ['timestamp', 'source_ip', 'destination_ip']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        logger.warning(f"Missing columns: {missing_columns}")
                    
                    # Basic data cleaning
                    df = self._clean_dataframe(df)
                    
                    # Cache the data
                    cache_key = hashlib.md5(file_path.encode()).hexdigest()
                    self.loaded_data_cache[cache_key] = df
                    
                    execution_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        data=df,
                        operation_type=OperationType.DATA_LOADING,
                        retry_count=retry_count,
                        execution_time=execution_time,
                        metadata={
                            "file_path": file_path,
                            "rows": len(df),
                            "columns": len(df.columns),
                            "file_size_mb": file_size / 1024 / 1024
                        }
                    )
                    
            except Exception as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                # Log error with context
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.DATA,
                    context={
                        "operation": "load_csv_data",
                        "file_path": file_path,
                        "retry_count": retry_count,
                        "execution_time": execution_time
                    }
                )
                
                # Attempt recovery
                recovery_successful = await self.error_system.error_recovery.attempt_recovery(
                    e, {"operation": "load_csv_data", "file_path": file_path}
                )
                
                if recovery_successful and retry_count < max_retries:
                    logger.info(f"Recovery successful, retrying CSV load (attempt {retry_count + 1})")
                    time.sleep(1 * retry_count)  # Exponential backoff
                    continue
                
                return OperationResult(
                    success=False,
                    error=e,
                    error_message=str(e),
                    operation_type=OperationType.DATA_LOADING,
                    retry_count=retry_count,
                    execution_time=execution_time,
                    metadata={"file_path": file_path, "error_id": error_id}
                )
        
        return OperationResult(
            success=False,
            error_message="Max retries exceeded",
            operation_type=OperationType.DATA_LOADING,
            retry_count=retry_count,
            execution_time=time.time() - start_time
        )
    
    def load_json_data(
        self, 
        file_path: str,
        max_retries: int = 3
    ) -> OperationResult:
        """Load JSON data with comprehensive error handling."""
        start_time = time.time()
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.error_system.error_context("json_data_loading", ErrorSeverity.ERROR):
                    # Validate file path
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"JSON file not found: {file_path}")
                    
                    # Load JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        data = json.load(f)
                    
                    # Validate JSON structure
                    if not isinstance(data, (dict, list)):
                        raise ValueError("Invalid JSON structure")
                    
                    execution_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        data=data,
                        operation_type=OperationType.DATA_LOADING,
                        retry_count=retry_count,
                        execution_time=execution_time,
                        metadata={"file_path": file_path, "data_type": type(data).__name__}
                    )
                    
            except json.JSONDecodeError as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.DATA,
                    context={"operation": "load_json_data", "file_path": file_path}
                )
                
                logger.error(f"JSON decode error: {str(e)}")
                
            except Exception as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.DATA,
                    context={"operation": "load_json_data", "file_path": file_path}
                )
                
                # Attempt recovery
                recovery_successful = await self.error_system.error_recovery.attempt_recovery(
                    e, {"operation": "load_json_data", "file_path": file_path}
                )
                
                if recovery_successful and retry_count < max_retries:
                    logger.info(f"Recovery successful, retrying JSON load (attempt {retry_count + 1})")
                    time.sleep(1 * retry_count)
                    continue
                
                return OperationResult(
                    success=False,
                    error=e,
                    error_message=str(e),
                    operation_type=OperationType.DATA_LOADING,
                    retry_count=retry_count,
                    execution_time=execution_time,
                    metadata={"file_path": file_path, "error_id": error_id}
                )
        
        return OperationResult(
            success=False,
            error_message="Max retries exceeded",
            operation_type=OperationType.DATA_LOADING,
            retry_count=retry_count,
            execution_time=time.time() - start_time
        )
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe."""
        try:
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Basic type conversion for common cybersecurity fields
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                except Exception as e:
                    logger.warning(f"Timestamp conversion failed: {str(e)}")
            
            # Validate IP addresses
            ip_columns = [col for col in df.columns if 'ip' in col.lower()]
            for col in ip_columns:
                df[col] = df[col].astype(str).apply(self._validate_ip_address)
            
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return df  # Return original if cleaning fails
    
    def _validate_ip_address(self, ip: str) -> str:
        """Validate and clean IP address."""
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            return "invalid_ip"


class RobustModelInference:
    """Robust model inference with comprehensive error handling."""
    
    def __init__(self, error_system: ErrorHandlingDebuggingSystem):
        
    """__init__ function."""
self.error_system = error_system
        self.model_cache = {}
        self.inference_history = []
        
    def safe_inference(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        device: torch.device = None,
        max_retries: int = 3,
        fallback_model: Optional[nn.Module] = None
    ) -> OperationResult:
        """Perform safe model inference with comprehensive error handling."""
        start_time = time.time()
        retry_count = 0
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        while retry_count < max_retries:
            try:
                with self.error_system.error_context("model_inference", ErrorSeverity.ERROR):
                    # Validate input
                    if not isinstance(input_data, torch.Tensor):
                        raise ValueError("Input must be a torch.Tensor")
                    
                    if input_data.dim() == 0:
                        raise ValueError("Input tensor must have at least 1 dimension")
                    
                    # Check for NaN/Inf in input
                    if torch.isnan(input_data).any() or torch.isinf(input_data).any():
                        raise ValueError("Input contains NaN or infinite values")
                    
                    # Move model and data to device
                    model = model.to(device)
                    input_data = input_data.to(device)
                    
                    # Set model to evaluation mode
                    model.eval()
                    
                    # Perform inference with gradient computation disabled
                    with torch.no_grad():
                        # Check memory before inference
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
                                raise RuntimeError("Insufficient GPU memory")
                        
                        # Perform inference
                        output = model(input_data)
                        
                        # Validate output
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            raise ValueError("Model output contains NaN or infinite values")
                        
                        # Move output back to CPU if needed
                        if device.type == 'cuda':
                            output = output.cpu()
                    
                    execution_time = time.time() - start_time
                    
                    # Record successful inference
                    self.inference_history.append({
                        "timestamp": time.time(),
                        "input_shape": input_data.shape,
                        "output_shape": output.shape,
                        "execution_time": execution_time,
                        "device": str(device)
                    })
                    
                    return OperationResult(
                        success=True,
                        data=output,
                        operation_type=OperationType.MODEL_INFERENCE,
                        retry_count=retry_count,
                        execution_time=execution_time,
                        metadata={
                            "input_shape": input_data.shape,
                            "output_shape": output.shape,
                            "device": str(device),
                            "model_type": type(model).__name__
                        }
                    )
                    
            except RuntimeError as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                # Handle CUDA out of memory
                if "out of memory" in str(e).lower():
                    logger.warning("CUDA out of memory, attempting recovery")
                    
                    # Clear CUDA cache
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Try with smaller batch or fallback model
                    if fallback_model and retry_count == max_retries - 1:
                        logger.info("Using fallback model")
                        return self.safe_inference(fallback_model, input_data, device, 1)
                    
                    if retry_count < max_retries:
                        time.sleep(1 * retry_count)
                        continue
                
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.MODEL,
                    context={
                        "operation": "model_inference",
                        "input_shape": input_data.shape if 'input_data' in locals() else None,
                        "device": str(device),
                        "retry_count": retry_count
                    }
                )
                
                return OperationResult(
                    success=False,
                    error=e,
                    error_message=str(e),
                    operation_type=OperationType.MODEL_INFERENCE,
                    retry_count=retry_count,
                    execution_time=execution_time,
                    metadata={"error_id": error_id, "device": str(device)}
                )
                
            except Exception as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.MODEL,
                    context={
                        "operation": "model_inference",
                        "input_shape": input_data.shape if 'input_data' in locals() else None,
                        "device": str(device),
                        "retry_count": retry_count
                    }
                )
                
                # Attempt recovery
                recovery_successful = await self.error_system.error_recovery.attempt_recovery(
                    e, {"operation": "model_inference", "device": str(device)}
                )
                
                if recovery_successful and retry_count < max_retries:
                    logger.info(f"Recovery successful, retrying inference (attempt {retry_count + 1})")
                    time.sleep(1 * retry_count)
                    continue
                
                # Use fallback model if available
                if fallback_model and retry_count == max_retries - 1:
                    logger.info("Using fallback model due to repeated failures")
                    return self.safe_inference(fallback_model, input_data, device, 1)
                
                return OperationResult(
                    success=False,
                    error=e,
                    error_message=str(e),
                    operation_type=OperationType.MODEL_INFERENCE,
                    retry_count=retry_count,
                    execution_time=execution_time,
                    metadata={"error_id": error_id, "device": str(device)}
                )
        
        return OperationResult(
            success=False,
            error_message="Max retries exceeded",
            operation_type=OperationType.MODEL_INFERENCE,
            retry_count=retry_count,
            execution_time=time.time() - start_time
        )
    
    def batch_inference(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device = None,
        batch_size: int = 32,
        max_workers: int = 4
    ) -> OperationResult:
        """Perform batch inference with error handling."""
        start_time = time.time()
        
        try:
            with self.error_system.error_context("batch_inference", ErrorSeverity.ERROR):
                if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                model = model.to(device)
                model.eval()
                
                all_outputs = []
                all_targets = []
                failed_batches = 0
                
                for batch_idx, (data, targets) in enumerate(dataloader):
                    try:
                        # Process single batch
                        result = self.safe_inference(model, data, device)
                        
                        if result.success:
                            all_outputs.append(result.data)
                            all_targets.append(targets)
                        else:
                            failed_batches += 1
                            logger.warning(f"Batch {batch_idx} failed: {result.error_message}")
                            
                            # Skip failed batch and continue
                            continue
                            
                    except Exception as e:
                        failed_batches += 1
                        logger.error(f"Batch {batch_idx} error: {str(e)}")
                        continue
                
                # Concatenate results
                if all_outputs:
                    outputs = torch.cat(all_outputs, dim=0)
                    targets = torch.cat(all_targets, dim=0)
                    
                    execution_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        data={"outputs": outputs, "targets": targets},
                        operation_type=OperationType.MODEL_INFERENCE,
                        execution_time=execution_time,
                        metadata={
                            "total_batches": len(dataloader),
                            "successful_batches": len(all_outputs),
                            "failed_batches": failed_batches,
                            "success_rate": len(all_outputs) / len(dataloader)
                        }
                    )
                else:
                    raise ValueError("All batches failed")
                    
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_id = self.error_system.error_tracker.track_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.MODEL,
                context={"operation": "batch_inference", "device": str(device)}
            )
            
            return OperationResult(
                success=False,
                error=e,
                error_message=str(e),
                operation_type=OperationType.MODEL_INFERENCE,
                execution_time=execution_time,
                metadata={"error_id": error_id}
            )


class RobustFileOperations:
    """Robust file operations with comprehensive error handling."""
    
    def __init__(self, error_system: ErrorHandlingDebuggingSystem):
        
    """__init__ function."""
self.error_system = error_system
        
    def safe_save_model(
        self,
        model: nn.Module,
        file_path: str,
        max_retries: int = 3
    ) -> OperationResult:
        """Save model with comprehensive error handling."""
        start_time = time.time()
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.error_system.error_context("model_saving", ErrorSeverity.ERROR):
                    # Create directory if it doesn't exist
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check available disk space
                    disk_usage = Path(file_path).parent.stat()
                    if hasattr(disk_usage, 'st_size'):
                        free_space = psutil.disk_usage(Path(file_path).parent).free
                        if free_space < 100 * 1024 * 1024:  # 100MB minimum
                            raise RuntimeError("Insufficient disk space")
                    
                    # Save model with error handling
                    torch.save(model.state_dict(), file_path)
                    
                    # Verify save was successful
                    if not Path(file_path).exists():
                        raise RuntimeError("Model file was not created")
                    
                    # Verify file can be loaded
                    test_load = torch.load(file_path, map_location='cpu')
                    if not isinstance(test_load, dict):
                        raise RuntimeError("Saved model has invalid format")
                    
                    execution_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        data=file_path,
                        operation_type=OperationType.FILE_OPERATION,
                        retry_count=retry_count,
                        execution_time=execution_time,
                        metadata={
                            "file_path": file_path,
                            "file_size_mb": Path(file_path).stat().st_size / 1024 / 1024
                        }
                    )
                    
            except Exception as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.FILE_OPERATION,
                    context={
                        "operation": "save_model",
                        "file_path": file_path,
                        "retry_count": retry_count
                    }
                )
                
                # Attempt recovery
                recovery_successful = await self.error_system.error_recovery.attempt_recovery(
                    e, {"operation": "save_model", "file_path": file_path}
                )
                
                if recovery_successful and retry_count < max_retries:
                    logger.info(f"Recovery successful, retrying save (attempt {retry_count + 1})")
                    time.sleep(1 * retry_count)
                    continue
                
                return OperationResult(
                    success=False,
                    error=e,
                    error_message=str(e),
                    operation_type=OperationType.FILE_OPERATION,
                    retry_count=retry_count,
                    execution_time=execution_time,
                    metadata={"file_path": file_path, "error_id": error_id}
                )
        
        return OperationResult(
            success=False,
            error_message="Max retries exceeded",
            operation_type=OperationType.FILE_OPERATION,
            retry_count=retry_count,
            execution_time=time.time() - start_time
        )
    
    def safe_load_model(
        self,
        model_class: type,
        file_path: str,
        device: torch.device = None,
        max_retries: int = 3
    ) -> OperationResult:
        """Load model with comprehensive error handling."""
        start_time = time.time()
        retry_count = 0
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        while retry_count < max_retries:
            try:
                with self.error_system.error_context("model_loading", ErrorSeverity.ERROR):
                    # Check if file exists
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"Model file not found: {file_path}")
                    
                    # Check file size
                    file_size = Path(file_path).stat().st_size
                    if file_size == 0:
                        raise ValueError("Model file is empty")
                    
                    # Load model state dict
                    state_dict = torch.load(file_path, map_location=device)
                    
                    # Validate state dict
                    if not isinstance(state_dict, dict):
                        raise ValueError("Invalid model file format")
                    
                    # Create model instance
                    model = model_class()
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()
                    
                    execution_time = time.time() - start_time
                    
                    return OperationResult(
                        success=True,
                        data=model,
                        operation_type=OperationType.FILE_OPERATION,
                        retry_count=retry_count,
                        execution_time=execution_time,
                        metadata={
                            "file_path": file_path,
                            "file_size_mb": file_size / 1024 / 1024,
                            "device": str(device)
                        }
                    )
                    
            except Exception as e:
                retry_count += 1
                execution_time = time.time() - start_time
                
                error_id = self.error_system.error_tracker.track_error(
                    error=e,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.FILE_OPERATION,
                    context={
                        "operation": "load_model",
                        "file_path": file_path,
                        "device": str(device),
                        "retry_count": retry_count
                    }
                )
                
                # Attempt recovery
                recovery_successful = await self.error_system.error_recovery.attempt_recovery(
                    e, {"operation": "load_model", "file_path": file_path}
                )
                
                if recovery_successful and retry_count < max_retries:
                    logger.info(f"Recovery successful, retrying load (attempt {retry_count + 1})")
                    time.sleep(1 * retry_count)
                    continue
                
                return OperationResult(
                    success=False,
                    error=e,
                    error_message=str(e),
                    operation_type=OperationType.FILE_OPERATION,
                    retry_count=retry_count,
                    execution_time=execution_time,
                    metadata={"file_path": file_path, "error_id": error_id}
                )
        
        return OperationResult(
            success=False,
            error_message="Max retries exceeded",
            operation_type=OperationType.FILE_OPERATION,
            retry_count=retry_count,
            execution_time=time.time() - start_time
        )


class RobustOperations:
    """Main class for robust operations with comprehensive error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.error_system = ErrorHandlingDebuggingSystem(self.config)
        
        # Initialize components
        self.data_loader = RobustDataLoader(self.error_system)
        self.model_inference = RobustModelInference(self.error_system)
        self.file_operations = RobustFileOperations(self.error_system)
        
        logger.info("RobustOperations initialized", config=self.config)
    
    @contextmanager
    def operation_context(self, operation_name: str, operation_type: OperationType):
        """Context manager for robust operations."""
        start_time = time.time()
        
        try:
            with self.error_system.error_context(operation_name, ErrorSeverity.ERROR):
                yield
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_id = self.error_system.error_tracker.track_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=self._get_category_for_operation(operation_type),
                context={
                    "operation": operation_name,
                    "operation_type": operation_type.value,
                    "execution_time": execution_time
                }
            )
            
            # Attempt recovery
            recovery_successful = await self.error_system.error_recovery.attempt_recovery(
                e, {"operation": operation_name, "operation_type": operation_type.value}
            )
            
            if not recovery_successful:
                raise
    
    def _get_category_for_operation(self, operation_type: OperationType) -> ErrorCategory:
        """Map operation type to error category."""
        mapping = {
            OperationType.DATA_LOADING: ErrorCategory.DATA,
            OperationType.MODEL_INFERENCE: ErrorCategory.MODEL,
            OperationType.FILE_OPERATION: ErrorCategory.SYSTEM,
            OperationType.NETWORK_OPERATION: ErrorCategory.NETWORK,
            OperationType.MEMORY_OPERATION: ErrorCategory.MEMORY,
            OperationType.PREPROCESSING: ErrorCategory.DATA,
            OperationType.POSTPROCESSING: ErrorCategory.DATA
        }
        return mapping.get(operation_type, ErrorCategory.UNKNOWN)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "error_system": self.error_system.get_system_status(),
            "data_loader": {
                "cache_size": len(self.data_loader.loaded_data_cache),
                "validation_rules": len(self.data_loader.data_validation_rules)
            },
            "model_inference": {
                "cache_size": len(self.model_inference.model_cache),
                "inference_history_length": len(self.model_inference.inference_history)
            },
            "timestamp": time.time()
        }
    
    def cleanup(self) -> Any:
        """Cleanup resources."""
        self.error_system.cleanup()
        logger.info("RobustOperations cleanup completed")


# Utility decorators for easy integration
def robust_operation(operation_type: OperationType, max_retries: int = 3):
    """Decorator for robust operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get or create robust operations instance
            if not hasattr(wrapper, '_robust_ops'):
                wrapper._robust_ops = RobustOperations()
            
            retry_count = 0
            while retry_count < max_retries:
                try:
                    with wrapper._robust_ops.operation_context(func.__name__, operation_type):
                        return await func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        raise
                    
                    logger.warning(f"Operation failed, retrying ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(1 * retry_count)  # Exponential backoff
        
        return wrapper
    return decorator


def safe_data_loading(max_retries: int = 3):
    """Decorator for safe data loading operations."""
    return robust_operation(OperationType.DATA_LOADING, max_retries)


def safe_model_inference(max_retries: int = 3):
    """Decorator for safe model inference operations."""
    return robust_operation(OperationType.MODEL_INFERENCE, max_retries)


def safe_file_operation(max_retries: int = 3):
    """Decorator for safe file operations."""
    return robust_operation(OperationType.FILE_OPERATION, max_retries)


# Example usage
if __name__ == "__main__":
    # Initialize robust operations
    robust_ops = RobustOperations({
        "max_errors": 5000,
        "enable_persistence": True,
        "enable_profiling": True
    })
    
    # Example: Safe data loading
    @safe_data_loading(max_retries=3)
    async def load_cybersecurity_data(file_path: str):
        
    """load_cybersecurity_data function."""
result = robust_ops.data_loader.load_csv_data(file_path)
        if not result.success:
            raise Exception(f"Data loading failed: {result.error_message}")
        return result.data
    
    # Example: Safe model inference
    @safe_model_inference(max_retries=3)
    async def run_model_inference(model: nn.Module, data: torch.Tensor):
        
    """run_model_inference function."""
result = robust_ops.model_inference.safe_inference(model, data)
        if not result.success:
            raise Exception(f"Model inference failed: {result.error_message}")
        return result.data
    
    # Example: Safe file operation
    @safe_file_operation(max_retries=3)
    async def save_model_safely(model: nn.Module, file_path: str):
        
    """save_model_safely function."""
result = robust_ops.file_operations.safe_save_model(model, file_path)
        if not result.success:
            raise Exception(f"Model saving failed: {result.error_message}")
        return result.data 