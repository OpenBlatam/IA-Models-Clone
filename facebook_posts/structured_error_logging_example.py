from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import traceback
import sys
import inspect
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import conint, confloat, constr
import numpy as np
from pathlib import Path
from enum import Enum
from functools import wraps
import json
import uuid
from datetime import datetime, timezone
import structlog
            import inspect
from typing import Any, List, Dict, Optional
"""
Structured Error Logging - Complete Patterns
===========================================

This file demonstrates structured error logging with comprehensive context:
- Structured logging with module, function, parameters
- Error context preservation
- Performance monitoring integration
- Comprehensive error tracking
- Debugging information capture
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Structured Logging
    "StructuredLogger",
    "ErrorLogger",
    "PerformanceLogger",
    "ContextLogger",
    
    # Error Context
    "ErrorContext",
    "FunctionContext",
    "ModuleContext",
    "ParameterContext",
    
    # Logging Patterns
    "LoggingPatterns",
    "StructuredLoggingPatterns",
    "ErrorLoggingPatterns",
    
    # Common utilities
    "LoggingResult",
    "LoggingConfig",
    "LoggingType"
]

# ============================================================================
# COMMON UTILITIES
# ============================================================================

class LoggingResult(BaseModel):
    """Pydantic model for logging results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether logging was successful")
    log_id: str = Field(description="Unique log identifier")
    log_type: str = Field(description="Type of log entry")
    message: str = Field(description="Log message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Log context")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if applicable")

class LoggingConfig(BaseModel):
    """Pydantic model for logging configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level"
    )
    include_stack_trace: bool = Field(default=True, description="Include stack trace in errors")
    include_function_context: bool = Field(default=True, description="Include function context")
    include_parameter_values: bool = Field(default=True, description="Include parameter values")
    include_performance_metrics: bool = Field(default=True, description="Include performance metrics")
    max_context_size: conint(ge=100, le=10000) = Field(default=1000, description="Maximum context size")
    log_format: Literal["json", "text", "structured"] = Field(default="json", description="Log format")
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")

class LoggingType(BaseModel):
    """Pydantic model for logging type validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    log_type: constr(strip_whitespace=True) = Field(
        pattern=r"^(error|info|warning|debug|performance|security|audit)$"
    )
    description: Optional[str] = Field(default=None)
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium")

# ============================================================================
# ERROR CONTEXT
# ============================================================================

@dataclass
class FunctionContext:
    """Function context for structured logging."""
    
    function_name: str
    module_name: str
    class_name: Optional[str] = None
    file_path: str
    line_number: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "module_name": self.module_name,
            "class_name": self.class_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "docstring": self.docstring
        }

@dataclass
class ModuleContext:
    """Module context for structured logging."""
    
    module_name: str
    module_path: str
    module_version: Optional[str] = None
    module_dependencies: List[str] = field(default_factory=list)
    module_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "module_name": self.module_name,
            "module_path": self.module_path,
            "module_version": self.module_version,
            "module_dependencies": self.module_dependencies,
            "module_config": self.module_config
        }

@dataclass
class ParameterContext:
    """Parameter context for structured logging."""
    
    parameter_name: str
    parameter_type: str
    parameter_value: Any
    is_required: bool = True
    validation_rules: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter_name": self.parameter_name,
            "parameter_type": self.parameter_type,
            "parameter_value": str(self.parameter_value)[:200],  # Truncate long values
            "is_required": self.is_required,
            "validation_rules": self.validation_rules
        }

@dataclass
class ErrorContext:
    """Error context for structured logging."""
    
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    error_severity: Literal["low", "medium", "high", "critical"] = "medium"
    function_context: FunctionContext
    module_context: ModuleContext
    parameter_context: List[ParameterContext] = field(default_factory=list)
    stack_trace: Optional[str] = None
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "error_severity": self.error_severity,
            "function_context": self.function_context.to_dict(),
            "module_context": self.module_context.to_dict(),
            "parameter_context": [param.to_dict() for param in self.parameter_context],
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
            "additional_context": self.additional_context
        }

# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

class StructuredLogger:
    """Structured logger with comprehensive context."""
    
    def __init__(self, config: LoggingConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logger()
        self.error_count = 0
        self.performance_metrics = {}
    
    def _setup_logger(self) -> structlog.BoundLogger:
        """Setup structured logger."""
        if self.config.enable_structured_logging:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            return structlog.get_logger()
        else:
            # Fallback to standard logging
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return structlog.wrap(logging.getLogger(__name__))
    
    def _get_function_context(self, frame_depth: int = 1) -> FunctionContext:
        """Get function context from call stack."""
        try:
            frame = inspect.currentframe()
            for _ in range(frame_depth):
                frame = frame.f_back
            
            if frame:
                function_name = frame.f_code.co_name
                module_name = frame.f_globals.get('__name__', 'unknown')
                file_path = frame.f_code.co_filename
                line_number = frame.f_lineno
                
                # Get class name if in a class method
                class_name = None
                if 'self' in frame.f_locals:
                    class_name = frame.f_locals['self'].__class__.__name__
                
                # Get parameters
                parameters = {}
                if self.config.include_parameter_values:
                    for name, value in frame.f_locals.items():
                        if not name.startswith('_'):
                            parameters[name] = str(value)[:100]  # Truncate long values
                
                return FunctionContext(
                    function_name=function_name,
                    module_name=module_name,
                    class_name=class_name,
                    file_path=file_path,
                    line_number=line_number,
                    parameters=parameters
                )
        except Exception:
            pass
        
        return FunctionContext(
            function_name="unknown",
            module_name="unknown",
            file_path="unknown",
            line_number=0
        )
    
    def _get_module_context(self, module_name: str) -> ModuleContext:
        """Get module context."""
        try:
            module = sys.modules.get(module_name)
            if module:
                return ModuleContext(
                    module_name=module_name,
                    module_path=getattr(module, '__file__', 'unknown'),
                    module_version=getattr(module, '__version__', None),
                    module_dependencies=list(module.__dict__.keys()) if module else [],
                    module_config={}
                )
        except Exception:
            pass
        
        return ModuleContext(
            module_name=module_name,
            module_path="unknown"
        )
    
    def _get_parameter_context(self, func: Callable, args: tuple, kwargs: dict) -> List[ParameterContext]:
        """Get parameter context."""
        if not self.config.include_parameter_values:
            return []
        
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            parameter_context = []
            for name, value in bound_args.arguments.items():
                param = sig.parameters[name]
                parameter_context.append(ParameterContext(
                    parameter_name=name,
                    parameter_type=str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    parameter_value=value,
                    is_required=param.default == inspect.Parameter.empty,
                    validation_rules=None
                ))
            
            return parameter_context
        except Exception:
            return []
    
    def log_error(
        self,
        error: Exception,
        function_context: Optional[FunctionContext] = None,
        module_context: Optional[ModuleContext] = None,
        parameter_context: Optional[List[ParameterContext]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> LoggingResult:
        """Log error with structured context."""
        try:
            start_time = time.time()
            
            # Get contexts if not provided
            if function_context is None:
                function_context = self._get_function_context()
            
            if module_context is None:
                module_context = self._get_module_context(function_context.module_name)
            
            if parameter_context is None:
                parameter_context = []
            
            if additional_context is None:
                additional_context = {}
            
            # Create error context
            error_context = ErrorContext(
                error_type=type(error).__name__,
                error_message=str(error),
                function_context=function_context,
                module_context=module_context,
                parameter_context=parameter_context,
                stack_trace=traceback.format_exc() if self.config.include_stack_trace else None,
                additional_context=additional_context
            )
            
            # Log the error
            log_data = error_context.to_dict()
            
            if self.config.log_format == "json":
                self.logger.error("Error occurred", **log_data)
            else:
                self.logger.error(f"Error in {function_context.function_name}: {error}")
            
            execution_time = time.time() - start_time
            
            # Update metrics
            self.error_count += 1
            error_type = type(error).__name__
            if error_type not in self.performance_metrics:
                self.performance_metrics[error_type] = 0
            self.performance_metrics[error_type] += 1
            
            return LoggingResult(
                is_successful=True,
                log_id=error_context.error_id,
                log_type="error",
                message=str(error),
                context=log_data,
                execution_time=execution_time
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="error",
                message=f"Error logging failed: {str(exc)}",
                error_details={"original_error": str(error)}
            )
    
    def log_info(
        self,
        message: str,
        function_context: Optional[FunctionContext] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> LoggingResult:
        """Log info with structured context."""
        try:
            start_time = time.time()
            
            if function_context is None:
                function_context = self._get_function_context()
            
            if additional_context is None:
                additional_context = {}
            
            log_data = {
                "message": message,
                "function_context": function_context.to_dict(),
                "additional_context": additional_context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if self.config.log_format == "json":
                self.logger.info(message, **log_data)
            else:
                self.logger.info(f"[{function_context.function_name}] {message}")
            
            execution_time = time.time() - start_time
            
            return LoggingResult(
                is_successful=True,
                log_id=str(uuid.uuid4()),
                log_type="info",
                message=message,
                context=log_data,
                execution_time=execution_time
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="info",
                message=f"Info logging failed: {str(exc)}",
                error_details={"original_message": message}
            )
    
    def log_performance(
        self,
        operation_name: str,
        execution_time: float,
        function_context: Optional[FunctionContext] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> LoggingResult:
        """Log performance metrics with structured context."""
        try:
            start_time = time.time()
            
            if function_context is None:
                function_context = self._get_function_context()
            
            if additional_metrics is None:
                additional_metrics = {}
            
            log_data = {
                "operation_name": operation_name,
                "execution_time": execution_time,
                "function_context": function_context.to_dict(),
                "additional_metrics": additional_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if self.config.log_format == "json":
                self.logger.info("Performance metric", **log_data)
            else:
                self.logger.info(f"[{function_context.function_name}] {operation_name}: {execution_time:.3f}s")
            
            log_execution_time = time.time() - start_time
            
            return LoggingResult(
                is_successful=True,
                log_id=str(uuid.uuid4()),
                log_type="performance",
                message=f"{operation_name}: {execution_time:.3f}s",
                context=log_data,
                execution_time=log_execution_time
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="performance",
                message=f"Performance logging failed: {str(exc)}",
                error_details={"operation_name": operation_name, "execution_time": execution_time}
            )

class ErrorLogger:
    """Specialized error logger with comprehensive context."""
    
    def __init__(self, config: LoggingConfig):
        
    """__init__ function."""
self.config = config
        self.structured_logger = StructuredLogger(config)
        self.error_patterns = {}
    
    def log_validation_error(
        self,
        error: Exception,
        validation_type: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        suggestions: Optional[List[str]] = None
    ) -> LoggingResult:
        """Log validation error with structured context."""
        try:
            function_context = self.structured_logger._get_function_context()
            
            additional_context = {
                "validation_type": validation_type,
                "field_name": field_name,
                "field_value": str(field_value)[:200] if field_value is not None else None,
                "suggestions": suggestions or [],
                "error_category": "validation"
            }
            
            return self.structured_logger.log_error(
                error=error,
                function_context=function_context,
                additional_context=additional_context
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="validation_error",
                message=f"Validation error logging failed: {str(exc)}",
                error_details={"original_error": str(error)}
            )
    
    def log_processing_error(
        self,
        error: Exception,
        operation_name: str,
        operation_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> LoggingResult:
        """Log processing error with structured context."""
        try:
            function_context = self.structured_logger._get_function_context()
            
            additional_context = {
                "operation_name": operation_name,
                "operation_data": operation_data or {},
                "retry_count": retry_count,
                "error_category": "processing"
            }
            
            return self.structured_logger.log_error(
                error=error,
                function_context=function_context,
                additional_context=additional_context
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="processing_error",
                message=f"Processing error logging failed: {str(exc)}",
                error_details={"original_error": str(error)}
            )
    
    def log_network_error(
        self,
        error: Exception,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None
    ) -> LoggingResult:
        """Log network error with structured context."""
        try:
            function_context = self.structured_logger._get_function_context()
            
            additional_context = {
                "url": url,
                "status_code": status_code,
                "response_time": response_time,
                "error_category": "network"
            }
            
            return self.structured_logger.log_error(
                error=error,
                function_context=function_context,
                additional_context=additional_context
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="network_error",
                message=f"Network error logging failed: {str(exc)}",
                error_details={"original_error": str(error)}
            )

class PerformanceLogger:
    """Specialized performance logger with metrics."""
    
    def __init__(self, config: LoggingConfig):
        
    """__init__ function."""
self.config = config
        self.structured_logger = StructuredLogger(config)
        self.performance_metrics = {}
    
    def log_function_performance(
        self,
        function_name: str,
        execution_time: float,
        parameters: Optional[Dict[str, Any]] = None,
        return_value: Optional[Any] = None
    ) -> LoggingResult:
        """Log function performance with structured context."""
        try:
            function_context = self.structured_logger._get_function_context()
            
            additional_metrics = {
                "parameters": parameters or {},
                "return_value": str(return_value)[:200] if return_value is not None else None,
                "performance_category": "function_execution"
            }
            
            return self.structured_logger.log_performance(
                operation_name=function_name,
                execution_time=execution_time,
                function_context=function_context,
                additional_metrics=additional_metrics
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="performance",
                message=f"Performance logging failed: {str(exc)}",
                error_details={"function_name": function_name, "execution_time": execution_time}
            )
    
    def log_operation_performance(
        self,
        operation_name: str,
        execution_time: float,
        operation_type: str,
        success: bool,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> LoggingResult:
        """Log operation performance with structured context."""
        try:
            function_context = self.structured_logger._get_function_context()
            
            metrics = {
                "operation_type": operation_type,
                "success": success,
                "performance_category": "operation_execution"
            }
            
            if additional_metrics:
                metrics.update(additional_metrics)
            
            return self.structured_logger.log_performance(
                operation_name=operation_name,
                execution_time=execution_time,
                function_context=function_context,
                additional_metrics=metrics
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="performance",
                message=f"Operation performance logging failed: {str(exc)}",
                error_details={"operation_name": operation_name, "execution_time": execution_time}
            )

class ContextLogger:
    """Context-aware logger with comprehensive information."""
    
    def __init__(self, config: LoggingConfig):
        
    """__init__ function."""
self.config = config
        self.structured_logger = StructuredLogger(config)
        self.context_stack = []
    
    def log_with_context(
        self,
        message: str,
        log_level: str = "INFO",
        context_data: Optional[Dict[str, Any]] = None,
        include_stack: bool = True
    ) -> LoggingResult:
        """Log message with comprehensive context."""
        try:
            function_context = self.structured_logger._get_function_context()
            
            # Build comprehensive context
            full_context = {
                "message": message,
                "log_level": log_level,
                "context_stack": self.context_stack.copy(),
                "include_stack": include_stack
            }
            
            if context_data:
                full_context.update(context_data)
            
            if include_stack:
                full_context["stack_info"] = traceback.format_stack()
            
            # Log based on level
            if log_level.upper() == "ERROR":
                return self.structured_logger.log_error(
                    error=Exception(message),
                    function_context=function_context,
                    additional_context=full_context
                )
            else:
                return self.structured_logger.log_info(
                    message=message,
                    function_context=function_context,
                    additional_context=full_context
                )
                
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="context_log",
                message=f"Context logging failed: {str(exc)}",
                error_details={"original_message": message}
            )
    
    def add_context(self, context_name: str, context_value: Any) -> None:
        """Add context to the current context stack."""
        self.context_stack.append({
            "name": context_name,
            "value": str(context_value)[:200],  # Truncate long values
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def clear_context(self) -> None:
        """Clear the current context stack."""
        self.context_stack.clear()

# ============================================================================
# LOGGING PATTERNS
# ============================================================================

class LoggingPatterns:
    """Logging patterns with comprehensive context."""
    
    @staticmethod
    def log_function_entry(
        logger: StructuredLogger,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> LoggingResult:
        """Log function entry with parameters."""
        try:
            function_context = logger._get_function_context()
            
            additional_context = {
                "event_type": "function_entry",
                "parameters": parameters or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return logger.log_info(
                message=f"Entering function: {function_name}",
                function_context=function_context,
                additional_context=additional_context
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="function_entry",
                message=f"Function entry logging failed: {str(exc)}",
                error_details={"function_name": function_name}
            )
    
    @staticmethod
    def log_function_exit(
        logger: StructuredLogger,
        function_name: str,
        execution_time: float,
        return_value: Optional[Any] = None,
        success: bool = True
    ) -> LoggingResult:
        """Log function exit with results."""
        try:
            function_context = logger._get_function_context()
            
            additional_context = {
                "event_type": "function_exit",
                "execution_time": execution_time,
                "return_value": str(return_value)[:200] if return_value is not None else None,
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return logger.log_info(
                message=f"Exiting function: {function_name} (took {execution_time:.3f}s)",
                function_context=function_context,
                additional_context=additional_context
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="function_exit",
                message=f"Function exit logging failed: {str(exc)}",
                error_details={"function_name": function_name, "execution_time": execution_time}
            )

class StructuredLoggingPatterns:
    """Structured logging patterns with comprehensive context."""
    
    @staticmethod
    def log_operation_with_context(
        logger: StructuredLogger,
        operation_name: str,
        operation_data: Dict[str, Any],
        success: bool = True,
        error: Optional[Exception] = None
    ) -> LoggingResult:
        """Log operation with comprehensive context."""
        try:
            function_context = logger._get_function_context()
            
            additional_context = {
                "operation_name": operation_name,
                "operation_data": operation_data,
                "success": success,
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if success:
                return logger.log_info(
                    message=f"Operation completed: {operation_name}",
                    function_context=function_context,
                    additional_context=additional_context
                )
            else:
                return logger.log_error(
                    error=error or Exception(f"Operation failed: {operation_name}"),
                    function_context=function_context,
                    additional_context=additional_context
                )
                
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="operation_log",
                message=f"Operation logging failed: {str(exc)}",
                error_details={"operation_name": operation_name}
            )

class ErrorLoggingPatterns:
    """Error logging patterns with comprehensive context."""
    
    @staticmethod
    def log_error_with_context(
        logger: ErrorLogger,
        error: Exception,
        context_data: Dict[str, Any],
        error_category: str = "general"
    ) -> LoggingResult:
        """Log error with comprehensive context."""
        try:
            function_context = logger.structured_logger._get_function_context()
            
            additional_context = {
                "error_category": error_category,
                "context_data": context_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return logger.structured_logger.log_error(
                error=error,
                function_context=function_context,
                additional_context=additional_context
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="error_context",
                message=f"Error context logging failed: {str(exc)}",
                error_details={"original_error": str(error)}
            )

# ============================================================================
# MAIN STRUCTURED LOGGING MODULE
# ============================================================================

class MainStructuredLoggingModule:
    """Main structured logging module with proper imports and exports."""
    
    # Define main exports
    __all__ = [
        # Structured Logging
        "StructuredLogger",
        "ErrorLogger",
        "PerformanceLogger",
        "ContextLogger",
        
        # Error Context
        "ErrorContext",
        "FunctionContext",
        "ModuleContext",
        "ParameterContext",
        
        # Logging Patterns
        "LoggingPatterns",
        "StructuredLoggingPatterns",
        "ErrorLoggingPatterns",
        
        # Common utilities
        "LoggingResult",
        "LoggingConfig",
        "LoggingType",
        
        # Main functions
        "log_error_with_context",
        "log_performance_with_context",
        "log_function_with_context"
    ]
    
    def __init__(self, config: LoggingConfig):
        
    """__init__ function."""
self.config = config
        self.structured_logger = StructuredLogger(config)
        self.error_logger = ErrorLogger(config)
        self.performance_logger = PerformanceLogger(config)
        self.context_logger = ContextLogger(config)
    
    def log_error_with_context(
        self,
        error: Exception,
        context_data: Optional[Dict[str, Any]] = None,
        error_category: str = "general"
    ) -> LoggingResult:
        """Log error with comprehensive context."""
        try:
            if context_data is None:
                context_data = {}
            
            return ErrorLoggingPatterns.log_error_with_context(
                self.error_logger,
                error=error,
                context_data=context_data,
                error_category=error_category
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="error_logging",
                message=f"Error logging failed: {str(exc)}",
                error_details={"original_error": str(error)}
            )
    
    def log_performance_with_context(
        self,
        operation_name: str,
        execution_time: float,
        context_data: Optional[Dict[str, Any]] = None
    ) -> LoggingResult:
        """Log performance with comprehensive context."""
        try:
            if context_data is None:
                context_data = {}
            
            return self.performance_logger.log_operation_performance(
                operation_name=operation_name,
                execution_time=execution_time,
                operation_type="custom",
                success=True,
                additional_metrics=context_data
            )
            
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="performance_logging",
                message=f"Performance logging failed: {str(exc)}",
                error_details={"operation_name": operation_name, "execution_time": execution_time}
            )
    
    def log_function_with_context(
        self,
        function_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        return_value: Optional[Any] = None,
        execution_time: Optional[float] = None,
        success: bool = True,
        error: Optional[Exception] = None
    ) -> LoggingResult:
        """Log function execution with comprehensive context."""
        try:
            context_data = {
                "function_name": function_name,
                "parameters": parameters or {},
                "return_value": str(return_value)[:200] if return_value is not None else None,
                "execution_time": execution_time,
                "success": success
            }
            
            if error:
                return self.log_error_with_context(
                    error=error,
                    context_data=context_data,
                    error_category="function_execution"
                )
            else:
                return self.log_performance_with_context(
                    operation_name=function_name,
                    execution_time=execution_time or 0.0,
                    context_data=context_data
                )
                
        except Exception as exc:
            return LoggingResult(
                is_successful=False,
                log_id=str(uuid.uuid4()),
                log_type="function_logging",
                message=f"Function logging failed: {str(exc)}",
                error_details={"function_name": function_name}
            )

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_structured_error_logging():
    """Demonstrate structured error logging patterns."""
    
    print("📝 Demonstrating Structured Error Logging Patterns")
    print("=" * 60)
    
    # Initialize structured logging module
    config = LoggingConfig(
        log_level="INFO",
        include_stack_trace=True,
        include_function_context=True,
        include_parameter_values=True,
        include_performance_metrics=True,
        max_context_size=1000,
        log_format="json",
        enable_structured_logging=True
    )
    
    main_module = MainStructuredLoggingModule(config)
    
    # Example 1: Log error with context
    print("\n🚨 Error Logging with Context:")
    
    try:
        # Simulate an error
        raise ValueError("Invalid input parameter")
    except Exception as error:
        context_data = {
            "user_id": "12345",
            "operation": "data_processing",
            "input_data": {"field": "value"},
            "timestamp": datetime.now().isoformat()
        }
        
        result = main_module.log_error_with_context(
            error=error,
            context_data=context_data,
            error_category="validation"
        )
        print(f"Error logging result: {result.is_successful}")
        print(f"Log ID: {result.log_id}")
    
    # Example 2: Log performance with context
    print("\n⚡ Performance Logging with Context:")
    
    execution_time = 2.5
    context_data = {
        "database_queries": 5,
        "cache_hits": 3,
        "memory_usage": "150MB",
        "user_id": "12345"
    }
    
    result = main_module.log_performance_with_context(
        operation_name="user_data_processing",
        execution_time=execution_time,
        context_data=context_data
    )
    print(f"Performance logging result: {result.is_successful}")
    print(f"Log ID: {result.log_id}")
    
    # Example 3: Log function with context
    print("\n🔧 Function Logging with Context:")
    
    parameters = {"user_id": "12345", "include_profile": True}
    return_value = {"user": "john_doe", "email": "john@example.com"}
    
    result = main_module.log_function_with_context(
        function_name="get_user_data",
        parameters=parameters,
        return_value=return_value,
        execution_time=1.2,
        success=True
    )
    print(f"Function logging result: {result.is_successful}")
    print(f"Log ID: {result.log_id}")
    
    # Example 4: Log validation error
    print("\n✅ Validation Error Logging:")
    
    try:
        raise ValueError("Email format is invalid")
    except Exception as error:
        result = main_module.error_logger.log_validation_error(
            error=error,
            validation_type="email_format",
            field_name="email",
            field_value="invalid-email",
            suggestions=["Use a valid email format (e.g., user@example.com)"]
        )
        print(f"Validation error logging result: {result.is_successful}")
        print(f"Log ID: {result.log_id}")
    
    # Example 5: Log processing error
    print("\n⚙️ Processing Error Logging:")
    
    try:
        raise RuntimeError("Database connection failed")
    except Exception as error:
        result = main_module.error_logger.log_processing_error(
            error=error,
            operation_name="database_query",
            operation_data={"table": "users", "query": "SELECT * FROM users"},
            retry_count=2
        )
        print(f"Processing error logging result: {result.is_successful}")
        print(f"Log ID: {result.log_id}")

def show_structured_logging_benefits():
    """Show the benefits of structured error logging."""
    
    benefits = {
        "structured_logging": [
            "Comprehensive error context preservation",
            "Module, function, and parameter tracking",
            "Structured JSON log format",
            "Performance metrics integration"
        ],
        "error_context": [
            "Function call stack information",
            "Parameter values and types",
            "Module and class context",
            "Error categorization and severity"
        ],
        "debugging": [
            "Detailed stack traces",
            "Parameter validation context",
            "Performance timing information",
            "Error correlation and tracking"
        ],
        "monitoring": [
            "Error pattern analysis",
            "Performance trend tracking",
            "Function execution profiling",
            "System health monitoring"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate structured error logging
    asyncio.run(demonstrate_structured_error_logging())
    
    benefits = show_structured_logging_benefits()
    
    print("\n🎯 Key Structured Error Logging Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Structured error logging patterns completed successfully!") 