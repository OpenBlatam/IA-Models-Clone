from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
import time
import traceback
import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import hashlib
import secrets
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc
import psutil
import threading
from typing import Any, List, Dict, Optional
"""
Instagram Captions API v14.0 - Prioritized Error Handling and Edge Case Management
Advanced error handling with priority-based processing and comprehensive edge case coverage
"""


logger = logging.getLogger(__name__)

class ErrorPriority(Enum):
    """Error priority levels for handling order"""
    CRITICAL = 0      # System failure, immediate attention required
    HIGH = 1          # Service degradation, high impact
    MEDIUM = 2        # Functionality affected, moderate impact
    LOW = 3           # Minor issues, low impact
    INFO = 4          # Informational, no impact

class ErrorCategory(Enum):
    """Error categories for specialized handling"""
    SECURITY = "security"           # Security threats and violations
    VALIDATION = "validation"       # Input validation errors
    RESOURCE = "resource"           # Resource exhaustion and limits
    NETWORK = "network"             # Network connectivity issues
    AI_MODEL = "ai_model"           # AI model failures
    CACHE = "cache"                 # Cache-related issues
    DATABASE = "database"           # Database connection issues
    MEMORY = "memory"               # Memory management issues
    TIMEOUT = "timeout"             # Request timeout issues
    RATE_LIMIT = "rate_limit"       # Rate limiting violations
    BATCH_PROCESSING = "batch_processing" # Batch processing errors
    SYSTEM = "system"               # System-level errors

@dataclass
class PrioritizedError:
    """Structured error with priority and handling information"""
    error_id: str
    category: ErrorCategory
    priority: ErrorPriority
    message: str
    exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    handled: bool = False
    recovery_action: Optional[str] = None
    impact_score: float = 0.0

@dataclass
class EdgeCase:
    """Edge case definition with handling strategy"""
    name: str
    description: str
    detection_pattern: str
    severity: ErrorPriority
    handler: Callable
    recovery_strategy: str
    monitoring: bool = True

class PrioritizedErrorHandler:
    """Advanced error handler with priority-based processing"""
    
    def __init__(self) -> Any:
        self.error_queue: deque = deque()
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self.edge_cases: Dict[str, EdgeCase] = {}
        self.error_stats: Dict[str, int] = defaultdict(int)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.monitoring_enabled: bool = True
        self.max_queue_size: int = 10000
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Initialize handlers
        self._initialize_handlers()
        self._initialize_edge_cases()
        self._initialize_recovery_strategies()
        
        # Start processing thread
        self._start_processing_thread()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def _initialize_handlers(self) -> Any:
        """Initialize error handlers for each category"""
        self.error_handlers[ErrorCategory.SECURITY] = [
            self._handle_security_error,
            self._log_security_incident,
            self._trigger_security_alert
        ]
        
        self.error_handlers[ErrorCategory.VALIDATION] = [
            self._handle_validation_error,
            self._provide_user_feedback,
            self._log_validation_issue
        ]
        
        self.error_handlers[ErrorCategory.RESOURCE] = [
            self._handle_resource_error,
            self._trigger_resource_cleanup,
            self._scale_resources
        ]
        
        self.error_handlers[ErrorCategory.AI_MODEL] = [
            self._handle_ai_model_error,
            self._fallback_to_template,
            self._retry_with_different_model
        ]
        
        self.error_handlers[ErrorCategory.CACHE] = [
            self._handle_cache_error,
            self._rebuild_cache,
            self._fallback_to_database
        ]
        
        self.error_handlers[ErrorCategory.TIMEOUT] = [
            self._handle_timeout_error,
            self._extend_timeout,
            self._cancel_operation
        ]
    
    def _initialize_edge_cases(self) -> Any:
        """Initialize edge case definitions"""
        self.edge_cases = {
            "memory_exhaustion": EdgeCase(
                name="Memory Exhaustion",
                description="System running out of memory",
                detection_pattern=r"MemoryError|OutOfMemoryError",
                severity=ErrorPriority.CRITICAL,
                handler=self._handle_memory_exhaustion,
                recovery_strategy="Force garbage collection and reduce cache size"
            ),
            "network_timeout": EdgeCase(
                name="Network Timeout",
                description="Network request timeout",
                detection_pattern=r"TimeoutError|asyncio\.TimeoutError",
                severity=ErrorPriority.HIGH,
                handler=self._handle_network_timeout,
                recovery_strategy="Retry with exponential backoff"
            ),
            "model_loading_failure": EdgeCase(
                name="Model Loading Failure",
                description="AI model failed to load",
                detection_pattern=r"ModelNotFoundError|LoadError",
                severity=ErrorPriority.CRITICAL,
                handler=self._handle_model_loading_failure,
                recovery_strategy="Fallback to template-based generation"
            ),
            "cache_corruption": EdgeCase(
                name="Cache Corruption",
                description="Cache data corruption detected",
                detection_pattern=r"CacheCorruptionError|InvalidCacheData",
                severity=ErrorPriority.HIGH,
                handler=self._handle_cache_corruption,
                recovery_strategy="Clear cache and rebuild"
            ),
            "rate_limit_exceeded": EdgeCase(
                name="Rate Limit Exceeded",
                description="API rate limit exceeded",
                detection_pattern=r"RateLimitExceeded|TooManyRequests",
                severity=ErrorPriority.MEDIUM,
                handler=self._handle_rate_limit_exceeded,
                recovery_strategy="Implement exponential backoff"
            ),
            "invalid_input_format": EdgeCase(
                name="Invalid Input Format",
                description="Input data format is invalid",
                detection_pattern=r"ValueError|TypeError|JSONDecodeError",
                severity=ErrorPriority.LOW,
                handler=self._handle_invalid_input_format,
                recovery_strategy="Sanitize and validate input"
            ),
            "concurrent_access": EdgeCase(
                name="Concurrent Access",
                description="Concurrent access to shared resources",
                detection_pattern=r"ConcurrentModificationError|LockError",
                severity=ErrorPriority.MEDIUM,
                handler=self._handle_concurrent_access,
                recovery_strategy="Implement proper locking mechanism"
            ),
            "disk_space_full": EdgeCase(
                name="Disk Space Full",
                description="Disk space exhausted",
                detection_pattern=r"OSError.*No space left|DiskFullError",
                severity=ErrorPriority.CRITICAL,
                handler=self._handle_disk_space_full,
                recovery_strategy="Clean up temporary files and logs"
            ),
            "database_connection_lost": EdgeCase(
                name="Database Connection Lost",
                description="Database connection failure",
                detection_pattern=r"ConnectionError|DatabaseError|OperationalError",
                severity=ErrorPriority.HIGH,
                handler=self._handle_database_connection_lost,
                recovery_strategy="Reconnect with retry mechanism"
            ),
            "gpu_memory_exhaustion": EdgeCase(
                name="GPU Memory Exhaustion",
                description="GPU memory exhausted",
                detection_pattern=r"CUDA.*out of memory|GPU.*memory",
                severity=ErrorPriority.HIGH,
                handler=self._handle_gpu_memory_exhaustion,
                recovery_strategy="Switch to CPU or reduce batch size"
            )
        }
    
    def _initialize_recovery_strategies(self) -> Any:
        """Initialize recovery strategies"""
        self.recovery_strategies = {
            "retry": self._retry_operation,
            "fallback": self._fallback_operation,
            "degrade": self._degrade_service,
            "restart": self._restart_component,
            "cleanup": self._cleanup_resources,
            "scale": self._scale_resources,
            "cache_clear": self._clear_cache,
            "memory_cleanup": self._cleanup_memory,
            "timeout_extend": self._extend_timeout,
            "rate_limit_backoff": self._implement_backoff
        }
    
    def _start_processing_thread(self) -> Any:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        """Start background error processing thread"""
        self.processing_thread = threading.Thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            target=self._process_error_queue,
            daemon=True,
            name="ErrorProcessor"
        )
        self.processing_thread.start()
    
    def _process_error_queue(self) -> Any:
        """Background thread for processing error queue"""
        while not self.shutdown_event.is_set():
            try:
                if self.error_queue:
                    error = self.error_queue.popleft()
                    self._handle_error(error)
                else:
                    time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
                time.sleep(1)  # Longer delay on error
    
    def _handle_error(self, error: PrioritizedError):
        """Handle a single error with appropriate strategy"""
        try:
            # Update error stats
            self.error_stats[error.category.value] += 1
            
            # Check for edge cases
            edge_case = self._detect_edge_case(error)
            if edge_case:
                edge_case.handler(error)
                return
            
            # Apply category-specific handlers
            handlers = self.error_handlers.get(error.category, [])
            for handler in handlers:
                try:
                    handler(error)
                except Exception as handler_error:
                    logger.error(f"Handler {handler.__name__} failed: {handler_error}")
            
            # Apply recovery strategy if available
            if error.recovery_action and error.recovery_action in self.recovery_strategies:
                try:
                    self.recovery_strategies[error.recovery_action](error)
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy {error.recovery_action} failed: {recovery_error}")
            
            # Mark as handled
            error.handled = True
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
    
    def _detect_edge_case(self, error: PrioritizedError) -> Optional[EdgeCase]:
        """Detect if error matches any edge case patterns"""
        error_message = str(error.exception) if error.exception else error.message
        
        for edge_case in self.edge_cases.values():
            if re.search(edge_case.detection_pattern, error_message, re.IGNORECASE):
                return edge_case
        
        return None
    
    def add_error(self, error: PrioritizedError):
        """Add error to processing queue with priority ordering"""
        if len(self.error_queue) >= self.max_queue_size:
            # Remove lowest priority error if queue is full
            self.error_queue.remove(min(self.error_queue, key=lambda x: x.priority.value))
        
        # Insert based on priority (lower value = higher priority)
        insert_index = 0
        for i, existing_error in enumerate(self.error_queue):
            if error.priority.value < existing_error.priority.value:
                insert_index = i
                break
            elif error.priority.value == existing_error.priority.value:
                # Same priority, insert after existing ones
                insert_index = i + 1
        
        self.error_queue.insert(insert_index, error)
    
    def create_error(self, category: ErrorCategory, priority: ErrorPriority, 
                    message: str, exception: Optional[Exception] = None,
                    context: Optional[Dict[str, Any]] = None, 
                    request_id: Optional[str] = None) -> PrioritizedError:
        """Create a new prioritized error"""
        error_id = f"{category.value}-{int(time.time() * 1000)}-{secrets.token_hex(4)}"
        
        # Calculate impact score based on priority and category
        impact_score = self._calculate_impact_score(priority, category)
        
        # Determine recovery action based on category
        recovery_action = self._determine_recovery_action(category)
        
        error = PrioritizedError(
            error_id=error_id,
            category=category,
            priority=priority,
            message=message,
            exception=exception,
            context=context or {},
            request_id=request_id,
            impact_score=impact_score,
            recovery_action=recovery_action
        )
        
        self.add_error(error)
        return error
    
    def _calculate_impact_score(self, priority: ErrorPriority, category: ErrorCategory) -> float:
        """Calculate impact score for error prioritization"""
        base_score = 10 - priority.value  # Higher priority = higher score
        
        # Category multipliers
        category_multipliers = {
            ErrorCategory.SECURITY: 2.0,
            ErrorCategory.RESOURCE: 1.8,
            ErrorCategory.AI_MODEL: 1.5,
            ErrorCategory.NETWORK: 1.3,
            ErrorCategory.VALIDATION: 1.0,
            ErrorCategory.CACHE: 0.8,
            ErrorCategory.SYSTEM: 1.2
        }
        
        return base_score * category_multipliers.get(category, 1.0)
    
    def _determine_recovery_action(self, category: ErrorCategory) -> Optional[str]:
        """Determine appropriate recovery action for error category"""
        recovery_mapping = {
            ErrorCategory.SECURITY: "restart",
            ErrorCategory.RESOURCE: "scale",
            ErrorCategory.AI_MODEL: "fallback",
            ErrorCategory.NETWORK: "retry",
            ErrorCategory.CACHE: "cache_clear",
            ErrorCategory.MEMORY: "memory_cleanup",
            ErrorCategory.TIMEOUT: "timeout_extend",
            ErrorCategory.RATE_LIMIT: "rate_limit_backoff",
            ErrorCategory.VALIDATION: None,  # No recovery needed
            ErrorCategory.SYSTEM: "restart"
        }
        
        return recovery_mapping.get(category)
    
    # Error handlers for different categories
    def _handle_security_error(self, error: PrioritizedError):
        """Handle security-related errors"""
        logger.critical(f"SECURITY ERROR: {error.message}")
        # Trigger security alert
        self._trigger_security_alert(error)
    
    def _handle_validation_error(self, error: PrioritizedError):
        """Handle validation errors"""
        logger.warning(f"VALIDATION ERROR: {error.message}")
        # Provide user feedback
        self._provide_user_feedback(error)
    
    def _handle_resource_error(self, error: PrioritizedError):
        """Handle resource-related errors"""
        logger.error(f"RESOURCE ERROR: {error.message}")
        # Trigger resource cleanup
        self._trigger_resource_cleanup(error)
    
    def _handle_ai_model_error(self, error: PrioritizedError):
        """Handle AI model errors"""
        logger.error(f"AI MODEL ERROR: {error.message}")
        # Fallback to template generation
        self._fallback_to_template(error)
    
    def _handle_cache_error(self, error: PrioritizedError):
        """Handle cache-related errors"""
        logger.warning(f"CACHE ERROR: {error.message}")
        # Rebuild cache
        self._rebuild_cache(error)
    
    def _handle_timeout_error(self, error: PrioritizedError):
        """Handle timeout errors"""
        logger.warning(f"TIMEOUT ERROR: {error.message}")
        # Extend timeout
        self._extend_timeout(error)
    
    # Edge case handlers
    def _handle_memory_exhaustion(self, error: PrioritizedError):
        """Handle memory exhaustion edge case"""
        logger.critical("MEMORY EXHAUSTION DETECTED - Initiating cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        self._clear_cache(error)
        
        # Reduce memory usage
        self._cleanup_memory(error)
        
        # Monitor memory usage
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"Memory usage after cleanup: {memory_usage}%")
    
    def _handle_network_timeout(self, error: PrioritizedError):
        """Handle network timeout edge case"""
        logger.warning("NETWORK TIMEOUT DETECTED - Implementing retry strategy")
        
        # Implement exponential backoff
        self._implement_backoff(error)
        
        # Extend timeout for retry
        self._extend_timeout(error)
    
    def _handle_model_loading_failure(self, error: PrioritizedError):
        """Handle AI model loading failure edge case"""
        logger.critical("MODEL LOADING FAILURE DETECTED - Switching to fallback")
        
        # Switch to template-based generation
        self._fallback_to_template(error)
        
        # Attempt model reload in background
        asyncio.create_task(self._reload_model_async())
    
    def _handle_cache_corruption(self, error: PrioritizedError):
        """Handle cache corruption edge case"""
        logger.error("CACHE CORRUPTION DETECTED - Clearing and rebuilding")
        
        # Clear corrupted cache
        self._clear_cache(error)
        
        # Rebuild cache from database
        self._rebuild_cache(error)
    
    def _handle_rate_limit_exceeded(self, error: PrioritizedError):
        """Handle rate limit exceeded edge case"""
        logger.warning("RATE LIMIT EXCEEDED - Implementing backoff")
        
        # Implement exponential backoff
        self._implement_backoff(error)
        
        # Queue request for later processing
        self._queue_for_later_processing(error)
    
    def _handle_invalid_input_format(self, error: PrioritizedError):
        """Handle invalid input format edge case"""
        logger.info("INVALID INPUT FORMAT - Sanitizing input")
        
        # Sanitize input data
        self._sanitize_input(error)
        
        # Retry with sanitized input
        self._retry_operation(error)
    
    def _handle_concurrent_access(self, error: PrioritizedError):
        """Handle concurrent access edge case"""
        logger.warning("CONCURRENT ACCESS DETECTED - Implementing locking")
        
        # Implement proper locking mechanism
        self._implement_locking(error)
        
        # Retry operation with lock
        self._retry_operation(error)
    
    def _handle_disk_space_full(self, error: PrioritizedError):
        """Handle disk space full edge case"""
        logger.critical("DISK SPACE FULL - Cleaning up files")
        
        # Clean up temporary files
        self._cleanup_temp_files(error)
        
        # Clean up old logs
        self._cleanup_old_logs(error)
        
        # Monitor disk usage
        disk_usage = psutil.disk_usage('/').percent
        logger.info(f"Disk usage after cleanup: {disk_usage}%")
    
    def _handle_database_connection_lost(self, error: PrioritizedError):
        """Handle database connection lost edge case"""
        logger.error("DATABASE CONNECTION LOST - Attempting reconnection")
        
        # Attempt reconnection with retry
        self._reconnect_database(error)
        
        # Fallback to cache if available
        self._fallback_to_cache(error)
    
    def _handle_gpu_memory_exhaustion(self, error: PrioritizedError):
        """Handle GPU memory exhaustion edge case"""
        logger.warning("GPU MEMORY EXHAUSTED - Switching to CPU")
        
        # Switch to CPU processing
        self._switch_to_cpu(error)
        
        # Reduce batch size
        self._reduce_batch_size(error)
    
    # Recovery strategies
    def _retry_operation(self, error: PrioritizedError):
        """Retry operation with exponential backoff"""
        if error.retry_count < error.max_retries:
            error.retry_count += 1
            delay = min(2 ** error.retry_count, 60)  # Max 60 seconds
            logger.info(f"Retrying operation in {delay} seconds (attempt {error.retry_count})")
            time.sleep(delay)
            return True
        return False
    
    def _fallback_operation(self, error: PrioritizedError):
        """Fallback to alternative operation"""
        logger.info("Implementing fallback operation")
        # Implementation depends on specific operation
        return True
    
    def _degrade_service(self, error: PrioritizedError):
        """Degrade service gracefully"""
        logger.warning("Degrading service to maintain availability")
        # Reduce functionality but maintain core service
        return True
    
    def _restart_component(self, error: PrioritizedError):
        """Restart failed component"""
        logger.warning("Restarting failed component")
        # Implementation depends on component
        return True
    
    def _cleanup_resources(self, error: PrioritizedError):
        """Clean up system resources"""
        logger.info("Cleaning up system resources")
        gc.collect()
        return True
    
    def _scale_resources(self, error: PrioritizedError):
        """Scale resources up or down"""
        logger.info("Scaling resources")
        # Implementation depends on infrastructure
        return True
    
    def _clear_cache(self, error: PrioritizedError):
        """Clear system cache"""
        logger.info("Clearing system cache")
        # Implementation depends on cache system
        return True
    
    def _cleanup_memory(self, error: PrioritizedError):
        """Clean up memory"""
        logger.info("Cleaning up memory")
        gc.collect()
        return True
    
    def _extend_timeout(self, error: PrioritizedError):
        """Extend operation timeout"""
        logger.info("Extending operation timeout")
        # Implementation depends on operation
        return True
    
    def _implement_backoff(self, error: PrioritizedError):
        """Implement exponential backoff"""
        logger.info("Implementing exponential backoff")
        # Implementation depends on rate limiting
        return True
    
    # Utility methods
    def _trigger_security_alert(self, error: PrioritizedError):
        """Trigger security alert"""
        logger.critical(f"SECURITY ALERT: {error.message}")
        # Implementation for security alerting
    
    def _provide_user_feedback(self, error: PrioritizedError):
        """Provide user-friendly feedback"""
        logger.info(f"User feedback: {error.message}")
        # Implementation for user feedback
    
    def _trigger_resource_cleanup(self, error: PrioritizedError):
        """Trigger resource cleanup"""
        logger.info("Triggering resource cleanup")
        # Implementation for resource cleanup
    
    def _fallback_to_template(self, error: PrioritizedError):
        """Fallback to template-based generation"""
        logger.info("Falling back to template-based generation")
        # Implementation for template fallback
    
    def _rebuild_cache(self, error: PrioritizedError):
        """Rebuild cache from source"""
        logger.info("Rebuilding cache from source")
        # Implementation for cache rebuilding
    
    def _fallback_to_database(self, error: PrioritizedError):
        """Fallback to database when cache fails"""
        logger.info("Falling back to database")
        # Implementation for database fallback
    
    def _cancel_operation(self, error: PrioritizedError):
        """Cancel ongoing operation"""
        logger.info("Cancelling operation")
        # Implementation for operation cancellation
    
    def _sanitize_input(self, error: PrioritizedError):
        """Sanitize input data"""
        logger.info("Sanitizing input data")
        # Implementation for input sanitization
    
    def _implement_locking(self, error: PrioritizedError):
        """Implement proper locking mechanism"""
        logger.info("Implementing locking mechanism")
        # Implementation for locking
    
    def _cleanup_temp_files(self, error: PrioritizedError):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files")
        # Implementation for temp file cleanup
    
    def _cleanup_old_logs(self, error: PrioritizedError):
        """Clean up old log files"""
        logger.info("Cleaning up old log files")
        # Implementation for log cleanup
    
    def _reconnect_database(self, error: PrioritizedError):
        """Reconnect to database"""
        logger.info("Reconnecting to database")
        # Implementation for database reconnection
    
    def _fallback_to_cache(self, error: PrioritizedError):
        """Fallback to cache when database fails"""
        logger.info("Falling back to cache")
        # Implementation for cache fallback
    
    def _switch_to_cpu(self, error: PrioritizedError):
        """Switch processing to CPU"""
        logger.info("Switching to CPU processing")
        # Implementation for CPU switch
    
    def _reduce_batch_size(self, error: PrioritizedError):
        """Reduce batch size to conserve memory"""
        logger.info("Reducing batch size")
        # Implementation for batch size reduction
    
    def _queue_for_later_processing(self, error: PrioritizedError):
        """Queue request for later processing"""
        logger.info("Queueing request for later processing")
        # Implementation for request queuing
    
    async def _reload_model_async(self) -> Any:
        """Reload AI model asynchronously"""
        logger.info("Reloading AI model asynchronously")
        # Implementation for model reload
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "total_errors": sum(self.error_stats.values()),
            "error_counts": dict(self.error_stats),
            "queue_size": len(self.error_queue),
            "handled_errors": len([e for e in self.error_queue if e.handled]),
            "pending_errors": len([e for e in self.error_queue if not e.handled]),
            "priority_distribution": self._get_priority_distribution(),
            "category_distribution": self._get_category_distribution(),
            "recovery_success_rate": self._get_recovery_success_rate()
        }
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of errors by priority"""
        distribution = defaultdict(int)
        for error in self.error_queue:
            distribution[error.priority.value] += 1
        return dict(distribution)
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of errors by category"""
        distribution = defaultdict(int)
        for error in self.error_queue:
            distribution[error.category.value] += 1
        return dict(distribution)
    
    def _get_recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        total_errors = len(self.error_queue)
        if total_errors == 0:
            return 100.0
        
        handled_errors = len([e for e in self.error_queue if e.handled])
        return (handled_errors / total_errors) * 100
    
    def shutdown(self) -> Any:
        """Shutdown error handler gracefully"""
        logger.info("Shutting down error handler")
        self.shutdown_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

# Global prioritized error handler
prioritized_error_handler = PrioritizedErrorHandler()

# Context managers for prioritized error handling
@contextmanager
def prioritized_error_context(operation: str, category: ErrorCategory, 
                            priority: ErrorPriority = ErrorPriority.MEDIUM,
                            request_id: Optional[str] = None):
    """Context manager for prioritized error handling"""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error = prioritized_error_handler.create_error(
            category=category,
            priority=priority,
            message=f"Error in {operation}: {str(e)}",
            exception=e,
            context={"operation": operation, "duration": time.time() - start_time},
            request_id=request_id
        )
        raise
    finally:
        # Record performance metrics
        duration = time.time() - start_time
        if duration > 1.0:  # Log slow operations
            logger.warning(f"Slow operation detected: {operation} took {duration:.2f}s")

@asynccontextmanager
async def async_prioritized_error_context(operation: str, category: ErrorCategory,
                                        priority: ErrorPriority = ErrorPriority.MEDIUM,
                                        request_id: Optional[str] = None):
    """Async context manager for prioritized error handling"""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error = prioritized_error_handler.create_error(
            category=category,
            priority=priority,
            message=f"Async error in {operation}: {str(e)}",
            exception=e,
            context={"operation": operation, "duration": time.time() - start_time},
            request_id=request_id
        )
        raise
    finally:
        # Record performance metrics
        duration = time.time() - start_time
        if duration > 1.0:  # Log slow operations
            logger.warning(f"Slow async operation detected: {operation} took {duration:.2f}s") 