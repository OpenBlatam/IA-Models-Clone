from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import Union, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math
from decimal import Decimal, DivisionByZero, InvalidOperation
from concurrent.futures import ThreadPoolExecutor
import time
import json
import hashlib
import threading
from collections import OrderedDict
import statistics
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Math Service Core
Core mathematical operations with enhanced caching, error handling, and production features.
"""


logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Supported mathematical operations."""
    ADD = "add"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    SQRT = "sqrt"
    LOG = "log"
    BATCH = "batch"
    FACTORIAL = "factorial"
    GCD = "gcd"
    LCM = "lcm"


class CalculationMethod(Enum):
    """Available calculation methods."""
    BASIC = "basic"
    NUMPY = "numpy"
    MATH = "math"
    DECIMAL = "decimal"
    OPTIMIZED = "optimized"


class OperationStatus(Enum):
    """Operation execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class MathOperation:
    """Data structure for mathematical operations with enhanced features."""
    operation_type: OperationType
    operands: List[Union[int, float]]
    method: CalculationMethod = CalculationMethod.BASIC
    precision: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    operation_id: str = field(default_factory=lambda: hashlib.md5(f"{time.time()}_{id(threading.current_thread())}".encode()).hexdigest()[:12])
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def __post_init__(self) -> Any:
        """Validate operation parameters."""
        if not self.operands:
            raise ValueError("Operands cannot be empty")
        
        if self.precision < 0:
            raise ValueError("Precision must be non-negative")
        
        # Validate operands for specific operations
        if self.operation_type == OperationType.DIVIDE and len(self.operands) != 2:
            raise ValueError("Division requires exactly 2 operands")
        
        if self.operation_type == OperationType.POWER and len(self.operands) != 2:
            raise ValueError("Power operation requires exactly 2 operands")
        
        if self.operation_type in [OperationType.SQRT, OperationType.LOG] and len(self.operands) != 1:
            raise ValueError(f"{self.operation_type.value} requires exactly 1 operand")


@dataclass
class MathResult:
    """Result of mathematical operation with enhanced metadata."""
    value: Union[int, float, List[Union[int, float]]]
    operation: MathOperation
    execution_time: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    status: OperationStatus = OperationStatus.COMPLETED
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self) -> Any:
        """Validate result."""
        if not self.success and not self.error_message:
            self.error_message = "Unknown error occurred"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    result: MathResult
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def update_access(self) -> Any:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class PerformanceMetrics:
    """Performance metrics for the math service."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_execution_time: float = 0.0
    total_execution_time: float = 0.0
    operation_counts: Dict[str, int] = field(default_factory=dict)
    method_usage: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)


class LRUCache:
    """LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        
    """__init__ function."""
self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry, updating access statistics."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.update_access()
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return entry
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """Put entry in cache, evicting if necessary."""
        with self._lock:
            if key in self.cache:
                # Update existing entry
                self.cache.move_to_end(key)
                self.cache[key] = entry
            else:
                # Add new entry
                self.cache[key] = entry
                
                # Evict if cache is full
                if len(self.cache) > self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
    
    def clear(self) -> Any:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            if not self.cache:
                return {"size": 0, "max_size": self.max_size, "hit_rate": 0.0}
            
            total_access = sum(entry.access_count for entry in self.cache.values())
            avg_access = total_access / len(self.cache) if self.cache else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_access": total_access,
                "average_access": avg_access,
                "utilization": len(self.cache) / self.max_size
            }


class MathProcessor:
    """Core mathematical processing engine with enhanced features."""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        
    """__init__ function."""
self.max_workers = max_workers
        self.cache = LRUCache(cache_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = PerformanceMetrics()
        self._lock = threading.RLock()
        
        # Operation handlers
        self.operation_handlers: Dict[OperationType, Callable] = {
            OperationType.ADD: self._add_operation,
            OperationType.MULTIPLY: self._multiply_operation,
            OperationType.DIVIDE: self._divide_operation,
            OperationType.POWER: self._power_operation,
            OperationType.SQRT: self._sqrt_operation,
            OperationType.LOG: self._log_operation,
            OperationType.BATCH: self._batch_operation,
            OperationType.FACTORIAL: self._factorial_operation,
            OperationType.GCD: self._gcd_operation,
            OperationType.LCM: self._lcm_operation
        }
        
        logger.info(f"MathProcessor initialized with {max_workers} workers and {cache_size} cache size")
    
    def _generate_cache_key(self, operation: MathOperation) -> str:
        """Generate cache key for operation."""
        key_data = {
            "type": operation.operation_type.value,
            "operands": operation.operands,
            "method": operation.method.value,
            "precision": operation.precision
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _get_cached_result(self, operation: MathOperation) -> Optional[MathResult]:
        """Get cached result if available."""
        cache_key = self._generate_cache_key(operation)
        cache_entry = self.cache.get(cache_key)
        
        if cache_entry:
            # Create new result with cache hit flag
            cached_result = cache_entry.result
            result = MathResult(
                value=cached_result.value,
                operation=operation,
                execution_time=0.0,  # No execution time for cache hits
                success=cached_result.success,
                error_message=cached_result.error_message,
                metadata=cached_result.metadata,
                cache_hit=True,
                status=OperationStatus.CACHED
            )
            
            with self._lock:
                self.metrics.cache_hits += 1
            
            logger.debug(f"Cache hit for operation {operation.operation_type.value}")
            return result
        
        with self._lock:
            self.metrics.cache_misses += 1
        
        return None
    
    def _cache_result(self, operation: MathOperation, result: MathResult):
        """Cache operation result."""
        cache_key = self._generate_cache_key(operation)
        
        # Estimate size (simplified)
        size_bytes = len(str(result.value)) + len(str(operation.operands))
        
        cache_entry = CacheEntry(
            result=result,
            size_bytes=size_bytes
        )
        
        self.cache.put(cache_key, cache_entry)
        logger.debug(f"Cached result for operation {operation.operation_type.value}")
    
    async def process_operation(self, operation: MathOperation) -> MathResult:
        """Process a mathematical operation with enhanced error handling."""
        start_time = time.time()
        
        # Update metrics
        with self._lock:
            self.metrics.total_operations += 1
            self.metrics.operation_counts[operation.operation_type.value] = \
                self.metrics.operation_counts.get(operation.operation_type.value, 0) + 1
            self.metrics.method_usage[operation.method.value] = \
                self.metrics.method_usage.get(operation.method.value, 0) + 1
        
        try:
            # Check cache first
            cached_result = self._get_cached_result(operation)
            if cached_result:
                return cached_result
            
            # Execute operation
            operation.status = OperationStatus.RUNNING
            
            if operation.operation_type not in self.operation_handlers:
                raise ValueError(f"Unsupported operation type: {operation.operation_type}")
            
            handler = self.operation_handlers[operation.operation_type]
            value = await handler(operation)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = MathResult(
                value=value,
                operation=operation,
                execution_time=execution_time,
                success=True,
                status=OperationStatus.COMPLETED
            )
            
            # Cache successful result
            self._cache_result(operation, result)
            
            # Update metrics
            with self._lock:
                self.metrics.successful_operations += 1
                self.metrics.total_execution_time += execution_time
                self._update_average_execution_time(execution_time)
            
            logger.debug(f"Operation {operation.operation_type.value} completed in {execution_time:.6f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Update error metrics
            with self._lock:
                self.metrics.failed_operations += 1
                self.metrics.error_counts[operation.operation_type.value] = \
                    self.metrics.error_counts.get(operation.operation_type.value, 0) + 1
            
            logger.error(f"Operation {operation.operation_type.value} failed: {error_message}")
            
            return MathResult(
                value=0,
                operation=operation,
                execution_time=execution_time,
                success=False,
                error_message=error_message,
                status=OperationStatus.FAILED
            )
    
    async def _add_operation(self, operation: MathOperation) -> Union[int, float]:
        """Add operation with multiple calculation methods."""
        if operation.method == CalculationMethod.NUMPY:
            return float(np.sum(operation.operands))
        elif operation.method == CalculationMethod.DECIMAL:
            decimals = [Decimal(str(op)) for op in operation.operands]
            result = sum(decimals)
            return float(result.quantize(Decimal(f'1e-{operation.precision}')))
        else:  # BASIC or MATH
            return sum(operation.operands)
    
    async def _multiply_operation(self, operation: MathOperation) -> Union[int, float]:
        """Multiply operation with multiple calculation methods."""
        if operation.method == CalculationMethod.NUMPY:
            return float(np.prod(operation.operands))
        elif operation.method == CalculationMethod.DECIMAL:
            decimals = [Decimal(str(op)) for op in operation.operands]
            result = math.prod(decimals)
            return float(result.quantize(Decimal(f'1e-{operation.precision}')))
        else:  # BASIC or MATH
            return math.prod(operation.operands)
    
    async def _divide_operation(self, operation: MathOperation) -> float:
        """Divide operation with multiple calculation methods."""
        a, b = operation.operands[0], operation.operands[1]
        
        if b == 0:
            raise ValueError("Division by zero")
        
        if operation.method == CalculationMethod.NUMPY:
            return float(np.divide(a, b))
        elif operation.method == CalculationMethod.DECIMAL:
            decimal_a = Decimal(str(a))
            decimal_b = Decimal(str(b))
            result = decimal_a / decimal_b
            return float(result.quantize(Decimal(f'1e-{operation.precision}')))
        else:  # BASIC or MATH
            return a / b
    
    async def _power_operation(self, operation: MathOperation) -> float:
        """Power operation with multiple calculation methods."""
        base, exponent = operation.operands[0], operation.operands[1]
        
        if operation.method == CalculationMethod.NUMPY:
            return float(np.power(base, exponent))
        elif operation.method == CalculationMethod.DECIMAL:
            decimal_base = Decimal(str(base))
            decimal_exponent = Decimal(str(exponent))
            result = decimal_base ** decimal_exponent
            return float(result.quantize(Decimal(f'1e-{operation.precision}')))
        else:  # BASIC or MATH
            return math.pow(base, exponent)
    
    async def _sqrt_operation(self, operation: MathOperation) -> float:
        """Square root operation with multiple calculation methods."""
        x = operation.operands[0]
        
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        
        if operation.method == CalculationMethod.NUMPY:
            return float(np.sqrt(x))
        elif operation.method == CalculationMethod.DECIMAL:
            decimal_x = Decimal(str(x))
            result = decimal_x.sqrt()
            return float(result.quantize(Decimal(f'1e-{operation.precision}')))
        else:  # BASIC or MATH
            return math.sqrt(x)
    
    async def _log_operation(self, operation: MathOperation) -> float:
        """Logarithm operation with multiple calculation methods."""
        x = operation.operands[0]
        base = operation.operands[1] if len(operation.operands) > 1 else math.e
        
        if x <= 0:
            raise ValueError("Cannot calculate logarithm of non-positive number")
        
        if operation.method == CalculationMethod.NUMPY:
            if base == math.e:
                return float(np.log(x))
            else:
                return float(np.log(x) / np.log(base))
        elif operation.method == CalculationMethod.DECIMAL:
            decimal_x = Decimal(str(x))
            decimal_base = Decimal(str(base))
            result = decimal_x.ln() / decimal_base.ln()
            return float(result.quantize(Decimal(f'1e-{operation.precision}')))
        else:  # BASIC or MATH
            if base == math.e:
                return math.log(x)
            else:
                return math.log(x, base)
    
    async def _batch_operation(self, operation: MathOperation) -> List[Union[int, float]]:
        """Batch operation processing."""
        # For batch operations, we expect operands to be lists of operations
        # This is a simplified implementation
        results = []
        for operand in operation.operands:
            if isinstance(operand, (list, tuple)):
                # Treat as sub-operation
                sub_result = sum(operand) if operation.method == CalculationMethod.BASIC else float(np.sum(operand))
                results.append(sub_result)
            else:
                results.append(operand)
        return results
    
    async def _factorial_operation(self, operation: MathOperation) -> int:
        """Factorial operation."""
        n = int(operation.operands[0])
        
        if n < 0:
            raise ValueError("Cannot calculate factorial of negative number")
        
        if operation.method == CalculationMethod.NUMPY:
            return int(np.math.factorial(n))
        else:  # BASIC or MATH
            return math.factorial(n)
    
    async def _gcd_operation(self, operation: MathOperation) -> int:
        """Greatest Common Divisor operation."""
        if len(operation.operands) < 2:
            raise ValueError("GCD requires at least 2 operands")
        
        if operation.method == CalculationMethod.NUMPY:
            return int(np.gcd.reduce([int(op) for op in operation.operands]))
        else:  # BASIC or MATH
            result = math.gcd(int(operation.operands[0]), int(operation.operands[1]))
            for operand in operation.operands[2:]:
                result = math.gcd(result, int(operand))
            return result
    
    async def _lcm_operation(self, operation: MathOperation) -> int:
        """Least Common Multiple operation."""
        if len(operation.operands) < 2:
            raise ValueError("LCM requires at least 2 operands")
        
        if operation.method == CalculationMethod.NUMPY:
            return int(np.lcm.reduce([int(op) for op in operation.operands]))
        else:  # BASIC or MATH
            result = math.lcm(int(operation.operands[0]), int(operation.operands[1]))
            for operand in operation.operands[2:]:
                result = math.lcm(result, int(operand))
            return result
    
    def _update_average_execution_time(self, new_time: float):
        """Update average execution time."""
        total_ops = self.metrics.successful_operations
        if total_ops > 0:
            current_avg = self.metrics.average_execution_time
            new_avg = ((current_avg * (total_ops - 1)) + new_time) / total_ops
            self.metrics.average_execution_time = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            total_ops = self.metrics.total_operations
            cache_total = self.metrics.cache_hits + self.metrics.cache_misses
            
            return {
                "operations": {
                    "total": self.metrics.total_operations,
                    "successful": self.metrics.successful_operations,
                    "failed": self.metrics.failed_operations,
                    "success_rate": self.metrics.successful_operations / total_ops if total_ops > 0 else 0.0
                },
                "cache": {
                    "hits": self.metrics.cache_hits,
                    "misses": self.metrics.cache_misses,
                    "hit_rate": self.metrics.cache_hits / cache_total if cache_total > 0 else 0.0,
                    "stats": self.cache.get_stats()
                },
                "performance": {
                    "average_execution_time": self.metrics.average_execution_time,
                    "total_execution_time": self.metrics.total_execution_time
                },
                "usage": {
                    "operation_counts": dict(self.metrics.operation_counts),
                    "method_usage": dict(self.metrics.method_usage),
                    "error_counts": dict(self.metrics.error_counts)
                }
            }
    
    def clear_cache(self) -> Any:
        """Clear operation cache."""
        self.cache.clear()
        logger.info("Operation cache cleared")
    
    async def shutdown(self) -> Any:
        """Shutdown the processor gracefully."""
        logger.info("Shutting down MathProcessor...")
        self.executor.shutdown(wait=True)
        logger.info("MathProcessor shutdown completed")


class MathService:
    """High-level math service with enhanced features."""
    
    def __init__(self, processor: Optional[MathProcessor] = None):
        
    """__init__ function."""
self.processor = processor or MathProcessor()
    
    async def add(self, *operands: Union[int, float], 
                  method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Add multiple numbers."""
        operation = MathOperation(OperationType.ADD, list(operands), method)
        return await self.processor.process_operation(operation)
    
    async def multiply(self, *operands: Union[int, float],
                       method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Multiply multiple numbers."""
        operation = MathOperation(OperationType.MULTIPLY, list(operands), method)
        return await self.processor.process_operation(operation)
    
    async def divide(self, a: Union[int, float], b: Union[int, float],
                     method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Divide two numbers."""
        operation = MathOperation(OperationType.DIVIDE, [a, b], method)
        return await self.processor.process_operation(operation)
    
    async def power(self, base: Union[int, float], exponent: Union[int, float],
                    method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Calculate power."""
        operation = MathOperation(OperationType.POWER, [base, exponent], method)
        return await self.processor.process_operation(operation)
    
    async def sqrt(self, x: Union[int, float],
                   method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Calculate square root."""
        operation = MathOperation(OperationType.SQRT, [x], method)
        return await self.processor.process_operation(operation)
    
    async def log(self, x: Union[int, float], base: Union[int, float] = math.e,
                  method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Calculate logarithm."""
        operation = MathOperation(OperationType.LOG, [x, base], method)
        return await self.processor.process_operation(operation)
    
    async def factorial(self, n: int, method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Calculate factorial."""
        operation = MathOperation(OperationType.FACTORIAL, [n], method)
        return await self.processor.process_operation(operation)
    
    async def gcd(self, *operands: int, method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Calculate Greatest Common Divisor."""
        operation = MathOperation(OperationType.GCD, list(operands), method)
        return await self.processor.process_operation(operation)
    
    async def lcm(self, *operands: int, method: CalculationMethod = CalculationMethod.BASIC) -> MathResult:
        """Calculate Least Common Multiple."""
        operation = MathOperation(OperationType.LCM, list(operands), method)
        return await self.processor.process_operation(operation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.processor.get_performance_stats()
    
    async def shutdown(self) -> Any:
        """Shutdown the service."""
        await self.processor.shutdown()


def create_math_service(max_workers: int = 4, cache_size: int = 1000) -> MathService:
    """Factory function to create a math service."""
    processor = MathProcessor(max_workers=max_workers, cache_size=cache_size)
    return MathService(processor)


async def main():
    """Main function for testing."""
    # Create math service
    service = create_math_service()
    
    try:
        # Test operations
        result1 = await service.add(1, 2, 3, 4, 5, method=CalculationMethod.NUMPY)
        print(f"Add result: {result1.value}")
        
        result2 = await service.multiply(2, 3, 4, method=CalculationMethod.BASIC)
        print(f"Multiply result: {result2.value}")
        
        result3 = await service.divide(10, 3, method=CalculationMethod.DECIMAL)
        print(f"Divide result: {result3.value}")
        
        # Get statistics
        stats = service.get_stats()
        print(f"Service statistics: {stats}")
        
    finally:
        await service.shutdown()


match __name__:
    case "__main__":
    asyncio.run(main()) 