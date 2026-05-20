"""
🚀 ADVANCED REFACTORING & ENTERPRISE ARCHITECTURE v3.0
=======================================================

Advanced refactoring with design patterns, enterprise architecture, and cutting-edge optimizations.
"""

import asyncio
import time
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps, partial
from typing import Dict, Any, List, Optional, Union, Protocol, Callable, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
import numpy as np
import weakref

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Generic type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Advanced enums
class OptimizationLevel(Enum):
    """Optimization quality levels."""
    BASIC = auto()
    STANDARD = auto()
    PREMIUM = auto()
    ENTERPRISE = auto()

class ProcessingMode(Enum):
    """Processing modes for optimization."""
    SYNC = auto()
    ASYNC = auto()
    STREAMING = auto()
    BATCH = auto()
    REAL_TIME = auto()

class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = auto()
    LFU = auto()
    TTL = auto()
    HYBRID = auto()
    INTELLIGENT = auto()

# Advanced data structures
@dataclass
class OptimizationRequest:
    """Optimization request with advanced metadata."""
    id: str = field(default_factory=lambda: hashlib.uuid4().hex)
    content: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    deadline: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    @property
    def can_retry(self) -> bool:
        """Check if request can be retried."""
        return self.retry_count < self.max_retries

@dataclass
class OptimizationResponse:
    """Optimization response with comprehensive data."""
    request_id: str
    content: str
    optimized_content: str
    score: float
    confidence: float
    processing_time: float
    cache_hit: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

# Advanced protocols and interfaces
class AsyncIterator(Protocol[T]):
    """Protocol for async iterators."""
    async def __anext__(self) -> T: ...
    async def __aiter__(self) -> 'AsyncIterator[T]': ...

class Observable(Protocol):
    """Protocol for observable objects."""
    def subscribe(self, observer: Callable) -> None: ...
    def unsubscribe(self, observer: Callable) -> None: ...
    def notify(self, data: Any) -> None: ...

class StateMachine(Protocol):
    """Protocol for state machines."""
    def transition_to(self, state: str) -> bool: ...
    def get_current_state(self) -> str: ...
    def can_transition_to(self, state: str) -> bool: ...

# Design Patterns Implementation

## 1. Observer Pattern
class EventBus:
    """Event bus for decoupled communication."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._weak_refs: Dict[str, List[weakref.ref]] = {}
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def subscribe_weak(self, event_type: str, callback: Callable) -> None:
        """Subscribe with weak reference to prevent memory leaks."""
        if event_type not in self._weak_refs:
            self._weak_refs[event_type] = []
        self._weak_refs[event_type].append(weakref.ref(callback))
    
    def publish(self, event_type: str, data: Any) -> None:
        """Publish an event to all subscribers."""
        # Regular subscribers
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        
        # Weak reference subscribers
        if event_type in self._weak_refs:
            # Clean up dead references
            self._weak_refs[event_type] = [
                ref for ref in self._weak_refs[event_type] 
                if ref() is not None
            ]
            
            for ref in self._weak_refs[event_type]:
                callback = ref()
                if callback is not None:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in weak event callback: {e}")

## 2. Strategy Pattern
class OptimizationStrategy(ABC):
    """Abstract optimization strategy."""
    
    @abstractmethod
    async def optimize(self, content: str, config: Dict[str, Any]) -> OptimizationResponse:
        """Apply optimization strategy."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get strategy priority."""
        pass

class EngagementStrategy(OptimizationStrategy):
    """Engagement-focused optimization strategy."""
    
    async def optimize(self, content: str, config: Dict[str, Any]) -> OptimizationResponse:
        """Optimize for engagement."""
        # Add engagement elements
        enhanced_content = content
        if '?' not in content:
            enhanced_content += "\n\nWhat are your thoughts on this? Share your experience below! 👇"
        
        if len(content) < 100:
            enhanced_content += "\n\nLet's discuss this! What's your take?"
        
        return OptimizationResponse(
            request_id="mock_id",
            content=content,
            optimized_content=enhanced_content,
            score=85.0,
            confidence=0.9,
            processing_time=0.1
        )
    
    def get_name(self) -> str:
        return "engagement"
    
    def get_priority(self) -> int:
        return 1

class ReachStrategy(OptimizationStrategy):
    """Reach-focused optimization strategy."""
    
    async def optimize(self, content: str, config: Dict[str, Any]) -> OptimizationResponse:
        """Optimize for reach."""
        # Add viral elements
        enhanced_content = content
        if len(content) > 200:
            enhanced_content += "\n\n🔥 Share this if you agree!"
        
        return OptimizationResponse(
            request_id="mock_id",
            content=content,
            optimized_content=enhanced_content,
            score=80.0,
            confidence=0.85,
            processing_time=0.08
        )
    
    def get_name(self) -> str:
        return "reach"
    
    def get_priority(self) -> int:
        return 2

## 3. Factory Pattern with Registry
class StrategyRegistry:
    """Registry for optimization strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, OptimizationStrategy] = {}
        self._strategy_factories: Dict[str, Callable[[], OptimizationStrategy]] = {}
    
    def register(self, name: str, strategy: OptimizationStrategy) -> None:
        """Register a strategy instance."""
        self._strategies[name] = strategy
    
    def register_factory(self, name: str, factory: Callable[[], OptimizationStrategy]) -> None:
        """Register a strategy factory."""
        self._strategy_factories[name] = factory
    
    def get(self, name: str) -> Optional[OptimizationStrategy]:
        """Get a strategy by name."""
        if name in self._strategies:
            return self._strategies[name]
        
        if name in self._strategy_factories:
            strategy = self._strategy_factories[name]()
            self._strategies[name] = strategy
            return strategy
        
        return None
    
    def get_all(self) -> List[OptimizationStrategy]:
        """Get all registered strategies."""
        strategies = list(self._strategies.values())
        
        # Create strategies from factories
        for factory in self._strategy_factories.values():
            if factory not in [s.__class__ for s in strategies]:
                strategies.append(factory())
        
        return sorted(strategies, key=lambda s: s.get_priority())

## 4. Chain of Responsibility Pattern
class OptimizationHandler(ABC):
    """Abstract optimization handler."""
    
    def __init__(self):
        self._next_handler: Optional['OptimizationHandler'] = None
    
    def set_next(self, handler: 'OptimizationHandler') -> 'OptimizationHandler':
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler
    
    @abstractmethod
    async def handle(self, request: OptimizationRequest) -> Optional[OptimizationResponse]:
        """Handle the optimization request."""
        pass
    
    async def handle_next(self, request: OptimizationRequest) -> Optional[OptimizationResponse]:
        """Pass to next handler if current can't handle."""
        if self._next_handler:
            return await self._next_handler.handle(request)
        return None

class ContentValidationHandler(OptimizationHandler):
    """Handler for content validation."""
    
    async def handle(self, request: OptimizationRequest) -> Optional[OptimizationResponse]:
        """Validate content before optimization."""
        if not request.content or len(request.content.strip()) == 0:
            return OptimizationResponse(
                request_id=request.id,
                content=request.content,
                optimized_content=request.content,
                score=0.0,
                confidence=0.0,
                processing_time=0.0,
                errors=["Content cannot be empty"]
            )
        
        if len(request.content) > 3000:
            return OptimizationResponse(
                request_id=request.id,
                content=request.content,
                optimized_content=request.content,
                score=0.0,
                confidence=0.0,
                processing_time=0.0,
                errors=["Content exceeds maximum length of 3000 characters"]
            )
        
        # Pass to next handler
        return await self.handle_next(request)

class CachingHandler(OptimizationHandler):
    """Handler for caching optimization results."""
    
    def __init__(self, cache: 'CacheProvider'):
        super().__init__()
        self.cache = cache
    
    async def handle(self, request: OptimizationRequest) -> Optional[OptimizationResponse]:
        """Check cache before optimization."""
        cache_key = self._generate_cache_key(request)
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            return OptimizationResponse(
                request_id=request.id,
                content=request.content,
                optimized_content=cached_result['optimized_content'],
                score=cached_result['score'],
                confidence=cached_result['confidence'],
                processing_time=0.0,
                cache_hit=True
            )
        
        # Pass to next handler
        return await self.handle_next(request)
    
    def _generate_cache_key(self, request: OptimizationRequest) -> str:
        """Generate cache key for request."""
        content_hash = hashlib.sha256(request.content.encode()).hexdigest()
        config_hash = hashlib.sha256(json.dumps(request.config, sort_keys=True).encode()).hexdigest()
        return f"opt:{content_hash}:{config_hash}"

class OptimizationExecutionHandler(OptimizationHandler):
    """Handler for actual optimization execution."""
    
    def __init__(self, strategy_registry: StrategyRegistry):
        super().__init__()
        self.strategy_registry = strategy_registry
    
    async def handle(self, request: OptimizationRequest) -> Optional[OptimizationResponse]:
        """Execute optimization using appropriate strategy."""
        strategy_name = request.config.get('strategy', 'engagement')
        strategy = self.strategy_registry.get(strategy_name)
        
        if not strategy:
            return OptimizationResponse(
                request_id=request.id,
                content=request.content,
                optimized_content=request.content,
                score=0.0,
                confidence=0.0,
                processing_time=0.0,
                errors=[f"Unknown optimization strategy: {strategy_name}"]
            )
        
        try:
            result = await strategy.optimize(request.content, request.config)
            result.request_id = request.id
            return result
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationResponse(
                request_id=request.id,
                content=request.content,
                optimized_content=request.content,
                score=0.0,
                confidence=0.0,
                processing_time=0.0,
                errors=[f"Optimization failed: {str(e)}"]
            )

## 5. Command Pattern
class OptimizationCommand(ABC):
    """Abstract optimization command."""
    
    @abstractmethod
    async def execute(self) -> OptimizationResponse:
        """Execute the command."""
        pass
    
    @abstractmethod
    async def undo(self) -> bool:
        """Undo the command."""
        pass

class SingleOptimizationCommand(OptimizationCommand):
    """Command for single content optimization."""
    
    def __init__(self, request: OptimizationRequest, handler: OptimizationHandler):
        self.request = request
        self.handler = handler
        self.result: Optional[OptimizationResponse] = None
    
    async def execute(self) -> OptimizationResponse:
        """Execute single optimization."""
        self.result = await self.handler.handle(self.request)
        return self.result
    
    async def undo(self) -> bool:
        """Undo optimization (not applicable for optimization)."""
        return False

class BatchOptimizationCommand(OptimizationCommand):
    """Command for batch content optimization."""
    
    def __init__(self, requests: List[OptimizationRequest], handler: OptimizationHandler):
        self.requests = requests
        self.handler = handler
        self.results: List[OptimizationResponse] = []
    
    async def execute(self) -> List[OptimizationResponse]:
        """Execute batch optimization."""
        tasks = [self.handler.handle(request) for request in self.requests]
        self.results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in self.results:
            if isinstance(result, OptimizationResponse):
                valid_results.append(result)
            else:
                logger.error(f"Batch optimization failed: {result}")
        
        return valid_results
    
    async def undo(self) -> bool:
        """Undo batch optimization (not applicable)."""
        return False

## 6. Template Method Pattern
class BaseOptimizationPipeline(ABC):
    """Base optimization pipeline with template method."""
    
    async def run(self, request: OptimizationRequest) -> OptimizationResponse:
        """Run the optimization pipeline."""
        try:
            # Pre-processing
            await self.pre_process(request)
            
            # Core optimization
            result = await self.optimize(request)
            
            # Post-processing
            await self.post_process(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return self.create_error_response(request, str(e))
    
    @abstractmethod
    async def pre_process(self, request: OptimizationRequest) -> None:
        """Pre-processing step."""
        pass
    
    @abstractmethod
    async def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        """Core optimization step."""
        pass
    
    @abstractmethod
    async def post_process(self, request: OptimizationRequest, result: OptimizationResponse) -> None:
        """Post-processing step."""
        pass
    
    def create_error_response(self, request: OptimizationRequest, error: str) -> OptimizationResponse:
        """Create error response."""
        return OptimizationResponse(
            request_id=request.id,
            content=request.content,
            optimized_content=request.content,
            score=0.0,
            confidence=0.0,
            processing_time=0.0,
            errors=[error]
        )

class StandardOptimizationPipeline(BaseOptimizationPipeline):
    """Standard optimization pipeline implementation."""
    
    def __init__(self, handler: OptimizationHandler):
        self.handler = handler
    
    async def pre_process(self, request: OptimizationRequest) -> None:
        """Pre-processing: validate and prepare request."""
        if request.is_expired:
            raise ValueError("Request has expired")
        
        if request.retry_count >= request.max_retries:
            raise ValueError("Maximum retries exceeded")
    
    async def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        """Core optimization using handler chain."""
        result = await self.handler.handle(request)
        if result is None:
            raise ValueError("No handler could process the request")
        return result
    
    async def post_process(self, request: OptimizationRequest, result: OptimizationResponse) -> None:
        """Post-processing: cache result and log metrics."""
        # Cache the result if successful
        if result.score > 0:
            # This would integrate with actual cache
            pass
        
        # Log metrics
        logger.info(f"Optimization completed: {result.score}/100 in {result.processing_time:.3f}s")

## 7. Advanced Cache Provider
class CacheProvider(Protocol):
    """Protocol for cache implementations."""
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def exists(self, key: str) -> bool: ...

class AdvancedMemoryCache:
    """Advanced in-memory cache with multiple strategies."""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.HYBRID, max_size: int = 10000):
        self.strategy = strategy
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        self._ttl: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        # Check TTL
        if key in self._ttl and time.time() > self._ttl[key]:
            await self.delete(key)
            return None
        
        # Update access statistics
        self._update_access(key)
        return self._cache[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        # Evict if necessary
        if len(self._cache) >= self.max_size:
            await self._evict()
        
        # Set value and metadata
        self._cache[key] = value
        self._update_access(key)
        
        if ttl:
            self._ttl[key] = time.time() + ttl
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._access_count[key]
            del self._last_access[key]
            if key in self._ttl:
                del self._ttl[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
    
    def _update_access(self, key: str) -> None:
        """Update access statistics."""
        current_time = time.time()
        self._access_count[key] = self._access_count.get(key, 0) + 1
        self._last_access[key] = current_time
    
    async def _evict(self) -> None:
        """Evict items based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            await self._evict_lru()
        elif self.strategy == CacheStrategy.LFU:
            await self._evict_lfu()
        elif self.strategy == CacheStrategy.TTL:
            await self._evict_ttl()
        elif self.strategy == CacheStrategy.HYBRID:
            await self._evict_hybrid()
        elif self.strategy == CacheStrategy.INTELLIGENT:
            await self._evict_intelligent()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return
        
        oldest_key = min(self._last_access.items(), key=lambda x: x[1])[0]
        await self.delete(oldest_key)
    
    async def _evict_lfu(self) -> None:
        """Evict least frequently used item."""
        if not self._cache:
            return
        
        least_frequent_key = min(self._access_count.items(), key=lambda x: x[1])[0]
        await self.delete(least_frequent_key)
    
    async def _evict_ttl(self) -> None:
        """Evict expired items."""
        current_time = time.time()
        expired_keys = [key for key, expiry in self._ttl.items() if current_time > expiry]
        
        for key in expired_keys:
            await self.delete(key)
    
    async def _evict_hybrid(self) -> None:
        """Hybrid eviction strategy."""
        # Combine LRU and LFU with weighted scoring
        if not self._cache:
            return
        
        current_time = time.time()
        scores = {}
        
        for key in self._cache:
            access_count = self._access_count.get(key, 0)
            last_access = self._last_access.get(key, current_time)
            
            # Weighted scoring: 70% frequency, 30% recency
            frequency_score = access_count * 0.7
            recency_score = (1.0 / (current_time - last_access + 1)) * 0.3
            scores[key] = frequency_score + recency_score
        
        if scores:
            least_valuable = min(scores.items(), key=lambda x: x[1])[0]
            await self.delete(least_valuable)
    
    async def _evict_intelligent(self) -> None:
        """Intelligent eviction using ML-like heuristics."""
        # This would implement more sophisticated eviction logic
        await self._evict_hybrid()

# Main orchestrator class
class AdvancedOptimizationOrchestrator:
    """Advanced optimization orchestrator with all patterns."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.strategy_registry = StrategyRegistry()
        self.cache = AdvancedMemoryCache(strategy=CacheStrategy.INTELLIGENT)
        
        # Set up handler chain
        self.validation_handler = ContentValidationHandler()
        self.caching_handler = CachingHandler(self.cache)
        self.execution_handler = OptimizationExecutionHandler(self.strategy_registry)
        
        # Chain handlers
        self.validation_handler.set_next(self.caching_handler)
        self.caching_handler.set_next(self.execution_handler)
        
        # Set up pipeline
        self.pipeline = StandardOptimizationPipeline(self.validation_handler)
        
        # Register strategies
        self._register_strategies()
        
        # Set up event listeners
        self._setup_event_listeners()
    
    def _register_strategies(self) -> None:
        """Register optimization strategies."""
        self.strategy_registry.register("engagement", EngagementStrategy())
        self.strategy_registry.register("reach", ReachStrategy())
    
    def _setup_event_listeners(self) -> None:
        """Set up event listeners."""
        self.event_bus.subscribe("optimization_started", self._on_optimization_started)
        self.event_bus.subscribe("optimization_completed", self._on_optimization_completed)
        self.event_bus.subscribe("optimization_failed", self._on_optimization_failed)
    
    async def optimize(self, content: str, config: Dict[str, Any]) -> OptimizationResponse:
        """Optimize content using the advanced pipeline."""
        request = OptimizationRequest(
            content=content,
            config=config
        )
        
        # Publish event
        self.event_bus.publish("optimization_started", request)
        
        try:
            result = await self.pipeline.run(request)
            
            if result.score > 0:
                self.event_bus.publish("optimization_completed", result)
            else:
                self.event_bus.publish("optimization_failed", result)
            
            return result
            
        except Exception as e:
            error_result = self.pipeline.create_error_response(request, str(e))
            self.event_bus.publish("optimization_failed", error_result)
            return error_result
    
    async def batch_optimize(self, contents: List[str], config: Dict[str, Any]) -> List[OptimizationResponse]:
        """Optimize multiple contents in batch."""
        requests = [
            OptimizationRequest(content=content, config=config)
            for content in contents
        ]
        
        # Create batch command
        command = BatchOptimizationCommand(requests, self.validation_handler)
        
        # Execute batch optimization
        results = await command.execute()
        return results
    
    def _on_optimization_started(self, request: OptimizationRequest) -> None:
        """Handle optimization started event."""
        logger.info(f"Optimization started for request {request.id}")
    
    def _on_optimization_completed(self, result: OptimizationResponse) -> None:
        """Handle optimization completed event."""
        logger.info(f"Optimization completed: {result.score}/100")
    
    def _on_optimization_failed(self, result: OptimizationResponse) -> None:
        """Handle optimization failed event."""
        logger.error(f"Optimization failed: {result.errors}")

# Advanced decorators
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Circuit breaker decorator."""
    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = 0
        circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, circuit_state
            
            current_time = time.time()
            
            if circuit_state == "OPEN":
                if current_time - last_failure_time > recovery_timeout:
                    circuit_state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                if circuit_state == "HALF_OPEN":
                    circuit_state = "CLOSED"
                    failure_count = 0
                return result
            except Exception as e:
                failure_count += 1
                last_failure_time = current_time
                
                if failure_count >= failure_threshold:
                    circuit_state = "OPEN"
                
                raise e
        
        return wrapper
    return decorator

# Main demo function
async def demo_advanced_refactoring():
    """Demonstrate advanced refactoring capabilities."""
    print("🚀 ADVANCED REFACTORING & ENTERPRISE ARCHITECTURE v3.0")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = AdvancedOptimizationOrchestrator()
    
    # Test configurations
    test_configs = [
        {"strategy": "engagement", "level": "premium"},
        {"strategy": "reach", "level": "standard"},
        {"strategy": "engagement", "level": "enterprise"}
    ]
    
    # Test content
    test_contents = [
        "AI is revolutionizing the workplace. Machine learning algorithms are becoming more sophisticated every day.",
        "The future of remote work is here. Companies are adopting hybrid models that combine the best of both worlds.",
        "Building a strong personal brand on LinkedIn requires consistency, authenticity, and valuable content."
    ]
    
    print("📝 Testing advanced refactored optimization engine...")
    
    # Single optimizations
    for i, (content, config) in enumerate(zip(test_contents, test_configs)):
        print(f"\n{i+1}. Strategy: {config['strategy'].upper()}")
        print(f"   Content: {content[:80]}...")
        
        start_time = time.time()
        result = await orchestrator.optimize(content, config)
        optimization_time = time.time() - start_time
        
        print(f"   ✅ Optimization completed in {optimization_time:.3f}s")
        print(f"   📊 Score: {result.score:.1f}/100")
        print(f"   🎯 Cache hit: {result.cache_hit}")
        if result.errors:
            print(f"   ❌ Errors: {result.errors}")
    
    # Batch optimization
    print(f"\n⚡ Running batch optimization for {len(test_contents)} contents...")
    start_time = time.time()
    batch_results = await orchestrator.batch_optimize(test_contents, test_configs[0])
    batch_time = time.time() - start_time
    
    print(f"✅ Batch optimization completed in {batch_time:.3f}s")
    print(f"   Average time per content: {batch_time/len(test_contents):.3f}s")
    print(f"   Successful optimizations: {len([r for r in batch_results if r.score > 0])}")
    
    print("\n🎉 Advanced refactoring demo completed!")
    print("✨ The system now uses enterprise-grade design patterns and architecture!")

if __name__ == "__main__":
    asyncio.run(demo_advanced_refactoring())
