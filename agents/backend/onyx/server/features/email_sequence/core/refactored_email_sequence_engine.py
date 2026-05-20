"""
Refactored Email Sequence Engine

A modern, highly optimized email sequence engine with:
- Clean Architecture and Dependency Injection
- Advanced Error Handling and Resilience
- Performance Optimization with Cutting-edge Libraries
- Modern Python Patterns and Best Practices
- Comprehensive Monitoring and Observability
"""

import asyncio
import logging
import time
import gc
import psutil
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, 
    AsyncGenerator, Protocol, runtime_checkable
)
from uuid import UUID
import weakref

# High-performance libraries
import orjson
import msgspec
import uvloop
import structlog
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)
import pybreaker
from cachetools import TTLCache, LRUCache
import asyncio_mqtt

# Models and services
from ..models.sequence import EmailSequence, SequenceStep, SequenceStatus, StepType
from ..models.template import EmailTemplate, TemplateStatus
from ..models.subscriber import Subscriber, SubscriberStatus
from ..models.campaign import EmailCampaign, CampaignMetrics
from ..services.langchain_service import LangChainEmailService
from ..services.delivery_service import EmailDeliveryService
from ..services.analytics_service import EmailAnalyticsService

# Configure structured logging
logger = structlog.get_logger(__name__)

# Constants
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
BATCH_SIZE = 100
MAX_MEMORY_USAGE = 0.8  # 80% of available memory
CACHE_TTL = 3600  # 1 hour
CACHE_SIZE = 1000


class EngineStatus(Enum):
    """Engine status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ProcessingResult:
    """Enhanced result of sequence processing with detailed metadata"""
    
    def __init__(
        self,
        success: bool,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.message = message
        self.data = data or {}
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time
        }


@dataclass
class EngineConfig:
    """Configuration for the email sequence engine"""
    max_concurrent_sequences: int = 50
    max_queue_size: int = 1000
    batch_size: int = BATCH_SIZE
    timeout_seconds: int = TIMEOUT_SECONDS
    max_retries: int = MAX_RETRIES
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_circuit_breaker: bool = True
    memory_threshold: float = MAX_MEMORY_USAGE
    cache_ttl: int = CACHE_TTL
    cache_size: int = CACHE_SIZE


@dataclass
class EngineMetrics:
    """Comprehensive engine metrics"""
    sequences_processed: int = 0
    emails_sent: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    start_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    processing_times: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)


@runtime_checkable
class EmailSequenceProcessor(Protocol):
    """Protocol for email sequence processors"""
    
    async def process_sequence(self, sequence: EmailSequence) -> ProcessingResult:
        """Process a single email sequence"""
        ...
    
    async def process_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process a single sequence step"""
        ...


@runtime_checkable
class EmailDeliveryProcessor(Protocol):
    """Protocol for email delivery processors"""
    
    async def send_email(self, email_data: Dict[str, Any]) -> ProcessingResult:
        """Send a single email"""
        ...
    
    async def send_bulk_emails(self, emails: List[Dict[str, Any]]) -> ProcessingResult:
        """Send multiple emails in bulk"""
        ...


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        
        return (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.recovery_timeout


class MemoryManager:
    """Advanced memory management with automatic cleanup"""
    
    def __init__(self, threshold: float = MAX_MEMORY_USAGE):
        self.threshold = threshold
        self.last_cleanup = datetime.utcnow()
        self.cleanup_interval = timedelta(minutes=5)
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is above threshold"""
        memory = psutil.virtual_memory()
        return memory.percent / 100 > self.threshold
    
    async def perform_cleanup(self) -> None:
        """Perform memory cleanup"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            self.last_cleanup = datetime.utcnow()
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        return (
            self.check_memory_pressure() or
            datetime.utcnow() - self.last_cleanup > self.cleanup_interval
        )


class CacheManager:
    """Advanced caching with multiple strategies"""
    
    def __init__(self, ttl: int = CACHE_TTL, size: int = CACHE_SIZE):
        self.ttl_cache = TTLCache(maxsize=size, ttl=ttl)
        self.lru_cache = LRUCache(maxsize=size)
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try TTL cache first
        if key in self.ttl_cache:
            self.hits += 1
            return self.ttl_cache[key]
        
        # Try LRU cache
        if key in self.lru_cache:
            self.hits += 1
            return self.lru_cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, use_ttl: bool = True) -> None:
        """Set value in cache"""
        if use_ttl:
            self.ttl_cache[key] = value
        else:
            self.lru_cache[key] = value
    
    def clear(self) -> None:
        """Clear all caches"""
        self.ttl_cache.clear()
        self.lru_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_cache_size": len(self.ttl_cache),
            "lru_cache_size": len(self.lru_cache)
        }


class RetryManager:
    """Advanced retry logic with exponential backoff"""
    
    def __init__(self, max_attempts: int = MAX_RETRIES, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
    
    def __call__(self, func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper


class EmailSequenceEngine:
    """
    Refactored email sequence engine with modern architecture and advanced features.
    """
    
    def __init__(
        self,
        config: EngineConfig,
        langchain_service: LangChainEmailService,
        delivery_service: EmailDeliveryService,
        analytics_service: EmailAnalyticsService,
        processor: Optional[EmailSequenceProcessor] = None,
        delivery_processor: Optional[EmailDeliveryProcessor] = None
    ):
        """
        Initialize the refactored email sequence engine.
        
        Args:
            config: Engine configuration
            langchain_service: LangChain service for AI-powered features
            delivery_service: Email delivery service
            analytics_service: Analytics service for tracking
            processor: Optional custom sequence processor
            delivery_processor: Optional custom delivery processor
        """
        self.config = config
        self.langchain_service = langchain_service
        self.delivery_service = delivery_service
        self.analytics_service = analytics_service
        
        # Use dependency injection for processors
        self.sequence_processor = processor or self._create_default_processor()
        self.delivery_processor = delivery_processor or self._create_default_delivery_processor()
        
        # Advanced components
        self.memory_manager = MemoryManager(config.memory_threshold)
        self.cache_manager = CacheManager(config.cache_ttl, config.cache_size)
        self.retry_manager = RetryManager(config.max_retries)
        self.circuit_breaker = CircuitBreaker()
        
        # State management
        self.status = EngineStatus.IDLE
        self.active_sequences: Dict[UUID, EmailSequence] = {}
        self.active_campaigns: Dict[UUID, EmailCampaign] = {}
        
        # Processing queues with backpressure
        self.sequence_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.email_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Background tasks with proper lifecycle management
        self.background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Comprehensive metrics
        self.metrics = EngineMetrics()
        self.metrics.start_time = datetime.utcnow()
        
        # Circuit breakers for different operations
        self.sequence_circuit_breaker = CircuitBreaker(failure_threshold=3)
        self.email_circuit_breaker = CircuitBreaker(failure_threshold=5)
        self.analytics_circuit_breaker = CircuitBreaker(failure_threshold=2)
        
        logger.info("Refactored Email Sequence Engine initialized", 
                   config=config.__dict__, 
                   status=self.status.value)
    
    def _create_default_processor(self) -> EmailSequenceProcessor:
        """Create default sequence processor"""
        return DefaultSequenceProcessor(self)
    
    def _create_default_delivery_processor(self) -> EmailDeliveryProcessor:
        """Create default delivery processor"""
        return DefaultDeliveryProcessor(self)
    
    @asynccontextmanager
    async def lifecycle(self):
        """Context manager for engine lifecycle"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    async def start(self) -> ProcessingResult:
        """Start the email sequence engine with enhanced error handling"""
        start_time = time.time()
        
        try:
            if self.status != EngineStatus.IDLE:
                return ProcessingResult(
                    success=False,
                    message=f"Engine is not in IDLE state. Current state: {self.status.value}",
                    error=RuntimeError("Invalid engine state")
                )
            
            self.status = EngineStatus.RUNNING
            self.metrics.start_time = datetime.utcnow()
            
            # Start background tasks
            await self._start_background_tasks()
            
            execution_time = time.time() - start_time
            result = ProcessingResult(
                success=True,
                message="Engine started successfully",
                data={"status": self.status.value},
                metadata={"execution_time": execution_time}
            )
            result.execution_time = execution_time
            
            logger.info("Engine started successfully", 
                       execution_time=execution_time,
                       status=self.status.value)
            
            return result
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            execution_time = time.time() - start_time
            
            logger.error("Failed to start engine", 
                        error=str(e), 
                        execution_time=execution_time)
            
            return ProcessingResult(
                success=False,
                message=f"Failed to start engine: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    async def stop(self) -> ProcessingResult:
        """Stop the email sequence engine gracefully"""
        start_time = time.time()
        
        try:
            if self.status == EngineStatus.STOPPING:
                return ProcessingResult(
                    success=False,
                    message="Engine is already stopping",
                    error=RuntimeError("Engine already stopping")
                )
            
            self.status = EngineStatus.STOPPING
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=self.config.timeout_seconds
                )
            
            # Cleanup resources
            await self._cleanup_resources()
            
            self.status = EngineStatus.IDLE
            execution_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                message="Engine stopped successfully",
                data={"status": self.status.value},
                metadata={"execution_time": execution_time}
            )
            result.execution_time = execution_time
            
            logger.info("Engine stopped successfully", 
                       execution_time=execution_time,
                       status=self.status.value)
            
            return result
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            execution_time = time.time() - start_time
            
            logger.error("Failed to stop engine", 
                        error=str(e), 
                        execution_time=execution_time)
            
            return ProcessingResult(
                success=False,
                message=f"Failed to stop engine: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        tasks = [
            asyncio.create_task(self._process_sequence_queue(), name="sequence_processor"),
            asyncio.create_task(self._process_email_queue(), name="email_processor"),
            asyncio.create_task(self._process_analytics(), name="analytics_processor"),
            asyncio.create_task(self._monitor_system(), name="system_monitor"),
            asyncio.create_task(self._cleanup_loop(), name="cleanup_loop")
        ]
        
        self.background_tasks.extend(tasks)
        logger.info("Background tasks started", task_count=len(tasks))
    
    async def _cleanup_resources(self) -> None:
        """Cleanup engine resources"""
        try:
            # Clear caches
            self.cache_manager.clear()
            
            # Clear queues
            while not self.sequence_queue.empty():
                self.sequence_queue.get_nowait()
            
            while not self.email_queue.empty():
                self.email_queue.get_nowait()
            
            # Clear active sequences
            self.active_sequences.clear()
            self.active_campaigns.clear()
            
            # Perform memory cleanup
            await self.memory_manager.perform_cleanup()
            
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.error("Resource cleanup failed", error=str(e))
    
    @circuit_breaker
    async def create_sequence(
        self,
        name: str,
        target_audience: str,
        goals: List[str],
        tone: str = "professional",
        templates: List[EmailTemplate] = None,
        user_id: Optional[str] = None
    ) -> ProcessingResult:
        """Create a new email sequence with enhanced validation"""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not name or not target_audience or not goals:
                return ProcessingResult(
                    success=False,
                    message="Missing required parameters: name, target_audience, goals",
                    error=ValueError("Missing required parameters")
                )
            
            # Create sequence using LangChain
            sequence_data = await self.langchain_service.create_sequence(
                name=name,
                target_audience=target_audience,
                goals=goals,
                tone=tone,
                templates=templates or []
            )
            
            # Create EmailSequence object
            sequence = EmailSequence(
                name=name,
                target_audience=target_audience,
                goals=goals,
                tone=tone,
                steps=sequence_data.get("steps", []),
                personalization_variables=sequence_data.get("personalization_variables", {})
            )
            
            # Apply templates if provided
            if templates:
                await self._apply_templates_to_sequence(sequence, templates)
            
            # Store in cache
            cache_key = f"sequence:{sequence.id}"
            self.cache_manager.set(cache_key, sequence)
            
            execution_time = time.time() - start_time
            self.metrics.sequences_processed += 1
            
            result = ProcessingResult(
                success=True,
                message="Sequence created successfully",
                data={"sequence_id": str(sequence.id), "sequence": sequence.to_dict()},
                metadata={
                    "execution_time": execution_time,
                    "user_id": user_id,
                    "templates_count": len(templates) if templates else 0
                }
            )
            result.execution_time = execution_time
            
            logger.info("Sequence created successfully", 
                       sequence_id=str(sequence.id),
                       execution_time=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.errors += 1
            self.metrics.error_types[type(e).__name__] = self.metrics.error_types.get(type(e).__name__, 0) + 1
            
            logger.error("Failed to create sequence", 
                        error=str(e), 
                        execution_time=execution_time)
            
            return ProcessingResult(
                success=False,
                message=f"Failed to create sequence: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    @circuit_breaker
    async def activate_sequence(self, sequence_id: UUID, user_id: Optional[str] = None) -> ProcessingResult:
        """Activate a sequence with enhanced error handling"""
        start_time = time.time()
        
        try:
            # Get sequence from cache or active sequences
            sequence = self.active_sequences.get(sequence_id)
            if not sequence:
                cache_key = f"sequence:{sequence_id}"
                sequence = self.cache_manager.get(cache_key)
            
            if not sequence:
                return ProcessingResult(
                    success=False,
                    message=f"Sequence {sequence_id} not found",
                    error=ValueError("Sequence not found")
                )
            
            # Validate sequence can be activated
            if sequence.status != SequenceStatus.DRAFT:
                return ProcessingResult(
                    success=False,
                    message=f"Sequence {sequence_id} cannot be activated. Current status: {sequence.status.value}",
                    error=ValueError("Invalid sequence status")
                )
            
            # Activate sequence
            sequence.status = SequenceStatus.ACTIVE
            sequence.activated_at = datetime.utcnow()
            self.active_sequences[sequence_id] = sequence
            
            # Add to processing queue
            await self.sequence_queue.put(sequence)
            
            execution_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                message="Sequence activated successfully",
                data={"sequence_id": str(sequence_id)},
                metadata={
                    "execution_time": execution_time,
                    "user_id": user_id,
                    "status": sequence.status.value
                }
            )
            result.execution_time = execution_time
            
            logger.info("Sequence activated successfully", 
                       sequence_id=str(sequence_id),
                       execution_time=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.metrics.errors += 1
            
            logger.error("Failed to activate sequence", 
                        sequence_id=str(sequence_id),
                        error=str(e), 
                        execution_time=execution_time)
            
            return ProcessingResult(
                success=False,
                message=f"Failed to activate sequence: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    async def _process_sequence_queue(self) -> None:
        """Process sequences from the queue with enhanced error handling"""
        while not self._shutdown_event.is_set():
            try:
                # Check memory pressure
                if self.memory_manager.should_cleanup():
                    await self.memory_manager.perform_cleanup()
                
                # Get sequence from queue with timeout
                try:
                    sequence = await asyncio.wait_for(
                        self.sequence_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process sequence
                result = await self.sequence_processor.process_sequence(sequence)
                
                if result.success:
                    self.metrics.sequences_processed += 1
                else:
                    self.metrics.errors += 1
                
                # Update metrics
                self.metrics.last_activity = datetime.utcnow()
                if result.execution_time:
                    self.metrics.processing_times.append(result.execution_time)
                
                # Mark task as done
                self.sequence_queue.task_done()
                
            except Exception as e:
                self.metrics.errors += 1
                logger.error("Error processing sequence queue", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_email_queue(self) -> None:
        """Process emails from the queue with batching"""
        batch = []
        batch_size = self.config.batch_size
        
        while not self._shutdown_event.is_set():
            try:
                # Collect batch of emails
                while len(batch) < batch_size:
                    try:
                        email_data = await asyncio.wait_for(
                            self.email_queue.get(),
                            timeout=0.1
                        )
                        batch.append(email_data)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process batch
                result = await self.delivery_processor.send_bulk_emails(batch)
                
                if result.success:
                    self.metrics.emails_sent += len(batch)
                else:
                    self.metrics.errors += len(batch)
                
                # Clear batch
                batch.clear()
                
                # Mark tasks as done
                for _ in range(len(batch)):
                    self.email_queue.task_done()
                
            except Exception as e:
                self.metrics.errors += len(batch)
                logger.error("Error processing email queue", error=str(e))
                batch.clear()
                await asyncio.sleep(1)
    
    async def _process_analytics(self) -> None:
        """Process analytics with circuit breaker protection"""
        while not self._shutdown_event.is_set():
            try:
                # Process analytics every 30 seconds
                await asyncio.sleep(30)
                
                # Update system metrics
                await self._update_system_metrics()
                
            except Exception as e:
                logger.error("Error processing analytics", error=str(e))
                await asyncio.sleep(5)
    
    async def _monitor_system(self) -> None:
        """Monitor system health and performance"""
        while not self._shutdown_event.is_set():
            try:
                # Update memory and CPU usage
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                
                self.metrics.memory_usage = memory.percent / 100
                self.metrics.cpu_usage = cpu / 100
                
                # Log metrics every minute
                await asyncio.sleep(60)
                
                logger.info("System metrics updated",
                           memory_usage=self.metrics.memory_usage,
                           cpu_usage=self.metrics.cpu_usage,
                           cache_stats=self.cache_manager.get_stats())
                
            except Exception as e:
                logger.error("Error monitoring system", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup every 5 minutes
                await asyncio.sleep(300)
                
                # Perform memory cleanup if needed
                if self.memory_manager.should_cleanup():
                    await self.memory_manager.perform_cleanup()
                
                # Trim processing times list
                if len(self.metrics.processing_times) > 1000:
                    self.metrics.processing_times = self.metrics.processing_times[-500:]
                
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _update_system_metrics(self) -> None:
        """Update system metrics"""
        try:
            # Get cache statistics
            cache_stats = self.cache_manager.get_stats()
            
            # Update metrics
            self.metrics.cache_hits = cache_stats["hits"]
            self.metrics.cache_misses = cache_stats["misses"]
            
        except Exception as e:
            logger.error("Error updating system metrics", error=str(e))
    
    async def _apply_templates_to_sequence(
        self,
        sequence: EmailSequence,
        templates: List[EmailTemplate]
    ) -> None:
        """Apply templates to sequence with validation"""
        try:
            for template in templates:
                if template.status == TemplateStatus.ACTIVE:
                    # Apply template to sequence steps
                    for step in sequence.steps:
                        if step.step_type == StepType.EMAIL and not step.template_id:
                            step.template_id = template.id
                            step.subject = template.subject
                            step.content = template.html_content
            
            logger.info("Templates applied to sequence", 
                       sequence_id=str(sequence.id),
                       templates_count=len(templates))
            
        except Exception as e:
            logger.error("Error applying templates to sequence", 
                        sequence_id=str(sequence.id),
                        error=str(e))
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        stats = {
            "engine_status": self.status.value,
            "metrics": {
                "sequences_processed": self.metrics.sequences_processed,
                "emails_sent": self.metrics.emails_sent,
                "errors": self.metrics.errors,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "memory_usage": self.metrics.memory_usage,
                "cpu_usage": self.metrics.cpu_usage,
                "error_types": self.metrics.error_types
            },
            "queue_status": {
                "sequence_queue_size": self.sequence_queue.qsize(),
                "email_queue_size": self.email_queue.qsize()
            },
            "cache_stats": self.cache_manager.get_stats(),
            "active_sequences": len(self.active_sequences),
            "active_campaigns": len(self.active_campaigns),
            "background_tasks": len(self.background_tasks),
            "uptime": (datetime.utcnow() - self.metrics.start_time).total_seconds() if self.metrics.start_time else 0
        }
        
        # Add processing time statistics
        if self.metrics.processing_times:
            stats["metrics"]["avg_processing_time"] = sum(self.metrics.processing_times) / len(self.metrics.processing_times)
            stats["metrics"]["min_processing_time"] = min(self.metrics.processing_times)
            stats["metrics"]["max_processing_time"] = max(self.metrics.processing_times)
        
        return stats


class DefaultSequenceProcessor:
    """Default implementation of EmailSequenceProcessor"""
    
    def __init__(self, engine: EmailSequenceEngine):
        self.engine = engine
    
    async def process_sequence(self, sequence: EmailSequence) -> ProcessingResult:
        """Process a single email sequence"""
        start_time = time.time()
        
        try:
            for step in sequence.steps:
                if step.is_active:
                    step_result = await self.process_step(sequence, step)
                    if not step_result.success:
                        return step_result
            
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                message="Sequence processed successfully",
                data={"sequence_id": str(sequence.id)},
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                message=f"Failed to process sequence: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    async def process_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process a single sequence step"""
        start_time = time.time()
        
        try:
            if step.step_type == StepType.EMAIL:
                return await self._process_email_step(sequence, step)
            elif step.step_type == StepType.DELAY:
                return await self._process_delay_step(sequence, step)
            elif step.step_type == StepType.CONDITION:
                return await self._process_condition_step(sequence, step)
            elif step.step_type == StepType.ACTION:
                return await self._process_action_step(sequence, step)
            elif step.step_type == StepType.WEBHOOK:
                return await self._process_webhook_step(sequence, step)
            else:
                return ProcessingResult(
                    success=False,
                    message=f"Unknown step type: {step.step_type}",
                    error=ValueError(f"Unknown step type: {step.step_type}")
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                message=f"Failed to process step: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    async def _process_email_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process email step with personalization"""
        try:
            # Get active subscribers
            subscribers = [sub for sub in sequence.subscribers if sub.status == SubscriberStatus.ACTIVE]
            
            for subscriber in subscribers:
                # Personalize content
                personalized_content = await self.engine.langchain_service.personalize_content(
                    step.content,
                    subscriber,
                    sequence.personalization_variables
                )
                
                # Add to email queue
                await self.engine.email_queue.put({
                    "sequence_id": sequence.id,
                    "step_id": step.id,
                    "subscriber_id": subscriber.id,
                    "subject": step.subject,
                    "content": personalized_content,
                    "template_id": step.template_id
                })
            
            return ProcessingResult(
                success=True,
                message=f"Email step processed for {len(subscribers)} subscribers"
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to process email step: {str(e)}",
                error=e
            )
    
    async def _process_delay_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process delay step"""
        try:
            delay_seconds = (step.delay_hours or 0) * 3600 + (step.delay_days or 0) * 86400
            await asyncio.sleep(delay_seconds)
            
            return ProcessingResult(
                success=True,
                message=f"Delay step completed ({delay_seconds} seconds)"
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to process delay step: {str(e)}",
                error=e
            )
    
    async def _process_condition_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process condition step"""
        try:
            # Evaluate condition using LangChain
            condition_result = await self.engine.langchain_service.evaluate_condition(
                step.condition_expression,
                sequence,
                step
            )
            
            return ProcessingResult(
                success=True,
                message=f"Condition evaluated: {condition_result}",
                data={"condition_result": condition_result}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to process condition step: {str(e)}",
                error=e
            )
    
    async def _process_action_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process action step"""
        try:
            # Execute action using LangChain
            action_result = await self.engine.langchain_service.execute_action(
                step.action_type,
                step.action_data,
                sequence,
                step
            )
            
            return ProcessingResult(
                success=True,
                message=f"Action executed: {step.action_type}",
                data={"action_result": action_result}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to process action step: {str(e)}",
                error=e
            )
    
    async def _process_webhook_step(self, sequence: EmailSequence, step: SequenceStep) -> ProcessingResult:
        """Process webhook step"""
        try:
            # Execute webhook
            webhook_result = await self.engine.langchain_service.execute_webhook(
                step.webhook_url,
                step.webhook_data,
                sequence,
                step
            )
            
            return ProcessingResult(
                success=True,
                message=f"Webhook executed: {step.webhook_url}",
                data={"webhook_result": webhook_result}
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to process webhook step: {str(e)}",
                error=e
            )


class DefaultDeliveryProcessor:
    """Default implementation of EmailDeliveryProcessor"""
    
    def __init__(self, engine: EmailSequenceEngine):
        self.engine = engine
    
    async def send_email(self, email_data: Dict[str, Any]) -> ProcessingResult:
        """Send a single email"""
        start_time = time.time()
        
        try:
            # Send email using delivery service
            result = await self.engine.delivery_service.send_email(
                to_email=email_data["subscriber_email"],
                subject=email_data["subject"],
                content=email_data["content"],
                template_id=email_data.get("template_id")
            )
            
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=result.success,
                message=result.message,
                data=result.data,
                error=result.error,
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                message=f"Failed to send email: {str(e)}",
                error=e,
                metadata={"execution_time": execution_time}
            )
    
    async def send_bulk_emails(self, emails: List[Dict[str, Any]]) -> ProcessingResult:
        """Send multiple emails in bulk"""
        start_time = time.time()
        
        try:
            # Send bulk emails using delivery service
            result = await self.engine.delivery_service.send_bulk_emails(emails)
            
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=result.success,
                message=result.message,
                data=result.data,
                error=result.error,
                metadata={
                    "execution_time": execution_time,
                    "emails_count": len(emails)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                message=f"Failed to send bulk emails: {str(e)}",
                error=e,
                metadata={
                    "execution_time": execution_time,
                    "emails_count": len(emails)
                }
            ) 