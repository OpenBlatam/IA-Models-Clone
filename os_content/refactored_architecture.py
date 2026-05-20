from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Dict, List, Optional, Any, Protocol, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import weakref
from contextlib import asynccontextmanager
import time
import uuid
from typing import Any, List, Dict, Optional
"""
Refactored OS Content Architecture
Clean Architecture with Dependency Injection and Advanced Optimization
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic components
T = TypeVar('T')
R = TypeVar('R')

# ============================================================================
# CORE DOMAIN LAYER
# ============================================================================

class ProcessingMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAM = "stream"

class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ProcessingRequest:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Any = None
    priority: Priority = Priority.NORMAL
    mode: ProcessingMode = ProcessingMode.ASYNC
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class ProcessingResult:
    request_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# APPLICATION LAYER - USE CASES
# ============================================================================

class VideoProcessingUseCase(ABC):
    """Video processing use case interface"""
    
    @abstractmethod
    async def generate_video(self, prompt: str, duration: int, **kwargs) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def get_video_status(self, video_id: str) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def cancel_video(self, video_id: str) -> ProcessingResult:
        pass

class NLPProcessingUseCase(ABC):
    """NLP processing use case interface"""
    
    @abstractmethod
    async def analyze_text(self, text: str, analysis_type: str) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def batch_analyze(self, texts: List[str]) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def answer_question(self, question: str, context: str) -> ProcessingResult:
        pass

class CacheManagementUseCase(ABC):
    """Cache management use case interface"""
    
    @abstractmethod
    async def get(self, key: str) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def clear(self) -> ProcessingResult:
        pass

class PerformanceMonitoringUseCase(ABC):
    """Performance monitoring use case interface"""
    
    @abstractmethod
    async def get_metrics(self, metric_names: List[str]) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def get_alerts(self, level: Optional[str] = None) -> ProcessingResult:
        pass
    
    @abstractmethod
    async def generate_report(self) -> ProcessingResult:
        pass

# ============================================================================
# INFRASTRUCTURE LAYER - REPOSITORIES & SERVICES
# ============================================================================

class VideoRepository(ABC):
    """Video repository interface"""
    
    @abstractmethod
    async async def save_video_request(self, request: ProcessingRequest) -> str:
        pass
    
    @abstractmethod
    async async def get_video_request(self, video_id: str) -> Optional[ProcessingRequest]:
        pass
    
    @abstractmethod
    async def update_video_status(self, video_id: str, status: str) -> bool:
        pass
    
    @abstractmethod
    async def get_video_result(self, video_id: str) -> Optional[ProcessingResult]:
        pass

class NLPRepository(ABC):
    """NLP repository interface"""
    
    @abstractmethod
    async async def save_analysis_request(self, request: ProcessingRequest) -> str:
        pass
    
    @abstractmethod
    async def get_analysis_result(self, analysis_id: str) -> Optional[ProcessingResult]:
        pass
    
    @abstractmethod
    async def save_batch_analysis(self, requests: List[ProcessingRequest]) -> List[str]:
        pass

class CacheRepository(ABC):
    """Cache repository interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass

class MetricsRepository(ABC):
    """Metrics repository interface"""
    
    @abstractmethod
    async def save_metric(self, name: str, value: float, labels: Dict[str, str]) -> bool:
        pass
    
    @abstractmethod
    async def get_metric(self, name: str, start_time: Optional[float] = None, 
                        end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_metric_statistics(self, name: str) -> Dict[str, float]:
        pass
    
    @abstractmethod
    async def save_alert(self, alert: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def get_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

# ============================================================================
# INFRASTRUCTURE LAYER - EXTERNAL SERVICES
# ============================================================================

class VideoProcessingService(ABC):
    """Video processing service interface"""
    
    @abstractmethod
    async def create_video(self, prompt: str, duration: int, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def get_processing_stats(self) -> Dict[str, Any]:
        pass

class NLPProcessingService(ABC):
    """NLP processing service interface"""
    
    @abstractmethod
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        pass

class CacheService(ABC):
    """Cache service interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass

class PerformanceMonitoringService(ABC):
    """Performance monitoring service interface"""
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, float]:
        pass
    
    @abstractmethod
    async def check_alerts(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def generate_report(self) -> str:
        pass

# ============================================================================
# INFRASTRUCTURE LAYER - TASK PROCESSING
# ============================================================================

class TaskProcessor(ABC):
    """Task processor interface"""
    
    @abstractmethod
    async def submit_task(self, func: callable, *args, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> str:
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        pass

# ============================================================================
# APPLICATION LAYER - IMPLEMENTATIONS
# ============================================================================

class VideoProcessingUseCaseImpl(VideoProcessingUseCase):
    """Video processing use case implementation"""
    
    def __init__(self, 
                 video_repo: VideoRepository,
                 video_service: VideoProcessingService,
                 task_processor: TaskProcessor,
                 cache_service: CacheService):
        
    """__init__ function."""
self.video_repo = video_repo
        self.video_service = video_service
        self.task_processor = task_processor
        self.cache_service = cache_service
    
    async def generate_video(self, prompt: str, duration: int, **kwargs) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"video:{hash(prompt + str(duration))}"
            cached_result = await self.cache_service.get(cache_key)
            
            if cached_result:
                return ProcessingResult(
                    request_id=cached_result['request_id'],
                    success=True,
                    data=cached_result['data'],
                    processing_time=time.time() - start_time,
                    metadata={'source': 'cache'}
                )
            
            # Create processing request
            request = ProcessingRequest(
                data={'prompt': prompt, 'duration': duration, **kwargs},
                priority=Priority.HIGH,
                mode=ProcessingMode.ASYNC
            )
            
            # Save to repository
            video_id = await self.video_repo.save_video_request(request)
            
            # Submit to task processor
            task_id = await self.task_processor.submit_task(
                self.video_service.create_video,
                prompt, duration, **kwargs
            )
            
            # Update status
            await self.video_repo.update_video_status(video_id, 'processing')
            
            return ProcessingResult(
                request_id=video_id,
                success=True,
                data={'task_id': task_id, 'status': 'processing'},
                processing_time=time.time() - start_time,
                metadata={'task_id': task_id}
            )
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            return ProcessingResult(
                request_id=request.id if 'request' in locals() else str(uuid.uuid4()),
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def get_video_status(self, video_id: str) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Get from repository
            request = await self.video_repo.get_video_request(video_id)
            if not request:
                return ProcessingResult(
                    request_id=video_id,
                    success=False,
                    error="Video request not found",
                    processing_time=time.time() - start_time
                )
            
            # Get result
            result = await self.video_repo.get_video_result(video_id)
            
            return ProcessingResult(
                request_id=video_id,
                success=True,
                data=result.data if result else None,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error getting video status: {e}")
            return ProcessingResult(
                request_id=video_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def cancel_video(self, video_id: str) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Cancel task
            success = await self.task_processor.cancel_task(video_id)
            
            if success:
                await self.video_repo.update_video_status(video_id, 'cancelled')
            
            return ProcessingResult(
                request_id=video_id,
                success=success,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error cancelling video: {e}")
            return ProcessingResult(
                request_id=video_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

class NLPProcessingUseCaseImpl(NLPProcessingUseCase):
    """NLP processing use case implementation"""
    
    def __init__(self, 
                 nlp_repo: NLPRepository,
                 nlp_service: NLPProcessingService,
                 cache_service: CacheService):
        
    """__init__ function."""
self.nlp_repo = nlp_repo
        self.nlp_service = nlp_service
        self.cache_service = cache_service
    
    async def analyze_text(self, text: str, analysis_type: str) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"nlp:{hash(text + analysis_type)}"
            cached_result = await self.cache_service.get(cache_key)
            
            if cached_result:
                return ProcessingResult(
                    request_id=cached_result['request_id'],
                    success=True,
                    data=cached_result['data'],
                    processing_time=time.time() - start_time,
                    metadata={'source': 'cache'}
                )
            
            # Create request
            request = ProcessingRequest(
                data={'text': text, 'analysis_type': analysis_type},
                priority=Priority.NORMAL,
                mode=ProcessingMode.SYNC
            )
            
            # Save to repository
            analysis_id = await self.nlp_repo.save_analysis_request(request)
            
            # Process
            result = await self.nlp_service.analyze_text(text)
            
            # Save result
            processing_result = ProcessingResult(
                request_id=analysis_id,
                success=True,
                data=result,
                processing_time=time.time() - start_time
            )
            
            # Cache result
            await self.cache_service.set(cache_key, {
                'request_id': analysis_id,
                'data': result
            }, ttl=1800)
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return ProcessingResult(
                request_id=request.id if 'request' in locals() else str(uuid.uuid4()),
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def batch_analyze(self, texts: List[str]) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Create batch requests
            requests = [
                ProcessingRequest(
                    data={'text': text},
                    priority=Priority.NORMAL,
                    mode=ProcessingMode.BATCH
                )
                for text in texts
            ]
            
            # Save batch
            analysis_ids = await self.nlp_repo.save_batch_analysis(requests)
            
            # Process batch
            results = await self.nlp_service.batch_analyze(texts)
            
            return ProcessingResult(
                request_id=','.join(analysis_ids),
                success=True,
                data=results,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return ProcessingResult(
                request_id=str(uuid.uuid4()),
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def answer_question(self, question: str, context: str) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"qa:{hash(question + context)}"
            cached_result = await self.cache_service.get(cache_key)
            
            if cached_result:
                return ProcessingResult(
                    request_id=cached_result['request_id'],
                    success=True,
                    data=cached_result['data'],
                    processing_time=time.time() - start_time,
                    metadata={'source': 'cache'}
                )
            
            # Create request
            request = ProcessingRequest(
                data={'question': question, 'context': context},
                priority=Priority.NORMAL,
                mode=ProcessingMode.SYNC
            )
            
            # Save to repository
            qa_id = await self.nlp_repo.save_analysis_request(request)
            
            # Process
            result = await self.nlp_service.answer_question(question, context)
            
            # Save result
            processing_result = ProcessingResult(
                request_id=qa_id,
                success=True,
                data=result,
                processing_time=time.time() - start_time
            )
            
            # Cache result
            await self.cache_service.set(cache_key, {
                'request_id': qa_id,
                'data': result
            }, ttl=3600)
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return ProcessingResult(
                request_id=request.id if 'request' in locals() else str(uuid.uuid4()),
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

class CacheManagementUseCaseImpl(CacheManagementUseCase):
    """Cache management use case implementation"""
    
    def __init__(self, cache_service: CacheService):
        
    """__init__ function."""
self.cache_service = cache_service
    
    async def get(self, key: str) -> ProcessingResult:
        start_time = time.time()
        
        try:
            value = await self.cache_service.get(key)
            
            return ProcessingResult(
                request_id=key,
                success=value is not None,
                data=value,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return ProcessingResult(
                request_id=key,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> ProcessingResult:
        start_time = time.time()
        
        try:
            success = await self.cache_service.set(key, value, ttl)
            
            return ProcessingResult(
                request_id=key,
                success=success,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return ProcessingResult(
                request_id=key,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def delete(self, key: str) -> ProcessingResult:
        start_time = time.time()
        
        try:
            success = await self.cache_service.delete(key)
            
            return ProcessingResult(
                request_id=key,
                success=success,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return ProcessingResult(
                request_id=key,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def clear(self) -> ProcessingResult:
        start_time = time.time()
        
        try:
            success = await self.cache_service.clear()
            
            return ProcessingResult(
                request_id="clear_all",
                success=success,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return ProcessingResult(
                request_id="clear_all",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

class PerformanceMonitoringUseCaseImpl(PerformanceMonitoringUseCase):
    """Performance monitoring use case implementation"""
    
    def __init__(self, metrics_repo: MetricsRepository, perf_service: PerformanceMonitoringService):
        
    """__init__ function."""
self.metrics_repo = metrics_repo
        self.perf_service = perf_service
    
    async def get_metrics(self, metric_names: List[str]) -> ProcessingResult:
        start_time = time.time()
        
        try:
            # Collect current metrics
            current_metrics = await self.perf_service.collect_metrics()
            
            # Get historical data
            historical_data = {}
            for metric_name in metric_names:
                if metric_name in current_metrics:
                    historical_data[metric_name] = await self.metrics_repo.get_metric(metric_name)
            
            return ProcessingResult(
                request_id="metrics_collection",
                success=True,
                data={
                    'current': current_metrics,
                    'historical': historical_data
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return ProcessingResult(
                request_id="metrics_collection",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def get_alerts(self, level: Optional[str] = None) -> ProcessingResult:
        start_time = time.time()
        
        try:
            alerts = await self.perf_service.check_alerts()
            
            if level:
                alerts = [alert for alert in alerts if alert.get('level') == level]
            
            return ProcessingResult(
                request_id="alerts_check",
                success=True,
                data=alerts,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return ProcessingResult(
                request_id="alerts_check",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def generate_report(self) -> ProcessingResult:
        start_time = time.time()
        
        try:
            report = await self.perf_service.generate_report()
            
            return ProcessingResult(
                request_id="report_generation",
                success=True,
                data=report,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ProcessingResult(
                request_id="report_generation",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

class DependencyContainer:
    """Dependency injection container"""
    
    def __init__(self) -> Any:
        self._services = {}
        self._singletons = {}
    
    def register(self, interface: type, implementation: type):
        """Register a service implementation"""
        self._services[interface] = implementation
    
    def register_singleton(self, interface: type, instance: Any):
        """Register a singleton instance"""
        self._singletons[interface] = instance
    
    def resolve(self, interface: type) -> Any:
        """Resolve a service instance"""
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check registered implementations
        if interface in self._services:
            implementation = self._services[interface]
            return implementation()
        
        raise ValueError(f"No implementation registered for {interface}")
    
    async def initialize(self) -> Any:
        """Initialize all services"""
        # Initialize singletons
        for interface, instance in self._singletons.items():
            if hasattr(instance, 'initialize'):
                await instance.initialize()
    
    async def shutdown(self) -> Any:
        """Shutdown all services"""
        # Shutdown singletons
        for interface, instance in self._singletons.items():
            if hasattr(instance, 'shutdown'):
                await instance.shutdown()

# ============================================================================
# APPLICATION FACTORY
# ============================================================================

class ApplicationFactory:
    """Application factory for creating use cases with dependencies"""
    
    def __init__(self, container: DependencyContainer):
        
    """__init__ function."""
self.container = container
    
    def create_video_use_case(self) -> VideoProcessingUseCase:
        """Create video processing use case with dependencies"""
        return VideoProcessingUseCaseImpl(
            video_repo=self.container.resolve(VideoRepository),
            video_service=self.container.resolve(VideoProcessingService),
            task_processor=self.container.resolve(TaskProcessor),
            cache_service=self.container.resolve(CacheService)
        )
    
    def create_nlp_use_case(self) -> NLPProcessingUseCase:
        """Create NLP processing use case with dependencies"""
        return NLPProcessingUseCaseImpl(
            nlp_repo=self.container.resolve(NLPRepository),
            nlp_service=self.container.resolve(NLPProcessingService),
            cache_service=self.container.resolve(CacheService)
        )
    
    def create_cache_use_case(self) -> CacheManagementUseCase:
        """Create cache management use case with dependencies"""
        return CacheManagementUseCaseImpl(
            cache_service=self.container.resolve(CacheService)
        )
    
    def create_performance_use_case(self) -> PerformanceMonitoringUseCase:
        """Create performance monitoring use case with dependencies"""
        return PerformanceMonitoringUseCaseImpl(
            metrics_repo=self.container.resolve(MetricsRepository),
            perf_service=self.container.resolve(PerformanceMonitoringService)
        )

# ============================================================================
# APPLICATION CONTEXT
# ============================================================================

class ApplicationContext:
    """Application context for managing the application lifecycle"""
    
    def __init__(self) -> Any:
        self.container = DependencyContainer()
        self.factory = ApplicationFactory(self.container)
        self._use_cases = {}
    
    def register_services(self) -> Any:
        """Register all services in the container"""
        # This would be implemented with actual service implementations
        # For now, we'll use placeholder registrations
        pass
    
    def get_use_case(self, use_case_type: type) -> Optional[Dict[str, Any]]:
        """Get a use case instance"""
        if use_case_type not in self._use_cases:
            if use_case_type == VideoProcessingUseCase:
                self._use_cases[use_case_type] = self.factory.create_video_use_case()
            elif use_case_type == NLPProcessingUseCase:
                self._use_cases[use_case_type] = self.factory.create_nlp_use_case()
            elif use_case_type == CacheManagementUseCase:
                self._use_cases[use_case_type] = self.factory.create_cache_use_case()
            elif use_case_type == PerformanceMonitoringUseCase:
                self._use_cases[use_case_type] = self.factory.create_performance_use_case()
            else:
                raise ValueError(f"Unknown use case type: {use_case_type}")
        
        return self._use_cases[use_case_type]
    
    async def initialize(self) -> Any:
        """Initialize the application context"""
        await self.container.initialize()
    
    async def shutdown(self) -> Any:
        """Shutdown the application context"""
        await self.container.shutdown()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class RefactoredOSContentApplication:
    """Main refactored OS Content application"""
    
    def __init__(self) -> Any:
        self.context = ApplicationContext()
        self._initialized = False
    
    async def initialize(self) -> Any:
        """Initialize the application"""
        if self._initialized:
            return
        
        logger.info("Initializing refactored OS Content application...")
        
        # Register services
        self.context.register_services()
        
        # Initialize context
        await self.context.initialize()
        
        self._initialized = True
        logger.info("Application initialized successfully")
    
    async def shutdown(self) -> Any:
        """Shutdown the application"""
        if not self._initialized:
            return
        
        logger.info("Shutting down application...")
        await self.context.shutdown()
        self._initialized = False
        logger.info("Application shut down successfully")
    
    def get_video_use_case(self) -> VideoProcessingUseCase:
        """Get video processing use case"""
        return self.context.get_use_case(VideoProcessingUseCase)
    
    def get_nlp_use_case(self) -> NLPProcessingUseCase:
        """Get NLP processing use case"""
        return self.context.get_use_case(NLPProcessingUseCase)
    
    def get_cache_use_case(self) -> CacheManagementUseCase:
        """Get cache management use case"""
        return self.context.get_use_case(CacheManagementUseCase)
    
    def get_performance_use_case(self) -> PerformanceMonitoringUseCase:
        """Get performance monitoring use case"""
        return self.context.get_use_case(PerformanceMonitoringUseCase)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of the refactored architecture"""
    
    # Create application
    app = RefactoredOSContentApplication()
    
    try:
        # Initialize application
        await app.initialize()
        
        # Get use cases
        video_use_case = app.get_video_use_case()
        nlp_use_case = app.get_nlp_use_case()
        cache_use_case = app.get_cache_use_case()
        perf_use_case = app.get_performance_use_case()
        
        # Example: Generate video
        video_result = await video_use_case.generate_video(
            prompt="Beautiful sunset over mountains",
            duration=10
        )
        print(f"Video generation result: {video_result.success}")
        
        # Example: Analyze text
        nlp_result = await nlp_use_case.analyze_text(
            text="I love this amazing product!",
            analysis_type="sentiment"
        )
        print(f"NLP analysis result: {nlp_result.success}")
        
        # Example: Cache operations
        cache_result = await cache_use_case.set("test_key", "test_value")
        print(f"Cache set result: {cache_result.success}")
        
        # Example: Performance monitoring
        perf_result = await perf_use_case.get_metrics(["system.cpu.usage"])
        print(f"Performance metrics result: {perf_result.success}")
        
    finally:
        # Shutdown application
        await app.shutdown()

match __name__:
    case "__main__":
    asyncio.run(main()) 