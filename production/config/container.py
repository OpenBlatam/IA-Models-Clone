from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from domain.interfaces import (
from infrastructure.repositories import PostgresCopywritingRepository
from infrastructure.cache import RedisCacheService
from infrastructure.ai import DevinAIService
from infrastructure.events import AsyncEventPublisher
from infrastructure.monitoring import PrometheusMonitoringService
from application.services import CopywritingApplicationService
from application.use_cases import (
from presentation.controllers import CopywritingController
from typing import Any, List, Dict, Optional
import logging
"""
Dependency Injection Container
==============================

Production dependency injection container with all services properly configured.
"""


    CopywritingRepository,
    CacheService,
    AIService,
    EventPublisher,
    MonitoringService
)
    GenerateCopywritingUseCase,
    GetCopywritingHistoryUseCase
)


@dataclass
class Container:
    """Dependency injection container for production."""
    
    # Settings
    settings: Any
    
    # Infrastructure services
    repository: Optional[CopywritingRepository] = None
    cache_service: Optional[CacheService] = None
    ai_service: Optional[AIService] = None
    event_publisher: Optional[EventPublisher] = None
    monitoring_service: Optional[MonitoringService] = None
    
    # Application services
    application_service: Optional[CopywritingApplicationService] = None
    
    # Use cases
    generate_use_case: Optional[GenerateCopywritingUseCase] = None
    history_use_case: Optional[GetCopywritingHistoryUseCase] = None
    
    # Controllers
    copywriting_controller: Optional[CopywritingController] = None
    
    # State
    is_initialized: bool = False
    
    async def initialize(self) -> Any:
        """Initialize all services in dependency order."""
        if self.is_initialized:
            return
        
        try:
            # Initialize infrastructure services
            await self._initialize_infrastructure()
            
            # Initialize application services
            await self._initialize_application()
            
            # Initialize controllers
            await self._initialize_controllers()
            
            self.is_initialized = True
            
        except Exception as e:
            raise Exception(f"Failed to initialize container: {e}")
    
    async def _initialize_infrastructure(self) -> Any:
        """Initialize infrastructure services."""
        # Repository
        self.repository = PostgresCopywritingRepository(
            database_url=self.settings.database_url,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow
        )
        await self.repository.initialize()
        
        # Cache service
        self.cache_service = RedisCacheService(
            redis_url=self.settings.redis_url,
            pool_size=self.settings.redis_pool_size,
            max_connections=self.settings.redis_max_connections,
            ttl=self.settings.cache_ttl,
            max_size=self.settings.cache_max_size
        )
        await self.cache_service.initialize()
        
        # AI service
        self.ai_service = DevinAIService(
            model_name=self.settings.ai_model_name,
            cache_dir=self.settings.ai_model_cache_dir,
            max_length=self.settings.ai_max_length,
            temperature=self.settings.ai_temperature
        )
        await self.ai_service.initialize()
        
        # Event publisher
        self.event_publisher = AsyncEventPublisher()
        await self.event_publisher.initialize()
        
        # Monitoring service
        self.monitoring_service = PrometheusMonitoringService()
        await self.monitoring_service.initialize()
    
    async def _initialize_application(self) -> Any:
        """Initialize application services."""
        # Application service
        self.application_service = CopywritingApplicationService(
            repository=self.repository,
            ai_service=self.ai_service,
            cache_service=self.cache_service,
            event_publisher=self.event_publisher,
            monitoring_service=self.monitoring_service
        )
        
        # Use cases
        self.generate_use_case = GenerateCopywritingUseCase(
            repository=self.repository,
            ai_service=self.ai_service,
            cache_service=self.cache_service,
            event_publisher=self.event_publisher
        )
        
        self.history_use_case = GetCopywritingHistoryUseCase(
            repository=self.repository
        )
    
    async def _initialize_controllers(self) -> Any:
        """Initialize controllers."""
        self.copywriting_controller = CopywritingController(
            generate_use_case=self.generate_use_case,
            history_use_case=self.history_use_case
        )
    
    async def cleanup(self) -> Any:
        """Cleanup all services."""
        if not self.is_initialized:
            return
        
        try:
            # Cleanup in reverse order
            if self.repository:
                await self.repository.cleanup()
            
            if self.cache_service:
                await self.cache_service.cleanup()
            
            if self.ai_service:
                await self.ai_service.cleanup()
            
            if self.event_publisher:
                await self.event_publisher.cleanup()
            
            if self.monitoring_service:
                await self.monitoring_service.cleanup()
            
            self.is_initialized = False
            
        except Exception as e:
            raise Exception(f"Failed to cleanup container: {e}")
    
    def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service by name."""
        services = {
            "repository": self.repository,
            "cache_service": self.cache_service,
            "ai_service": self.ai_service,
            "event_publisher": self.event_publisher,
            "monitoring_service": self.monitoring_service,
            "application_service": self.application_service,
            "generate_use_case": self.generate_use_case,
            "history_use_case": self.history_use_case,
            "copywriting_controller": self.copywriting_controller,
        }
        
        if service_name not in services:
            raise ValueError(f"Service '{service_name}' not found")
        
        service = services[service_name]
        if service is None:
            raise ValueError(f"Service '{service_name}' not initialized")
        
        return service


def create_container(settings: Any) -> Container:
    """Create and return a new container instance."""
    return Container(settings=settings) 