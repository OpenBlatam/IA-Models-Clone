"""
Base classes for the AI History Comparison System

This module provides base classes that implement common functionality
for analyzers, engines, and services.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime
from dataclasses import dataclass

from .config import SystemConfig, get_config
from .interfaces import IAnalyzer, IEngine, IService
from .exceptions import AIHistoryException, ValidationError

T = TypeVar('T')
logger = logging.getLogger(__name__)


@dataclass
class ComponentStatus:
    """Status information for a component"""
    name: str
    status: str  # "initialized", "running", "stopped", "error"
    last_updated: datetime
    metrics: Dict[str, Any]
    error_message: Optional[str] = None


class BaseComponent(ABC):
    """Base class for all components"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._status = ComponentStatus(
            name=self.__class__.__name__,
            status="initialized",
            last_updated=datetime.utcnow(),
            metrics={}
        )
        self._initialized = False
    
    @property
    def status(self) -> ComponentStatus:
        """Get component status"""
        return self._status
    
    def _update_status(self, status: str, error_message: Optional[str] = None):
        """Update component status"""
        self._status.status = status
        self._status.last_updated = datetime.utcnow()
        if error_message:
            self._status.error_message = error_message
        self.logger.info(f"Status updated to: {status}")
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update component metrics"""
        self._status.metrics.update(metrics)
        self._status.last_updated = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """Initialize the component"""
        try:
            self._update_status("initializing")
            result = await self._initialize()
            if result:
                self._initialized = True
                self._update_status("initialized")
                self.logger.info("Component initialized successfully")
            else:
                self._update_status("error", "Initialization failed")
            return result
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    @abstractmethod
    async def _initialize(self) -> bool:
        """Component-specific initialization"""
        pass
    
    async def shutdown(self) -> bool:
        """Shutdown the component"""
        try:
            self._update_status("shutting_down")
            result = await self._shutdown()
            self._update_status("stopped")
            self.logger.info("Component shutdown successfully")
            return result
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Shutdown failed: {e}")
            return False
    
    @abstractmethod
    async def _shutdown(self) -> bool:
        """Component-specific shutdown"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status"""
        return {
            "name": self._status.name,
            "status": self._status.status,
            "initialized": self._initialized,
            "last_updated": self._status.last_updated.isoformat(),
            "metrics": self._status.metrics,
            "error_message": self._status.error_message
        }


class BaseAnalyzer(BaseComponent, IAnalyzer[T], Generic[T]):
    """Base class for all analyzers"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._analysis_count = 0
        self._total_analysis_time = 0.0
    
    async def analyze(self, data: T, **kwargs) -> Dict[str, Any]:
        """Analyze the provided data and return results"""
        if not self._initialized:
            raise AIHistoryException("Analyzer not initialized")
        
        if not self.validate_input(data):
            raise ValidationError("Invalid input data")
        
        start_time = datetime.utcnow()
        try:
            self._update_status("analyzing")
            result = await self._analyze(data, **kwargs)
            self._analysis_count += 1
            
            # Update metrics
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            self._total_analysis_time += analysis_time
            self._update_metrics({
                "analysis_count": self._analysis_count,
                "total_analysis_time": self._total_analysis_time,
                "average_analysis_time": self._total_analysis_time / self._analysis_count,
                "last_analysis_time": analysis_time
            })
            
            self._update_status("initialized")
            return result
            
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    @abstractmethod
    async def _analyze(self, data: T, **kwargs) -> Dict[str, Any]:
        """Component-specific analysis implementation"""
        pass
    
    async def batch_analyze(self, data_list: List[T], **kwargs) -> List[Dict[str, Any]]:
        """Analyze multiple data items in batch"""
        if not self._initialized:
            raise AIHistoryException("Analyzer not initialized")
        
        results = []
        for data in data_list:
            if self.validate_input(data):
                result = await self.analyze(data, **kwargs)
                results.append(result)
            else:
                self.logger.warning(f"Skipping invalid data item")
                results.append({"error": "Invalid input data"})
        
        return results
    
    @abstractmethod
    def get_analysis_metrics(self) -> List[str]:
        """Get list of metrics this analyzer produces"""
        pass
    
    def validate_input(self, data: T) -> bool:
        """Validate input data before analysis"""
        return data is not None
    
    async def _initialize(self) -> bool:
        """Initialize the analyzer"""
        return True
    
    async def _shutdown(self) -> bool:
        """Shutdown the analyzer"""
        return True


class BaseEngine(BaseComponent, IEngine[T], Generic[T]):
    """Base class for all engines"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._processing_count = 0
        self._total_processing_time = 0.0
    
    async def process(self, data: T, **kwargs) -> Dict[str, Any]:
        """Process the provided data"""
        if not self._initialized:
            raise AIHistoryException("Engine not initialized")
        
        start_time = datetime.utcnow()
        try:
            self._update_status("processing")
            result = await self._process(data, **kwargs)
            self._processing_count += 1
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._total_processing_time += processing_time
            self._update_metrics({
                "processing_count": self._processing_count,
                "total_processing_time": self._total_processing_time,
                "average_processing_time": self._total_processing_time / self._processing_count,
                "last_processing_time": processing_time
            })
            
            self._update_status("initialized")
            return result
            
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Processing failed: {e}")
            raise
    
    @abstractmethod
    async def _process(self, data: T, **kwargs) -> Dict[str, Any]:
        """Component-specific processing implementation"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return self.get_health_status()
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of engine capabilities"""
        pass
    
    async def _initialize(self) -> bool:
        """Initialize the engine"""
        return True
    
    async def _shutdown(self) -> bool:
        """Shutdown the engine"""
        return True


class BaseService(BaseComponent, IService[T], Generic[T]):
    """Base class for all services"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._running = False
        self._start_time: Optional[datetime] = None
    
    async def start(self) -> bool:
        """Start the service"""
        if self._running:
            self.logger.warning("Service is already running")
            return True
        
        try:
            self._update_status("starting")
            result = await self._start()
            if result:
                self._running = True
                self._start_time = datetime.utcnow()
                self._update_status("running")
                self.logger.info("Service started successfully")
            else:
                self._update_status("error", "Start failed")
            return result
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Start failed: {e}")
            return False
    
    @abstractmethod
    async def _start(self) -> bool:
        """Component-specific start implementation"""
        pass
    
    async def stop(self) -> bool:
        """Stop the service"""
        if not self._running:
            self.logger.warning("Service is not running")
            return True
        
        try:
            self._update_status("stopping")
            result = await self._stop()
            self._running = False
            self._start_time = None
            self._update_status("stopped")
            self.logger.info("Service stopped successfully")
            return result
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Stop failed: {e}")
            return False
    
    @abstractmethod
    async def _stop(self) -> bool:
        """Component-specific stop implementation"""
        pass
    
    async def restart(self) -> bool:
        """Restart the service"""
        self.logger.info("Restarting service")
        await self.stop()
        await asyncio.sleep(1)  # Brief pause
        return await self.start()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        base_status = super().get_health_status()
        base_status.update({
            "running": self._running,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "uptime": (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
        })
        return base_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return self._status.metrics
    
    def is_running(self) -> bool:
        """Check if service is running"""
        return self._running
    
    async def _initialize(self) -> bool:
        """Initialize the service"""
        return True
    
    async def _shutdown(self) -> bool:
        """Shutdown the service"""
        if self._running:
            await self.stop()
        return True


class BaseRepository(BaseComponent, Generic[T, str]):
    """Base class for all repositories"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._operation_count = 0
        self._total_operation_time = 0.0
    
    async def _execute_operation(self, operation_name: str, operation_func):
        """Execute a repository operation with metrics tracking"""
        start_time = datetime.utcnow()
        try:
            self._update_status(f"executing_{operation_name}")
            result = await operation_func()
            self._operation_count += 1
            
            # Update metrics
            operation_time = (datetime.utcnow() - start_time).total_seconds()
            self._total_operation_time += operation_time
            self._update_metrics({
                "operation_count": self._operation_count,
                "total_operation_time": self._total_operation_time,
                "average_operation_time": self._total_operation_time / self._operation_count,
                "last_operation_time": operation_time,
                "last_operation": operation_name
            })
            
            self._update_status("initialized")
            return result
            
        except Exception as e:
            self._update_status("error", str(e))
            self.logger.error(f"Operation {operation_name} failed: {e}")
            raise
    
    async def _initialize(self) -> bool:
        """Initialize the repository"""
        return True
    
    async def _shutdown(self) -> bool:
        """Shutdown the repository"""
        return True





















