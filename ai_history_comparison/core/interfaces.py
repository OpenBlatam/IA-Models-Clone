"""
Core interfaces for the AI History Comparison System

This module defines the core interfaces that all components should implement
to ensure consistency and interoperability across the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime

T = TypeVar('T')
K = TypeVar('K')


class IAnalyzer(ABC, Generic[T]):
    """Base interface for all analyzers"""
    
    @abstractmethod
    async def analyze(self, data: T, **kwargs) -> Dict[str, Any]:
        """Analyze the provided data and return results"""
        pass
    
    @abstractmethod
    async def batch_analyze(self, data_list: List[T], **kwargs) -> List[Dict[str, Any]]:
        """Analyze multiple data items in batch"""
        pass
    
    @abstractmethod
    def get_analysis_metrics(self) -> List[str]:
        """Get list of metrics this analyzer produces"""
        pass
    
    @abstractmethod
    def validate_input(self, data: T) -> bool:
        """Validate input data before analysis"""
        pass


class IEngine(ABC, Generic[T]):
    """Base interface for all engines"""
    
    @abstractmethod
    async def process(self, data: T, **kwargs) -> Dict[str, Any]:
        """Process the provided data"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the engine"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the engine"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of engine capabilities"""
        pass


class IService(ABC, Generic[T]):
    """Base interface for all services"""
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the service"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service"""
        pass
    
    @abstractmethod
    async def restart(self) -> bool:
        """Restart the service"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        pass


class IRepository(ABC, Generic[T, K]):
    """Base interface for all repositories"""
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity"""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: K) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[T]:
        """Get all entities with optional pagination"""
        pass
    
    @abstractmethod
    async def update(self, entity_id: K, updates: Dict[str, Any]) -> Optional[T]:
        """Update entity by ID"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: K) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    async def search(self, filters: Dict[str, Any], limit: Optional[int] = None) -> List[T]:
        """Search entities with filters"""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters"""
        pass


class IContentAnalyzer(IAnalyzer[str]):
    """Interface for content analysis"""
    
    @abstractmethod
    async def analyze_quality(self, content: str) -> Dict[str, Any]:
        """Analyze content quality"""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze content sentiment"""
        pass
    
    @abstractmethod
    async def analyze_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze content complexity"""
        pass
    
    @abstractmethod
    async def analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability"""
        pass


class IComparisonEngine(IEngine[Dict[str, Any]]):
    """Interface for comparison operations"""
    
    @abstractmethod
    async def compare_content(self, content1: str, content2: str) -> Dict[str, Any]:
        """Compare two content pieces"""
        pass
    
    @abstractmethod
    async def compare_models(self, model1_results: Dict[str, Any], model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two model results"""
        pass
    
    @abstractmethod
    async def find_similar_content(self, content: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar content pieces"""
        pass


class ITrendAnalyzer(IAnalyzer[List[Dict[str, Any]]]):
    """Interface for trend analysis"""
    
    @abstractmethod
    async def analyze_trends(self, data: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        """Analyze trends in the data"""
        pass
    
    @abstractmethod
    async def predict_future(self, data: List[Dict[str, Any]], metric: str, days: int = 7) -> Dict[str, Any]:
        """Predict future values"""
        pass
    
    @abstractmethod
    async def detect_anomalies(self, data: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        pass


class IReportingService(IService[Dict[str, Any]]):
    """Interface for reporting services"""
    
    @abstractmethod
    async def generate_report(self, data: Dict[str, Any], template: str) -> Dict[str, Any]:
        """Generate a report from data"""
        pass
    
    @abstractmethod
    async def schedule_report(self, data_source: str, template: str, schedule: str) -> str:
        """Schedule a recurring report"""
        pass
    
    @abstractmethod
    async def export_report(self, report_id: str, format: str) -> bytes:
        """Export report in specified format"""
        pass


class INotificationService(IService[Dict[str, Any]]):
    """Interface for notification services"""
    
    @abstractmethod
    async def send_notification(self, message: Dict[str, Any], channels: List[str]) -> bool:
        """Send notification to specified channels"""
        pass
    
    @abstractmethod
    async def subscribe(self, user_id: str, event_types: List[str]) -> bool:
        """Subscribe user to event types"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, user_id: str, event_types: List[str]) -> bool:
        """Unsubscribe user from event types"""
        pass


class IDataProcessor(IEngine[Any]):
    """Interface for data processing"""
    
    @abstractmethod
    async def process_data(self, data: Any, pipeline: str) -> Any:
        """Process data through specified pipeline"""
        pass
    
    @abstractmethod
    async def validate_data(self, data: Any) -> bool:
        """Validate data format and content"""
        pass
    
    @abstractmethod
    async def transform_data(self, data: Any, transformation: str) -> Any:
        """Transform data using specified transformation"""
        pass


class ICacheService(IService[Any]):
    """Interface for caching services"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass


class IIntegrationService(IService[Dict[str, Any]]):
    """Interface for external integrations"""
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to external service"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from external service"""
        pass
    
    @abstractmethod
    async def sync_data(self, direction: str) -> Dict[str, Any]:
        """Sync data with external service"""
        pass
    
    @abstractmethod
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        pass


class IMonitoringService(IService[Dict[str, Any]]):
    """Interface for monitoring services"""
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        pass
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check system health"""
        pass
    
    @abstractmethod
    async def log_event(self, event: Dict[str, Any]) -> bool:
        """Log an event"""
        pass
    
    @abstractmethod
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        pass


class IWorkflowEngine(IEngine[Dict[str, Any]]):
    """Interface for workflow engines"""
    
    @abstractmethod
    async def execute_workflow(self, workflow_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        pass
    
    @abstractmethod
    async def create_workflow(self, definition: Dict[str, Any]) -> str:
        """Create a new workflow"""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        pass
    
    @abstractmethod
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause workflow execution"""
        pass
    
    @abstractmethod
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume workflow execution"""
        pass





















