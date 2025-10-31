"""
Enhanced Email Sequence Engine

Integrates all advanced improvements including database integration,
message queue, security enhancements, and real-time streaming.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

# Import all the new enhancement modules
from .database_integration import (
    DatabaseConnectionPool, DatabaseConfig, DatabaseType, CacheStrategy,
    EmailSequenceRepository, SubscriberRepository, TemplateRepository
)
from .message_queue_integration import (
    MessageQueueManager, QueueConfig, QueueType, MessagePriority,
    EmailSequenceQueueService
)
from .security_enhancements import (
    SecurityManager, SecurityConfig, SecurityLevel, EncryptionType,
    EmailSequenceSecurityService
)
from .streaming_optimization import (
    RealTimeStreamManager, StreamConfig, StreamType, StreamEventType,
    EmailSequenceStreamService
)

# Import existing core components
from .email_sequence_engine import EmailSequenceEngine, ProcessingResult, EngineStatus
from .performance_optimizer import OptimizedPerformanceOptimizer, OptimizationConfig
from .advanced_optimizer import AdvancedOptimizer, AdvancedOptimizationConfig
from .intelligent_monitor import IntelligentMonitor, MonitoringConfig

# Import models
from ..models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from ..models.subscriber import Subscriber, SubscriberSegment
from ..models.template import EmailTemplate, TemplateVariable
from ..models.campaign import EmailCampaign, CampaignMetrics

# Import services
from ..services.langchain_service import LangChainEmailService
from ..services.delivery_service import EmailDeliveryService
from ..services.analytics_service import EmailAnalyticsService

logger = logging.getLogger(__name__)


@dataclass
class EnhancedEngineConfig:
    """Enhanced engine configuration"""
    # Database configuration
    database_type: DatabaseType = DatabaseType.POSTGRESQL
    database_url: str = "postgresql://user:password@localhost/email_sequence"
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    
    # Message queue configuration
    queue_type: QueueType = QueueType.REDIS_STREAMS
    queue_url: str = "redis://localhost:6379"
    
    # Security configuration
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    
    # Streaming configuration
    stream_type: StreamType = StreamType.WEBSOCKET
    enable_real_time: bool = True
    
    # Performance configuration
    max_concurrent_sequences: int = 50
    enable_ml_optimization: bool = True
    enable_monitoring: bool = True


class EnhancedEmailSequenceEngine:
    """
    Enhanced email sequence engine with all advanced features integrated.
    """
    
    def __init__(self, config: EnhancedEngineConfig):
        self.config = config
        self.status = EngineStatus.IDLE
        
        # Initialize all enhancement components
        self._initialize_components()
        
        # Core engine (will be initialized after components)
        self.core_engine = None
        
        # Statistics
        self.stats = {
            "sequences_processed": 0,
            "emails_sent": 0,
            "errors": 0,
            "start_time": None,
            "enhancements_active": []
        }
        
        logger.info("Enhanced Email Sequence Engine initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all enhancement components"""
        try:
            # Database integration
            db_config = DatabaseConfig(
                database_type=self.config.database_type,
                connection_string=self.config.database_url,
                cache_strategy=self.config.cache_strategy,
                enable_caching=True,
                enable_metrics=True
            )
            self.db_pool = DatabaseConnectionPool(db_config)
            self.sequence_repo = EmailSequenceRepository(self.db_pool)
            self.subscriber_repo = SubscriberRepository(self.db_pool)
            self.template_repo = TemplateRepository(self.db_pool)
            
            # Message queue integration
            queue_config = QueueConfig(
                queue_type=self.config.queue_type,
                connection_string=self.config.queue_url,
                enable_metrics=True,
                enable_dlq=True
            )
            self.queue_manager = MessageQueueManager(queue_config)
            self.queue_service = EmailSequenceQueueService(self.queue_manager)
            
            # Security enhancements
            security_config = SecurityConfig(
                security_level=self.config.security_level,
                enable_encryption=self.config.enable_encryption,
                enable_rate_limiting=self.config.enable_rate_limiting,
                enable_audit_logging=True,
                enable_jwt_auth=True
            )
            self.security_manager = SecurityManager(security_config)
            self.security_service = EmailSequenceSecurityService(self.security_manager)
            
            # Streaming optimization
            stream_config = StreamConfig(
                stream_type=self.config.stream_type,
                enable_metrics=True,
                enable_heartbeat=True
            )
            self.stream_manager = RealTimeStreamManager(stream_config)
            self.stream_service = EmailSequenceStreamService(self.stream_manager)
            
            # Performance optimization
            perf_config = OptimizationConfig(
                max_memory_usage=0.8,
                cache_size=1000,
                batch_size=64,
                max_concurrent_tasks=10,
                enable_caching=True,
                enable_memory_optimization=True,
                enable_batch_processing=True
            )
            self.performance_optimizer = OptimizedPerformanceOptimizer(perf_config)
            
            # Advanced ML optimization
            ml_config = AdvancedOptimizationConfig(
                enable_ml_optimization=self.config.enable_ml_optimization,
                enable_predictive_caching=True,
                enable_adaptive_batching=True,
                enable_intelligent_resource_management=True,
                enable_performance_prediction=True
            )
            self.advanced_optimizer = AdvancedOptimizer(ml_config)
            
            # Intelligent monitoring
            monitor_config = MonitoringConfig(
                monitoring_interval=5,
                alert_threshold=0.8,
                auto_optimization_enabled=True,
                enable_real_time_alerts=True,
                enable_performance_tracking=True,
                enable_resource_monitoring=True,
                enable_ml_insights=True
            )
            self.intelligent_monitor = IntelligentMonitor(
                monitor_config,
                self.performance_optimizer,
                self.advanced_optimizer
            )
            
            logger.info("All enhancement components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing enhancement components: {e}")
            raise
    
    async def initialize(self) -> ProcessingResult:
        """Initialize the enhanced engine"""
        try:
            self.status = EngineStatus.RUNNING
            self.stats["start_time"] = datetime.utcnow()
            
            # Initialize all components
            await self.db_pool.initialize()
            await self.queue_manager.initialize()
            await self.security_manager.initialize()
            await self.stream_manager.initialize()
            await self.intelligent_monitor.start_monitoring()
            
            # Initialize core engine with enhanced services
            self.core_engine = EmailSequenceEngine(
                langchain_service=LangChainEmailService(),
                delivery_service=EmailDeliveryService(),
                analytics_service=EmailAnalyticsService(),
                max_concurrent_sequences=self.config.max_concurrent_sequences
            )
            
            # Start core engine
            await self.core_engine.start()
            
            # Set up event handlers
            await self._setup_event_handlers()
            
            self.stats["enhancements_active"] = [
                "database_integration",
                "message_queue",
                "security_enhancements",
                "streaming_optimization",
                "performance_optimization",
                "ml_optimization",
                "intelligent_monitoring"
            ]
            
            logger.info("Enhanced Email Sequence Engine initialized successfully")
            return ProcessingResult(
                success=True,
                message="Enhanced engine initialized successfully",
                data={"status": self.status.value, "enhancements": self.stats["enhancements_active"]}
            )
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Error initializing enhanced engine: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to initialize enhanced engine: {str(e)}",
                error=e
            )
    
    async def _setup_event_handlers(self) -> None:
        """Set up event handlers for real-time updates"""
        try:
            # Subscribe to sequence events
            await self.stream_service.subscribe_to_sequence_events(
                "all",
                self._handle_sequence_event
            )
            
            # Subscribe to email events
            await self.stream_service.subscribe_to_email_events(
                self._handle_email_event
            )
            
            logger.info("Event handlers set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up event handlers: {e}")
    
    async def _handle_sequence_event(self, event) -> None:
        """Handle sequence events"""
        try:
            # Log security event
            await self.security_manager.log_security_event(
                event_type="sequence_event",
                user_id=event.metadata.get("user_id", "system"),
                ip_address="internal",
                user_agent="enhanced_engine",
                details=event.data,
                severity=SecurityLevel.MEDIUM,
                success=True
            )
            
            # Publish to queue
            await self.queue_service.publish_sequence_event(
                event.event_type.value,
                event.data
            )
            
        except Exception as e:
            logger.error(f"Error handling sequence event: {e}")
    
    async def _handle_email_event(self, event) -> None:
        """Handle email events"""
        try:
            # Publish to queue
            await self.queue_service.publish_email_event(
                event.event_type.value,
                event.data
            )
            
        except Exception as e:
            logger.error(f"Error handling email event: {e}")
    
    async def create_sequence(
        self,
        name: str,
        target_audience: str,
        goals: List[str],
        tone: str = "professional",
        templates: List[EmailTemplate] = None,
        user_id: str = None
    ) -> ProcessingResult:
        """Create a new email sequence with enhanced features"""
        try:
            # Security validation
            if user_id:
                is_valid = await self.security_manager.check_rate_limit(user_id, "internal")
                if not is_valid:
                    return ProcessingResult(
                        success=False,
                        message="Rate limit exceeded",
                        error=Exception("Rate limit exceeded")
                    )
            
            # Create sequence using core engine
            result = await self.core_engine.create_sequence(
                name, target_audience, goals, tone, templates
            )
            
            if result.success:
                # Store in database
                sequence = result.data.get("sequence")
                if sequence:
                    await self.sequence_repo.create_sequence(sequence)
                    
                    # Secure the sequence data
                    secured_sequence = await self.security_service.secure_sequence_data(sequence)
                    
                    # Publish event
                    await self.stream_service.publish_sequence_event(
                        StreamEventType.SEQUENCE_CREATED,
                        str(sequence.id),
                        {
                            "name": sequence.name,
                            "target_audience": target_audience,
                            "goals": goals,
                            "tone": tone
                        }
                    )
                    
                    # Optimize with ML
                    if self.config.enable_ml_optimization:
                        await self.advanced_optimizer.optimize_with_ml([sequence])
                
                self.stats["sequences_processed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating sequence: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to create sequence: {str(e)}",
                error=e
            )
    
    async def activate_sequence(self, sequence_id: UUID, user_id: str = None) -> ProcessingResult:
        """Activate a sequence with enhanced features"""
        try:
            # Security validation
            if user_id:
                access_valid = await self.security_service.validate_sequence_access(
                    user_id, str(sequence_id), "activate"
                )
                if not access_valid:
                    return ProcessingResult(
                        success=False,
                        message="Access denied",
                        error=Exception("Access denied")
                    )
            
            # Get sequence from database
            sequence = await self.sequence_repo.get_sequence(str(sequence_id))
            if not sequence:
                return ProcessingResult(
                    success=False,
                    message="Sequence not found",
                    error=Exception("Sequence not found")
                )
            
            # Activate using core engine
            result = await self.core_engine.activate_sequence(sequence_id)
            
            if result.success:
                # Update in database
                sequence.status = "active"
                await self.sequence_repo.update_sequence(sequence)
                
                # Publish event
                await self.stream_service.publish_sequence_event(
                    StreamEventType.SEQUENCE_UPDATED,
                    str(sequence_id),
                    {"status": "active"}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error activating sequence: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to activate sequence: {str(e)}",
                error=e
            )
    
    async def add_subscribers_to_sequence(
        self,
        sequence_id: UUID,
        subscribers: List[Subscriber],
        user_id: str = None
    ) -> ProcessingResult:
        """Add subscribers to sequence with enhanced features"""
        try:
            # Security validation
            if user_id:
                access_valid = await self.security_service.validate_sequence_access(
                    user_id, str(sequence_id), "add_subscribers"
                )
                if not access_valid:
                    return ProcessingResult(
                        success=False,
                        message="Access denied",
                        error=Exception("Access denied")
                    )
            
            # Secure subscriber data
            secured_subscribers = []
            for subscriber in subscribers:
                secured_subscriber = await self.security_service.secure_subscriber_data(subscriber)
                secured_subscribers.append(secured_subscriber)
                
                # Store in database
                await self.subscriber_repo.create_subscriber(secured_subscriber)
            
            # Add using core engine
            result = await self.core_engine.add_subscribers_to_sequence(sequence_id, secured_subscribers)
            
            if result.success:
                # Publish events
                for subscriber in secured_subscribers:
                    await self.stream_service.publish_sequence_event(
                        StreamEventType.SUBSCRIBER_ADDED,
                        str(sequence_id),
                        {
                            "subscriber_id": str(subscriber.id),
                            "email": subscriber.email
                        }
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding subscribers: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to add subscribers: {str(e)}",
                error=e
            )
    
    async def get_sequence_analytics(
        self,
        sequence_id: UUID,
        user_id: str = None
    ) -> ProcessingResult:
        """Get sequence analytics with enhanced features"""
        try:
            # Security validation
            if user_id:
                access_valid = await self.security_service.validate_sequence_access(
                    user_id, str(sequence_id), "view_analytics"
                )
                if not access_valid:
                    return ProcessingResult(
                        success=False,
                        message="Access denied",
                        error=Exception("Access denied")
                    )
            
            # Get from core engine
            result = await self.core_engine.get_sequence_analytics(sequence_id)
            
            if result.success:
                # Add real-time streaming analytics
                analytics_data = result.data
                analytics_data["real_time_metrics"] = self.stream_service.get_stream_metrics()
                analytics_data["security_metrics"] = self.security_service.get_security_metrics()
                analytics_data["performance_metrics"] = self.performance_optimizer.get_performance_stats()
                
                result.data = analytics_data
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to get analytics: {str(e)}",
                error=e
            )
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        try:
            stats = {
                "engine_status": self.status.value,
                "start_time": self.stats["start_time"],
                "sequences_processed": self.stats["sequences_processed"],
                "emails_sent": self.stats["emails_sent"],
                "errors": self.stats["errors"],
                "enhancements_active": self.stats["enhancements_active"],
                
                # Database metrics
                "database_metrics": self.db_pool.get_performance_stats(),
                
                # Queue metrics
                "queue_metrics": self.queue_manager.get_metrics(),
                
                # Security metrics
                "security_metrics": self.security_manager.get_security_metrics(),
                
                # Stream metrics
                "stream_metrics": self.stream_manager.get_stream_metrics(),
                
                # Performance metrics
                "performance_metrics": self.performance_optimizer.get_performance_stats(),
                
                # ML metrics
                "ml_metrics": self.advanced_optimizer.get_advanced_metrics(),
                
                # Monitoring metrics
                "monitoring_metrics": self.intelligent_monitor.get_monitoring_metrics()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> ProcessingResult:
        """Cleanup all components"""
        try:
            self.status = EngineStatus.STOPPING
            
            # Cleanup core engine
            if self.core_engine:
                await self.core_engine.stop()
            
            # Cleanup all enhancement components
            await self.db_pool.cleanup()
            await self.queue_manager.cleanup()
            await self.security_manager.cleanup()
            await self.stream_manager.cleanup()
            await self.intelligent_monitor.stop_monitoring()
            
            self.status = EngineStatus.IDLE
            
            logger.info("Enhanced Email Sequence Engine cleaned up successfully")
            return ProcessingResult(
                success=True,
                message="Enhanced engine cleaned up successfully",
                data={"status": self.status.value}
            )
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"Error during cleanup: {e}")
            return ProcessingResult(
                success=False,
                message=f"Failed to cleanup enhanced engine: {str(e)}",
                error=e
            ) 