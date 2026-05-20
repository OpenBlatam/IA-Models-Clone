"""
Component Manager
Manages initialization and access to various AI components.
"""

import logging
from typing import Optional

from ..config import UnifiedAIConfig
from .llm_service import LLMService
from .knowledge_base import KnowledgeBase
from .learning_engine import LearningEngine
from .reasoning_engine import ReasoningEngine
from .self_reflection import SelfReflectionEngine
from .world_model import ContinualWorldModel
from .experience_driven_learning import ExperienceDrivenLearning
from .workflow_engine import WorkflowEngine
from .analytics import AnalyticsSystem
from .autonomous_operation_handler import AutonomousOperationHandler
from .data_processing import DataProcessor
from .resilience import ResilienceManager
from .health_check import HealthChecker
from .rate_limiter import RateLimiter
from .document_exporter import DocumentExporter
from .http_client import AsyncHttpClient
from .template_engine import TemplateEngine
from .event_bus import EventBus

logger = logging.getLogger(__name__)

class ComponentManager:
    """
    Manages the lifecycle and access to AI components.
    """
    
    def __init__(
        self, 
        config: UnifiedAIConfig,
        llm_service: LLMService,
        agent_id: str,
        system_prompt: str
    ):
        self.config = config
        self.llm_service = llm_service
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        
        # Initialize components
        self.knowledge_base = self._init_knowledge_base()
        self.learning_engine = self._init_learning_engine()
        self.reasoning_engine = self._init_reasoning_engine()
        
        # Optional components
        self.world_model = self._init_world_model()
        self.self_reflection_engine = self._init_self_reflection_engine()
        self.experience_learning = self._init_experience_learning()
        
        # Bulk features
        self.workflow_engine = self._init_workflow_engine()
        self.analytics = self._init_analytics()
        
        # Data processing & export
        self.data_processor = self._init_data_processor()
        self.document_exporter = self._init_document_exporter()
        
        # Utilities
        self.http_client = self._init_http_client()
        self.template_engine = self._init_template_engine()
        self.event_bus = self._init_event_bus()
        
        # Resilience & Monitoring
        self.resilience = self._init_resilience()
        self.health_checker = self._init_health_checker()
        self.rate_limiter = self._init_rate_limiter()
        
        # Handler
        self.autonomous_handler = self._init_autonomous_handler()
        
        logger.info("ComponentManager initialized successfully")

    def _init_knowledge_base(self) -> KnowledgeBase:
        return KnowledgeBase(
            max_entries=self.config.autonomous.max_knowledge_entries,
            retention_days=self.config.autonomous.knowledge_base_retention_days
        )

    def _init_learning_engine(self) -> LearningEngine:
        return LearningEngine(
            adaptation_rate=self.config.autonomous.learning_adaptation_rate,
            learning_enabled=self.config.autonomous.learning_enabled
        )

    def _init_reasoning_engine(self) -> ReasoningEngine:
        return ReasoningEngine(
            knowledge_base=self.knowledge_base,
            llm_service=self.llm_service
        )

    def _init_world_model(self) -> Optional[ContinualWorldModel]:
        if self.config.autonomous.enable_world_model:
            return ContinualWorldModel()
        return None

    def _init_self_reflection_engine(self) -> Optional[SelfReflectionEngine]:
        if self.config.autonomous.enable_self_reflection:
            return SelfReflectionEngine()
        return None

    def _init_experience_learning(self) -> Optional[ExperienceDrivenLearning]:
        if self.config.autonomous.enable_experience_learning:
            return ExperienceDrivenLearning()
        return None

    def _init_workflow_engine(self) -> WorkflowEngine:
        return WorkflowEngine(llm_service=self.llm_service)

    def _init_analytics(self) -> AnalyticsSystem:
        return AnalyticsSystem()

    def _init_data_processor(self) -> DataProcessor:
        return DataProcessor()

    def _init_document_exporter(self) -> DocumentExporter:
        return DocumentExporter()

    def _init_http_client(self) -> AsyncHttpClient:
        return AsyncHttpClient()

    def _init_template_engine(self) -> TemplateEngine:
        return TemplateEngine()

    def _init_event_bus(self) -> EventBus:
        return EventBus()

    def _init_resilience(self) -> ResilienceManager:
        return ResilienceManager()

    def _init_health_checker(self) -> HealthChecker:
        return HealthChecker()

    def _init_rate_limiter(self) -> RateLimiter:
        return RateLimiter()

    def _init_autonomous_handler(self) -> AutonomousOperationHandler:
        return AutonomousOperationHandler(
            learning_engine=self.learning_engine,
            world_model=self.world_model,
            agent_id=self.agent_id,
            instruction=self.system_prompt
        )
