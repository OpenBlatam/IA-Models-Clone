"""
Dependency Injection Container
Configure and manage dependencies following Clean Architecture
"""

from typing import Any, Dict
from ..layers import Container

from .domain.entities import DocumentEntity, VariantEntity, TopicEntity
from .domain.services import (
    DocumentAccessService,
    VariantQualityService,
    TopicRelevanceService
)
from .domain.events import EventBus
from .application.use_cases import (
    UploadDocumentUseCase,
    GetDocumentUseCase,
    ListDocumentsUseCase,
    GenerateVariantsUseCase,
    ExtractTopicsUseCase
)
from .application.handlers import (
    UploadDocumentUseCaseImpl,
    GetDocumentUseCaseImpl,
    ListDocumentsUseCaseImpl,
    GenerateVariantsUseCaseImpl,
    ExtractTopicsUseCaseImpl
)
from .infrastructure.repositories import (
    DocumentRepository,
    VariantRepository,
    TopicRepository
)
from .infrastructure.event_bus import (
    InMemoryEventBus,
    DocumentEventHandler,
    VariantEventHandler,
    TopicEventHandler
)
from .presentation.controllers import (
    DocumentController,
    VariantController,
    TopicController
)


class ApplicationContainer(Container):
    """Application dependency injection container"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self._event_bus = None
        
        # Setup in order: infrastructure -> domain -> application -> presentation
        self._setup_repositories()
        self._setup_domain_services()
        self._setup_use_cases()
        self._setup_controllers()
    
    def _setup_repositories(self):
        """Setup repository layer"""
        # Register repositories
        # In a real implementation, these would be properly initialized
        # with database connections, etc.
        
        # For now, register as singletons
        doc_repo = DocumentRepository()
        variant_repo = VariantRepository()
        topic_repo = TopicRepository()
        
        self.register_instance(DocumentRepository, doc_repo)
        self.register_instance(VariantRepository, variant_repo)
        self.register_instance(TopicRepository, topic_repo)
    
    def _setup_event_bus(self):
        """Setup event bus and handlers"""
        event_bus = InMemoryEventBus()
        
        # Register event handlers
        doc_handler = DocumentEventHandler()
        variant_handler = VariantEventHandler()
        topic_handler = TopicEventHandler()
        
        # Subscribe handlers (in real impl, would use domain event types)
        from .domain.events import (
            DocumentUploadedEvent,
            DocumentProcessedEvent,
            DocumentDeletedEvent,
            VariantsGeneratedEvent,
            TopicsExtractedEvent
        )
        
        # Register handlers
        self.register_instance(EventBus, event_bus)
        self._event_bus = event_bus
        
        # Subscribe handlers (would be done properly in real impl)
        return event_bus
    
    def _setup_domain_services(self):
        """Setup domain services"""
        access_service = DocumentAccessService()
        quality_service = VariantQualityService()
        relevance_service = TopicRelevanceService()
        
        self.register_instance(DocumentAccessService, access_service)
        self.register_instance(VariantQualityService, quality_service)
        self.register_instance(TopicRelevanceService, relevance_service)
    
    def _setup_use_cases(self):
        """Setup use case layer with dependencies"""
        # Get dependencies
        doc_repo = self.resolve(DocumentRepository)
        variant_repo = self.resolve(VariantRepository)
        topic_repo = self.resolve(TopicRepository)
        event_bus = self._setup_event_bus()
        access_service = self.resolve(DocumentAccessService)
        quality_service = self.resolve(VariantQualityService)
        relevance_service = self.resolve(TopicRelevanceService)
        
        # Create use cases
        upload_use_case = UploadDocumentUseCaseImpl(
            doc_repo, event_bus, access_service
        )
        get_use_case = GetDocumentUseCaseImpl(doc_repo, access_service)
        list_use_case = ListDocumentsUseCaseImpl(doc_repo, access_service)
        generate_use_case = GenerateVariantsUseCaseImpl(
            doc_repo, variant_repo, quality_service, event_bus
        )
        extract_use_case = ExtractTopicsUseCaseImpl(
            doc_repo, topic_repo, relevance_service, event_bus
        )
        
        # Register use cases
        self.register_instance(UploadDocumentUseCase, upload_use_case)
        self.register_instance(GetDocumentUseCase, get_use_case)
        self.register_instance(ListDocumentsUseCase, list_use_case)
        self.register_instance(GenerateVariantsUseCase, generate_use_case)
        self.register_instance(ExtractTopicsUseCase, extract_use_case)
    
    def _setup_controllers(self):
        """Setup controller layer"""
        # Get use cases
        upload_use_case = self.resolve(UploadDocumentUseCase)
        get_use_case = self.resolve(GetDocumentUseCase)
        list_use_case = self.resolve(ListDocumentsUseCase)
        generate_use_case = self.resolve(GenerateVariantsUseCase)
        extract_use_case = self.resolve(ExtractTopicsUseCase)
        
        # Create controllers
        doc_controller = DocumentController(
            upload_use_case, get_use_case, list_use_case
        )
        variant_controller = VariantController(generate_use_case)
        topic_controller = TopicController(extract_use_case)
        
        # Register controllers
        self.register_instance(DocumentController, doc_controller)
        self.register_instance(VariantController, variant_controller)
        self.register_instance(TopicController, topic_controller)
    
    def get_document_controller(self) -> DocumentController:
        """Get document controller with dependencies"""
        return self.resolve(DocumentController)
    
    def get_variant_controller(self) -> VariantController:
        """Get variant controller with dependencies"""
        return self.resolve(VariantController)
    
    def get_topic_controller(self) -> TopicController:
        """Get topic controller with dependencies"""
        return self.resolve(TopicController)

