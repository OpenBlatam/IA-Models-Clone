"""
Application Layer - Command and Query Handlers
Concrete implementations of use cases
"""

from typing import List, Optional
from datetime import datetime
from uuid import uuid4

from ..domain.entities import DocumentEntity, VariantEntity, TopicEntity
from ..domain.value_objects import DocumentId, UserId, Filename, FileSize
from ..domain.services import (
    DocumentAccessService,
    VariantQualityService,
    TopicRelevanceService
)
from ..domain.events import (
    DocumentUploadedEvent,
    DocumentProcessedEvent,
    VariantsGeneratedEvent,
    TopicsExtractedEvent,
    EventBus
)
from .use_cases import (
    UploadDocumentUseCase,
    UploadDocumentCommand,
    GetDocumentUseCase,
    GetDocumentQuery,
    ListDocumentsUseCase,
    ListDocumentsQuery,
    GenerateVariantsUseCase,
    GenerateVariantsCommand,
    ExtractTopicsUseCase,
    ExtractTopicsCommand
)
from ..infrastructure.repositories import (
    DocumentRepository,
    VariantRepository,
    TopicRepository
)


class UploadDocumentUseCaseImpl(UploadDocumentUseCase):
    """Implementation of upload document use case"""
    
    def __init__(
        self,
        repository: DocumentRepository,
        event_bus: EventBus,
        access_service: DocumentAccessService
    ):
        self.repository = repository
        self.event_bus = event_bus
        self.access_service = access_service
    
    async def execute(self, command: UploadDocumentCommand) -> DocumentEntity:
        """Execute upload document use case"""
        # Validate filename
        filename_vo = Filename(command.filename)
        file_size_vo = FileSize(len(command.file_content))
        user_id_vo = UserId(command.user_id)
        
        # Create domain entity
        document = DocumentEntity(
            id=str(uuid4()),
            user_id=user_id_vo.value,
            filename=filename_vo.value,
            file_path=f"uploads/{str(uuid4())}.pdf",  # In real impl, save file
            file_size=file_size_vo.bytes,
            content_type="application/pdf",
            status="uploaded",
            created_at=datetime.utcnow(),
            metadata={}
        )
        
        # Validate for processing
        can_process, error = self.access_service.validate_document_for_processing(document)
        if not can_process and error:
            raise ValueError(error)
        
        # Save document
        saved_document = await self.repository.save(document)
        
        # Publish domain event
        event = DocumentUploadedEvent(
            document_id=saved_document.id,
            user_id=saved_document.user_id,
            filename=saved_document.filename,
            file_size=saved_document.file_size
        )
        await self.event_bus.publish(event)
        
        # Auto-process if requested
        if command.auto_process:
            saved_document.status = "processing"
            saved_document = await self.repository.save(saved_document)
            
            # Publish processing event (in real impl would process)
            process_event = DocumentProcessedEvent(
                document_id=saved_document.id,
                user_id=saved_document.user_id,
                processing_time=0.0,
                success=True
            )
            await self.event_bus.publish(process_event)
            saved_document.mark_as_processed()
            saved_document = await self.repository.save(saved_document)
        
        return saved_document


class GetDocumentUseCaseImpl(GetDocumentUseCase):
    """Implementation of get document use case"""
    
    def __init__(
        self,
        repository: DocumentRepository,
        access_service: DocumentAccessService
    ):
        self.repository = repository
        self.access_service = access_service
    
    async def execute(self, query: GetDocumentQuery) -> Optional[DocumentEntity]:
        """Execute get document use case"""
        document = await self.repository.get_by_id(query.document_id)
        
        if not document:
            return None
        
        # Check access
        if not self.access_service.can_user_access_document(document, query.user_id):
            raise PermissionError("User does not have access to this document")
        
        return document


class ListDocumentsUseCaseImpl(ListDocumentsUseCase):
    """Implementation of list documents use case"""
    
    def __init__(
        self,
        repository: DocumentRepository,
        access_service: DocumentAccessService
    ):
        self.repository = repository
        self.access_service = access_service
    
    async def execute(self, query: ListDocumentsQuery) -> List[DocumentEntity]:
        """Execute list documents use case"""
        # Get documents from repository
        documents = await self.repository.find_by_user(
            query.user_id,
            limit=query.limit,
            offset=query.offset,
            search=query.search
        )
        
        # Filter by access (in real impl, repository would handle this)
        accessible = [
            doc for doc in documents
            if self.access_service.can_user_access_document(doc, query.user_id)
        ]
        
        return accessible


class GenerateVariantsUseCaseImpl(GenerateVariantsUseCase):
    """Implementation of generate variants use case"""
    
    def __init__(
        self,
        document_repo: DocumentRepository,
        variant_repo: VariantRepository,
        quality_service: VariantQualityService,
        event_bus: EventBus
    ):
        self.document_repo = document_repo
        self.variant_repo = variant_repo
        self.quality_service = quality_service
        self.event_bus = event_bus
    
    async def execute(self, command: GenerateVariantsCommand) -> List[VariantEntity]:
        """Execute generate variants use case"""
        # Verify document exists
        document = await self.document_repo.get_by_id(command.document_id)
        if not document:
            raise ValueError(f"Document {command.document_id} not found")
        
        # Generate variants (simplified - in real impl would use AI)
        variants = []
        start_time = datetime.utcnow()
        
        # In real implementation, this would generate actual variants
        for i in range(command.variant_count):
            variant = VariantEntity(
                id=str(uuid4()),
                document_id=command.document_id,
                variant_type=command.variant_type,
                content=f"Variant {i+1} content",  # Placeholder
                similarity_score=0.85,  # Placeholder
                status="completed",
                created_at=datetime.utcnow()
            )
            
            # Save variant
            saved_variant = await self.variant_repo.save(variant)
            variants.append(saved_variant)
        
        # Filter by quality
        quality_variants = self.quality_service.filter_variants_by_quality(
            variants,
            min_similarity=0.7
        )
        
        # Publish event
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        event = VariantsGeneratedEvent(
            document_id=command.document_id,
            user_id=command.user_id,
            variant_count=len(quality_variants),
            generation_time=generation_time
        )
        await self.event_bus.publish(event)
        
        return quality_variants


class ExtractTopicsUseCaseImpl(ExtractTopicsUseCase):
    """Implementation of extract topics use case"""
    
    def __init__(
        self,
        document_repo: DocumentRepository,
        topic_repo: TopicRepository,
        relevance_service: TopicRelevanceService,
        event_bus: EventBus
    ):
        self.document_repo = document_repo
        self.topic_repo = topic_repo
        self.relevance_service = relevance_service
        self.event_bus = event_bus
    
    async def execute(self, command: ExtractTopicsCommand) -> List[TopicEntity]:
        """Execute extract topics use case"""
        # Verify document exists
        document = await self.document_repo.get_by_id(command.document_id)
        if not document:
            raise ValueError(f"Document {command.document_id} not found")
        
        # In real implementation, this would extract topics using NLP/AI
        # For now, return placeholder topics
        topics = []
        start_time = datetime.utcnow()
        
        # Simplified topic extraction
        placeholder_topics = ["AI", "Machine Learning", "PDF Processing", "Data Analysis"]
        
        for topic_name in placeholder_topics:
            # Calculate relevance (would use actual document content)
            relevance = self.relevance_service.calculate_relevance(
                topic_name,
                "document content placeholder"  # Would use actual content
            )
            
            topic = TopicEntity(
                id=str(uuid4()),
                document_id=command.document_id,
                topic=topic_name,
                relevance_score=float(relevance),
                category="technology",
                created_at=datetime.utcnow()
            )
            
            # Filter by min relevance
            if topic.is_relevant(command.min_relevance):
                saved_topic = await self.topic_repo.save(topic)
                topics.append(saved_topic)
        
        # Rank by relevance
        ranked_topics = self.relevance_service.rank_topics_by_relevance(topics)
        
        # Publish event
        extraction_time = (datetime.utcnow() - start_time).total_seconds()
        event = TopicsExtractedEvent(
            document_id=command.document_id,
            user_id=command.user_id,
            topic_count=len(ranked_topics),
            extraction_time=extraction_time
        )
        await self.event_bus.publish(event)
        
        return ranked_topics






