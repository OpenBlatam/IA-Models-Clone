"""
Infrastructure Layer - Event Bus Implementation
In-memory event bus for domain events
"""

from typing import Dict, List, Type
import asyncio
import logging

from ..domain.events import DomainEvent, EventHandler, EventBus

logger = logging.getLogger(__name__)


class InMemoryEventBus(EventBus):
    """In-memory event bus implementation"""
    
    def __init__(self):
        self._handlers: Dict[Type[DomainEvent], List[EventHandler]] = {}
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event to all subscribers"""
        event_type = type(event)
        
        if event_type not in self._handlers:
            logger.debug(f"No handlers registered for {event_type.__name__}")
            return
        
        handlers = self._handlers[event_type]
        logger.info(f"Publishing {event_type.__name__} to {len(handlers)} handlers")
        
        # Execute all handlers
        tasks = [handler.handle(event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def subscribe(
        self,
        event_type: Type[DomainEvent],
        handler: EventHandler
    ) -> None:
        """Subscribe handler to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        logger.info(f"Subscribed {handler.__class__.__name__} to {event_type.__name__}")


# Event handlers implementations
class DocumentEventHandler(EventHandler):
    """Handle document-related events"""
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event"""
        logger.info(f"Handling event: {event.__class__.__name__}")
        
        if isinstance(event, DocumentUploadedEvent):
            await self._on_document_uploaded(event)
        elif isinstance(event, DocumentProcessedEvent):
            await self._on_document_processed(event)
        elif isinstance(event, DocumentDeletedEvent):
            await self._on_document_deleted(event)
    
    async def _on_document_uploaded(self, event: DocumentUploadedEvent) -> None:
        """Handle document uploaded"""
        logger.info(f"Document uploaded: {event.document_id}")
        # In real impl, would update analytics, send notifications, etc.
    
    async def _on_document_processed(self, event: DocumentProcessedEvent) -> None:
        """Handle document processed"""
        logger.info(f"Document processed: {event.document_id}")
        # In real impl, would trigger next steps, update status, etc.
    
    async def _on_document_deleted(self, event: DocumentDeletedEvent) -> None:
        """Handle document deleted"""
        logger.info(f"Document deleted: {event.document_id}")
        # In real impl, would cleanup resources, update analytics, etc.


class VariantEventHandler(EventHandler):
    """Handle variant-related events"""
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event"""
        if isinstance(event, VariantsGeneratedEvent):
            await self._on_variants_generated(event)
    
    async def _on_variants_generated(self, event: VariantsGeneratedEvent) -> None:
        """Handle variants generated"""
        logger.info(f"Variants generated: {event.document_id}, count: {event.variant_count}")
        # In real impl, would notify user, update analytics, etc.


class TopicEventHandler(EventHandler):
    """Handle topic-related events"""
    
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event"""
        if isinstance(event, TopicsExtractedEvent):
            await self._on_topics_extracted(event)
    
    async def _on_topics_extracted(self, event: TopicsExtractedEvent) -> None:
        """Handle topics extracted"""
        logger.info(f"Topics extracted: {event.document_id}, count: {event.topic_count}")
        # In real impl, would update document metadata, notify user, etc.






