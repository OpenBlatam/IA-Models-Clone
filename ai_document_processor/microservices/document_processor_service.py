"""
Document Processor Microservice - Ultra-Modular Service
======================================================

Independent document processor microservice with full modularity.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json

from ..modules.microservices import Microservice, ServiceConfiguration, ServiceStatus
from ..modules.events import EventBus, Event, EventType, get_event_bus
from ..modules.registry import ComponentRegistry, get_component_registry

logger = logging.getLogger(__name__)


class DocumentProcessorService(Microservice):
    """Ultra-modular document processor microservice."""
    
    def __init__(self, configuration: ServiceConfiguration):
        super().__init__(configuration)
        self.processor_components = {}
        self.processing_queue = asyncio.Queue()
        self.processing_workers = []
        self.max_workers = configuration.custom_config.get('max_workers', 4)
        self.event_bus = get_event_bus()
        self.component_registry = get_component_registry()
    
    async def start(self) -> bool:
        """Start the document processor service."""
        try:
            self.status = ServiceStatus.STARTING
            logger.info(f"Starting document processor service: {self.configuration.name}")
            
            # Initialize components
            await self._initialize_components()
            
            # Start processing workers
            await self._start_processing_workers()
            
            # Register with component registry
            await self._register_components()
            
            # Start event handlers
            await self._start_event_handlers()
            
            self.status = ServiceStatus.RUNNING
            self.start_time = datetime.utcnow()
            self.health_score = 1.0
            
            # Publish service started event
            await self.event_bus.publish(Event(
                type=EventType.SERVICE_STARTED,
                source=self.configuration.name,
                data={
                    'service_name': self.configuration.name,
                    'service_type': self.configuration.service_type.value,
                    'start_time': self.start_time.isoformat()
                }
            ))
            
            logger.info(f"Document processor service started: {self.configuration.name}")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to start document processor service: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the document processor service."""
        try:
            self.status = ServiceStatus.STOPPING
            logger.info(f"Stopping document processor service: {self.configuration.name}")
            
            # Stop processing workers
            await self._stop_processing_workers()
            
            # Stop event handlers
            await self._stop_event_handlers()
            
            # Unregister components
            await self._unregister_components()
            
            # Cleanup components
            await self._cleanup_components()
            
            self.status = ServiceStatus.STOPPED
            self.stop_time = datetime.utcnow()
            
            # Publish service stopped event
            await self.event_bus.publish(Event(
                type=EventType.SERVICE_STOPPED,
                source=self.configuration.name,
                data={
                    'service_name': self.configuration.name,
                    'stop_time': self.stop_time.isoformat()
                }
            ))
            
            logger.info(f"Document processor service stopped: {self.configuration.name}")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Failed to stop document processor service: {e}")
            return False
    
    async def health_check(self) -> float:
        """Perform health check."""
        try:
            if self.status != ServiceStatus.RUNNING:
                return 0.0
            
            health_score = 1.0
            
            # Check processing workers
            if len(self.processing_workers) < self.max_workers:
                health_score -= 0.2
            
            # Check component health
            for component_name, component in self.processor_components.items():
                if hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_score = min(health_score, component_health)
            
            # Check queue size
            queue_size = self.processing_queue.qsize()
            if queue_size > 1000:  # Queue too full
                health_score -= 0.3
            
            self.health_score = max(0.0, health_score)
            return self.health_score
            
        except Exception as e:
            self.health_score = 0.0
            self.last_error = str(e)
            self.error_count += 1
            logger.error(f"Health check failed for document processor service: {e}")
            return 0.0
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document processing request."""
        try:
            start_time = datetime.utcnow()
            
            # Validate request
            if not self._validate_request(request):
                return {
                    'status': 'error',
                    'error': 'invalid_request',
                    'message': 'Invalid request format'
                }
            
            # Create processing task
            task_id = str(uuid.uuid4())
            processing_task = {
                'task_id': task_id,
                'request': request,
                'created_at': start_time,
                'status': 'queued'
            }
            
            # Add to processing queue
            await self.processing_queue.put(processing_task)
            
            # Publish processing started event
            await self.event_bus.publish(Event(
                type=EventType.PROCESSING_STARTED,
                source=self.configuration.name,
                data={
                    'task_id': task_id,
                    'document_id': request.get('document_id'),
                    'start_time': start_time.isoformat()
                }
            ))
            
            # Record metrics
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            self.response_times.append(response_time)
            self.request_count += 1
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            return {
                'status': 'accepted',
                'task_id': task_id,
                'message': 'Processing task queued successfully'
            }
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Request processing failed: {e}")
            return {
                'status': 'error',
                'error': 'processing_failed',
                'message': str(e)
            }
    
    async def _initialize_components(self):
        """Initialize processor components."""
        try:
            # Initialize text extractor
            self.processor_components['text_extractor'] = await self._create_text_extractor()
            
            # Initialize AI classifier
            self.processor_components['ai_classifier'] = await self._create_ai_classifier()
            
            # Initialize document transformer
            self.processor_components['document_transformer'] = await self._create_document_transformer()
            
            # Initialize validation service
            self.processor_components['validator'] = await self._create_validator()
            
            logger.info("Document processor components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _create_text_extractor(self):
        """Create text extractor component."""
        # In a real implementation, this would create actual components
        return {
            'name': 'text_extractor',
            'health_check': lambda: 1.0,
            'extract_text': self._extract_text_impl
        }
    
    async def _create_ai_classifier(self):
        """Create AI classifier component."""
        return {
            'name': 'ai_classifier',
            'health_check': lambda: 1.0,
            'classify_document': self._classify_document_impl
        }
    
    async def _create_document_transformer(self):
        """Create document transformer component."""
        return {
            'name': 'document_transformer',
            'health_check': lambda: 1.0,
            'transform_document': self._transform_document_impl
        }
    
    async def _create_validator(self):
        """Create validator component."""
        return {
            'name': 'validator',
            'health_check': lambda: 1.0,
            'validate_document': self._validate_document_impl
        }
    
    async def _start_processing_workers(self):
        """Start processing workers."""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._processing_worker(f"worker-{i}"))
            self.processing_workers.append(worker)
        
        logger.info(f"Started {self.max_workers} processing workers")
    
    async def _stop_processing_workers(self):
        """Stop processing workers."""
        for worker in self.processing_workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.processing_workers:
            await asyncio.gather(*self.processing_workers, return_exceptions=True)
        
        self.processing_workers.clear()
        logger.info("Processing workers stopped")
    
    async def _processing_worker(self, worker_name: str):
        """Processing worker coroutine."""
        logger.info(f"Processing worker {worker_name} started")
        
        while True:
            try:
                # Get task from queue
                task = await self.processing_queue.get()
                
                # Process task
                await self._process_task(task, worker_name)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Processing worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Processing worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _process_task(self, task: Dict[str, Any], worker_name: str):
        """Process a single task."""
        try:
            task_id = task['task_id']
            request = task['request']
            
            logger.info(f"Processing task {task_id} with {worker_name}")
            
            # Update task status
            task['status'] = 'processing'
            task['worker'] = worker_name
            task['processing_started'] = datetime.utcnow()
            
            # Process document
            result = await self._process_document(request)
            
            # Update task status
            task['status'] = 'completed'
            task['result'] = result
            task['completed_at'] = datetime.utcnow()
            
            # Publish processing completed event
            await self.event_bus.publish(Event(
                type=EventType.PROCESSING_COMPLETED,
                source=self.configuration.name,
                data={
                    'task_id': task_id,
                    'document_id': request.get('document_id'),
                    'result': result,
                    'worker': worker_name,
                    'completed_at': task['completed_at'].isoformat()
                }
            ))
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Update task status
            task['status'] = 'failed'
            task['error'] = str(e)
            task['failed_at'] = datetime.utcnow()
            
            # Publish processing failed event
            await self.event_bus.publish(Event(
                type=EventType.PROCESSING_FAILED,
                source=self.configuration.name,
                data={
                    'task_id': task['task_id'],
                    'document_id': request.get('document_id'),
                    'error': str(e),
                    'worker': worker_name,
                    'failed_at': task['failed_at'].isoformat()
                }
            ))
            
            logger.error(f"Task {task['task_id']} failed: {e}")
    
    async def _process_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document through all stages."""
        try:
            document_id = request.get('document_id')
            document_data = request.get('document_data', {})
            
            # Stage 1: Validate document
            validator = self.processor_components['validator']
            is_valid = await validator['validate_document'](document_data)
            if not is_valid:
                raise ValueError("Document validation failed")
            
            # Stage 2: Extract text
            text_extractor = self.processor_components['text_extractor']
            extracted_text = await text_extractor['extract_text'](document_data)
            
            # Stage 3: Classify document
            ai_classifier = self.processor_components['ai_classifier']
            document_type = await ai_classifier['classify_document'](extracted_text)
            
            # Stage 4: Transform document
            document_transformer = self.processor_components['document_transformer']
            transformed_content = await document_transformer['transform_document'](
                extracted_text, document_type
            )
            
            return {
                'document_id': document_id,
                'extracted_text': extracted_text,
                'document_type': document_type,
                'transformed_content': transformed_content,
                'processing_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise
    
    async def _register_components(self):
        """Register components with the registry."""
        try:
            for component_name, component in self.processor_components.items():
                await self.component_registry.register_component(
                    name=component_name,
                    component_type=component['name'],
                    service_type=self.configuration.service_type,
                    instance=component,
                    metadata={
                        'name': component['name'],
                        'version': '1.0.0',
                        'description': f'{component_name} component',
                        'author': 'AI Document Processor Team'
                    }
                )
            
            logger.info("Components registered with registry")
            
        except Exception as e:
            logger.error(f"Failed to register components: {e}")
            raise
    
    async def _unregister_components(self):
        """Unregister components from the registry."""
        try:
            for component_name in self.processor_components.keys():
                # Find and unregister component
                components = await self.component_registry.get_components_by_type(component_name)
                for component in components:
                    await self.component_registry.unregister_component(component.id)
            
            logger.info("Components unregistered from registry")
            
        except Exception as e:
            logger.error(f"Failed to unregister components: {e}")
    
    async def _start_event_handlers(self):
        """Start event handlers."""
        # Event handlers would be registered here
        pass
    
    async def _stop_event_handlers(self):
        """Stop event handlers."""
        # Event handlers would be unregistered here
        pass
    
    async def _cleanup_components(self):
        """Cleanup components."""
        self.processor_components.clear()
        logger.info("Components cleaned up")
    
    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate processing request."""
        required_fields = ['document_id', 'document_data']
        return all(field in request for field in required_fields)
    
    # Component implementation methods (simplified)
    async def _extract_text_impl(self, document_data: Dict[str, Any]) -> str:
        """Extract text from document (simplified implementation)."""
        return document_data.get('content', 'Sample extracted text')
    
    async def _classify_document_impl(self, text: str) -> str:
        """Classify document type (simplified implementation)."""
        return 'document'  # Simplified classification
    
    async def _transform_document_impl(self, text: str, doc_type: str) -> str:
        """Transform document content (simplified implementation)."""
        return f"Transformed {doc_type}: {text}"
    
    async def _validate_document_impl(self, document_data: Dict[str, Any]) -> bool:
        """Validate document (simplified implementation)."""
        return 'content' in document_data

















