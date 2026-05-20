"""
TruthGPT-Inspired Bulk Document Processor
=========================================

A continuous document generation system that creates multiple documents
with a single request without stopping, inspired by TruthGPT architecture.

Features:
- Continuous document generation
- Multiple document types per request
- Asynchronous processing
- Auto-scaling document creation
- Real-time progress tracking
- Error recovery and retry mechanisms
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from langchain_community.callbacks import get_openai_callback

from ..config.openrouter_config import OpenRouterConfig
from ..config.bul_config import BULConfig
from ..utils.document_processor import DocumentProcessor
from .base_processor import BaseBulkProcessor
from .langchain_setup import LangChainSetup
from .task_creator import TaskCreator
from .prompt_templates import PromptTemplates
from .task_error_handler import TaskErrorHandler
from .stats_helper import StatsHelper
from .request_validator import RequestValidator
from .request_submitter import RequestSubmitter
from .request_query_helper import RequestQueryHelper
from .task_queue_helper import TaskQueueHelper
from .processing_loop import ProcessingLoop
from .content_generator import ContentGenerator
from .constants import DEFAULT_LOOP_SLEEP_SECONDS

logger = logging.getLogger(__name__)

@dataclass
class BulkDocumentRequest:
    """Request for bulk document generation."""
    id: str
    query: str
    document_types: List[str]
    business_areas: List[str]
    max_documents: int = 100
    continuous_mode: bool = True
    priority: int = 1
    created_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DocumentGenerationTask:
    """Individual document generation task."""
    id: str
    request_id: str
    document_type: str
    business_area: str
    query: str
    priority: int
    status: str = "pending"  # pending, processing, completed, failed
    content: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class BulkGenerationResult:
    """Result of bulk document generation."""
    request_id: str
    total_documents_requested: int
    documents_generated: int
    documents_failed: int
    processing_time: float
    start_time: datetime
    end_time: Optional[datetime] = None
    documents: List[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.documents is None:
            self.documents = []
        if self.errors is None:
            self.errors = []

class TruthGPTBulkProcessor(BaseBulkProcessor):
    """
    TruthGPT-inspired bulk document processor that generates multiple documents
    continuously with a single request.
    """
    
    def __init__(self, config: Optional[BULConfig] = None):
        super().__init__(config)
        
        self.active_requests: Dict[str, BulkDocumentRequest] = {}
        self.active_tasks: Dict[str, DocumentGenerationTask] = {}
        self.completed_tasks: Dict[str, DocumentGenerationTask] = {}
        self.results: Dict[str, BulkGenerationResult] = {}
        
        self.task_queue: List[DocumentGenerationTask] = []
        self.processing_stats = {
            "total_requests": 0,
            "total_documents_generated": 0,
            "total_documents_failed": 0,
            "average_processing_time": 0.0,
            "active_requests": 0,
            "queued_tasks": 0
        }
        
        self._setup_langchain()
        
        logger.info("TruthGPT Bulk Processor initialized")
    
    def _setup_langchain(self):
        """Setup LangChain with OpenRouter configuration."""
        self._setup_langchain_base()
        self.llm = LangChainSetup.create_llm(self.openrouter_config)
        self._create_prompt_templates()
        
        logger.info(f"LangChain configured with model: {self.openrouter_config.default_model}")
    
    def _create_prompt_templates(self):
        """Create prompt templates for different document types."""
        self.document_prompt = PromptTemplates.create_document_prompt()
        self.variation_prompt = PromptTemplates.create_variation_prompt()
    
    async def start_continuous_processing(self):
        """Start the continuous processing loop."""
        await ProcessingLoop.run_processing_loop(
            processor=self,
            process_func=self._process_next_tasks,
            sleep_seconds=DEFAULT_LOOP_SLEEP_SECONDS,
            loop_name="TruthGPT continuous processing"
        )
    
    async def _process_next_tasks(self):
        """Process the next batch of tasks."""
        if not self.task_queue:
            return
        
        tasks_to_process, self.task_queue = TaskQueueHelper.get_batch_from_queue(
            task_queue=self.task_queue,
            batch_size=self.config.processing.max_concurrent_tasks
        )
        
        if tasks_to_process:
            processing_tasks = [
                self._process_single_task(task) for task in tasks_to_process
            ]
            await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    async def _process_single_task(self, task: DocumentGenerationTask):
        """Process a single document generation task."""
        try:
            task.status = "processing"
            self.active_tasks[task.id] = task
            
            content = await self._generate_document_content(task)
            
            if content:
                task.content = content
                task.status = "completed"
                task.completed_at = datetime.now()
                
                processed_doc = await self.document_processor.process_document(
                    content=content,
                    document_type=task.document_type,
                    business_area=task.business_area,
                    query=task.query
                )
                
                TaskErrorHandler.mark_task_completed(
                    task=task,
                    completed_tasks=self.completed_tasks,
                    processing_stats=self.processing_stats
                )
                
                await self.callbacks.execute_document_callback(task, processed_doc)
                
                logger.info(f"Document generated: {task.id} - {task.document_type}")
            else:
                raise Exception("Failed to generate document content")
                
        except Exception as e:
            logger.error(f"Task failed: {task.id} - {e}")
            
            async def error_callback(t, err):
                await self.callbacks.execute_error_callback(t, err)
            
            should_retry = await TaskErrorHandler.handle_task_error(
                task=task,
                error=e,
                task_queue=self.task_queue,
                max_retries=task.max_retries,
                on_error_callback=error_callback
            )
            
            if not should_retry:
                TaskErrorHandler.mark_task_failed(
                    task=task,
                    completed_tasks=self.completed_tasks,
                    processing_stats=self.processing_stats
                )
        
        finally:
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _generate_document_content(self, task: DocumentGenerationTask) -> Optional[str]:
        """Generate document content using LangChain."""
        context = ContentGenerator.build_task_context(
            task_id=task.id,
            priority=task.priority,
            created_at=str(task.created_at) if task.created_at else None
        )
        
        return await ContentGenerator.generate_content(
            prompt_template=self.document_prompt,
            llm=self.llm,
            output_parser=self.output_parser,
            task_data={
                "business_area": task.business_area,
                "document_type": task.document_type,
                "query": task.query,
                "context": context
            }
        )
    
    async def submit_bulk_request(self, 
                                query: str,
                                document_types: List[str],
                                business_areas: List[str],
                                max_documents: int = 100,
                                continuous_mode: bool = True,
                                priority: int = 1,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a bulk document generation request.
        
        Args:
            query: The main query/topic for document generation
            document_types: List of document types to generate
            business_areas: List of business areas to focus on
            max_documents: Maximum number of documents to generate
            continuous_mode: Whether to continue generating until max_documents
            priority: Priority level (1-5, where 1 is highest)
            metadata: Additional metadata for the request
        
        Returns:
            Request ID for tracking
            
        Raises:
            ValueError: If validation fails
        """
        is_valid, error_msg, request_id = RequestSubmitter.validate_and_prepare_request(
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            priority=priority
        )
        
        if not is_valid:
            raise ValueError(error_msg)
        
        request = RequestSubmitter.create_request_object(
            request_class=BulkDocumentRequest,
            request_id=request_id,
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            continuous_mode=continuous_mode,
            priority=priority,
            metadata=metadata
        )
        
        def update_stats():
            self.processing_stats["total_requests"] += 1
            self.processing_stats["active_requests"] += 1
        
        await RequestSubmitter.register_request_and_start_processing(
            request=request,
            active_requests=self.active_requests,
            stats_updater=update_stats,
            task_creator=self._create_initial_tasks,
            processor=self,
            start_processing_func=self.start_continuous_processing
        )
        
        logger.info(f"Bulk request submitted: {request_id} - {max_documents} documents requested")
        
        return request_id
    
    async def _create_initial_tasks(self, request: BulkDocumentRequest):
        """Create initial tasks for a bulk request."""
        initial_tasks = TaskCreator.create_initial_tasks(
            request_id=request.id,
            query=request.query,
            document_types=request.document_types,
            business_areas=request.business_areas,
            max_documents=request.max_documents,
            priority=request.priority,
            task_class=DocumentGenerationTask
        )
        
        self.task_queue.extend(initial_tasks)
        tasks_created = len(initial_tasks)
        
        if request.continuous_mode and tasks_created < request.max_documents:
            additional_tasks = TaskCreator.create_additional_tasks(
                request_id=request.id,
                query=request.query,
                document_types=request.document_types,
                business_areas=request.business_areas,
                current_count=tasks_created,
                max_documents=request.max_documents,
                priority=request.priority,
                task_class=DocumentGenerationTask
            )
            self.task_queue.extend(additional_tasks)
        
        self.processing_stats["queued_tasks"] = len(self.task_queue)
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a bulk request."""
        request = RequestQueryHelper.find_request(request_id, self.active_requests)
        if not request:
            return None
        
        status = StatsHelper.get_request_status_base(
            request_id=request_id,
            request=request,
            completed_tasks=self.completed_tasks,
            active_tasks=self.active_tasks,
            task_queue=self.task_queue
        )
        
        status.update({
            "continuous_mode": request.continuous_mode
        })
        
        return status
    
    async def get_request_documents(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all generated documents for a request."""
        return RequestQueryHelper.get_request_documents(
            request_id=request_id,
            completed_tasks=self.completed_tasks,
            include_metadata=False
        )
    
    async def stop_request(self, request_id: str) -> bool:
        """Stop a specific bulk request."""
        if request_id not in self.active_requests:
            return False
        
        del self.active_requests[request_id]
        self.processing_stats["active_requests"] -= 1
        
        self.task_queue = TaskQueueHelper.remove_tasks_by_request_id(
            task_queue=self.task_queue,
            request_id=request_id
        )
        self.processing_stats["queued_tasks"] = len(self.task_queue)
        
        logger.info(f"Request stopped: {request_id}")
        return True
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        base_stats = StatsHelper.get_base_stats(
            active_requests=self.active_requests,
            active_tasks=self.active_tasks,
            task_queue=self.task_queue,
            completed_tasks=self.completed_tasks,
            is_running=self.is_running
        )
        
        return {
            **self.processing_stats,
            **base_stats
        }
    
    def set_request_callback(self, callback: Callable):
        """Set callback for when requests are completed."""
        self.callbacks.set_task_callback(callback)

_global_truthgpt_processor: Optional[TruthGPTBulkProcessor] = None

def get_global_truthgpt_processor() -> TruthGPTBulkProcessor:
    """Get the global TruthGPT bulk processor instance."""
    global _global_truthgpt_processor
    if _global_truthgpt_processor is None:
        _global_truthgpt_processor = TruthGPTBulkProcessor()
    return _global_truthgpt_processor

async def start_global_truthgpt_processor():
    """Start the global TruthGPT bulk processor."""
    processor = get_global_truthgpt_processor()
    await processor.start_continuous_processing()

def stop_global_truthgpt_processor():
    """Stop the global TruthGPT bulk processor."""
    processor = get_global_truthgpt_processor()
    processor.stop_processing()





























