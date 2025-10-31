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
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from ..config.openrouter_config import OpenRouterConfig
from ..config.bul_config import BULConfig
from ..utils.document_processor import DocumentProcessor

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

class TruthGPTBulkProcessor:
    """
    TruthGPT-inspired bulk document processor that generates multiple documents
    continuously with a single request.
    """
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.openrouter_config = OpenRouterConfig()
        self.document_processor = DocumentProcessor(self.config)
        
        # Processing state
        self.is_running = False
        self.active_requests: Dict[str, BulkDocumentRequest] = {}
        self.active_tasks: Dict[str, DocumentGenerationTask] = {}
        self.completed_tasks: Dict[str, DocumentGenerationTask] = {}
        self.results: Dict[str, BulkGenerationResult] = {}
        
        # Task queue for continuous processing
        self.task_queue: List[DocumentGenerationTask] = []
        self.processing_stats = {
            "total_requests": 0,
            "total_documents_generated": 0,
            "total_documents_failed": 0,
            "average_processing_time": 0.0,
            "active_requests": 0,
            "queued_tasks": 0
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.processing.max_concurrent_tasks)
        
        # Callbacks
        self.on_document_generated: Optional[Callable] = None
        self.on_request_completed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Initialize LangChain
        self._setup_langchain()
        
        logger.info("TruthGPT Bulk Processor initialized")
    
    def _setup_langchain(self):
        """Setup LangChain with OpenRouter configuration."""
        if not self.openrouter_config.is_configured():
            raise ValueError("OpenRouter API key not configured")
        
        # Create ChatOpenAI instance with OpenRouter
        self.llm = ChatOpenAI(
            model=self.openrouter_config.default_model,
            openai_api_key=self.openrouter_config.api_key,
            openai_api_base=self.openrouter_config.base_url,
            temperature=0.7,
            max_tokens=4096,
            headers=self.openrouter_config.get_headers()
        )
        
        # Create output parser
        self.output_parser = StrOutputParser()
        
        # Create prompt templates
        self._create_prompt_templates()
        
        logger.info(f"LangChain configured with model: {self.openrouter_config.default_model}")
    
    def _create_prompt_templates(self):
        """Create prompt templates for different document types."""
        
        # TruthGPT-inspired system prompt
        self.truthgpt_system_prompt = """You are TruthGPT, an advanced AI system specialized in generating comprehensive, accurate, and detailed business documents. Your mission is to create high-quality content that provides real value to businesses.

Core Principles:
1. TRUTH: Always provide accurate, factual, and verifiable information
2. COMPREHENSIVENESS: Create detailed, thorough documents that cover all aspects
3. PRACTICALITY: Focus on actionable insights and real-world applications
4. QUALITY: Maintain high standards in content structure and presentation
5. CONTINUITY: Generate content that flows naturally and builds upon itself

Document Generation Guidelines:
- Create professional, well-structured documents
- Include relevant examples, case studies, and best practices
- Provide actionable recommendations and next steps
- Ensure content is current and relevant to the business context
- Use clear, professional language suitable for business use
- Structure content with logical flow and clear headings
- Include practical implementation guidance

You are generating documents as part of a continuous bulk generation process. Each document should be comprehensive and valuable on its own while contributing to the overall knowledge base."""

        # Document generation prompt
        self.document_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.truthgpt_system_prompt),
            HumanMessage(content="""Business Area: {business_area}
Document Type: {document_type}
Query/Topic: {query}
Context: {context}

Generate a comprehensive {document_type} document for the {business_area} area based on the query: "{query}"

Requirements:
1. Create a detailed, professional document
2. Include practical examples and case studies
3. Provide actionable recommendations
4. Structure content with clear headings and sections
5. Ensure the document is comprehensive and valuable
6. Focus on real-world applicability
7. Include implementation guidance where relevant

Make this document a valuable resource that businesses can immediately use and implement.""")
        ])
        
        # Document variation prompt for continuous generation
        self.variation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.truthgpt_system_prompt),
            HumanMessage(content="""Business Area: {business_area}
Document Type: {document_type}
Base Query: {query}
Variation Number: {variation_number}
Previous Content: {previous_content}

Create a variation of the {document_type} document for {business_area}. This should be a different perspective or approach to the same topic while maintaining quality and comprehensiveness.

Requirements:
1. Provide a fresh perspective on the topic
2. Include different examples and case studies
3. Offer alternative approaches or methodologies
4. Maintain the same high quality standards
5. Ensure the content is distinct but complementary
6. Focus on different aspects or applications

This is variation {variation_number} of the document generation process.""")
        ])
    
    async def start_continuous_processing(self):
        """Start the continuous processing loop."""
        if self.is_running:
            logger.warning("Continuous processing is already running")
            return
        
        self.is_running = True
        logger.info("Starting TruthGPT continuous processing mode...")
        
        try:
            while self.is_running:
                await self._process_next_tasks()
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            if self.on_error:
                await self._safe_callback(self.on_error, e)
        finally:
            self.is_running = False
            logger.info("Continuous processing stopped")
    
    async def _process_next_tasks(self):
        """Process the next batch of tasks."""
        if not self.task_queue:
            return
        
        # Process multiple tasks concurrently
        batch_size = min(self.config.processing.max_concurrent_tasks, len(self.task_queue))
        tasks_to_process = self.task_queue[:batch_size]
        
        # Remove tasks from queue
        self.task_queue = self.task_queue[batch_size:]
        
        # Process tasks concurrently
        processing_tasks = []
        for task in tasks_to_process:
            processing_tasks.append(self._process_single_task(task))
        
        if processing_tasks:
            await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    async def _process_single_task(self, task: DocumentGenerationTask):
        """Process a single document generation task."""
        try:
            task.status = "processing"
            self.active_tasks[task.id] = task
            
            # Generate document content
            content = await self._generate_document_content(task)
            
            if content:
                task.content = content
                task.status = "completed"
                task.completed_at = datetime.now()
                
                # Process and save document
                processed_doc = await self.document_processor.process_document(
                    content=content,
                    document_type=task.document_type,
                    business_area=task.business_area,
                    query=task.query
                )
                
                # Store completed task
                self.completed_tasks[task.id] = task
                
                # Update stats
                self.processing_stats["total_documents_generated"] += 1
                
                # Callback for document generated
                if self.on_document_generated:
                    await self._safe_callback(self.on_document_generated, task, processed_doc)
                
                logger.info(f"Document generated: {task.id} - {task.document_type}")
            else:
                raise Exception("Failed to generate document content")
                
        except Exception as e:
            logger.error(f"Task failed: {task.id} - {e}")
            task.status = "failed"
            task.error = str(e)
            task.retry_count += 1
            
            # Retry if under limit
            if task.retry_count < task.max_retries:
                self.task_queue.append(task)
                logger.info(f"Retrying task: {task.id} (attempt {task.retry_count + 1})")
            else:
                self.processing_stats["total_documents_failed"] += 1
                logger.error(f"Task failed permanently: {task.id}")
            
            if self.on_error:
                await self._safe_callback(self.on_error, task, e)
        
        finally:
            # Remove from active tasks
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _generate_document_content(self, task: DocumentGenerationTask) -> Optional[str]:
        """Generate document content using LangChain."""
        try:
            # Create the processing chain
            chain = self.document_prompt | self.llm | self.output_parser
            
            # Generate content
            content = await chain.ainvoke({
                "business_area": task.business_area,
                "document_type": task.document_type,
                "query": task.query,
                "context": f"Task ID: {task.id}, Priority: {task.priority}, Created: {task.created_at}"
            })
            
            return content
            
        except Exception as e:
            logger.error(f"Content generation failed for task {task.id}: {e}")
            return None
    
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
        """
        request_id = str(uuid.uuid4())
        
        # Create bulk request
        request = BulkDocumentRequest(
            id=request_id,
            query=query,
            document_types=document_types,
            business_areas=business_areas,
            max_documents=max_documents,
            continuous_mode=continuous_mode,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store request
        self.active_requests[request_id] = request
        self.processing_stats["total_requests"] += 1
        self.processing_stats["active_requests"] += 1
        
        # Create initial tasks
        await self._create_initial_tasks(request)
        
        # Start continuous processing if not running
        if not self.is_running:
            asyncio.create_task(self.start_continuous_processing())
        
        logger.info(f"Bulk request submitted: {request_id} - {max_documents} documents requested")
        
        return request_id
    
    async def _create_initial_tasks(self, request: BulkDocumentRequest):
        """Create initial tasks for a bulk request."""
        tasks_created = 0
        
        # Create tasks for each combination of document type and business area
        for doc_type in request.document_types:
            for business_area in request.business_areas:
                if tasks_created >= request.max_documents:
                    break
                
                task = DocumentGenerationTask(
                    id=str(uuid.uuid4()),
                    request_id=request.id,
                    document_type=doc_type,
                    business_area=business_area,
                    query=request.query,
                    priority=request.priority
                )
                
                self.task_queue.append(task)
                tasks_created += 1
        
        # If in continuous mode and we haven't reached max_documents, create more tasks
        if request.continuous_mode and tasks_created < request.max_documents:
            await self._create_additional_tasks(request, tasks_created)
        
        self.processing_stats["queued_tasks"] = len(self.task_queue)
    
    async def _create_additional_tasks(self, request: BulkDocumentRequest, current_count: int):
        """Create additional tasks for continuous generation."""
        remaining = request.max_documents - current_count
        
        # Create variations of existing combinations
        for i in range(remaining):
            doc_type = request.document_types[i % len(request.document_types)]
            business_area = request.business_areas[i % len(request.business_areas)]
            variation_number = (i // len(request.document_types)) + 1
            
            task = DocumentGenerationTask(
                id=str(uuid.uuid4()),
                request_id=request.id,
                document_type=doc_type,
                business_area=business_area,
                query=request.query,
                priority=request.priority
            )
            
            # Add variation metadata
            task.metadata = {"variation_number": variation_number}
            
            self.task_queue.append(task)
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a bulk request."""
        if request_id not in self.active_requests:
            return None
        
        request = self.active_requests[request_id]
        
        # Count completed and failed tasks for this request
        completed_count = sum(1 for task in self.completed_tasks.values() 
                            if task.request_id == request_id and task.status == "completed")
        failed_count = sum(1 for task in self.completed_tasks.values() 
                         if task.request_id == request_id and task.status == "failed")
        active_count = sum(1 for task in self.active_tasks.values() 
                         if task.request_id == request_id)
        queued_count = sum(1 for task in self.task_queue 
                         if task.request_id == request_id)
        
        return {
            "request_id": request_id,
            "status": "active" if request_id in self.active_requests else "completed",
            "query": request.query,
            "max_documents": request.max_documents,
            "documents_generated": completed_count,
            "documents_failed": failed_count,
            "active_tasks": active_count,
            "queued_tasks": queued_count,
            "progress_percentage": (completed_count / request.max_documents) * 100,
            "created_at": request.created_at.isoformat(),
            "continuous_mode": request.continuous_mode
        }
    
    async def get_request_documents(self, request_id: str) -> List[Dict[str, Any]]:
        """Get all generated documents for a request."""
        documents = []
        
        for task in self.completed_tasks.values():
            if task.request_id == request_id and task.status == "completed" and task.content:
                documents.append({
                    "task_id": task.id,
                    "document_type": task.document_type,
                    "business_area": task.business_area,
                    "content": task.content,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                })
        
        return documents
    
    async def stop_request(self, request_id: str) -> bool:
        """Stop a specific bulk request."""
        if request_id not in self.active_requests:
            return False
        
        # Remove request
        del self.active_requests[request_id]
        self.processing_stats["active_requests"] -= 1
        
        # Remove queued tasks for this request
        self.task_queue = [task for task in self.task_queue if task.request_id != request_id]
        self.processing_stats["queued_tasks"] = len(self.task_queue)
        
        logger.info(f"Request stopped: {request_id}")
        return True
    
    def stop_processing(self):
        """Stop all processing."""
        self.is_running = False
        logger.info("Processing stop requested")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.processing_stats,
            "active_requests": len(self.active_requests),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "is_running": self.is_running
        }
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute a callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    def set_document_callback(self, callback: Callable):
        """Set callback for when documents are generated."""
        self.on_document_generated = callback
    
    def set_request_callback(self, callback: Callable):
        """Set callback for when requests are completed."""
        self.on_request_completed = callback
    
    def set_error_callback(self, callback: Callable):
        """Set callback for when errors occur."""
        self.on_error = callback

# Global processor instance
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



























