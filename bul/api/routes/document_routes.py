"""
Advanced Document Routes - Functional Programming Approach
========================================================

Implementation of document generation routes following FastAPI best practices:
- Functional programming patterns
- RORO pattern (Receive an Object, Return an Object)
- Advanced error handling
- Performance optimization
- Comprehensive validation
"""

from typing import Dict, List, Optional, Any, Union, Callable
from functools import partial
import asyncio
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Request, Response, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt

from ..advanced_patterns import (
    RequestContext,
    ResponseContext,
    APIError,
    ValidationError,
    BusinessLogicError,
    monitor_performance,
    cache_result,
    retry_on_failure,
    create_response_context,
    extract_request_context,
    validate_required_fields,
    validate_field_types,
    serialize_response,
    batch_process,
    optimize_json_serialization
)

# Advanced Pydantic Models with RORO pattern
class DocumentRequest(BaseModel):
    """Document generation request following RORO pattern"""
    query: str = Field(..., min_length=10, max_length=2000, description="Business query or requirement")
    business_area: Optional[str] = Field(None, description="Business area")
    document_type: Optional[str] = Field(None, description="Type of document to generate")
    company_name: Optional[str] = Field(None, max_length=100, description="Company name")
    industry: Optional[str] = Field(None, max_length=100, description="Industry sector")
    company_size: Optional[str] = Field(None, description="Company size")
    target_audience: Optional[str] = Field(None, max_length=200, description="Target audience")
    language: str = Field("es", description="Language for document generation")
    format: str = Field("markdown", description="Output format")
    style: str = Field("professional", description="Document style")
    priority: str = Field("normal", description="Processing priority")
    cache_ttl: int = Field(3600, ge=60, le=86400, description="Cache TTL in seconds")
    include_metadata: bool = Field(True, description="Include metadata in response")
    
    @validator('language')
    def validate_language(cls, v):
        allowed_languages = ['es', 'en', 'pt', 'fr', 'de', 'it', 'ru', 'zh', 'ja']
        if v not in allowed_languages:
            raise ValueError(f'Language must be one of: {allowed_languages}')
        return v
    
    @validator('business_area')
    def validate_business_area(cls, v):
        if v is not None:
            allowed_areas = ['marketing', 'sales', 'operations', 'hr', 'finance', 'legal', 'technical', 'content', 'strategy', 'customer_service']
            if v.lower() not in allowed_areas:
                raise ValueError(f'Business area must be one of: {allowed_areas}')
        return v
    
    @validator('document_type')
    def validate_document_type(cls, v):
        if v is not None:
            allowed_types = ['business_plan', 'marketing_strategy', 'sales_proposal', 'operational_manual', 'hr_policy', 'financial_report', 'legal_contract', 'technical_specification', 'content_strategy', 'strategic_plan', 'customer_service_guide']
            if v.lower() not in allowed_types:
                raise ValueError(f'Document type must be one of: {allowed_types}')
        return v

class DocumentResponse(BaseModel):
    """Document generation response following RORO pattern"""
    id: str
    request_id: str
    content: str
    title: str
    summary: str
    business_area: str
    document_type: str
    word_count: int
    processing_time: float
    confidence_score: float
    created_at: datetime
    agent_used: Optional[str] = None
    format: str
    style: str
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    readability_score: Optional[float] = None

class BatchDocumentRequest(BaseModel):
    """Batch document generation request"""
    requests: List[DocumentRequest] = Field(..., max_items=10, description="List of document requests")
    parallel: bool = Field(True, description="Process requests in parallel")
    priority: str = Field("normal", description="Overall batch priority")
    max_concurrent: int = Field(5, ge=1, le=10, description="Maximum concurrent requests")
    
    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError('At least one request is required')
        return v

# Advanced Functional Utilities
def create_document_processor(engine: Any, agent_manager: Any) -> Callable:
    """Create document processor function with dependency injection"""
    async def process_document(request: DocumentRequest, context: RequestContext) -> DocumentResponse:
        """Process single document with advanced error handling"""
        try:
            # Validate request
            validate_required_fields(request.dict(), ['query'])
            validate_field_types(request.dict(), {
                'query': str,
                'language': str,
                'format': str
            })
            
            # Process document
            start_time = time.time()
            
            # Get best agent
            best_agent = await agent_manager.get_best_agent(request)
            
            # Generate document
            response = await engine.generate_document(request)
            
            processing_time = time.time() - start_time
            
            # Calculate quality metrics
            quality_score = _calculate_quality_score(response.content)
            readability_score = _calculate_readability_score(response.content)
            
            return DocumentResponse(
                id=response.id,
                request_id=context.request_id,
                content=response.content,
                title=response.title,
                summary=response.summary,
                business_area=response.business_area,
                document_type=response.document_type,
                word_count=response.word_count,
                processing_time=processing_time,
                confidence_score=response.confidence_score,
                created_at=response.created_at,
                agent_used=best_agent.name if best_agent else "Default Agent",
                format=request.format,
                style=request.style,
                metadata=response.metadata,
                quality_score=quality_score,
                readability_score=readability_score
            )
            
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except BusinessLogicError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error during document generation")
    
    return process_document

def create_batch_processor(processor: Callable) -> Callable:
    """Create batch processor function"""
    async def process_batch(request: BatchDocumentRequest, context: RequestContext) -> List[DocumentResponse]:
        """Process batch documents with advanced error handling"""
        try:
            if request.parallel:
                # Process in parallel with concurrency limit
                semaphore = asyncio.Semaphore(request.max_concurrent)
                
                async def process_single(doc_request: DocumentRequest) -> DocumentResponse:
                    async with semaphore:
                        return await processor(doc_request, context)
                
                tasks = [process_single(req) for req in request.requests]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append({
                            "error": str(result),
                            "request_index": i,
                            "success": False
                        })
                    else:
                        processed_results.append(result)
                
                return processed_results
            else:
                # Process sequentially
                results = []
                for i, doc_request in enumerate(request.requests):
                    try:
                        result = await processor(doc_request, context)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "error": str(e),
                            "request_index": i,
                            "success": False
                        })
                
                return results
                
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error during batch processing")
    
    return process_batch

# Advanced Quality Metrics
def _calculate_quality_score(content: str) -> float:
    """Calculate quality score for content using advanced metrics"""
    if not content:
        return 0.0
    
    # Word count factor
    word_count = len(content.split())
    word_score = min(1.0, word_count / 1000)  # Normalize to 1000 words
    
    # Sentence structure factor
    sentences = [s for s in content.split('.') if s.strip()]
    if not sentences:
        return 0.0
    
    avg_sentence_length = word_count / len(sentences)
    structure_score = 1.0 if 10 <= avg_sentence_length <= 25 else 0.5
    
    # Content depth factor
    depth_score = min(1.0, len(content) / 5000)  # Normalize to 5000 characters
    
    # Overall quality score
    quality_score = (word_score * 0.4 + structure_score * 0.3 + depth_score * 0.3)
    return round(quality_score, 2)

def _calculate_readability_score(content: str) -> float:
    """Calculate readability score using advanced algorithms"""
    if not content:
        return 0.0
    
    words = content.split()
    sentences = [s for s in content.split('.') if s.strip()]
    
    if not words or not sentences:
        return 0.0
    
    # Average words per sentence
    avg_words_per_sentence = len(words) / len(sentences)
    
    # Syllable estimation (simplified)
    total_syllables = sum(_estimate_syllables(word) for word in words)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch Reading Ease Score (simplified)
    readability_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
    
    # Normalize to 0-1 scale
    normalized_score = max(0.0, min(1.0, readability_score / 100))
    return round(normalized_score, 2)

def _estimate_syllables(word: str) -> int:
    """Estimate syllables in a word"""
    word = word.lower()
    vowels = 'aeiouy'
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    # Handle silent 'e'
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count)

# Advanced Caching Functions
@cache_result(ttl=1800, key_func=lambda req, ctx: f"doc:{hash(req.query + req.business_area + req.document_type)}")
async def get_cached_document(request: DocumentRequest, context: RequestContext) -> Optional[DocumentResponse]:
    """Get cached document if available"""
    # This would integrate with actual cache system
    return None

async def cache_document_response(response: DocumentResponse, ttl: int = 1800) -> None:
    """Cache document response"""
    # This would integrate with actual cache system
    pass

# Advanced Validation Functions
def validate_document_request(request: DocumentRequest) -> None:
    """Advanced validation for document request"""
    # Required field validation
    validate_required_fields(request.dict(), ['query'])
    
    # Business logic validation
    if request.business_area and request.document_type:
        # Validate business area and document type compatibility
        compatible_types = {
            'marketing': ['marketing_strategy', 'content_strategy'],
            'sales': ['sales_proposal', 'business_plan'],
            'operations': ['operational_manual', 'technical_specification'],
            'hr': ['hr_policy', 'operational_manual'],
            'finance': ['financial_report', 'business_plan'],
            'legal': ['legal_contract', 'hr_policy'],
            'technical': ['technical_specification', 'operational_manual'],
            'content': ['content_strategy', 'marketing_strategy'],
            'strategy': ['business_plan', 'strategic_plan'],
            'customer_service': ['customer_service_guide', 'operational_manual']
        }
        
        area_types = compatible_types.get(request.business_area.lower(), [])
        if request.document_type.lower() not in area_types:
            raise BusinessLogicError(
                f"Document type '{request.document_type}' is not compatible with business area '{request.business_area}'"
            )

def validate_batch_request(request: BatchDocumentRequest) -> None:
    """Advanced validation for batch request"""
    # Validate individual requests
    for i, doc_request in enumerate(request.requests):
        try:
            validate_document_request(doc_request)
        except (ValidationError, BusinessLogicError) as e:
            raise ValidationError(f"Request {i}: {str(e)}")
    
    # Validate batch constraints
    if request.max_concurrent > len(request.requests):
        raise ValidationError("max_concurrent cannot be greater than number of requests")

# Advanced Error Handling Functions
def handle_document_generation_error(error: Exception, context: RequestContext) -> HTTPException:
    """Handle document generation errors with context"""
    if isinstance(error, ValidationError):
        return HTTPException(
            status_code=400,
            detail={
                "error": "Validation Error",
                "message": str(error),
                "request_id": context.request_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    elif isinstance(error, BusinessLogicError):
        return HTTPException(
            status_code=422,
            detail={
                "error": "Business Logic Error",
                "message": str(error),
                "request_id": context.request_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    else:
        return HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred during document generation",
                "request_id": context.request_id,
                "timestamp": datetime.now().isoformat()
            }
        )

# Advanced Response Functions
def create_success_response(data: Any, context: RequestContext, metadata: Optional[Dict[str, Any]] = None) -> ResponseContext:
    """Create success response following RORO pattern"""
    return create_response_context(
        data=data,
        request_id=context.request_id,
        success=True,
        metadata=metadata or {}
    )

def create_error_response(error: str, context: RequestContext, status_code: int = 500) -> ResponseContext:
    """Create error response following RORO pattern"""
    return create_response_context(
        data=None,
        request_id=context.request_id,
        success=False,
        error=error,
        metadata={"status_code": status_code}
    )

# Advanced Performance Functions
@monitor_performance("document_generation")
async def generate_document_with_metrics(request: DocumentRequest, context: RequestContext, processor: Callable) -> DocumentResponse:
    """Generate document with performance monitoring"""
    start_time = time.time()
    
    try:
        # Check cache first
        cached_response = await get_cached_document(request, context)
        if cached_response:
            return cached_response
        
        # Process document
        response = await processor(request, context)
        
        # Cache response
        await cache_document_response(response, request.cache_ttl)
        
        return response
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        # This would integrate with metrics system
        raise

@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
async def generate_document_with_retry(request: DocumentRequest, context: RequestContext, processor: Callable) -> DocumentResponse:
    """Generate document with retry logic"""
    return await processor(request, context)

# Advanced Route Handlers
async def handle_single_document_generation(
    request: DocumentRequest,
    context: RequestContext,
    processor: Callable,
    background_tasks: BackgroundTasks
) -> ResponseContext:
    """Handle single document generation with advanced features"""
    try:
        # Validate request
        validate_document_request(request)
        
        # Generate document with retry and metrics
        response = await generate_document_with_retry(request, context, processor)
        
        # Background tasks
        background_tasks.add_task(
            _log_document_generation,
            request,
            response,
            context
        )
        
        return create_success_response(response, context)
        
    except Exception as e:
        error_response = handle_document_generation_error(e, context)
        return create_error_response(str(e), context, error_response.status_code)

async def handle_batch_document_generation(
    request: BatchDocumentRequest,
    context: RequestContext,
    processor: Callable,
    background_tasks: BackgroundTasks
) -> ResponseContext:
    """Handle batch document generation with advanced features"""
    try:
        # Validate batch request
        validate_batch_request(request)
        
        # Create batch processor
        batch_processor = create_batch_processor(processor)
        
        # Process batch
        responses = await batch_processor(request, context)
        
        # Background tasks
        background_tasks.add_task(
            _log_batch_generation,
            request,
            responses,
            context
        )
        
        return create_success_response(responses, context)
        
    except Exception as e:
        error_response = handle_document_generation_error(e, context)
        return create_error_response(str(e), context, error_response.status_code)

# Background Task Functions
async def _log_document_generation(request: DocumentRequest, response: DocumentResponse, context: RequestContext) -> None:
    """Log document generation for analytics"""
    # This would integrate with logging system
    pass

async def _log_batch_generation(request: BatchDocumentRequest, responses: List[Any], context: RequestContext) -> None:
    """Log batch generation for analytics"""
    # This would integrate with logging system
    pass

# Advanced Utility Functions
def create_document_router(engine: Any, agent_manager: Any) -> APIRouter:
    """Create document router with dependency injection"""
    router = APIRouter(prefix="/documents", tags=["documents"])
    
    # Create processors
    document_processor = create_document_processor(engine, agent_manager)
    
    @router.post("/generate", response_model=ResponseContext[DocumentResponse])
    async def generate_document(
        request: DocumentRequest,
        background_tasks: BackgroundTasks,
        req: Request
    ):
        """Generate single document with advanced features"""
        context = extract_request_context(req)
        return await handle_single_document_generation(
            request, context, document_processor, background_tasks
        )
    
    @router.post("/generate/batch", response_model=ResponseContext[List[DocumentResponse]])
    async def generate_documents_batch(
        request: BatchDocumentRequest,
        background_tasks: BackgroundTasks,
        req: Request
    ):
        """Generate multiple documents in batch"""
        context = extract_request_context(req)
        return await handle_batch_document_generation(
            request, context, document_processor, background_tasks
        )
    
    return router

# Export functions
__all__ = [
    "DocumentRequest",
    "DocumentResponse", 
    "BatchDocumentRequest",
    "create_document_processor",
    "create_batch_processor",
    "validate_document_request",
    "validate_batch_request",
    "handle_document_generation_error",
    "create_success_response",
    "create_error_response",
    "generate_document_with_metrics",
    "generate_document_with_retry",
    "handle_single_document_generation",
    "handle_batch_document_generation",
    "create_document_router"
]












