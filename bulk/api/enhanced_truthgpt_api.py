"""
Enhanced TruthGPT Bulk Document Generation API
=============================================

API mejorada con características avanzadas:
- Caching inteligente
- Optimización de prompts
- Balanceo de carga de modelos
- Métricas de calidad
- Monitoreo avanzado
- Recuperación de errores mejorada
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import asyncio
import json
import logging
from datetime import datetime
import uuid

from ..core.enhanced_truthgpt_processor import (
    EnhancedTruthGPTProcessor, 
    get_global_enhanced_processor,
    EnhancedBulkDocumentRequest,
    EnhancedDocumentTask
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v2/truthgpt", tags=["Enhanced TruthGPT"])

# Enhanced Pydantic models
class EnhancedBulkRequestModel(BaseModel):
    """Enhanced request model for bulk document generation."""
    query: str = Field(..., description="Main query/topic for document generation", min_length=1)
    document_types: List[str] = Field(..., description="List of document types to generate", min_items=1)
    business_areas: List[str] = Field(..., description="List of business areas to focus on", min_items=1)
    max_documents: int = Field(100, description="Maximum number of documents to generate", ge=1, le=1000)
    continuous_mode: bool = Field(True, description="Whether to continue generating until max_documents")
    priority: int = Field(1, description="Priority level (1-5, where 1 is highest)", ge=1, le=5)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    # Enhanced features
    enable_caching: bool = Field(True, description="Enable intelligent caching")
    enable_optimization: bool = Field(True, description="Enable prompt optimization")
    quality_threshold: float = Field(0.85, description="Minimum quality score threshold", ge=0.0, le=1.0)
    enable_variations: bool = Field(True, description="Enable document variations")
    max_variations: int = Field(5, description="Maximum number of variations", ge=1, le=20)
    enable_cross_referencing: bool = Field(True, description="Enable cross-referencing between documents")
    enable_evolution: bool = Field(True, description="Enable content evolution")
    target_audience: Optional[str] = Field(None, description="Target audience for documents")
    language: str = Field("es", description="Language for document generation")
    tone: str = Field("professional", description="Tone for document generation")
    
    @validator('document_types')
    def validate_document_types(cls, v):
        valid_types = [
            "business_plan", "marketing_strategy", "sales_presentation", 
            "financial_analysis", "operational_manual", "hr_policy",
            "technical_documentation", "content_strategy", "legal_document",
            "customer_service_guide", "product_description", "proposal",
            "report", "white_paper", "case_study", "best_practices_guide",
            "implementation_plan", "risk_assessment", "quality_manual",
            "innovation_framework", "compliance_guide", "training_material"
        ]
        for doc_type in v:
            if doc_type not in valid_types:
                raise ValueError(f"Invalid document type: {doc_type}. Valid types: {valid_types}")
        return v
    
    @validator('business_areas')
    def validate_business_areas(cls, v):
        valid_areas = [
            "marketing", "sales", "operations", "hr", "finance", 
            "legal", "technical", "content", "strategy", "customer_service",
            "product_development", "business_development", "management",
            "innovation", "quality_assurance", "risk_management"
        ]
        for area in v:
            if area not in valid_areas:
                raise ValueError(f"Invalid business area: {area}. Valid areas: {valid_areas}")
        return v
    
    @validator('tone')
    def validate_tone(cls, v):
        valid_tones = ["professional", "casual", "formal", "friendly", "authoritative", "conversational"]
        if v not in valid_tones:
            raise ValueError(f"Invalid tone: {v}. Valid tones: {valid_tones}")
        return v

class EnhancedBulkResponseModel(BaseModel):
    """Enhanced response model for bulk document generation request."""
    request_id: str
    status: str
    message: str
    estimated_completion_time: Optional[str] = None
    enhanced_features: Dict[str, Any] = None
    quality_metrics: Dict[str, Any] = None

class EnhancedRequestStatusModel(BaseModel):
    """Enhanced model for request status response."""
    request_id: str
    status: str
    query: str
    max_documents: int
    documents_generated: int
    documents_failed: int
    active_tasks: int
    queued_tasks: int
    progress_percentage: float
    created_at: str
    continuous_mode: bool
    
    # Enhanced metrics
    average_quality_score: float
    cache_hit_rate: float
    optimization_enabled: bool
    variations_enabled: bool
    target_audience: Optional[str]
    language: str
    tone: str
    processing_efficiency: float
    estimated_remaining_time: Optional[str] = None

class EnhancedDocumentModel(BaseModel):
    """Enhanced model for individual document."""
    task_id: str
    document_type: str
    business_area: str
    content: str
    created_at: str
    completed_at: Optional[str] = None
    
    # Enhanced features
    quality_score: float
    processing_time: float
    model_used: Optional[str]
    tokens_used: int
    cost_estimate: float
    cache_hit: bool
    optimization_applied: bool
    variations_generated: int
    cross_references: List[str]

class EnhancedProcessingStatsModel(BaseModel):
    """Enhanced model for processing statistics."""
    total_requests: int
    total_documents_generated: int
    total_documents_failed: int
    average_processing_time: float
    average_quality_score: float
    cache_hit_rate: float
    optimization_success_rate: float
    model_usage_stats: Dict[str, int]
    error_analysis: Dict[str, int]
    active_requests: int
    queued_tasks: int
    completed_tasks: int
    is_running: bool
    cache_enabled: bool
    models_available: int
    performance_trends: List[float]

class QualityAnalysisModel(BaseModel):
    """Model for quality analysis."""
    overall_quality: float
    quality_breakdown: Dict[str, float]
    recommendations: List[str]
    improvement_suggestions: List[str]

# Dependency to get enhanced processor
def get_enhanced_processor() -> EnhancedTruthGPTProcessor:
    """Get the global enhanced TruthGPT processor."""
    return get_global_enhanced_processor()

@router.post("/generate", response_model=EnhancedBulkResponseModel)
async def submit_enhanced_bulk_request(
    request: EnhancedBulkRequestModel,
    background_tasks: BackgroundTasks,
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Submit an enhanced bulk document generation request with advanced features.
    
    This endpoint provides:
    - Intelligent caching
    - Prompt optimization
    - Quality assessment
    - Model load balancing
    - Advanced error recovery
    """
    try:
        # Submit the enhanced bulk request
        request_id = await processor.submit_enhanced_bulk_request(
            query=request.query,
            document_types=request.document_types,
            business_areas=request.business_areas,
            max_documents=request.max_documents,
            continuous_mode=request.continuous_mode,
            priority=request.priority,
            metadata=request.metadata,
            enable_caching=request.enable_caching,
            enable_optimization=request.enable_optimization,
            quality_threshold=request.quality_threshold,
            enable_variations=request.enable_variations,
            max_variations=request.max_variations,
            enable_cross_referencing=request.enable_cross_referencing,
            enable_evolution=request.enable_evolution,
            target_audience=request.target_audience,
            language=request.language,
            tone=request.tone
        )
        
        # Calculate estimated completion time based on enhanced metrics
        stats = processor.get_enhanced_processing_stats()
        avg_time = stats.get("average_processing_time", 30)
        estimated_time = request.max_documents * avg_time
        estimated_completion = datetime.now().timestamp() + estimated_time
        
        # Enhanced features summary
        enhanced_features = {
            "caching_enabled": request.enable_caching,
            "optimization_enabled": request.enable_optimization,
            "quality_threshold": request.quality_threshold,
            "variations_enabled": request.enable_variations,
            "max_variations": request.max_variations,
            "cross_referencing_enabled": request.enable_cross_referencing,
            "evolution_enabled": request.enable_evolution,
            "target_audience": request.target_audience,
            "language": request.language,
            "tone": request.tone
        }
        
        # Quality metrics
        quality_metrics = {
            "target_quality": request.quality_threshold,
            "current_system_quality": stats.get("average_quality_score", 0.0),
            "optimization_rate": stats.get("optimization_success_rate", 0.0)
        }
        
        return EnhancedBulkResponseModel(
            request_id=request_id,
            status="accepted",
            message=f"Enhanced bulk generation request accepted. {request.max_documents} documents will be generated with advanced features.",
            estimated_completion_time=datetime.fromtimestamp(estimated_completion).isoformat(),
            enhanced_features=enhanced_features,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Error submitting enhanced bulk request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit enhanced request: {str(e)}")

@router.get("/status/{request_id}", response_model=EnhancedRequestStatusModel)
async def get_enhanced_request_status(
    request_id: str = Path(..., description="Request ID to check status for"),
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Get the enhanced status of a bulk generation request with detailed metrics.
    """
    try:
        status = await processor.get_enhanced_request_status(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Request not found")
        
        # Calculate processing efficiency
        if status["documents_generated"] > 0:
            efficiency = status["documents_generated"] / (status["documents_generated"] + status["documents_failed"])
        else:
            efficiency = 0.0
        
        # Estimate remaining time
        remaining_docs = status["max_documents"] - status["documents_generated"]
        if remaining_docs > 0 and status["active_tasks"] > 0:
            stats = processor.get_enhanced_processing_stats()
            avg_time = stats.get("average_processing_time", 30)
            estimated_remaining = remaining_docs * avg_time
            estimated_remaining_time = datetime.fromtimestamp(
                datetime.now().timestamp() + estimated_remaining
            ).isoformat()
        else:
            estimated_remaining_time = None
        
        return EnhancedRequestStatusModel(
            **status,
            processing_efficiency=efficiency,
            estimated_remaining_time=estimated_remaining_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced request status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced status: {str(e)}")

@router.get("/documents/{request_id}", response_model=List[EnhancedDocumentModel])
async def get_enhanced_request_documents(
    request_id: str = Path(..., description="Request ID to get documents for"),
    min_quality: float = Query(0.0, description="Minimum quality score filter", ge=0.0, le=1.0),
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Get all generated documents for a specific request with enhanced metadata.
    """
    try:
        # Check if request exists
        status = await processor.get_enhanced_request_status(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Request not found")
        
        # Get documents with quality filter
        documents = []
        for task in processor.completed_tasks.values():
            if (task.request_id == request_id and 
                task.status == "completed" and 
                task.content and 
                task.quality_score >= min_quality):
                
                documents.append(EnhancedDocumentModel(
                    task_id=task.id,
                    document_type=task.document_type,
                    business_area=task.business_area,
                    content=task.content,
                    created_at=task.created_at.isoformat(),
                    completed_at=task.completed_at.isoformat() if task.completed_at else None,
                    quality_score=task.quality_score,
                    processing_time=task.processing_time,
                    model_used=task.model_used,
                    tokens_used=task.tokens_used,
                    cost_estimate=task.cost_estimate,
                    cache_hit=task.cache_hit,
                    optimization_applied=task.optimization_applied,
                    variations_generated=task.variations_generated,
                    cross_references=task.cross_references
                ))
        
        return documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced request documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced documents: {str(e)}")

@router.get("/documents/{request_id}/stream")
async def stream_enhanced_documents(
    request_id: str = Path(..., description="Request ID to stream documents for"),
    include_quality_metrics: bool = Query(True, description="Include quality metrics in stream"),
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Stream documents as they are generated with enhanced metadata.
    """
    try:
        # Check if request exists
        status = await processor.get_enhanced_request_status(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Request not found")
        
        async def generate_enhanced_document_stream():
            """Generate a stream of enhanced documents as they are completed."""
            seen_documents = set()
            
            while True:
                # Get current documents
                current_documents = []
                for task in processor.completed_tasks.values():
                    if (task.request_id == request_id and 
                        task.status == "completed" and 
                        task.content and 
                        task.id not in seen_documents):
                        
                        document_data = {
                            "type": "document",
                            "task_id": task.id,
                            "document_type": task.document_type,
                            "business_area": task.business_area,
                            "content": task.content,
                            "created_at": task.created_at.isoformat(),
                            "completed_at": task.completed_at.isoformat() if task.completed_at else None
                        }
                        
                        if include_quality_metrics:
                            document_data.update({
                                "quality_score": task.quality_score,
                                "processing_time": task.processing_time,
                                "model_used": task.model_used,
                                "cache_hit": task.cache_hit,
                                "optimization_applied": task.optimization_applied
                            })
                        
                        current_documents.append(document_data)
                        seen_documents.add(task.id)
                
                # Send new documents
                for doc in current_documents:
                    yield f"data: {json.dumps(doc)}\n\n"
                
                # Check if request is complete
                current_status = await processor.get_enhanced_request_status(request_id)
                if current_status and current_status["documents_generated"] >= current_status["max_documents"]:
                    yield f"data: {json.dumps({'type': 'complete', 'message': 'All documents generated', 'final_stats': current_status})}\n\n"
                    break
                
                # Wait before next check
                await asyncio.sleep(2)
        
        return StreamingResponse(
            generate_enhanced_document_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming enhanced documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stream enhanced documents: {str(e)}")

@router.get("/quality-analysis/{request_id}", response_model=QualityAnalysisModel)
async def get_quality_analysis(
    request_id: str = Path(..., description="Request ID to analyze quality for"),
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Get detailed quality analysis for a request.
    """
    try:
        # Check if request exists
        status = await processor.get_enhanced_request_status(request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Request not found")
        
        # Analyze quality of completed documents
        quality_scores = []
        quality_breakdown = {
            "excellent": 0,  # >= 0.9
            "good": 0,       # >= 0.8
            "acceptable": 0, # >= 0.7
            "poor": 0        # < 0.7
        }
        
        for task in processor.completed_tasks.values():
            if task.request_id == request_id and task.status == "completed":
                quality_scores.append(task.quality_score)
                
                if task.quality_score >= 0.9:
                    quality_breakdown["excellent"] += 1
                elif task.quality_score >= 0.8:
                    quality_breakdown["good"] += 1
                elif task.quality_score >= 0.7:
                    quality_breakdown["acceptable"] += 1
                else:
                    quality_breakdown["poor"] += 1
        
        # Calculate overall quality
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Generate recommendations
        recommendations = []
        if overall_quality < 0.8:
            recommendations.append("Consider enabling prompt optimization for better quality")
        if quality_breakdown["poor"] > 0:
            recommendations.append("Some documents have low quality scores - review and retry")
        if status["cache_hit_rate"] < 0.3:
            recommendations.append("Low cache hit rate - consider enabling caching")
        
        # Improvement suggestions
        improvement_suggestions = []
        if overall_quality < 0.9:
            improvement_suggestions.append("Increase quality threshold to 0.9 for premium documents")
        if status["processing_efficiency"] < 0.9:
            improvement_suggestions.append("Review error patterns to improve processing efficiency")
        
        return QualityAnalysisModel(
            overall_quality=overall_quality,
            quality_breakdown=quality_breakdown,
            recommendations=recommendations,
            improvement_suggestions=improvement_suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quality analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get quality analysis: {str(e)}")

@router.get("/stats", response_model=EnhancedProcessingStatsModel)
async def get_enhanced_processing_stats(
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Get enhanced processing statistics with detailed metrics.
    """
    try:
        stats = processor.get_enhanced_processing_stats()
        return EnhancedProcessingStatsModel(**stats)
        
    except Exception as e:
        logger.error(f"Error getting enhanced processing stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced stats: {str(e)}")

@router.post("/optimize-prompt")
async def optimize_prompt(
    prompt: str = Query(..., description="Prompt to optimize"),
    document_type: str = Query(..., description="Document type for optimization"),
    business_area: str = Query(..., description="Business area for optimization"),
    target_audience: Optional[str] = Query(None, description="Target audience"),
    language: str = Query("es", description="Language"),
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Optimize a prompt using the advanced optimization system.
    """
    try:
        optimized_prompt = processor.prompt_optimizer.optimize_prompt(
            prompt, document_type, business_area, target_audience, language
        )
        
        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized_prompt,
            "document_type": document_type,
            "business_area": business_area,
            "target_audience": target_audience,
            "language": language,
            "optimization_applied": True
        }
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize prompt: {str(e)}")

@router.get("/cache-stats")
async def get_cache_statistics(
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Get cache statistics and performance metrics.
    """
    try:
        stats = processor.get_enhanced_processing_stats()
        
        return {
            "cache_enabled": stats["cache_enabled"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "total_requests": stats["total_requests"],
            "cache_performance": {
                "hit_rate_percentage": stats["cache_hit_rate"] * 100,
                "estimated_savings": stats["cache_hit_rate"] * stats["total_documents_generated"] * 0.3,  # 30% time savings
                "cache_efficiency": "high" if stats["cache_hit_rate"] > 0.5 else "medium" if stats["cache_hit_rate"] > 0.2 else "low"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")

@router.post("/start-enhanced-processing")
async def start_enhanced_processing(
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Start the enhanced continuous processing system.
    """
    try:
        if not processor.is_running:
            asyncio.create_task(processor.start_enhanced_processing())
            return {"message": "Enhanced continuous processing started"}
        else:
            return {"message": "Enhanced continuous processing is already running"}
            
    except Exception as e:
        logger.error(f"Error starting enhanced processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start enhanced processing: {str(e)}")

@router.post("/stop-enhanced-processing")
async def stop_enhanced_processing(
    processor: EnhancedTruthGPTProcessor = Depends(get_enhanced_processor)
):
    """
    Stop the enhanced continuous processing system.
    """
    try:
        processor.stop_processing()
        return {"message": "Enhanced continuous processing stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping enhanced processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop enhanced processing: {str(e)}")

@router.get("/health")
async def enhanced_health_check():
    """
    Enhanced health check endpoint with detailed system status.
    """
    try:
        processor = get_enhanced_processor()
        stats = processor.get_enhanced_processing_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Enhanced TruthGPT Bulk Document Generation API",
            "version": "2.0.0",
            "system_status": {
                "is_running": stats["is_running"],
                "models_available": stats["models_available"],
                "cache_enabled": stats["cache_enabled"],
                "active_requests": stats["active_requests"],
                "queued_tasks": stats["queued_tasks"]
            },
            "performance_metrics": {
                "average_quality_score": stats["average_quality_score"],
                "cache_hit_rate": stats["cache_hit_rate"],
                "optimization_success_rate": stats["optimization_success_rate"],
                "processing_efficiency": 1.0 - (stats["total_documents_failed"] / max(1, stats["total_documents_generated"] + stats["total_documents_failed"]))
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Additional utility endpoints
@router.get("/document-types")
async def get_available_document_types():
    """
    Get list of available document types with descriptions.
    """
    return {
        "document_types": [
            {"type": "business_plan", "description": "Plan de negocio completo"},
            {"type": "marketing_strategy", "description": "Estrategia de marketing integral"},
            {"type": "sales_presentation", "description": "Presentación de ventas profesional"},
            {"type": "financial_analysis", "description": "Análisis financiero detallado"},
            {"type": "operational_manual", "description": "Manual operativo completo"},
            {"type": "hr_policy", "description": "Política de recursos humanos"},
            {"type": "technical_documentation", "description": "Documentación técnica detallada"},
            {"type": "content_strategy", "description": "Estrategia de contenido"},
            {"type": "legal_document", "description": "Documento legal"},
            {"type": "customer_service_guide", "description": "Guía de atención al cliente"},
            {"type": "product_description", "description": "Descripción de producto"},
            {"type": "proposal", "description": "Propuesta comercial"},
            {"type": "report", "description": "Reporte ejecutivo"},
            {"type": "white_paper", "description": "Libro blanco técnico"},
            {"type": "case_study", "description": "Estudio de caso"},
            {"type": "best_practices_guide", "description": "Guía de mejores prácticas"},
            {"type": "implementation_plan", "description": "Plan de implementación"},
            {"type": "risk_assessment", "description": "Evaluación de riesgos"},
            {"type": "quality_manual", "description": "Manual de calidad"},
            {"type": "innovation_framework", "description": "Marco de innovación"},
            {"type": "compliance_guide", "description": "Guía de cumplimiento"},
            {"type": "training_material", "description": "Material de capacitación"}
        ]
    }

@router.get("/business-areas")
async def get_available_business_areas():
    """
    Get list of available business areas with descriptions.
    """
    return {
        "business_areas": [
            {"area": "marketing", "description": "Marketing y promoción"},
            {"area": "sales", "description": "Ventas y comercialización"},
            {"area": "operations", "description": "Operaciones y procesos"},
            {"area": "hr", "description": "Recursos humanos"},
            {"area": "finance", "description": "Finanzas y contabilidad"},
            {"area": "legal", "description": "Legal y cumplimiento"},
            {"area": "technical", "description": "Tecnología e IT"},
            {"area": "content", "description": "Contenido y comunicación"},
            {"area": "strategy", "description": "Estrategia y planificación"},
            {"area": "customer_service", "description": "Atención al cliente"},
            {"area": "product_development", "description": "Desarrollo de productos"},
            {"area": "business_development", "description": "Desarrollo de negocios"},
            {"area": "management", "description": "Gestión y liderazgo"},
            {"area": "innovation", "description": "Innovación y R&D"},
            {"area": "quality_assurance", "description": "Aseguramiento de calidad"},
            {"area": "risk_management", "description": "Gestión de riesgos"}
        ]
    }



























