"""
Enhanced API for AI Document Classifier
=======================================

Advanced FastAPI endpoints with improved functionality, batch processing,
external service integration, and comprehensive analytics.
"""

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import logging
import asyncio
from datetime import datetime, timedelta
import json
import io
import csv
from pathlib import Path

# Import our enhanced components
from .document_classifier_engine import DocumentClassifierEngine, DocumentType, ClassificationResult
from .models.advanced_classifier import AdvancedDocumentClassifier, ClassificationMethod
from .templates.dynamic_template_generator import (
    DynamicTemplateGenerator, TemplateComplexity, DocumentFormat, DynamicTemplate
)
from .utils.batch_processor import BatchProcessor, BatchResult
from .integrations.external_services import ExternalServiceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize enhanced router
router = APIRouter(prefix="/ai-document-classifier/v2", tags=["Enhanced AI Document Classifier"])

# Initialize enhanced components
classifier_engine = DocumentClassifierEngine()
advanced_classifier = AdvancedDocumentClassifier()
template_generator = DynamicTemplateGenerator()
batch_processor = BatchProcessor(classifier_engine)
service_manager = ExternalServiceManager()

# Enhanced Pydantic models
class EnhancedClassificationRequest(BaseModel):
    """Enhanced request model for document classification"""
    query: str = Field(..., description="Text query describing the document to classify")
    use_ai: bool = Field(True, description="Whether to use AI for classification")
    use_advanced: bool = Field(False, description="Whether to use advanced ML classification")
    classification_method: Optional[str] = Field(None, description="Specific classification method to use")
    extract_features: bool = Field(True, description="Whether to extract detailed features")
    external_service: Optional[str] = Field(None, description="External service to use for classification")
    language: Optional[str] = Field("en", description="Language of the query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for classification")

class EnhancedClassificationResponse(BaseModel):
    """Enhanced response model for document classification"""
    document_type: str
    confidence: float
    method_used: str
    keywords: List[str]
    reasoning: str
    features: Optional[Dict[str, Any]] = None
    processing_time: float
    alternative_types: List[Dict[str, Any]] = field(default_factory=list)
    template_suggestions: List[str] = field(default_factory=list)
    external_service_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class BatchClassificationRequest(BaseModel):
    """Request model for batch classification"""
    queries: List[str] = Field(..., description="List of text queries to classify")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    use_cache: bool = Field(True, description="Whether to use cached results")
    use_advanced: bool = Field(False, description="Whether to use advanced classification")
    max_workers: Optional[int] = Field(None, description="Maximum number of worker threads")
    progress_callback_url: Optional[str] = Field(None, description="URL for progress callbacks")

class TemplateGenerationRequest(BaseModel):
    """Request model for template generation"""
    document_type: str = Field(..., description="Type of document")
    complexity: str = Field("intermediate", description="Template complexity level")
    style_preset: str = Field("business", description="Style preset to use")
    custom_requirements: Optional[Dict[str, Any]] = Field(None, description="Custom requirements")
    genre: Optional[str] = Field(None, description="Document genre")
    industry: Optional[str] = Field(None, description="Target industry")
    language: Optional[str] = Field("en", description="Template language")

class ServiceConfigurationRequest(BaseModel):
    """Request model for service configuration"""
    service_name: str = Field(..., description="Name of the service")
    api_key: Optional[str] = Field(None, description="API key for the service")
    enabled: bool = Field(True, description="Whether to enable the service")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")

# Enhanced endpoints
@router.post("/classify/enhanced", response_model=EnhancedClassificationResponse)
async def enhanced_classify_document(request: EnhancedClassificationRequest):
    """
    Enhanced document classification with advanced features
    
    Args:
        request: EnhancedClassificationRequest with classification parameters
        
    Returns:
        EnhancedClassificationResponse with detailed results
    """
    try:
        logger.info(f"Enhanced classification request: {request.query[:100]}...")
        
        start_time = datetime.now()
        
        # Choose classification method
        if request.external_service:
            # Use external service
            external_result = service_manager.classify_with_external_ai(
                request.query, 
                request.external_service
            )
            
            if external_result.success:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return EnhancedClassificationResponse(
                    document_type=external_result.data.get("document_type", "unknown"),
                    confidence=external_result.data.get("confidence", 0.0),
                    method_used=f"external_{request.external_service}",
                    keywords=external_result.data.get("keywords", []),
                    reasoning=external_result.data.get("reasoning", "External service classification"),
                    processing_time=processing_time,
                    external_service_used=request.external_service,
                    template_suggestions=[]  # Would need to be generated separately
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"External service error: {external_result.error}"
                )
        
        elif request.use_advanced:
            # Use advanced ML classification
            result = advanced_classifier.classify_with_ml(
                request.query,
                ClassificationMethod(request.classification_method) if request.classification_method 
                else ClassificationMethod.ENSEMBLE
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedClassificationResponse(
                document_type=result.document_type,
                confidence=result.confidence,
                method_used=result.method_used.value,
                keywords=result.keywords,
                reasoning=result.reasoning,
                features=result.features.__dict__ if request.extract_features else None,
                processing_time=processing_time,
                alternative_types=[
                    {"type": alt_type, "confidence": conf} 
                    for alt_type, conf in result.alternative_types
                ],
                template_suggestions=template_generator._get_template_suggestions(
                    DocumentType(result.document_type)
                )
            )
        
        else:
            # Use standard classification
            result = classifier_engine.classify_document(
                query=request.query,
                use_ai=request.use_ai
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedClassificationResponse(
                document_type=result.document_type.value,
                confidence=result.confidence,
                method_used="standard",
                keywords=result.keywords,
                reasoning=result.reasoning,
                processing_time=processing_time,
                template_suggestions=result.template_suggestions
            )
        
    except Exception as e:
        logger.error(f"Enhanced classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/classify/batch", response_model=Dict[str, Any])
async def batch_classify_documents(request: BatchClassificationRequest):
    """
    Batch classification of multiple documents
    
    Args:
        request: BatchClassificationRequest with batch parameters
        
    Returns:
        Batch processing result with analytics
    """
    try:
        logger.info(f"Batch classification request: {len(request.queries)} queries")
        
        # Process batch
        batch_result = batch_processor.process_batch(
            queries=request.queries,
            batch_id=request.batch_id,
            use_cache=request.use_cache
        )
        
        return {
            "batch_id": batch_result.batch_id,
            "total_jobs": batch_result.total_jobs,
            "completed_jobs": batch_result.completed_jobs,
            "failed_jobs": batch_result.failed_jobs,
            "processing_time": batch_result.processing_time,
            "success_rate": batch_result.completed_jobs / batch_result.total_jobs,
            "analytics": batch_result.analytics,
            "results": batch_result.results[:10],  # Limit results for response size
            "errors": batch_result.errors[:5] if batch_result.errors else []
        }
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str = Path(..., description="Batch identifier")):
    """
    Get status of a batch processing job
    
    Args:
        batch_id: Batch identifier
        
    Returns:
        Batch status information
    """
    try:
        status = batch_processor.get_batch_status(batch_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")

@router.post("/templates/generate")
async def generate_dynamic_template(request: TemplateGenerationRequest):
    """
    Generate a dynamic template based on requirements
    
    Args:
        request: TemplateGenerationRequest with generation parameters
        
    Returns:
        Generated dynamic template
    """
    try:
        logger.info(f"Template generation request: {request.document_type}")
        
        # Generate template
        template = template_generator.generate_template(
            document_type=request.document_type,
            complexity=TemplateComplexity(request.complexity),
            style_preset=request.style_preset,
            custom_requirements=request.custom_requirements,
            genre=request.genre,
            industry=request.industry
        )
        
        # Export template as JSON
        template_json = template_generator.export_template(template, DocumentFormat.JSON)
        
        return {
            "template": json.loads(template_json),
            "generated_at": datetime.now().isoformat(),
            "template_id": f"{template.document_type}_{template.complexity.value}_{int(datetime.now().timestamp())}"
        }
        
    except Exception as e:
        logger.error(f"Template generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Template generation failed: {str(e)}")

@router.get("/templates/export/{template_id}")
async def export_template(
    template_id: str = Path(..., description="Template identifier"),
    format: str = Query("json", description="Export format: json, yaml, markdown, html")
):
    """
    Export a template in specified format
    
    Args:
        template_id: Template identifier
        format: Export format
        
    Returns:
        Exported template
    """
    try:
        # This would need to be implemented with template storage
        # For now, return a placeholder response
        return {
            "error": "Template export not fully implemented",
            "template_id": template_id,
            "format": format
        }
        
    except Exception as e:
        logger.error(f"Template export error: {e}")
        raise HTTPException(status_code=500, detail=f"Template export failed: {str(e)}")

@router.post("/services/configure")
async def configure_external_service(request: ServiceConfigurationRequest):
    """
    Configure an external service
    
    Args:
        request: ServiceConfigurationRequest with service configuration
        
    Returns:
        Service configuration result
    """
    try:
        logger.info(f"Service configuration request: {request.service_name}")
        
        if request.enabled and request.api_key:
            service_manager.enable_service(request.service_name, request.api_key)
        elif not request.enabled:
            service_manager.disable_service(request.service_name)
        
        # Test service connection
        test_result = service_manager.test_service_connection(request.service_name)
        
        return {
            "service_name": request.service_name,
            "enabled": request.enabled,
            "connection_test": {
                "success": test_result.success,
                "error": test_result.error,
                "processing_time": test_result.processing_time
            },
            "configured_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Service configuration error: {e}")
        raise HTTPException(status_code=500, detail=f"Service configuration failed: {str(e)}")

@router.get("/services/status")
async def get_services_status():
    """
    Get status of all external services
    
    Returns:
        Status of all configured services
    """
    try:
        status = service_manager.get_all_services_status()
        
        return {
            "services": status,
            "total_services": len(status),
            "enabled_services": len([s for s in status.values() if s.get("enabled", False)]),
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting services status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get services status: {str(e)}")

@router.get("/analytics")
async def get_analytics():
    """
    Get comprehensive analytics and performance metrics
    
    Returns:
        Analytics and performance data
    """
    try:
        # Get batch processor analytics
        batch_analytics = batch_processor.get_analytics()
        
        # Get advanced classifier performance
        classifier_performance = advanced_classifier.get_model_performance()
        
        # Get service status
        services_status = service_manager.get_all_services_status()
        
        return {
            "batch_processing": batch_analytics,
            "classifier_performance": classifier_performance,
            "services": {
                "total": len(services_status),
                "enabled": len([s for s in services_status.values() if s.get("enabled", False)]),
                "status": services_status
            },
            "system": {
                "uptime": "N/A",  # Would need to track system uptime
                "version": "2.0.0",
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/analytics/export")
async def export_analytics(format: str = Query("json", description="Export format: json, csv")):
    """
    Export analytics data in specified format
    
    Args:
        format: Export format (json, csv)
        
    Returns:
        Exported analytics data
    """
    try:
        analytics_data = batch_processor.export_analytics(format)
        
        if format == "csv":
            # Return as downloadable file
            output = io.StringIO()
            output.write(analytics_data)
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        else:
            return JSONResponse(content=json.loads(analytics_data))
        
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export analytics: {str(e)}")

@router.post("/models/train")
async def train_models(background_tasks: BackgroundTasks):
    """
    Train machine learning models in the background
    
    Args:
        background_tasks: FastAPI background tasks
        
    Returns:
        Training initiation response
    """
    try:
        # Start training in background
        background_tasks.add_task(advanced_classifier.train_models)
        
        return {
            "message": "Model training started in background",
            "started_at": datetime.now().isoformat(),
            "status": "training"
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model training: {str(e)}")

@router.get("/models/performance")
async def get_model_performance():
    """
    Get machine learning model performance metrics
    
    Returns:
        Model performance data
    """
    try:
        performance = advanced_classifier.get_model_performance()
        
        return {
            "performance": performance,
            "models_available": ML_AVAILABLE,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached results and data
    
    Returns:
        Cache clearing result
    """
    try:
        batch_processor.clear_cache()
        
        return {
            "message": "Cache cleared successfully",
            "cleared_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/health/enhanced")
async def enhanced_health_check():
    """
    Enhanced health check with detailed system status
    
    Returns:
        Comprehensive health status
    """
    try:
        # Check all components
        classifier_health = "healthy"
        advanced_classifier_health = "healthy"
        batch_processor_health = "healthy"
        services_health = "healthy"
        
        # Test basic classification
        try:
            test_result = classifier_engine.classify_document("test query", use_ai=False)
            if not test_result:
                classifier_health = "degraded"
        except:
            classifier_health = "unhealthy"
        
        # Test advanced classifier
        try:
            performance = advanced_classifier.get_model_performance()
            if not performance:
                advanced_classifier_health = "degraded"
        except:
            advanced_classifier_health = "unhealthy"
        
        # Test batch processor
        try:
            analytics = batch_processor.get_analytics()
            if not analytics:
                batch_processor_health = "degraded"
        except:
            batch_processor_health = "unhealthy"
        
        # Test services
        try:
            services_status = service_manager.get_all_services_status()
            enabled_services = [s for s in services_status.values() if s.get("enabled", False)]
            if not enabled_services:
                services_health = "degraded"
        except:
            services_health = "unhealthy"
        
        overall_health = "healthy"
        if any(status == "unhealthy" for status in [classifier_health, advanced_classifier_health, batch_processor_health]):
            overall_health = "unhealthy"
        elif any(status == "degraded" for status in [classifier_health, advanced_classifier_health, batch_processor_health, services_health]):
            overall_health = "degraded"
        
        return {
            "status": overall_health,
            "version": "2.0.0",
            "components": {
                "classifier_engine": classifier_health,
                "advanced_classifier": advanced_classifier_health,
                "batch_processor": batch_processor_health,
                "external_services": services_health
            },
            "timestamp": datetime.now().isoformat(),
            "uptime": "N/A"  # Would need to track actual uptime
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Error handlers
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"}
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Import ML_AVAILABLE from advanced_classifier
try:
    from .models.advanced_classifier import ML_AVAILABLE
except ImportError:
    ML_AVAILABLE = False



























