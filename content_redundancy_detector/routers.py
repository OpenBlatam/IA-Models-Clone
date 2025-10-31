"""
Route handlers for the content redundancy detector API
"""

import logging
import time
import base64
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, WebSocket
from fastapi.responses import JSONResponse

from types import (
    ContentInput, SimilarityInput, BatchAnalysisInput, TopicExtractionInput,
    PlagiarismDetectionInput, SummaryInput, AnalysisResult, 
    SimilarityResult, QualityResult, SentimentResult, LanguageResult,
    TopicResult, SemanticSimilarityResult, PlagiarismResult, EntityResult,
    SummaryResult, ReadabilityResult, ComprehensiveAnalysisResult,
    BatchAnalysisResult, ErrorResponse, HealthResponse, StatsResponse
)
from services import (
    analyze_content, detect_similarity, assess_quality,
    get_system_stats, get_health_status, analyze_sentiment,
    detect_language, extract_topics, calculate_semantic_similarity,
    detect_plagiarism, extract_entities, generate_summary,
    analyze_readability_advanced, comprehensive_analysis, batch_analyze_content
)
from metrics import get_system_metrics, get_endpoint_metrics, get_health_metrics
from cache import get_cache_stats, clear_cache
from batch_processor import (
    create_batch_request, create_batch_job, process_batch_async,
    get_batch_status, get_all_batches, cancel_batch
)
try:
    from webhooks import (
        WebhookEndpoint, WebhookEvent, register_webhook_endpoint,
        unregister_webhook_endpoint, get_webhook_endpoints, get_webhook_stats
    )
except ImportError:
    # Fallback if webhooks module not available
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("webhooks module not available, using fallbacks")
    
    class WebhookEndpoint:
        pass
    class WebhookEvent:
        pass
    def register_webhook_endpoint(*args, **kwargs):
        pass
    def unregister_webhook_endpoint(*args, **kwargs):
        return False
    def get_webhook_endpoints(*args, **kwargs):
        return []
    def get_webhook_stats(*args, **kwargs):
        return {"status": "disabled"}
from export import (
    create_export, process_export, get_export, get_all_exports,
    ExportFormat
)
from analytics import (
    generate_performance_report, generate_content_insights_report,
    generate_similarity_insights_report, generate_quality_insights_report,
    get_analytics_report, get_all_analytics_reports
)
from ai_ml_engine import ai_ml_engine
from real_time_engine import real_time_engine
from cloud_integration import cloud_manager
from security_advanced import security_manager
from monitoring_advanced import monitoring_system
from automation_engine import automation_engine
from realtime_analysis import websocket_endpoint, realtime_engine
from multimodal_analysis import multimodal_engine, MultimodalInput
from custom_model_training import custom_training_engine, TrainingDataset, TrainingConfig
from advanced_analytics_dashboard import analytics_dashboard, AnalyticsQuery, Dashboard, ReportConfig

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


def create_error_response(error: str, detail: str = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error=error,
        detail=detail,
        timestamp=time.time()
    )


def create_success_response(data: Any, message: str = None) -> Dict[str, Any]:
    """Create standardized success response for frontend"""
    response = {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": time.time()
    }
    if message:
        response["message"] = message
    return response


def create_error_response_dict(message: str, status_code: int = 500, error_type: str = "Error", detail: Any = None) -> Dict[str, Any]:
    """Create standardized error response dict for frontend"""
    error = {
        "message": message,
        "status_code": status_code,
        "type": error_type
    }
    if detail:
        error["detail"] = detail
    
    return {
        "success": False,
        "data": None,
        "error": error,
        "timestamp": time.time()
    }


@router.get("/", response_model=Dict[str, Any])
async def get_root() -> Dict[str, Any]:
    """Root endpoint with API information for frontend"""
    logger.info("Root endpoint accessed")
    from config import settings
    
    return {
        "success": True,
        "data": {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "Advanced Content Redundancy Detector API with AI/ML capabilities",
            "status": "ready",
            "frontend_ready": True,
            "cors_enabled": True,
            "docs": "/docs",
            "api_docs": "/redoc",
            "health": "/health",
            "api_health": "/api/v1/health",
            "api_version": "v1",
            "timestamp": time.time()
        },
        "error": None,
        "endpoints": {
            "base": "/api/v1",
            "health": "/api/v1/health",
            "analyze": "/api/v1/analyze",
            "similarity": "/api/v1/similarity",
            "quality": "/api/v1/quality",
            "stats": "/api/v1/stats",
            "metrics": "/api/v1/metrics",
            "ai_sentiment": "/api/v1/ai/sentiment",
            "ai_language": "/api/v1/ai/language",
            "ai_topics": "/api/v1/ai/topics",
            "ai_semantic_similarity": "/api/v1/ai/semantic-similarity",
            "ai_plagiarism": "/api/v1/ai/plagiarism",
            "ai_entities": "/api/v1/ai/entities",
            "ai_summary": "/api/v1/ai/summary",
            "ai_readability": "/api/v1/ai/readability",
            "ai_comprehensive": "/api/v1/ai/comprehensive",
            "ai_batch": "/api/v1/ai/batch",
            "batch_process": "/api/v1/batch/process",
            "realtime_start": "/api/v1/realtime/start",
            "multimodal_analyze": "/api/v1/multimodal/analyze"
        },
        "features": [
            "Content redundancy analysis",
            "Text similarity comparison",
            "Content quality assessment",
            "AI-powered sentiment analysis",
            "Language detection",
            "Topic extraction",
            "Semantic similarity",
            "Plagiarism detection",
            "Entity extraction",
            "Text summarization",
            "Advanced readability analysis",
            "Comprehensive AI analysis",
            "Batch processing",
            "Real-time analysis",
            "Multimodal analysis",
            "Custom model training",
            "Analytics dashboard",
            "Webhook support",
            "Export capabilities",
            "Cloud integration",
            "Security features",
            "Monitoring and alerts",
            "Automation workflows"
        ]
    }


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint - Frontend-friendly format"""
    logger.info("Health check requested")
    
    try:
        health_data = get_health_status()
        
        return {
            "success": True,
            "data": {
                **health_data,
                "api_version": "v1",
                "api_ready": True,
                "frontend_compatible": True
            },
            "error": None,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "data": None,
            "error": {
                "message": "Health check failed",
                "status_code": 500,
                "type": "HealthCheckError"
            },
            "timestamp": time.time()
        }


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_content_endpoint(input_data: ContentInput) -> Dict[str, Any]:
    """Analyze content for redundancy - Frontend-ready"""
    logger.info(f"Content analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = analyze_content(input_data.content)
        logger.info(f"Analysis completed - Redundancy: {result['redundancy_score']:.2f}")
        
        # Wrap in frontend-friendly format
        return create_success_response(AnalysisResult(**result).model_dump())
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/similarity", response_model=Dict[str, Any])
async def check_similarity_endpoint(input_data: SimilarityInput) -> Dict[str, Any]:
    """Check similarity between two texts - Frontend-ready"""
    logger.info(f"Similarity check requested - Text1: {len(input_data.text1)}, Text2: {len(input_data.text2)}")
    
    try:
        result = detect_similarity(input_data.text1, input_data.text2, input_data.threshold)
        logger.info(f"Similarity check completed - Score: {result['similarity_score']:.2f}")
        
        # Wrap in frontend-friendly format
        return create_success_response(SimilarityResult(**result).model_dump())
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Similarity check error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quality", response_model=Dict[str, Any])
async def assess_quality_endpoint(input_data: ContentInput) -> Dict[str, Any]:
    """Assess content quality - Frontend-ready"""
    logger.info(f"Quality assessment requested - Length: {len(input_data.content)}")
    
    try:
        result = assess_quality(input_data.content)
        logger.info(f"Quality assessment completed - Readability: {result['readability_score']:.2f}")
        
        # Wrap in frontend-friendly format
        return create_success_response(QualityResult(**result).model_dump())
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quality assessment error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=StatsResponse)
async def get_stats_endpoint() -> StatsResponse:
    """Get system statistics"""
    logger.info("System statistics requested")
    
    try:
        stats_data = get_system_stats()
        return StatsResponse(**stats_data)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/metrics")
async def get_metrics_endpoint():
    """Get system metrics"""
    logger.info("System metrics requested")
    
    try:
        return {
            "system": get_system_metrics(),
            "endpoints": get_endpoint_metrics(),
            "health": get_health_metrics(),
            "cache": get_cache_stats()
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


@router.post("/cache/clear")
async def clear_cache_endpoint():
    """Clear system cache"""
    logger.info("Cache clear requested")
    
    try:
        clear_cache()
        return {"message": "Cache cleared successfully", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# =============================================================================
# BATCH PROCESSING ENDPOINTS
# =============================================================================

@router.post("/batch/process")
async def process_batch_endpoint(batch_data: dict):
    """Process a batch of analysis jobs"""
    logger.info("Batch processing requested")
    
    try:
        jobs_data = batch_data.get("jobs", [])
        if not jobs_data:
            raise ValueError("No jobs provided")
        
        # Create batch jobs
        jobs = []
        for i, job_data in enumerate(jobs_data):
            job_id = f"job_{i}_{int(time.time())}"
            operation = job_data.get("operation")
            input_data = job_data.get("input")
            
            if operation == "analyze":
                job = create_batch_job(job_id, ContentInput(**input_data), operation)
            elif operation == "similarity":
                job = create_batch_job(job_id, SimilarityInput(**input_data), operation)
            elif operation == "quality":
                job = create_batch_job(job_id, ContentInput(**input_data), operation)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            jobs.append(job)
        
        # Create and process batch
        batch_id = f"batch_{int(time.time())}"
        batch_request = create_batch_request(batch_id, jobs)
        result = await process_batch_async(batch_request)
        
        return {
            "batch_id": batch_id,
            "status": result.status.value,
            "total_jobs": result.total_jobs,
            "completed_jobs": result.completed_jobs,
            "failed_jobs": result.failed_jobs,
            "progress": result.progress
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/batch/{batch_id}")
async def get_batch_status_endpoint(batch_id: str):
    """Get batch processing status"""
    logger.info(f"Batch status requested: {batch_id}")
    
    try:
        batch = get_batch_status(batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "batch_id": batch_id,
            "status": batch.status.value,
            "total_jobs": batch.total_jobs,
            "completed_jobs": batch.completed_jobs,
            "failed_jobs": batch.failed_jobs,
            "progress": batch.progress,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch status error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/batch")
async def get_all_batches_endpoint():
    """Get all batches"""
    logger.info("All batches requested")
    
    try:
        batches = get_all_batches()
        return {
            "batches": [
                {
                    "batch_id": batch_id,
                    "status": batch.status.value,
                    "total_jobs": batch.total_jobs,
                    "progress": batch.progress,
                    "created_at": batch.created_at
                }
                for batch_id, batch in batches.items()
            ]
        }
    except Exception as e:
        logger.error(f"Get batches error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch/{batch_id}/cancel")
async def cancel_batch_endpoint(batch_id: str):
    """Cancel a batch"""
    logger.info(f"Batch cancellation requested: {batch_id}")
    
    try:
        success = cancel_batch(batch_id)
        if not success:
            raise HTTPException(status_code=404, detail="Batch not found or already completed")
        
        return {"message": "Batch cancelled successfully", "batch_id": batch_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch cancellation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# WEBHOOK ENDPOINTS
# =============================================================================

@router.post("/webhooks/register")
async def register_webhook_endpoint(webhook_data: dict):
    """Register a webhook endpoint"""
    logger.info("Webhook registration requested")
    
    try:
        endpoint = WebhookEndpoint(
            id=webhook_data["id"],
            url=webhook_data["url"],
            events=[WebhookEvent(event) for event in webhook_data.get("events", [])],
            secret=webhook_data.get("secret"),
            timeout=webhook_data.get("timeout", 30),
            retry_count=webhook_data.get("retry_count", 3)
        )
        
        register_webhook_endpoint(endpoint)
        return {"message": "Webhook registered successfully", "endpoint_id": endpoint.id}
        
    except Exception as e:
        logger.error(f"Webhook registration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/webhooks/{endpoint_id}")
async def unregister_webhook_endpoint(endpoint_id: str):
    """Unregister a webhook endpoint"""
    logger.info(f"Webhook unregistration requested: {endpoint_id}")
    
    try:
        success = unregister_webhook_endpoint(endpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook endpoint not found")
        
        return {"message": "Webhook unregistered successfully", "endpoint_id": endpoint_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook unregistration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/webhooks")
async def get_webhooks_endpoint():
    """Get all webhook endpoints"""
    logger.info("Webhooks list requested")
    
    try:
        endpoints = get_webhook_endpoints()
        return {
            "endpoints": [
                {
                    "id": endpoint.id,
                    "url": endpoint.url,
                    "events": [event.value for event in endpoint.events],
                    "is_active": endpoint.is_active,
                    "created_at": endpoint.created_at
                }
                for endpoint in endpoints
            ]
        }
    except Exception as e:
        logger.error(f"Get webhooks error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/webhooks/stats")
async def get_webhook_stats_endpoint():
    """Get webhook statistics"""
    logger.info("Webhook stats requested")
    
    try:
        stats = get_webhook_stats()
        return stats
    except Exception as e:
        logger.error(f"Webhook stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# EXPORT ENDPOINTS
# =============================================================================

@router.post("/export")
async def export_data_endpoint(export_data: dict):
    """Export analysis data"""
    logger.info("Data export requested")
    
    try:
        data = export_data.get("data", [])
        format_type = ExportFormat(export_data.get("format", "json"))
        filename = export_data.get("filename")
        
        if not data:
            raise ValueError("No data provided for export")
        
        # Create and process export
        export_request = create_export(data, format_type, filename)
        content = process_export(export_request)
        
        return {
            "export_id": export_request.id,
            "format": format_type.value,
            "file_size": len(content),
            "filename": export_request.filename,
            "created_at": export_request.created_at
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export/{export_id}")
async def get_export_endpoint(export_id: str):
    """Get export by ID"""
    logger.info(f"Export requested: {export_id}")
    
    try:
        export_request = get_export(export_id)
        if not export_request:
            raise HTTPException(status_code=404, detail="Export not found")
        
        return {
            "export_id": export_id,
            "format": export_request.format.value,
            "filename": export_request.filename,
            "file_size": export_request.file_size,
            "created_at": export_request.created_at,
            "completed_at": export_request.completed_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get export error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export")
async def get_all_exports_endpoint():
    """Get all exports"""
    logger.info("All exports requested")
    
    try:
        exports = get_all_exports()
        return {
            "exports": [
                {
                    "export_id": export.id,
                    "format": export.format.value,
                    "filename": export.filename,
                    "file_size": export.file_size,
                    "created_at": export.created_at
                }
                for export in exports
            ]
        }
    except Exception as e:
        logger.error(f"Get exports error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics/performance")
async def get_performance_analytics_endpoint():
    """Get performance analytics report"""
    logger.info("Performance analytics requested")
    
    try:
        report = generate_performance_report()
        return report.data
    except Exception as e:
        logger.error(f"Performance analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/content")
async def get_content_analytics_endpoint():
    """Get content analytics report"""
    logger.info("Content analytics requested")
    
    try:
        report = generate_content_insights_report()
        return report.data
    except Exception as e:
        logger.error(f"Content analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/similarity")
async def get_similarity_analytics_endpoint():
    """Get similarity analytics report"""
    logger.info("Similarity analytics requested")
    
    try:
        report = generate_similarity_insights_report()
        return report.data
    except Exception as e:
        logger.error(f"Similarity analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/quality")
async def get_quality_analytics_endpoint():
    """Get quality analytics report"""
    logger.info("Quality analytics requested")
    
    try:
        report = generate_quality_insights_report()
        return report.data
    except Exception as e:
        logger.error(f"Quality analytics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/reports")
async def get_all_analytics_reports_endpoint():
    """Get all analytics reports"""
    logger.info("All analytics reports requested")
    
    try:
        reports = get_all_analytics_reports()
        return {
            "reports": [
                {
                    "report_id": report.id,
                    "report_type": report.report_type,
                    "generated_at": report.generated_at,
                    "period_start": report.period_start,
                    "period_end": report.period_end
                }
                for report in reports
            ]
        }
    except Exception as e:
        logger.error(f"Get analytics reports error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# AI/ML ENDPOINTS
# =============================================================================

@router.post("/ai/predict/similarity")
async def predict_similarity_ai_endpoint(prediction_data: dict):
    """Predict similarity using AI/ML models"""
    logger.info("AI similarity prediction requested")
    
    try:
        text1 = prediction_data.get("text1", "")
        text2 = prediction_data.get("text2", "")
        model_id = prediction_data.get("model_id", "default")
        
        if not text1 or not text2:
            raise ValueError("Both text1 and text2 are required")
        
        result = await ai_ml_engine.predict_similarity(text1, text2, model_id)
        
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI similarity prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/predict/quality")
async def predict_quality_ai_endpoint(prediction_data: dict):
    """Predict content quality using AI/ML models"""
    logger.info("AI quality prediction requested")
    
    try:
        content = prediction_data.get("content", "")
        model_id = prediction_data.get("model_id", "default")
        
        if not content:
            raise ValueError("Content is required")
        
        result = await ai_ml_engine.predict_quality(content, model_id)
        
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI quality prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/predict/sentiment")
async def predict_sentiment_ai_endpoint(prediction_data: dict):
    """Predict sentiment using AI/ML models"""
    logger.info("AI sentiment prediction requested")
    
    try:
        content = prediction_data.get("content", "")
        model_id = prediction_data.get("model_id", "default")
        
        if not content:
            raise ValueError("Content is required")
        
        result = await ai_ml_engine.predict_sentiment(content, model_id)
        
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI sentiment prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/predict/topics")
async def predict_topics_ai_endpoint(prediction_data: dict):
    """Predict topics using AI/ML models"""
    logger.info("AI topic prediction requested")
    
    try:
        content = prediction_data.get("content", "")
        num_topics = prediction_data.get("num_topics", 5)
        model_id = prediction_data.get("model_id", "default")
        
        if not content:
            raise ValueError("Content is required")
        
        result = await ai_ml_engine.predict_topics(content, num_topics, model_id)
        
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI topic prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/cluster")
async def cluster_content_ai_endpoint(clustering_data: dict):
    """Cluster content using AI/ML models"""
    logger.info("AI content clustering requested")
    
    try:
        contents = clustering_data.get("contents", [])
        num_clusters = clustering_data.get("num_clusters", 3)
        model_id = clustering_data.get("model_id", "default")
        
        if not contents or len(contents) < 2:
            raise ValueError("At least 2 contents are required for clustering")
        
        result = await ai_ml_engine.cluster_content(contents, num_clusters, model_id)
        
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI clustering error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/generate")
async def generate_ai_response_endpoint(generation_data: dict):
    """Generate AI response using AI models"""
    logger.info("AI response generation requested")
    
    try:
        prompt = generation_data.get("prompt", "")
        model_id = generation_data.get("model_id", "default")
        
        if not prompt:
            raise ValueError("Prompt is required")
        
        result = await ai_ml_engine.generate_ai_response(prompt, model_id)
        
        return {
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_name": result.model_name,
            "processing_time": result.processing_time,
            "metadata": result.metadata
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/ai/models")
async def get_ai_models_endpoint():
    """Get all AI/ML models"""
    logger.info("AI models list requested")
    
    try:
        ml_models = ai_ml_engine.get_models()
        ai_models = ai_ml_engine.get_ai_models()
        model_stats = ai_ml_engine.get_model_stats()
        
        return {
            "ml_models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "type": model.model_type.value,
                    "version": model.version,
                    "accuracy": model.accuracy,
                    "is_active": model.is_active
                }
                for model in ml_models.values()
            ],
            "ai_models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "type": model.model_type.value,
                    "provider": model.provider,
                    "is_active": model.is_active
                }
                for model in ai_models.values()
            ],
            "stats": model_stats
        }
    except Exception as e:
        logger.error(f"Get AI models error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# REAL-TIME ENDPOINTS
# =============================================================================

@router.post("/realtime/stream/create")
async def create_realtime_stream_endpoint(stream_data: dict):
    """Create real-time stream"""
    logger.info("Real-time stream creation requested")
    
    try:
        stream_id = stream_data.get("stream_id", f"stream_{int(time.time())}")
        name = stream_data.get("name", "Unnamed Stream")
        stream_type = stream_data.get("type", "content_analysis")
        
        # Map string to enum
        from real_time_engine import StreamType
        type_mapping = {
            "content_analysis": StreamType.CONTENT_ANALYSIS,
            "similarity_detection": StreamType.SIMILARITY_DETECTION,
            "quality_assessment": StreamType.QUALITY_ASSESSMENT,
            "batch_processing": StreamType.BATCH_PROCESSING,
            "system_monitoring": StreamType.SYSTEM_MONITORING,
            "ai_ml_processing": StreamType.AI_ML_PROCESSING
        }
        
        stream_type_enum = type_mapping.get(stream_type, StreamType.CONTENT_ANALYSIS)
        
        stream = await real_time_engine.create_stream(stream_id, name, stream_type_enum)
        
        return {
            "stream_id": stream.id,
            "name": stream.name,
            "type": stream.stream_type.value,
            "status": stream.status.value,
            "created_at": stream.created_at
        }
        
    except Exception as e:
        logger.error(f"Create real-time stream error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/realtime/stream/{stream_id}")
async def get_realtime_stream_endpoint(stream_id: str):
    """Get real-time stream status"""
    logger.info(f"Real-time stream status requested: {stream_id}")
    
    try:
        stats = await real_time_engine.get_stream_stats(stream_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get real-time stream error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/realtime/streams")
async def get_all_realtime_streams_endpoint():
    """Get all real-time streams"""
    logger.info("All real-time streams requested")
    
    try:
        streams = await real_time_engine.get_all_streams()
        return {"streams": streams}
    except Exception as e:
        logger.error(f"Get real-time streams error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/realtime/stream/{stream_id}/subscribe")
async def subscribe_to_stream_endpoint(stream_id: str, subscription_data: dict):
    """Subscribe to real-time stream"""
    logger.info(f"Stream subscription requested: {stream_id}")
    
    try:
        subscriber_id = subscription_data.get("subscriber_id", f"sub_{int(time.time())}")
        filters = subscription_data.get("filters", {})
        
        # Simple callback for demonstration
        def callback(event):
            logger.info(f"Event received: {event.event_type} - {event.data}")
        
        success = await real_time_engine.subscribe_to_stream(
            subscriber_id, stream_id, callback, filters
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return {
            "subscriber_id": subscriber_id,
            "stream_id": stream_id,
            "subscribed": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subscribe to stream error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/realtime/events/{stream_id}")
async def get_stream_events_endpoint(stream_id: str, limit: int = 100):
    """Get stream events"""
    logger.info(f"Stream events requested: {stream_id}")
    
    try:
        events = await real_time_engine.get_stream_events(stream_id, limit)
        return {
            "stream_id": stream_id,
            "events": [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "data": event.data,
                    "timestamp": event.timestamp,
                    "metadata": event.metadata
                }
                for event in events
            ]
        }
    except Exception as e:
        logger.error(f"Get stream events error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/realtime/engine/stats")
async def get_realtime_engine_stats_endpoint():
    """Get real-time engine statistics"""
    logger.info("Real-time engine stats requested")
    
    try:
        stats = real_time_engine.get_engine_stats()
        return stats
    except Exception as e:
        logger.error(f"Get real-time engine stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# CLOUD INTEGRATION ENDPOINTS
# =============================================================================

@router.post("/cloud/config/add")
async def add_cloud_config_endpoint(config_data: dict):
    """Add cloud configuration"""
    logger.info("Cloud configuration addition requested")
    
    try:
        name = config_data.get("name")
        provider = config_data.get("provider")
        region = config_data.get("region")
        credentials = config_data.get("credentials", {})
        
        if not all([name, provider, region]):
            raise ValueError("Name, provider, and region are required")
        
        from cloud_integration import CloudConfig, CloudProvider
        cloud_config = CloudConfig(
            provider=CloudProvider(provider),
            region=region,
            credentials=credentials
        )
        
        cloud_manager.add_cloud_config(name, cloud_config)
        
        return {
            "message": "Cloud configuration added successfully",
            "config_name": name,
            "provider": provider,
            "region": region
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Add cloud config error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cloud/upload")
async def upload_to_cloud_endpoint(upload_data: dict):
    """Upload data to cloud storage"""
    logger.info("Cloud upload requested")
    
    try:
        config_name = upload_data.get("config_name")
        bucket = upload_data.get("bucket")
        data = upload_data.get("data", {})
        
        if not all([config_name, bucket]):
            raise ValueError("Config name and bucket are required")
        
        success = await cloud_manager.upload_analysis_result(config_name, bucket, data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Upload failed")
        
        return {
            "message": "Data uploaded successfully",
            "config_name": config_name,
            "bucket": bucket
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud upload error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cloud/download")
async def download_from_cloud_endpoint(download_data: dict):
    """Download data from cloud storage"""
    logger.info("Cloud download requested")
    
    try:
        config_name = download_data.get("config_name")
        bucket = download_data.get("bucket")
        key = download_data.get("key")
        
        if not all([config_name, bucket, key]):
            raise ValueError("Config name, bucket, and key are required")
        
        data = await cloud_manager.download_analysis_result(config_name, bucket, key)
        
        if not data:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return {
            "message": "Data downloaded successfully",
            "data": data,
            "config_name": config_name,
            "bucket": bucket,
            "key": key
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud download error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cloud/backup")
async def backup_to_cloud_endpoint(backup_data: dict):
    """Backup data to cloud storage"""
    logger.info("Cloud backup requested")
    
    try:
        config_name = backup_data.get("config_name")
        bucket = backup_data.get("bucket")
        data = backup_data.get("data", {})
        
        if not all([config_name, bucket]):
            raise ValueError("Config name and bucket are required")
        
        success = await cloud_manager.backup_system_data(config_name, bucket, data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Backup failed")
        
        return {
            "message": "Data backed up successfully",
            "config_name": config_name,
            "bucket": bucket
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cloud backup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cloud/list/{config_name}/{bucket}")
async def list_cloud_files_endpoint(config_name: str, bucket: str, prefix: str = ""):
    """List files in cloud storage"""
    logger.info(f"Cloud file listing requested: {config_name}/{bucket}")
    
    try:
        files = await cloud_manager.list_analysis_results(config_name, bucket, prefix)
        
        return {
            "config_name": config_name,
            "bucket": bucket,
            "prefix": prefix,
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        logger.error(f"Cloud file listing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/cloud/cleanup/{config_name}/{bucket}")
async def cleanup_cloud_files_endpoint(config_name: str, bucket: str, days_old: int = 30):
    """Cleanup old files in cloud storage"""
    logger.info(f"Cloud cleanup requested: {config_name}/{bucket}")
    
    try:
        deleted_count = await cloud_manager.delete_old_analysis_results(config_name, bucket, days_old)
        
        return {
            "message": "Cloud cleanup completed",
            "config_name": config_name,
            "bucket": bucket,
            "days_old": days_old,
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Cloud cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cloud/configs")
async def get_cloud_configs_endpoint():
    """Get all cloud configurations"""
    logger.info("Cloud configurations requested")
    
    try:
        configs = cloud_manager.get_cloud_configs()
        stats = cloud_manager.get_cloud_stats()
        
        return {
            "configs": [
                {
                    "name": name,
                    "provider": config.provider.value,
                    "region": config.region,
                    "endpoint": config.endpoint
                }
                for name, config in configs.items()
            ],
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Get cloud configs error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# SECURITY ENDPOINTS
# =============================================================================

@router.post("/security/api-key/create")
async def create_api_key_endpoint(key_data: dict):
    """Create API key"""
    logger.info("API key creation requested")
    
    try:
        user_id = key_data.get("user_id")
        name = key_data.get("name")
        permissions = key_data.get("permissions", [])
        rate_limit = key_data.get("rate_limit", 1000)
        expires_days = key_data.get("expires_days")
        
        if not all([user_id, name]):
            raise ValueError("User ID and name are required")
        
        api_key = security_manager.create_api_key(user_id, name, permissions, rate_limit, expires_days)
        
        return {
            "message": "API key created successfully",
            "api_key": api_key,
            "user_id": user_id,
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create API key error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/security/api-key/validate")
async def validate_api_key_endpoint(validation_data: dict):
    """Validate API key"""
    logger.info("API key validation requested")
    
    try:
        api_key = validation_data.get("api_key")
        
        if not api_key:
            raise ValueError("API key is required")
        
        key_record = security_manager.validate_api_key(api_key)
        
        if not key_record:
            raise HTTPException(status_code=401, detail="Invalid or expired API key")
        
        return {
            "valid": True,
            "key_id": key_record.id,
            "name": key_record.name,
            "user_id": key_record.user_id,
            "permissions": key_record.permissions,
            "rate_limit": key_record.rate_limit,
            "is_active": key_record.is_active
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validate API key error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/security/session/create")
async def create_session_endpoint(session_data: dict):
    """Create user session"""
    logger.info("User session creation requested")
    
    try:
        user_id = session_data.get("user_id")
        ip_address = session_data.get("ip_address", "127.0.0.1")
        user_agent = session_data.get("user_agent", "Unknown")
        
        if not user_id:
            raise ValueError("User ID is required")
        
        session_id = security_manager.create_user_session(user_id, ip_address, user_agent)
        
        return {
            "message": "Session created successfully",
            "session_id": session_id,
            "user_id": user_id,
            "ip_address": ip_address
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/security/session/validate")
async def validate_session_endpoint(validation_data: dict):
    """Validate user session"""
    logger.info("Session validation requested")
    
    try:
        session_id = validation_data.get("session_id")
        
        if not session_id:
            raise ValueError("Session ID is required")
        
        session = security_manager.validate_session(session_id)
        
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return {
            "valid": True,
            "session_id": session.id,
            "user_id": session.user_id,
            "ip_address": session.ip_address,
            "is_active": session.is_active,
            "created_at": session.created_at,
            "last_activity": session.last_activity
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validate session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/security/events")
async def get_security_events_endpoint(limit: int = 100):
    """Get security events"""
    logger.info("Security events requested")
    
    try:
        events = security_manager.get_security_events(limit)
        
        return {
            "events": [
                {
                    "id": event.id,
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "source_ip": event.source_ip,
                    "user_id": event.user_id,
                    "description": event.description,
                    "timestamp": event.timestamp,
                    "resolved": event.resolved
                }
                for event in events
            ],
            "count": len(events)
        }
    except Exception as e:
        logger.error(f"Get security events error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/security/audit")
async def get_audit_log_endpoint(limit: int = 100):
    """Get audit log"""
    logger.info("Audit log requested")
    
    try:
        audit_entries = security_manager.get_audit_log(limit)
        
        return {
            "audit_entries": audit_entries,
            "count": len(audit_entries)
        }
    except Exception as e:
        logger.error(f"Get audit log error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/security/stats")
async def get_security_stats_endpoint():
    """Get security statistics"""
    logger.info("Security stats requested")
    
    try:
        stats = security_manager.get_security_stats()
        return stats
    except Exception as e:
        logger.error(f"Get security stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# MONITORING ENDPOINTS
# =============================================================================

@router.get("/monitoring/metrics")
async def get_monitoring_metrics_endpoint():
    """Get monitoring metrics"""
    logger.info("Monitoring metrics requested")
    
    try:
        metrics = monitoring_system.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Get monitoring metrics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/alerts")
async def get_monitoring_alerts_endpoint(limit: int = 100):
    """Get monitoring alerts"""
    logger.info("Monitoring alerts requested")
    
    try:
        alerts = monitoring_system.get_alerts(limit)
        
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ],
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Get monitoring alerts error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/monitoring/alert/create")
async def create_monitoring_alert_endpoint(alert_data: dict):
    """Create monitoring alert"""
    logger.info("Monitoring alert creation requested")
    
    try:
        name = alert_data.get("name")
        level = alert_data.get("level", "info")
        message = alert_data.get("message")
        metadata = alert_data.get("metadata", {})
        
        if not all([name, message]):
            raise ValueError("Name and message are required")
        
        from monitoring_advanced import AlertLevel
        level_mapping = {
            "info": AlertLevel.INFO,
            "warning": AlertLevel.WARNING,
            "error": AlertLevel.ERROR,
            "critical": AlertLevel.CRITICAL
        }
        
        alert_level = level_mapping.get(level, AlertLevel.INFO)
        
        alert = monitoring_system.create_alert(name, alert_level, message, metadata)
        
        return {
            "message": "Alert created successfully",
            "alert_id": alert.id,
            "name": alert.name,
            "level": alert.level.value,
            "message": alert.message
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create monitoring alert error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/monitoring/alert/{alert_id}/resolve")
async def resolve_monitoring_alert_endpoint(alert_id: str):
    """Resolve monitoring alert"""
    logger.info(f"Monitoring alert resolution requested: {alert_id}")
    
    try:
        success = monitoring_system.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "message": "Alert resolved successfully",
            "alert_id": alert_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resolve monitoring alert error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/health")
async def get_monitoring_health_endpoint():
    """Get monitoring health status"""
    logger.info("Monitoring health status requested")
    
    try:
        health_status = monitoring_system.get_health_status()
        return health_status
    except Exception as e:
        logger.error(f"Get monitoring health error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/performance")
async def get_monitoring_performance_endpoint():
    """Get monitoring performance metrics"""
    logger.info("Monitoring performance metrics requested")
    
    try:
        performance_metrics = monitoring_system.get_performance_metrics()
        return performance_metrics
    except Exception as e:
        logger.error(f"Get monitoring performance error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/system")
async def get_monitoring_system_metrics_endpoint():
    """Get monitoring system metrics"""
    logger.info("Monitoring system metrics requested")
    
    try:
        system_metrics = monitoring_system.get_system_metrics()
        return system_metrics
    except Exception as e:
        logger.error(f"Get monitoring system metrics error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/stats")
async def get_monitoring_stats_endpoint():
    """Get monitoring statistics"""
    logger.info("Monitoring stats requested")
    
    try:
        stats = monitoring_system.get_monitoring_stats()
        return stats
    except Exception as e:
        logger.error(f"Get monitoring stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =============================================================================
# AUTOMATION ENDPOINTS
# =============================================================================

@router.post("/automation/workflow/create")
async def create_automation_workflow_endpoint(workflow_data: dict):
    """Create automation workflow"""
    logger.info("Automation workflow creation requested")
    
    try:
        name = workflow_data.get("name")
        description = workflow_data.get("description", "")
        steps = workflow_data.get("steps", [])
        
        if not all([name, steps]):
            raise ValueError("Name and steps are required")
        
        workflow = automation_engine.create_workflow(name, description, steps)
        
        return {
            "message": "Workflow created successfully",
            "workflow_id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "steps_count": len(workflow.steps)
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create automation workflow error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/automation/workflow/{workflow_id}/execute")
async def execute_automation_workflow_endpoint(workflow_id: str, execution_data: dict = None):
    """Execute automation workflow"""
    logger.info(f"Automation workflow execution requested: {workflow_id}")
    
    try:
        context = execution_data.get("context", {}) if execution_data else {}
        
        result = await automation_engine.execute_workflow(workflow_id, context)
        
        if not result:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "message": "Workflow executed successfully",
            "workflow_id": workflow_id,
            "execution_id": result.execution_id,
            "status": result.status.value,
            "steps_completed": result.steps_completed,
            "total_steps": result.total_steps,
            "result": result.result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execute automation workflow error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/automation/workflow/{workflow_id}")
async def get_automation_workflow_endpoint(workflow_id: str):
    """Get automation workflow"""
    logger.info(f"Automation workflow requested: {workflow_id}")
    
    try:
        workflow = automation_engine.get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "type": step.step_type.value,
                    "config": step.config
                }
                for step in workflow.steps
            ],
            "created_at": workflow.created_at,
            "is_active": workflow.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get automation workflow error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/automation/workflows")
async def get_automation_workflows_endpoint():
    """Get all automation workflows"""
    logger.info("All automation workflows requested")
    
    try:
        workflows = automation_engine.get_workflows()
        
        return {
            "workflows": [
                {
                    "workflow_id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "steps_count": len(workflow.steps),
                    "created_at": workflow.created_at,
                    "is_active": workflow.is_active
                }
                for workflow in workflows.values()
            ],
            "count": len(workflows)
        }
    except Exception as e:
        logger.error(f"Get automation workflows error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/automation/rule/create")
async def create_automation_rule_endpoint(rule_data: dict):
    """Create automation rule"""
    logger.info("Automation rule creation requested")
    
    try:
        name = rule_data.get("name")
        condition = rule_data.get("condition")
        actions = rule_data.get("actions", [])
        is_active = rule_data.get("is_active", True)
        
        if not all([name, condition, actions]):
            raise ValueError("Name, condition, and actions are required")
        
        rule = automation_engine.create_rule(name, condition, actions, is_active)
        
        return {
            "message": "Rule created successfully",
            "rule_id": rule.id,
            "name": rule.name,
            "condition": rule.condition,
            "actions_count": len(rule.actions),
            "is_active": rule.is_active
        }
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create automation rule error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/automation/rules")
async def get_automation_rules_endpoint():
    """Get all automation rules"""
    logger.info("All automation rules requested")
    
    try:
        rules = automation_engine.get_rules()
        
        return {
            "rules": [
                {
                    "rule_id": rule.id,
                    "name": rule.name,
                    "condition": rule.condition,
                    "actions_count": len(rule.actions),
                    "is_active": rule.is_active,
                    "created_at": rule.created_at,
                    "last_triggered": rule.last_triggered,
                    "trigger_count": rule.trigger_count
                }
                for rule in rules.values()
            ],
            "count": len(rules)
        }
    except Exception as e:
        logger.error(f"Get automation rules error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/automation/executions")
async def get_automation_executions_endpoint(limit: int = 100):
    """Get automation executions"""
    logger.info("Automation executions requested")
    
    try:
        executions = automation_engine.get_executions(limit)
        
        return {
            "executions": [
                {
                    "execution_id": execution.execution_id,
                    "workflow_id": execution.workflow_id,
                    "status": execution.status.value,
                    "steps_completed": execution.steps_completed,
                    "total_steps": execution.total_steps,
                    "started_at": execution.started_at,
                    "completed_at": execution.completed_at,
                    "result": execution.result
                }
                for execution in executions
            ],
            "count": len(executions)
        }
    except Exception as e:
        logger.error(f"Get automation executions error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/automation/stats")
async def get_automation_stats_endpoint():
    """Get automation statistics"""
    logger.info("Automation stats requested")
    
    try:
        stats = automation_engine.get_automation_stats()
        return stats
    except Exception as e:
        logger.error(f"Get automation stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# AI/ML Enhanced Endpoints
# ============================================================================

@router.post("/ai/sentiment", response_model=SentimentResult)
async def analyze_sentiment_endpoint(input_data: ContentInput) -> SentimentResult:
    """Analyze sentiment of content using AI/ML"""
    logger.info(f"Sentiment analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = await analyze_sentiment(input_data.content)
        logger.info(f"Sentiment analysis completed - Sentiment: {result.get('dominant_sentiment', 'unknown')}")
        return SentimentResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/language", response_model=LanguageResult)
async def detect_language_endpoint(input_data: ContentInput) -> LanguageResult:
    """Detect language of content"""
    logger.info(f"Language detection requested - Length: {len(input_data.content)}")
    
    try:
        result = await detect_language(input_data.content)
        logger.info(f"Language detection completed - Language: {result.get('language', 'unknown')}")
        return LanguageResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/topics", response_model=TopicResult)
async def extract_topics_endpoint(input_data: TopicExtractionInput) -> TopicResult:
    """Extract topics from a collection of texts"""
    logger.info(f"Topic extraction requested - Texts: {len(input_data.texts)}, Topics: {input_data.num_topics}")
    
    try:
        result = await extract_topics(input_data.texts, input_data.num_topics)
        logger.info(f"Topic extraction completed - Topics found: {result.get('num_topics', 0)}")
        return TopicResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/semantic-similarity", response_model=SemanticSimilarityResult)
async def calculate_semantic_similarity_endpoint(input_data: SimilarityInput) -> SemanticSimilarityResult:
    """Calculate semantic similarity between two texts using AI/ML"""
    logger.info(f"Semantic similarity requested - Text1: {len(input_data.text1)}, Text2: {len(input_data.text2)}")
    
    try:
        result = await calculate_semantic_similarity(input_data.text1, input_data.text2)
        logger.info(f"Semantic similarity completed - Score: {result.get('similarity_score', 0):.3f}")
        return SemanticSimilarityResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Semantic similarity error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/plagiarism", response_model=PlagiarismResult)
async def detect_plagiarism_endpoint(input_data: PlagiarismDetectionInput) -> PlagiarismResult:
    """Detect potential plagiarism in content"""
    logger.info(f"Plagiarism detection requested - Content: {len(input_data.content)}, References: {len(input_data.reference_texts)}")
    
    try:
        result = await detect_plagiarism(input_data.content, input_data.reference_texts, input_data.threshold)
        logger.info(f"Plagiarism detection completed - Plagiarized: {result.get('is_plagiarized', False)}")
        return PlagiarismResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Plagiarism detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/entities", response_model=EntityResult)
async def extract_entities_endpoint(input_data: ContentInput) -> EntityResult:
    """Extract named entities from content"""
    logger.info(f"Entity extraction requested - Length: {len(input_data.content)}")
    
    try:
        result = await extract_entities(input_data.content)
        logger.info(f"Entity extraction completed - Entities: {result.get('entity_count', 0)}")
        return EntityResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/summary", response_model=SummaryResult)
async def generate_summary_endpoint(input_data: SummaryInput) -> SummaryResult:
    """Generate summary of content using AI/ML"""
    logger.info(f"Text summarization requested - Length: {len(input_data.content)}, Max length: {input_data.max_length}")
    
    try:
        result = await generate_summary(input_data.content, input_data.max_length)
        logger.info(f"Text summarization completed - Summary length: {result.get('summary_length', 0)}")
        return SummaryResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text summarization error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/readability", response_model=ReadabilityResult)
async def analyze_readability_advanced_endpoint(input_data: ContentInput) -> ReadabilityResult:
    """Advanced readability analysis using AI/ML"""
    logger.info(f"Advanced readability analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = await analyze_readability_advanced(input_data.content)
        logger.info(f"Advanced readability analysis completed - Grade level: {result.get('grade_level', 0):.1f}")
        return ReadabilityResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Advanced readability analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/comprehensive", response_model=ComprehensiveAnalysisResult)
async def comprehensive_analysis_endpoint(input_data: ContentInput) -> ComprehensiveAnalysisResult:
    """Perform comprehensive analysis combining all AI/ML features"""
    logger.info(f"Comprehensive analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = await comprehensive_analysis(input_data.content)
        logger.info(f"Comprehensive analysis completed - Hash: {result.get('text_hash', 'unknown')}")
        return ComprehensiveAnalysisResult(**result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ai/batch", response_model=BatchAnalysisResult)
async def batch_analyze_content_endpoint(input_data: BatchAnalysisInput) -> BatchAnalysisResult:
    """Analyze multiple texts in batch for efficiency"""
    logger.info(f"Batch analysis requested - Texts: {len(input_data.texts)}")
    
    try:
        results = await batch_analyze_content(input_data.texts)
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        logger.info(f"Batch analysis completed - Successful: {successful}, Failed: {failed}")
        
        return BatchAnalysisResult(
            results=results,
            total_processed=len(input_data.texts),
            successful_analyses=successful,
            failed_analyses=failed,
            timestamp=time.time()
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Real-time Analysis Endpoints
# ============================================================================

@router.websocket("/ws/realtime/{session_id}")
async def websocket_realtime_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time analysis"""
    await websocket_endpoint(websocket, session_id)


@router.post("/realtime/start")
async def start_realtime_analysis(request: Dict[str, Any]):
    """Start real-time analysis session"""
    logger.info("Starting real-time analysis session")
    
    try:
        from realtime_analysis import RealTimeAnalysisRequest
        analysis_request = RealTimeAnalysisRequest(**request)
        session_id = await realtime_engine.start_analysis_session(analysis_request)
        
        return {
            "session_id": session_id,
            "status": "started",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error starting real-time analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/realtime/stop/{session_id}")
async def stop_realtime_analysis(session_id: str):
    """Stop real-time analysis session"""
    logger.info(f"Stopping real-time analysis session: {session_id}")
    
    try:
        stopped = await realtime_engine.stop_analysis_session(session_id)
        
        return {
            "session_id": session_id,
            "stopped": stopped,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error stopping real-time analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/realtime/sessions")
async def get_realtime_sessions():
    """Get active real-time analysis sessions"""
    logger.info("Getting active real-time sessions")
    
    try:
        sessions = await realtime_engine.get_active_sessions()
        
        return {
            "sessions": sessions,
            "count": len(sessions),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting real-time sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Multimodal Analysis Endpoints
# ============================================================================

@router.post("/multimodal/analyze")
async def analyze_multimodal_content(input_data: MultimodalInput):
    """Analyze multimodal content (text, image, audio, video)"""
    logger.info(f"Multimodal analysis requested - Type: {input_data.content_type}")
    
    try:
        result = await multimodal_engine.analyze_multimodal(input_data)
        
        return {
            "content_type": result.content_type,
            "content_hash": result.content_hash,
            "analysis_results": result.analysis_results,
            "cross_modal_insights": result.cross_modal_insights,
            "confidence_scores": result.confidence_scores,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multimodal/image")
async def analyze_image_content(input_data: Dict[str, Any]):
    """Analyze image content specifically"""
    logger.info("Image analysis requested")
    
    try:
        from multimodal_analysis import ImageAnalysisResult
        
        image_data = input_data.get("image_data")
        analysis_types = input_data.get("analysis_types", ["all"])
        
        if not image_data:
            raise ValueError("image_data is required")
        
        result = await multimodal_engine.analyze_image(image_data, analysis_types)
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multimodal/audio")
async def analyze_audio_content(input_data: Dict[str, Any]):
    """Analyze audio content specifically"""
    logger.info("Audio analysis requested")
    
    try:
        from multimodal_analysis import AudioAnalysisResult
        
        audio_data = input_data.get("audio_data")
        analysis_types = input_data.get("analysis_types", ["all"])
        
        if not audio_data:
            raise ValueError("audio_data is required")
        
        result = await multimodal_engine.analyze_audio(audio_data, analysis_types)
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error in audio analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multimodal/video")
async def analyze_video_content(input_data: Dict[str, Any]):
    """Analyze video content specifically"""
    logger.info("Video analysis requested")
    
    try:
        from multimodal_analysis import VideoAnalysisResult
        
        video_data = input_data.get("video_data")
        analysis_types = input_data.get("analysis_types", ["all"])
        
        if not video_data:
            raise ValueError("video_data is required")
        
        result = await multimodal_engine.analyze_video(video_data, analysis_types)
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error in video analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Custom Model Training Endpoints
# ============================================================================

@router.post("/training/create-job")
async def create_training_job(dataset: TrainingDataset, config: TrainingConfig):
    """Create a new model training job"""
    logger.info(f"Creating training job for dataset: {dataset.name}")
    
    try:
        job_id = await custom_training_engine.create_training_job(dataset, config)
        
        return {
            "job_id": job_id,
            "status": "created",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/training/start/{job_id}")
async def start_training_job(job_id: str):
    """Start a training job"""
    logger.info(f"Starting training job: {job_id}")
    
    try:
        started = await custom_training_engine.start_training(job_id)
        
        return {
            "job_id": job_id,
            "started": started,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error starting training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/training/jobs")
async def list_training_jobs():
    """List all training jobs"""
    logger.info("Listing training jobs")
    
    try:
        jobs = await custom_training_engine.list_training_jobs()
        
        return {
            "jobs": [job.model_dump() for job in jobs],
            "count": len(jobs),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/training/jobs/{job_id}")
async def get_training_job_status(job_id: str):
    """Get training job status"""
    logger.info(f"Getting training job status: {job_id}")
    
    try:
        job = await custom_training_engine.get_training_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        return job.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/training/deploy/{job_id}")
async def deploy_trained_model(job_id: str, model_name: str):
    """Deploy a trained model"""
    logger.info(f"Deploying model from job: {job_id}")
    
    try:
        deployed = await custom_training_engine.deploy_model(job_id, model_name)
        
        return {
            "job_id": job_id,
            "model_name": model_name,
            "deployed": deployed,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/training/models")
async def list_deployed_models():
    """List deployed models"""
    logger.info("Listing deployed models")
    
    try:
        models = await custom_training_engine.list_deployed_models()
        
        return {
            "models": models,
            "count": len(models),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error listing deployed models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/training/predict/{model_name}")
async def predict_with_custom_model(model_name: str, text: str):
    """Make prediction with custom trained model"""
    logger.info(f"Making prediction with model: {model_name}")
    
    try:
        result = await custom_training_engine.predict_with_custom_model(model_name, text)
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Advanced Analytics Dashboard Endpoints
# ============================================================================

@router.post("/analytics/query")
async def execute_analytics_query(query: AnalyticsQuery):
    """Execute analytics query"""
    logger.info(f"Executing analytics query: {query.query_type}")
    
    try:
        result = await analytics_dashboard.execute_analytics_query(query)
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error executing analytics query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analytics/dashboards")
async def create_dashboard(name: str, description: str, created_by: str, is_public: bool = False):
    """Create analytics dashboard"""
    logger.info(f"Creating dashboard: {name}")
    
    try:
        dashboard_id = await analytics_dashboard.create_dashboard(name, description, created_by, is_public)
        
        return {
            "dashboard_id": dashboard_id,
            "name": name,
            "status": "created",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/dashboards")
async def list_dashboards():
    """List all dashboards"""
    logger.info("Listing dashboards")
    
    try:
        dashboards = await analytics_dashboard.list_dashboards()
        
        return {
            "dashboards": [dashboard.model_dump() for dashboard in dashboards],
            "count": len(dashboards),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error listing dashboards: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    """Get dashboard by ID"""
    logger.info(f"Getting dashboard: {dashboard_id}")
    
    try:
        dashboard = await analytics_dashboard.get_dashboard(dashboard_id)
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return dashboard.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/dashboards/{dashboard_id}/html")
async def get_dashboard_html(dashboard_id: str):
    """Get dashboard as HTML"""
    logger.info(f"Getting dashboard HTML: {dashboard_id}")
    
    try:
        html = await analytics_dashboard.generate_dashboard_html(dashboard_id)
        
        return {
            "dashboard_id": dashboard_id,
            "html": html,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error generating dashboard HTML: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analytics/reports")
async def create_report(name: str, description: str, query: AnalyticsQuery, 
                       format: str = "pdf", schedule: str = None, recipients: List[str] = None):
    """Create analytics report"""
    logger.info(f"Creating report: {name}")
    
    try:
        report_id = await analytics_dashboard.create_report(name, description, query, format, schedule, recipients)
        
        return {
            "report_id": report_id,
            "name": name,
            "status": "created",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error creating report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/reports")
async def list_reports():
    """List all reports"""
    logger.info("Listing reports")
    
    try:
        reports = await analytics_dashboard.list_reports()
        
        return {
            "reports": [report.model_dump() for report in reports],
            "count": len(reports),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/reports/{report_id}/generate")
async def generate_report(report_id: str):
    """Generate report"""
    logger.info(f"Generating report: {report_id}")
    
    try:
        report_data = await analytics_dashboard.generate_report(report_id)
        
        return {
            "report_id": report_id,
            "data": base64.b64encode(report_data).decode(),
            "size": len(report_data),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
