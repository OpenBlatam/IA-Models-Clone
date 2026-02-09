"""
REST API Endpoints for AI History Analyzer and Model Comparison
=============================================================

This module provides comprehensive REST API endpoints for the AI history analyzer,
enabling external access to performance tracking, model comparison, and analysis features.

Features:
- Performance data recording and retrieval
- Model comparison endpoints
- Trend analysis and forecasting
- Comprehensive reporting
- Configuration management
- Real-time monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
from datetime import datetime, timedelta
import json

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric, 
    get_ai_history_analyzer
)
from .config import (
    AIHistoryConfig, ModelProvider, ModelCategory,
    get_ai_history_config
)

logger = logging.getLogger(__name__)

# Import NLP endpoints
try:
    from .nlp_endpoints import router as nlp_router
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLP endpoints not available - nlp_endpoints module not found")

# Initialize FastAPI app
app = FastAPI(
    title="AI History Analyzer API",
    description="Comprehensive API for AI model performance tracking and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include NLP router if available
if NLP_AVAILABLE:
    app.include_router(nlp_router)
    logger.info("NLP endpoints included successfully")
else:
    logger.warning("NLP endpoints not included - module not available")

# Security
security = HTTPBearer(auto_error=False)

# Global instances
analyzer: Optional[AIHistoryAnalyzer] = None
config: Optional[AIHistoryConfig] = None


# Pydantic Models
class PerformanceRecord(BaseModel):
    model_name: str = Field(..., description="Name of the AI model")
    model_type: str = Field(..., description="Type of the model")
    metric: str = Field(..., description="Performance metric name")
    value: float = Field(..., description="Performance value")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelComparisonRequest(BaseModel):
    model_a: str = Field(..., description="First model to compare")
    model_b: str = Field(..., description="Second model to compare")
    metric: str = Field(..., description="Metric to compare on")
    days: int = Field(30, description="Number of days to analyze")


class TrendAnalysisRequest(BaseModel):
    model_name: str = Field(..., description="Model to analyze")
    metric: str = Field(..., description="Metric to analyze")
    days: int = Field(90, description="Number of days to analyze")


class PerformanceSummaryRequest(BaseModel):
    model_name: str = Field(..., description="Model to summarize")
    days: int = Field(30, description="Number of days to analyze")


class ComprehensiveReportRequest(BaseModel):
    days: int = Field(30, description="Number of days to analyze")
    include_forecasts: bool = Field(True, description="Include forecast data")
    include_anomalies: bool = Field(True, description="Include anomaly detection")


class ModelRankingRequest(BaseModel):
    metric: str = Field(..., description="Metric to rank by")
    days: int = Field(30, description="Number of days to analyze")
    limit: int = Field(10, description="Maximum number of results")


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Dependency functions
async def get_analyzer() -> AIHistoryAnalyzer:
    """Get or initialize analyzer"""
    global analyzer
    if analyzer is None:
        analyzer = get_ai_history_analyzer()
    return analyzer


async def get_config() -> AIHistoryConfig:
    """Get or initialize configuration"""
    global config
    if config is None:
        config = get_ai_history_config()
    return config


async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication (implement your auth logic here)"""
    # For now, we'll accept any token or no token
    # In production, implement proper JWT or API key validation
    return True


# Health and Status Endpoints
@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    try:
        analyzer = await get_analyzer()
        config = await get_config()
        
        return APIResponse(
            success=True,
            message="Service is healthy",
            data={
                "status": "healthy",
                "analyzer_initialized": analyzer is not None,
                "config_loaded": config is not None,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/status", response_model=APIResponse)
async def get_system_status(auth: bool = Depends(verify_auth)):
    """Get comprehensive system status"""
    try:
        analyzer = await get_analyzer()
        config = await get_config()
        
        # Get analyzer stats
        stats = analyzer.performance_stats
        
        # Get configuration summary
        config_summary = config.get_configuration_summary()
        
        return APIResponse(
            success=True,
            message="System status retrieved",
            data={
                "analyzer_stats": {
                    "total_measurements": stats["total_measurements"],
                    "models_tracked": len(stats["models_tracked"]),
                    "metrics_tracked": len(stats["metrics_tracked"]),
                    "last_analysis": stats["last_analysis"]
                },
                "configuration": config_summary,
                "system_uptime": "N/A"  # Implement uptime tracking
            }
        )
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Data Endpoints
@app.post("/performance/record", response_model=APIResponse)
async def record_performance(
    record: PerformanceRecord,
    background_tasks: BackgroundTasks,
    auth: bool = Depends(verify_auth)
):
    """Record performance data for a model"""
    try:
        analyzer = await get_analyzer()
        
        # Validate model type and metric
        try:
            model_type = ModelType(record.model_type)
            metric = PerformanceMetric(record.metric)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid model type or metric: {str(e)}")
        
        # Record performance
        success = analyzer.record_performance(
            model_name=record.model_name,
            model_type=model_type,
            metric=metric,
            value=record.value,
            context=record.context,
            metadata=record.metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record performance data")
        
        # Add background task for analysis
        background_tasks.add_task(log_performance_record, record.model_name, record.metric)
        
        return APIResponse(
            success=True,
            message="Performance data recorded successfully",
            data={
                "model_name": record.model_name,
                "metric": record.metric,
                "value": record.value,
                "timestamp": datetime.now().isoformat()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/{model_name}/{metric}", response_model=APIResponse)
async def get_performance_data(
    model_name: str = Path(..., description="Model name"),
    metric: str = Path(..., description="Metric name"),
    days: int = Query(30, description="Number of days to retrieve"),
    auth: bool = Depends(verify_auth)
):
    """Get performance data for a specific model and metric"""
    try:
        analyzer = await get_analyzer()
        
        # Validate metric
        try:
            metric_enum = PerformanceMetric(metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {metric}")
        
        # Get performance data
        performance_data = analyzer.get_model_performance(model_name, metric_enum, days)
        
        # Convert to serializable format
        data = []
        for p in performance_data:
            data.append({
                "timestamp": p.timestamp.isoformat(),
                "value": p.value,
                "context": p.context,
                "metadata": p.metadata
            })
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(data)} performance records",
            data={
                "model_name": model_name,
                "metric": metric,
                "days": days,
                "records": data,
                "summary": {
                    "count": len(data),
                    "latest_value": data[-1]["value"] if data else None,
                    "date_range": {
                        "start": data[0]["timestamp"] if data else None,
                        "end": data[-1]["timestamp"] if data else None
                    }
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Comparison Endpoints
@app.post("/comparison/compare", response_model=APIResponse)
async def compare_models(
    request: ModelComparisonRequest,
    auth: bool = Depends(verify_auth)
):
    """Compare two models on a specific metric"""
    try:
        analyzer = await get_analyzer()
        
        # Validate metric
        try:
            metric_enum = PerformanceMetric(request.metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {request.metric}")
        
        # Perform comparison
        comparison = analyzer.compare_models(
            model_a=request.model_a,
            model_b=request.model_b,
            metric=metric_enum,
            days=request.days
        )
        
        if not comparison:
            raise HTTPException(status_code=404, detail="Insufficient data for comparison")
        
        return APIResponse(
            success=True,
            message="Model comparison completed",
            data={
                "model_a": comparison.model_a,
                "model_b": comparison.model_b,
                "metric": comparison.metric.value,
                "comparison_score": comparison.comparison_score,
                "confidence": comparison.confidence,
                "sample_size": comparison.sample_size,
                "timestamp": comparison.timestamp.isoformat(),
                "details": comparison.details,
                "interpretation": {
                    "winner": comparison.model_a if comparison.comparison_score > 0 else comparison.model_b,
                    "margin": abs(comparison.comparison_score),
                    "confidence_level": "high" if comparison.confidence > 0.8 else "medium" if comparison.confidence > 0.5 else "low"
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Trend Analysis Endpoints
@app.post("/trends/analyze", response_model=APIResponse)
async def analyze_trends(
    request: TrendAnalysisRequest,
    auth: bool = Depends(verify_auth)
):
    """Analyze performance trends for a model"""
    try:
        analyzer = await get_analyzer()
        
        # Validate metric
        try:
            metric_enum = PerformanceMetric(request.metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {request.metric}")
        
        # Perform trend analysis
        trend_analysis = analyzer.analyze_trends(
            model_name=request.model_name,
            metric=metric_enum,
            days=request.days
        )
        
        if not trend_analysis:
            raise HTTPException(status_code=404, detail="Insufficient data for trend analysis")
        
        return APIResponse(
            success=True,
            message="Trend analysis completed",
            data={
                "model_name": trend_analysis.model_name,
                "metric": trend_analysis.metric.value,
                "trend_direction": trend_analysis.trend_direction,
                "trend_strength": trend_analysis.trend_strength,
                "confidence": trend_analysis.confidence,
                "forecast": [
                    {"date": date.isoformat(), "predicted_value": value}
                    for date, value in trend_analysis.forecast
                ],
                "anomalies": [
                    {"date": date.isoformat(), "value": value}
                    for date, value in trend_analysis.anomalies
                ],
                "interpretation": {
                    "trend_description": f"{trend_analysis.trend_direction} with {trend_analysis.trend_strength:.1%} strength",
                    "confidence_level": "high" if trend_analysis.confidence > 0.8 else "medium" if trend_analysis.confidence > 0.5 else "low",
                    "anomaly_count": len(trend_analysis.anomalies),
                    "forecast_days": len(trend_analysis.forecast)
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Summary Endpoints
@app.post("/summary/model", response_model=APIResponse)
async def get_model_summary(
    request: PerformanceSummaryRequest,
    auth: bool = Depends(verify_auth)
):
    """Get comprehensive performance summary for a model"""
    try:
        analyzer = await get_analyzer()
        
        # Get performance summary
        summary = analyzer.get_performance_summary(request.model_name, request.days)
        
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return APIResponse(
            success=True,
            message="Model performance summary retrieved",
            data=summary
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Ranking Endpoints
@app.post("/rankings/models", response_model=APIResponse)
async def get_model_rankings(
    request: ModelRankingRequest,
    auth: bool = Depends(verify_auth)
):
    """Get rankings of models for a specific metric"""
    try:
        analyzer = await get_analyzer()
        
        # Validate metric
        try:
            metric_enum = PerformanceMetric(request.metric)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {request.metric}")
        
        # Get rankings
        rankings = analyzer.get_model_rankings(metric_enum, request.days)
        
        # Limit results
        rankings = rankings[:request.limit]
        
        return APIResponse(
            success=True,
            message=f"Retrieved rankings for {request.metric}",
            data={
                "metric": request.metric,
                "days": request.days,
                "rankings": rankings,
                "total_models": len(rankings),
                "top_performer": rankings[0] if rankings else None
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model rankings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Comprehensive Report Endpoints
@app.post("/reports/comprehensive", response_model=APIResponse)
async def get_comprehensive_report(
    request: ComprehensiveReportRequest,
    background_tasks: BackgroundTasks,
    auth: bool = Depends(verify_auth)
):
    """Generate comprehensive analysis report"""
    try:
        analyzer = await get_analyzer()
        
        # Generate comprehensive report
        report = analyzer.get_comprehensive_report(request.days)
        
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])
        
        # Add background task for report processing
        background_tasks.add_task(process_comprehensive_report, report)
        
        return APIResponse(
            success=True,
            message="Comprehensive report generated",
            data=report
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration Endpoints
@app.get("/config/models", response_model=APIResponse)
async def get_models_config(auth: bool = Depends(verify_auth)):
    """Get model configuration"""
    try:
        config = await get_config()
        
        models_data = []
        for name, model in config.models.items():
            models_data.append({
                "name": model.name,
                "provider": model.provider.value,
                "category": model.category.value,
                "version": model.version,
                "context_length": model.context_length,
                "parameters": model.parameters,
                "release_date": model.release_date,
                "description": model.description,
                "capabilities": model.capabilities,
                "limitations": model.limitations,
                "cost_per_1k_tokens": model.cost_per_1k_tokens,
                "max_requests_per_minute": model.max_requests_per_minute,
                "is_active": model.is_active
            })
        
        return APIResponse(
            success=True,
            message="Model configuration retrieved",
            data={
                "models": models_data,
                "total_models": len(models_data),
                "active_models": len([m for m in models_data if m["is_active"]])
            }
        )
    except Exception as e:
        logger.error(f"Failed to get models config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/metrics", response_model=APIResponse)
async def get_metrics_config(auth: bool = Depends(verify_auth)):
    """Get metrics configuration"""
    try:
        config = await get_config()
        
        metrics_data = []
        for name, metric in config.metrics.items():
            metrics_data.append({
                "name": metric.name,
                "description": metric.description,
                "unit": metric.unit,
                "min_value": metric.min_value,
                "max_value": metric.max_value,
                "optimal_range": metric.optimal_range,
                "weight": metric.weight,
                "higher_is_better": metric.higher_is_better,
                "calculation_method": metric.calculation_method,
                "alert_thresholds": metric.alert_thresholds
            })
        
        return APIResponse(
            success=True,
            message="Metrics configuration retrieved",
            data={
                "metrics": metrics_data,
                "total_metrics": len(metrics_data),
                "weighted_metrics": len([m for m in metrics_data if m["weight"] > 0])
            }
        )
    except Exception as e:
        logger.error(f"Failed to get metrics config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/summary", response_model=APIResponse)
async def get_config_summary(auth: bool = Depends(verify_auth)):
    """Get configuration summary"""
    try:
        config = await get_config()
        summary = config.get_configuration_summary()
        
        return APIResponse(
            success=True,
            message="Configuration summary retrieved",
            data=summary
        )
    except Exception as e:
        logger.error(f"Failed to get config summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Export Endpoints
@app.get("/export/data", response_model=APIResponse)
async def export_data(
    format: str = Query("json", description="Export format"),
    auth: bool = Depends(verify_auth)
):
    """Export all analysis data"""
    try:
        analyzer = await get_analyzer()
        
        # Export data
        export_data = analyzer.export_data(format)
        
        if isinstance(export_data, dict) and "error" in export_data:
            raise HTTPException(status_code=500, detail=export_data["error"])
        
        return APIResponse(
            success=True,
            message=f"Data exported in {format} format",
            data={
                "format": format,
                "export_size": len(str(export_data)),
                "export_timestamp": datetime.now().isoformat()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def log_performance_record(model_name: str, metric: str):
    """Background task to log performance record"""
    logger.info(f"Performance recorded: {model_name} - {metric}")


async def process_comprehensive_report(report: Dict[str, Any]):
    """Background task to process comprehensive report"""
    logger.info(f"Processing comprehensive report with {len(report.get('model_performances', {}))} models")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI History Analyzer API")
    
    # Initialize analyzer and config
    await get_analyzer()
    await get_config()
    
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI History Analyzer API")
    logger.info("API shutdown completed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)