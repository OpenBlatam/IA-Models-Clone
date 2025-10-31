"""
Advanced Endpoints for AI History Comparison System
Endpoints avanzados para el sistema de análisis de historial de IA
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for advanced endpoints
router = APIRouter(prefix="/advanced", tags=["Advanced Analytics"])

# ============================================================================
# ADVANCED ANALYTICS ENDPOINTS
# ============================================================================

@router.post("/analytics")
async def perform_advanced_analytics(
    analysis_types: List[str],
    documents: Optional[List[Dict[str, Any]]] = None,
    **kwargs
):
    """Realizar análisis avanzado de documentos"""
    try:
        from advanced_analytics import AdvancedAnalyticsEngine, AnalysisType
        
        # Inicializar motor de análisis avanzado
        analytics_engine = AdvancedAnalyticsEngine()
        
        # Obtener documentos si no se proporcionan
        if documents is None:
            # Importar analyzer desde el módulo principal
            from api_endpoints import get_analyzer
            analyzer = await get_analyzer()
            documents = []
            for doc in analyzer.document_history.values():
                documents.append({
                    "id": doc.id,
                    "content": doc.content,
                    "query": doc.query,
                    "quality_score": doc.quality_score,
                    "readability_score": doc.readability_score,
                    "originality_score": doc.originality_score,
                    "word_count": doc.word_count,
                    "timestamp": doc.timestamp.isoformat(),
                    "metadata": doc.metadata
                })
        
        # Convertir tipos de análisis
        analysis_type_enums = [AnalysisType(at) for at in analysis_types]
        
        # Realizar análisis
        results = await analytics_engine.analyze_documents(documents, analysis_type_enums, **kwargs)
        
        return {
            "success": True,
            "analysis_results": results,
            "documents_analyzed": len(documents),
            "analysis_types": analysis_types,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in advanced analytics: {e}")
        return {"error": str(e)}

@router.get("/analytics/clustering")
async def get_document_clusters(
    method: str = "kmeans",
    n_clusters: Optional[int] = None
):
    """Obtener clusters de documentos"""
    try:
        from advanced_analytics import AdvancedAnalyticsEngine, AnalysisType, ClusteringMethod
        from api_endpoints import get_analyzer
        
        analytics_engine = AdvancedAnalyticsEngine()
        
        # Obtener documentos
        analyzer = await get_analyzer()
        documents = []
        for doc in analyzer.document_history.values():
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        # Realizar clustering
        results = await analytics_engine.analyze_documents(
            documents, 
            [AnalysisType.CLUSTERING],
            method=ClusteringMethod(method),
            n_clusters=n_clusters
        )
        
        return {
            "success": True,
            "clustering_results": results.get("clustering", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        return {"error": str(e)}

@router.get("/analytics/anomalies")
async def detect_anomalies(
    method: str = "isolation_forest"
):
    """Detectar anomalías en documentos"""
    try:
        from advanced_analytics import AdvancedAnalyticsEngine, AnalysisType, AnomalyMethod
        from api_endpoints import get_analyzer
        
        analytics_engine = AdvancedAnalyticsEngine()
        
        # Obtener documentos
        analyzer = await get_analyzer()
        documents = []
        for doc in analyzer.document_history.values():
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        # Detectar anomalías
        results = await analytics_engine.analyze_documents(
            documents, 
            [AnalysisType.ANOMALY_DETECTION],
            method=AnomalyMethod(method)
        )
        
        return {
            "success": True,
            "anomaly_results": results.get("anomaly_detection", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return {"error": str(e)}

@router.get("/analytics/sentiment")
async def analyze_sentiment():
    """Analizar sentimientos de documentos"""
    try:
        from advanced_analytics import AdvancedAnalyticsEngine, AnalysisType
        from api_endpoints import get_analyzer
        
        analytics_engine = AdvancedAnalyticsEngine()
        
        # Obtener documentos
        analyzer = await get_analyzer()
        documents = []
        for doc in analyzer.document_history.values():
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        # Analizar sentimientos
        results = await analytics_engine.analyze_documents(
            documents, 
            [AnalysisType.SENTIMENT_ANALYSIS]
        )
        
        return {
            "success": True,
            "sentiment_results": results.get("sentiment_analysis", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"error": str(e)}

@router.get("/analytics/segmentation")
async def perform_document_segmentation():
    """Realizar segmentación de documentos"""
    try:
        from advanced_analytics import AdvancedAnalyticsEngine, AnalysisType
        from api_endpoints import get_analyzer
        
        analytics_engine = AdvancedAnalyticsEngine()
        
        # Obtener documentos
        analyzer = await get_analyzer()
        documents = []
        for doc in analyzer.document_history.values():
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        # Realizar segmentación
        results = await analytics_engine.analyze_documents(
            documents, 
            [AnalysisType.SEGMENTATION]
        )
        
        return {
            "success": True,
            "segmentation_results": results.get("segmentation", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in document segmentation: {e}")
        return {"error": str(e)}

@router.get("/analytics/trends")
async def analyze_trends():
    """Analizar tendencias en documentos"""
    try:
        from advanced_analytics import AdvancedAnalyticsEngine, AnalysisType
        from api_endpoints import get_analyzer
        
        analytics_engine = AdvancedAnalyticsEngine()
        
        # Obtener documentos
        analyzer = await get_analyzer()
        documents = []
        for doc in analyzer.document_history.values():
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        # Analizar tendencias
        results = await analytics_engine.analyze_documents(
            documents, 
            [AnalysisType.TREND_ANALYSIS]
        )
        
        return {
            "success": True,
            "trend_results": results.get("trend_analysis", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return {"error": str(e)}

# ============================================================================
# AUTOMATED INSIGHTS ENDPOINTS
# ============================================================================

@router.post("/insights/generate")
async def generate_automated_insights():
    """Generar insights automatizados"""
    try:
        from automated_insights import AutomatedInsightsGenerator
        from api_endpoints import get_analyzer
        
        # Inicializar generador de insights
        insights_generator = AutomatedInsightsGenerator()
        
        # Obtener documentos
        analyzer = await get_analyzer()
        documents = []
        for doc in analyzer.document_history.values():
            documents.append({
                "id": doc.id,
                "content": doc.content,
                "query": doc.query,
                "quality_score": doc.quality_score,
                "readability_score": doc.readability_score,
                "originality_score": doc.originality_score,
                "word_count": doc.word_count,
                "timestamp": doc.timestamp.isoformat(),
                "metadata": doc.metadata
            })
        
        # Generar insights
        insights = await insights_generator.generate_insights(documents)
        
        return {
            "success": True,
            "insights": [
                {
                    "id": insight.id,
                    "type": insight.type.value,
                    "category": insight.category.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "recommendations": insight.recommendations,
                    "actionable_items": insight.actionable_items,
                    "tags": insight.tags,
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in insights
            ],
            "total_insights": len(insights),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating automated insights: {e}")
        return {"error": str(e)}

@router.get("/insights")
async def get_automated_insights(
    priority: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50
):
    """Obtener insights automatizados"""
    try:
        from automated_insights import AutomatedInsightsGenerator, InsightPriority, InsightCategory
        
        insights_generator = AutomatedInsightsGenerator()
        
        # Filtrar insights
        all_insights = list(insights_generator.insights.values())
        
        if priority:
            priority_enum = InsightPriority(priority)
            all_insights = [i for i in all_insights if i.priority == priority_enum]
        
        if category:
            category_enum = InsightCategory(category)
            all_insights = [i for i in all_insights if i.category == category_enum]
        
        # Limitar resultados
        all_insights = all_insights[:limit]
        
        return {
            "success": True,
            "insights": [
                {
                    "id": insight.id,
                    "type": insight.type.value,
                    "category": insight.category.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "recommendations": insight.recommendations,
                    "actionable_items": insight.actionable_items,
                    "tags": insight.tags,
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in all_insights
            ],
            "total_insights": len(all_insights),
            "filters": {
                "priority": priority,
                "category": category,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting automated insights: {e}")
        return {"error": str(e)}

@router.get("/insights/summary")
async def get_insights_summary():
    """Obtener resumen de insights automatizados"""
    try:
        from automated_insights import AutomatedInsightsGenerator
        
        insights_generator = AutomatedInsightsGenerator()
        summary = insights_generator.get_insights_summary()
        
        return {
            "success": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting insights summary: {e}")
        return {"error": str(e)}

@router.get("/insights/recent")
async def get_recent_insights(hours: int = 24):
    """Obtener insights recientes"""
    try:
        from automated_insights import AutomatedInsightsGenerator
        
        insights_generator = AutomatedInsightsGenerator()
        recent_insights = insights_generator.get_recent_insights(hours)
        
        return {
            "success": True,
            "recent_insights": [
                {
                    "id": insight.id,
                    "type": insight.type.value,
                    "category": insight.category.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in recent_insights
            ],
            "total_recent": len(recent_insights),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent insights: {e}")
        return {"error": str(e)}

@router.get("/insights/priority/{priority}")
async def get_insights_by_priority(priority: str):
    """Obtener insights por prioridad"""
    try:
        from automated_insights import AutomatedInsightsGenerator, InsightPriority
        
        insights_generator = AutomatedInsightsGenerator()
        priority_enum = InsightPriority(priority)
        insights = insights_generator.get_insights_by_priority(priority_enum)
        
        return {
            "success": True,
            "insights": [
                {
                    "id": insight.id,
                    "type": insight.type.value,
                    "category": insight.category.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in insights
            ],
            "total_insights": len(insights),
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting insights by priority: {e}")
        return {"error": str(e)}

@router.get("/insights/category/{category}")
async def get_insights_by_category(category: str):
    """Obtener insights por categoría"""
    try:
        from automated_insights import AutomatedInsightsGenerator, InsightCategory
        
        insights_generator = AutomatedInsightsGenerator()
        category_enum = InsightCategory(category)
        insights = insights_generator.get_insights_by_category(category_enum)
        
        return {
            "success": True,
            "insights": [
                {
                    "id": insight.id,
                    "type": insight.type.value,
                    "category": insight.category.value,
                    "priority": insight.priority.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in insights
            ],
            "total_insights": len(insights),
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting insights by category: {e}")
        return {"error": str(e)}

# ============================================================================
# PERFORMANCE OPTIMIZATION ENDPOINTS
# ============================================================================

@router.post("/optimization/performance")
async def optimize_performance():
    """Optimizar rendimiento del sistema"""
    try:
        # Implementar optimizaciones de rendimiento
        optimizations = []
        
        # Limpiar cache
        import gc
        gc.collect()
        optimizations.append("Cache limpiado")
        
        # Optimizar base de datos
        from api_endpoints import get_analyzer
        analyzer = await get_analyzer()
        if hasattr(analyzer, 'optimize_database'):
            analyzer.optimize_database()
            optimizations.append("Base de datos optimizada")
        
        # Limpiar insights expirados
        from automated_insights import AutomatedInsightsGenerator
        insights_generator = AutomatedInsightsGenerator()
        expired_count = insights_generator.cleanup_expired_insights()
        optimizations.append(f"Insights expirados limpiados: {expired_count}")
        
        return {
            "success": True,
            "optimizations_applied": optimizations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error optimizing performance: {e}")
        return {"error": str(e)}

@router.get("/optimization/status")
async def get_optimization_status():
    """Obtener estado de optimización del sistema"""
    try:
        import psutil
        import os
        
        # Obtener métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Obtener métricas de la aplicación
        from api_endpoints import get_analyzer
        analyzer = await get_analyzer()
        
        status = {
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "application_metrics": {
                "total_documents": len(analyzer.document_history),
                "total_insights": len(analyzer.insights),
                "total_recommendations": len(analyzer.recommendations)
            },
            "optimization_recommendations": []
        }
        
        # Generar recomendaciones de optimización
        if cpu_percent > 80:
            status["optimization_recommendations"].append("CPU usage is high - consider scaling")
        
        if memory.percent > 80:
            status["optimization_recommendations"].append("Memory usage is high - consider cleanup")
        
        if disk.percent > 90:
            status["optimization_recommendations"].append("Disk space is low - consider cleanup")
        
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        return {"error": str(e)}



























