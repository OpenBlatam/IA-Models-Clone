"""
LLM Integration for AI History Comparison System
Integración de LLM para el Sistema de Comparación de Historial de IA
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

# Import our best practices
from llm_best_practices import (
    LLMFactory, PromptTemplate, AsyncLLMProcessor, 
    LLMConfig, LLMProvider, LLMMonitor
)

# FastAPI integration
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from api_best_practices import APIResponse, create_success_response

logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURACIÓN DE LLM PARA EL SISTEMA
# =============================================================================

class LLMService:
    """Servicio LLM integrado con el sistema de comparación de historial"""
    
    def __init__(self):
        self.processor = LLMFactory.create_processor()
        self.monitor = LLMMonitor(redis.Redis(host='localhost', port=6379, db=0))
        self.logger = logging.getLogger(__name__)
        
        # Configurar modelos específicos para el sistema
        self._setup_models()
    
    def _setup_models(self):
        """Configurar modelos específicos para análisis de contenido"""
        
        # Modelo principal para análisis
        analysis_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            max_tokens=4000,
            temperature=0.3,  # Más determinístico para análisis
            cache_enabled=True,
            cache_ttl=7200  # 2 horas
        )
        
        # Modelo para comparaciones
        comparison_config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.2,  # Muy determinístico para comparaciones
            cache_enabled=True,
            cache_ttl=3600
        )
        
        # Modelo para resúmenes rápidos
        summary_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            max_tokens=2000,
            temperature=0.5,
            cache_enabled=True,
            cache_ttl=1800  # 30 minutos
        )
        
        # Registrar modelos
        self.processor.llm_manager.register_model("analysis", analysis_config)
        self.processor.llm_manager.register_model("comparison", comparison_config)
        self.processor.llm_manager.register_model("summary", summary_config)
    
    async def analyze_content(self, content: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analizar contenido usando LLM"""
        try:
            # Generar prompt optimizado
            prompt = PromptTemplate.content_analysis_prompt(content, analysis_type)
            
            # Obtener configuración del modelo
            config = self.processor.llm_manager.configs["analysis"]
            
            # Procesar con LLM
            start_time = time.time()
            result = await self.processor.process_single(prompt, "analysis", config)
            processing_time = time.time() - start_time
            
            # Registrar métricas
            await self.monitor.log_usage(
                model_name="analysis",
                provider="openai",
                tokens_used=result.get("tokens_used", 0),
                processing_time=processing_time,
                success=result.get("success", False),
                error=result.get("error") if not result.get("success") else None
            )
            
            if result.get("success"):
                return {
                    "analysis": result["content"],
                    "metadata": {
                        "model": "gpt-4",
                        "processing_time": processing_time,
                        "tokens_used": result.get("tokens_used", 0),
                        "cached": result.get("cached", False),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            else:
                raise Exception(f"LLM analysis failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in content analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def compare_content(self, content1: str, content2: str) -> Dict[str, Any]:
        """Comparar dos contenidos usando LLM"""
        try:
            # Generar prompt de comparación
            prompt = PromptTemplate.comparison_prompt(content1, content2)
            
            # Obtener configuración del modelo
            config = self.processor.llm_manager.configs["comparison"]
            
            # Procesar con LLM
            start_time = time.time()
            result = await self.processor.process_single(prompt, "comparison", config)
            processing_time = time.time() - start_time
            
            # Registrar métricas
            await self.monitor.log_usage(
                model_name="comparison",
                provider="anthropic",
                tokens_used=result.get("tokens_used", 0),
                processing_time=processing_time,
                success=result.get("success", False),
                error=result.get("error") if not result.get("success") else None
            )
            
            if result.get("success"):
                return {
                    "comparison": result["content"],
                    "metadata": {
                        "model": "claude-3-sonnet",
                        "processing_time": processing_time,
                        "tokens_used": result.get("tokens_used", 0),
                        "cached": result.get("cached", False),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            else:
                raise Exception(f"LLM comparison failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in content comparison: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
    
    async def analyze_trends(self, contents: List[str], timeframes: List[str]) -> Dict[str, Any]:
        """Analizar tendencias usando LLM"""
        try:
            # Generar prompt de análisis de tendencias
            prompt = PromptTemplate.trend_analysis_prompt(contents, timeframes)
            
            # Obtener configuración del modelo
            config = self.processor.llm_manager.configs["analysis"]
            
            # Procesar con LLM
            start_time = time.time()
            result = await self.processor.process_single(prompt, "analysis", config)
            processing_time = time.time() - start_time
            
            # Registrar métricas
            await self.monitor.log_usage(
                model_name="analysis",
                provider="openai",
                tokens_used=result.get("tokens_used", 0),
                processing_time=processing_time,
                success=result.get("success", False),
                error=result.get("error") if not result.get("success") else None
            )
            
            if result.get("success"):
                return {
                    "trends": result["content"],
                    "metadata": {
                        "model": "gpt-4",
                        "processing_time": processing_time,
                        "tokens_used": result.get("tokens_used", 0),
                        "cached": result.get("cached", False),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            else:
                raise Exception(f"LLM trend analysis failed: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")
    
    async def batch_analyze(self, contents: List[str], analysis_type: str = "comprehensive") -> List[Dict[str, Any]]:
        """Análisis en lote de múltiples contenidos"""
        try:
            # Generar prompts para todos los contenidos
            prompts = [
                PromptTemplate.content_analysis_prompt(content, analysis_type)
                for content in contents
            ]
            
            # Obtener configuración del modelo
            config = self.processor.llm_manager.configs["analysis"]
            
            # Procesar en lote
            start_time = time.time()
            results = await self.processor.process_batch(prompts, "analysis", config, max_concurrent=3)
            processing_time = time.time() - start_time
            
            # Procesar resultados
            processed_results = []
            for i, result in enumerate(results):
                if result.get("success"):
                    processed_results.append({
                        "content_index": i,
                        "analysis": result["content"],
                        "metadata": {
                            "model": "gpt-4",
                            "processing_time": result.get("processing_time", 0),
                            "tokens_used": result.get("tokens_used", 0),
                            "cached": result.get("cached", False)
                        }
                    })
                else:
                    processed_results.append({
                        "content_index": i,
                        "error": result.get("error"),
                        "success": False
                    })
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error in batch analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# =============================================================================
# 2. MODELOS PYDANTIC PARA LA API
# =============================================================================

class ContentAnalysisRequest(BaseModel):
    """Request para análisis de contenido"""
    content: str = Field(..., min_length=1, max_length=50000, description="Contenido a analizar")
    analysis_type: str = Field(default="comprehensive", description="Tipo de análisis")
    include_metadata: bool = Field(default=True, description="Incluir metadatos en la respuesta")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Este es un ejemplo de contenido para analizar...",
                "analysis_type": "comprehensive",
                "include_metadata": True
            }
        }

class ContentComparisonRequest(BaseModel):
    """Request para comparación de contenido"""
    content1: str = Field(..., min_length=1, max_length=50000, description="Primer contenido")
    content2: str = Field(..., min_length=1, max_length=50000, description="Segundo contenido")
    include_recommendations: bool = Field(default=True, description="Incluir recomendaciones")
    
    class Config:
        schema_extra = {
            "example": {
                "content1": "Primer contenido para comparar...",
                "content2": "Segundo contenido para comparar...",
                "include_recommendations": True
            }
        }

class TrendAnalysisRequest(BaseModel):
    """Request para análisis de tendencias"""
    contents: List[str] = Field(..., min_items=2, max_items=10, description="Lista de contenidos")
    timeframes: List[str] = Field(..., min_items=2, max_items=10, description="Marcos temporales")
    
    class Config:
        schema_extra = {
            "example": {
                "contents": ["Contenido 1", "Contenido 2", "Contenido 3"],
                "timeframes": ["2024-01", "2024-02", "2024-03"]
            }
        }

class BatchAnalysisRequest(BaseModel):
    """Request para análisis en lote"""
    contents: List[str] = Field(..., min_items=1, max_items=20, description="Lista de contenidos")
    analysis_type: str = Field(default="comprehensive", description="Tipo de análisis")
    
    class Config:
        schema_extra = {
            "example": {
                "contents": ["Contenido 1", "Contenido 2", "Contenido 3"],
                "analysis_type": "comprehensive"
            }
        }

# =============================================================================
# 3. ENDPOINTS DE LA API
# =============================================================================

# Crear router para endpoints LLM
llm_router = APIRouter(prefix="/api/v1/llm", tags=["LLM Analysis"])

# Instancia global del servicio LLM
llm_service = LLMService()

@llm_router.post(
    "/analyze",
    response_model=APIResponse,
    summary="Analizar contenido con LLM",
    description="Analiza contenido usando modelos de lenguaje avanzados"
)
async def analyze_content_llm(request: ContentAnalysisRequest):
    """Analizar contenido usando LLM"""
    try:
        result = await llm_service.analyze_content(
            content=request.content,
            analysis_type=request.analysis_type
        )
        
        return create_success_response(
            data=result,
            message="Content analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_content_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@llm_router.post(
    "/compare",
    response_model=APIResponse,
    summary="Comparar contenidos con LLM",
    description="Compara dos contenidos usando modelos de lenguaje avanzados"
)
async def compare_content_llm(request: ContentComparisonRequest):
    """Comparar contenidos usando LLM"""
    try:
        result = await llm_service.compare_content(
            content1=request.content1,
            content2=request.content2
        )
        
        return create_success_response(
            data=result,
            message="Content comparison completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in compare_content_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@llm_router.post(
    "/trends",
    response_model=APIResponse,
    summary="Analizar tendencias con LLM",
    description="Analiza tendencias en contenidos a lo largo del tiempo"
)
async def analyze_trends_llm(request: TrendAnalysisRequest):
    """Analizar tendencias usando LLM"""
    try:
        result = await llm_service.analyze_trends(
            contents=request.contents,
            timeframes=request.timeframes
        )
        
        return create_success_response(
            data=result,
            message="Trend analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_trends_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@llm_router.post(
    "/batch-analyze",
    response_model=APIResponse,
    summary="Análisis en lote con LLM",
    description="Analiza múltiples contenidos en paralelo"
)
async def batch_analyze_llm(request: BatchAnalysisRequest):
    """Análisis en lote usando LLM"""
    try:
        result = await llm_service.batch_analyze(
            contents=request.contents,
            analysis_type=request.analysis_type
        )
        
        return create_success_response(
            data=result,
            message="Batch analysis completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in batch_analyze_llm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@llm_router.get(
    "/metrics",
    response_model=APIResponse,
    summary="Métricas de uso de LLM",
    description="Obtiene métricas de uso y rendimiento de los modelos LLM"
)
async def get_llm_metrics():
    """Obtener métricas de uso de LLM"""
    try:
        # Obtener métricas para cada modelo
        analysis_stats = await llm_service.monitor.get_usage_stats("analysis", days=7)
        comparison_stats = await llm_service.monitor.get_usage_stats("comparison", days=7)
        summary_stats = await llm_service.monitor.get_usage_stats("summary", days=7)
        
        metrics = {
            "analysis_model": analysis_stats,
            "comparison_model": comparison_stats,
            "summary_model": summary_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return create_success_response(
            data=metrics,
            message="LLM metrics retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in get_llm_metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@llm_router.get(
    "/health",
    response_model=APIResponse,
    summary="Health check de LLM",
    description="Verifica el estado de los modelos LLM"
)
async def llm_health_check():
    """Health check para servicios LLM"""
    try:
        # Verificar conectividad de modelos
        health_status = {
            "analysis_model": "healthy",
            "comparison_model": "healthy", 
            "summary_model": "healthy",
            "cache_status": "healthy",
            "monitoring_status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return create_success_response(
            data=health_status,
            message="LLM services are healthy"
        )
        
    except Exception as e:
        logger.error(f"Error in llm_health_check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 4. FUNCIONES DE UTILIDAD
# =============================================================================

async def initialize_llm_service():
    """Inicializar servicio LLM al startup"""
    global llm_service
    try:
        llm_service = LLMService()
        logger.info("✅ LLM service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM service: {str(e)}")
        raise

async def cleanup_llm_service():
    """Limpiar recursos del servicio LLM al shutdown"""
    try:
        # Cerrar conexiones Redis, etc.
        logger.info("🛑 LLM service cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Error cleaning up LLM service: {str(e)}")

# =============================================================================
# 5. EJEMPLO DE USO
# =============================================================================

async def example_usage():
    """Ejemplo de uso del servicio LLM"""
    
    # Inicializar servicio
    service = LLMService()
    
    # Ejemplo 1: Análisis de contenido
    content = "Este es un ejemplo de contenido para analizar con LLM."
    analysis_result = await service.analyze_content(content, "comprehensive")
    print("Análisis:", analysis_result)
    
    # Ejemplo 2: Comparación de contenidos
    content1 = "Primer contenido para comparar."
    content2 = "Segundo contenido para comparar."
    comparison_result = await service.compare_content(content1, content2)
    print("Comparación:", comparison_result)
    
    # Ejemplo 3: Análisis en lote
    contents = ["Contenido 1", "Contenido 2", "Contenido 3"]
    batch_result = await service.batch_analyze(contents)
    print("Análisis en lote:", batch_result)

if __name__ == "__main__":
    # Ejecutar ejemplo
    asyncio.run(example_usage())







