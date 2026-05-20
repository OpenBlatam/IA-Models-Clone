"""
FastAPI endpoints for Contabilidad Mexicana AI.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from ..core.contador_ai import ContadorAI
from ..config.contador_config import ContadorConfig
from .response_helpers import handle_service_call
from .error_handlers import handle_contador_errors

router = APIRouter(prefix="/api/contador", tags=["Contador AI"])

contador_instance: Optional[ContadorAI] = None


def get_contador() -> ContadorAI:
    """Dependency to get contador instance."""
    global contador_instance
    if contador_instance is None:
        config = ContadorConfig()
        contador_instance = ContadorAI(config)
    return contador_instance


# Request Models
class CalculoImpuestosRequest(BaseModel):
    """Request model for tax calculation."""
    regimen: str = Field(..., description="Régimen fiscal (RESICO, PFAE, etc.)")
    tipo_impuesto: str = Field(..., description="Tipo de impuesto (ISR, IVA, IEPS)")
    datos: Dict[str, Any] = Field(..., description="Datos para el cálculo")


class AsesoriaFiscalRequest(BaseModel):
    """Request model for fiscal advice."""
    pregunta: str = Field(..., description="Pregunta o situación fiscal")
    contexto: Optional[Dict[str, Any]] = Field(None, description="Contexto adicional")


class GuiaFiscalRequest(BaseModel):
    """Request model for fiscal guide."""
    tema: str = Field(..., description="Tema fiscal")
    nivel_detalle: str = Field("completo", description="Nivel de detalle (básico, intermedio, completo)")


class TramiteSATRequest(BaseModel):
    """Request model for SAT procedure information."""
    tipo_tramite: str = Field(..., description="Tipo de trámite del SAT")
    detalles: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales")


class AyudaDeclaracionRequest(BaseModel):
    """Request model for declaration assistance."""
    tipo_declaracion: str = Field(..., description="Tipo de declaración (mensual, anual, etc.)")
    periodo: str = Field(..., description="Período fiscal")
    datos: Optional[Dict[str, Any]] = Field(None, description="Datos del contribuyente")


class CompararRegimenesRequest(BaseModel):
    """Request model for regime comparison."""
    regimenes: List[str] = Field(..., description="Lista de regímenes a comparar", min_items=2)
    datos: Dict[str, Any] = Field(..., description="Datos del contribuyente")


# Endpoints
@router.post("/calcular-impuestos", response_model=Dict[str, Any])
async def calcular_impuestos(
    request: CalculoImpuestosRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Calcular impuestos para un régimen fiscal específico.
    
    Ejemplo:
    ```json
    {
        "regimen": "RESICO",
        "tipo_impuesto": "ISR",
        "datos": {
            "ingresos_mensuales": 50000,
            "gastos_deducibles": 10000
        }
    }
    ```
    """
    return await handle_service_call(
        contador.calcular_impuestos,
        "calcular_impuestos",
        regimen=request.regimen,
        tipo_impuesto=request.tipo_impuesto,
        datos=request.datos
    )


@router.post("/asesoria-fiscal", response_model=Dict[str, Any])
async def asesoria_fiscal(
    request: AsesoriaFiscalRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Obtener asesoría fiscal personalizada.
    
    Ejemplo:
    ```json
    {
        "pregunta": "¿Qué deducciones puedo aplicar en RESICO?",
        "contexto": {
            "regimen": "RESICO",
            "actividad": "Servicios profesionales"
        }
    }
    ```
    """
    return await handle_service_call(
        contador.asesoria_fiscal,
        "asesoria_fiscal",
        pregunta=request.pregunta,
        contexto=request.contexto
    )


@router.post("/guia-fiscal", response_model=Dict[str, Any])
async def guia_fiscal(
    request: GuiaFiscalRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Generar guía fiscal sobre un tema específico.
    
    Ejemplo:
    ```json
    {
        "tema": "Deducciones para emprendedores en RESICO",
        "nivel_detalle": "completo"
    }
    ```
    """
    return await handle_service_call(
        contador.guia_fiscal,
        "guia_fiscal",
        tema=request.tema,
        nivel_detalle=request.nivel_detalle
    )


@router.post("/tramite-sat", response_model=Dict[str, Any])
async def tramite_sat(
    request: TramiteSATRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Obtener información sobre un trámite del SAT.
    
    Ejemplo:
    ```json
    {
        "tipo_tramite": "Alta en RFC",
        "detalles": {
            "tipo_persona": "Persona Física"
        }
    }
    ```
    """
    return await handle_service_call(
        contador.tramite_sat,
        "tramite_sat",
        tipo_tramite=request.tipo_tramite,
        detalles=request.detalles
    )


@router.post("/ayuda-declaracion", response_model=Dict[str, Any])
async def ayuda_declaracion(
    request: AyudaDeclaracionRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Ayudar con la preparación de una declaración fiscal.
    
    Ejemplo:
    ```json
    {
        "tipo_declaracion": "mensual",
        "periodo": "2024-01",
        "datos": {
            "regimen": "RESICO",
            "ingresos": 50000
        }
    }
    ```
    """
    return await handle_service_call(
        contador.ayuda_declaracion,
        "ayuda_declaracion",
        tipo_declaracion=request.tipo_declaracion,
        periodo=request.periodo,
        datos=request.datos
    )


@router.get("/regimenes", response_model=List[str])
async def listar_regimenes(contador: ContadorAI = Depends(get_contador)):
    """Listar regímenes fiscales soportados."""
    return contador.config.regimenes_fiscales


@router.get("/tipos-impuestos", response_model=List[str])
async def listar_tipos_impuestos(contador: ContadorAI = Depends(get_contador)):
    """Listar tipos de impuestos soportados."""
    return contador.config.tipos_impuestos


@router.get("/servicios", response_model=List[str])
async def listar_servicios(contador: ContadorAI = Depends(get_contador)):
    """Listar servicios disponibles."""
    return contador.config.servicios


@router.post("/comparar-regimenes", response_model=Dict[str, Any])
async def comparar_regimenes(
    request: CompararRegimenesRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Comparar diferentes regímenes fiscales.
    
    Ejemplo:
    ```json
    {
        "regimenes": ["RESICO", "PFAE"],
        "datos": {
            "ingresos_mensuales": 50000,
            "gastos_deducibles": 10000
        }
    }
    ```
    """
    return await handle_service_call(
        contador.comparar_regimenes,
        "comparar_regimenes",
        regimenes=request.regimenes,
        datos=request.datos
    )


@router.get("/cache/stats", response_model=Dict[str, Any])
async def cache_stats(contador: ContadorAI = Depends(get_contador)):
    """Get cache statistics."""
    stats = contador.get_cache_stats()
    if stats:
        return JSONResponse(content=stats)
    return JSONResponse(content={"message": "Cache is disabled"})


@router.get("/metrics/stats", response_model=Dict[str, Any])
async def metrics_stats(contador: ContadorAI = Depends(get_contador)):
    """Get metrics statistics."""
    stats = contador.get_metrics_stats()
    if stats:
        return JSONResponse(content=stats)
    return JSONResponse(content={"message": "Metrics are disabled"})


@router.post("/analizar-deducciones", response_model=Dict[str, Any])
async def analizar_deducciones(
    request: AnalizarDeduccionesRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Analizar deducciones fiscales y optimización.
    
    Ejemplo:
    ```json
    {
        "regimen": "RESICO",
        "ingresos": 50000,
        "gastos_actuales": {
            "Renta de local": 5000,
            "Materiales": 2000
        },
        "gastos_potenciales": {
            "Publicidad": 1000
        }
    }
    ```
    """
    try:
        from ..services.analizador_deducciones import AnalizadorDeducciones
        analizador = AnalizadorDeducciones()
        
        resultado = analizador.analizar_deducciones(
            regimen=request.regimen,
            ingresos=request.ingresos,
            gastos_actuales=request.gastos_actuales,
            gastos_potenciales=request.gastos_potenciales
        )
        
        return JSONResponse(content={
            "success": True,
            **resultado,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error analyzing deductions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exportar-resultado", response_model=Dict[str, Any])
async def exportar_resultado(
    request: ExportarResultadoRequest,
    contador: ContadorAI = Depends(get_contador)
):
    """
    Exportar resultado a diferentes formatos.
    
    Ejemplo:
    ```json
    {
        "resultado": {...},
        "formato": "markdown"
    }
    ```
    """
    try:
        contenido = contador.exportar_resultado(request.resultado, request.formato)
        return JSONResponse(content={
            "success": True,
            "formato": request.formato,
            "contenido": contenido,
            "timestamp": datetime.now().isoformat()
        })
    except ValueError as e:
        logger.warning(f"Invalid format: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/clear")
async def clear_cache(contador: ContadorAI = Depends(get_contador)):
    """Clear cache."""
    if contador.cache:
        contador.cache.clear()
        return JSONResponse(content={"message": "Cache cleared successfully"})
    return JSONResponse(content={"message": "Cache is disabled"})


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Contador AI"}

