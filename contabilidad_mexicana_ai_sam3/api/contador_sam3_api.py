"""
REST API for Contabilidad Mexicana AI SAM3
==========================================

Optional REST API wrapper for the agent.

Refactored to:
- Use centralized API helpers for common patterns
- Eliminate duplicate error handling
- Standardize response formats
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..core.contador_sam3_agent import ContadorSAM3Agent
from ..config.contador_sam3_config import ContadorSAM3Config
from .api_helpers import require_agent, handle_task_operation
from .error_handlers import handle_task_errors
from .response_builder import ResponseBuilder

logger = logging.getLogger(__name__)

app = FastAPI(title="Contabilidad Mexicana AI SAM3 API", version="1.0.0")

# Global agent instance
_agent: Optional[ContadorSAM3Agent] = None


# Request models
class CalcularImpuestosRequest(BaseModel):
    regimen: str
    tipo_impuesto: str
    datos: Dict[str, Any]
    priority: int = 0


class AsesoriaFiscalRequest(BaseModel):
    pregunta: str
    contexto: Optional[Dict[str, Any]] = None
    priority: int = 0


class GuiaFiscalRequest(BaseModel):
    tema: str
    nivel_detalle: str = "completo"
    priority: int = 0


class TramiteSATRequest(BaseModel):
    tipo_tramite: str
    detalles: Optional[Dict[str, Any]] = None
    priority: int = 0


class AyudaDeclaracionRequest(BaseModel):
    tipo_declaracion: str
    periodo: str
    datos: Optional[Dict[str, Any]] = None
    priority: int = 0


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global _agent
    config = ContadorSAM3Config()
    _agent = ContadorSAM3Agent(config=config)
    logger.info("ContadorSAM3Agent initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _agent
    if _agent:
        await _agent.close()
        _agent = None


@app.post("/calcular-impuestos")
async def calcular_impuestos(request: CalcularImpuestosRequest):
    """Calculate taxes."""
    agent = require_agent(_agent)
    
    task_id = await agent.calcular_impuestos(
        regimen=request.regimen,
        tipo_impuesto=request.tipo_impuesto,
        datos=request.datos,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.post("/asesoria-fiscal")
async def asesoria_fiscal(request: AsesoriaFiscalRequest):
    """Get fiscal advice."""
    agent = require_agent(_agent)
    
    task_id = await agent.asesoria_fiscal(
        pregunta=request.pregunta,
        contexto=request.contexto,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.post("/guia-fiscal")
async def guia_fiscal(request: GuiaFiscalRequest):
    """Get fiscal guide."""
    agent = require_agent(_agent)
    
    task_id = await agent.guia_fiscal(
        tema=request.tema,
        nivel_detalle=request.nivel_detalle,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.post("/tramite-sat")
async def tramite_sat(request: TramiteSATRequest):
    """Get SAT procedure information."""
    agent = require_agent(_agent)
    
    task_id = await agent.tramite_sat(
        tipo_tramite=request.tipo_tramite,
        detalles=request.detalles,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.post("/ayuda-declaracion")
async def ayuda_declaracion(request: AyudaDeclaracionRequest):
    """Get declaration assistance."""
    agent = require_agent(_agent)
    
    task_id = await agent.ayuda_declaracion(
        tipo_declaracion=request.tipo_declaracion,
        periodo=request.periodo,
        datos=request.datos,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task status."""
    agent = require_agent(_agent)
    
    return await handle_task_operation(
        agent,
        "get_task_status",
        agent.get_task_status,
        task_id
    )


@app.get("/task/{task_id}/result")
@handle_task_errors
async def get_task_result(task_id: str):
    """
    Get task result.
    
    Uses helper for consistent error handling.
    """
    agent = require_agent(_agent)
    result = await agent.get_task_result(task_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Task not completed yet")
    
    return result


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Uses ResponseBuilder for consistent response format.
    """
    agent_running = _agent is not None and _agent.running if _agent else False
    return ResponseBuilder.health_check(agent_running)
