"""
Agent Routes - Rutas para control del agente
============================================

Endpoints para controlar el ciclo de vida del agente.
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..models import StatusResponse
from ..templates import get_dashboard_html
from ..utils import get_agent, handle_route_errors, create_success_response, AgentDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])


@router.get("/", response_class=HTMLResponse)
async def root():
    """
    Página principal con dashboard del agente.
    
    Returns:
        HTML del dashboard.
    """
    return get_dashboard_html()


@router.post("/start")
@handle_route_errors("starting agent")
async def start_agent(agent = AgentDep):
    """
    Iniciar el agente.
    
    Returns:
        Mensaje de confirmación.
    
    Raises:
        HTTPException: Si hay error al iniciar el agente.
    """
    await agent.start()
    return create_success_response(
        status="started",
        message="Agent started successfully"
    )


@router.post("/stop")
@handle_route_errors("stopping agent")
async def stop_agent(agent = AgentDep):
    """
    Detener el agente.
    
    Returns:
        Mensaje de confirmación.
    
    Raises:
        HTTPException: Si hay error al detener el agente.
    """
    await agent.stop()
    return create_success_response(
        status="stopped",
        message="Agent stopped successfully"
    )


@router.post("/pause")
@handle_route_errors("pausing agent")
async def pause_agent(agent = AgentDep):
    """
    Pausar el agente.
    
    Returns:
        Mensaje de confirmación.
    
    Raises:
        HTTPException: Si hay error al pausar el agente.
    """
    await agent.pause()
    return create_success_response(
        status="paused",
        message="Agent paused successfully"
    )


@router.post("/resume")
@handle_route_errors("resuming agent")
async def resume_agent(agent = AgentDep):
    """
    Reanudar el agente.
    
    Returns:
        Mensaje de confirmación.
    
    Raises:
        HTTPException: Si hay error al reanudar el agente.
    """
    await agent.resume()
    return create_success_response(
        status="resumed",
        message="Agent resumed successfully"
    )


@router.get("/status", response_model=StatusResponse)
@handle_route_errors("getting agent status")
async def get_status(agent = AgentDep):
    """
    Obtener estado del agente.
    
    Returns:
        Estado actual del agente.
    
    Raises:
        HTTPException: Si hay error al obtener el estado.
    """
    return await agent.get_status()

