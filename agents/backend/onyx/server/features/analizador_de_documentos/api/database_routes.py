"""
Rutas para Integración con Bases de Datos
==========================================

Endpoints para integración con bases de datos.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.database_integration import (
    get_database_integration,
    DatabaseIntegration,
    DatabaseType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/database",
    tags=["Database Integration"]
)


class RegisterConnectionRequest(BaseModel):
    """Request para registrar conexión"""
    connection_id: str = Field(..., description="ID de conexión")
    db_type: str = Field(..., description="Tipo de BD (postgresql, mysql, mongodb, redis, elasticsearch)")
    connection_string: str = Field(..., description="String de conexión")


class ExecuteQueryRequest(BaseModel):
    """Request para ejecutar query"""
    query: str = Field(..., description="Query a ejecutar")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parámetros")


@router.post("/connections")
async def register_connection(
    request: RegisterConnectionRequest,
    integration: DatabaseIntegration = Depends(get_database_integration)
):
    """Registrar conexión a base de datos"""
    try:
        db_type = DatabaseType(request.db_type)
        connection = integration.register_connection(
            request.connection_id,
            db_type,
            request.connection_string
        )
        
        return {
            "status": "registered",
            "connection_id": connection.connection_id,
            "db_type": connection.db_type.value
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Tipo de BD inválido: {request.db_type}")
    except Exception as e:
        logger.error(f"Error registrando conexión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/query")
async def execute_query(
    connection_id: str,
    request: ExecuteQueryRequest,
    integration: DatabaseIntegration = Depends(get_database_integration)
):
    """Ejecutar query"""
    try:
        result = integration.execute_query(
            connection_id,
            request.query,
            request.parameters
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error ejecutando query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/connections/{connection_id}/save-analysis")
async def save_analysis_result(
    connection_id: str,
    analysis_result: Dict[str, Any],
    integration: DatabaseIntegration = Depends(get_database_integration)
):
    """Guardar resultado de análisis"""
    try:
        success = integration.save_analysis_result(connection_id, analysis_result)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error guardando resultado")
        
        return {"status": "saved", "connection_id": connection_id}
    except Exception as e:
        logger.error(f"Error guardando resultado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections")
async def list_connections(
    integration: DatabaseIntegration = Depends(get_database_integration)
):
    """Listar todas las conexiones"""
    connections = integration.list_connections()
    return {"connections": connections}














