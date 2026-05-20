"""
Rutas para Knowledge Graph
===========================

Endpoints para knowledge graph.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.knowledge_graph import get_knowledge_graph, KnowledgeGraph

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/knowledge-graph",
    tags=["Knowledge Graph"]
)


class AddNodeRequest(BaseModel):
    """Request para agregar nodo"""
    node_id: str = Field(..., description="ID del nodo")
    label: str = Field(..., description="Etiqueta del nodo")
    node_type: str = Field(..., description="Tipo de nodo")
    properties: Optional[Dict[str, Any]] = Field(None, description="Propiedades")


class AddEdgeRequest(BaseModel):
    """Request para agregar arista"""
    source_id: str = Field(..., description="ID del nodo origen")
    target_id: str = Field(..., description="ID del nodo destino")
    relation: str = Field(..., description="Tipo de relación")
    properties: Optional[Dict[str, Any]] = Field(None, description="Propiedades")


@router.post("/nodes")
async def add_node(
    request: AddNodeRequest,
    graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Agregar nodo al knowledge graph"""
    try:
        node = graph.add_node(
            request.node_id,
            request.label,
            request.node_type,
            request.properties
        )
        
        return {
            "status": "added",
            "node_id": node.node_id,
            "label": node.label
        }
    except Exception as e:
        logger.error(f"Error agregando nodo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edges")
async def add_edge(
    request: AddEdgeRequest,
    graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Agregar arista al knowledge graph"""
    try:
        edge = graph.add_edge(
            request.source_id,
            request.target_id,
            request.relation,
            request.properties
        )
        
        return {
            "status": "added",
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation": edge.relation
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error agregando arista: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/path/{source_id}/{target_id}")
async def find_path(
    source_id: str,
    target_id: str,
    max_depth: int = 5,
    graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Encontrar camino entre nodos"""
    try:
        path = graph.find_path(source_id, target_id, max_depth)
        
        if path is None:
            return {"path": None, "message": "No se encontró camino"}
        
        return {"path": path, "length": len(path) - 1}
    except Exception as e:
        logger.error(f"Error encontrando camino: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}/related")
async def get_related_nodes(
    node_id: str,
    relation: Optional[str] = None,
    max_depth: int = 1,
    graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Obtener nodos relacionados"""
    try:
        related = graph.get_related_nodes(node_id, relation, max_depth)
        
        return {
            "node_id": node_id,
            "related_nodes": related,
            "count": len(related)
        }
    except Exception as e:
        logger.error(f"Error obteniendo nodos relacionados: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_stats(
    graph: KnowledgeGraph = Depends(get_knowledge_graph)
):
    """Obtener estadísticas del knowledge graph"""
    stats = graph.get_graph_stats()
    return stats














