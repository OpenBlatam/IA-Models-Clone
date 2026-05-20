"""
Rutas para Graph Neural Networks
==================================

Endpoints para GNN.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.graph_neural_networks import (
    get_gnn,
    GraphNeuralNetwork,
    GraphNode,
    GraphEdge,
    GNNType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/gnn",
    tags=["Graph Neural Networks"]
)


class CreateGraphRequest(BaseModel):
    """Request para crear grafo"""
    nodes: List[Dict[str, Any]] = Field(..., description="Nodos")
    edges: List[Dict[str, Any]] = Field(..., description="Aristas")


@router.post("/graphs/{graph_id}")
async def create_graph(
    graph_id: str,
    request: CreateGraphRequest,
    gnn: GraphNeuralNetwork = Depends(get_gnn)
):
    """Crear grafo"""
    try:
        nodes = [
            GraphNode(
                node_id=node.get("node_id", ""),
                features=node.get("features", []),
                label=node.get("label")
            )
            for node in request.nodes
        ]
        
        edges = [
            GraphEdge(
                source=edge.get("source", ""),
                target=edge.get("target", ""),
                weight=edge.get("weight", 1.0),
                features=edge.get("features")
            )
            for edge in request.edges
        ]
        
        gnn.create_graph(graph_id, nodes, edges)
        
        return {"status": "created", "graph_id": graph_id}
    except Exception as e:
        logger.error(f"Error creando grafo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graphs/{graph_id}/train")
async def train_gnn(
    graph_id: str,
    gnn_type: str = Field("gcn", description="Tipo de GNN"),
    epochs: int = Field(10, description="Número de épocas"),
    gnn: GraphNeuralNetwork = Depends(get_gnn)
):
    """Entrenar GNN"""
    try:
        gnn_type_enum = GNNType(gnn_type)
        result = gnn.train_gnn(graph_id, gnn_type_enum, epochs)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error entrenando GNN: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/predict-node")
async def predict_node(
    model_id: str,
    node_id: str = Field(..., description="ID del nodo"),
    gnn: GraphNeuralNetwork = Depends(get_gnn)
):
    """Predecir etiqueta de nodo"""
    try:
        prediction = gnn.predict_node(model_id, node_id)
        
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error prediciendo nodo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/predict-link")
async def predict_link(
    model_id: str,
    source_node: str = Field(..., description="Nodo origen"),
    target_node: str = Field(..., description="Nodo destino"),
    gnn: GraphNeuralNetwork = Depends(get_gnn)
):
    """Predecir existencia de enlace"""
    try:
        prediction = gnn.predict_link(model_id, source_node, target_node)
        
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error prediciendo enlace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


