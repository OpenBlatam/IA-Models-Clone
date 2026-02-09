"""
Knowledge API Endpoints
=======================

Endpoints para base de conocimientos y recomendaciones.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import logging

from ..core.knowledge_base import get_knowledge_base
from ..core.recommendation_engine import get_recommendation_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post("/entries")
async def create_knowledge_entry(
    entry_id: str,
    title: str,
    content: str,
    category: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear entrada de conocimiento."""
    try:
        kb = get_knowledge_base()
        entry = kb.add_entry(
            entry_id=entry_id,
            title=title,
            content=content,
            category=category,
            tags=tags,
            metadata=metadata
        )
        return {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "category": entry.category
        }
    except Exception as e:
        logger.error(f"Error creating knowledge entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entries/search")
async def search_knowledge(
    query: str,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = Query(10, ge=1, le=100)
) -> Dict[str, Any]:
    """Buscar en base de conocimientos."""
    try:
        kb = get_knowledge_base()
        results = kb.search(query, category=category, tags=tags, limit=limit)
        return {
            "query": query,
            "results": [
                {
                    "entry_id": r.entry_id,
                    "title": r.title,
                    "content": r.content[:200],  # Primeros 200 caracteres
                    "category": r.category,
                    "tags": r.tags
                }
                for r in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entries")
async def list_knowledge_entries(
    category: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Listar entradas de conocimiento."""
    try:
        kb = get_knowledge_base()
        entries = kb.list_entries(category=category, limit=limit)
        return {
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "title": e.title,
                    "category": e.category,
                    "tags": e.tags,
                    "created_at": e.created_at
                }
                for e in entries
            ],
            "count": len(entries)
        }
    except Exception as e:
        logger.error(f"Error listing knowledge entries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(
    recommendation_type: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=100)
) -> Dict[str, Any]:
    """Obtener recomendaciones."""
    try:
        engine = get_recommendation_engine()
        recommendations = engine.get_recommendations(
            recommendation_type=recommendation_type,
            min_confidence=min_confidence,
            limit=limit
        )
        return {
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "type": r.type,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "action": r.action,
                    "timestamp": r.timestamp
                }
                for r in recommendations
            ],
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/generate/performance")
async def generate_performance_recommendations() -> Dict[str, Any]:
    """Generar recomendaciones de performance."""
    try:
        engine = get_recommendation_engine()
        recommendations = engine.generate_performance_recommendations()
        return {
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "action": r.action
                }
                for r in recommendations
            ],
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error generating performance recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/generate/optimization")
async def generate_optimization_recommendations() -> Dict[str, Any]:
    """Generar recomendaciones de optimización."""
    try:
        engine = get_recommendation_engine()
        recommendations = engine.generate_optimization_recommendations()
        return {
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "action": r.action
                }
                for r in recommendations
            ],
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error generating optimization recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






