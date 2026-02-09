"""
Rutas para NLP Avanzado
========================

Endpoints para procesamiento avanzado de lenguaje natural.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_nlp import get_advanced_nlp, AdvancedNLProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/nlp",
    tags=["Advanced NLP"]
)


class AnalyzeTextRequest(BaseModel):
    """Request para análisis de texto"""
    text: str = Field(..., description="Texto a analizar")


@router.post("/entities")
async def extract_entities(
    request: AnalyzeTextRequest,
    nlp: AdvancedNLProcessor = Depends(get_advanced_nlp)
):
    """Extraer entidades avanzadas"""
    try:
        entities = nlp.extract_entities_advanced(request.text)
        
        return {
            "entities": entities,
            "count": len(entities)
        }
    except Exception as e:
        logger.error(f"Error extrayendo entidades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relations")
async def extract_relations(
    request: AnalyzeTextRequest,
    entities: Optional[list] = None,
    nlp: AdvancedNLProcessor = Depends(get_advanced_nlp)
):
    """Extraer relaciones entre entidades"""
    try:
        relations = nlp.extract_relations(request.text, entities)
        
        return {
            "relations": relations,
            "count": len(relations)
        }
    except Exception as e:
        logger.error(f"Error extrayendo relaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/coreferences")
async def resolve_coreferences(
    request: AnalyzeTextRequest,
    nlp: AdvancedNLProcessor = Depends(get_advanced_nlp)
):
    """Resolver coreferencias"""
    try:
        coreferences = nlp.resolve_coreferences(request.text)
        
        return {
            "coreferences": coreferences,
            "count": len(coreferences)
        }
    except Exception as e:
        logger.error(f"Error resolviendo coreferencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discourse")
async def analyze_discourse(
    request: AnalyzeTextRequest,
    nlp: AdvancedNLProcessor = Depends(get_advanced_nlp)
):
    """Analizar estructura discursiva"""
    try:
        structure = nlp.analyze_discourse_structure(request.text)
        
        return structure
    except Exception as e:
        logger.error(f"Error analizando discurso: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic-roles")
async def analyze_semantic_roles(
    request: AnalyzeTextRequest,
    nlp: AdvancedNLProcessor = Depends(get_advanced_nlp)
):
    """Analizar roles semánticos"""
    try:
        roles = nlp.analyze_semantic_roles(request.text)
        
        return {
            "semantic_roles": roles,
            "count": len(roles)
        }
    except Exception as e:
        logger.error(f"Error analizando roles semánticos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comprehensive")
async def comprehensive_nlp_analysis(
    request: AnalyzeTextRequest,
    nlp: AdvancedNLProcessor = Depends(get_advanced_nlp)
):
    """Análisis completo de NLP"""
    try:
        analysis = nlp.comprehensive_nlp_analysis(request.text)
        
        return {
            "entities": analysis.entities,
            "relations": analysis.relations,
            "coreferences": analysis.coreferences,
            "discourse_structure": analysis.discourse_structure,
            "semantic_roles": analysis.semantic_roles
        }
    except Exception as e:
        logger.error(f"Error en análisis completo: {e}")
        raise HTTPException(status_code=500, detail=str(e))














