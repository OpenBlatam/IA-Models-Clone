"""
Network Analysis endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.network_analysis import NetworkAnalysisService

router = APIRouter()
network_service = NetworkAnalysisService()


@router.post("/add-contact/{user_id}")
async def add_contact(
    user_id: str,
    name: str,
    title: str,
    company: str,
    industry: str,
    connection_strength: float = 0.5
) -> Dict[str, Any]:
    """Agregar contacto a la red"""
    try:
        contact = network_service.add_contact(
            user_id, name, title, company, industry, connection_strength
        )
        return {
            "id": contact.id,
            "name": contact.name,
            "title": contact.title,
            "company": contact.company,
            "connection_strength": contact.connection_strength,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{user_id}")
async def analyze_network(user_id: str) -> Dict[str, Any]:
    """Analizar red profesional"""
    try:
        insight = network_service.analyze_network(user_id)
        return {
            "total_contacts": insight.total_contacts,
            "strong_connections": insight.strong_connections,
            "weak_connections": insight.weak_connections,
            "industries": insight.industries,
            "companies": insight.companies,
            "recommendations": insight.recommendations,
            "network_score": insight.network_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/introductions/{user_id}")
async def find_introductions(
    user_id: str,
    target_company: str,
    target_title: Optional[str] = None
) -> Dict[str, Any]:
    """Encontrar posibles introducciones"""
    try:
        introductions = network_service.find_introductions(user_id, target_company, target_title)
        return {
            "introductions": introductions,
            "total": len(introductions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




