"""
Dashboard API Routes
====================

Endpoints para el dashboard del artista.
"""

import os
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def get_artist_manager(artist_id: str) -> "ArtistManager":
    """Dependency para obtener ArtistManager."""
    from ...core.artist_manager import ArtistManager
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    return ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key)


@router.get("/{artist_id}", response_model=Dict[str, Any])
async def get_dashboard(artist_id: str):
    """Obtener datos del dashboard."""
    try:
        manager = get_artist_manager(artist_id)
        return manager.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/daily-summary", response_model=Dict[str, Any])
async def get_daily_summary(artist_id: str):
    """Obtener resumen diario generado por IA."""
    try:
        manager = get_artist_manager(artist_id)
        summary = await manager.generate_daily_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




