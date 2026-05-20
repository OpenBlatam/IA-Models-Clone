"""
Advanced API Routes
===================

Endpoints avanzados: ML, búsqueda, alertas, sincronización.
"""

import os
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

router = APIRouter(prefix="/advanced", tags=["advanced"])


class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None


def get_artist_manager(artist_id: str):
    """Dependency para obtener ArtistManager."""
    from ...core.artist_manager import ArtistManager
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    manager = ArtistManager(artist_id=artist_id, openrouter_api_key=openrouter_key)
    return manager


@router.post("/{artist_id}/search/events", response_model=List[Dict[str, Any]])
async def search_events(artist_id: str, search_request: SearchRequest):
    """Buscar eventos."""
    try:
        from ...services.search_service import SearchService
        
        manager, _, _ = get_artist_manager(artist_id)
        search_service = SearchService()
        
        events = [e.to_dict() for e in manager.calendar.get_all_events()]
        results = search_service.search_events(
            events,
            search_request.query,
            search_request.filters
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(artist_id: str):
    """Obtener alertas del artista."""
    try:
        from ...services.alert_service import AlertService
        
        manager, _, _ = get_artist_manager(artist_id)
        alert_service = AlertService()
        
        # Verificar conflictos
        events = [e.to_dict() for e in manager.calendar.get_all_events()]
        conflicts = alert_service.check_conflicts(artist_id, events)
        
        # Verificar rutinas vencidas
        routines = [r.to_dict() for r in manager.routines.get_all_routines()]
        completions = [c.to_dict() for c in manager.routines.get_completion_history()]
        overdue = alert_service.check_overdue_routines(artist_id, routines, completions)
        
        # Verificar sobrecarga
        overload = alert_service.check_schedule_overload(artist_id, events)
        
        all_alerts = conflicts + overdue + overload
        return all_alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/predictions/event-duration", response_model=Dict[str, Any])
async def predict_event_duration(artist_id: str, event_type: str):
    """Predecir duración de evento."""
    try:
        from ...ml.prediction_service import PredictionService
        
        manager, _, _ = get_artist_manager(artist_id)
        prediction_service = PredictionService()
        
        # Obtener eventos históricos del mismo tipo
        all_events = manager.calendar.get_all_events()
        historical = [
            e.to_dict() for e in all_events
            if e.event_type.value == event_type
        ]
        
        prediction = prediction_service.predict_event_duration(event_type, historical)
        return prediction.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{artist_id}/predictions/routine-completion", response_model=Dict[str, Any])
async def predict_routine_completion(artist_id: str, routine_id: str, days: int = 30):
    """Predecir tasa de completación de rutina."""
    try:
        from ...ml.prediction_service import PredictionService
        
        manager, _, _ = get_artist_manager(artist_id)
        prediction_service = PredictionService()
        
        completions = manager.routines.get_completion_history(routine_id, days=days)
        completion_data = [c.to_dict() for c in completions]
        
        prediction = prediction_service.predict_routine_completion(
            routine_id,
            completion_data,
            days=days
        )
        return prediction.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{artist_id}/sync/calendar", response_model=Dict[str, Any])
async def sync_calendar(artist_id: str, provider: str, credentials: Dict[str, Any]):
    """Sincronizar calendario externo."""
    try:
        from ...services.sync_service import SyncService
        from ...integrations.calendar_integrations import (
            GoogleCalendarIntegration,
            OutlookCalendarIntegration
        )
        
        sync_service = SyncService()
        
        if provider == "google":
            integration = GoogleCalendarIntegration(credentials)
        elif provider == "outlook":
            integration = OutlookCalendarIntegration(credentials)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
        
        result = await sync_service.sync_calendar(artist_id, integration)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




