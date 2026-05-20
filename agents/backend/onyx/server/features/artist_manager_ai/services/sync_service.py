"""
Sync Service
============

Servicio de sincronización automática con calendarios externos.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SyncService:
    """Servicio de sincronización."""
    
    def __init__(self):
        """Inicializar servicio de sincronización."""
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        self.sync_intervals: Dict[str, int] = {}  # segundos
        self._logger = logger
    
    async def sync_calendar(
        self,
        artist_id: str,
        calendar_integration,
        merge_strategy: str = "merge"
    ) -> Dict[str, Any]:
        """
        Sincronizar calendario externo.
        
        Args:
            artist_id: ID del artista
            calendar_integration: Integración de calendario
            merge_strategy: Estrategia de merge (merge, replace, skip)
        
        Returns:
            Resultado de sincronización
        """
        try:
            # Obtener eventos del calendario externo
            external_events = await calendar_integration.sync_events(artist_id)
            
            result = {
                "artist_id": artist_id,
                "synced_at": datetime.now().isoformat(),
                "external_events_count": len(external_events),
                "merged": 0,
                "created": 0,
                "skipped": 0,
                "errors": []
            }
            
            # Aquí se integraría con CalendarManager para merge
            # Por ahora retornamos estructura básica
            result["created"] = len(external_events)
            
            self._logger.info(
                f"Synced {len(external_events)} events for artist {artist_id}"
            )
            return result
        
        except Exception as e:
            self._logger.error(f"Error syncing calendar for artist {artist_id}: {str(e)}")
            return {
                "artist_id": artist_id,
                "error": str(e),
                "synced_at": datetime.now().isoformat()
            }
    
    def start_auto_sync(
        self,
        artist_id: str,
        calendar_integration,
        interval_seconds: int = 3600
    ):
        """
        Iniciar sincronización automática.
        
        Args:
            artist_id: ID del artista
            calendar_integration: Integración de calendario
            interval_seconds: Intervalo en segundos
        """
        async def sync_loop():
            while True:
                try:
                    await self.sync_calendar(artist_id, calendar_integration)
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    self._logger.error(f"Error in auto-sync loop: {str(e)}")
                    await asyncio.sleep(interval_seconds)
        
        task_key = f"{artist_id}_calendar"
        if task_key in self.sync_tasks:
            self.stop_auto_sync(artist_id)
        
        self.sync_tasks[task_key] = asyncio.create_task(sync_loop())
        self.sync_intervals[task_key] = interval_seconds
        self._logger.info(f"Started auto-sync for artist {artist_id} (interval: {interval_seconds}s)")
    
    def stop_auto_sync(self, artist_id: str):
        """
        Detener sincronización automática.
        
        Args:
            artist_id: ID del artista
        """
        task_key = f"{artist_id}_calendar"
        if task_key in self.sync_tasks:
            self.sync_tasks[task_key].cancel()
            del self.sync_tasks[task_key]
            if task_key in self.sync_intervals:
                del self.sync_intervals[task_key]
            self._logger.info(f"Stopped auto-sync for artist {artist_id}")




