"""
Calendar Integrations
=====================

Integraciones con calendarios externos (Google Calendar, Outlook).
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CalendarIntegration(ABC):
    """Clase base para integraciones de calendario."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Inicializar integración.
        
        Args:
            credentials: Credenciales de la integración
        """
        self.credentials = credentials
        self._logger = logger
    
    @abstractmethod
    async def sync_events(self, artist_id: str) -> List[Dict[str, Any]]:
        """
        Sincronizar eventos desde el calendario externo.
        
        Args:
            artist_id: ID del artista
        
        Returns:
            Lista de eventos sincronizados
        """
        pass
    
    @abstractmethod
    async def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crear evento en el calendario externo.
        
        Args:
            event_data: Datos del evento
        
        Returns:
            Evento creado
        """
        pass
    
    @abstractmethod
    async def update_event(self, event_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualizar evento en el calendario externo.
        
        Args:
            event_id: ID del evento
            event_data: Datos actualizados
        
        Returns:
            Evento actualizado
        """
        pass
    
    @abstractmethod
    async def delete_event(self, event_id: str) -> bool:
        """
        Eliminar evento del calendario externo.
        
        Args:
            event_id: ID del evento
        
        Returns:
            True si se eliminó
        """
        pass


class GoogleCalendarIntegration(CalendarIntegration):
    """Integración con Google Calendar."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Inicializar integración con Google Calendar.
        
        Args:
            credentials: Debe contener 'access_token' y opcionalmente 'refresh_token'
        """
        super().__init__(credentials)
        self.access_token = credentials.get("access_token")
        self.refresh_token = credentials.get("refresh_token")
        self.api_base = "https://www.googleapis.com/calendar/v3"
    
    async def sync_events(self, artist_id: str) -> List[Dict[str, Any]]:
        """Sincronizar eventos desde Google Calendar."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/calendars/primary/events",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={
                        "timeMin": datetime.now().isoformat() + "Z",
                        "maxResults": 100,
                        "singleEvents": True,
                        "orderBy": "startTime"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                events = []
                for item in data.get("items", []):
                    event = {
                        "id": item.get("id"),
                        "title": item.get("summary", ""),
                        "description": item.get("description", ""),
                        "start_time": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                        "end_time": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                        "location": item.get("location", ""),
                        "external_id": item.get("id"),
                        "external_source": "google_calendar"
                    }
                    events.append(event)
                
                self._logger.info(f"Synced {len(events)} events from Google Calendar for artist {artist_id}")
                return events
        
        except Exception as e:
            self._logger.error(f"Error syncing Google Calendar events: {str(e)}")
            return []
    
    async def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crear evento en Google Calendar."""
        try:
            import httpx
            
            google_event = {
                "summary": event_data.get("title"),
                "description": event_data.get("description", ""),
                "start": {
                    "dateTime": event_data.get("start_time"),
                    "timeZone": "UTC"
                },
                "end": {
                    "dateTime": event_data.get("end_time"),
                    "timeZone": "UTC"
                },
                "location": event_data.get("location", "")
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/calendars/primary/events",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    },
                    json=google_event
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            self._logger.error(f"Error creating Google Calendar event: {str(e)}")
            raise
    
    async def update_event(self, event_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Actualizar evento en Google Calendar."""
        try:
            import httpx
            
            google_event = {
                "summary": event_data.get("title"),
                "description": event_data.get("description", ""),
                "start": {
                    "dateTime": event_data.get("start_time"),
                    "timeZone": "UTC"
                },
                "end": {
                    "dateTime": event_data.get("end_time"),
                    "timeZone": "UTC"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.api_base}/calendars/primary/events/{event_id}",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    },
                    json=google_event
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            self._logger.error(f"Error updating Google Calendar event: {str(e)}")
            raise
    
    async def delete_event(self, event_id: str) -> bool:
        """Eliminar evento de Google Calendar."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base}/calendars/primary/events/{event_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                response.raise_for_status()
                return True
        
        except Exception as e:
            self._logger.error(f"Error deleting Google Calendar event: {str(e)}")
            return False


class OutlookCalendarIntegration(CalendarIntegration):
    """Integración con Outlook Calendar."""
    
    def __init__(self, credentials: Dict[str, Any]):
        """
        Inicializar integración con Outlook Calendar.
        
        Args:
            credentials: Debe contener 'access_token'
        """
        super().__init__(credentials)
        self.access_token = credentials.get("access_token")
        self.api_base = "https://graph.microsoft.com/v1.0/me/calendar"
    
    async def sync_events(self, artist_id: str) -> List[Dict[str, Any]]:
        """Sincronizar eventos desde Outlook Calendar."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/events",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={
                        "$filter": f"start/dateTime ge '{datetime.now().isoformat()}'",
                        "$orderby": "start/dateTime",
                        "$top": 100
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                events = []
                for item in data.get("value", []):
                    event = {
                        "id": item.get("id"),
                        "title": item.get("subject", ""),
                        "description": item.get("body", {}).get("content", ""),
                        "start_time": item.get("start", {}).get("dateTime"),
                        "end_time": item.get("end", {}).get("dateTime"),
                        "location": item.get("location", {}).get("displayName", ""),
                        "external_id": item.get("id"),
                        "external_source": "outlook_calendar"
                    }
                    events.append(event)
                
                self._logger.info(f"Synced {len(events)} events from Outlook Calendar for artist {artist_id}")
                return events
        
        except Exception as e:
            self._logger.error(f"Error syncing Outlook Calendar events: {str(e)}")
            return []
    
    async def create_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crear evento en Outlook Calendar."""
        try:
            import httpx
            
            outlook_event = {
                "subject": event_data.get("title"),
                "body": {
                    "contentType": "HTML",
                    "content": event_data.get("description", "")
                },
                "start": {
                    "dateTime": event_data.get("start_time"),
                    "timeZone": "UTC"
                },
                "end": {
                    "dateTime": event_data.get("end_time"),
                    "timeZone": "UTC"
                },
                "location": {
                    "displayName": event_data.get("location", "")
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/events",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    },
                    json=outlook_event
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            self._logger.error(f"Error creating Outlook Calendar event: {str(e)}")
            raise
    
    async def update_event(self, event_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Actualizar evento en Outlook Calendar."""
        try:
            import httpx
            
            outlook_event = {
                "subject": event_data.get("title"),
                "body": {
                    "contentType": "HTML",
                    "content": event_data.get("description", "")
                },
                "start": {
                    "dateTime": event_data.get("start_time"),
                    "timeZone": "UTC"
                },
                "end": {
                    "dateTime": event_data.get("end_time"),
                    "timeZone": "UTC"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f"{self.api_base}/events/{event_id}",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    },
                    json=outlook_event
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            self._logger.error(f"Error updating Outlook Calendar event: {str(e)}")
            raise
    
    async def delete_event(self, event_id: str) -> bool:
        """Eliminar evento de Outlook Calendar."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.api_base}/events/{event_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                response.raise_for_status()
                return True
        
        except Exception as e:
            self._logger.error(f"Error deleting Outlook Calendar event: {str(e)}")
            return False




