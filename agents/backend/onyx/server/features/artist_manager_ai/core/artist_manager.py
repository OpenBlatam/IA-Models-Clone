"""
Artist Manager
==============

Gestor principal que integra todas las funcionalidades del manager de artistas.
Utiliza OpenRouter para generar recomendaciones inteligentes.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from ..infrastructure.openrouter_client import OpenRouterClient
from ..services.database_service import DatabaseService
from ..services.notification_service import NotificationService, NotificationType
from ..services.analytics_service import AnalyticsService
from ..utils.cache import CacheManager
from ..utils.validators import Validator
from ..utils.ai_helpers import AIHelper
from .calendar_manager import CalendarManager, CalendarEvent, EventType
from .routine_manager import RoutineManager, RoutineTask, RoutineType, RoutineStatus
from .protocol_manager import ProtocolManager, Protocol, ProtocolCategory, ProtocolPriority
from .wardrobe_manager import WardrobeManager, WardrobeItem, Outfit, DressCode, WardrobeRecommendation

logger = logging.getLogger(__name__)


class ArtistManager:
    """Gestor principal de artista con integración de IA."""
    
    def __init__(
        self,
        artist_id: str,
        openrouter_api_key: Optional[str] = None,
        enable_persistence: bool = True,
        enable_notifications: bool = True,
        enable_analytics: bool = True
    ):
        """
        Inicializar gestor de artista.
        
        Args:
            artist_id: ID del artista
            openrouter_api_key: API key de OpenRouter (opcional)
            enable_persistence: Habilitar persistencia en BD
            enable_notifications: Habilitar notificaciones
            enable_analytics: Habilitar analytics
        """
        if not Validator.validate_artist_id(artist_id):
            raise ValueError(f"Invalid artist_id: {artist_id}")
        
        self.artist_id = artist_id
        self.calendar = CalendarManager(artist_id)
        self.routines = RoutineManager(artist_id)
        self.protocols = ProtocolManager(artist_id)
        self.wardrobe = WardrobeManager(artist_id)
        
        self.openrouter = OpenRouterClient(openrouter_api_key) if openrouter_api_key else None
        self.cache = CacheManager(default_ttl_seconds=1800)  # 30 minutos
        
        # Servicios opcionales
        self.db_service = DatabaseService() if enable_persistence else None
        self.notification_service = NotificationService() if enable_notifications else None
        self.analytics = AnalyticsService() if enable_analytics else None
        
        self._logger = logger
        
        # Cargar datos desde BD si está habilitada
        if self.db_service:
            self._load_from_database()
    
    def _load_from_database(self):
        """Cargar datos desde base de datos."""
        try:
            if not self.db_service:
                return
            
            # Cargar eventos
            events_data = self.db_service.load_events(self.artist_id)
            for event_data in events_data:
                try:
                    event = CalendarEvent(
                        id=event_data['id'],
                        title=event_data['title'],
                        description=event_data.get('description', ''),
                        event_type=EventType(event_data['event_type']),
                        start_time=datetime.fromisoformat(event_data['start_time']),
                        end_time=datetime.fromisoformat(event_data['end_time']),
                        location=event_data.get('location'),
                        attendees=event_data.get('attendees', []),
                        protocol_requirements=event_data.get('protocol_requirements', []),
                        wardrobe_requirements=event_data.get('wardrobe_requirements'),
                        notes=event_data.get('notes')
                    )
                    self.calendar.events[event.id] = event
                except Exception as e:
                    self._logger.warning(f"Error loading event {event_data.get('id')}: {e}")
            
            # Cargar rutinas
            routines_data = self.db_service.load_routines(self.artist_id)
            for routine_data in routines_data:
                try:
                    from datetime import time
                    routine = RoutineTask(
                        id=routine_data['id'],
                        title=routine_data['title'],
                        description=routine_data.get('description', ''),
                        routine_type=RoutineType(routine_data['routine_type']),
                        scheduled_time=time.fromisoformat(routine_data['scheduled_time']),
                        duration_minutes=routine_data['duration_minutes'],
                        priority=routine_data.get('priority', 5),
                        days_of_week=routine_data.get('days_of_week', []),
                        is_required=bool(routine_data.get('is_required', True)),
                        notes=routine_data.get('notes')
                    )
                    self.routines.routines[routine.id] = routine
                except Exception as e:
                    self._logger.warning(f"Error loading routine {routine_data.get('id')}: {e}")
            
            self._logger.info(f"Loaded data from database for artist {self.artist_id}")
        except Exception as e:
            self._logger.error(f"Error loading from database: {e}")
    
    def _save_to_database(self):
        """Guardar datos en base de datos."""
        try:
            if not self.db_service:
                return
            
            # Guardar eventos
            for event in self.calendar.events.values():
                self.db_service.save_event(self.artist_id, event.to_dict())
            
            # Guardar rutinas
            for routine in self.routines.routines.values():
                self.db_service.save_routine(self.artist_id, routine.to_dict())
            
        except Exception as e:
            self._logger.error(f"Error saving to database: {e}")
    
    async def generate_daily_summary(self) -> Dict[str, Any]:
        """
        Generar resumen diario usando IA.
        
        Returns:
            Resumen del día con recomendaciones
        """
        # Usar cache si está disponible
        cache_key = f"daily_summary:{self.artist_id}:{datetime.now().date().isoformat()}"
        cached_summary = self.cache.get(cache_key)
        if cached_summary:
            return cached_summary
        
        today = datetime.now()
        today_events = self.calendar.get_events_by_date(today)
        pending_routines = self.routines.get_pending_routines()
        
        # Registrar métrica
        if self.analytics:
            self.analytics.record_metric(
                "daily_summary_generated",
                1,
                tags={"artist_id": self.artist_id}
            )
        
        summary_prompt = f"""Eres un asistente de manager de artistas profesional. Genera un resumen diario motivacional y útil para el artista {self.artist_id}.

Eventos de hoy ({today.strftime('%Y-%m-%d')}):
{json.dumps([e.to_dict() for e in today_events], indent=2, default=str)}

Rutinas pendientes:
{json.dumps([r.to_dict() for r in pending_routines], indent=2, default=str)}

Genera un resumen conciso y motivacional que incluya:
1. Resumen de eventos del día
2. Recordatorios de rutinas importantes
3. Recomendaciones generales
4. Motivación positiva

Formato: JSON con campos: summary, events_summary, routines_reminder, recommendations, motivation"""

        try:
            if not self.openrouter:
                return {
                    "error": "OpenRouter not configured",
                    "summary": "OpenRouter API key required for AI summaries"
                }
            
            response = await self.openrouter.generate_text(
                prompt=summary_prompt,
                model="anthropic/claude-3-haiku",
                max_tokens=1500,
                temperature=0.7
            )
            
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Intentar parsear JSON de la respuesta
            try:
                summary_data = json.loads(content)
            except json.JSONDecodeError:
                # Si no es JSON válido, usar el texto como está
                summary_data = {
                    "summary": content,
                    "events_summary": f"{len(today_events)} eventos programados",
                    "routines_reminder": f"{len(pending_routines)} rutinas pendientes",
                    "recommendations": "Revisa tus eventos y rutinas del día",
                    "motivation": "¡Tienes un gran día por delante!"
                }
            
            result = {
                "artist_id": self.artist_id,
                "date": today.isoformat(),
                **summary_data,
                "events_count": len(today_events),
                "pending_routines_count": len(pending_routines)
            }
            
            # Guardar en cache
            self.cache.set(cache_key, result, ttl_seconds=3600)
            
            return result
        
        except Exception as e:
            self._logger.error(f"Error generating daily summary: {str(e)}")
            return {
                "error": str(e),
                "summary": "Error al generar resumen",
                "events_count": len(today_events),
                "pending_routines_count": len(pending_routines)
            }
    
    async def generate_wardrobe_recommendation(
        self,
        event_id: str,
        occasion: Optional[str] = None
    ) -> WardrobeRecommendation:
        """
        Generar recomendación de vestimenta usando IA.
        
        Args:
            event_id: ID del evento
            occasion: Ocasión del evento (opcional)
        
        Returns:
            Recomendación de vestimenta
        """
        event = self.calendar.get_event(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")
        
        # Obtener protocolos del evento
        event_protocols = self.protocols.get_protocols_for_event(event_id)
        
        # Usar helper mejorado para crear prompt
        prompt = AIHelper.improve_prompt_for_wardrobe(
            event_data=event.to_dict(),
            wardrobe_items=[item.to_dict() for item in self.wardrobe.get_all_items()[:30]],
            protocols=[p.to_dict() for p in event_protocols]
        )
        
        # Registrar métrica
        if self.analytics:
            self.analytics.record_metric(
                "wardrobe_recommendation_generated",
                1,
                tags={"artist_id": self.artist_id, "event_type": event.event_type.value}
            )
        
        # Construir prompt para IA (fallback si helper no funciona)
        if not prompt:
            prompt = f"""Eres un estilista profesional para artistas. Recomienda vestimenta para el siguiente evento:

Evento: {event.title}
Tipo: {event.event_type.value}
Descripción: {event.description}
Fecha: {event.start_time.strftime('%Y-%m-%d %H:%M')}
Ubicación: {event.location or 'No especificada'}

Protocolos aplicables:
{json.dumps([p.to_dict() for p in event_protocols], indent=2, default=str)}

Items disponibles en el guardarropa:
{json.dumps([item.to_dict() for item in self.wardrobe.get_all_items()[:20]], indent=2, default=str)}

Genera una recomendación de vestimenta que incluya:
1. Código de vestimenta apropiado (formal, casual, etc.)
2. Recomendación específica basada en los items disponibles
3. Razón de la recomendación
4. Consideraciones del clima si aplica

Formato: JSON con campos: dress_code, recommendation, reasoning, weather_considerations"""

        try:
            if not self.openrouter:
                # Sin IA, usar recomendación básica
                dress_code = DressCode.SMART_CASUAL
                if event.event_type == EventType.CONCERT:
                    dress_code = DressCode.ARTISTIC
                elif event.event_type == EventType.INTERVIEW:
                    dress_code = DressCode.BUSINESS_CASUAL
                
                return self.wardrobe.create_recommendation(
                    occasion=occasion or event.title,
                    dress_code=dress_code,
                    event_id=event_id,
                    reasoning="Recomendación básica basada en tipo de evento"
                )
            
            response = await self.openrouter.generate_text(
                prompt=prompt,
                model="anthropic/claude-3-haiku",
                max_tokens=1000,
                temperature=0.7
            )
            
            # Usar helper mejorado para parsear respuesta
            rec_data = AIHelper.parse_ai_response(response, expected_fields=["dress_code", "reasoning"])
            
            # Extraer dress_code
            dress_code_str = rec_data.get("dress_code", "smart_casual")
            dress_code = DressCode(dress_code_str) if dress_code_str in [dc.value for dc in DressCode] else DressCode.SMART_CASUAL
            
            # Si no se pudo parsear bien, usar fallback
            if not rec_data.get("reasoning"):
                rec_data["reasoning"] = rec_data.get("raw_content", "Recomendación generada por IA")[:200]
            
            recommendation = self.wardrobe.create_recommendation(
                occasion=occasion or event.title,
                dress_code=dress_code,
                event_id=event_id,
                reasoning=rec_data.get("reasoning", "Recomendación generada por IA"),
                weather_considerations=rec_data.get("weather_considerations")
            )
            
            return recommendation
        
        except Exception as e:
            self._logger.error(f"Error generating wardrobe recommendation: {str(e)}")
            # Fallback a recomendación básica
            dress_code = DressCode.SMART_CASUAL
            return self.wardrobe.create_recommendation(
                occasion=occasion or event.title,
                dress_code=dress_code,
                event_id=event_id,
                reasoning=f"Error al generar recomendación: {str(e)}"
            )
    
    async def check_protocol_compliance(self, event_id: str) -> Dict[str, Any]:
        """
        Verificar cumplimiento de protocolos para un evento usando IA.
        
        Args:
            event_id: ID del evento
        
        Returns:
            Reporte de cumplimiento
        """
        event = self.calendar.get_event(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")
        
        event_protocols = self.protocols.get_protocols_for_event(event_id)
        
        if not event_protocols:
            return {
                "event_id": event_id,
                "compliant": True,
                "message": "No hay protocolos específicos para este evento",
                "checked_protocols": []
            }
        
        # Usar helper mejorado para crear prompt
        prompt = AIHelper.improve_prompt_for_compliance(
            event_data=event.to_dict(),
            protocols=[p.to_dict() for p in event_protocols]
        )
        
        # Registrar métrica
        if self.analytics:
            self.analytics.record_metric(
                "protocol_compliance_checked",
                1,
                tags={"artist_id": self.artist_id, "event_id": event_id}
            )
        
        # Fallback si helper no funciona
        if not prompt:
            prompt = f"""Eres un auditor de protocolos para artistas. Evalúa el cumplimiento de protocolos para el siguiente evento:

Evento: {event.title}
Tipo: {event.event_type.value}
Fecha: {event.start_time.strftime('%Y-%m-%d %H:%M')}

Protocolos a verificar:
{json.dumps([p.to_dict() for p in event_protocols], indent=2, default=str)}

Genera un reporte de cumplimiento que incluya:
1. Estado general (compliant/non-compliant)
2. Lista de protocolos verificados con su estado
3. Violaciones encontradas (si las hay)
4. Recomendaciones

Formato: JSON con campos: overall_compliant, protocol_checks (array con protocol_id, compliant, violations), recommendations"""

        try:
            if not self.openrouter:
                # Sin IA, marcar todos como cumplidos
                for protocol in event_protocols:
                    self.protocols.record_compliance(
                        protocol_id=protocol.id,
                        is_compliant=True,
                        event_id=event_id
                    )
                
                return {
                    "event_id": event_id,
                    "compliant": True,
                    "message": "Protocolos verificados (sin IA)",
                    "checked_protocols": [p.id for p in event_protocols]
                }
            
            response = await self.openrouter.generate_text(
                prompt=prompt,
                model="anthropic/claude-3-haiku",
                max_tokens=1500,
                temperature=0.3  # Más determinístico para auditoría
            )
            
            # Usar helper mejorado para parsear respuesta
            compliance_data = AIHelper.parse_ai_response(
                response,
                expected_fields=["overall_compliant", "protocol_checks", "recommendations"]
            )
            
            # Si no se pudo parsear bien, usar fallback
            if compliance_data.get("parsed") is False:
                compliance_data = {
                    "overall_compliant": True,
                    "protocol_checks": [],
                    "recommendations": compliance_data.get("raw_content", "No se pudo analizar automáticamente")
                }
            
            # Registrar cumplimiento para cada protocolo
            checked_protocols = []
            for check in compliance_data.get("protocol_checks", []):
                protocol_id = check.get("protocol_id")
                is_compliant = check.get("compliant", True)
                violations = check.get("violations", [])
                
                if protocol_id:
                    self.protocols.record_compliance(
                        protocol_id=protocol_id,
                        is_compliant=is_compliant,
                        event_id=event_id,
                        violations=violations
                    )
                    checked_protocols.append(protocol_id)
            
            return {
                "event_id": event_id,
                "compliant": compliance_data.get("overall_compliant", True),
                "checked_protocols": checked_protocols,
                "recommendations": compliance_data.get("recommendations", ""),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            self._logger.error(f"Error checking protocol compliance: {str(e)}")
            return {
                "event_id": event_id,
                "compliant": False,
                "error": str(e),
                "checked_protocols": []
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Obtener datos para el dashboard.
        
        Returns:
            Datos consolidados para dashboard
        """
        today = datetime.now()
        upcoming_events = self.calendar.get_upcoming_events(days=7)
        pending_routines = self.routines.get_pending_routines()
        critical_protocols = self.protocols.get_protocols_by_priority(ProtocolPriority.CRITICAL)
        
        return {
            "artist_id": self.artist_id,
            "today": today.isoformat(),
            "upcoming_events": {
                "count": len(upcoming_events),
                "events": [e.to_dict() for e in upcoming_events[:5]]
            },
            "routines": {
                "pending_count": len(pending_routines),
                "pending": [r.to_dict() for r in pending_routines[:5]]
            },
            "protocols": {
                "critical_count": len(critical_protocols),
                "critical": [p.to_dict() for p in critical_protocols[:5]]
            },
            "wardrobe": {
                "total_items": len(self.wardrobe.get_all_items()),
                "total_outfits": len(self.wardrobe.get_all_outfits())
            }
        }
    
    def create_event_with_reminders(
        self,
        event: CalendarEvent,
        reminder_minutes: List[int] = [60, 30, 15]
    ) -> CalendarEvent:
        """
        Crear evento con recordatorios automáticos.
        
        Args:
            event: Evento a crear
            reminder_minutes: Minutos antes del evento para recordatorios
        
        Returns:
            Evento creado
        """
        # Validar tiempos
        is_valid, error_msg = Validator.validate_time_range(
            event.start_time,
            event.end_time,
            min_duration_minutes=0,
            max_duration_hours=24
        )
        if not is_valid:
            raise ValueError(error_msg)
        
        # Agregar evento
        created_event = self.calendar.add_event(event)
        
        # Crear recordatorios
        if self.notification_service:
            for minutes in reminder_minutes:
                if event.start_time > datetime.now() + timedelta(minutes=minutes):
                    self.notification_service.create_event_reminder(
                        artist_id=self.artist_id,
                        event_title=event.title,
                        event_time=event.start_time,
                        minutes_before=minutes
                    )
        
        # Guardar en BD
        if self.db_service:
            self.db_service.save_event(self.artist_id, created_event.to_dict())
        
        # Registrar métrica
        if self.analytics:
            self.analytics.record_metric(
                "event_created",
                1,
                tags={"artist_id": self.artist_id, "event_type": event.event_type.value}
            )
        
        return created_event
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtener estadísticas del artista.
        
        Args:
            days: Días hacia atrás
        
        Returns:
            Estadísticas
        """
        stats = {
            "artist_id": self.artist_id,
            "period_days": days,
            "events": {
                "total": len(self.calendar.events),
                "upcoming": len(self.calendar.get_upcoming_events(days=days)),
                "by_type": {}
            },
            "routines": {
                "total": len(self.routines.routines),
                "pending": len(self.routines.get_pending_routines()),
                "completion_rate": 0.0
            },
            "protocols": {
                "total": len(self.protocols.protocols),
                "critical": len(self.protocols.get_protocols_by_priority(ProtocolPriority.CRITICAL))
            },
            "wardrobe": {
                "items": len(self.wardrobe.get_all_items()),
                "outfits": len(self.wardrobe.get_all_outfits())
            }
        }
        
        # Estadísticas de eventos por tipo
        for event_type in EventType:
            events_of_type = self.calendar.get_events_by_type(event_type)
            stats["events"]["by_type"][event_type.value] = len(events_of_type)
        
        # Tasa de completación de rutinas
        if self.routines.routines:
            total_completion = sum(
                self.routines.get_completion_rate(routine_id, days=days)
                for routine_id in self.routines.routines.keys()
            )
            stats["routines"]["completion_rate"] = total_completion / len(self.routines.routines)
        
        # Agregar analytics si está disponible
        if self.analytics:
            analytics_stats = self.analytics.get_artist_statistics(self.artist_id, days=days)
            stats["analytics"] = analytics_stats
        
        return stats
    
    async def close(self):
        """Cerrar conexiones y guardar datos."""
        # Guardar en BD
        self._save_to_database()
        
        # Cerrar OpenRouter
        if self.openrouter:
            await self.openrouter.close()
        
        # Cerrar BD
        if self.db_service:
            self.db_service.close()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

