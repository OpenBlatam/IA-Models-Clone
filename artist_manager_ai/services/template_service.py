"""
Template Service
================

Servicio de plantillas para eventos y rutinas.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EventTemplate:
    """Plantilla de evento."""
    id: str
    name: str
    title: str
    description: str
    event_type: str
    default_duration_hours: float = 2.0
    default_location: Optional[str] = None
    default_protocol_requirements: List[str] = None
    default_wardrobe_requirements: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.default_protocol_requirements is None:
            self.default_protocol_requirements = []
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)


@dataclass
class RoutineTemplate:
    """Plantilla de rutina."""
    id: str
    name: str
    title: str
    description: str
    routine_type: str
    scheduled_time: str  # HH:MM format
    duration_minutes: int
    priority: int = 5
    days_of_week: List[int] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.days_of_week is None:
            self.days_of_week = list(range(7))
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return asdict(self)


class TemplateService:
    """Servicio de plantillas."""
    
    def __init__(self):
        """Inicializar servicio de plantillas."""
        self.event_templates: Dict[str, EventTemplate] = {}
        self.routine_templates: Dict[str, RoutineTemplate] = {}
        self._logger = logger
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Cargar plantillas por defecto."""
        # Plantillas de eventos por defecto
        default_event_templates = [
            EventTemplate(
                id="concert",
                name="Concierto",
                title="Concierto",
                description="Concierto en vivo",
                event_type="concert",
                default_duration_hours=3.0,
                default_protocol_requirements=["Llegar 2 horas antes", "Soundcheck obligatorio"],
                default_wardrobe_requirements="Vestimenta artística",
                tags=["performance", "music"]
            ),
            EventTemplate(
                id="interview",
                name="Entrevista",
                title="Entrevista",
                description="Entrevista con medios",
                event_type="interview",
                default_duration_hours=1.0,
                default_protocol_requirements=["Llegar 15 minutos antes", "No usar teléfono"],
                default_wardrobe_requirements="Vestimenta profesional",
                tags=["media", "promotion"]
            ),
            EventTemplate(
                id="photoshoot",
                name="Sesión de Fotos",
                title="Sesión de Fotos",
                description="Sesión fotográfica",
                event_type="photoshoot",
                default_duration_hours=2.0,
                default_protocol_requirements=["Llegar 1 hora antes", "Traer cambios de ropa"],
                tags=["media", "promotion"]
            )
        ]
        
        for template in default_event_templates:
            self.event_templates[template.id] = template
        
        # Plantillas de rutinas por defecto
        default_routine_templates = [
            RoutineTemplate(
                id="morning_exercise",
                name="Ejercicio Matutino",
                title="Ejercicio Matutino",
                description="30 minutos de ejercicio",
                routine_type="morning",
                scheduled_time="07:00",
                duration_minutes=30,
                priority=8,
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                tags=["health", "fitness"]
            ),
            RoutineTemplate(
                id="vocal_warmup",
                name="Calentamiento Vocal",
                title="Calentamiento Vocal",
                description="Calentamiento vocal antes de presentaciones",
                routine_type="morning",
                scheduled_time="09:00",
                duration_minutes=20,
                priority=9,
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                tags=["vocal", "performance"]
            )
        ]
        
        for template in default_routine_templates:
            self.routine_templates[template.id] = template
    
    def add_event_template(self, template: EventTemplate) -> EventTemplate:
        """
        Agregar plantilla de evento.
        
        Args:
            template: Plantilla a agregar
        
        Returns:
            Plantilla agregada
        """
        self.event_templates[template.id] = template
        self._logger.info(f"Added event template: {template.name}")
        return template
    
    def get_event_template(self, template_id: str) -> Optional[EventTemplate]:
        """
        Obtener plantilla de evento.
        
        Args:
            template_id: ID de la plantilla
        
        Returns:
            Plantilla o None
        """
        return self.event_templates.get(template_id)
    
    def list_event_templates(self, tags: Optional[List[str]] = None) -> List[EventTemplate]:
        """
        Listar plantillas de eventos.
        
        Args:
            tags: Filtrar por tags (opcional)
        
        Returns:
            Lista de plantillas
        """
        templates = list(self.event_templates.values())
        
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]
        
        return templates
    
    def create_event_from_template(
        self,
        template_id: str,
        start_time: datetime,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear evento desde plantilla.
        
        Args:
            template_id: ID de la plantilla
            start_time: Hora de inicio
            customizations: Personalizaciones (opcional)
        
        Returns:
            Datos del evento
        """
        template = self.get_event_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        from datetime import timedelta
        
        end_time = start_time + timedelta(hours=template.default_duration_hours)
        
        event_data = {
            "title": customizations.get("title") if customizations else template.title,
            "description": customizations.get("description") if customizations else template.description,
            "event_type": template.event_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "location": customizations.get("location") if customizations else template.default_location,
            "protocol_requirements": customizations.get("protocol_requirements") if customizations else template.default_protocol_requirements,
            "wardrobe_requirements": customizations.get("wardrobe_requirements") if customizations else template.default_wardrobe_requirements
        }
        
        return event_data
    
    def add_routine_template(self, template: RoutineTemplate) -> RoutineTemplate:
        """
        Agregar plantilla de rutina.
        
        Args:
            template: Plantilla a agregar
        
        Returns:
            Plantilla agregada
        """
        self.routine_templates[template.id] = template
        self._logger.info(f"Added routine template: {template.name}")
        return template
    
    def get_routine_template(self, template_id: str) -> Optional[RoutineTemplate]:
        """
        Obtener plantilla de rutina.
        
        Args:
            template_id: ID de la plantilla
        
        Returns:
            Plantilla o None
        """
        return self.routine_templates.get(template_id)
    
    def list_routine_templates(self, tags: Optional[List[str]] = None) -> List[RoutineTemplate]:
        """
        Listar plantillas de rutinas.
        
        Args:
            tags: Filtrar por tags (opcional)
        
        Returns:
            Lista de plantillas
        """
        templates = list(self.routine_templates.values())
        
        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]
        
        return templates
    
    def create_routine_from_template(
        self,
        template_id: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear rutina desde plantilla.
        
        Args:
            template_id: ID de la plantilla
            customizations: Personalizaciones (opcional)
        
        Returns:
            Datos de la rutina
        """
        template = self.get_routine_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        routine_data = {
            "title": customizations.get("title") if customizations else template.title,
            "description": customizations.get("description") if customizations else template.description,
            "routine_type": template.routine_type,
            "scheduled_time": customizations.get("scheduled_time") if customizations else template.scheduled_time,
            "duration_minutes": customizations.get("duration_minutes") if customizations else template.duration_minutes,
            "priority": customizations.get("priority") if customizations else template.priority,
            "days_of_week": customizations.get("days_of_week") if customizations else template.days_of_week
        }
        
        return routine_data




