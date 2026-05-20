"""
Artist Manager AI
=================

Sistema de IA para gestión de artistas que ayuda a cumplir:
- Rutinas diarias
- Calendarios de eventos
- Protocolos de comportamiento
- Recomendaciones de vestimenta
- Recordatorios y notificaciones
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered artist management system for daily routines, calendars, protocols, wardrobe, and notifications"

# Try to import components with error handling
try:
    from .core.artist_manager import ArtistManager
    from .core.calendar_manager import CalendarManager
    from .core.routine_manager import RoutineManager
    from .core.protocol_manager import ProtocolManager
    from .core.wardrobe_manager import WardrobeManager
except ImportError:
    ArtistManager = None
    CalendarManager = None
    RoutineManager = None
    ProtocolManager = None
    WardrobeManager = None

try:
    from .api import create_app
except ImportError:
    create_app = None

__all__ = [
    "ArtistManager",
    "CalendarManager",
    "RoutineManager",
    "ProtocolManager",
    "WardrobeManager",
    "create_app",
]
