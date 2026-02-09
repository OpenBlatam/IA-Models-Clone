"""
i18n Service - Sistema de internacionalización
==============================================

Sistema multi-idioma para soportar diferentes idiomas.
"""

import logging
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Idiomas soportados"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"


class I18nService:
    """Servicio de internacionalización"""
    
    def __init__(self, default_language: Language = Language.SPANISH):
        """Inicializar servicio"""
        self.default_language = default_language
        self.translations = self._initialize_translations()
        logger.info(f"I18nService initialized with default language: {default_language.value}")
    
    def _initialize_translations(self) -> Dict[str, Dict[str, str]]:
        """Inicializar traducciones"""
        return {
            "en": {
                "welcome": "Welcome to AI Job Replacement Helper",
                "dashboard": "Dashboard",
                "profile": "Profile",
                "jobs": "Jobs",
                "steps": "Steps",
                "community": "Community",
                "settings": "Settings",
                "logout": "Logout",
            },
            "es": {
                "welcome": "Bienvenido a AI Job Replacement Helper",
                "dashboard": "Panel de Control",
                "profile": "Perfil",
                "jobs": "Trabajos",
                "steps": "Pasos",
                "community": "Comunidad",
                "settings": "Configuración",
                "logout": "Cerrar Sesión",
            },
            "fr": {
                "welcome": "Bienvenue dans AI Job Replacement Helper",
                "dashboard": "Tableau de bord",
                "profile": "Profil",
                "jobs": "Emplois",
                "steps": "Étapes",
                "community": "Communauté",
                "settings": "Paramètres",
                "logout": "Déconnexion",
            },
        }
    
    def translate(self, key: str, language: Optional[Language] = None) -> str:
        """Traducir clave"""
        lang = (language or self.default_language).value
        
        if lang in self.translations and key in self.translations[lang]:
            return self.translations[lang][key]
        
        # Fallback a idioma por defecto
        if key in self.translations[self.default_language.value]:
            return self.translations[self.default_language.value][key]
        
        # Fallback a la clave misma
        return key
    
    def get_available_languages(self) -> List[Dict[str, str]]:
        """Obtener idiomas disponibles"""
        return [
            {"code": lang.value, "name": lang.name}
            for lang in Language
        ]

