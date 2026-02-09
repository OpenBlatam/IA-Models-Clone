"""
Internationalization (i18n) - Sistema de Internacionalización
================================================================

Sistema de traducción y soporte multi-idioma.
"""

import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Idiomas soportados."""
    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    ZH = "zh"  # Chinese
    JA = "ja"  # Japanese
    RU = "ru"  # Russian


@dataclass
class Translation:
    """Traducción."""
    key: str
    language: Language
    value: str
    context: Optional[str] = None


class I18nManager:
    """Gestor de internacionalización."""
    
    def __init__(self, default_language: Language = Language.EN):
        self.default_language = default_language
        self.translations: Dict[str, Dict[Language, str]] = {}
        self._load_default_translations()
    
    def _load_default_translations(self):
        """Cargar traducciones por defecto."""
        default_translations = {
            "chat.paused": {
                Language.EN: "Chat paused",
                Language.ES: "Chat pausado",
                Language.FR: "Chat en pause",
                Language.DE: "Chat pausiert",
            },
            "chat.resumed": {
                Language.EN: "Chat resumed",
                Language.ES: "Chat reanudado",
                Language.FR: "Chat repris",
                Language.DE: "Chat fortgesetzt",
            },
            "chat.stopped": {
                Language.EN: "Chat stopped",
                Language.ES: "Chat detenido",
                Language.FR: "Chat arrêté",
                Language.DE: "Chat gestoppt",
            },
            "error.session_not_found": {
                Language.EN: "Session not found",
                Language.ES: "Sesión no encontrada",
                Language.FR: "Session non trouvée",
                Language.DE: "Sitzung nicht gefunden",
            },
            "error.unauthorized": {
                Language.EN: "Unauthorized access",
                Language.ES: "Acceso no autorizado",
                Language.FR: "Accès non autorisé",
                Language.DE: "Unbefugter Zugriff",
            },
        }
        
        for key, translations in default_translations.items():
            self.translations[key] = translations
    
    def translate(
        self,
        key: str,
        language: Optional[Language] = None,
        **kwargs,
    ) -> str:
        """
        Traducir clave.
        
        Args:
            key: Clave de traducción
            language: Idioma (opcional, usa default si no se especifica)
            **kwargs: Variables para interpolación
        
        Returns:
            Texto traducido
        """
        lang = language or self.default_language
        
        # Buscar traducción
        translation = None
        if key in self.translations:
            translation = self.translations[key].get(lang)
        
        # Si no hay traducción, usar inglés o default
        if not translation:
            if key in self.translations:
                translation = self.translations[key].get(Language.EN)
        
        # Si aún no hay, usar la clave
        if not translation:
            translation = key
        
        # Interpolación de variables
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except KeyError:
                logger.warning(f"Missing variable in translation {key}")
        
        return translation
    
    def add_translation(
        self,
        key: str,
        language: Language,
        value: str,
    ):
        """Agregar traducción."""
        if key not in self.translations:
            self.translations[key] = {}
        
        self.translations[key][language] = value
        logger.debug(f"Added translation: {key} ({language.value})")
    
    def get_supported_languages(self) -> list:
        """Obtener idiomas soportados."""
        return [lang.value for lang in Language]
    
    def set_default_language(self, language: Language):
        """Establecer idioma por defecto."""
        self.default_language = language
        logger.info(f"Default language set to: {language.value}")
































