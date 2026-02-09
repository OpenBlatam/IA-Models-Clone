"""
Sistema de Internacionalización
================================
Soporte multi-idioma
"""

from typing import Dict, Any, Optional
from enum import Enum
import structlog
import json
from pathlib import Path

logger = structlog.get_logger()


class Language(str, Enum):
    """Idiomas soportados"""
    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    PT = "pt"  # Portuguese
    IT = "it"  # Italian


class Translator:
    """Traductor"""
    
    def __init__(self, default_language: Language = Language.EN):
        """
        Inicializar traductor
        
        Args:
            default_language: Idioma por defecto
        """
        self.default_language = default_language
        self._translations: Dict[Language, Dict[str, str]] = {}
        self._load_translations()
        logger.info("Translator initialized", default_language=default_language.value)
    
    def _load_translations(self) -> None:
        """Cargar traducciones"""
        # Traducciones básicas
        self._translations[Language.EN] = {
            "validation_completed": "Validation completed",
            "profile_generated": "Profile generated",
            "report_ready": "Report ready",
            "recommendations_available": "Recommendations available",
            "high_confidence": "High confidence",
            "medium_confidence": "Medium confidence",
            "low_confidence": "Low confidence",
            "personality_traits": "Personality Traits",
            "emotional_state": "Emotional State",
            "risk_factors": "Risk Factors",
            "strengths": "Strengths",
            "recommendations": "Recommendations"
        }
        
        self._translations[Language.ES] = {
            "validation_completed": "Validación completada",
            "profile_generated": "Perfil generado",
            "report_ready": "Reporte listo",
            "recommendations_available": "Recomendaciones disponibles",
            "high_confidence": "Alta confianza",
            "medium_confidence": "Confianza media",
            "low_confidence": "Baja confianza",
            "personality_traits": "Rasgos de Personalidad",
            "emotional_state": "Estado Emocional",
            "risk_factors": "Factores de Riesgo",
            "strengths": "Fortalezas",
            "recommendations": "Recomendaciones"
        }
        
        self._translations[Language.FR] = {
            "validation_completed": "Validation terminée",
            "profile_generated": "Profil généré",
            "report_ready": "Rapport prêt",
            "recommendations_available": "Recommandations disponibles",
            "high_confidence": "Confiance élevée",
            "medium_confidence": "Confiance moyenne",
            "low_confidence": "Faible confiance",
            "personality_traits": "Traits de Personnalité",
            "emotional_state": "État Émotionnel",
            "risk_factors": "Facteurs de Risque",
            "strengths": "Forces",
            "recommendations": "Recommandations"
        }
    
    def translate(
        self,
        key: str,
        language: Optional[Language] = None,
        default: Optional[str] = None
    ) -> str:
        """
        Traducir clave
        
        Args:
            key: Clave a traducir
            language: Idioma (opcional)
            default: Valor por defecto (opcional)
            
        Returns:
            Texto traducido
        """
        lang = language or self.default_language
        translations = self._translations.get(lang, self._translations[Language.EN])
        
        return translations.get(key, default or key)
    
    def translate_dict(
        self,
        data: Dict[str, Any],
        language: Optional[Language] = None
    ) -> Dict[str, Any]:
        """
        Traducir diccionario
        
        Args:
            data: Diccionario a traducir
            language: Idioma (opcional)
            
        Returns:
            Diccionario traducido
        """
        lang = language or self.default_language
        translated = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                translated[key] = self.translate(value, language=lang, default=value)
            elif isinstance(value, dict):
                translated[key] = self.translate_dict(value, language=lang)
            elif isinstance(value, list):
                translated[key] = [
                    self.translate_dict(item, language=lang) if isinstance(item, dict)
                    else self.translate(item, language=lang, default=str(item)) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                translated[key] = value
        
        return translated
    
    def get_available_languages(self) -> List[str]:
        """
        Obtener idiomas disponibles
        
        Returns:
            Lista de códigos de idioma
        """
        return [lang.value for lang in self._translations.keys()]


# Instancia global del traductor
translator = Translator()




