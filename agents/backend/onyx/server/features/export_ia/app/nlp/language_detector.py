"""
Language Detector - Detector de idiomas
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import LanguageDetectionResult, Language

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detector de idiomas."""
    
    def __init__(self):
        self._initialized = False
        self.language_patterns = {
            Language.ENGLISH: {
                "common_words": ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"],
                "characters": "abcdefghijklmnopqrstuvwxyz"
            },
            Language.SPANISH: {
                "common_words": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para"],
                "characters": "abcdefghijklmnñopqrstuvwxyz"
            },
            Language.FRENCH: {
                "common_words": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne", "se"],
                "characters": "abcdefghijklmnopqrstuvwxyz"
            }
        }
    
    async def initialize(self):
        """Inicializar el detector."""
        self._initialized = True
        logger.info("Language Detector inicializado")
    
    async def shutdown(self):
        """Cerrar el detector."""
        self._initialized = False
        logger.info("Language Detector cerrado")
    
    async def detect(self, text: str) -> LanguageDetectionResult:
        """Detectar idioma del texto."""
        if not self._initialized:
            await self.initialize()
        
        # Convertir a minúsculas
        text_lower = text.lower()
        
        # Calcular puntuaciones para cada idioma
        language_scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            
            # Puntuación por palabras comunes
            for word in patterns["common_words"]:
                if word in text_lower:
                    score += 1
            
            # Puntuación por caracteres
            char_count = sum(1 for char in text_lower if char in patterns["characters"])
            char_score = char_count / len(text_lower) if text_lower else 0
            score += char_score * 10
            
            language_scores[language] = score
        
        # Determinar idioma con mayor puntuación
        if language_scores:
            detected_language = max(language_scores, key=language_scores.get)
            confidence = language_scores[detected_language] / sum(language_scores.values())
        else:
            detected_language = Language.UNKNOWN
            confidence = 0.0
        
        # Crear lista de idiomas alternativos
        alternative_languages = [
            {"language": lang.value, "score": score}
            for lang, score in sorted(language_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        
        return LanguageDetectionResult(
            text=text,
            detected_language=detected_language,
            confidence=confidence,
            alternative_languages=alternative_languages
        )
    
    async def health_check(self) -> bool:
        """Verificar salud del detector."""
        return self._initialized




