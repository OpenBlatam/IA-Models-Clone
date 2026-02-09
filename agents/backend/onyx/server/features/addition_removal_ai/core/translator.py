"""
Translator - Sistema de traducción automática
"""

import logging
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)


class Translator:
    """Traductor automático"""

    def __init__(self):
        """Inicializar traductor"""
        # Diccionario básico de traducciones comunes
        self.translation_dict = {
            # Español -> Inglés
            'hola': 'hello',
            'adiós': 'goodbye',
            'gracias': 'thank you',
            'por favor': 'please',
            'sí': 'yes',
            'no': 'no',
            'buenos días': 'good morning',
            'buenas tardes': 'good afternoon',
            'buenas noches': 'good night',
            
            # Inglés -> Español
            'hello': 'hola',
            'goodbye': 'adiós',
            'thank you': 'gracias',
            'please': 'por favor',
            'yes': 'sí',
            'no': 'no',
            'good morning': 'buenos días',
            'good afternoon': 'buenas tardes',
            'good night': 'buenas noches',
        }

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Traducir texto.

        Args:
            text: Texto a traducir
            target_language: Idioma destino (es, en, fr, etc.)
            source_language: Idioma origen (opcional, auto-detecta)

        Returns:
            Traducción
        """
        # Detectar idioma si no se especifica
        if not source_language:
            source_language = self._detect_language(text)
        
        # Si es el mismo idioma, devolver original
        if source_language == target_language:
            return {
                "original": text,
                "translated": text,
                "source_language": source_language,
                "target_language": target_language,
                "confidence": 1.0,
                "method": "no_translation_needed"
            }
        
        # Traducción básica usando diccionario
        # En producción, se usaría un servicio de traducción real
        translated = self._basic_translate(text, source_language, target_language)
        
        return {
            "original": text,
            "translated": translated,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": 0.7,  # Baja confianza para traducción básica
            "method": "dictionary_based"
        }

    def _detect_language(self, text: str) -> str:
        """
        Detectar idioma del texto.

        Args:
            text: Texto

        Returns:
            Código de idioma
        """
        # Detección básica basada en caracteres comunes
        text_lower = text.lower()
        
        # Caracteres específicos del español
        spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü', '¿', '¡']
        spanish_words = ['el', 'la', 'los', 'las', 'de', 'del', 'en', 'y', 'que', 'es']
        
        # Caracteres específicos del francés
        french_chars = ['à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ']
        french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'que', 'est']
        
        # Contar ocurrencias
        spanish_score = sum(1 for char in spanish_chars if char in text_lower)
        spanish_score += sum(1 for word in spanish_words if word in text_lower.split())
        
        french_score = sum(1 for char in french_chars if char in text_lower)
        french_score += sum(1 for word in french_words if word in text_lower.split())
        
        # Determinar idioma
        if spanish_score > french_score and spanish_score > 0:
            return 'es'
        elif french_score > 0:
            return 'fr'
        else:
            # Por defecto, inglés
            return 'en'

    def _basic_translate(
        self,
        text: str,
        source: str,
        target: str
    ) -> str:
        """
        Traducción básica usando diccionario.

        Args:
            text: Texto
            source: Idioma origen
            target: Idioma destino

        Returns:
            Texto traducido
        """
        # Dividir en palabras
        words = text.split()
        translated_words = []
        
        for word in words:
            # Limpiar palabra
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            # Buscar en diccionario
            if clean_word in self.translation_dict:
                translated = self.translation_dict[clean_word]
                # Preservar mayúsculas
                if word[0].isupper():
                    translated = translated.capitalize()
                translated_words.append(translated)
            else:
                # Mantener palabra original si no se encuentra
                translated_words.append(word)
        
        return ' '.join(translated_words)

    def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Traducir múltiples textos.

        Args:
            texts: Lista de textos
            target_language: Idioma destino
            source_language: Idioma origen

        Returns:
            Lista de traducciones
        """
        return [
            self.translate(text, target_language, source_language)
            for text in texts
        ]

    def get_supported_languages(self) -> List[str]:
        """
        Obtener idiomas soportados.

        Returns:
            Lista de códigos de idioma
        """
        return ['es', 'en', 'fr', 'de', 'it', 'pt']






