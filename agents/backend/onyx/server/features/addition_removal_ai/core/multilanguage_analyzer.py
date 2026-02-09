"""
Multilanguage Analyzer - Sistema de análisis de contenido multiidioma
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class MultilanguageAnalyzer:
    """Analizador multiidioma"""

    def __init__(self):
        """Inicializar analizador"""
        # Patrones de idiomas comunes
        self.language_patterns = {
            "es": {
                "common_words": {"el", "la", "los", "las", "de", "del", "en", "a", "y", "o"},
                "characters": "áéíóúñü",
                "name": "Español"
            },
            "en": {
                "common_words": {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for"},
                "characters": "",
                "name": "English"
            },
            "fr": {
                "common_words": {"le", "la", "les", "de", "du", "des", "et", "ou", "dans", "sur"},
                "characters": "àâäéèêëïîôùûüÿç",
                "name": "Français"
            },
            "pt": {
                "common_words": {"o", "a", "os", "as", "de", "do", "da", "em", "e", "ou"},
                "characters": "áàâãéêíóôõúüç",
                "name": "Português"
            },
            "it": {
                "common_words": {"il", "la", "lo", "gli", "le", "di", "del", "in", "e", "o"},
                "characters": "àèéìíîòóùú",
                "name": "Italiano"
            },
            "de": {
                "common_words": {"der", "die", "das", "und", "oder", "in", "auf", "zu", "für"},
                "characters": "äöüß",
                "name": "Deutsch"
            }
        }

    def detect_language(self, content: str) -> Dict[str, Any]:
        """
        Detectar idioma del contenido.

        Args:
            content: Contenido

        Returns:
            Detección de idioma
        """
        content_lower = content.lower()
        words = content_lower.split()
        
        if not words:
            return {"error": "Contenido vacío"}
        
        language_scores = {}
        
        for lang_code, lang_info in self.language_patterns.items():
            score = 0.0
            
            # Contar palabras comunes
            common_words = lang_info["common_words"]
            word_matches = sum(1 for word in words if word in common_words)
            word_score = word_matches / len(words) if words else 0
            
            # Contar caracteres especiales
            special_chars = lang_info["characters"]
            if special_chars:
                char_matches = sum(1 for char in content if char.lower() in special_chars)
                char_score = char_matches / len(content) if content else 0
            else:
                char_score = 0.5  # Neutral para idiomas sin caracteres especiales
            
            # Score combinado
            score = (word_score * 0.7) + (char_score * 0.3)
            language_scores[lang_code] = {
                "score": score,
                "name": lang_info["name"],
                "word_matches": word_matches,
                "char_matches": char_matches if special_chars else 0
            }
        
        # Obtener idioma con mayor score
        best_language = max(language_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "detected_language": best_language[0],
            "language_name": best_language[1]["name"],
            "confidence": best_language[1]["score"],
            "all_scores": {
                lang: info["score"]
                for lang, info in language_scores.items()
            }
        }

    def analyze_multilanguage_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido multiidioma.

        Args:
            content: Contenido

        Returns:
            Análisis multiidioma
        """
        # Detectar idioma principal
        language_detection = self.detect_language(content)
        
        # Detectar múltiples idiomas (si hay mezcla)
        paragraphs = content.split('\n\n')
        language_distribution = Counter()
        
        for paragraph in paragraphs:
            if paragraph.strip():
                para_lang = self.detect_language(paragraph)
                if "detected_language" in para_lang:
                    language_distribution[para_lang["detected_language"]] += 1
        
        # Calcular porcentaje de cada idioma
        total_paragraphs = len([p for p in paragraphs if p.strip()])
        language_percentages = {
            lang: (count / total_paragraphs * 100) if total_paragraphs > 0 else 0
            for lang, count in language_distribution.items()
        }
        
        # Determinar si es multiidioma
        is_multilanguage = len(language_distribution) > 1
        
        return {
            "primary_language": language_detection.get("detected_language"),
            "primary_language_name": language_detection.get("language_name"),
            "confidence": language_detection.get("confidence", 0),
            "is_multilanguage": is_multilanguage,
            "language_distribution": language_percentages,
            "detected_languages": list(language_distribution.keys()),
            "total_paragraphs": total_paragraphs
        }

    def compare_languages(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """
        Comparar contenido en diferentes idiomas.

        Args:
            content1: Contenido 1
            content2: Contenido 2

        Returns:
            Comparación de idiomas
        """
        lang1 = self.detect_language(content1)
        lang2 = self.detect_language(content2)
        
        same_language = (
            lang1.get("detected_language") == lang2.get("detected_language")
        )
        
        return {
            "content1_language": lang1.get("detected_language"),
            "content1_name": lang1.get("language_name"),
            "content1_confidence": lang1.get("confidence", 0),
            "content2_language": lang2.get("detected_language"),
            "content2_name": lang2.get("language_name"),
            "content2_confidence": lang2.get("confidence", 0),
            "same_language": same_language,
            "language_match": same_language
        }

    def get_language_statistics(
        self,
        contents: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de idiomas de múltiples contenidos.

        Args:
            contents: Lista de contenidos (cada uno con 'id' y 'content')

        Returns:
            Estadísticas de idiomas
        """
        language_counts = Counter()
        language_details = {}
        
        for content_item in contents:
            content_id = content_item.get("id", "unknown")
            content = content_item.get("content", "")
            
            lang_detection = self.detect_language(content)
            detected_lang = lang_detection.get("detected_language", "unknown")
            
            language_counts[detected_lang] += 1
            
            if detected_lang not in language_details:
                language_details[detected_lang] = {
                    "name": lang_detection.get("language_name", "Unknown"),
                    "content_ids": []
                }
            
            language_details[detected_lang]["content_ids"].append(content_id)
        
        total = len(contents)
        
        return {
            "total_contents": total,
            "language_distribution": {
                lang: {
                    "count": count,
                    "percentage": (count / total * 100) if total > 0 else 0,
                    "name": language_details.get(lang, {}).get("name", "Unknown")
                }
                for lang, count in language_counts.items()
            },
            "language_details": language_details,
            "most_common_language": language_counts.most_common(1)[0][0] if language_counts else None
        }






