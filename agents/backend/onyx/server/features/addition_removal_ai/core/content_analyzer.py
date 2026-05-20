"""
Content Analyzer - Análisis avanzado de contenido
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedContentAnalyzer:
    """Analizador avanzado de contenido"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze_readability(self, content: str) -> Dict[str, Any]:
        """
        Analizar legibilidad del contenido.

        Args:
            content: Contenido a analizar

        Returns:
            Métricas de legibilidad
        """
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        # Fórmula de Flesch Reading Ease (simplificada)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Calcular sílabas aproximadas (simplificado)
        syllables = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllables / len(words) if words else 0
        
        # Flesch Reading Ease (simplificado)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Clasificar legibilidad
        if flesch_score >= 90:
            readability_level = "Muy fácil"
        elif flesch_score >= 80:
            readability_level = "Fácil"
        elif flesch_score >= 70:
            readability_level = "Bastante fácil"
        elif flesch_score >= 60:
            readability_level = "Estándar"
        elif flesch_score >= 50:
            readability_level = "Bastante difícil"
        elif flesch_score >= 30:
            readability_level = "Difícil"
        else:
            readability_level = "Muy difícil"
        
        return {
            "flesch_score": round(flesch_score, 2),
            "readability_level": readability_level,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "total_words": len(words),
            "total_sentences": len([s for s in sentences if s.strip()])
        }

    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra (aproximado)"""
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Ajustar para palabras que terminan en 'e'
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)

    def analyze_sentiment_basic(self, content: str) -> Dict[str, Any]:
        """
        Análisis básico de sentimiento.

        Args:
            content: Contenido

        Returns:
            Análisis de sentimiento
        """
        # Palabras positivas y negativas básicas (español e inglés)
        positive_words = [
            'bueno', 'excelente', 'genial', 'fantástico', 'maravilloso',
            'good', 'excellent', 'great', 'fantastic', 'wonderful', 'amazing'
        ]
        negative_words = [
            'malo', 'terrible', 'horrible', 'pésimo', 'mal',
            'bad', 'terrible', 'horrible', 'awful', 'worst'
        ]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
        else:
            score = (positive_count - negative_count) / total_sentiment_words
            if score > 0.3:
                sentiment = "positive"
            elif score < -0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "positive_words": positive_count,
            "negative_words": negative_count,
            "confidence": min(abs(score) * 2, 1.0)
        }

    def extract_keywords(self, content: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extraer palabras clave.

        Args:
            content: Contenido
            top_n: Número de palabras clave

        Returns:
            Lista de palabras clave
        """
        # Remover stopwords básicas
        stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it'
        }
        
        words = re.findall(r'\b\w{4,}\b', content.lower())
        words = [w for w in words if w not in stopwords]
        
        word_freq = Counter(words)
        top_words = word_freq.most_common(top_n)
        
        return [
            {
                "word": word,
                "frequency": count,
                "percentage": round((count / len(words)) * 100, 2) if words else 0
            }
            for word, count in top_words
        ]

    def analyze_structure(self, content: str) -> Dict[str, Any]:
        """
        Analizar estructura del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura
        """
        lines = content.split('\n')
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        # Detectar encabezados
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        
        # Detectar listas
        list_items = re.findall(r'^[\*\-\+]\s+(.+)$', content, re.MULTILINE)
        
        # Detectar enlaces
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        
        # Detectar código
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        inline_code = re.findall(r'`([^`]+)`', content)
        
        return {
            "total_lines": len(lines),
            "total_paragraphs": len(paragraphs),
            "headers": {
                "count": len(headers),
                "titles": headers[:5]  # Primeros 5
            },
            "lists": {
                "count": len(list_items),
                "items": list_items[:5]
            },
            "links": {
                "count": len(links),
                "links": links[:5]
            },
            "code": {
                "blocks": len(code_blocks),
                "inline": len(inline_code)
            },
            "avg_paragraph_length": sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            "avg_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0
        }






