"""
Content Metrics - Sistema de métricas de contenido
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class ContentMetrics:
    """Métricas de contenido"""

    def __init__(self):
        """Inicializar métricas"""
        pass

    def calculate_comprehensive_metrics(self, content: str) -> Dict[str, Any]:
        """
        Calcular métricas completas del contenido.

        Args:
            content: Contenido

        Returns:
            Métricas completas
        """
        # Métricas básicas
        basic_metrics = self._calculate_basic_metrics(content)
        
        # Métricas de estructura
        structure_metrics = self._calculate_structure_metrics(content)
        
        # Métricas de legibilidad
        readability_metrics = self._calculate_readability_metrics(content)
        
        # Métricas de contenido
        content_metrics = self._calculate_content_metrics(content)
        
        # Métricas de formato
        format_metrics = self._calculate_format_metrics(content)
        
        # Score general
        overall_score = self._calculate_overall_score(
            basic_metrics,
            structure_metrics,
            readability_metrics,
            content_metrics,
            format_metrics
        )
        
        return {
            "overall_score": overall_score,
            "basic": basic_metrics,
            "structure": structure_metrics,
            "readability": readability_metrics,
            "content": content_metrics,
            "format": format_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _calculate_basic_metrics(self, content: str) -> Dict[str, Any]:
        """Calcular métricas básicas"""
        char_count = len(content)
        char_count_no_spaces = len(content.replace(' ', ''))
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        line_count = len(content.split('\n'))
        
        return {
            "char_count": char_count,
            "char_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "line_count": line_count,
            "avg_words_per_sentence": word_count / sentence_count if sentence_count > 0 else 0,
            "avg_sentences_per_paragraph": sentence_count / paragraph_count if paragraph_count > 0 else 0
        }

    def _calculate_structure_metrics(self, content: str) -> Dict[str, Any]:
        """Calcular métricas de estructura"""
        lines = content.split('\n')
        
        # Headers
        headers = [line for line in lines if line.startswith('#')]
        header_count = len(headers)
        
        # Listas
        lists = [line for line in lines if re.match(r'^[\*\-\+]\s', line) or re.match(r'^\d+\.\s', line)]
        list_count = len(lists)
        
        # Links
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        link_count = len(links)
        
        # Imágenes
        images = re.findall(r'!\[([^\]]*)\]\(([^\)]+)\)', content)
        image_count = len(images)
        
        # Tablas
        tables = [line for line in lines if '|' in line and line.strip().startswith('|')]
        table_count = len(set(tables))  # Aproximación
        
        return {
            "header_count": header_count,
            "list_count": list_count,
            "link_count": link_count,
            "image_count": image_count,
            "table_count": table_count,
            "has_title": any(line.startswith('#') for line in lines[:3]),
            "structure_score": min(1.0, (header_count * 0.2 + list_count * 0.1 + link_count * 0.1) / 10)
        }

    def _calculate_readability_metrics(self, content: str) -> Dict[str, Any]:
        """Calcular métricas de legibilidad"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = content.split()
        
        if not sentences or not words:
            return {"readability_score": 0.5}
        
        # Longitud promedio de oraciones
        avg_sentence_length = len(words) / len(sentences)
        
        # Longitud promedio de palabras
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Palabras complejas (más de 3 sílabas estimadas)
        complex_words = sum(1 for w in words if self._estimate_syllables(w) > 3)
        complex_ratio = complex_words / len(words) if words else 0
        
        # Score de legibilidad (simplificado)
        readability_score = 1.0
        if avg_sentence_length > 25:
            readability_score -= 0.2
        if avg_word_length > 6:
            readability_score -= 0.2
        if complex_ratio > 0.2:
            readability_score -= 0.2
        
        readability_score = max(0.0, min(1.0, readability_score))
        
        return {
            "readability_score": readability_score,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "complex_word_ratio": complex_ratio
        }

    def _calculate_content_metrics(self, content: str) -> Dict[str, Any]:
        """Calcular métricas de contenido"""
        words = content.lower().split()
        
        # Diversidad de vocabulario
        unique_words = len(set(words))
        total_words = len(words)
        diversity_ratio = unique_words / total_words if total_words > 0 else 0
        
        # Palabras más frecuentes
        word_freq = Counter(words)
        most_common = word_freq.most_common(5)
        
        # Repetición
        if most_common:
            max_freq = most_common[0][1]
            repetition_ratio = max_freq / total_words if total_words > 0 else 0
        else:
            repetition_ratio = 0
        
        return {
            "diversity_ratio": diversity_ratio,
            "unique_words": unique_words,
            "total_words": total_words,
            "repetition_ratio": repetition_ratio,
            "most_common_words": [{"word": word, "count": count} for word, count in most_common]
        }

    def _calculate_format_metrics(self, content: str) -> Dict[str, Any]:
        """Calcular métricas de formato"""
        # Espacios dobles
        double_spaces = content.count('  ')
        
        # Puntuación al final
        has_final_punctuation = content and content[-1] in '.!?'
        
        # Mayúsculas al inicio de oraciones
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        capitalized = sum(1 for s in sentences if s and s[0].isupper())
        capitalization_ratio = capitalized / len(sentences) if sentences else 0
        
        return {
            "double_spaces": double_spaces,
            "has_final_punctuation": has_final_punctuation,
            "capitalization_ratio": capitalization_ratio,
            "format_score": (
                0.5 if has_final_punctuation else 0.0 +
                0.3 * capitalization_ratio +
                0.2 * (1.0 if double_spaces == 0 else 0.0)
            )
        }

    def _estimate_syllables(self, word: str) -> int:
        """Estimar sílabas"""
        word = word.lower()
        if not word:
            return 0
        
        if word.endswith('e'):
            word = word[:-1]
        
        vowels = re.findall(r'[aeiouáéíóúü]+', word)
        return len(vowels) if vowels else 1

    def _calculate_overall_score(
        self,
        basic: Dict[str, Any],
        structure: Dict[str, Any],
        readability: Dict[str, Any],
        content: Dict[str, Any],
        format_metrics: Dict[str, Any]
    ) -> float:
        """Calcular score general"""
        score = (
            structure.get("structure_score", 0.5) * 0.25 +
            readability.get("readability_score", 0.5) * 0.25 +
            content.get("diversity_ratio", 0.5) * 0.25 +
            format_metrics.get("format_score", 0.5) * 0.25
        )
        
        return max(0.0, min(1.0, score))

    def compare_metrics(
        self,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comparar dos conjuntos de métricas.

        Args:
            metrics1: Métricas 1
            metrics2: Métricas 2

        Returns:
            Comparación
        """
        basic1 = metrics1.get("basic", {})
        basic2 = metrics2.get("basic", {})
        
        return {
            "word_count_diff": basic2.get("word_count", 0) - basic1.get("word_count", 0),
            "sentence_count_diff": basic2.get("sentence_count", 0) - basic1.get("sentence_count", 0),
            "readability_diff": (
                metrics2.get("readability", {}).get("readability_score", 0) -
                metrics1.get("readability", {}).get("readability_score", 0)
            ),
            "overall_score_diff": (
                metrics2.get("overall_score", 0) -
                metrics1.get("overall_score", 0)
            )
        }






