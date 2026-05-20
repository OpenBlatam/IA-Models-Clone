"""
Coherence Analyzer - Sistema de análisis de coherencia textual
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class CoherenceAnalyzer:
    """Analizador de coherencia textual"""

    def __init__(self):
        """Inicializar analizador"""
        # Palabras de transición
        self.transitions = {
            'además', 'también', 'asimismo', 'igualmente', 'del mismo modo',
            'sin embargo', 'no obstante', 'pero', 'aunque', 'a pesar de',
            'por lo tanto', 'en consecuencia', 'así que', 'por eso', 'por ello',
            'en primer lugar', 'en segundo lugar', 'finalmente', 'en conclusión',
            'además', 'also', 'furthermore', 'moreover', 'however', 'nevertheless',
            'therefore', 'consequently', 'thus', 'first', 'second', 'finally'
        }
        
        # Conectores de coherencia
        self.coherence_connectors = {
            'esto', 'eso', 'aquello', 'estos', 'esos', 'aquellos',
            'este', 'esta', 'ese', 'esa', 'aquel', 'aquella',
            'this', 'that', 'these', 'those', 'it', 'they'
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar coherencia del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de coherencia
        """
        sentences = self._split_sentences(content)
        
        if len(sentences) < 2:
            return {
                "coherence_score": 0.5,
                "is_coherent": True,
                "error": "Contenido muy corto para análisis de coherencia"
            }
        
        # Análisis de transiciones
        transition_score = self._analyze_transitions(content, sentences)
        
        # Análisis de referencias
        reference_score = self._analyze_references(content, sentences)
        
        # Análisis de flujo temático
        thematic_score = self._analyze_thematic_flow(sentences)
        
        # Análisis de conectores
        connector_score = self._analyze_connectors(content, sentences)
        
        # Calcular score general
        coherence_score = (
            transition_score * 0.3 +
            reference_score * 0.25 +
            thematic_score * 0.25 +
            connector_score * 0.2
        )
        
        return {
            "coherence_score": coherence_score,
            "is_coherent": coherence_score >= 0.6,
            "transition_score": transition_score,
            "reference_score": reference_score,
            "thematic_score": thematic_score,
            "connector_score": connector_score,
            "suggestions": self._generate_coherence_suggestions(
                coherence_score,
                transition_score,
                reference_score,
                thematic_score
            )
        }

    def _split_sentences(self, content: str) -> List[str]:
        """Dividir en oraciones"""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _analyze_transitions(self, content: str, sentences: List[str]) -> float:
        """Analizar uso de transiciones"""
        content_lower = content.lower()
        
        # Contar transiciones
        transition_count = sum(1 for trans in self.transitions if trans in content_lower)
        
        # Calcular ratio
        if len(sentences) < 2:
            return 0.5
        
        # Ideal: al menos una transición cada 3-4 oraciones
        ideal_transitions = len(sentences) / 3.5
        transition_ratio = min(1.0, transition_count / ideal_transitions) if ideal_transitions > 0 else 0.0
        
        return transition_ratio

    def _analyze_references(self, content: str, sentences: List[str]) -> float:
        """Analizar referencias (pronombres, demostrativos)"""
        content_lower = content.lower()
        
        # Contar referencias
        reference_count = sum(1 for ref in self.coherence_connectors if ref in content_lower)
        
        # Calcular score
        if len(sentences) < 2:
            return 0.5
        
        # Ideal: algunas referencias pero no demasiadas
        ideal_references = len(sentences) * 0.3
        if ideal_references == 0:
            return 0.5
        
        reference_ratio = reference_count / ideal_references
        # Normalizar: demasiadas o muy pocas es malo
        if reference_ratio > 2.0:
            score = 1.0 - (reference_ratio - 2.0) / 2.0
        else:
            score = min(1.0, reference_ratio / 2.0)
        
        return max(0.0, min(1.0, score))

    def _analyze_thematic_flow(self, sentences: List[str]) -> float:
        """Analizar flujo temático"""
        if len(sentences) < 2:
            return 0.5
        
        # Extraer palabras clave de cada oración
        sentence_keywords = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            # Filtrar stop words
            stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o', 'que', 'es', 'son',
                         'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            sentence_keywords.append(set(keywords[:5]))  # Top 5 keywords
        
        # Calcular solapamiento entre oraciones adyacentes
        overlaps = []
        for i in range(len(sentence_keywords) - 1):
            set1 = sentence_keywords[i]
            set2 = sentence_keywords[i + 1]
            
            if set1 and set2:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                overlap = intersection / union if union > 0 else 0.0
                overlaps.append(overlap)
        
        if not overlaps:
            return 0.5
        
        # Score basado en solapamiento promedio
        avg_overlap = sum(overlaps) / len(overlaps)
        return min(1.0, avg_overlap * 2)  # Normalizar

    def _analyze_connectors(self, content: str, sentences: List[str]) -> float:
        """Analizar conectores de coherencia"""
        content_lower = content.lower()
        
        # Conectores comunes
        connectors = {
            'y', 'o', 'pero', 'aunque', 'sin embargo', 'por lo tanto',
            'and', 'or', 'but', 'although', 'however', 'therefore'
        }
        
        connector_count = sum(1 for conn in connectors if conn in content_lower)
        
        if len(sentences) < 2:
            return 0.5
        
        # Ideal: algunos conectores
        ideal_connectors = len(sentences) * 0.2
        if ideal_connectors == 0:
            return 0.5
        
        connector_ratio = connector_count / ideal_connectors
        score = min(1.0, connector_ratio)
        
        return score

    def _generate_coherence_suggestions(
        self,
        coherence_score: float,
        transition_score: float,
        reference_score: float,
        thematic_score: float
    ) -> List[str]:
        """Generar sugerencias de coherencia"""
        suggestions = []
        
        if coherence_score < 0.6:
            suggestions.append("El contenido tiene baja coherencia. Considera mejorar la conexión entre ideas.")
        
        if transition_score < 0.5:
            suggestions.append("Faltan palabras de transición. Considera agregar conectores como 'además', 'sin embargo', 'por lo tanto'.")
        
        if reference_score < 0.4:
            suggestions.append("Faltan referencias entre oraciones. Considera usar pronombres y demostrativos para conectar ideas.")
        
        if thematic_score < 0.5:
            suggestions.append("El flujo temático es débil. Considera mantener un tema consistente entre oraciones.")
        
        return suggestions

    def analyze_paragraph_coherence(self, content: str) -> Dict[str, Any]:
        """
        Analizar coherencia entre párrafos.

        Args:
            content: Contenido

        Returns:
            Análisis de coherencia entre párrafos
        """
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return {
                "paragraph_coherence": 0.5,
                "error": "Se necesitan al menos 2 párrafos"
            }
        
        # Calcular similitud temática entre párrafos adyacentes
        paragraph_similarities = []
        for i in range(len(paragraphs) - 1):
            similarity = self._calculate_paragraph_similarity(paragraphs[i], paragraphs[i + 1])
            paragraph_similarities.append(similarity)
        
        avg_similarity = sum(paragraph_similarities) / len(paragraph_similarities) if paragraph_similarities else 0.5
        
        return {
            "paragraph_coherence": avg_similarity,
            "paragraph_count": len(paragraphs),
            "similarities": paragraph_similarities
        }

    def _calculate_paragraph_similarity(self, para1: str, para2: str) -> float:
        """Calcular similitud entre párrafos"""
        words1 = set(re.findall(r'\b\w+\b', para1.lower()))
        words2 = set(re.findall(r'\b\w+\b', para2.lower()))
        
        # Filtrar stop words
        stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a', 'y', 'o', 'que', 'es', 'son',
                     'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
        
        words1 = {w for w in words1 if w not in stop_words and len(w) > 3}
        words2 = {w for w in words2 if w not in stop_words and len(w) > 3}
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0






