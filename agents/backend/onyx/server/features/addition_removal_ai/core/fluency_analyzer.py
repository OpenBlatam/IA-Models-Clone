"""
Fluency Analyzer - Sistema de análisis de fluidez
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class FluencyAnalyzer:
    """Analizador de fluidez"""

    def __init__(self):
        """Inicializar analizador"""
        # Conectores de fluidez
        self.fluency_connectors = {
            'además', 'también', 'asimismo', 'igualmente', 'del mismo modo',
            'sin embargo', 'no obstante', 'pero', 'aunque', 'a pesar de',
            'por lo tanto', 'en consecuencia', 'así que', 'por eso', 'por ello',
            'en primer lugar', 'en segundo lugar', 'finalmente', 'en conclusión',
            'also', 'furthermore', 'moreover', 'however', 'nevertheless',
            'therefore', 'consequently', 'thus', 'first', 'second', 'finally'
        }
        
        # Palabras de transición
        self.transition_words = {
            'entonces', 'luego', 'después', 'posteriormente', 'seguidamente',
            'then', 'next', 'after', 'subsequently', 'following'
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar fluidez del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de fluidez
        """
        sentences = self._split_sentences(content)
        
        if len(sentences) < 2:
            return {
                "fluency_score": 0.5,
                "error": "Se necesitan al menos 2 oraciones para analizar fluidez"
            }
        
        # Análisis de variación de longitud
        length_variation = self._analyze_length_variation(sentences)
        
        # Análisis de conectores
        connector_analysis = self._analyze_connectors(content, sentences)
        
        # Análisis de repetición de palabras
        repetition_analysis = self._analyze_repetition(sentences)
        
        # Análisis de inicio de oraciones
        sentence_start_analysis = self._analyze_sentence_starts(sentences)
        
        # Análisis de ritmo
        rhythm_analysis = self._analyze_rhythm(sentences)
        
        # Calcular score de fluidez
        fluency_score = (
            length_variation["score"] * 0.25 +
            connector_analysis["score"] * 0.25 +
            repetition_analysis["score"] * 0.20 +
            sentence_start_analysis["score"] * 0.15 +
            rhythm_analysis["score"] * 0.15
        )
        
        return {
            "fluency_score": fluency_score,
            "length_variation": length_variation,
            "connectors": connector_analysis,
            "repetition": repetition_analysis,
            "sentence_starts": sentence_start_analysis,
            "rhythm": rhythm_analysis,
            "suggestions": self._generate_fluency_suggestions(
                fluency_score,
                length_variation,
                connector_analysis,
                repetition_analysis
            )
        }

    def _split_sentences(self, content: str) -> List[str]:
        """Dividir en oraciones"""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]

    def _analyze_length_variation(self, sentences: List[str]) -> Dict[str, Any]:
        """Analizar variación de longitud"""
        lengths = [len(s.split()) for s in sentences]
        
        if not lengths:
            return {"score": 0.0, "avg_length": 0, "variation": 0}
        
        avg_length = sum(lengths) / len(lengths)
        
        # Calcular desviación estándar
        variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
        std_dev = variance ** 0.5
        
        # Score: más variación es mejor (hasta cierto punto)
        # Normalizar: score alto si hay variación moderada
        if std_dev == 0:
            score = 0.3  # Sin variación es malo
        else:
            # Ideal: std_dev entre 5-15 palabras
            if 5 <= std_dev <= 15:
                score = 1.0
            elif std_dev < 5:
                score = std_dev / 5
            else:
                score = max(0.5, 1.0 - (std_dev - 15) / 20)
        
        return {
            "score": score,
            "avg_length": avg_length,
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_deviation": std_dev
        }

    def _analyze_connectors(self, content: str, sentences: List[str]) -> Dict[str, Any]:
        """Analizar conectores"""
        content_lower = content.lower()
        
        # Contar conectores
        connector_count = sum(1 for conn in self.fluency_connectors if conn in content_lower)
        transition_count = sum(1 for trans in self.transition_words if trans in content_lower)
        
        total_connectors = connector_count + transition_count
        
        # Score: idealmente 1 conector cada 3-4 oraciones
        ideal_connectors = len(sentences) / 3.5
        if ideal_connectors == 0:
            return {"score": 0.5, "connector_count": 0, "transition_count": 0}
        
        connector_ratio = total_connectors / ideal_connectors
        score = min(1.0, connector_ratio)
        
        return {
            "score": score,
            "connector_count": connector_count,
            "transition_count": transition_count,
            "total_connectors": total_connectors
        }

    def _analyze_repetition(self, sentences: List[str]) -> Dict[str, Any]:
        """Analizar repetición de palabras"""
        # Contar palabras repetidas al inicio de oraciones
        sentence_starts = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                first_word = words[0].lower().rstrip('.,;:')
                sentence_starts.append(first_word)
        
        # Contar repeticiones
        start_counts = Counter(sentence_starts)
        repeated_starts = {word: count for word, count in start_counts.items() if count > 1}
        
        # Score: menos repeticiones es mejor
        if not sentence_starts:
            return {"score": 0.5, "repeated_starts": {}}
        
        repetition_ratio = sum(repeated_starts.values()) / len(sentence_starts)
        score = 1.0 - min(1.0, repetition_ratio * 2)  # Penalizar repeticiones
        
        return {
            "score": score,
            "repeated_starts": dict(list(repeated_starts.items())[:5])  # Top 5
        }

    def _analyze_sentence_starts(self, sentences: List[str]) -> Dict[str, Any]:
        """Analizar inicio de oraciones"""
        starts = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                first_word = words[0].lower().rstrip('.,;:')
                starts.append(first_word)
        
        # Diversidad de inicios
        unique_starts = len(set(starts))
        total_starts = len(starts)
        
        if total_starts == 0:
            return {"score": 0.0, "diversity": 0}
        
        diversity_ratio = unique_starts / total_starts
        score = diversity_ratio
        
        return {
            "score": score,
            "diversity": diversity_ratio,
            "unique_starts": unique_starts,
            "total_starts": total_starts
        }

    def _analyze_rhythm(self, sentences: List[str]) -> Dict[str, Any]:
        """Analizar ritmo"""
        # Calcular longitud de oraciones
        lengths = [len(s.split()) for s in sentences]
        
        if len(lengths) < 3:
            return {"score": 0.5, "rhythm_pattern": "insufficient_data"}
        
        # Analizar patrón de alternancia
        alternations = 0
        for i in range(len(lengths) - 1):
            # Verificar si alterna entre corta y larga
            if (lengths[i] < lengths[i+1] and i > 0 and lengths[i-1] > lengths[i]) or \
               (lengths[i] > lengths[i+1] and i > 0 and lengths[i-1] < lengths[i]):
                alternations += 1
        
        alternation_ratio = alternations / (len(lengths) - 1) if len(lengths) > 1 else 0
        
        # Score: algo de alternación es bueno, pero no demasiada
        if 0.3 <= alternation_ratio <= 0.7:
            score = 1.0
        else:
            score = min(1.0, alternation_ratio * 2)
        
        return {
            "score": score,
            "alternation_ratio": alternation_ratio,
            "rhythm_pattern": "varied" if alternation_ratio > 0.3 else "monotonous"
        }

    def _generate_fluency_suggestions(
        self,
        fluency_score: float,
        length_variation: Dict[str, Any],
        connectors: Dict[str, Any],
        repetition: Dict[str, Any]
    ) -> List[str]:
        """Generar sugerencias de fluidez"""
        suggestions = []
        
        if fluency_score < 0.6:
            suggestions.append("La fluidez del contenido puede mejorarse.")
        
        if length_variation.get("score", 0) < 0.5:
            suggestions.append("Varía la longitud de las oraciones para mejorar el ritmo.")
        
        if connectors.get("score", 0) < 0.5:
            suggestions.append("Agrega más conectores y palabras de transición.")
        
        if repetition.get("score", 0) < 0.5:
            suggestions.append("Evita repetir las mismas palabras al inicio de oraciones.")
        
        return suggestions






