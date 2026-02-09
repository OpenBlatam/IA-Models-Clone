"""
Redundancy Analyzer - Sistema de análisis de redundancia
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class RedundancyAnalyzer:
    """Analizador de redundancia"""

    def __init__(self):
        """Inicializar analizador"""
        pass

    def analyze(
        self,
        content: str,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analizar redundancia del contenido.

        Args:
            content: Contenido a analizar
            threshold: Umbral de redundancia

        Returns:
            Análisis de redundancia
        """
        if not content:
            return {
                "redundancy_score": 0.0,
                "is_redundant": False,
                "redundant_phrases": []
            }
        
        # Analizar repetición de palabras
        word_redundancy = self._analyze_word_redundancy(content)
        
        # Analizar repetición de frases
        phrase_redundancy = self._analyze_phrase_redundancy(content)
        
        # Analizar repetición de ideas
        idea_redundancy = self._analyze_idea_redundancy(content)
        
        # Calcular score general
        redundancy_score = (
            word_redundancy["score"] * 0.4 +
            phrase_redundancy["score"] * 0.35 +
            idea_redundancy["score"] * 0.25
        )
        
        # Combinar frases redundantes
        redundant_phrases = (
            word_redundancy["redundant_items"][:5] +
            phrase_redundancy["redundant_items"][:5] +
            idea_redundancy["redundant_items"][:5]
        )
        
        return {
            "redundancy_score": redundancy_score,
            "is_redundant": redundancy_score >= threshold,
            "word_redundancy": word_redundancy,
            "phrase_redundancy": phrase_redundancy,
            "idea_redundancy": idea_redundancy,
            "redundant_phrases": redundant_phrases[:10],
            "suggestions": self._generate_suggestions(redundancy_score, threshold)
        }

    def _analyze_word_redundancy(self, content: str) -> Dict[str, Any]:
        """Analizar redundancia de palabras"""
        words = re.findall(r'\b\w+\b', content.lower())
        
        if not words:
            return {"score": 0.0, "redundant_items": []}
        
        # Contar frecuencias
        word_freq = Counter(words)
        total_words = len(words)
        
        # Filtrar stop words
        stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'a',
            'y', 'o', 'que', 'es', 'son', 'the', 'a', 'an', 'and', 'or',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'
        }
        
        # Encontrar palabras repetidas excesivamente
        redundant_words = []
        for word, count in word_freq.items():
            if word not in stop_words and len(word) > 3:
                ratio = count / total_words
                if ratio > 0.05:  # Más del 5% del contenido
                    redundant_words.append({
                        "word": word,
                        "count": count,
                        "ratio": ratio
                    })
        
        # Calcular score
        if redundant_words:
            max_ratio = max(item["ratio"] for item in redundant_words)
            score = min(1.0, max_ratio * 5)  # Normalizar
        else:
            score = 0.0
        
        return {
            "score": score,
            "redundant_items": redundant_words[:10]
        }

    def _analyze_phrase_redundancy(self, content: str) -> Dict[str, Any]:
        """Analizar redundancia de frases"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {"score": 0.0, "redundant_items": []}
        
        # Generar n-gramas de palabras (frases de 3-5 palabras)
        redundant_phrases = []
        
        for n in range(3, 6):
            ngrams = []
            for sentence in sentences:
                words = sentence.lower().split()
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngrams.append(ngram)
            
            # Contar frecuencias
            ngram_freq = Counter(ngrams)
            
            # Encontrar n-gramas repetidos
            for ngram, count in ngram_freq.items():
                if count > 1 and len(ngram) > 10:  # Frases significativas
                    redundant_phrases.append({
                        "phrase": ngram,
                        "count": count
                    })
        
        # Calcular score
        if redundant_phrases:
            max_count = max(item["count"] for item in redundant_phrases)
            score = min(1.0, max_count / 5)  # Normalizar
        else:
            score = 0.0
        
        return {
            "score": score,
            "redundant_items": redundant_phrases[:10]
        }

    def _analyze_idea_redundancy(self, content: str) -> Dict[str, Any]:
        """Analizar redundancia de ideas"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return {"score": 0.0, "redundant_items": []}
        
        # Calcular similitud entre párrafos
        similarities = []
        redundant_paragraphs = []
        
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                similarity = self._calculate_similarity(paragraphs[i], paragraphs[j])
                similarities.append(similarity)
                
                if similarity > 0.7:  # Alta similitud
                    redundant_paragraphs.append({
                        "paragraph1": paragraphs[i][:50] + "...",
                        "paragraph2": paragraphs[j][:50] + "...",
                        "similarity": similarity
                    })
        
        # Calcular score
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            score = min(1.0, avg_similarity * 1.2)  # Normalizar
        else:
            score = 0.0
        
        return {
            "score": score,
            "redundant_items": redundant_paragraphs[:5]
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud entre dos textos"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _generate_suggestions(
        self,
        redundancy_score: float,
        threshold: float
    ) -> List[str]:
        """Generar sugerencias"""
        suggestions = []
        
        if redundancy_score >= threshold:
            suggestions.append("El contenido tiene alta redundancia. Considera eliminar repeticiones.")
            suggestions.append("Revisa las palabras y frases que se repiten frecuentemente.")
            suggestions.append("Combina párrafos similares para reducir redundancia.")
        elif redundancy_score > threshold * 0.7:
            suggestions.append("Hay cierta redundancia en el contenido. Revisa las repeticiones.")
        
        return suggestions

    def find_redundant_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Encontrar secciones redundantes.

        Args:
            content: Contenido

        Returns:
            Lista de secciones redundantes
        """
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        redundant_sections = []
        
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                similarity = self._calculate_similarity(paragraphs[i], paragraphs[j])
                
                if similarity > 0.7:
                    redundant_sections.append({
                        "section1_index": i,
                        "section2_index": j,
                        "section1_preview": paragraphs[i][:100],
                        "section2_preview": paragraphs[j][:100],
                        "similarity": similarity
                    })
        
        return redundant_sections






