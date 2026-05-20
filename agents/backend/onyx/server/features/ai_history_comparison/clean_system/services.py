"""
Servicios de negocio simples y funcionales
=========================================

Solo la lógica esencial para analizar y comparar contenido.
"""

import re
import statistics
from typing import List, Dict, Any
from collections import Counter

from .models import HistoryEntry, ComparisonResult


class ContentAnalyzer:
    """Analizador de contenido simple y efectivo."""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor',
            'worst', 'dreadful', 'atrocious', 'appalling', 'deplorable'
        }
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar contenido y devolver métricas.
        
        Args:
            content: Contenido a analizar
            
        Returns:
            Diccionario con métricas
        """
        if not content or not content.strip():
            return self._empty_metrics()
        
        # Métricas básicas
        words = self._extract_words(content)
        sentences = self._extract_sentences(content)
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Calcular puntuaciones
        readability_score = self._calculate_readability(word_count, sentence_count, content)
        sentiment_score = self._calculate_sentiment(words)
        quality_score = self._calculate_quality(content, words, sentences)
        
        return {
            'quality_score': quality_score,
            'word_count': word_count,
            'readability_score': readability_score,
            'sentiment_score': sentiment_score,
            'metadata': {
                'sentence_count': sentence_count,
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'unique_words': len(set(words))
            }
        }
    
    def _extract_words(self, content: str) -> List[str]:
        """Extraer palabras del contenido."""
        return re.findall(r'\b[a-zA-Z]+\b', content.lower())
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extraer oraciones del contenido."""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_readability(self, word_count: int, sentence_count: int, content: str) -> float:
        """Calcular puntuación de legibilidad (simplificada)."""
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Longitud promedio de oración
        avg_sentence_length = word_count / sentence_count
        
        # Contar sílabas (simplificado)
        syllables = self._count_syllables(content)
        avg_syllables_per_word = syllables / word_count if word_count > 0 else 0
        
        # Fórmula simplificada de Flesch Reading Ease
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalizar a escala 0-1
        return max(0.0, min(1.0, readability / 100))
    
    def _calculate_sentiment(self, words: List[str]) -> float:
        """Calcular puntuación de sentimiento."""
        if not words:
            return 0.5  # Neutral
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return (sentiment + 1) / 2  # Normalizar a 0-1
    
    def _calculate_quality(self, content: str, words: List[str], sentences: List[str]) -> float:
        """Calcular puntuación de calidad general."""
        if not words:
            return 0.0
        
        # Factores de calidad
        readability = self._calculate_readability(len(words), len(sentences), content)
        sentiment = self._calculate_sentiment(words)
        
        # Diversidad de vocabulario
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words)
        
        # Consistencia de longitud de oraciones
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        if len(sentence_lengths) > 1:
            mean_length = statistics.mean(sentence_lengths)
            std_length = statistics.stdev(sentence_lengths)
            consistency = max(0.0, 1.0 - (std_length / mean_length)) if mean_length > 0 else 1.0
        else:
            consistency = 1.0
        
        # Puntuación de calidad ponderada
        quality = (
            readability * 0.4 +
            sentiment * 0.2 +
            vocabulary_diversity * 0.2 +
            consistency * 0.2
        )
        
        return min(1.0, quality)
    
    def _count_syllables(self, text: str) -> int:
        """Contar sílabas en el texto (simplificado)."""
        vowels = 'aeiouy'
        count = 0
        
        for word in text.lower().split():
            word_count = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        word_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Ajustar por 'e' silenciosa
            if word.endswith('e'):
                word_count -= 1
            
            # Cada palabra tiene al menos una sílaba
            if word_count == 0:
                word_count = 1
            
            count += word_count
        
        return count
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Métricas vacías para contenido vacío."""
        return {
            'quality_score': 0.0,
            'word_count': 0,
            'readability_score': 0.0,
            'sentiment_score': 0.5,
            'metadata': {}
        }


class ModelComparator:
    """Comparador de modelos simple y efectivo."""
    
    def compare(self, entry1: HistoryEntry, entry2: HistoryEntry) -> ComparisonResult:
        """
        Comparar dos entradas de historial.
        
        Args:
            entry1: Primera entrada
            entry2: Segunda entrada
            
        Returns:
            Resultado de la comparación
        """
        # Calcular similitud de contenido
        similarity = self._calculate_similarity(entry1.content, entry2.content)
        
        # Calcular diferencia de calidad
        quality_diff = abs(entry1.quality_score - entry2.quality_score)
        
        # Crear resultado
        result = ComparisonResult.create(
            model_a=entry1.model_version,
            model_b=entry2.model_version,
            similarity=similarity,
            quality_diff=quality_diff
        )
        
        # Agregar detalles
        result.details = {
            'entry1_quality': entry1.quality_score,
            'entry2_quality': entry2.quality_score,
            'entry1_word_count': entry1.word_count,
            'entry2_word_count': entry2.word_count,
            'entry1_readability': entry1.readability_score,
            'entry2_readability': entry2.readability_score,
            'entry1_sentiment': entry1.sentiment_score,
            'entry2_sentiment': entry2.sentiment_score
        }
        
        return result
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud entre dos contenidos."""
        if not content1 or not content2:
            return 0.0
        
        # Extraer palabras
        words1 = set(re.findall(r'\b[a-zA-Z]+\b', content1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]+\b', content2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calcular similitud de Jaccard
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


class QualityAssessor:
    """Evaluador de calidad simple y efectivo."""
    
    def assess(self, entry: HistoryEntry) -> Dict[str, Any]:
        """
        Evaluar la calidad de una entrada.
        
        Args:
            entry: Entrada a evaluar
            
        Returns:
            Evaluación de calidad
        """
        quality_score = entry.quality_score
        
        # Determinar nivel de calidad
        if quality_score >= 0.8:
            level = "excellent"
        elif quality_score >= 0.6:
            level = "good"
        elif quality_score >= 0.4:
            level = "fair"
        else:
            level = "poor"
        
        # Identificar fortalezas y debilidades
        strengths = []
        weaknesses = []
        
        if entry.readability_score >= 0.7:
            strengths.append("high_readability")
        else:
            weaknesses.append("low_readability")
        
        if entry.sentiment_score >= 0.6:
            strengths.append("positive_sentiment")
        elif entry.sentiment_score <= 0.4:
            weaknesses.append("negative_sentiment")
        
        if entry.word_count >= 50:
            strengths.append("adequate_length")
        else:
            weaknesses.append("too_short")
        
        return {
            'quality_score': quality_score,
            'quality_level': level,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': self._get_recommendations(weaknesses)
        }
    
    def _get_recommendations(self, weaknesses: List[str]) -> List[str]:
        """Obtener recomendaciones basadas en debilidades."""
        recommendations = []
        
        if "low_readability" in weaknesses:
            recommendations.append("Improve sentence structure and word choice")
        
        if "negative_sentiment" in weaknesses:
            recommendations.append("Consider more positive language")
        
        if "too_short" in weaknesses:
            recommendations.append("Add more detail and context")
        
        return recommendations




