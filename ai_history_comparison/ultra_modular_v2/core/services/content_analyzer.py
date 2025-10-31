"""
Content Analyzer Service
========================

Servicio para analizar contenido de texto.
Responsabilidad única: Analizar contenido y extraer métricas.
"""

import re
import statistics
from typing import Dict, Any, List
from collections import Counter


class ContentAnalyzer:
    """
    Servicio para analizar contenido de texto.
    
    Responsabilidad única: Analizar contenido y extraer métricas de calidad,
    legibilidad, sentimiento y otras características del texto.
    """
    
    def __init__(self):
        """Inicializar el analizador de contenido."""
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent',
            'perfect', 'incredible', 'marvelous', 'exceptional', 'remarkable'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor',
            'worst', 'dreadful', 'atrocious', 'appalling', 'deplorable',
            'terrible', 'horrible', 'disgusting', 'repulsive', 'abominable'
        }
    
    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analizar contenido y devolver métricas.
        
        Args:
            content: Contenido a analizar
            
        Returns:
            Diccionario con métricas de análisis
        """
        if not content or not content.strip():
            return self._empty_metrics()
        
        # Extraer elementos del texto
        words = self._extract_words(content)
        sentences = self._extract_sentences(content)
        
        # Calcular métricas básicas
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Calcular puntuaciones
        readability_score = self._calculate_readability(word_count, sentence_count, content)
        sentiment_score = self._calculate_sentiment(words)
        complexity_score = self._calculate_complexity(content, words)
        quality_score = self._calculate_quality(content, words, sentences)
        
        return {
            'quality_score': quality_score,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'readability_score': readability_score,
            'sentiment_score': sentiment_score,
            'complexity_score': complexity_score,
            'metadata': {
                'avg_word_length': self._calculate_avg_word_length(words),
                'unique_words': len(set(words)),
                'vocabulary_diversity': self._calculate_vocabulary_diversity(words),
                'sentence_length_variance': self._calculate_sentence_variance(sentences)
            }
        }
    
    def _extract_words(self, content: str) -> List[str]:
        """
        Extraer palabras del contenido.
        
        Args:
            content: Contenido de texto
            
        Returns:
            Lista de palabras
        """
        return re.findall(r'\b[a-zA-Z]+\b', content.lower())
    
    def _extract_sentences(self, content: str) -> List[str]:
        """
        Extraer oraciones del contenido.
        
        Args:
            content: Contenido de texto
            
        Returns:
            Lista de oraciones
        """
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_readability(self, word_count: int, sentence_count: int, content: str) -> float:
        """
        Calcular puntuación de legibilidad (Flesch Reading Ease simplificado).
        
        Args:
            word_count: Número de palabras
            sentence_count: Número de oraciones
            content: Contenido de texto
            
        Returns:
            Puntuación de legibilidad (0.0 - 1.0)
        """
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
        """
        Calcular puntuación de sentimiento.
        
        Args:
            words: Lista de palabras
            
        Returns:
            Puntuación de sentimiento (0.0 - 1.0)
        """
        if not words:
            return 0.5  # Neutral
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return (sentiment + 1) / 2  # Normalizar a 0-1
    
    def _calculate_complexity(self, content: str, words: List[str]) -> float:
        """
        Calcular puntuación de complejidad.
        
        Args:
            content: Contenido de texto
            words: Lista de palabras
            
        Returns:
            Puntuación de complejidad (0.0 - 1.0)
        """
        if not words:
            return 0.0
        
        # Factores de complejidad
        avg_word_length = sum(len(word) for word in words) / len(words)
        vocabulary_diversity = len(set(words)) / len(words)
        
        # Puntuación de complejidad normalizada
        complexity = (avg_word_length / 10 + vocabulary_diversity) / 2
        return min(1.0, complexity)
    
    def _calculate_quality(self, content: str, words: List[str], sentences: List[str]) -> float:
        """
        Calcular puntuación de calidad general.
        
        Args:
            content: Contenido de texto
            words: Lista de palabras
            sentences: Lista de oraciones
            
        Returns:
            Puntuación de calidad (0.0 - 1.0)
        """
        if not words:
            return 0.0
        
        # Calcular métricas individuales
        readability = self._calculate_readability(len(words), len(sentences), content)
        sentiment = self._calculate_sentiment(words)
        complexity = self._calculate_complexity(content, words)
        consistency = self._calculate_consistency(sentences)
        
        # Puntuación de calidad ponderada
        quality = (
            readability * 0.3 +
            sentiment * 0.2 +
            complexity * 0.2 +
            consistency * 0.3
        )
        
        return min(1.0, quality)
    
    def _calculate_consistency(self, sentences: List[str]) -> float:
        """
        Calcular puntuación de consistencia.
        
        Args:
            sentences: Lista de oraciones
            
        Returns:
            Puntuación de consistencia (0.0 - 1.0)
        """
        if len(sentences) < 2:
            return 1.0
        
        # Verificar consistencia de longitud de oraciones
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        if not sentence_lengths:
            return 1.0
        
        # Calcular coeficiente de variación
        mean_length = statistics.mean(sentence_lengths)
        if mean_length == 0:
            return 1.0
        
        std_length = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        cv = std_length / mean_length
        
        # Convertir a puntuación de consistencia (CV más bajo = mayor consistencia)
        return max(0.0, 1.0 - cv)
    
    def _calculate_avg_word_length(self, words: List[str]) -> float:
        """
        Calcular longitud promedio de palabras.
        
        Args:
            words: Lista de palabras
            
        Returns:
            Longitud promedio de palabras
        """
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _calculate_vocabulary_diversity(self, words: List[str]) -> float:
        """
        Calcular diversidad de vocabulario.
        
        Args:
            words: Lista de palabras
            
        Returns:
            Diversidad de vocabulario (0.0 - 1.0)
        """
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _calculate_sentence_variance(self, sentences: List[str]) -> float:
        """
        Calcular varianza de longitud de oraciones.
        
        Args:
            sentences: Lista de oraciones
            
        Returns:
            Varianza de longitud de oraciones
        """
        if len(sentences) < 2:
            return 0.0
        
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        return statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    
    def _count_syllables(self, text: str) -> int:
        """
        Contar sílabas en el texto (simplificado).
        
        Args:
            text: Texto a analizar
            
        Returns:
            Número de sílabas
        """
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
        """
        Métricas vacías para contenido vacío.
        
        Returns:
            Diccionario con métricas vacías
        """
        return {
            'quality_score': 0.0,
            'word_count': 0,
            'sentence_count': 0,
            'readability_score': 0.0,
            'sentiment_score': 0.5,
            'complexity_score': 0.0,
            'metadata': {}
        }




