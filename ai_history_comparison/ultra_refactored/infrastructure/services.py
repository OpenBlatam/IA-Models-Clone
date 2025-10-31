"""
Infrastructure Services - Servicios de Infraestructura
====================================================

Implementaciones concretas de servicios de análisis y procesamiento.
"""

import re
import math
from typing import List, Dict, Any
from collections import Counter
import asyncio

from ..domain.models import HistoryEntry
from ..domain.value_objects import (
    ContentMetrics, 
    QualityScore, 
    SimilarityScore,
    SentimentAnalysis,
    TextComplexity
)
from ..domain.exceptions import AnalysisException
from ..application.interfaces import (
    IContentAnalyzer,
    IQualityAssessor,
    ISimilarityCalculator
)


class TextContentAnalyzer(IContentAnalyzer):
    """
    Analizador de contenido de texto.
    
    Implementación básica para análisis de contenido textual.
    """
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'can', 'only', 'other',
            'new', 'some', 'could', 'these', 'may', 'say', 'her', 'than',
            'would', 'like', 'so', 'these', 'him', 'into', 'has', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
            'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day',
            'did', 'get', 'come', 'made', 'may', 'part'
        }
    
    async def analyze_content(self, content: str) -> ContentMetrics:
        """Analizar contenido y extraer métricas."""
        try:
            # Métricas básicas
            words = self._extract_words(content)
            sentences = self._extract_sentences(content)
            paragraphs = self._extract_paragraphs(content)
            
            word_count = len(words)
            sentence_count = len(sentences)
            character_count = len(content)
            paragraph_count = len(paragraphs)
            
            # Métricas calculadas
            avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            unique_words = len(set(word.lower() for word in words))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            return ContentMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                character_count=character_count,
                paragraph_count=paragraph_count,
                avg_word_length=avg_word_length,
                avg_sentence_length=avg_sentence_length,
                unique_words=unique_words,
                vocabulary_richness=vocabulary_richness
            )
            
        except Exception as e:
            raise AnalysisException(f"Failed to analyze content: {e}", "content_analysis")
    
    async def extract_keywords(self, content: str) -> List[str]:
        """Extraer palabras clave del contenido."""
        try:
            words = self._extract_words(content)
            
            # Filtrar stop words y palabras cortas
            keywords = [
                word.lower() for word in words
                if len(word) > 3 and word.lower() not in self.stop_words
            ]
            
            # Contar frecuencia y devolver las más comunes
            word_freq = Counter(keywords)
            return [word for word, freq in word_freq.most_common(10)]
            
        except Exception as e:
            raise AnalysisException(f"Failed to extract keywords: {e}", "keyword_extraction")
    
    async def detect_language(self, content: str) -> str:
        """Detectar idioma del contenido (implementación básica)."""
        try:
            # Implementación básica basada en palabras comunes
            words = self._extract_words(content)
            word_count = len(words)
            
            if word_count == 0:
                return "unknown"
            
            # Palabras comunes en inglés
            english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            english_count = sum(1 for word in words if word.lower() in english_words)
            
            # Palabras comunes en español
            spanish_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para'}
            spanish_count = sum(1 for word in words if word.lower() in spanish_words)
            
            # Determinar idioma basado en la proporción
            if english_count / word_count > 0.1:
                return "en"
            elif spanish_count / word_count > 0.1:
                return "es"
            else:
                return "unknown"
                
        except Exception as e:
            raise AnalysisException(f"Failed to detect language: {e}", "language_detection")
    
    async def analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analizar sentimiento del contenido."""
        try:
            words = self._extract_words(content)
            word_count = len(words)
            
            if word_count == 0:
                return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
            # Palabras positivas y negativas básicas
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'awesome', 'brilliant', 'outstanding', 'perfect', 'love', 'like',
                'happy', 'joy', 'pleasure', 'satisfied', 'content', 'pleased'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
                'dislike', 'angry', 'sad', 'disappointed', 'frustrated', 'annoyed',
                'upset', 'worried', 'concerned', 'problem', 'issue', 'error'
            }
            
            positive_count = sum(1 for word in words if word.lower() in positive_words)
            negative_count = sum(1 for word in words if word.lower() in negative_words)
            
            positive_score = positive_count / word_count
            negative_score = negative_count / word_count
            neutral_score = 1.0 - positive_score - negative_score
            
            return {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": max(0.0, neutral_score)
            }
            
        except Exception as e:
            raise AnalysisException(f"Failed to analyze sentiment: {e}", "sentiment_analysis")
    
    def _extract_words(self, content: str) -> List[str]:
        """Extraer palabras del contenido."""
        return re.findall(r'\b\w+\b', content)
    
    def _extract_sentences(self, content: str) -> List[str]:
        """Extraer oraciones del contenido."""
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """Extraer párrafos del contenido."""
        paragraphs = content.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]


class BasicQualityAssessor(IQualityAssessor):
    """
    Evaluador básico de calidad.
    
    Implementación simple para evaluación de calidad de contenido.
    """
    
    def __init__(self):
        self.content_analyzer = TextContentAnalyzer()
    
    async def assess_quality(self, entry: HistoryEntry) -> QualityScore:
        """Evaluar calidad de una entrada."""
        try:
            # Obtener métricas de contenido
            content_metrics = await self.content_analyzer.analyze_content(entry.content)
            
            # Evaluar diferentes aspectos
            readability_score = await self.assess_readability(entry.content)
            coherence_score = await self.assess_coherence(entry.content)
            relevance_score = await self.assess_relevance(entry.content)
            completeness_score = await self._assess_completeness(content_metrics)
            accuracy_score = await self._assess_accuracy(entry.content)
            
            # Calcular score overall
            overall_score = (
                readability_score + coherence_score + relevance_score + 
                completeness_score + accuracy_score
            ) / 5.0
            
            return QualityScore(
                overall_score=overall_score,
                readability_score=readability_score,
                coherence_score=coherence_score,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score
            )
            
        except Exception as e:
            raise AnalysisException(f"Failed to assess quality: {e}", "quality_assessment")
    
    async def assess_readability(self, content: str) -> float:
        """Evaluar legibilidad del contenido."""
        try:
            words = re.findall(r'\b\w+\b', content)
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not words or not sentences:
                return 0.0
            
            # Fórmula simplificada de Flesch Reading Ease
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            # Fórmula de Flesch Reading Ease
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalizar a 0-1
            return max(0.0, min(1.0, score / 100.0))
            
        except Exception as e:
            raise AnalysisException(f"Failed to assess readability: {e}", "readability_assessment")
    
    async def assess_coherence(self, content: str) -> float:
        """Evaluar coherencia del contenido."""
        try:
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 1.0  # Contenido muy corto se considera coherente
            
            # Análisis básico de coherencia
            coherence_score = 0.0
            
            # Verificar conectores
            connectors = {'however', 'therefore', 'moreover', 'furthermore', 'additionally', 'consequently'}
            connector_count = sum(1 for sentence in sentences for connector in connectors if connector in sentence.lower())
            coherence_score += min(0.3, connector_count * 0.1)
            
            # Verificar repetición de palabras clave
            words = re.findall(r'\b\w+\b', content.lower())
            word_freq = Counter(words)
            common_words = [word for word, freq in word_freq.most_common(5) if freq > 1]
            coherence_score += min(0.3, len(common_words) * 0.06)
            
            # Verificar longitud de oraciones (consistencia)
            sentence_lengths = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]
            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
                consistency = max(0.0, 1.0 - (variance / (avg_length ** 2)))
                coherence_score += consistency * 0.4
            
            return min(1.0, coherence_score)
            
        except Exception as e:
            raise AnalysisException(f"Failed to assess coherence: {e}", "coherence_assessment")
    
    async def assess_relevance(self, content: str, context: str = None) -> float:
        """Evaluar relevancia del contenido."""
        try:
            # Análisis básico de relevancia
            words = re.findall(r'\b\w+\b', content.lower())
            word_count = len(words)
            
            if word_count == 0:
                return 0.0
            
            relevance_score = 0.0
            
            # Verificar densidad de palabras clave
            word_freq = Counter(words)
            total_words = sum(word_freq.values())
            
            # Palabras más frecuentes (posiblemente relevantes)
            top_words = word_freq.most_common(5)
            keyword_density = sum(freq for word, freq in top_words) / total_words
            relevance_score += min(0.4, keyword_density)
            
            # Verificar longitud apropiada (ni muy corto ni muy largo)
            if 50 <= word_count <= 1000:
                relevance_score += 0.3
            elif 20 <= word_count < 50 or 1000 < word_count <= 2000:
                relevance_score += 0.2
            else:
                relevance_score += 0.1
            
            # Verificar estructura (párrafos, oraciones)
            paragraphs = content.split('\n\n')
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(paragraphs) > 1 and len(sentences) > 2:
                relevance_score += 0.3
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            raise AnalysisException(f"Failed to assess relevance: {e}", "relevance_assessment")
    
    async def _assess_completeness(self, content_metrics: ContentMetrics) -> float:
        """Evaluar completitud del contenido."""
        try:
            completeness_score = 0.0
            
            # Verificar longitud mínima
            if content_metrics.word_count >= 50:
                completeness_score += 0.3
            elif content_metrics.word_count >= 20:
                completeness_score += 0.2
            else:
                completeness_score += 0.1
            
            # Verificar estructura
            if content_metrics.sentence_count >= 3:
                completeness_score += 0.3
            elif content_metrics.sentence_count >= 2:
                completeness_score += 0.2
            else:
                completeness_score += 0.1
            
            # Verificar párrafos
            if content_metrics.paragraph_count >= 2:
                completeness_score += 0.2
            elif content_metrics.paragraph_count >= 1:
                completeness_score += 0.1
            
            # Verificar riqueza del vocabulario
            if content_metrics.vocabulary_richness >= 0.5:
                completeness_score += 0.2
            elif content_metrics.vocabulary_richness >= 0.3:
                completeness_score += 0.1
            
            return min(1.0, completeness_score)
            
        except Exception as e:
            raise AnalysisException(f"Failed to assess completeness: {e}", "completeness_assessment")
    
    async def _assess_accuracy(self, content: str) -> float:
        """Evaluar precisión del contenido."""
        try:
            # Análisis básico de precisión
            accuracy_score = 0.5  # Score base
            
            # Verificar ortografía básica (palabras muy cortas o muy largas)
            words = re.findall(r'\b\w+\b', content)
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if 3 <= avg_word_length <= 8:
                    accuracy_score += 0.2
                elif 2 <= avg_word_length <= 10:
                    accuracy_score += 0.1
            
            # Verificar puntuación
            if re.search(r'[.!?]', content):
                accuracy_score += 0.2
            
            # Verificar mayúsculas al inicio de oraciones
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            proper_capitalization = sum(1 for s in sentences if s and s[0].isupper())
            if sentences:
                capitalization_ratio = proper_capitalization / len(sentences)
                accuracy_score += capitalization_ratio * 0.1
            
            return min(1.0, accuracy_score)
            
        except Exception as e:
            raise AnalysisException(f"Failed to assess accuracy: {e}", "accuracy_assessment")
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra (aproximación)."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Ajustar para palabras que terminan en 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)


class CosineSimilarityCalculator(ISimilarityCalculator):
    """
    Calculador de similitud basado en coseno.
    
    Implementación básica para cálculo de similitud entre contenidos.
    """
    
    def __init__(self):
        self.content_analyzer = TextContentAnalyzer()
    
    async def calculate_similarity(self, entry1: HistoryEntry, entry2: HistoryEntry) -> SimilarityScore:
        """Calcular similitud entre dos entradas."""
        try:
            # Calcular diferentes tipos de similitud
            content_similarity = await self.calculate_content_similarity(entry1.content, entry2.content)
            semantic_similarity = await self.calculate_semantic_similarity(entry1.content, entry2.content)
            structural_similarity = await self.calculate_structural_similarity(entry1.content, entry2.content)
            style_similarity = await self._calculate_style_similarity(entry1, entry2)
            
            # Calcular similitud overall
            overall_similarity = (
                content_similarity + semantic_similarity + 
                structural_similarity + style_similarity
            ) / 4.0
            
            return SimilarityScore(
                overall_similarity=overall_similarity,
                content_similarity=content_similarity,
                semantic_similarity=semantic_similarity,
                structural_similarity=structural_similarity,
                style_similarity=style_similarity
            )
            
        except Exception as e:
            raise AnalysisException(f"Failed to calculate similarity: {e}", "similarity_calculation")
    
    async def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud de contenido."""
        try:
            # Obtener vectores de palabras
            words1 = set(re.findall(r'\b\w+\b', content1.lower()))
            words2 = set(re.findall(r'\b\w+\b', content2.lower()))
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            # Calcular similitud de Jaccard
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            raise AnalysisException(f"Failed to calculate content similarity: {e}", "content_similarity")
    
    async def calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud semántica."""
        try:
            # Implementación básica basada en palabras clave
            keywords1 = await self.content_analyzer.extract_keywords(content1)
            keywords2 = await self.content_analyzer.extract_keywords(content2)
            
            if not keywords1 and not keywords2:
                return 1.0
            if not keywords1 or not keywords2:
                return 0.0
            
            # Calcular similitud de palabras clave
            set1 = set(keywords1)
            set2 = set(keywords2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            raise AnalysisException(f"Failed to calculate semantic similarity: {e}", "semantic_similarity")
    
    async def calculate_structural_similarity(self, content1: str, content2: str) -> float:
        """Calcular similitud estructural."""
        try:
            # Analizar estructura
            sentences1 = re.split(r'[.!?]+', content1)
            sentences2 = re.split(r'[.!?]+', content2)
            sentences1 = [s.strip() for s in sentences1 if s.strip()]
            sentences2 = [s.strip() for s in sentences2 if s.strip()]
            
            paragraphs1 = content1.split('\n\n')
            paragraphs2 = content2.split('\n\n')
            paragraphs1 = [p.strip() for p in paragraphs1 if p.strip()]
            paragraphs2 = [p.strip() for p in paragraphs2 if p.strip()]
            
            # Calcular similitud de estructura
            structure_score = 0.0
            
            # Similitud en número de oraciones
            if sentences1 and sentences2:
                sentence_ratio = min(len(sentences1), len(sentences2)) / max(len(sentences1), len(sentences2))
                structure_score += sentence_ratio * 0.5
            
            # Similitud en número de párrafos
            if paragraphs1 and paragraphs2:
                paragraph_ratio = min(len(paragraphs1), len(paragraphs2)) / max(len(paragraphs1), len(paragraphs2))
                structure_score += paragraph_ratio * 0.5
            
            return structure_score
            
        except Exception as e:
            raise AnalysisException(f"Failed to calculate structural similarity: {e}", "structural_similarity")
    
    async def _calculate_style_similarity(self, entry1: HistoryEntry, entry2: HistoryEntry) -> float:
        """Calcular similitud de estilo."""
        try:
            # Analizar métricas de estilo
            metrics1 = await self.content_analyzer.analyze_content(entry1.content)
            metrics2 = await self.content_analyzer.analyze_content(entry2.content)
            
            # Calcular similitud de métricas
            style_score = 0.0
            
            # Similitud en longitud promedio de palabras
            if metrics1.avg_word_length > 0 and metrics2.avg_word_length > 0:
                word_length_ratio = min(metrics1.avg_word_length, metrics2.avg_word_length) / max(metrics1.avg_word_length, metrics2.avg_word_length)
                style_score += word_length_ratio * 0.3
            
            # Similitud en longitud promedio de oraciones
            if metrics1.avg_sentence_length > 0 and metrics2.avg_sentence_length > 0:
                sentence_length_ratio = min(metrics1.avg_sentence_length, metrics2.avg_sentence_length) / max(metrics1.avg_sentence_length, metrics2.avg_sentence_length)
                style_score += sentence_length_ratio * 0.3
            
            # Similitud en riqueza del vocabulario
            if metrics1.vocabulary_richness > 0 and metrics2.vocabulary_richness > 0:
                vocabulary_ratio = min(metrics1.vocabulary_richness, metrics2.vocabulary_richness) / max(metrics1.vocabulary_richness, metrics2.vocabulary_richness)
                style_score += vocabulary_ratio * 0.4
            
            return style_score
            
        except Exception as e:
            raise AnalysisException(f"Failed to calculate style similarity: {e}", "style_similarity")




