"""
Analysis Service - Servicio de Análisis
======================================

Servicio real para análisis de contenido de IA con tecnologías funcionales.
"""

import asyncio
import re
import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

from ..core.models import HistoryEntry, ComparisonResult, QualityReport, AnalysisJob, TrendAnalysis


class ContentAnalyzer:
    """Analizador de contenido real y funcional."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analizar contenido de texto."""
        try:
            # Métricas básicas
            word_count = len(word_tokenize(content))
            sentence_count = len(sent_tokenize(content))
            char_count = len(content)
            
            # Métricas de legibilidad
            flesch_score = flesch_reading_ease(content)
            fk_grade = flesch_kincaid_grade(content)
            
            # Análisis de sentimiento
            sentiment_scores = self.sia.polarity_scores(content)
            
            # Análisis de complejidad
            avg_word_length = sum(len(word) for word in word_tokenize(content)) / word_count if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Análisis de vocabulario
            unique_words = len(set(word.lower() for word in word_tokenize(content)))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "char_count": char_count,
                "flesch_reading_ease": flesch_score,
                "flesch_kincaid_grade": fk_grade,
                "sentiment_positive": sentiment_scores["pos"],
                "sentiment_neutral": sentiment_scores["neu"],
                "sentiment_negative": sentiment_scores["neg"],
                "sentiment_compound": sentiment_scores["compound"],
                "avg_word_length": avg_word_length,
                "avg_sentence_length": avg_sentence_length,
                "vocabulary_richness": vocabulary_richness,
                "unique_words": unique_words
            }
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Obtener análisis por defecto en caso de error."""
        return {
            "word_count": 0,
            "sentence_count": 0,
            "char_count": 0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "sentiment_positive": 0.0,
            "sentiment_neutral": 1.0,
            "sentiment_negative": 0.0,
            "sentiment_compound": 0.0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "vocabulary_richness": 0.0,
            "unique_words": 0
        }


class ComparisonService:
    """Servicio de comparación real y funcional."""
    
    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def compare_entries(self, entry1: HistoryEntry, entry2: HistoryEntry) -> ComparisonResult:
        """Comparar dos entradas de historial."""
        try:
            # Análisis de contenido
            analysis1 = self.analyzer.analyze_content(entry1.response)
            analysis2 = self.analyzer.analyze_content(entry2.response)
            
            # Similitud semántica usando TF-IDF
            semantic_similarity = self._calculate_semantic_similarity(
                entry1.response, entry2.response
            )
            
            # Similitud léxica
            lexical_similarity = self._calculate_lexical_similarity(
                entry1.response, entry2.response
            )
            
            # Similitud estructural
            structural_similarity = self._calculate_structural_similarity(
                analysis1, analysis2
            )
            
            # Similitud general (promedio ponderado)
            overall_similarity = (
                semantic_similarity * 0.5 +
                lexical_similarity * 0.3 +
                structural_similarity * 0.2
            )
            
            # Detectar diferencias
            differences = self._detect_differences(analysis1, analysis2)
            improvements = self._detect_improvements(analysis1, analysis2)
            
            return ComparisonResult(
                entry_1_id=entry1.id,
                entry_2_id=entry2.id,
                semantic_similarity=semantic_similarity,
                lexical_similarity=lexical_similarity,
                structural_similarity=structural_similarity,
                overall_similarity=overall_similarity,
                differences=differences,
                improvements=improvements,
                analysis_details={
                    "entry1_analysis": analysis1,
                    "entry2_analysis": analysis2,
                    "comparison_timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            print(f"Error comparing entries: {e}")
            return self._get_default_comparison(entry1.id, entry2.id)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud semántica usando TF-IDF."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud léxica."""
        try:
            words1 = set(word.lower() for word in word_tokenize(text1))
            words2 = set(word.lower() for word in word_tokenize(text2))
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_structural_similarity(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> float:
        """Calcular similitud estructural."""
        try:
            # Comparar métricas estructurales
            metrics = [
                'word_count', 'sentence_count', 'avg_word_length', 
                'avg_sentence_length', 'vocabulary_richness'
            ]
            
            similarities = []
            for metric in metrics:
                val1 = analysis1.get(metric, 0)
                val2 = analysis2.get(metric, 0)
                
                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                elif val1 == 0 or val2 == 0:
                    similarity = 0.0
                else:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                
                similarities.append(similarity)
            
            return statistics.mean(similarities)
        except Exception:
            return 0.0
    
    def _detect_differences(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> List[str]:
        """Detectar diferencias significativas."""
        differences = []
        
        # Diferencia en longitud
        word_diff = abs(analysis1.get('word_count', 0) - analysis2.get('word_count', 0))
        if word_diff > 50:
            differences.append(f"Significant word count difference: {word_diff} words")
        
        # Diferencia en legibilidad
        flesch_diff = abs(analysis1.get('flesch_reading_ease', 0) - analysis2.get('flesch_reading_ease', 0))
        if flesch_diff > 20:
            differences.append(f"Significant readability difference: {flesch_diff:.1f} points")
        
        # Diferencia en sentimiento
        sentiment_diff = abs(analysis1.get('sentiment_compound', 0) - analysis2.get('sentiment_compound', 0))
        if sentiment_diff > 0.3:
            differences.append(f"Significant sentiment difference: {sentiment_diff:.2f}")
        
        return differences
    
    def _detect_improvements(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> List[str]:
        """Detectar mejoras."""
        improvements = []
        
        # Mejora en legibilidad
        if analysis2.get('flesch_reading_ease', 0) > analysis1.get('flesch_reading_ease', 0) + 10:
            improvements.append("Improved readability")
        
        # Mejora en riqueza de vocabulario
        if analysis2.get('vocabulary_richness', 0) > analysis1.get('vocabulary_richness', 0) + 0.1:
            improvements.append("Enhanced vocabulary richness")
        
        # Mejora en sentimiento positivo
        if analysis2.get('sentiment_compound', 0) > analysis1.get('sentiment_compound', 0) + 0.2:
            improvements.append("More positive sentiment")
        
        return improvements
    
    def _get_default_comparison(self, entry1_id: str, entry2_id: str) -> ComparisonResult:
        """Obtener comparación por defecto."""
        return ComparisonResult(
            entry_1_id=entry1_id,
            entry_2_id=entry2_id,
            semantic_similarity=0.0,
            lexical_similarity=0.0,
            structural_similarity=0.0,
            overall_similarity=0.0,
            differences=["Analysis failed"],
            improvements=[],
            analysis_details={"error": "Comparison analysis failed"}
        )


class QualityAssessmentService:
    """Servicio de evaluación de calidad real y funcional."""
    
    def __init__(self):
        self.analyzer = ContentAnalyzer()
    
    def assess_quality(self, entry: HistoryEntry) -> QualityReport:
        """Evaluar la calidad de una entrada."""
        try:
            analysis = self.analyzer.analyze_content(entry.response)
            
            # Calcular puntuaciones de calidad
            coherence = self._calculate_coherence_score(analysis)
            relevance = self._calculate_relevance_score(entry.prompt, entry.response)
            creativity = self._calculate_creativity_score(analysis)
            accuracy = self._calculate_accuracy_score(analysis)
            clarity = self._calculate_clarity_score(analysis)
            
            # Puntuación general
            overall_quality = (coherence + relevance + creativity + accuracy + clarity) / 5
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(analysis, {
                'coherence': coherence,
                'relevance': relevance,
                'creativity': creativity,
                'accuracy': accuracy,
                'clarity': clarity
            })
            
            # Identificar fortalezas y debilidades
            strengths, weaknesses = self._identify_strengths_weaknesses(analysis, {
                'coherence': coherence,
                'relevance': relevance,
                'creativity': creativity,
                'accuracy': accuracy,
                'clarity': clarity
            })
            
            return QualityReport(
                entry_id=entry.id,
                overall_quality=overall_quality,
                coherence=coherence,
                relevance=relevance,
                creativity=creativity,
                accuracy=accuracy,
                clarity=clarity,
                recommendations=recommendations,
                strengths=strengths,
                weaknesses=weaknesses,
                confidence_score=0.8  # Confianza en el análisis automatizado
            )
        except Exception as e:
            print(f"Error assessing quality: {e}")
            return self._get_default_quality_report(entry.id)
    
    def _calculate_coherence_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular puntuación de coherencia."""
        # Basado en métricas de estructura y legibilidad
        flesch_score = analysis.get('flesch_reading_ease', 0)
        avg_sentence_length = analysis.get('avg_sentence_length', 0)
        
        # Normalizar puntuación Flesch (0-100 -> 0-1)
        flesch_normalized = max(0, min(1, flesch_score / 100))
        
        # Penalizar oraciones muy largas o muy cortas
        sentence_penalty = 0
        if avg_sentence_length > 25 or avg_sentence_length < 5:
            sentence_penalty = 0.2
        
        return max(0, flesch_normalized - sentence_penalty)
    
    def _calculate_relevance_score(self, prompt: str, response: str) -> float:
        """Calcular puntuación de relevancia."""
        try:
            # Análisis simple de relevancia basado en palabras clave
            prompt_words = set(word.lower() for word in word_tokenize(prompt))
            response_words = set(word.lower() for word in word_tokenize(response))
            
            # Remover palabras comunes
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            prompt_words -= common_words
            response_words -= common_words
            
            if not prompt_words:
                return 0.5  # Puntuación neutral si no hay palabras clave
            
            overlap = len(prompt_words.intersection(response_words))
            relevance = overlap / len(prompt_words)
            
            return min(1.0, relevance)
        except Exception:
            return 0.5
    
    def _calculate_creativity_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular puntuación de creatividad."""
        # Basado en riqueza de vocabulario y variación
        vocabulary_richness = analysis.get('vocabulary_richness', 0)
        unique_words = analysis.get('unique_words', 0)
        word_count = analysis.get('word_count', 1)
        
        # Normalizar riqueza de vocabulario
        creativity = min(1.0, vocabulary_richness * 2)
        
        # Bonus por diversidad de palabras
        if unique_words > word_count * 0.7:
            creativity += 0.1
        
        return min(1.0, creativity)
    
    def _calculate_accuracy_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular puntuación de precisión."""
        # Basado en métricas de estructura y coherencia
        avg_word_length = analysis.get('avg_word_length', 0)
        avg_sentence_length = analysis.get('avg_sentence_length', 0)
        
        # Palabras de longitud apropiada (4-8 caracteres)
        word_score = 1.0 - abs(avg_word_length - 6) / 6
        word_score = max(0, word_score)
        
        # Oraciones de longitud apropiada (10-20 palabras)
        sentence_score = 1.0 - abs(avg_sentence_length - 15) / 15
        sentence_score = max(0, sentence_score)
        
        return (word_score + sentence_score) / 2
    
    def _calculate_clarity_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular puntuación de claridad."""
        flesch_score = analysis.get('flesch_reading_ease', 0)
        
        # Puntuación Flesch ideal: 60-80
        if 60 <= flesch_score <= 80:
            return 1.0
        elif 40 <= flesch_score < 60 or 80 < flesch_score <= 90:
            return 0.8
        else:
            return 0.5
    
    def _generate_recommendations(self, analysis: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Generar recomendaciones de mejora."""
        recommendations = []
        
        # Recomendaciones basadas en puntuaciones bajas
        if scores['coherence'] < 0.6:
            recommendations.append("Improve text coherence by using shorter sentences and clearer transitions")
        
        if scores['relevance'] < 0.6:
            recommendations.append("Ensure response directly addresses the prompt and stays on topic")
        
        if scores['creativity'] < 0.6:
            recommendations.append("Enhance creativity by using more diverse vocabulary and varied sentence structures")
        
        if scores['accuracy'] < 0.6:
            recommendations.append("Improve accuracy by using more precise language and appropriate word choices")
        
        if scores['clarity'] < 0.6:
            recommendations.append("Enhance clarity by simplifying complex sentences and using more common words")
        
        # Recomendaciones basadas en métricas específicas
        if analysis.get('flesch_reading_ease', 0) < 30:
            recommendations.append("Text is too complex - consider using simpler language")
        elif analysis.get('flesch_reading_ease', 0) > 90:
            recommendations.append("Text may be too simple - consider adding more sophisticated vocabulary")
        
        if analysis.get('avg_sentence_length', 0) > 25:
            recommendations.append("Sentences are too long - consider breaking them into shorter ones")
        
        return recommendations
    
    def _identify_strengths_weaknesses(self, analysis: Dict[str, Any], scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identificar fortalezas y debilidades."""
        strengths = []
        weaknesses = []
        
        # Fortalezas
        if scores['coherence'] > 0.8:
            strengths.append("Excellent text coherence and flow")
        if scores['relevance'] > 0.8:
            strengths.append("Highly relevant to the prompt")
        if scores['creativity'] > 0.8:
            strengths.append("Creative and engaging content")
        if scores['accuracy'] > 0.8:
            strengths.append("Precise and accurate language")
        if scores['clarity'] > 0.8:
            strengths.append("Clear and easy to understand")
        
        # Debilidades
        if scores['coherence'] < 0.5:
            weaknesses.append("Poor text coherence and structure")
        if scores['relevance'] < 0.5:
            weaknesses.append("Low relevance to the prompt")
        if scores['creativity'] < 0.5:
            weaknesses.append("Lacks creativity and originality")
        if scores['accuracy'] < 0.5:
            weaknesses.append("Imprecise or inaccurate language")
        if scores['clarity'] < 0.5:
            weaknesses.append("Unclear or confusing content")
        
        return strengths, weaknesses
    
    def _get_default_quality_report(self, entry_id: str) -> QualityReport:
        """Obtener reporte de calidad por defecto."""
        return QualityReport(
            entry_id=entry_id,
            overall_quality=0.5,
            coherence=0.5,
            relevance=0.5,
            creativity=0.5,
            accuracy=0.5,
            clarity=0.5,
            recommendations=["Quality assessment failed"],
            strengths=[],
            weaknesses=["Unable to assess quality"],
            confidence_score=0.0
        )




