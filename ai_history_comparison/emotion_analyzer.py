"""
Advanced Emotion and Sentiment Analysis System for AI History Comparison
Sistema avanzado de análisis de emociones y sentimientos para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict
import math

# NLP imports
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Tipos de emociones"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"

class SentimentIntensity(Enum):
    """Intensidades de sentimiento"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class EmotionalTone(Enum):
    """Tonos emocionales"""
    ENTHUSIASTIC = "enthusiastic"
    CONFIDENT = "confident"
    CONCERNED = "concerned"
    FRUSTRATED = "frustrated"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    ANALYTICAL = "analytical"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"
    CASUAL = "casual"

@dataclass
class EmotionScore:
    """Score de emoción"""
    emotion: EmotionType
    score: float
    confidence: float
    intensity: str

@dataclass
class SentimentAnalysis:
    """Análisis de sentimiento"""
    polarity: float
    subjectivity: float
    intensity: SentimentIntensity
    emotional_tone: EmotionalTone
    dominant_emotion: EmotionType
    emotion_scores: Dict[EmotionType, float]
    confidence: float
    analyzed_at: datetime = field(default_factory=datetime.now)

@dataclass
class EmotionalInsight:
    """Insight emocional"""
    id: str
    text_segment: str
    emotion_type: EmotionType
    intensity: float
    context: str
    implications: List[str]
    recommendations: List[str]
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedEmotionAnalyzer:
    """
    Analizador avanzado de emociones y sentimientos
    """
    
    def __init__(
        self,
        enable_deep_emotion_analysis: bool = True,
        enable_context_analysis: bool = True,
        enable_emotional_insights: bool = True
    ):
        self.enable_deep_emotion_analysis = enable_deep_emotion_analysis
        self.enable_context_analysis = enable_context_analysis
        self.enable_emotional_insights = enable_emotional_insights
        
        # Inicializar analizadores
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_lexicon = self._initialize_emotion_lexicon()
        self.emotional_patterns = self._initialize_emotional_patterns()
        
        # Almacenamiento de análisis
        self.emotion_analyses: Dict[str, SentimentAnalysis] = {}
        self.emotional_insights: Dict[str, EmotionalInsight] = {}
        
        # Configuración
        self.config = {
            "min_confidence_threshold": 0.3,
            "emotion_intensity_threshold": 0.5,
            "context_window_size": 3,
            "max_insights_per_analysis": 10
        }
    
    def _initialize_emotion_lexicon(self) -> Dict[str, Dict[EmotionType, float]]:
        """Inicializar léxico de emociones"""
        return {
            # Palabras de alegría
            "excellent": {EmotionType.JOY: 0.8, EmotionType.TRUST: 0.6},
            "amazing": {EmotionType.JOY: 0.9, EmotionType.SURPRISE: 0.5},
            "fantastic": {EmotionType.JOY: 0.8, EmotionType.ANTICIPATION: 0.4},
            "wonderful": {EmotionType.JOY: 0.7, EmotionType.TRUST: 0.5},
            "brilliant": {EmotionType.JOY: 0.8, EmotionType.TRUST: 0.6},
            "outstanding": {EmotionType.JOY: 0.7, EmotionType.TRUST: 0.7},
            "perfect": {EmotionType.JOY: 0.9, EmotionType.TRUST: 0.8},
            "incredible": {EmotionType.JOY: 0.8, EmotionType.SURPRISE: 0.7},
            "awesome": {EmotionType.JOY: 0.8, EmotionType.ANTICIPATION: 0.4},
            "great": {EmotionType.JOY: 0.6, EmotionType.TRUST: 0.5},
            
            # Palabras de tristeza
            "terrible": {EmotionType.SADNESS: 0.8, EmotionType.DISGUST: 0.6},
            "awful": {EmotionType.SADNESS: 0.7, EmotionType.DISGUST: 0.7},
            "horrible": {EmotionType.SADNESS: 0.8, EmotionType.FEAR: 0.5},
            "disappointing": {EmotionType.SADNESS: 0.7, EmotionType.ANTICIPATION: -0.6},
            "frustrating": {EmotionType.SADNESS: 0.6, EmotionType.ANGER: 0.7},
            "concerning": {EmotionType.SADNESS: 0.5, EmotionType.FEAR: 0.6},
            "worried": {EmotionType.SADNESS: 0.6, EmotionType.FEAR: 0.7},
            "disappointed": {EmotionType.SADNESS: 0.7, EmotionType.ANTICIPATION: -0.5},
            "sad": {EmotionType.SADNESS: 0.8},
            "unfortunate": {EmotionType.SADNESS: 0.6},
            
            # Palabras de ira
            "angry": {EmotionType.ANGER: 0.8},
            "furious": {EmotionType.ANGER: 0.9},
            "outraged": {EmotionType.ANGER: 0.8, EmotionType.SURPRISE: 0.4},
            "frustrated": {EmotionType.ANGER: 0.7, EmotionType.SADNESS: 0.4},
            "annoyed": {EmotionType.ANGER: 0.6},
            "irritated": {EmotionType.ANGER: 0.6},
            "mad": {EmotionType.ANGER: 0.7},
            "upset": {EmotionType.ANGER: 0.6, EmotionType.SADNESS: 0.5},
            "furious": {EmotionType.ANGER: 0.9},
            "livid": {EmotionType.ANGER: 0.8},
            
            # Palabras de miedo
            "scared": {EmotionType.FEAR: 0.8},
            "afraid": {EmotionType.FEAR: 0.7},
            "worried": {EmotionType.FEAR: 0.6, EmotionType.SADNESS: 0.4},
            "concerned": {EmotionType.FEAR: 0.5, EmotionType.SADNESS: 0.3},
            "anxious": {EmotionType.FEAR: 0.7, EmotionType.ANTICIPATION: -0.4},
            "nervous": {EmotionType.FEAR: 0.6},
            "terrified": {EmotionType.FEAR: 0.9},
            "frightened": {EmotionType.FEAR: 0.8},
            "alarmed": {EmotionType.FEAR: 0.7, EmotionType.SURPRISE: 0.5},
            "panicked": {EmotionType.FEAR: 0.9},
            
            # Palabras de sorpresa
            "surprised": {EmotionType.SURPRISE: 0.7},
            "shocked": {EmotionType.SURPRISE: 0.8, EmotionType.FEAR: 0.3},
            "amazed": {EmotionType.SURPRISE: 0.7, EmotionType.JOY: 0.5},
            "astonished": {EmotionType.SURPRISE: 0.8},
            "stunned": {EmotionType.SURPRISE: 0.7},
            "bewildered": {EmotionType.SURPRISE: 0.6, EmotionType.FEAR: 0.3},
            "confused": {EmotionType.SURPRISE: 0.5, EmotionType.FEAR: 0.3},
            "perplexed": {EmotionType.SURPRISE: 0.5, EmotionType.FEAR: 0.2},
            "startled": {EmotionType.SURPRISE: 0.6, EmotionType.FEAR: 0.4},
            "dumbfounded": {EmotionType.SURPRISE: 0.7},
            
            # Palabras de confianza
            "trusted": {EmotionType.TRUST: 0.8},
            "reliable": {EmotionType.TRUST: 0.7},
            "dependable": {EmotionType.TRUST: 0.7},
            "confident": {EmotionType.TRUST: 0.6, EmotionType.JOY: 0.4},
            "secure": {EmotionType.TRUST: 0.6, EmotionType.JOY: 0.3},
            "stable": {EmotionType.TRUST: 0.6},
            "solid": {EmotionType.TRUST: 0.5},
            "proven": {EmotionType.TRUST: 0.7},
            "verified": {EmotionType.TRUST: 0.6},
            "guaranteed": {EmotionType.TRUST: 0.8},
            
            # Palabras de anticipación
            "excited": {EmotionType.ANTICIPATION: 0.7, EmotionType.JOY: 0.5},
            "eager": {EmotionType.ANTICIPATION: 0.6, EmotionType.JOY: 0.3},
            "hopeful": {EmotionType.ANTICIPATION: 0.6, EmotionType.JOY: 0.4},
            "optimistic": {EmotionType.ANTICIPATION: 0.5, EmotionType.JOY: 0.4},
            "enthusiastic": {EmotionType.ANTICIPATION: 0.7, EmotionType.JOY: 0.6},
            "motivated": {EmotionType.ANTICIPATION: 0.6, EmotionType.JOY: 0.4},
            "inspired": {EmotionType.ANTICIPATION: 0.6, EmotionType.JOY: 0.5},
            "determined": {EmotionType.ANTICIPATION: 0.5, EmotionType.TRUST: 0.4},
            "committed": {EmotionType.ANTICIPATION: 0.5, EmotionType.TRUST: 0.5},
            "focused": {EmotionType.ANTICIPATION: 0.4, EmotionType.TRUST: 0.3}
        }
    
    def _initialize_emotional_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar patrones emocionales"""
        return {
            "enthusiasm_indicators": [
                r"\b(excited|enthusiastic|thrilled|delighted|ecstatic)\b",
                r"\b(can't wait|looking forward|eagerly awaiting)\b",
                r"\b(amazing|incredible|fantastic|wonderful)\b"
            ],
            "concern_indicators": [
                r"\b(concerned|worried|troubled|bothered)\b",
                r"\b(issue|problem|challenge|difficulty)\b",
                r"\b(need to address|should consider|important to note)\b"
            ],
            "confidence_indicators": [
                r"\b(confident|certain|sure|positive)\b",
                r"\b(proven|established|reliable|dependable)\b",
                r"\b(guaranteed|assured|confident that)\b"
            ],
            "frustration_indicators": [
                r"\b(frustrated|annoyed|irritated|upset)\b",
                r"\b(disappointed|let down|not working)\b",
                r"\b(struggling|having trouble|difficult)\b"
            ],
            "optimism_indicators": [
                r"\b(optimistic|hopeful|positive|upbeat)\b",
                r"\b(bright future|great potential|promising)\b",
                r"\b(improving|getting better|on the right track)\b"
            ],
            "pessimism_indicators": [
                r"\b(pessimistic|doubtful|skeptical|concerned)\b",
                r"\b(not sure|uncertain|worried about)\b",
                r"\b(risky|dangerous|problematic)\b"
            ]
        }
    
    async def analyze_emotions(
        self,
        text: str,
        document_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SentimentAnalysis:
        """
        Analizar emociones y sentimientos en el texto
        
        Args:
            text: Texto a analizar
            document_id: ID del documento
            context: Contexto adicional
            
        Returns:
            Análisis completo de emociones y sentimientos
        """
        try:
            logger.info(f"Analyzing emotions for document {document_id}")
            
            # Análisis básico de sentimiento
            polarity, subjectivity = self._analyze_basic_sentiment(text)
            
            # Análisis de emociones
            emotion_scores = await self._analyze_emotions(text)
            
            # Determinar emoción dominante
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            # Determinar intensidad de sentimiento
            intensity = self._determine_sentiment_intensity(polarity)
            
            # Determinar tono emocional
            emotional_tone = await self._determine_emotional_tone(text, emotion_scores)
            
            # Calcular confianza general
            confidence = self._calculate_confidence(emotion_scores, polarity)
            
            # Crear análisis
            analysis = SentimentAnalysis(
                polarity=polarity,
                subjectivity=subjectivity,
                intensity=intensity,
                emotional_tone=emotional_tone,
                dominant_emotion=dominant_emotion,
                emotion_scores=emotion_scores,
                confidence=confidence
            )
            
            # Almacenar análisis
            self.emotion_analyses[document_id] = analysis
            
            # Generar insights si está habilitado
            if self.enable_emotional_insights:
                await self._generate_emotional_insights(text, analysis, document_id)
            
            logger.info(f"Emotion analysis completed for document {document_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            raise
    
    def _analyze_basic_sentiment(self, text: str) -> Tuple[float, float]:
        """Analizar sentimiento básico"""
        # Usar TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Usar VADER para validación
        vader_scores = self.sia.polarity_scores(text)
        vader_polarity = vader_scores['compound']
        
        # Promedio de ambos métodos
        final_polarity = (polarity + vader_polarity) / 2
        
        return final_polarity, subjectivity
    
    async def _analyze_emotions(self, text: str) -> Dict[EmotionType, float]:
        """Analizar emociones específicas"""
        emotion_scores = {emotion: 0.0 for emotion in EmotionType}
        
        # Tokenizar texto
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Analizar cada palabra
        for word in words:
            if word in self.emotion_lexicon:
                word_emotions = self.emotion_lexicon[word]
                for emotion, score in word_emotions.items():
                    emotion_scores[emotion] += score
        
        # Normalizar scores
        total_words = len(words)
        if total_words > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total_words
        
        # Aplicar patrones emocionales
        pattern_scores = await self._analyze_emotional_patterns(text)
        for emotion, score in pattern_scores.items():
            emotion_scores[emotion] += score * 0.3  # Peso menor para patrones
        
        # Normalizar a rango [0, 1]
        max_score = max(emotion_scores.values()) if emotion_scores.values() else 1
        if max_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = min(1.0, emotion_scores[emotion] / max_score)
        
        return emotion_scores
    
    async def _analyze_emotional_patterns(self, text: str) -> Dict[EmotionType, float]:
        """Analizar patrones emocionales"""
        pattern_scores = {emotion: 0.0 for emotion in EmotionType}
        
        for pattern_name, patterns in self.emotional_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Mapear patrones a emociones
                    if "enthusiasm" in pattern_name:
                        pattern_scores[EmotionType.JOY] += len(matches) * 0.1
                        pattern_scores[EmotionType.ANTICIPATION] += len(matches) * 0.1
                    elif "concern" in pattern_name:
                        pattern_scores[EmotionType.FEAR] += len(matches) * 0.1
                        pattern_scores[EmotionType.SADNESS] += len(matches) * 0.1
                    elif "confidence" in pattern_name:
                        pattern_scores[EmotionType.TRUST] += len(matches) * 0.1
                        pattern_scores[EmotionType.JOY] += len(matches) * 0.05
                    elif "frustration" in pattern_name:
                        pattern_scores[EmotionType.ANGER] += len(matches) * 0.1
                        pattern_scores[EmotionType.SADNESS] += len(matches) * 0.05
                    elif "optimism" in pattern_name:
                        pattern_scores[EmotionType.JOY] += len(matches) * 0.1
                        pattern_scores[EmotionType.ANTICIPATION] += len(matches) * 0.1
                    elif "pessimism" in pattern_name:
                        pattern_scores[EmotionType.SADNESS] += len(matches) * 0.1
                        pattern_scores[EmotionType.FEAR] += len(matches) * 0.1
        
        return pattern_scores
    
    def _determine_sentiment_intensity(self, polarity: float) -> SentimentIntensity:
        """Determinar intensidad de sentimiento"""
        if polarity <= -0.6:
            return SentimentIntensity.VERY_NEGATIVE
        elif polarity <= -0.2:
            return SentimentIntensity.NEGATIVE
        elif polarity <= 0.2:
            return SentimentIntensity.NEUTRAL
        elif polarity <= 0.6:
            return SentimentIntensity.POSITIVE
        else:
            return SentimentIntensity.VERY_POSITIVE
    
    async def _determine_emotional_tone(
        self,
        text: str,
        emotion_scores: Dict[EmotionType, float]
    ) -> EmotionalTone:
        """Determinar tono emocional"""
        
        # Análisis basado en emociones dominantes
        joy_score = emotion_scores.get(EmotionType.JOY, 0)
        anger_score = emotion_scores.get(EmotionType.ANGER, 0)
        fear_score = emotion_scores.get(EmotionType.FEAR, 0)
        trust_score = emotion_scores.get(EmotionType.TRUST, 0)
        anticipation_score = emotion_scores.get(EmotionType.ANTICIPATION, 0)
        
        # Determinar tono basado en combinaciones
        if joy_score > 0.6 and anticipation_score > 0.4:
            return EmotionalTone.ENTHUSIASTIC
        elif trust_score > 0.6 and joy_score > 0.3:
            return EmotionalTone.CONFIDENT
        elif fear_score > 0.5 or (fear_score > 0.3 and anger_score > 0.3):
            return EmotionalTone.CONCERNED
        elif anger_score > 0.5:
            return EmotionalTone.FRUSTRATED
        elif joy_score > 0.4 and anticipation_score > 0.3:
            return EmotionalTone.OPTIMISTIC
        elif fear_score > 0.4 and anticipation_score < 0.2:
            return EmotionalTone.PESSIMISTIC
        elif trust_score > 0.4 and anger_score < 0.2:
            return EmotionalTone.ANALYTICAL
        elif joy_score > 0.3 and trust_score > 0.3:
            return EmotionalTone.PERSUASIVE
        elif trust_score > 0.3 and anger_score < 0.3:
            return EmotionalTone.INFORMATIVE
        else:
            return EmotionalTone.CASUAL
    
    def _calculate_confidence(
        self,
        emotion_scores: Dict[EmotionType, float],
        polarity: float
    ) -> float:
        """Calcular confianza del análisis"""
        # Confianza basada en la claridad de las emociones
        max_emotion_score = max(emotion_scores.values()) if emotion_scores.values() else 0
        emotion_clarity = max_emotion_score
        
        # Confianza basada en la consistencia del sentimiento
        polarity_confidence = abs(polarity)
        
        # Confianza combinada
        confidence = (emotion_clarity * 0.6 + polarity_confidence * 0.4)
        
        return min(1.0, confidence)
    
    async def _generate_emotional_insights(
        self,
        text: str,
        analysis: SentimentAnalysis,
        document_id: str
    ):
        """Generar insights emocionales"""
        insights = []
        
        # Insight 1: Emoción dominante
        if analysis.dominant_emotion != EmotionType.NEUTRAL:
            insight = EmotionalInsight(
                id=f"dominant_emotion_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                text_segment=text[:100] + "..." if len(text) > 100 else text,
                emotion_type=analysis.dominant_emotion,
                intensity=analysis.emotion_scores[analysis.dominant_emotion],
                context=f"Emoción dominante detectada en el documento",
                implications=[
                    f"El contenido muestra una tendencia hacia {analysis.dominant_emotion.value}",
                    f"Esto puede influir en la percepción del lector",
                    f"La intensidad es {analysis.emotion_scores[analysis.dominant_emotion]:.2f}"
                ],
                recommendations=[
                    f"Considerar el impacto de {analysis.dominant_emotion.value} en la audiencia",
                    "Evaluar si la emoción es apropiada para el contexto",
                    "Ajustar el tono si es necesario"
                ],
                confidence=analysis.confidence
            )
            insights.append(insight)
        
        # Insight 2: Tono emocional
        if analysis.emotional_tone != EmotionalTone.CASUAL:
            insight = EmotionalInsight(
                id=f"emotional_tone_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                text_segment=text[:100] + "..." if len(text) > 100 else text,
                emotion_type=analysis.dominant_emotion,
                intensity=analysis.confidence,
                context=f"Tono emocional: {analysis.emotional_tone.value}",
                implications=[
                    f"El tono {analysis.emotional_tone.value} puede afectar la recepción del mensaje",
                    "La audiencia puede responder de manera diferente según el tono"
                ],
                recommendations=[
                    f"Mantener consistencia con el tono {analysis.emotional_tone.value}",
                    "Considerar si el tono es apropiado para la audiencia objetivo"
                ],
                confidence=analysis.confidence
            )
            insights.append(insight)
        
        # Insight 3: Intensidad del sentimiento
        if analysis.intensity != SentimentIntensity.NEUTRAL:
            insight = EmotionalInsight(
                id=f"sentiment_intensity_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                text_segment=text[:100] + "..." if len(text) > 100 else text,
                emotion_type=analysis.dominant_emotion,
                intensity=abs(analysis.polarity),
                context=f"Intensidad de sentimiento: {analysis.intensity.value}",
                implications=[
                    f"La intensidad {analysis.intensity.value} puede generar respuestas fuertes",
                    "El contenido puede ser memorable debido a su intensidad emocional"
                ],
                recommendations=[
                    "Evaluar si la intensidad es apropiada para el contexto",
                    "Considerar el impacto en diferentes tipos de audiencia"
                ],
                confidence=analysis.confidence
            )
            insights.append(insight)
        
        # Almacenar insights
        for insight in insights[:self.config["max_insights_per_analysis"]]:
            self.emotional_insights[insight.id] = insight
    
    async def compare_emotional_profiles(
        self,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """Comparar perfiles emocionales de múltiples documentos"""
        try:
            if len(document_ids) < 2:
                raise ValueError("Se necesitan al menos 2 documentos para comparar")
            
            # Obtener análisis
            analyses = []
            for doc_id in document_ids:
                if doc_id in self.emotion_analyses:
                    analyses.append(self.emotion_analyses[doc_id])
                else:
                    logger.warning(f"Analysis not found for document {doc_id}")
            
            if len(analyses) < 2:
                raise ValueError("No hay suficientes análisis para comparar")
            
            # Calcular similitudes emocionales
            emotional_similarities = self._calculate_emotional_similarities(analyses)
            
            # Encontrar diferencias significativas
            significant_differences = self._find_significant_differences(analyses)
            
            # Generar insights comparativos
            comparative_insights = await self._generate_comparative_insights(analyses)
            
            return {
                "document_ids": document_ids,
                "analyses_count": len(analyses),
                "emotional_similarities": emotional_similarities,
                "significant_differences": significant_differences,
                "comparative_insights": comparative_insights,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing emotional profiles: {e}")
            raise
    
    def _calculate_emotional_similarities(self, analyses: List[SentimentAnalysis]) -> Dict[str, float]:
        """Calcular similitudes emocionales"""
        similarities = {}
        
        for i, analysis1 in enumerate(analyses):
            for j, analysis2 in enumerate(analyses[i+1:], i+1):
                # Similitud de polaridad
                polarity_sim = 1 - abs(analysis1.polarity - analysis2.polarity) / 2
                
                # Similitud de emociones
                emotion_sim = 0
                for emotion in EmotionType:
                    emotion_sim += abs(analysis1.emotion_scores[emotion] - analysis2.emotion_scores[emotion])
                emotion_sim = 1 - (emotion_sim / len(EmotionType))
                
                # Similitud combinada
                combined_sim = (polarity_sim * 0.4 + emotion_sim * 0.6)
                
                similarities[f"doc_{i}_vs_doc_{j}"] = combined_sim
        
        return similarities
    
    def _find_significant_differences(self, analyses: List[SentimentAnalysis]) -> List[Dict[str, Any]]:
        """Encontrar diferencias significativas"""
        differences = []
        
        # Comparar polaridades
        polarities = [analysis.polarity for analysis in analyses]
        if max(polarities) - min(polarities) > 0.5:
            differences.append({
                "type": "polarity",
                "description": "Diferencias significativas en polaridad de sentimiento",
                "range": f"{min(polarities):.2f} to {max(polarities):.2f}",
                "impact": "high"
            })
        
        # Comparar emociones dominantes
        dominant_emotions = [analysis.dominant_emotion for analysis in analyses]
        unique_emotions = set(dominant_emotions)
        if len(unique_emotions) > 1:
            differences.append({
                "type": "dominant_emotion",
                "description": "Diferentes emociones dominantes detectadas",
                "emotions": [emotion.value for emotion in unique_emotions],
                "impact": "medium"
            })
        
        # Comparar tonos emocionales
        emotional_tones = [analysis.emotional_tone for analysis in analyses]
        unique_tones = set(emotional_tones)
        if len(unique_tones) > 1:
            differences.append({
                "type": "emotional_tone",
                "description": "Diferentes tonos emocionales detectados",
                "tones": [tone.value for tone in unique_tones],
                "impact": "medium"
            })
        
        return differences
    
    async def _generate_comparative_insights(self, analyses: List[SentimentAnalysis]) -> List[str]:
        """Generar insights comparativos"""
        insights = []
        
        # Insight sobre variabilidad emocional
        polarities = [analysis.polarity for analysis in analyses]
        polarity_variance = np.var(polarities)
        
        if polarity_variance > 0.25:
            insights.append("Alta variabilidad emocional detectada entre documentos")
        elif polarity_variance < 0.05:
            insights.append("Baja variabilidad emocional - documentos muy consistentes")
        
        # Insight sobre emociones dominantes
        dominant_emotions = [analysis.dominant_emotion for analysis in analyses]
        emotion_counts = Counter(dominant_emotions)
        most_common_emotion = emotion_counts.most_common(1)[0]
        
        if most_common_emotion[1] > len(analyses) * 0.6:
            insights.append(f"Tendencia emocional consistente hacia {most_common_emotion[0].value}")
        else:
            insights.append("Variedad emocional significativa entre documentos")
        
        # Insight sobre confianza
        confidences = [analysis.confidence for analysis in analyses]
        avg_confidence = np.mean(confidences)
        
        if avg_confidence > 0.7:
            insights.append("Alta confianza en los análisis emocionales")
        elif avg_confidence < 0.4:
            insights.append("Baja confianza en los análisis - considerar más contexto")
        
        return insights
    
    async def get_emotion_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis emocionales"""
        if not self.emotion_analyses:
            return {"message": "No emotion analyses available"}
        
        # Estadísticas generales
        total_analyses = len(self.emotion_analyses)
        polarities = [analysis.polarity for analysis in self.emotion_analyses.values()]
        confidences = [analysis.confidence for analysis in self.emotion_analyses.values()]
        
        # Distribución de emociones dominantes
        dominant_emotions = [analysis.dominant_emotion for analysis in self.emotion_analyses.values()]
        emotion_distribution = Counter(dominant_emotions)
        
        # Distribución de tonos emocionales
        emotional_tones = [analysis.emotional_tone for analysis in self.emotion_analyses.values()]
        tone_distribution = Counter(emotional_tones)
        
        # Distribución de intensidades
        intensities = [analysis.intensity for analysis in self.emotion_analyses.values()]
        intensity_distribution = Counter(intensities)
        
        return {
            "total_analyses": total_analyses,
            "average_polarity": np.mean(polarities),
            "average_confidence": np.mean(confidences),
            "polarity_std": np.std(polarities),
            "emotion_distribution": {emotion.value: count for emotion, count in emotion_distribution.items()},
            "tone_distribution": {tone.value: count for tone, count in tone_distribution.items()},
            "intensity_distribution": {intensity.value: count for intensity, count in intensity_distribution.items()},
            "total_insights": len(self.emotional_insights),
            "last_analysis": max([analysis.analyzed_at for analysis in self.emotion_analyses.values()]).isoformat()
        }
    
    async def export_emotion_analysis(self, filepath: str = None) -> str:
        """Exportar análisis emocional"""
        try:
            if filepath is None:
                filepath = f"exports/emotion_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                "emotion_analyses": {
                    doc_id: {
                        "polarity": analysis.polarity,
                        "subjectivity": analysis.subjectivity,
                        "intensity": analysis.intensity.value,
                        "emotional_tone": analysis.emotional_tone.value,
                        "dominant_emotion": analysis.dominant_emotion.value,
                        "emotion_scores": {emotion.value: score for emotion, score in analysis.emotion_scores.items()},
                        "confidence": analysis.confidence,
                        "analyzed_at": analysis.analyzed_at.isoformat()
                    }
                    for doc_id, analysis in self.emotion_analyses.items()
                },
                "emotional_insights": {
                    insight_id: {
                        "text_segment": insight.text_segment,
                        "emotion_type": insight.emotion_type.value,
                        "intensity": insight.intensity,
                        "context": insight.context,
                        "implications": insight.implications,
                        "recommendations": insight.recommendations,
                        "confidence": insight.confidence,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.emotional_insights.items()
                },
                "summary": await self.get_emotion_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Emotion analysis exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting emotion analysis: {e}")
            raise

























