"""
Sentiment and Emotion Analysis System
====================================

This module provides advanced sentiment and emotion analysis capabilities for
AI model outputs, including:
- Multi-dimensional sentiment analysis
- Emotion detection and classification
- Tone analysis and personality insights
- Bias detection and fairness analysis
- Content safety and toxicity detection
- Cultural and linguistic adaptation
- Real-time emotion tracking
- Emotional intelligence metrics
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import os
from collections import Counter
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    text: str
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    confidence: float  # 0 to 1
    sentiment_label: str  # positive, negative, neutral
    emotional_intensity: float  # 0 to 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmotionAnalysis:
    """Emotion analysis result"""
    text: str
    emotions: Dict[str, float]  # emotion -> confidence
    dominant_emotion: str
    emotional_arousal: float  # 0 to 1
    emotional_valence: float  # -1 to 1
    emotional_dominance: float  # 0 to 1
    confidence: float  # 0 to 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToneAnalysis:
    """Tone analysis result"""
    text: str
    tones: Dict[str, float]  # tone -> confidence
    dominant_tone: str
    formality_level: float  # 0 to 1
    politeness_level: float  # 0 to 1
    assertiveness_level: float  # 0 to 1
    confidence: float  # 0 to 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BiasAnalysis:
    """Bias analysis result"""
    text: str
    bias_scores: Dict[str, float]  # bias_type -> score
    overall_bias_score: float  # 0 to 1
    detected_biases: List[str]
    fairness_score: float  # 0 to 1
    diversity_score: float  # 0 to 1
    confidence: float  # 0 to 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SafetyAnalysis:
    """Content safety analysis result"""
    text: str
    toxicity_score: float  # 0 to 1
    safety_level: str  # safe, caution, unsafe
    detected_issues: List[str]
    content_categories: Dict[str, float]
    moderation_flags: List[str]
    confidence: float  # 0 to 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SentimentEmotionAnalyzer:
    """Advanced sentiment and emotion analysis system"""
    
    def __init__(self, model_storage_path: str = "sentiment_models"):
        self.model_storage_path = model_storage_path
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # Analysis models
        self.sentiment_models: Dict[str, Any] = {}
        self.emotion_models: Dict[str, Any] = {}
        self.bias_models: Dict[str, Any] = {}
        self.safety_models: Dict[str, Any] = {}
        
        # NLP tools
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.nlp = None  # spaCy model
        self.emotion_classifier = None
        self.toxicity_classifier = None
        
        # Emotion and tone dictionaries
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.tone_lexicon = self._load_tone_lexicon()
        self.bias_patterns = self._load_bias_patterns()
        
        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Ensure model storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _load_emotion_lexicon(self) -> Dict[str, List[str]]:
        """Load emotion lexicon for emotion detection"""
        return {
            "joy": ["happy", "joyful", "excited", "pleased", "delighted", "cheerful", "elated", "thrilled"],
            "sadness": ["sad", "depressed", "melancholy", "gloomy", "sorrowful", "dejected", "mournful", "despondent"],
            "anger": ["angry", "furious", "irritated", "annoyed", "rage", "outraged", "livid", "incensed"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried", "nervous", "frightened", "alarmed"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "startled", "bewildered", "stunned"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "nauseated", "appalled", "horrified"],
            "trust": ["trusting", "confident", "reliable", "faithful", "loyal", "dependable", "secure"],
            "anticipation": ["anticipating", "expecting", "hopeful", "eager", "excited", "optimistic", "enthusiastic"]
        }
    
    def _load_tone_lexicon(self) -> Dict[str, List[str]]:
        """Load tone lexicon for tone analysis"""
        return {
            "formal": ["therefore", "furthermore", "consequently", "moreover", "nevertheless", "accordingly"],
            "informal": ["yeah", "cool", "awesome", "gonna", "wanna", "gotta", "kinda", "sorta"],
            "polite": ["please", "thank you", "excuse me", "pardon", "kindly", "respectfully", "courteously"],
            "assertive": ["must", "should", "will", "definitely", "certainly", "absolutely", "undoubtedly"],
            "tentative": ["maybe", "perhaps", "might", "could", "possibly", "potentially", "likely"],
            "enthusiastic": ["amazing", "fantastic", "incredible", "wonderful", "brilliant", "outstanding"],
            "critical": ["flawed", "problematic", "concerning", "inadequate", "insufficient", "disappointing"],
            "supportive": ["helpful", "beneficial", "valuable", "useful", "constructive", "positive"]
        }
    
    def _load_bias_patterns(self) -> Dict[str, List[str]]:
        """Load bias patterns for bias detection"""
        return {
            "gender_bias": ["he should", "she should", "men are", "women are", "boys will be", "girls are"],
            "racial_bias": ["they always", "those people", "typical", "you people", "your kind"],
            "age_bias": ["too old", "too young", "millennials", "boomers", "generation"],
            "cultural_bias": ["foreign", "exotic", "primitive", "backward", "uncivilized"],
            "religious_bias": ["heathen", "infidel", "godless", "sinful", "unholy"],
            "socioeconomic_bias": ["poor people", "rich people", "elite", "common folk", "working class"],
            "ability_bias": ["disabled", "handicapped", "retarded", "slow", "incompetent"],
            "appearance_bias": ["ugly", "beautiful", "attractive", "unattractive", "good looking"]
        }
    
    def _initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Initialize emotion classifier
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Could not load emotion classifier: {str(e)}")
                self.emotion_classifier = None
            
            # Initialize toxicity classifier
            try:
                self.toxicity_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Could not load toxicity classifier: {str(e)}")
                self.toxicity_classifier = None
            
            logger.info("Sentiment and emotion analysis models initialized")
        
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
    
    async def analyze_sentiment(self, text: str, model_name: str = "ensemble") -> SentimentAnalysis:
        """Analyze sentiment of text using multiple methods"""
        try:
            if not text or not text.strip():
                return SentimentAnalysis(
                    text=text,
                    polarity=0.0,
                    subjectivity=0.0,
                    confidence=0.0,
                    sentiment_label="neutral",
                    emotional_intensity=0.0
                )
            
            # Method 1: TextBlob
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Method 2: VADER
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_polarity = vader_scores['compound']
            
            # Method 3: spaCy (if available)
            spacy_polarity = 0.0
            if self.nlp:
                doc = self.nlp(text)
                # Simple rule-based sentiment using spaCy
                positive_words = 0
                negative_words = 0
                for token in doc:
                    if token.lemma_.lower() in ['good', 'great', 'excellent', 'amazing', 'wonderful']:
                        positive_words += 1
                    elif token.lemma_.lower() in ['bad', 'terrible', 'awful', 'horrible', 'disgusting']:
                        negative_words += 1
                
                if positive_words + negative_words > 0:
                    spacy_polarity = (positive_words - negative_words) / (positive_words + negative_words)
            
            # Ensemble method: combine all scores
            polarities = [textblob_polarity, vader_polarity, spacy_polarity]
            valid_polarities = [p for p in polarities if p is not None]
            
            if valid_polarities:
                ensemble_polarity = np.mean(valid_polarities)
                confidence = 1.0 - np.std(valid_polarities) if len(valid_polarities) > 1 else 0.8
            else:
                ensemble_polarity = 0.0
                confidence = 0.0
            
            # Determine sentiment label
            if ensemble_polarity > 0.1:
                sentiment_label = "positive"
            elif ensemble_polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Calculate emotional intensity
            emotional_intensity = abs(ensemble_polarity) + (1.0 - textblob_subjectivity) * 0.3
            
            return SentimentAnalysis(
                text=text,
                polarity=ensemble_polarity,
                subjectivity=textblob_subjectivity,
                confidence=confidence,
                sentiment_label=sentiment_label,
                emotional_intensity=emotional_intensity,
                metadata={
                    "textblob_polarity": textblob_polarity,
                    "vader_polarity": vader_polarity,
                    "spacy_polarity": spacy_polarity,
                    "model_name": model_name,
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return SentimentAnalysis(
                text=text,
                polarity=0.0,
                subjectivity=0.0,
                confidence=0.0,
                sentiment_label="neutral",
                emotional_intensity=0.0,
                metadata={"error": str(e)}
            )
    
    async def analyze_emotions(self, text: str) -> EmotionAnalysis:
        """Analyze emotions in text"""
        try:
            if not text or not text.strip():
                return EmotionAnalysis(
                    text=text,
                    emotions={},
                    dominant_emotion="neutral",
                    emotional_arousal=0.0,
                    emotional_valence=0.0,
                    emotional_dominance=0.0,
                    confidence=0.0
                )
            
            emotions = {}
            
            # Method 1: Emotion classifier (if available)
            if self.emotion_classifier:
                try:
                    emotion_scores = self.emotion_classifier(text)
                    for score in emotion_scores:
                        emotion_name = score['label'].lower()
                        emotions[emotion_name] = score['score']
                except Exception as e:
                    logger.warning(f"Error with emotion classifier: {str(e)}")
            
            # Method 2: Lexicon-based emotion detection
            text_lower = text.lower()
            for emotion, words in self.emotion_lexicon.items():
                emotion_count = sum(1 for word in words if word in text_lower)
                if emotion_count > 0:
                    lexicon_score = min(emotion_count / len(words), 1.0)
                    if emotion in emotions:
                        emotions[emotion] = (emotions[emotion] + lexicon_score) / 2
                    else:
                        emotions[emotion] = lexicon_score
            
            # If no emotions detected, use sentiment as base
            if not emotions:
                sentiment_analysis = await self.analyze_sentiment(text)
                if sentiment_analysis.polarity > 0.1:
                    emotions["joy"] = abs(sentiment_analysis.polarity)
                elif sentiment_analysis.polarity < -0.1:
                    emotions["sadness"] = abs(sentiment_analysis.polarity)
                else:
                    emotions["neutral"] = 1.0
            
            # Find dominant emotion
            if emotions:
                dominant_emotion = max(emotions, key=emotions.get)
                emotional_dominance = emotions[dominant_emotion]
            else:
                dominant_emotion = "neutral"
                emotional_dominance = 0.0
            
            # Calculate emotional arousal (intensity)
            emotional_arousal = max(emotions.values()) if emotions else 0.0
            
            # Calculate emotional valence (positive/negative)
            positive_emotions = ["joy", "trust", "anticipation", "surprise"]
            negative_emotions = ["sadness", "anger", "fear", "disgust"]
            
            positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
            negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
            
            if positive_score + negative_score > 0:
                emotional_valence = (positive_score - negative_score) / (positive_score + negative_score)
            else:
                emotional_valence = 0.0
            
            # Calculate confidence
            confidence = emotional_dominance if emotions else 0.0
            
            return EmotionAnalysis(
                text=text,
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                emotional_arousal=emotional_arousal,
                emotional_valence=emotional_valence,
                emotional_dominance=emotional_dominance,
                confidence=confidence,
                metadata={
                    "emotion_count": len(emotions),
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return EmotionAnalysis(
                text=text,
                emotions={},
                dominant_emotion="neutral",
                emotional_arousal=0.0,
                emotional_valence=0.0,
                emotional_dominance=0.0,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def analyze_tone(self, text: str) -> ToneAnalysis:
        """Analyze tone of text"""
        try:
            if not text or not text.strip():
                return ToneAnalysis(
                    text=text,
                    tones={},
                    dominant_tone="neutral",
                    formality_level=0.5,
                    politeness_level=0.5,
                    assertiveness_level=0.5,
                    confidence=0.0
                )
            
            tones = {}
            text_lower = text.lower()
            
            # Analyze each tone category
            for tone, words in self.tone_lexicon.items():
                tone_count = sum(1 for word in words if word in text_lower)
                if tone_count > 0:
                    tones[tone] = min(tone_count / len(words), 1.0)
            
            # Calculate specific tone metrics
            formality_level = self._calculate_formality(text)
            politeness_level = self._calculate_politeness(text)
            assertiveness_level = self._calculate_assertiveness(text)
            
            # Find dominant tone
            if tones:
                dominant_tone = max(tones, key=tones.get)
                confidence = tones[dominant_tone]
            else:
                dominant_tone = "neutral"
                confidence = 0.0
            
            return ToneAnalysis(
                text=text,
                tones=tones,
                dominant_tone=dominant_tone,
                formality_level=formality_level,
                politeness_level=politeness_level,
                assertiveness_level=assertiveness_level,
                confidence=confidence,
                metadata={
                    "tone_count": len(tones),
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing tone: {str(e)}")
            return ToneAnalysis(
                text=text,
                tones={},
                dominant_tone="neutral",
                formality_level=0.5,
                politeness_level=0.5,
                assertiveness_level=0.5,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_formality(self, text: str) -> float:
        """Calculate formality level of text"""
        try:
            formal_indicators = ["therefore", "furthermore", "consequently", "moreover", "nevertheless"]
            informal_indicators = ["yeah", "cool", "awesome", "gonna", "wanna", "gotta"]
            
            text_lower = text.lower()
            formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
            informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
            
            if formal_count + informal_count > 0:
                return formal_count / (formal_count + informal_count)
            else:
                # Use sentence structure as indicator
                sentences = text.split('.')
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
                return min(avg_sentence_length / 20, 1.0)  # Normalize to 0-1
        
        except Exception:
            return 0.5
    
    def _calculate_politeness(self, text: str) -> float:
        """Calculate politeness level of text"""
        try:
            polite_indicators = ["please", "thank you", "excuse me", "pardon", "kindly", "respectfully"]
            impolite_indicators = ["shut up", "stupid", "idiot", "damn", "hell", "crap"]
            
            text_lower = text.lower()
            polite_count = sum(1 for indicator in polite_indicators if indicator in text_lower)
            impolite_count = sum(1 for indicator in impolite_indicators if indicator in text_lower)
            
            if polite_count + impolite_count > 0:
                return polite_count / (polite_count + impolite_count)
            else:
                return 0.5  # Neutral
        
        except Exception:
            return 0.5
    
    def _calculate_assertiveness(self, text: str) -> float:
        """Calculate assertiveness level of text"""
        try:
            assertive_indicators = ["must", "should", "will", "definitely", "certainly", "absolutely"]
            tentative_indicators = ["maybe", "perhaps", "might", "could", "possibly", "potentially"]
            
            text_lower = text.lower()
            assertive_count = sum(1 for indicator in assertive_indicators if indicator in text_lower)
            tentative_count = sum(1 for indicator in tentative_indicators if indicator in text_lower)
            
            if assertive_count + tentative_count > 0:
                return assertive_count / (assertive_count + tentative_count)
            else:
                return 0.5  # Neutral
        
        except Exception:
            return 0.5
    
    async def analyze_bias(self, text: str) -> BiasAnalysis:
        """Analyze bias in text"""
        try:
            if not text or not text.strip():
                return BiasAnalysis(
                    text=text,
                    bias_scores={},
                    overall_bias_score=0.0,
                    detected_biases=[],
                    fairness_score=1.0,
                    diversity_score=1.0,
                    confidence=0.0
                )
            
            bias_scores = {}
            detected_biases = []
            text_lower = text.lower()
            
            # Check for each type of bias
            for bias_type, patterns in self.bias_patterns.items():
                bias_count = 0
                for pattern in patterns:
                    if pattern in text_lower:
                        bias_count += 1
                
                if bias_count > 0:
                    bias_score = min(bias_count / len(patterns), 1.0)
                    bias_scores[bias_type] = bias_score
                    if bias_score > 0.3:  # Threshold for detection
                        detected_biases.append(bias_type)
            
            # Calculate overall bias score
            overall_bias_score = max(bias_scores.values()) if bias_scores else 0.0
            
            # Calculate fairness score (inverse of bias)
            fairness_score = max(0.0, 1.0 - overall_bias_score)
            
            # Calculate diversity score (based on inclusive language)
            diversity_score = self._calculate_diversity_score(text)
            
            # Calculate confidence
            confidence = overall_bias_score if bias_scores else 0.0
            
            return BiasAnalysis(
                text=text,
                bias_scores=bias_scores,
                overall_bias_score=overall_bias_score,
                detected_biases=detected_biases,
                fairness_score=fairness_score,
                diversity_score=diversity_score,
                confidence=confidence,
                metadata={
                    "bias_types_detected": len(detected_biases),
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing bias: {str(e)}")
            return BiasAnalysis(
                text=text,
                bias_scores={},
                overall_bias_score=0.0,
                detected_biases=[],
                fairness_score=1.0,
                diversity_score=1.0,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_diversity_score(self, text: str) -> float:
        """Calculate diversity score based on inclusive language"""
        try:
            inclusive_indicators = ["inclusive", "diverse", "equitable", "accessible", "welcoming", "respectful"]
            exclusive_indicators = ["exclusive", "elite", "superior", "inferior", "excluded", "rejected"]
            
            text_lower = text.lower()
            inclusive_count = sum(1 for indicator in inclusive_indicators if indicator in text_lower)
            exclusive_count = sum(1 for indicator in exclusive_indicators if indicator in text_lower)
            
            if inclusive_count + exclusive_count > 0:
                return inclusive_count / (inclusive_count + exclusive_count)
            else:
                return 1.0  # Neutral/positive default
        
        except Exception:
            return 1.0
    
    async def analyze_safety(self, text: str) -> SafetyAnalysis:
        """Analyze content safety and toxicity"""
        try:
            if not text or not text.strip():
                return SafetyAnalysis(
                    text=text,
                    toxicity_score=0.0,
                    safety_level="safe",
                    detected_issues=[],
                    content_categories={},
                    moderation_flags=[],
                    confidence=0.0
                )
            
            toxicity_score = 0.0
            content_categories = {}
            moderation_flags = []
            
            # Method 1: Toxicity classifier (if available)
            if self.toxicity_classifier:
                try:
                    toxicity_scores = self.toxicity_classifier(text)
                    for score in toxicity_scores:
                        category = score['label'].lower()
                        confidence = score['score']
                        content_categories[category] = confidence
                        
                        if category in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                            toxicity_score = max(toxicity_score, confidence)
                            if confidence > 0.5:
                                moderation_flags.append(category)
                except Exception as e:
                    logger.warning(f"Error with toxicity classifier: {str(e)}")
            
            # Method 2: Rule-based safety checks
            safety_issues = self._check_safety_patterns(text)
            if safety_issues:
                toxicity_score = max(toxicity_score, 0.7)  # High confidence for rule-based detection
                moderation_flags.extend(safety_issues)
            
            # Determine safety level
            if toxicity_score > 0.7:
                safety_level = "unsafe"
            elif toxicity_score > 0.3:
                safety_level = "caution"
            else:
                safety_level = "safe"
            
            # Calculate confidence
            confidence = toxicity_score if toxicity_score > 0 else 0.0
            
            return SafetyAnalysis(
                text=text,
                toxicity_score=toxicity_score,
                safety_level=safety_level,
                detected_issues=safety_issues,
                content_categories=content_categories,
                moderation_flags=moderation_flags,
                confidence=confidence,
                metadata={
                    "moderation_flags_count": len(moderation_flags),
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing safety: {str(e)}")
            return SafetyAnalysis(
                text=text,
                toxicity_score=0.0,
                safety_level="safe",
                detected_issues=[],
                content_categories={},
                moderation_flags=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _check_safety_patterns(self, text: str) -> List[str]:
        """Check for safety issues using pattern matching"""
        try:
            issues = []
            text_lower = text.lower()
            
            # Check for profanity
            profanity_patterns = ["fuck", "shit", "damn", "hell", "bitch", "asshole"]
            if any(pattern in text_lower for pattern in profanity_patterns):
                issues.append("profanity")
            
            # Check for threats
            threat_patterns = ["kill you", "hurt you", "destroy you", "attack you"]
            if any(pattern in text_lower for pattern in threat_patterns):
                issues.append("threat")
            
            # Check for harassment
            harassment_patterns = ["stupid", "idiot", "moron", "loser", "pathetic"]
            if any(pattern in text_lower for pattern in harassment_patterns):
                issues.append("harassment")
            
            return issues
        
        except Exception:
            return []
    
    async def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment, emotion, tone, bias, and safety analysis"""
        try:
            # Run all analyses in parallel
            sentiment_task = asyncio.create_task(self.analyze_sentiment(text))
            emotion_task = asyncio.create_task(self.analyze_emotions(text))
            tone_task = asyncio.create_task(self.analyze_tone(text))
            bias_task = asyncio.create_task(self.analyze_bias(text))
            safety_task = asyncio.create_task(self.analyze_safety(text))
            
            # Wait for all analyses to complete
            sentiment_result = await sentiment_task
            emotion_result = await emotion_task
            tone_result = await tone_task
            bias_result = await bias_task
            safety_result = await safety_task
            
            # Combine results
            comprehensive_result = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "sentiment": asdict(sentiment_result),
                "emotions": asdict(emotion_result),
                "tone": asdict(tone_result),
                "bias": asdict(bias_result),
                "safety": asdict(safety_result),
                "overall_score": self._calculate_overall_score(
                    sentiment_result, emotion_result, tone_result, bias_result, safety_result
                )
            }
            
            # Store in history
            self.analysis_history.append(comprehensive_result)
            
            return comprehensive_result
        
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _calculate_overall_score(self, sentiment, emotion, tone, bias, safety) -> float:
        """Calculate overall quality score from all analyses"""
        try:
            # Weighted combination of all scores
            weights = {
                "sentiment": 0.25,
                "emotion": 0.20,
                "tone": 0.20,
                "bias": 0.20,
                "safety": 0.15
            }
            
            # Convert to 0-1 scale
            sentiment_score = (sentiment.polarity + 1) / 2  # -1 to 1 -> 0 to 1
            emotion_score = emotion.confidence
            tone_score = tone.confidence
            bias_score = bias.fairness_score  # Higher is better
            safety_score = 1.0 - safety.toxicity_score  # Higher is better
            
            overall_score = (
                weights["sentiment"] * sentiment_score +
                weights["emotion"] * emotion_score +
                weights["tone"] * tone_score +
                weights["bias"] * bias_score +
                weights["safety"] * safety_score
            )
            
            return overall_score
        
        except Exception:
            return 0.5
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return self.analysis_history[-limit:]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.analysis_history:
            return {}
        
        try:
            sentiments = [analysis["sentiment"]["sentiment_label"] for analysis in self.analysis_history]
            emotions = [analysis["emotions"]["dominant_emotion"] for analysis in self.analysis_history]
            tones = [analysis["tone"]["dominant_tone"] for analysis in self.analysis_history]
            safety_levels = [analysis["safety"]["safety_level"] for analysis in self.analysis_history]
            
            return {
                "total_analyses": len(self.analysis_history),
                "sentiment_distribution": dict(Counter(sentiments)),
                "emotion_distribution": dict(Counter(emotions)),
                "tone_distribution": dict(Counter(tones)),
                "safety_distribution": dict(Counter(safety_levels)),
                "average_overall_score": np.mean([analysis.get("overall_score", 0.5) for analysis in self.analysis_history])
            }
        
        except Exception as e:
            logger.error(f"Error calculating analysis statistics: {str(e)}")
            return {}


# Global sentiment emotion analyzer instance
_sentiment_analyzer: Optional[SentimentEmotionAnalyzer] = None


def get_sentiment_emotion_analyzer(model_storage_path: str = "sentiment_models") -> SentimentEmotionAnalyzer:
    """Get or create global sentiment emotion analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentEmotionAnalyzer(model_storage_path)
    return _sentiment_analyzer


# Example usage
async def main():
    """Example usage of sentiment emotion analyzer"""
    analyzer = get_sentiment_emotion_analyzer()
    
    # Test text
    test_text = "I'm absolutely thrilled with this amazing product! It's fantastic and exceeded all my expectations."
    
    # Comprehensive analysis
    result = await analyzer.comprehensive_analysis(test_text)
    
    print("Comprehensive Analysis Result:")
    print(f"Sentiment: {result['sentiment']['sentiment_label']} (polarity: {result['sentiment']['polarity']:.3f})")
    print(f"Dominant Emotion: {result['emotions']['dominant_emotion']}")
    print(f"Dominant Tone: {result['tone']['dominant_tone']}")
    print(f"Bias Score: {result['bias']['overall_bias_score']:.3f}")
    print(f"Safety Level: {result['safety']['safety_level']}")
    print(f"Overall Score: {result['overall_score']:.3f}")
    
    # Get statistics
    stats = analyzer.get_analysis_statistics()
    print(f"\nAnalysis Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())

























