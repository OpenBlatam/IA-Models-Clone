"""
AI Content Analyzer - Advanced AI-powered content analysis
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib
import re

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class AIContentAnalysis:
    """AI-powered content analysis result"""
    content_id: str
    sentiment_score: float
    sentiment_label: str
    emotion_scores: Dict[str, float]
    topic_classification: Dict[str, float]
    language_detection: str
    readability_score: float
    complexity_score: float
    key_phrases: List[str]
    named_entities: List[Dict[str, Any]]
    content_quality_score: float
    ai_confidence: float
    analysis_timestamp: datetime


@dataclass
class ContentInsights:
    """Comprehensive content insights"""
    content_id: str
    summary: str
    main_topics: List[str]
    key_insights: List[str]
    recommendations: List[str]
    content_type: str
    target_audience: str
    engagement_prediction: float
    seo_score: float
    brand_voice_alignment: float
    analysis_timestamp: datetime


class AIContentAnalyzer:
    """Advanced AI-powered content analyzer"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_analyzer = None
        self.topic_classifier = None
        self.language_detector = None
        self.ner_pipeline = None
        self.summarizer = None
        self.sentence_transformer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        
    async def initialize(self) -> None:
        """Initialize AI models"""
        try:
            logger.info("Initializing AI Content Analyzer...")
            
            # Load models asynchronously
            await asyncio.gather(
                self._load_sentiment_analyzer(),
                self._load_emotion_analyzer(),
                self._load_topic_classifier(),
                self._load_language_detector(),
                self._load_ner_pipeline(),
                self._load_summarizer(),
                self._load_sentence_transformer()
            )
            
            self.models_loaded = True
            logger.info("AI Content Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI Content Analyzer: {e}")
            raise
    
    async def _load_sentiment_analyzer(self) -> None:
        """Load sentiment analysis model"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    async def _load_emotion_analyzer(self) -> None:
        """Load emotion analysis model"""
        try:
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load emotion analyzer: {e}")
            self.emotion_analyzer = None
    
    async def _load_topic_classifier(self) -> None:
        """Load topic classification model"""
        try:
            self.topic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load topic classifier: {e}")
            self.topic_classifier = None
    
    async def _load_language_detector(self) -> None:
        """Load language detection model"""
        try:
            self.language_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load language detector: {e}")
            self.language_detector = None
    
    async def _load_ner_pipeline(self) -> None:
        """Load named entity recognition pipeline"""
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=self.device,
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.warning(f"Failed to load NER pipeline: {e}")
            self.ner_pipeline = None
    
    async def _load_summarizer(self) -> None:
        """Load text summarization model"""
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load summarizer: {e}")
            self.summarizer = None
    
    async def _load_sentence_transformer(self) -> None:
        """Load sentence transformer for embeddings"""
        try:
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    async def analyze_content_ai(self, content: str, content_id: str = "") -> AIContentAnalysis:
        """Perform comprehensive AI analysis of content"""
        
        if not self.models_loaded:
            raise Exception("AI models not loaded. Call initialize() first.")
        
        try:
            # Run all analyses in parallel
            results = await asyncio.gather(
                self._analyze_sentiment(content),
                self._analyze_emotions(content),
                self._classify_topics(content),
                self._detect_language(content),
                self._extract_entities(content),
                self._calculate_readability(content),
                self._calculate_complexity(content),
                self._extract_key_phrases(content),
                return_exceptions=True
            )
            
            # Extract results
            sentiment_score, sentiment_label = results[0] if not isinstance(results[0], Exception) else (0.0, "neutral")
            emotion_scores = results[1] if not isinstance(results[1], Exception) else {}
            topic_classification = results[2] if not isinstance(results[2], Exception) else {}
            language_detection = results[3] if not isinstance(results[3], Exception) else "unknown"
            named_entities = results[4] if not isinstance(results[4], Exception) else []
            readability_score = results[5] if not isinstance(results[5], Exception) else 0.0
            complexity_score = results[6] if not isinstance(results[6], Exception) else 0.0
            key_phrases = results[7] if not isinstance(results[7], Exception) else []
            
            # Calculate content quality score
            content_quality_score = await self._calculate_content_quality(
                sentiment_score, readability_score, complexity_score, len(key_phrases)
            )
            
            # Calculate AI confidence
            ai_confidence = await self._calculate_ai_confidence(results)
            
            return AIContentAnalysis(
                content_id=content_id,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                emotion_scores=emotion_scores,
                topic_classification=topic_classification,
                language_detection=language_detection,
                readability_score=readability_score,
                complexity_score=complexity_score,
                key_phrases=key_phrases,
                named_entities=named_entities,
                content_quality_score=content_quality_score,
                ai_confidence=ai_confidence,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in AI content analysis: {e}")
            raise
    
    async def _analyze_sentiment(self, content: str) -> Tuple[float, str]:
        """Analyze sentiment of content"""
        if not self.sentiment_analyzer:
            return 0.0, "neutral"
        
        try:
            result = self.sentiment_analyzer(content[:512])  # Limit length
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Convert to sentiment score (-1 to 1)
            if 'positive' in label:
                sentiment_score = score
            elif 'negative' in label:
                sentiment_score = -score
            else:
                sentiment_score = 0.0
            
            return sentiment_score, label
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0, "neutral"
    
    async def _analyze_emotions(self, content: str) -> Dict[str, float]:
        """Analyze emotions in content"""
        if not self.emotion_analyzer:
            return {}
        
        try:
            result = self.emotion_analyzer(content[:512])
            return {result[0]['label']: result[0]['score']}
            
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return {}
    
    async def _classify_topics(self, content: str) -> Dict[str, float]:
        """Classify topics in content"""
        if not self.topic_classifier:
            return {}
        
        try:
            candidate_labels = [
                "technology", "business", "health", "education", "entertainment",
                "sports", "politics", "science", "travel", "food", "fashion",
                "finance", "marketing", "news", "lifestyle"
            ]
            
            result = self.topic_classifier(content[:512], candidate_labels)
            
            return dict(zip(result['labels'], result['scores']))
            
        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            return {}
    
    async def _detect_language(self, content: str) -> str:
        """Detect language of content"""
        if not self.language_detector:
            return "unknown"
        
        try:
            result = self.language_detector(content[:512])
            return result[0]['label']
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content"""
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(content[:512])
            
            return [
                {
                    "text": entity['word'],
                    "label": entity['entity_group'],
                    "confidence": entity['score'],
                    "start": entity['start'],
                    "end": entity['end']
                }
                for entity in entities
            ]
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using AI"""
        try:
            # Simple readability calculation
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Flesch Reading Ease approximation
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            return max(0.0, min(100.0, readability))
            
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 0.0
    
    async def _calculate_complexity(self, content: str) -> float:
        """Calculate content complexity score"""
        try:
            words = content.split()
            unique_words = set(words)
            
            if not words:
                return 0.0
            
            # Lexical diversity
            lexical_diversity = len(unique_words) / len(words)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Sentence complexity
            sentences = re.split(r'[.!?]+', content)
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            
            # Combine metrics
            complexity = (lexical_diversity * 0.4 + 
                         (avg_word_length / 10) * 0.3 + 
                         (avg_sentence_length / 20) * 0.3)
            
            return min(1.0, complexity)
            
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            return 0.0
    
    async def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        try:
            # Simple key phrase extraction
            words = content.lower().split()
            
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
            }
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequency
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top key phrases
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            key_phrases = [word for word, freq in sorted_words[:10] if freq > 1]
            
            return key_phrases
            
        except Exception as e:
            logger.warning(f"Key phrase extraction failed: {e}")
            return []
    
    async def _calculate_content_quality(
        self, 
        sentiment_score: float, 
        readability_score: float, 
        complexity_score: float, 
        key_phrases_count: int
    ) -> float:
        """Calculate overall content quality score"""
        try:
            # Normalize scores
            sentiment_norm = (sentiment_score + 1) / 2  # -1 to 1 -> 0 to 1
            readability_norm = readability_score / 100  # 0 to 100 -> 0 to 1
            complexity_norm = complexity_score  # Already 0 to 1
            key_phrases_norm = min(1.0, key_phrases_count / 10)  # 0 to 10 -> 0 to 1
            
            # Weighted average
            quality_score = (
                sentiment_norm * 0.2 +
                readability_norm * 0.3 +
                complexity_norm * 0.3 +
                key_phrases_norm * 0.2
            )
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.warning(f"Content quality calculation failed: {e}")
            return 0.0
    
    async def _calculate_ai_confidence(self, results: List[Any]) -> float:
        """Calculate AI confidence based on successful analyses"""
        try:
            successful_analyses = sum(1 for result in results if not isinstance(result, Exception))
            total_analyses = len(results)
            
            return successful_analyses / total_analyses if total_analyses > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"AI confidence calculation failed: {e}")
            return 0.0
    
    async def generate_content_insights(self, content: str, content_id: str = "") -> ContentInsights:
        """Generate comprehensive content insights"""
        
        try:
            # Get AI analysis
            ai_analysis = await self.analyze_content_ai(content, content_id)
            
            # Generate summary
            summary = await self._generate_summary(content)
            
            # Extract main topics
            main_topics = list(ai_analysis.topic_classification.keys())[:3]
            
            # Generate key insights
            key_insights = await self._generate_key_insights(ai_analysis)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(ai_analysis)
            
            # Determine content type
            content_type = await self._determine_content_type(ai_analysis)
            
            # Predict target audience
            target_audience = await self._predict_target_audience(ai_analysis)
            
            # Predict engagement
            engagement_prediction = await self._predict_engagement(ai_analysis)
            
            # Calculate SEO score
            seo_score = await self._calculate_seo_score(content, ai_analysis)
            
            # Calculate brand voice alignment
            brand_voice_alignment = await self._calculate_brand_voice_alignment(ai_analysis)
            
            return ContentInsights(
                content_id=content_id,
                summary=summary,
                main_topics=main_topics,
                key_insights=key_insights,
                recommendations=recommendations,
                content_type=content_type,
                target_audience=target_audience,
                engagement_prediction=engagement_prediction,
                seo_score=seo_score,
                brand_voice_alignment=brand_voice_alignment,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            raise
    
    async def _generate_summary(self, content: str) -> str:
        """Generate content summary"""
        if not self.summarizer:
            # Fallback to simple summary
            sentences = content.split('.')
            return '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else content
        
        try:
            # Limit content length for summarization
            content_for_summary = content[:1024]
            summary = self.summarizer(content_for_summary, max_length=100, min_length=30)
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            sentences = content.split('.')
            return '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else content
    
    async def _generate_key_insights(self, ai_analysis: AIContentAnalysis) -> List[str]:
        """Generate key insights from AI analysis"""
        insights = []
        
        # Sentiment insights
        if ai_analysis.sentiment_score > 0.3:
            insights.append("Content has a positive tone that may engage readers")
        elif ai_analysis.sentiment_score < -0.3:
            insights.append("Content has a negative tone that may require attention")
        
        # Readability insights
        if ai_analysis.readability_score > 70:
            insights.append("Content is highly readable and accessible")
        elif ai_analysis.readability_score < 30:
            insights.append("Content may be too complex for general audience")
        
        # Topic insights
        if ai_analysis.topic_classification:
            top_topic = max(ai_analysis.topic_classification.items(), key=lambda x: x[1])
            insights.append(f"Primary topic focus: {top_topic[0]} (confidence: {top_topic[1]:.2f})")
        
        # Quality insights
        if ai_analysis.content_quality_score > 0.7:
            insights.append("High-quality content with good structure and engagement potential")
        elif ai_analysis.content_quality_score < 0.4:
            insights.append("Content may benefit from improvement in structure and clarity")
        
        return insights
    
    async def _generate_recommendations(self, ai_analysis: AIContentAnalysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Readability recommendations
        if ai_analysis.readability_score < 50:
            recommendations.append("Consider simplifying sentence structure and vocabulary")
        
        # Sentiment recommendations
        if abs(ai_analysis.sentiment_score) > 0.7:
            recommendations.append("Consider balancing emotional tone for broader appeal")
        
        # Complexity recommendations
        if ai_analysis.complexity_score > 0.8:
            recommendations.append("Content may be too complex - consider breaking into smaller sections")
        
        # Key phrases recommendations
        if len(ai_analysis.key_phrases) < 3:
            recommendations.append("Add more specific and descriptive key phrases")
        
        # Quality recommendations
        if ai_analysis.content_quality_score < 0.6:
            recommendations.append("Focus on improving content structure and engagement elements")
        
        return recommendations
    
    async def _determine_content_type(self, ai_analysis: AIContentAnalysis) -> str:
        """Determine content type based on analysis"""
        if not ai_analysis.topic_classification:
            return "general"
        
        top_topic = max(ai_analysis.topic_classification.items(), key=lambda x: x[1])
        
        content_type_mapping = {
            "technology": "technical",
            "business": "business",
            "health": "healthcare",
            "education": "educational",
            "entertainment": "entertainment",
            "news": "news",
            "marketing": "marketing"
        }
        
        return content_type_mapping.get(top_topic[0], "general")
    
    async def _predict_target_audience(self, ai_analysis: AIContentAnalysis) -> str:
        """Predict target audience based on analysis"""
        if ai_analysis.complexity_score > 0.7:
            return "expert"
        elif ai_analysis.complexity_score > 0.4:
            return "professional"
        else:
            return "general"
    
    async def _predict_engagement(self, ai_analysis: AIContentAnalysis) -> float:
        """Predict engagement potential"""
        # Combine multiple factors
        engagement_score = (
            (ai_analysis.sentiment_score + 1) / 2 * 0.3 +  # Sentiment (0-1)
            ai_analysis.readability_score / 100 * 0.3 +    # Readability (0-1)
            ai_analysis.content_quality_score * 0.4        # Quality (0-1)
        )
        
        return min(1.0, engagement_score)
    
    async def _calculate_seo_score(self, content: str, ai_analysis: AIContentAnalysis) -> float:
        """Calculate SEO score"""
        seo_factors = []
        
        # Content length
        word_count = len(content.split())
        if 300 <= word_count <= 2000:
            seo_factors.append(1.0)
        else:
            seo_factors.append(0.5)
        
        # Key phrases
        if len(ai_analysis.key_phrases) >= 3:
            seo_factors.append(1.0)
        else:
            seo_factors.append(0.6)
        
        # Readability
        if ai_analysis.readability_score > 60:
            seo_factors.append(1.0)
        else:
            seo_factors.append(0.7)
        
        # Named entities
        if len(ai_analysis.named_entities) > 0:
            seo_factors.append(1.0)
        else:
            seo_factors.append(0.8)
        
        return sum(seo_factors) / len(seo_factors)
    
    async def _calculate_brand_voice_alignment(self, ai_analysis: AIContentAnalysis) -> float:
        """Calculate brand voice alignment (placeholder)"""
        # This would typically compare against brand voice guidelines
        # For now, return a score based on content quality and consistency
        return ai_analysis.content_quality_score * 0.8 + 0.2
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of AI analyzer"""
        return {
            "status": "healthy" if self.models_loaded else "unhealthy",
            "models_loaded": self.models_loaded,
            "device": self.device,
            "available_models": {
                "sentiment_analyzer": self.sentiment_analyzer is not None,
                "emotion_analyzer": self.emotion_analyzer is not None,
                "topic_classifier": self.topic_classifier is not None,
                "language_detector": self.language_detector is not None,
                "ner_pipeline": self.ner_pipeline is not None,
                "summarizer": self.summarizer is not None,
                "sentence_transformer": self.sentence_transformer is not None
            },
            "timestamp": datetime.now().isoformat()
        }


# Global AI analyzer instance
ai_analyzer = AIContentAnalyzer()


async def initialize_ai_analyzer() -> None:
    """Initialize the global AI analyzer"""
    await ai_analyzer.initialize()


async def analyze_content_with_ai(content: str, content_id: str = "") -> AIContentAnalysis:
    """Analyze content using AI"""
    return await ai_analyzer.analyze_content_ai(content, content_id)


async def generate_ai_insights(content: str, content_id: str = "") -> ContentInsights:
    """Generate AI-powered content insights"""
    return await ai_analyzer.generate_content_insights(content, content_id)


async def get_ai_analyzer_health() -> Dict[str, Any]:
    """Get AI analyzer health status"""
    return await ai_analyzer.health_check()




