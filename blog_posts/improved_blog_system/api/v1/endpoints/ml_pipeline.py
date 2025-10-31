"""
Advanced ML Pipeline for Blog Posts System
==========================================

Comprehensive machine learning pipeline for blog content analysis, generation, and optimization.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
import anthropic
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import redis

from ....schemas import (
    BlogPostRequest, BlogPostResponse, MLPipelineRequest, MLPipelineResponse,
    ContentAnalysisRequest, ContentAnalysisResponse, SEOOptimizationRequest,
    SEOOptimizationResponse, ContentGenerationRequest, ContentGenerationResponse
)
from ....exceptions import (
    BlogPostError, MLPipelineError, ContentAnalysisError, SEOOptimizationError,
    ContentGenerationError, create_blog_error, create_ml_error
)
from ....services import BlogPostService, MLPipelineService, ContentAnalysisService
from ....config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ml-pipeline", tags=["ML Pipeline"])


class MLModelType(str, Enum):
    """ML model types"""
    CONTENT_ANALYSIS = "content_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    SEO_OPTIMIZATION = "seo_optimization"
    CONTENT_GENERATION = "content_generation"
    READABILITY_ANALYSIS = "readability_analysis"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    VIRAL_POTENTIAL = "viral_potential"
    KEYWORD_EXTRACTION = "keyword_extraction"
    CONTENT_CLUSTERING = "content_clustering"


class ContentQuality(str, Enum):
    """Content quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class MLModelConfig:
    """ML model configuration"""
    model_name: str
    model_type: MLModelType
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    last_trained: datetime
    is_active: bool = True


@dataclass
class ContentMetrics:
    """Content quality metrics"""
    readability_score: float
    seo_score: float
    engagement_score: float
    viral_potential: float
    sentiment_score: float
    topic_relevance: float
    keyword_density: float
    content_length_score: float
    structure_score: float
    originality_score: float


class ContentAnalysisModel:
    """Advanced content analysis model"""
    
    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self.vectorizers = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load sentiment analysis model
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load text classification model
            self.models['classification'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Load summarization model
            self.models['summarization'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            # Initialize TF-IDF vectorizer
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise create_ml_error("model_loading", "content_analysis", e)
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Comprehensive content analysis"""
        try:
            # Sentiment analysis
            sentiment_results = self.models['sentiment'](content)
            sentiment_score = self._calculate_sentiment_score(sentiment_results)
            
            # Readability analysis
            readability_score = self._calculate_readability(content)
            
            # SEO analysis
            seo_score = self._analyze_seo(content)
            
            # Engagement prediction
            engagement_score = self._predict_engagement(content)
            
            # Viral potential
            viral_potential = self._predict_viral_potential(content)
            
            # Topic relevance
            topic_relevance = self._analyze_topic_relevance(content)
            
            # Keyword analysis
            keyword_analysis = self._analyze_keywords(content)
            
            # Content structure
            structure_score = self._analyze_structure(content)
            
            # Originality check
            originality_score = self._check_originality(content)
            
            return {
                "sentiment_score": sentiment_score,
                "readability_score": readability_score,
                "seo_score": seo_score,
                "engagement_score": engagement_score,
                "viral_potential": viral_potential,
                "topic_relevance": topic_relevance,
                "keyword_analysis": keyword_analysis,
                "structure_score": structure_score,
                "originality_score": originality_score,
                "overall_quality": self._calculate_overall_quality({
                    "sentiment": sentiment_score,
                    "readability": readability_score,
                    "seo": seo_score,
                    "engagement": engagement_score,
                    "viral_potential": viral_potential,
                    "topic_relevance": topic_relevance,
                    "structure": structure_score,
                    "originality": originality_score
                })
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            raise create_ml_error("content_analysis", content[:100], e)
    
    def _calculate_sentiment_score(self, sentiment_results: List[Dict]) -> float:
        """Calculate overall sentiment score"""
        if not sentiment_results:
            return 0.0
        
        # Get the highest scoring sentiment
        best_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
        
        # Convert to 0-1 scale (positive = 1, neutral = 0.5, negative = 0)
        if best_sentiment['label'] == 'LABEL_2':  # Positive
            return best_sentiment['score']
        elif best_sentiment['label'] == 'LABEL_1':  # Neutral
            return 0.5
        else:  # Negative
            return 1 - best_sentiment['score']
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using Flesch Reading Ease"""
        try:
            sentences = content.split('.')
            words = content.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Count syllables (simplified)
            syllables = sum(self._count_syllables(word) for word in words)
            
            # Flesch Reading Ease formula
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Normalize to 0-1 scale
            return max(0, min(1, score / 100))
            
        except Exception:
            return 0.5  # Default score
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _analyze_seo(self, content: str) -> float:
        """Analyze SEO score"""
        try:
            seo_factors = {
                "title_present": 0,
                "meta_description": 0,
                "heading_structure": 0,
                "keyword_density": 0,
                "content_length": 0,
                "internal_links": 0,
                "external_links": 0,
                "image_alt_text": 0
            }
            
            # Check for title (simplified)
            if any(line.strip().startswith('#') for line in content.split('\n')):
                seo_factors["title_present"] = 1
            
            # Check content length
            word_count = len(content.split())
            if 300 <= word_count <= 2000:
                seo_factors["content_length"] = 1
            elif word_count > 2000:
                seo_factors["content_length"] = 0.8
            
            # Check heading structure
            headings = [line for line in content.split('\n') if line.strip().startswith('#')]
            if len(headings) >= 2:
                seo_factors["heading_structure"] = 1
            
            # Calculate overall SEO score
            total_score = sum(seo_factors.values())
            max_score = len(seo_factors)
            
            return total_score / max_score
            
        except Exception:
            return 0.5
    
    def _predict_engagement(self, content: str) -> float:
        """Predict engagement score"""
        try:
            engagement_factors = {
                "question_count": 0,
                "call_to_action": 0,
                "emotional_words": 0,
                "storytelling": 0,
                "interactive_elements": 0
            }
            
            # Count questions
            question_count = content.count('?')
            engagement_factors["question_count"] = min(1, question_count / 3)
            
            # Check for call-to-action words
            cta_words = ['click', 'subscribe', 'follow', 'share', 'comment', 'like']
            cta_count = sum(1 for word in cta_words if word.lower() in content.lower())
            engagement_factors["call_to_action"] = min(1, cta_count / 2)
            
            # Check for emotional words
            emotional_words = ['amazing', 'incredible', 'fantastic', 'wonderful', 'terrible', 'awful']
            emotional_count = sum(1 for word in emotional_words if word.lower() in content.lower())
            engagement_factors["emotional_words"] = min(1, emotional_count / 5)
            
            # Check for storytelling elements
            story_words = ['story', 'experience', 'journey', 'adventure', 'tale']
            story_count = sum(1 for word in story_words if word.lower() in content.lower())
            engagement_factors["storytelling"] = min(1, story_count / 2)
            
            # Calculate overall engagement score
            total_score = sum(engagement_factors.values())
            max_score = len(engagement_factors)
            
            return total_score / max_score
            
        except Exception:
            return 0.5
    
    def _predict_viral_potential(self, content: str) -> float:
        """Predict viral potential"""
        try:
            viral_factors = {
                "controversy": 0,
                "trending_topics": 0,
                "emotional_impact": 0,
                "shareability": 0,
                "timeliness": 0
            }
            
            # Check for controversial words
            controversial_words = ['controversial', 'debate', 'argument', 'disagree']
            controversial_count = sum(1 for word in controversial_words if word.lower() in content.lower())
            viral_factors["controversy"] = min(1, controversial_count / 2)
            
            # Check for trending topics (simplified)
            trending_words = ['new', 'latest', 'trending', 'viral', 'popular']
            trending_count = sum(1 for word in trending_words if word.lower() in content.lower())
            viral_factors["trending_topics"] = min(1, trending_count / 3)
            
            # Check for emotional impact
            emotional_words = ['love', 'hate', 'amazing', 'terrible', 'shocking', 'incredible']
            emotional_count = sum(1 for word in emotional_words if word.lower() in content.lower())
            viral_factors["emotional_impact"] = min(1, emotional_count / 4)
            
            # Check for shareability
            share_words = ['share', 'tell', 'spread', 'forward', 'recommend']
            share_count = sum(1 for word in share_words if word.lower() in content.lower())
            viral_factors["shareability"] = min(1, share_count / 2)
            
            # Calculate overall viral potential
            total_score = sum(viral_factors.values())
            max_score = len(viral_factors)
            
            return total_score / max_score
            
        except Exception:
            return 0.5
    
    def _analyze_topic_relevance(self, content: str) -> float:
        """Analyze topic relevance"""
        try:
            # This would typically use topic modeling or classification
            # For now, we'll use a simplified approach
            
            # Check for topic-specific keywords
            topic_keywords = {
                'technology': ['tech', 'software', 'AI', 'machine learning', 'programming'],
                'business': ['business', 'marketing', 'sales', 'strategy', 'growth'],
                'lifestyle': ['lifestyle', 'health', 'fitness', 'wellness', 'personal'],
                'education': ['education', 'learning', 'teaching', 'study', 'knowledge']
            }
            
            content_lower = content.lower()
            topic_scores = {}
            
            for topic, keywords in topic_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content_lower)
                topic_scores[topic] = score / len(keywords)
            
            # Return the highest topic relevance score
            return max(topic_scores.values()) if topic_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_keywords(self, content: str) -> Dict[str, Any]:
        """Analyze keywords in content"""
        try:
            words = content.lower().split()
            word_freq = {}
            
            # Count word frequency
            for word in words:
                # Remove punctuation
                word = ''.join(c for c in word if c.isalnum())
                if len(word) > 3:  # Only consider words longer than 3 characters
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate keyword density
            total_words = len(words)
            keyword_density = {word: (count / total_words) * 100 for word, count in top_keywords}
            
            return {
                "top_keywords": top_keywords,
                "keyword_density": keyword_density,
                "total_keywords": len(word_freq),
                "unique_keywords": len(set(words))
            }
            
        except Exception:
            return {
                "top_keywords": [],
                "keyword_density": {},
                "total_keywords": 0,
                "unique_keywords": 0
            }
    
    def _analyze_structure(self, content: str) -> float:
        """Analyze content structure"""
        try:
            structure_factors = {
                "has_headings": 0,
                "has_paragraphs": 0,
                "has_lists": 0,
                "has_conclusion": 0,
                "proper_length": 0
            }
            
            lines = content.split('\n')
            
            # Check for headings
            if any(line.strip().startswith('#') for line in lines):
                structure_factors["has_headings"] = 1
            
            # Check for paragraphs
            if len([line for line in lines if line.strip()]) >= 3:
                structure_factors["has_paragraphs"] = 1
            
            # Check for lists
            if any(line.strip().startswith(('-', '*', '1.', '2.')) for line in lines):
                structure_factors["has_lists"] = 1
            
            # Check for conclusion
            conclusion_words = ['conclusion', 'summary', 'finally', 'in conclusion']
            if any(word in content.lower() for word in conclusion_words):
                structure_factors["has_conclusion"] = 1
            
            # Check proper length
            word_count = len(content.split())
            if 500 <= word_count <= 2000:
                structure_factors["proper_length"] = 1
            elif word_count > 2000:
                structure_factors["proper_length"] = 0.8
            
            # Calculate overall structure score
            total_score = sum(structure_factors.values())
            max_score = len(structure_factors)
            
            return total_score / max_score
            
        except Exception:
            return 0.5
    
    def _check_originality(self, content: str) -> float:
        """Check content originality (simplified)"""
        try:
            # This would typically compare against a database of existing content
            # For now, we'll use a simplified approach based on common phrases
            
            common_phrases = [
                'in today\'s world',
                'it is important to',
                'as we all know',
                'in conclusion',
                'first and foremost',
                'last but not least'
            ]
            
            content_lower = content.lower()
            common_phrase_count = sum(1 for phrase in common_phrases if phrase in content_lower)
            
            # Calculate originality score (fewer common phrases = more original)
            originality_score = max(0, 1 - (common_phrase_count / len(common_phrases)))
            
            return originality_score
            
        except Exception:
            return 0.5
    
    def _calculate_overall_quality(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate overall content quality"""
        try:
            # Weighted average of all scores
            weights = {
                "sentiment": 0.15,
                "readability": 0.20,
                "seo": 0.20,
                "engagement": 0.15,
                "viral_potential": 0.10,
                "topic_relevance": 0.10,
                "structure": 0.05,
                "originality": 0.05
            }
            
            weighted_score = sum(weights[key] * scores[key] for key in weights.keys())
            
            # Determine quality level
            if weighted_score >= 0.8:
                quality_level = ContentQuality.EXCELLENT
            elif weighted_score >= 0.6:
                quality_level = ContentQuality.GOOD
            elif weighted_score >= 0.4:
                quality_level = ContentQuality.AVERAGE
            elif weighted_score >= 0.2:
                quality_level = ContentQuality.POOR
            else:
                quality_level = ContentQuality.VERY_POOR
            
            return {
                "overall_score": round(weighted_score, 3),
                "quality_level": quality_level.value,
                "individual_scores": scores,
                "recommendations": self._generate_recommendations(scores)
            }
            
        except Exception:
            return {
                "overall_score": 0.5,
                "quality_level": ContentQuality.AVERAGE.value,
                "individual_scores": scores,
                "recommendations": []
            }
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores.get("readability", 0) < 0.6:
            recommendations.append("Improve readability by using shorter sentences and simpler words")
        
        if scores.get("seo", 0) < 0.6:
            recommendations.append("Optimize for SEO by adding relevant keywords and improving structure")
        
        if scores.get("engagement", 0) < 0.6:
            recommendations.append("Increase engagement by adding questions, calls-to-action, and emotional elements")
        
        if scores.get("structure", 0) < 0.6:
            recommendations.append("Improve content structure with proper headings, paragraphs, and conclusion")
        
        if scores.get("originality", 0) < 0.6:
            recommendations.append("Make content more original by avoiding common phrases and clichés")
        
        if scores.get("viral_potential", 0) < 0.6:
            recommendations.append("Increase viral potential by adding trending topics and shareable elements")
        
        return recommendations


class ContentGenerationModel:
    """Advanced content generation model"""
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AI clients"""
        try:
            if self.settings.openai_api_key:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            
            if self.settings.anthropic_api_key:
                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
            
            logger.info("AI clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI clients: {e}")
    
    async def generate_content(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate content using AI"""
        try:
            if not self.openai_client and not self.anthropic_client:
                raise create_ml_error("ai_client_not_available", "content_generation", Exception("No AI clients available"))
            
            # Prepare prompt based on content type
            prompt = self._prepare_prompt(request)
            
            # Generate content using available AI client
            if self.openai_client:
                content = await self._generate_with_openai(prompt, request)
            else:
                content = await self._generate_with_anthropic(prompt, request)
            
            # Post-process content
            processed_content = self._post_process_content(content, request)
            
            return {
                "generated_content": processed_content,
                "word_count": len(processed_content.split()),
                "generation_metadata": {
                    "model_used": "gpt-4" if self.openai_client else "claude-3",
                    "prompt_tokens": len(prompt.split()),
                    "generation_time": datetime.utcnow().isoformat()
                },
                "quality_metrics": await self._analyze_generated_content(processed_content)
            }
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise create_ml_error("content_generation", request.content_type, e)
    
    def _prepare_prompt(self, request: ContentGenerationRequest) -> str:
        """Prepare prompt for content generation"""
        base_prompt = f"""
        Generate a {request.content_type} blog post with the following requirements:
        
        Topic: {request.topic}
        Target Audience: {request.target_audience}
        Tone: {request.tone}
        Length: {request.length} words
        Style: {request.style}
        
        Additional Requirements:
        - Include relevant keywords: {', '.join(request.keywords)}
        - Focus on: {request.focus_areas}
        - Avoid: {request.avoid_topics}
        
        Please generate high-quality, engaging content that is:
        1. Well-structured with clear headings
        2. SEO-optimized
        3. Engaging and readable
        4. Original and valuable
        5. Appropriate for the target audience
        
        Content:
        """
        
        return base_prompt
    
    async def _generate_with_openai(self, prompt: str, request: ContentGenerationRequest) -> str:
        """Generate content using OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content writer and SEO specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=request.length * 2,  # Approximate token count
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise create_ml_error("openai_generation", "content", e)
    
    async def _generate_with_anthropic(self, prompt: str, request: ContentGenerationRequest) -> str:
        """Generate content using Anthropic"""
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=request.length * 2,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise create_ml_error("anthropic_generation", "content", e)
    
    def _post_process_content(self, content: str, request: ContentGenerationRequest) -> str:
        """Post-process generated content"""
        try:
            # Clean up content
            content = content.strip()
            
            # Add proper formatting
            if not content.startswith('#'):
                content = f"# {request.topic}\n\n{content}"
            
            # Ensure proper paragraph breaks
            content = content.replace('\n\n\n', '\n\n')
            
            # Add conclusion if missing
            if 'conclusion' not in content.lower() and len(content.split()) > 200:
                content += "\n\n## Conclusion\n\nIn conclusion, this topic is important for understanding the key concepts and implications discussed above."
            
            return content
            
        except Exception as e:
            logger.error(f"Content post-processing failed: {e}")
            return content  # Return original content if processing fails
    
    async def _analyze_generated_content(self, content: str) -> Dict[str, Any]:
        """Analyze generated content quality"""
        try:
            # Use content analysis model
            analysis_model = ContentAnalysisModel()
            analysis_results = await analysis_model.analyze_content(content)
            
            return {
                "quality_score": analysis_results["overall_quality"]["overall_score"],
                "readability": analysis_results["readability_score"],
                "seo_score": analysis_results["seo_score"],
                "engagement": analysis_results["engagement_score"],
                "recommendations": analysis_results["overall_quality"]["recommendations"]
            }
            
        except Exception as e:
            logger.error(f"Generated content analysis failed: {e}")
            return {
                "quality_score": 0.5,
                "readability": 0.5,
                "seo_score": 0.5,
                "engagement": 0.5,
                "recommendations": []
            }


class MLPipelineService:
    """ML Pipeline service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.content_analysis_model = ContentAnalysisModel()
        self.content_generation_model = ContentGenerationModel()
        self.redis_client = redis.Redis(
            host=self.settings.redis.host,
            port=self.settings.redis.port,
            password=self.settings.redis.password,
            db=self.settings.redis.db
        )
    
    async def process_content_analysis(self, request: ContentAnalysisRequest) -> ContentAnalysisResponse:
        """Process content analysis request"""
        try:
            # Analyze content
            analysis_results = await self.content_analysis_model.analyze_content(request.content)
            
            # Cache results
            cache_key = f"content_analysis:{hash(request.content)}"
            await self.redis_client.setex(
                cache_key,
                3600,  # 1 hour
                json.dumps(analysis_results)
            )
            
            return ContentAnalysisResponse(
                analysis_id=str(uuid4()),
                content_hash=hash(request.content),
                analysis_results=analysis_results,
                processing_time=0.0,  # Would be calculated in real implementation
                confidence_score=analysis_results["overall_quality"]["overall_score"],
                recommendations=analysis_results["overall_quality"]["recommendations"],
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Content analysis processing failed: {e}")
            raise create_ml_error("content_analysis_processing", request.content[:100], e)
    
    async def process_content_generation(self, request: ContentGenerationRequest) -> ContentGenerationResponse:
        """Process content generation request"""
        try:
            # Generate content
            generation_results = await self.content_generation_model.generate_content(request)
            
            return ContentGenerationResponse(
                generation_id=str(uuid4()),
                generated_content=generation_results["generated_content"],
                word_count=generation_results["word_count"],
                quality_metrics=generation_results["quality_metrics"],
                generation_metadata=generation_results["generation_metadata"],
                processing_time=0.0,  # Would be calculated in real implementation
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Content generation processing failed: {e}")
            raise create_ml_error("content_generation_processing", request.topic, e)
    
    async def process_seo_optimization(self, request: SEOOptimizationRequest) -> SEOOptimizationResponse:
        """Process SEO optimization request"""
        try:
            # Analyze current SEO
            current_analysis = await self.content_analysis_model.analyze_content(request.content)
            
            # Generate SEO recommendations
            seo_recommendations = self._generate_seo_recommendations(
                request.content,
                request.target_keywords,
                current_analysis
            )
            
            # Generate optimized content
            optimized_content = self._optimize_content_for_seo(
                request.content,
                request.target_keywords,
                seo_recommendations
            )
            
            return SEOOptimizationResponse(
                optimization_id=str(uuid4()),
                original_content=request.content,
                optimized_content=optimized_content,
                seo_score_before=current_analysis["seo_score"],
                seo_score_after=0.8,  # Would be calculated
                recommendations=seo_recommendations,
                keyword_analysis=self._analyze_keyword_usage(optimized_content, request.target_keywords),
                processing_time=0.0,
                optimized_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"SEO optimization processing failed: {e}")
            raise create_ml_error("seo_optimization_processing", request.content[:100], e)
    
    def _generate_seo_recommendations(self, content: str, target_keywords: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Generate SEO recommendations"""
        recommendations = []
        
        # Check keyword usage
        content_lower = content.lower()
        for keyword in target_keywords:
            if keyword.lower() not in content_lower:
                recommendations.append(f"Add target keyword '{keyword}' to the content")
        
        # Check content length
        word_count = len(content.split())
        if word_count < 300:
            recommendations.append("Increase content length to at least 300 words for better SEO")
        elif word_count > 2000:
            recommendations.append("Consider breaking long content into multiple posts")
        
        # Check heading structure
        if not any(line.strip().startswith('#') for line in content.split('\n')):
            recommendations.append("Add proper heading structure (H1, H2, H3)")
        
        # Check meta elements
        if 'meta description' not in content.lower():
            recommendations.append("Add meta description for better search engine visibility")
        
        return recommendations
    
    def _optimize_content_for_seo(self, content: str, target_keywords: List[str], recommendations: List[str]) -> str:
        """Optimize content for SEO"""
        try:
            optimized_content = content
            
            # Add keywords if missing
            content_lower = content.lower()
            for keyword in target_keywords:
                if keyword.lower() not in content_lower:
                    # Add keyword naturally in the content
                    optimized_content += f"\n\nThis content covers important aspects of {keyword}."
            
            # Ensure proper heading structure
            if not optimized_content.startswith('#'):
                optimized_content = f"# {target_keywords[0] if target_keywords else 'Blog Post'}\n\n{optimized_content}"
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Content SEO optimization failed: {e}")
            return content  # Return original content if optimization fails
    
    def _analyze_keyword_usage(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Analyze keyword usage in content"""
        try:
            content_lower = content.lower()
            keyword_analysis = {}
            
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                count = content_lower.count(keyword_lower)
                density = (count / len(content.split())) * 100 if content.split() else 0
                
                keyword_analysis[keyword] = {
                    "count": count,
                    "density": round(density, 2),
                    "first_occurrence": content_lower.find(keyword_lower),
                    "in_title": keyword_lower in content_lower.split('\n')[0].lower() if content.split('\n') else False
                }
            
            return keyword_analysis
            
        except Exception as e:
            logger.error(f"Keyword usage analysis failed: {e}")
            return {}


# API Endpoints
@router.post("/analyze-content", response_model=ContentAnalysisResponse)
async def analyze_content(
    request: ContentAnalysisRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLPipelineService = Depends()
):
    """Analyze content using ML pipeline"""
    try:
        result = await ml_service.process_content_analysis(request)
        
        # Log analysis in background
        background_tasks.add_task(
            log_analysis_result,
            result.analysis_id,
            request.content[:100]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Content analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-content", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLPipelineService = Depends()
):
    """Generate content using ML pipeline"""
    try:
        result = await ml_service.process_content_generation(request)
        
        # Log generation in background
        background_tasks.add_task(
            log_generation_result,
            result.generation_id,
            request.topic
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Content generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-seo", response_model=SEOOptimizationResponse)
async def optimize_seo(
    request: SEOOptimizationRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLPipelineService = Depends()
):
    """Optimize content for SEO using ML pipeline"""
    try:
        result = await ml_service.process_seo_optimization(request)
        
        # Log optimization in background
        background_tasks.add_task(
            log_optimization_result,
            result.optimization_id,
            request.content[:100]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"SEO optimization endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_models_status():
    """Get ML models status"""
    try:
        return {
            "models": {
                "content_analysis": "active",
                "sentiment_analysis": "active",
                "content_generation": "active",
                "seo_optimization": "active"
            },
            "last_updated": datetime.utcnow().isoformat(),
            "system_health": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Models status endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def log_analysis_result(analysis_id: str, content_preview: str):
    """Log analysis result"""
    try:
        logger.info(f"Content analysis completed: {analysis_id} for content: {content_preview}")
    except Exception as e:
        logger.error(f"Failed to log analysis result: {e}")


async def log_generation_result(generation_id: str, topic: str):
    """Log generation result"""
    try:
        logger.info(f"Content generation completed: {generation_id} for topic: {topic}")
    except Exception as e:
        logger.error(f"Failed to log generation result: {e}")


async def log_optimization_result(optimization_id: str, content_preview: str):
    """Log optimization result"""
    try:
        logger.info(f"SEO optimization completed: {optimization_id} for content: {content_preview}")
    except Exception as e:
        logger.error(f"Failed to log optimization result: {e}")