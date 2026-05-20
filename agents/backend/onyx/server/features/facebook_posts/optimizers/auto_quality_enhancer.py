from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import statistics
        from collections import Counter
        from collections import Counter
from typing import Any, List, Dict, Optional
import logging
"""
🎯 Auto Quality Enhancer - Mejora Automática de Calidad
=====================================================

Sistema de mejora automática de calidad basado en feedback continuo
y aprendizaje de patrones de éxito.
"""


# ===== DATA STRUCTURES =====

@dataclass
class QualityMetrics:
    """Métricas de calidad detalladas."""
    grammar_score: float
    readability_score: float
    sentiment_score: float
    engagement_potential: float
    creativity_score: float
    relevance_score: float
    overall_score: float
    improvement_areas: List[str]

@dataclass
class EnhancementResult:
    """Resultado de mejora de calidad."""
    original_text: str
    enhanced_text: str
    quality_improvement: float
    enhancements_applied: List[str]
    confidence_score: float
    processing_time: float

@dataclass
class LearningPattern:
    """Patrón de aprendizaje identificado."""
    pattern_type: str
    success_rate: float
    avg_quality_improvement: float
    usage_count: int
    last_used: datetime
    effectiveness_score: float

# ===== AUTO QUALITY ENHANCER =====

class AutoQualityEnhancer:
    """Mejora automática de calidad basada en feedback."""
    
    def __init__(self) -> Any:
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "needs_improvement": 0.6
        }
        self.enhancement_strategies = {
            "grammar": GrammarEnhancer(),
            "readability": ReadabilityEnhancer(),
            "engagement": EngagementEnhancer(),
            "creativity": CreativityEnhancer(),
            "sentiment": SentimentEnhancer()
        }
        self.learning_patterns = {}
        self.feedback_database = []
        self.improvement_history = []
    
    async def auto_enhance(self, post: Dict[str, Any]) -> EnhancementResult:
        """Mejora automática de calidad."""
        start_time = time.time()
        
        # Analyze current quality
        quality_metrics = await self._analyze_quality(post)
        
        # Check if enhancement is needed
        if quality_metrics.overall_score >= self.quality_thresholds["good"]:
            return EnhancementResult(
                original_text=post.get("content", ""),
                enhanced_text=post.get("content", ""),
                quality_improvement=0.0,
                enhancements_applied=["no_enhancement_needed"],
                confidence_score=1.0,
                processing_time=time.time() - start_time
            )
        
        # Apply enhancements
        enhanced_text = post.get("content", "")
        enhancements_applied = []
        
        # Grammar enhancement
        if quality_metrics.grammar_score < 0.8:
            grammar_result = await self.enhancement_strategies["grammar"].enhance(enhanced_text)
            enhanced_text = grammar_result["enhanced_text"]
            enhancements_applied.append("grammar_improvement")
        
        # Readability enhancement
        if quality_metrics.readability_score < 0.7:
            readability_result = await self.enhancement_strategies["readability"].enhance(enhanced_text)
            enhanced_text = readability_result["enhanced_text"]
            enhancements_applied.append("readability_improvement")
        
        # Engagement enhancement
        if quality_metrics.engagement_potential < 0.6:
            engagement_result = await self.enhancement_strategies["engagement"].enhance(enhanced_text)
            enhanced_text = engagement_result["enhanced_text"]
            enhancements_applied.append("engagement_improvement")
        
        # Creativity enhancement
        if quality_metrics.creativity_score < 0.7:
            creativity_result = await self.enhancement_strategies["creativity"].enhance(enhanced_text)
            enhanced_text = creativity_result["enhanced_text"]
            enhancements_applied.append("creativity_improvement")
        
        # Sentiment enhancement
        if quality_metrics.sentiment_score < 0.6:
            sentiment_result = await self.enhancement_strategies["sentiment"].enhance(enhanced_text)
            enhanced_text = sentiment_result["enhanced_text"]
            enhancements_applied.append("sentiment_improvement")
        
        # Analyze enhanced quality
        enhanced_post = {**post, "content": enhanced_text}
        enhanced_metrics = await self._analyze_quality(enhanced_post)
        
        # Calculate improvement
        quality_improvement = enhanced_metrics.overall_score - quality_metrics.overall_score
        
        # Calculate confidence
        confidence_score = self._calculate_enhancement_confidence(enhancements_applied, quality_improvement)
        
        processing_time = time.time() - start_time
        
        # Store improvement history
        self.improvement_history.append({
            "original_score": quality_metrics.overall_score,
            "enhanced_score": enhanced_metrics.overall_score,
            "improvement": quality_improvement,
            "enhancements": enhancements_applied,
            "timestamp": datetime.now().isoformat()
        })
        
        return EnhancementResult(
            original_text=post.get("content", ""),
            enhanced_text=enhanced_text,
            quality_improvement=quality_improvement,
            enhancements_applied=enhancements_applied,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    async def _analyze_quality(self, post: Dict[str, Any]) -> QualityMetrics:
        """Analizar calidad del post."""
        content = post.get("content", "")
        
        # Grammar analysis
        grammar_score = await self._analyze_grammar(content)
        
        # Readability analysis
        readability_score = await self._analyze_readability(content)
        
        # Sentiment analysis
        sentiment_score = await self._analyze_sentiment(content)
        
        # Engagement potential analysis
        engagement_potential = await self._analyze_engagement_potential(content)
        
        # Creativity analysis
        creativity_score = await self._analyze_creativity(content)
        
        # Relevance analysis
        relevance_score = await self._analyze_relevance(content, post.get("topic", ""))
        
        # Calculate overall score
        overall_score = (
            grammar_score * 0.2 +
            readability_score * 0.2 +
            sentiment_score * 0.15 +
            engagement_potential * 0.2 +
            creativity_score * 0.15 +
            relevance_score * 0.1
        )
        
        # Identify improvement areas
        improvement_areas = []
        if grammar_score < 0.8:
            improvement_areas.append("grammar")
        if readability_score < 0.7:
            improvement_areas.append("readability")
        if sentiment_score < 0.6:
            improvement_areas.append("sentiment")
        if engagement_potential < 0.6:
            improvement_areas.append("engagement")
        if creativity_score < 0.7:
            improvement_areas.append("creativity")
        if relevance_score < 0.7:
            improvement_areas.append("relevance")
        
        return QualityMetrics(
            grammar_score=grammar_score,
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            engagement_potential=engagement_potential,
            creativity_score=creativity_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            improvement_areas=improvement_areas
        )
    
    async def _analyze_grammar(self, content: str) -> float:
        """Analizar gramática del contenido."""
        # Simple grammar analysis
        sentences = content.split('.')
        word_count = len(content.split())
        
        if word_count == 0:
            return 0.0
        
        # Basic grammar checks
        grammar_score = 0.8  # Base score
        
        # Check for common grammar issues
        if content.count('  ') > 0:  # Double spaces
            grammar_score -= 0.1
        
        if content.count('..') > 0:  # Double periods
            grammar_score -= 0.1
        
        # Check sentence structure
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        if 5 <= avg_sentence_length <= 25:
            grammar_score += 0.1
        else:
            grammar_score -= 0.1
        
        return max(0.0, min(1.0, grammar_score))
    
    async def _analyze_readability(self, content: str) -> float:
        """Analizar legibilidad del contenido."""
        words = content.split()
        sentences = content.split('.')
        
        if len(words) == 0:
            return 0.0
        
        # Flesch Reading Ease approximation
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate readability score
        if avg_sentence_length <= 10 and avg_word_length <= 5:
            readability_score = 0.9
        elif avg_sentence_length <= 15 and avg_word_length <= 6:
            readability_score = 0.8
        elif avg_sentence_length <= 20 and avg_word_length <= 7:
            readability_score = 0.7
        else:
            readability_score = 0.6
        
        return readability_score
    
    async def _analyze_sentiment(self, content: str) -> float:
        """Analizar sentimiento del contenido."""
        positive_words = ["great", "amazing", "excellent", "wonderful", "fantastic", "awesome", "love", "like", "good", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "dislike", "horrible", "terrible"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        total_words = len(content.split())
        if total_words == 0:
            return 0.5
        
        # Calculate sentiment score
        sentiment_ratio = (positive_count - negative_count) / total_words
        sentiment_score = 0.5 + (sentiment_ratio * 10)  # Normalize to 0-1
        
        return max(0.0, min(1.0, sentiment_score))
    
    async def _analyze_engagement_potential(self, content: str) -> float:
        """Analizar potencial de engagement."""
        engagement_score = 0.5  # Base score
        
        # Questions increase engagement
        if '?' in content:
            engagement_score += 0.2
        
        # Exclamations increase engagement
        if '!' in content:
            engagement_score += 0.1
        
        # Hashtags increase engagement
        if '#' in content:
            engagement_score += 0.1
        
        # Mentions increase engagement
        if '@' in content:
            engagement_score += 0.1
        
        # Emojis increase engagement
        emoji_count = sum(1 for c in content if ord(c) > 127)
        if emoji_count > 0:
            engagement_score += min(0.1, emoji_count * 0.02)
        
        # Call to action increases engagement
        cta_words = ["click", "share", "comment", "like", "follow", "subscribe"]
        if any(word in content.lower() for word in cta_words):
            engagement_score += 0.1
        
        return min(1.0, engagement_score)
    
    async def _analyze_creativity(self, content: str) -> float:
        """Analizar creatividad del contenido."""
        creativity_score = 0.5  # Base score
        
        # Unique words increase creativity
        words = content.split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness_ratio = len(unique_words) / len(words)
            creativity_score += uniqueness_ratio * 0.2
        
        # Metaphors and creative language
        creative_indicators = ["like", "as if", "imagine", "picture", "visualize", "think of"]
        if any(indicator in content.lower() for indicator in creative_indicators):
            creativity_score += 0.1
        
        # Emotional language
        emotional_words = ["feel", "emotion", "heart", "soul", "passion", "excitement"]
        if any(word in content.lower() for word in emotional_words):
            creativity_score += 0.1
        
        # Storytelling elements
        story_indicators = ["once", "when", "then", "finally", "suddenly", "meanwhile"]
        if any(indicator in content.lower() for indicator in story_indicators):
            creativity_score += 0.1
        
        return min(1.0, creativity_score)
    
    async def _analyze_relevance(self, content: str, topic: str) -> float:
        """Analizar relevancia del contenido."""
        if not topic:
            return 0.7  # Default score if no topic provided
        
        # Simple keyword matching
        topic_words = set(topic.lower().split())
        content_words = set(content.lower().split())
        
        if len(topic_words) == 0:
            return 0.7
        
        # Calculate relevance based on keyword overlap
        overlap = len(topic_words.intersection(content_words))
        relevance_score = min(1.0, overlap / len(topic_words))
        
        return relevance_score
    
    def _calculate_enhancement_confidence(self, enhancements_applied: List[str], quality_improvement: float) -> float:
        """Calcular confianza en la mejora aplicada."""
        base_confidence = 0.7
        
        # More enhancements = higher confidence
        enhancement_bonus = min(0.2, len(enhancements_applied) * 0.05)
        
        # Quality improvement = higher confidence
        improvement_bonus = min(0.1, quality_improvement * 0.5)
        
        # Historical success = higher confidence
        if self.improvement_history:
            recent_improvements = [h["improvement"] for h in self.improvement_history[-10:]]
            avg_improvement = statistics.mean(recent_improvements)
            history_bonus = min(0.1, avg_improvement * 0.2)
        else:
            history_bonus = 0.0
        
        confidence = base_confidence + enhancement_bonus + improvement_bonus + history_bonus
        
        return min(1.0, confidence)
    
    async def process_feedback(self, post_id: str, feedback: Dict[str, Any]):
        """Procesar feedback para mejorar el sistema."""
        self.feedback_database.append({
            "post_id": post_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update learning patterns
        await self._update_learning_patterns(post_id, feedback)
    
    async def _update_learning_patterns(self, post_id: str, feedback: Dict[str, Any]):
        """Actualizar patrones de aprendizaje."""
        # Extract patterns from feedback
        patterns = self._extract_patterns_from_feedback(feedback)
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type not in self.learning_patterns:
                self.learning_patterns[pattern_type] = LearningPattern(
                    pattern_type=pattern_type,
                    success_rate=0.5,
                    avg_quality_improvement=0.0,
                    usage_count=0,
                    last_used=datetime.now(),
                    effectiveness_score=0.5
                )
            
            pattern = self.learning_patterns[pattern_type]
            
            # Update pattern statistics
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            # Update success rate
            success = feedback.get("success", False)
            if pattern.usage_count == 1:
                pattern.success_rate = 1.0 if success else 0.0
            else:
                current_success = pattern.success_rate * (pattern.usage_count - 1)
                pattern.success_rate = (current_success + (1 if success else 0)) / pattern.usage_count
            
            # Update quality improvement
            quality_improvement = feedback.get("quality_improvement", 0.0)
            if pattern.usage_count == 1:
                pattern.avg_quality_improvement = quality_improvement
            else:
                current_improvement = pattern.avg_quality_improvement * (pattern.usage_count - 1)
                pattern.avg_quality_improvement = (current_improvement + quality_improvement) / pattern.usage_count
            
            # Update effectiveness score
            pattern.effectiveness_score = (pattern.success_rate + pattern.avg_quality_improvement) / 2
    
    def _extract_patterns_from_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Extraer patrones del feedback."""
        patterns = {}
        
        # Content type patterns
        if "content_type" in feedback:
            patterns["content_type"] = feedback["content_type"]
        
        # Enhancement patterns
        if "enhancements_applied" in feedback:
            patterns["enhancement_combination"] = tuple(feedback["enhancements_applied"])
        
        # Quality improvement patterns
        if "quality_improvement" in feedback:
            improvement_level = "high" if feedback["quality_improvement"] > 0.1 else "low"
            patterns["improvement_level"] = improvement_level
        
        return patterns
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de mejora."""
        if not self.improvement_history:
            return {"error": "No improvement history available"}
        
        recent_improvements = self.improvement_history[-50:]  # Last 50 improvements
        
        return {
            "total_enhancements": len(self.improvement_history),
            "recent_enhancements": len(recent_improvements),
            "avg_improvement": statistics.mean([h["improvement"] for h in recent_improvements]),
            "success_rate": sum(1 for h in recent_improvements if h["improvement"] > 0) / len(recent_improvements),
            "most_common_enhancements": self._get_most_common_enhancements(),
            "learning_patterns": len(self.learning_patterns),
            "feedback_count": len(self.feedback_database)
        }
    
    def _get_most_common_enhancements(self) -> List[Tuple[str, int]]:
        """Obtener mejoras más comunes."""
        enhancement_counts = defaultdict(int)
        
        for history in self.improvement_history:
            for enhancement in history["enhancements"]:
                enhancement_counts[enhancement] += 1
        
        return sorted(enhancement_counts.items(), key=lambda x: x[1], reverse=True)[:5]

# ===== ENHANCEMENT STRATEGIES =====

class GrammarEnhancer:
    """Estrategia de mejora de gramática."""
    
    async def enhance(self, text: str) -> Dict[str, Any]:
        """Mejorar gramática del texto."""
        enhanced_text = text
        
        # Fix common grammar issues
        enhanced_text = enhanced_text.replace('  ', ' ')  # Remove double spaces
        enhanced_text = enhanced_text.replace('..', '.')  # Fix double periods
        
        # Capitalize first letter of sentences
        sentences = enhanced_text.split('. ')
        enhanced_sentences = []
        for sentence in sentences:
            if sentence:
                enhanced_sentences.append(sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper())
        
        enhanced_text = '. '.join(enhanced_sentences)
        
        return {
            "enhanced_text": enhanced_text,
            "improvements": ["grammar_correction", "capitalization"],
            "confidence": 0.8
        }

class ReadabilityEnhancer:
    """Estrategia de mejora de legibilidad."""
    
    async def enhance(self, text: str) -> Dict[str, Any]:
        """Mejorar legibilidad del texto."""
        enhanced_text = text
        
        # Break long sentences
        sentences = enhanced_text.split('. ')
        enhanced_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 25:  # Long sentence
                # Simple sentence breaking
                words = sentence.split()
                mid_point = len(words) // 2
                first_part = ' '.join(words[:mid_point])
                second_part = ' '.join(words[mid_point:])
                enhanced_sentences.extend([first_part, second_part])
            else:
                enhanced_sentences.append(sentence)
        
        enhanced_text = '. '.join(enhanced_sentences)
        
        return {
            "enhanced_text": enhanced_text,
            "improvements": ["sentence_breaking", "readability"],
            "confidence": 0.7
        }

class EngagementEnhancer:
    """Estrategia de mejora de engagement."""
    
    async def enhance(self, text: str) -> Dict[str, Any]:
        """Mejorar engagement del texto."""
        enhanced_text = text
        
        # Add questions if none present
        if '?' not in enhanced_text:
            enhanced_text += " What do you think? 🤔"
        
        # Add call to action if none present
        cta_words = ["click", "share", "comment", "like", "follow", "subscribe"]
        if not any(word in enhanced_text.lower() for word in cta_words):
            enhanced_text += " Share your thoughts below! 💬"
        
        # Add emojis for visual appeal
        if sum(1 for c in enhanced_text if ord(c) > 127) < 2:
            enhanced_text += " ✨"
        
        return {
            "enhanced_text": enhanced_text,
            "improvements": ["engagement_boost", "call_to_action", "visual_appeal"],
            "confidence": 0.75
        }

class CreativityEnhancer:
    """Estrategia de mejora de creatividad."""
    
    async def enhance(self, text: str) -> Dict[str, Any]:
        """Mejorar creatividad del texto."""
        enhanced_text = text
        
        # Add creative language
        creative_phrases = [
            "Imagine this: ",
            "Picture this: ",
            "Think about it: ",
            "Here's the thing: "
        ]
        
        # Add creative phrase if text doesn't start with one
        if not any(phrase.lower() in enhanced_text.lower() for phrase in creative_phrases):
            enhanced_text = "Here's the thing: " + enhanced_text
        
        # Add metaphorical language
        if "like" not in enhanced_text.lower() and "as" not in enhanced_text.lower():
            # Simple metaphor addition
            enhanced_text += " It's like unlocking a new level of possibilities! 🚀"
        
        return {
            "enhanced_text": enhanced_text,
            "improvements": ["creative_language", "metaphorical_elements"],
            "confidence": 0.7
        }

class SentimentEnhancer:
    """Estrategia de mejora de sentimiento."""
    
    async def enhance(self, text: str) -> Dict[str, Any]:
        """Mejorar sentimiento del texto."""
        enhanced_text = text
        
        # Add positive language
        positive_enhancers = [
            "amazing", "incredible", "fantastic", "wonderful", "excellent"
        ]
        
        # Check if text has positive sentiment
        positive_words = ["great", "amazing", "excellent", "wonderful", "fantastic", "awesome", "love", "like", "good", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "dislike", "horrible", "terrible"]
        
        content_lower = enhanced_text.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        # Add positive language if sentiment is neutral or negative
        if positive_count <= negative_count:
            enhanced_text = "This is absolutely " + enhanced_text
        
        # Add emotional appeal
        if "feel" not in enhanced_text.lower() and "emotion" not in enhanced_text.lower():
            enhanced_text += " It's truly inspiring! 💫"
        
        return {
            "enhanced_text": enhanced_text,
            "improvements": ["positive_sentiment", "emotional_appeal"],
            "confidence": 0.75
        }

# ===== CONTINUOUS LEARNING OPTIMIZER =====

class ContinuousLearningOptimizer:
    """Loop de aprendizaje continuo para mejora automática."""
    
    def __init__(self, quality_enhancer: AutoQualityEnhancer):
        
    """__init__ function."""
self.quality_enhancer = quality_enhancer
        self.learning_interval = 3600  # 1 hour
        self.optimization_history = []
    
    async def learning_loop(self) -> Any:
        """Loop principal de aprendizaje continuo."""
        while True:
            try:
                print("🔄 Starting continuous learning cycle...")
                
                # Collect feedback
                feedback = await self._collect_user_feedback()
                
                # Analyze patterns
                patterns = await self._analyze_success_patterns(feedback)
                
                # Update models
                await self._update_models(patterns)
                
                # Optimize strategies
                await self._optimize_strategies(patterns)
                
                # Store optimization history
                self.optimization_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "patterns_analyzed": len(patterns),
                    "models_updated": True,
                    "strategies_optimized": True
                })
                
                print(f"✅ Learning cycle completed. Patterns analyzed: {len(patterns)}")
                
            except Exception as e:
                print(f"❌ Error in learning cycle: {e}")
            
            # Wait for next cycle
            await asyncio.sleep(self.learning_interval)
    
    async def _collect_user_feedback(self) -> List[Dict[str, Any]]:
        """Recolectar feedback de usuarios."""
        # In a real implementation, this would collect feedback from various sources
        return self.quality_enhancer.feedback_database[-100:]  # Last 100 feedback entries
    
    async def _analyze_success_patterns(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar patrones de éxito."""
        patterns = {
            "high_quality_indicators": [],
            "successful_enhancements": [],
            "user_preferences": [],
            "performance_metrics": {}
        }
        
        if not feedback:
            return patterns
        
        # Analyze high-quality indicators
        successful_posts = [f for f in feedback if f.get("feedback", {}).get("success", False)]
        if successful_posts:
            patterns["high_quality_indicators"] = self._extract_quality_indicators(successful_posts)
        
        # Analyze successful enhancements
        patterns["successful_enhancements"] = self._extract_successful_enhancements(feedback)
        
        # Analyze user preferences
        patterns["user_preferences"] = self._extract_user_preferences(feedback)
        
        # Calculate performance metrics
        patterns["performance_metrics"] = self._calculate_performance_metrics(feedback)
        
        return patterns
    
    def _extract_quality_indicators(self, successful_posts: List[Dict[str, Any]]) -> List[str]:
        """Extraer indicadores de calidad de posts exitosos."""
        indicators = []
        
        for post in successful_posts:
            content = post.get("feedback", {}).get("content", "")
            
            # Check for quality indicators
            if len(content) > 100:
                indicators.append("longer_content")
            
            if '?' in content:
                indicators.append("interactive_content")
            
            if '#' in content:
                indicators.append("hashtag_usage")
            
            if sum(1 for c in content if ord(c) > 127) > 2:
                indicators.append("emoji_usage")
        
        # Return most common indicators
        counter = Counter(indicators)
        return [indicator for indicator, count in counter.most_common(5)]
    
    def _extract_successful_enhancements(self, feedback: List[Dict[str, Any]]) -> List[str]:
        """Extraer mejoras exitosas."""
        successful_enhancements = []
        
        for post in feedback:
            if post.get("feedback", {}).get("success", False):
                enhancements = post.get("feedback", {}).get("enhancements_applied", [])
                successful_enhancements.extend(enhancements)
        
        # Return most common successful enhancements
        counter = Counter(successful_enhancements)
        return [enhancement for enhancement, count in counter.most_common(3)]
    
    def _extract_user_preferences(self, feedback: List[Dict[str, Any]]) -> List[str]:
        """Extraer preferencias de usuarios."""
        preferences = []
        
        for post in feedback:
            user_feedback = post.get("feedback", {})
            
            if user_feedback.get("liked_style"):
                preferences.append("style_preference")
            
            if user_feedback.get("liked_tone"):
                preferences.append("tone_preference")
            
            if user_feedback.get("liked_length"):
                preferences.append("length_preference")
        
        return list(set(preferences))
    
    def _calculate_performance_metrics(self, feedback: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcular métricas de performance."""
        if not feedback:
            return {}
        
        success_rate = sum(1 for f in feedback if f.get("feedback", {}).get("success", False)) / len(feedback)
        
        quality_improvements = [f.get("feedback", {}).get("quality_improvement", 0) for f in feedback]
        avg_improvement = statistics.mean(quality_improvements) if quality_improvements else 0
        
        return {
            "success_rate": success_rate,
            "avg_quality_improvement": avg_improvement,
            "feedback_count": len(feedback)
        }
    
    async def _update_models(self, patterns: Dict[str, Any]):
        """Actualizar modelos basado en patrones."""
        # Update enhancement strategies based on successful patterns
        successful_enhancements = patterns.get("successful_enhancements", [])
        
        for enhancement in successful_enhancements:
            if enhancement in self.quality_enhancer.enhancement_strategies:
                # Boost confidence for successful enhancements
                strategy = self.quality_enhancer.enhancement_strategies[enhancement]
                if hasattr(strategy, 'confidence'):
                    strategy.confidence = min(1.0, strategy.confidence + 0.1)
    
    async def _optimize_strategies(self, patterns: Dict[str, Any]):
        """Optimizar estrategias basado en patrones."""
        # Adjust quality thresholds based on user feedback
        performance_metrics = patterns.get("performance_metrics", {})
        success_rate = performance_metrics.get("success_rate", 0.5)
        
        if success_rate > 0.8:
            # High success rate - can be more selective
            self.quality_enhancer.quality_thresholds["good"] = min(0.85, self.quality_enhancer.quality_thresholds["good"] + 0.02)
        elif success_rate < 0.6:
            # Low success rate - be more lenient
            self.quality_enhancer.quality_thresholds["good"] = max(0.7, self.quality_enhancer.quality_thresholds["good"] - 0.02)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de aprendizaje."""
        return {
            "learning_cycles": len(self.optimization_history),
            "last_cycle": self.optimization_history[-1]["timestamp"] if self.optimization_history else None,
            "patterns_analyzed_total": sum(h["patterns_analyzed"] for h in self.optimization_history),
            "models_updated_count": sum(1 for h in self.optimization_history if h["models_updated"]),
            "strategies_optimized_count": sum(1 for h in self.optimization_history if h["strategies_optimized"])
        }

# ===== EXPORTS =====

__all__ = [
    "AutoQualityEnhancer",
    "ContinuousLearningOptimizer",
    "QualityMetrics",
    "EnhancementResult",
    "LearningPattern",
    "GrammarEnhancer",
    "ReadabilityEnhancer",
    "EngagementEnhancer",
    "CreativityEnhancer",
    "SentimentEnhancer"
] 