"""
Advanced AI Enhancement System for Facebook Posts
Following functional programming principles and AI best practices
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
import hashlib
from collections import Counter

logger = logging.getLogger(__name__)


# Pure functions for AI enhancement

class ContentQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class OptimizationStrategy(str, Enum):
    ENGAGEMENT = "engagement"
    READABILITY = "readability"
    VIRAL_POTENTIAL = "viral_potential"
    SENTIMENT = "sentiment"
    HASHTAG_OPTIMIZATION = "hashtag_optimization"
    EMOJI_OPTIMIZATION = "emoji_optimization"


@dataclass(frozen=True)
class ContentAnalysis:
    """Immutable content analysis - pure data structure"""
    content: str
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int
    readability_score: float
    engagement_score: float
    sentiment_score: float
    viral_potential: float
    quality_rating: ContentQuality
    hashtags: List[str]
    emojis: List[str]
    keywords: List[str]
    issues: List[str]
    suggestions: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "content": self.content,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "readability_score": self.readability_score,
            "engagement_score": self.engagement_score,
            "sentiment_score": self.sentiment_score,
            "viral_potential": self.viral_potential,
            "quality_rating": self.quality_rating.value,
            "hashtags": self.hashtags,
            "emojis": self.emojis,
            "keywords": self.keywords,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable optimization result - pure data structure"""
    original_content: str
    optimized_content: str
    strategy: OptimizationStrategy
    improvements: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement_percentage: float
    confidence_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "original_content": self.original_content,
            "optimized_content": self.optimized_content,
            "strategy": self.strategy.value,
            "improvements": self.improvements,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "improvement_percentage": self.improvement_percentage,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat()
        }


def analyze_content_structure(content: str) -> Dict[str, int]:
    """Analyze content structure - pure function"""
    # Word count
    words = content.split()
    word_count = len(words)
    
    # Character count
    character_count = len(content)
    
    # Sentence count (simple heuristic)
    sentences = re.split(r'[.!?]+', content)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Paragraph count
    paragraphs = content.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    
    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count
    }


def calculate_readability_score(content: str) -> float:
    """Calculate readability score - pure function"""
    structure = analyze_content_structure(content)
    
    if structure["sentence_count"] == 0 or structure["word_count"] == 0:
        return 0.0
    
    # Simple Flesch Reading Ease approximation
    avg_sentence_length = structure["word_count"] / structure["sentence_count"]
    avg_syllables_per_word = estimate_syllables_per_word(content)
    
    # Flesch Reading Ease formula (simplified)
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, score / 100.0))


def estimate_syllables_per_word(content: str) -> float:
    """Estimate syllables per word - pure function"""
    words = content.lower().split()
    if not words:
        return 0.0
    
    total_syllables = 0
    for word in words:
        # Simple syllable counting heuristic
        word = re.sub(r'[^a-z]', '', word)
        if not word:
            continue
        
        # Count vowel groups
        vowel_groups = len(re.findall(r'[aeiouy]+', word))
        total_syllables += max(1, vowel_groups)
    
    return total_syllables / len(words)


def extract_hashtags_advanced(content: str) -> List[str]:
    """Extract hashtags with advanced analysis - pure function"""
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, content)
    
    # Remove duplicates and sort by frequency
    hashtag_counts = Counter(hashtags)
    return [tag for tag, count in hashtag_counts.most_common()]


def extract_emojis_advanced(content: str) -> List[str]:
    """Extract emojis with advanced analysis - pure function"""
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
    emojis = re.findall(emoji_pattern, content)
    
    # Remove duplicates and sort by frequency
    emoji_counts = Counter(emojis)
    return [emoji for emoji, count in emoji_counts.most_common()]


def extract_keywords(content: str, min_length: int = 3) -> List[str]:
    """Extract keywords - pure function"""
    # Remove special characters and convert to lowercase
    clean_content = re.sub(r'[^\w\s]', ' ', content.lower())
    words = clean_content.split()
    
    # Filter by length and common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    keywords = [
        word for word in words
        if len(word) >= min_length and word not in common_words
    ]
    
    # Count frequency and return most common
    keyword_counts = Counter(keywords)
    return [word for word, count in keyword_counts.most_common(10)]


def calculate_engagement_score(content: str, hashtags: List[str], emojis: List[str]) -> float:
    """Calculate engagement score - pure function"""
    base_score = 0.5
    
    # Length factor (optimal range: 100-280 characters)
    char_count = len(content)
    if 100 <= char_count <= 280:
        base_score += 0.2
    elif 50 <= char_count < 100:
        base_score += 0.1
    elif char_count > 280:
        base_score += 0.05
    
    # Hashtag factor (optimal: 1-3 hashtags)
    hashtag_count = len(hashtags)
    if 1 <= hashtag_count <= 3:
        base_score += 0.2
    elif 4 <= hashtag_count <= 5:
        base_score += 0.1
    elif hashtag_count > 5:
        base_score -= 0.1
    
    # Emoji factor (optimal: 1-2 emojis)
    emoji_count = len(emojis)
    if 1 <= emoji_count <= 2:
        base_score += 0.1
    elif 3 <= emoji_count <= 4:
        base_score += 0.05
    elif emoji_count > 4:
        base_score -= 0.05
    
    # Question factor (questions increase engagement)
    if '?' in content:
        base_score += 0.1
    
    # Call-to-action factor
    cta_words = ['click', 'learn', 'discover', 'explore', 'join', 'follow', 'share', 'comment', 'like']
    if any(word in content.lower() for word in cta_words):
        base_score += 0.1
    
    return max(0.0, min(1.0, base_score))


def calculate_sentiment_score(content: str) -> float:
    """Calculate sentiment score - pure function"""
    positive_words = {
        'great', 'amazing', 'wonderful', 'excellent', 'fantastic', 'love', 'best', 'awesome',
        'incredible', 'outstanding', 'brilliant', 'perfect', 'superb', 'marvelous', 'fabulous',
        'terrific', 'magnificent', 'exceptional', 'remarkable', 'impressive', 'stunning'
    }
    
    negative_words = {
        'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'frustrating',
        'annoying', 'boring', 'stupid', 'ridiculous', 'pathetic', 'useless', 'waste',
        'disgusting', 'revolting', 'appalling', 'shocking', 'outrageous', 'unacceptable'
    }
    
    content_lower = content.lower()
    words = content_lower.split()
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words == 0:
        return 0.0
    
    # Normalize to -1 to 1 range
    sentiment = (positive_count - negative_count) / total_sentiment_words
    return max(-1.0, min(1.0, sentiment))


def calculate_viral_potential(content: str, hashtags: List[str], emojis: List[str]) -> float:
    """Calculate viral potential - pure function"""
    base_score = 0.3
    
    # Engagement factors
    engagement_score = calculate_engagement_score(content, hashtags, emojis)
    base_score += engagement_score * 0.4
    
    # Emotional impact
    sentiment_score = abs(calculate_sentiment_score(content))
    base_score += sentiment_score * 0.2
    
    # Controversy factor (moderate controversy increases viral potential)
    controversial_words = {'controversial', 'debate', 'opinion', 'think', 'believe', 'argue'}
    if any(word in content.lower() for word in controversial_words):
        base_score += 0.1
    
    # Trend factor (hashtags indicate trending topics)
    if hashtags:
        base_score += 0.1
    
    # Shareability factor
    share_words = ['share', 'spread', 'tell', 'friends', 'everyone', 'world']
    if any(word in content.lower() for word in share_words):
        base_score += 0.1
    
    return max(0.0, min(1.0, base_score))


def identify_content_issues(content: str) -> List[str]:
    """Identify content issues - pure function"""
    issues = []
    
    # Length issues
    if len(content) < 10:
        issues.append("Content too short")
    elif len(content) > 2000:
        issues.append("Content too long")
    
    # Readability issues
    readability = calculate_readability_score(content)
    if readability < 0.3:
        issues.append("Low readability score")
    
    # Hashtag issues
    hashtags = extract_hashtags_advanced(content)
    if len(hashtags) > 10:
        issues.append("Too many hashtags")
    elif len(hashtags) == 0:
        issues.append("No hashtags")
    
    # Emoji issues
    emojis = extract_emojis_advanced(content)
    if len(emojis) > 5:
        issues.append("Too many emojis")
    
    # Grammar issues (simple checks)
    if content.count('!') > 3:
        issues.append("Too many exclamation marks")
    
    if content.count('?') > 3:
        issues.append("Too many question marks")
    
    # Spacing issues
    if '  ' in content:
        issues.append("Multiple spaces detected")
    
    return issues


def generate_content_suggestions(content: str, analysis: ContentAnalysis) -> List[str]:
    """Generate content suggestions - pure function"""
    suggestions = []
    
    # Length suggestions
    if len(content) < 50:
        suggestions.append("Consider adding more details to make the post more engaging")
    elif len(content) > 500:
        suggestions.append("Consider shortening the post for better readability")
    
    # Hashtag suggestions
    if len(analysis.hashtags) == 0:
        suggestions.append("Add relevant hashtags to increase discoverability")
    elif len(analysis.hashtags) > 5:
        suggestions.append("Reduce hashtags to 3-5 for better engagement")
    
    # Emoji suggestions
    if len(analysis.emojis) == 0:
        suggestions.append("Add 1-2 relevant emojis to make the post more engaging")
    elif len(analysis.emojis) > 3:
        suggestions.append("Reduce emojis to 1-2 for better readability")
    
    # Readability suggestions
    if analysis.readability_score < 0.5:
        suggestions.append("Simplify sentence structure for better readability")
    
    # Engagement suggestions
    if analysis.engagement_score < 0.6:
        suggestions.append("Add a call-to-action or question to increase engagement")
    
    # Sentiment suggestions
    if abs(analysis.sentiment_score) > 0.8:
        suggestions.append("Consider balancing the emotional tone for broader appeal")
    
    return suggestions


def create_content_analysis(
    content: str,
    hashtags: Optional[List[str]] = None,
    emojis: Optional[List[str]] = None
) -> ContentAnalysis:
    """Create comprehensive content analysis - pure function"""
    # Extract elements if not provided
    if hashtags is None:
        hashtags = extract_hashtags_advanced(content)
    if emojis is None:
        emojis = extract_emojis_advanced(content)
    
    # Analyze structure
    structure = analyze_content_structure(content)
    
    # Calculate scores
    readability_score = calculate_readability_score(content)
    engagement_score = calculate_engagement_score(content, hashtags, emojis)
    sentiment_score = calculate_sentiment_score(content)
    viral_potential = calculate_viral_potential(content, hashtags, emojis)
    
    # Determine quality rating
    overall_score = (readability_score + engagement_score + viral_potential) / 3
    if overall_score >= 0.8:
        quality_rating = ContentQuality.EXCELLENT
    elif overall_score >= 0.6:
        quality_rating = ContentQuality.GOOD
    elif overall_score >= 0.4:
        quality_rating = ContentQuality.FAIR
    else:
        quality_rating = ContentQuality.POOR
    
    # Identify issues and generate suggestions
    issues = identify_content_issues(content)
    keywords = extract_keywords(content)
    
    # Create analysis
    analysis = ContentAnalysis(
        content=content,
        word_count=structure["word_count"],
        character_count=structure["character_count"],
        sentence_count=structure["sentence_count"],
        paragraph_count=structure["paragraph_count"],
        readability_score=readability_score,
        engagement_score=engagement_score,
        sentiment_score=sentiment_score,
        viral_potential=viral_potential,
        quality_rating=quality_rating,
        hashtags=hashtags,
        emojis=emojis,
        keywords=keywords,
        issues=issues,
        suggestions=[],
        timestamp=datetime.utcnow()
    )
    
    # Generate suggestions based on analysis
    suggestions = generate_content_suggestions(content, analysis)
    
    # Return updated analysis with suggestions
    return ContentAnalysis(
        content=analysis.content,
        word_count=analysis.word_count,
        character_count=analysis.character_count,
        sentence_count=analysis.sentence_count,
        paragraph_count=analysis.paragraph_count,
        readability_score=analysis.readability_score,
        engagement_score=analysis.engagement_score,
        sentiment_score=analysis.sentiment_score,
        viral_potential=analysis.viral_potential,
        quality_rating=analysis.quality_rating,
        hashtags=analysis.hashtags,
        emojis=analysis.emojis,
        keywords=analysis.keywords,
        issues=analysis.issues,
        suggestions=suggestions,
        timestamp=analysis.timestamp
    )


# Advanced AI Enhancement System Class

class AdvancedAIEnhancer:
    """Advanced AI Enhancement System following functional principles"""
    
    def __init__(self):
        self.optimization_strategies = {
            OptimizationStrategy.ENGAGEMENT: self._optimize_for_engagement,
            OptimizationStrategy.READABILITY: self._optimize_for_readability,
            OptimizationStrategy.VIRAL_POTENTIAL: self._optimize_for_viral_potential,
            OptimizationStrategy.SENTIMENT: self._optimize_for_sentiment,
            OptimizationStrategy.HASHTAG_OPTIMIZATION: self._optimize_hashtags,
            OptimizationStrategy.EMOJI_OPTIMIZATION: self._optimize_emojis
        }
        
        self.optimization_history: List[OptimizationResult] = []
    
    async def analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content comprehensively"""
        try:
            analysis = create_content_analysis(content)
            logger.info("Content analysis completed", content_length=len(content))
            return analysis
        except Exception as e:
            logger.error("Error analyzing content", error=str(e))
            raise
    
    async def optimize_content(
        self,
        content: str,
        strategy: OptimizationStrategy,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """Optimize content using specified strategy"""
        try:
            # Analyze original content
            original_analysis = await self.analyze_content(content)
            original_metrics = {
                "readability": original_analysis.readability_score,
                "engagement": original_analysis.engagement_score,
                "sentiment": original_analysis.sentiment_score,
                "viral_potential": original_analysis.viral_potential
            }
            
            # Apply optimization strategy
            optimization_func = self.optimization_strategies.get(strategy)
            if not optimization_func:
                raise ValueError(f"Unknown optimization strategy: {strategy}")
            
            optimized_content, improvements = await optimization_func(content, original_analysis, target_metrics)
            
            # Analyze optimized content
            optimized_analysis = await self.analyze_content(optimized_content)
            optimized_metrics = {
                "readability": optimized_analysis.readability_score,
                "engagement": optimized_analysis.engagement_score,
                "sentiment": optimized_analysis.sentiment_score,
                "viral_potential": optimized_analysis.viral_potential
            }
            
            # Calculate improvement percentage
            improvement_percentage = self._calculate_improvement_percentage(original_metrics, optimized_metrics, strategy)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(original_analysis, optimized_analysis)
            
            # Create optimization result
            result = OptimizationResult(
                original_content=content,
                optimized_content=optimized_content,
                strategy=strategy,
                improvements=improvements,
                metrics_before=original_metrics,
                metrics_after=optimized_metrics,
                improvement_percentage=improvement_percentage,
                confidence_score=confidence_score,
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self.optimization_history.append(result)
            
            logger.info("Content optimization completed", strategy=strategy.value, improvement=improvement_percentage)
            return result
            
        except Exception as e:
            logger.error("Error optimizing content", error=str(e), strategy=strategy.value)
            raise
    
    async def _optimize_for_engagement(
        self,
        content: str,
        analysis: ContentAnalysis,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[str, List[str]]:
        """Optimize content for engagement"""
        improvements = []
        optimized_content = content
        
        # Add call-to-action if missing
        if not any(word in content.lower() for word in ['click', 'learn', 'discover', 'explore', 'join', 'follow', 'share']):
            optimized_content += " What do you think? Share your thoughts below! 👇"
            improvements.append("Added call-to-action")
        
        # Optimize hashtags
        if len(analysis.hashtags) < 3:
            optimized_content += " #engagement #community #discussion"
            improvements.append("Added engagement hashtags")
        
        # Add emojis if missing
        if len(analysis.emojis) < 2:
            optimized_content += " 🚀💡"
            improvements.append("Added engaging emojis")
        
        # Add question if missing
        if '?' not in content:
            optimized_content += " What's your experience with this?"
            improvements.append("Added engaging question")
        
        return optimized_content, improvements
    
    async def _optimize_for_readability(
        self,
        content: str,
        analysis: ContentAnalysis,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[str, List[str]]:
        """Optimize content for readability"""
        improvements = []
        optimized_content = content
        
        # Break long sentences
        sentences = content.split('. ')
        if len(sentences) > 1:
            # Simple sentence shortening
            optimized_sentences = []
            for sentence in sentences:
                if len(sentence) > 100:
                    # Split long sentences
                    parts = sentence.split(', ')
                    if len(parts) > 1:
                        optimized_sentences.extend(parts)
                    else:
                        optimized_sentences.append(sentence)
                else:
                    optimized_sentences.append(sentence)
            
            optimized_content = '. '.join(optimized_sentences)
            improvements.append("Shortened long sentences")
        
        # Add paragraph breaks
        if len(content) > 200 and '\n\n' not in content:
            # Find natural break points
            words = content.split()
            if len(words) > 50:
                mid_point = len(words) // 2
                optimized_content = ' '.join(words[:mid_point]) + '\n\n' + ' '.join(words[mid_point:])
                improvements.append("Added paragraph breaks")
        
        return optimized_content, improvements
    
    async def _optimize_for_viral_potential(
        self,
        content: str,
        analysis: ContentAnalysis,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[str, List[str]]:
        """Optimize content for viral potential"""
        improvements = []
        optimized_content = content
        
        # Add trending hashtags
        trending_hashtags = ["#viral", "#trending", "#mustsee", "#share"]
        existing_hashtags = extract_hashtags_advanced(content)
        new_hashtags = [tag for tag in trending_hashtags if tag not in existing_hashtags]
        
        if new_hashtags:
            optimized_content += " " + " ".join(new_hashtags[:2])
            improvements.append("Added trending hashtags")
        
        # Add emotional triggers
        emotional_triggers = ["amazing", "incredible", "unbelievable", "mind-blowing"]
        if not any(trigger in content.lower() for trigger in emotional_triggers):
            optimized_content = f"🚀 {optimized_content}"
            improvements.append("Added emotional trigger")
        
        # Add share encouragement
        if "share" not in content.lower():
            optimized_content += " Don't forget to share! 🔄"
            improvements.append("Added share encouragement")
        
        return optimized_content, improvements
    
    async def _optimize_for_sentiment(
        self,
        content: str,
        analysis: ContentAnalysis,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[str, List[str]]:
        """Optimize content for sentiment"""
        improvements = []
        optimized_content = content
        
        # Balance sentiment if too extreme
        if analysis.sentiment_score > 0.8:
            # Too positive, add some balance
            optimized_content = optimized_content.replace("amazing", "interesting")
            optimized_content = optimized_content.replace("incredible", "notable")
            improvements.append("Balanced overly positive sentiment")
        
        elif analysis.sentiment_score < -0.8:
            # Too negative, add some positivity
            optimized_content = optimized_content.replace("terrible", "challenging")
            optimized_content = optimized_content.replace("awful", "difficult")
            improvements.append("Balanced overly negative sentiment")
        
        # Add neutral sentiment words if needed
        if abs(analysis.sentiment_score) < 0.2:
            optimized_content = f"Here's an interesting perspective: {optimized_content}"
            improvements.append("Added neutral sentiment framing")
        
        return optimized_content, improvements
    
    async def _optimize_hashtags(
        self,
        content: str,
        analysis: ContentAnalysis,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[str, List[str]]:
        """Optimize hashtags"""
        improvements = []
        optimized_content = content
        
        # Remove excessive hashtags
        hashtags = extract_hashtags_advanced(content)
        if len(hashtags) > 5:
            # Keep only the most relevant hashtags
            relevant_hashtags = hashtags[:5]
            # Remove all hashtags and add back the relevant ones
            content_without_hashtags = re.sub(r'#\w+', '', content).strip()
            optimized_content = content_without_hashtags + " " + " ".join(relevant_hashtags)
            improvements.append("Reduced excessive hashtags")
        
        # Add relevant hashtags if missing
        elif len(hashtags) < 2:
            relevant_hashtags = ["#content", "#socialmedia", "#engagement"]
            optimized_content += " " + " ".join(relevant_hashtags)
            improvements.append("Added relevant hashtags")
        
        return optimized_content, improvements
    
    async def _optimize_emojis(
        self,
        content: str,
        analysis: ContentAnalysis,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[str, List[str]]:
        """Optimize emojis"""
        improvements = []
        optimized_content = content
        
        # Remove excessive emojis
        emojis = extract_emojis_advanced(content)
        if len(emojis) > 3:
            # Keep only the first few emojis
            content_without_emojis = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', content).strip()
            optimized_content = content_without_emojis + " " + " ".join(emojis[:3])
            improvements.append("Reduced excessive emojis")
        
        # Add relevant emojis if missing
        elif len(emojis) == 0:
            optimized_content += " 🚀💡"
            improvements.append("Added relevant emojis")
        
        return optimized_content, improvements
    
    def _calculate_improvement_percentage(
        self,
        original_metrics: Dict[str, float],
        optimized_metrics: Dict[str, float],
        strategy: OptimizationStrategy
    ) -> float:
        """Calculate improvement percentage - pure function"""
        if strategy == OptimizationStrategy.ENGAGEMENT:
            key_metric = "engagement"
        elif strategy == OptimizationStrategy.READABILITY:
            key_metric = "readability"
        elif strategy == OptimizationStrategy.VIRAL_POTENTIAL:
            key_metric = "viral_potential"
        elif strategy == OptimizationStrategy.SENTIMENT:
            key_metric = "sentiment"
        else:
            # For hashtag and emoji optimization, use engagement
            key_metric = "engagement"
        
        original_value = original_metrics.get(key_metric, 0.0)
        optimized_value = optimized_metrics.get(key_metric, 0.0)
        
        if original_value == 0:
            return 0.0
        
        return ((optimized_value - original_value) / original_value) * 100
    
    def _calculate_confidence_score(
        self,
        original_analysis: ContentAnalysis,
        optimized_analysis: ContentAnalysis
    ) -> float:
        """Calculate confidence score - pure function"""
        # Base confidence on improvement in key metrics
        engagement_improvement = optimized_analysis.engagement_score - original_analysis.engagement_score
        readability_improvement = optimized_analysis.readability_score - original_analysis.readability_score
        viral_improvement = optimized_analysis.viral_potential - original_analysis.viral_potential
        
        # Calculate overall improvement
        overall_improvement = (engagement_improvement + readability_improvement + viral_improvement) / 3
        
        # Convert to confidence score (0-1)
        confidence = max(0.0, min(1.0, 0.5 + overall_improvement))
        
        return confidence
    
    def get_optimization_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return [result.to_dict() for result in self.optimization_history[-limit:]]
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        # Calculate statistics
        total_optimizations = len(self.optimization_history)
        average_improvement = sum(result.improvement_percentage for result in self.optimization_history) / total_optimizations
        average_confidence = sum(result.confidence_score for result in self.optimization_history) / total_optimizations
        
        # Strategy distribution
        strategy_counts = {}
        for result in self.optimization_history:
            strategy = result.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_optimizations": total_optimizations,
            "average_improvement_percentage": average_improvement,
            "average_confidence_score": average_confidence,
            "strategy_distribution": strategy_counts,
            "most_used_strategy": max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else None
        }


# Factory functions

def create_ai_enhancer() -> AdvancedAIEnhancer:
    """Create AI enhancer instance - pure function"""
    return AdvancedAIEnhancer()


async def get_ai_enhancer() -> AdvancedAIEnhancer:
    """Get AI enhancer instance"""
    return create_ai_enhancer()

