"""
Advanced Content Analysis and Optimization
==========================================

This module provides advanced content analysis, sentiment analysis,
readability scoring, and content optimization features.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import math

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ContentMetrics:
    """Comprehensive content metrics"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    avg_word_length: float
    readability_score: float
    sentiment_score: float
    keyword_density: Dict[str, float]
    topic_relevance: float
    engagement_score: float
    seo_score: float
    overall_quality: float

class ContentAnalyzer:
    """Advanced content analysis and optimization"""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'el', 'la', 'de', 'que', 'y', 'en',
            'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son'
        }
    
    async def analyze_content(self, content: str, title: str = "") -> ContentMetrics:
        """Perform comprehensive content analysis"""
        try:
            # Basic metrics
            word_count = len(content.split())
            sentence_count = self._count_sentences(content)
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
            # Calculate averages
            avg_sentence_length = word_count / max(sentence_count, 1)
            avg_word_length = sum(len(word) for word in content.split()) / max(word_count, 1)
            
            # Readability score
            readability_score = self._calculate_readability(content)
            
            # Sentiment analysis
            sentiment_score = self._analyze_sentiment(content)
            
            # Keyword analysis
            keyword_density = self._analyze_keywords(content)
            
            # Topic relevance
            topic_relevance = self._calculate_topic_relevance(content, title)
            
            # Engagement score
            engagement_score = self._calculate_engagement_score(content)
            
            # SEO score
            seo_score = self._calculate_seo_score(content, title)
            
            # Overall quality
            overall_quality = self._calculate_overall_quality(
                readability_score, sentiment_score, topic_relevance, 
                engagement_score, seo_score
            )
            
            return ContentMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                avg_sentence_length=avg_sentence_length,
                avg_word_length=avg_word_length,
                readability_score=readability_score,
                sentiment_score=sentiment_score,
                keyword_density=keyword_density,
                topic_relevance=topic_relevance,
                engagement_score=engagement_score,
                seo_score=seo_score,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return ContentMetrics(
                word_count=0, sentence_count=0, paragraph_count=0,
                avg_sentence_length=0, avg_word_length=0,
                readability_score=0.5, sentiment_score=0.5,
                keyword_density={}, topic_relevance=0.5,
                engagement_score=0.5, seo_score=0.5, overall_quality=0.5
            )
    
    def _count_sentences(self, content: str) -> int:
        """Count sentences in content"""
        sentences = re.split(r'[.!?]+', content)
        return len([s for s in sentences if s.strip()])
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate Flesch Reading Ease score"""
        try:
            words = content.split()
            sentences = self._count_sentences(content)
            syllables = sum(self._count_syllables(word) for word in words)
            
            if sentences == 0 or words == 0:
                return 0.5
            
            # Flesch Reading Ease formula
            score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            
            # Normalize to 0-1 scale
            return max(0, min(1, (score + 100) / 200))
            
        except Exception:
            return 0.5
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
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
    
    def _analyze_sentiment(self, content: str) -> float:
        """Simple sentiment analysis"""
        try:
            positive_words = {
                'excellent', 'amazing', 'great', 'wonderful', 'fantastic',
                'outstanding', 'brilliant', 'perfect', 'superb', 'incredible',
                'excelente', 'increíble', 'fantástico', 'maravilloso', 'perfecto'
            }
            
            negative_words = {
                'terrible', 'awful', 'horrible', 'bad', 'worst', 'disappointing',
                'poor', 'lousy', 'pathetic', 'disgusting', 'terrible', 'awful',
                'terrible', 'horrible', 'malo', 'pésimo', 'terrible', 'horrible'
            }
            
            words = content.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                return 0.5
            
            sentiment = (positive_count - negative_count) / total_sentiment_words
            return (sentiment + 1) / 2  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def _analyze_keywords(self, content: str) -> Dict[str, float]:
        """Analyze keyword density"""
        try:
            words = [word.lower().strip('.,!?;:"()[]{}') for word in content.split()]
            words = [word for word in words if word and word not in self.stop_words]
            
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            total_words = len(words)
            keyword_density = {}
            
            for word, freq in word_freq.items():
                if len(word) > 3 and freq > 1:  # Only significant words
                    density = freq / total_words
                    keyword_density[word] = density
            
            # Return top 10 keywords
            sorted_keywords = sorted(keyword_density.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_keywords[:10])
            
        except Exception:
            return {}
    
    def _calculate_topic_relevance(self, content: str, title: str) -> float:
        """Calculate topic relevance based on title-content alignment"""
        try:
            if not title:
                return 0.5
            
            title_words = set(title.lower().split())
            content_words = set(content.lower().split())
            
            # Remove stop words
            title_words = title_words - self.stop_words
            content_words = content_words - self.stop_words
            
            if not title_words:
                return 0.5
            
            overlap = len(title_words.intersection(content_words))
            relevance = overlap / len(title_words)
            
            return min(1.0, relevance)
            
        except Exception:
            return 0.5
    
    def _calculate_engagement_score(self, content: str) -> float:
        """Calculate engagement score based on content structure"""
        try:
            score = 0.0
            
            # Check for questions (encourage interaction)
            questions = content.count('?')
            if questions > 0:
                score += 0.2
            
            # Check for lists or bullet points
            if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):
                score += 0.2
            
            # Check for numbers (data-driven content)
            numbers = len(re.findall(r'\d+', content))
            if numbers > 3:
                score += 0.2
            
            # Check for examples or case studies
            example_indicators = ['for example', 'for instance', 'such as', 'case study', 'ejemplo', 'por ejemplo']
            if any(indicator in content.lower() for indicator in example_indicators):
                score += 0.2
            
            # Check for actionable content
            action_words = ['how to', 'steps', 'guide', 'tutorial', 'como', 'guía', 'pasos']
            if any(word in content.lower() for word in action_words):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _calculate_seo_score(self, content: str, title: str) -> float:
        """Calculate SEO score"""
        try:
            score = 0.0
            
            # Title length (optimal: 50-60 characters)
            if title:
                title_length = len(title)
                if 50 <= title_length <= 60:
                    score += 0.3
                elif 40 <= title_length <= 70:
                    score += 0.2
                else:
                    score += 0.1
            
            # Content length (optimal: 300+ words)
            word_count = len(content.split())
            if word_count >= 300:
                score += 0.3
            elif word_count >= 150:
                score += 0.2
            else:
                score += 0.1
            
            # Headings structure
            headings = len(re.findall(r'^#+\s', content, re.MULTILINE))
            if headings >= 2:
                score += 0.2
            elif headings >= 1:
                score += 0.1
            
            # Internal linking potential (mentions of other topics)
            topic_mentions = len(re.findall(r'\b(also|additionally|furthermore|moreover|however|therefore)\b', content, re.IGNORECASE))
            if topic_mentions >= 2:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _calculate_overall_quality(self, readability: float, sentiment: float, 
                                 topic_relevance: float, engagement: float, seo: float) -> float:
        """Calculate overall content quality score"""
        try:
            # Weighted average with emphasis on readability and engagement
            weights = {
                'readability': 0.25,
                'sentiment': 0.15,
                'topic_relevance': 0.20,
                'engagement': 0.25,
                'seo': 0.15
            }
            
            overall = (
                readability * weights['readability'] +
                sentiment * weights['sentiment'] +
                topic_relevance * weights['topic_relevance'] +
                engagement * weights['engagement'] +
                seo * weights['seo']
            )
            
            return min(1.0, max(0.0, overall))
            
        except Exception:
            return 0.5
    
    async def suggest_improvements(self, metrics: ContentMetrics, content: str) -> List[str]:
        """Suggest content improvements based on metrics"""
        suggestions = []
        
        try:
            # Readability suggestions
            if metrics.readability_score < 0.4:
                suggestions.append("Consider using shorter sentences and simpler words to improve readability")
            elif metrics.readability_score > 0.8:
                suggestions.append("Content might be too simple - consider adding more sophisticated vocabulary")
            
            # Engagement suggestions
            if metrics.engagement_score < 0.4:
                suggestions.append("Add more questions, examples, or actionable content to increase engagement")
            
            # SEO suggestions
            if metrics.seo_score < 0.5:
                suggestions.append("Improve SEO by adding more headings, increasing word count, or enhancing title")
            
            # Topic relevance suggestions
            if metrics.topic_relevance < 0.6:
                suggestions.append("Ensure content better aligns with the title and main topic")
            
            # Sentiment suggestions
            if metrics.sentiment_score < 0.3:
                suggestions.append("Consider adding more positive language to improve sentiment")
            elif metrics.sentiment_score > 0.8:
                suggestions.append("Content might be overly positive - consider a more balanced tone")
            
            # Structure suggestions
            if metrics.paragraph_count < 3:
                suggestions.append("Break content into more paragraphs for better readability")
            
            if metrics.avg_sentence_length > 25:
                suggestions.append("Consider breaking long sentences into shorter ones")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return ["Unable to generate suggestions at this time"]

# Example usage
if __name__ == "__main__":
    async def test_analyzer():
        analyzer = ContentAnalyzer()
        
        sample_content = """
        Artificial Intelligence is revolutionizing the way we create content. 
        This amazing technology allows us to generate high-quality articles, 
        blog posts, and marketing copy in minutes instead of hours.
        
        How does AI content creation work? The process involves several steps:
        1. Input a prompt or topic
        2. AI analyzes the request
        3. Content is generated using advanced algorithms
        4. Human review and editing
        
        For example, many companies now use AI to create product descriptions, 
        social media posts, and even entire blog series. The results are often 
        indistinguishable from human-written content.
        """
        
        metrics = await analyzer.analyze_content(sample_content, "AI Content Creation Guide")
        print(f"Overall Quality: {metrics.overall_quality:.2f}")
        print(f"Readability: {metrics.readability_score:.2f}")
        print(f"Engagement: {metrics.engagement_score:.2f}")
        print(f"SEO Score: {metrics.seo_score:.2f}")
        
        suggestions = await analyzer.suggest_improvements(metrics, sample_content)
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
    
    asyncio.run(test_analyzer())


