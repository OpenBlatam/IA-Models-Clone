"""
Content Analyzer Service
========================

Single responsibility: Analyze content and extract metrics.
"""

from typing import Dict, Any
import statistics
import re
from collections import Counter

from ..value_objects.content_metrics import ContentMetrics


class ContentAnalyzer:
    """
    Service for analyzing content and extracting metrics.
    
    Single Responsibility: Analyze text content and generate metrics.
    """
    
    def __init__(self):
        """Initialize the content analyzer."""
        self._positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent'
        }
        self._negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor',
            'worst', 'dreadful', 'atrocious', 'appalling', 'deplorable'
        }
    
    def analyze(self, content: str) -> ContentMetrics:
        """
        Analyze content and return comprehensive metrics.
        
        Args:
            content: Text content to analyze
            
        Returns:
            ContentMetrics with analysis results
        """
        if not content or not content.strip():
            return ContentMetrics.empty()
        
        # Basic text analysis
        words = self._extract_words(content)
        sentences = self._extract_sentences(content)
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = self._calculate_avg_word_length(words)
        
        # Calculate scores
        readability_score = self._calculate_readability(word_count, sentence_count, content)
        sentiment_score = self._calculate_sentiment(words)
        complexity_score = self._calculate_complexity(content, words)
        topic_diversity = self._calculate_topic_diversity(words)
        consistency_score = self._calculate_consistency(sentences)
        coherence_score = self._calculate_coherence(content, sentences)
        relevance_score = self._calculate_relevance(content)
        creativity_score = self._calculate_creativity(words)
        
        # Calculate overall quality
        quality_score = self._calculate_quality_score(
            readability_score, sentiment_score, complexity_score, consistency_score
        )
        
        return ContentMetrics(
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            complexity_score=complexity_score,
            topic_diversity=topic_diversity,
            consistency_score=consistency_score,
            quality_score=quality_score,
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            creativity_score=creativity_score
        )
    
    def _extract_words(self, content: str) -> list:
        """Extract words from content."""
        # Remove punctuation and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        return words
    
    def _extract_sentences(self, content: str) -> list:
        """Extract sentences from content."""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_avg_word_length(self, words: list) -> float:
        """Calculate average word length."""
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _calculate_readability(self, word_count: int, sentence_count: int, content: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Calculate average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Calculate average syllables per word (simplified)
        syllables = self._count_syllables(content)
        avg_syllables_per_word = syllables / word_count if word_count > 0 else 0
        
        # Simplified Flesch Reading Ease formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, readability / 100))
    
    def _calculate_sentiment(self, words: list) -> float:
        """Calculate sentiment score."""
        if not words:
            return 0.5  # Neutral
        
        positive_count = sum(1 for word in words if word in self._positive_words)
        negative_count = sum(1 for word in words if word in self._negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return (sentiment + 1) / 2  # Normalize to 0-1
    
    def _calculate_complexity(self, content: str, words: list) -> float:
        """Calculate content complexity."""
        if not words:
            return 0.0
        
        # Factors: average word length, vocabulary diversity
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words)
        
        # Normalize complexity score
        complexity = (avg_word_length / 10 + vocabulary_diversity) / 2
        return min(1.0, complexity)
    
    def _calculate_topic_diversity(self, words: list) -> float:
        """Calculate topic diversity."""
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        return min(1.0, unique_words / len(words))
    
    def _calculate_consistency(self, sentences: list) -> float:
        """Calculate content consistency."""
        if len(sentences) < 2:
            return 1.0
        
        # Check for consistent sentence length
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        if not sentence_lengths:
            return 1.0
        
        # Calculate coefficient of variation
        mean_length = statistics.mean(sentence_lengths)
        if mean_length == 0:
            return 1.0
        
        std_length = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        cv = std_length / mean_length
        
        # Convert to consistency score (lower CV = higher consistency)
        return max(0.0, 1.0 - cv)
    
    def _calculate_coherence(self, content: str, sentences: list) -> float:
        """Calculate content coherence."""
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'thus', 'meanwhile', 'additionally', 'similarly', 'conversely'
        }
        
        transition_count = sum(
            1 for sentence in sentences
            if any(word in sentence.lower() for word in transition_words)
        )
        
        return min(1.0, transition_count / len(sentences) + 0.5)
    
    def _calculate_relevance(self, content: str) -> float:
        """Calculate content relevance (placeholder)."""
        # In a real implementation, this would compare against a topic or context
        return 0.8  # Placeholder
    
    def _calculate_creativity(self, words: list) -> float:
        """Calculate content creativity."""
        if not words:
            return 0.0
        
        # Count unique word usage
        word_counts = Counter(words)
        unique_words = sum(1 for count in word_counts.values() if count == 1)
        
        return min(1.0, unique_words / len(words))
    
    def _calculate_quality_score(
        self,
        readability: float,
        sentiment: float,
        complexity: float,
        consistency: float
    ) -> float:
        """Calculate overall quality score."""
        # Weighted average of quality factors
        weights = {
            'readability': 0.3,
            'sentiment': 0.2,
            'complexity': 0.2,
            'consistency': 0.3
        }
        
        return (
            readability * weights['readability'] +
            sentiment * weights['sentiment'] +
            complexity * weights['complexity'] +
            consistency * weights['consistency']
        )
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified)."""
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
            
            # Adjust for silent 'e'
            if word.endswith('e'):
                word_count -= 1
            
            # Every word has at least one syllable
            if word_count == 0:
                word_count = 1
            
            count += word_count
        
        return count




