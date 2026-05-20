"""
Content Analyzer
================

Advanced content analysis system for quality assessment.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import re
import json
import nltk
from dataclasses import dataclass
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ContentAnalysis:
    """Content analysis result."""
    readability_score: float
    coherence_score: float
    engagement_score: float
    quality_score: float
    metrics: Dict[str, Any]
    suggestions: List[str]
    timestamp: datetime

class ContentAnalyzer:
    """
    Advanced content analysis system.
    
    Features:
    - Readability analysis
    - Coherence assessment
    - Engagement measurement
    - Quality scoring
    - Content suggestions
    - Language analysis
    """
    
    def __init__(self):
        self.analysis_cache = {}
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.0
        }
        
    async def initialize(self):
        """Initialize content analyzer."""
        logger.info("Initializing Content Analyzer...")
        
        try:
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {str(e)}")
            
            logger.info("Content Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Content Analyzer: {str(e)}")
            raise
    
    async def analyze_content(
        self,
        content: str,
        content_type: str = "general",
        language: str = "en"
    ) -> ContentAnalysis:
        """
        Analyze content quality and characteristics.
        
        Args:
            content: Content to analyze
            content_type: Type of content (general, technical, creative, etc.)
            language: Content language
            
        Returns:
            Content analysis result
        """
        try:
            # Check cache
            cache_key = f"{hash(content)}_{content_type}_{language}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Perform analysis
            readability_score = await self._analyze_readability(content)
            coherence_score = await self._analyze_coherence(content)
            engagement_score = await self._analyze_engagement(content)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                readability_score,
                coherence_score,
                engagement_score,
                content_type
            )
            
            # Generate metrics
            metrics = await self._generate_metrics(content)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(
                content,
                readability_score,
                coherence_score,
                engagement_score
            )
            
            # Create analysis result
            analysis = ContentAnalysis(
                readability_score=readability_score,
                coherence_score=coherence_score,
                engagement_score=engagement_score,
                quality_score=quality_score,
                metrics=metrics,
                suggestions=suggestions,
                timestamp=datetime.utcnow()
            )
            
            # Cache result
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {str(e)}")
            # Return default analysis
            return ContentAnalysis(
                readability_score=0.5,
                coherence_score=0.5,
                engagement_score=0.5,
                quality_score=0.5,
                metrics={},
                suggestions=["Analysis failed"],
                timestamp=datetime.utcnow()
            )
    
    async def _analyze_readability(self, content: str) -> float:
        """Analyze content readability."""
        try:
            # Basic readability metrics
            sentences = self._split_sentences(content)
            words = content.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.5
            
            # Average words per sentence
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Average syllables per word
            avg_syllables = self._calculate_avg_syllables(words)
            
            # Flesch Reading Ease Score
            flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables)
            
            # Normalize to 0-1 scale
            readability_score = max(0, min(1, flesch_score / 100))
            
            return readability_score
            
        except Exception as e:
            logger.error(f"Failed to analyze readability: {str(e)}")
            return 0.5
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            return [text]
    
    def _calculate_avg_syllables(self, words: List[str]) -> float:
        """Calculate average syllables per word."""
        try:
            total_syllables = sum(self._count_syllables(word) for word in words)
            return total_syllables / len(words) if words else 0
        except Exception:
            return 2.0  # Default average
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        try:
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
        except Exception:
            return 1
    
    async def _analyze_coherence(self, content: str) -> float:
        """Analyze content coherence."""
        try:
            sentences = self._split_sentences(content)
            if len(sentences) < 2:
                return 0.5
            
            # Check for transition words
            transition_words = [
                'however', 'therefore', 'moreover', 'furthermore', 'additionally',
                'consequently', 'meanwhile', 'subsequently', 'finally', 'in conclusion',
                'first', 'second', 'third', 'next', 'then', 'also', 'besides'
            ]
            
            transition_count = sum(1 for word in transition_words if word in content.lower())
            transition_score = min(1.0, transition_count / len(sentences))
            
            # Check for logical flow indicators
            flow_indicators = [
                'because', 'since', 'as a result', 'due to', 'in order to',
                'for example', 'for instance', 'specifically', 'in particular'
            ]
            
            flow_count = sum(1 for phrase in flow_indicators if phrase in content.lower())
            flow_score = min(1.0, flow_count / len(sentences))
            
            # Check for repetition (negative indicator)
            words = content.lower().split()
            word_counts = Counter(words)
            repetition_score = 1.0 - min(0.5, sum(1 for count in word_counts.values() if count > 3) / len(word_counts))
            
            # Combine scores
            coherence_score = (transition_score * 0.4 + flow_score * 0.4 + repetition_score * 0.2)
            
            return coherence_score
            
        except Exception as e:
            logger.error(f"Failed to analyze coherence: {str(e)}")
            return 0.5
    
    async def _analyze_engagement(self, content: str) -> float:
        """Analyze content engagement."""
        try:
            # Check for engaging elements
            questions = content.count('?')
            exclamations = content.count('!')
            quotes = content.count('"') + content.count("'")
            
            # Check for interactive elements
            interactive_words = [
                'you', 'your', 'imagine', 'consider', 'think', 'suppose',
                'what if', 'have you', 'do you', 'can you'
            ]
            
            interactive_count = sum(1 for word in interactive_words if word in content.lower())
            
            # Check for examples and stories
            example_indicators = [
                'for example', 'for instance', 'such as', 'like', 'including',
                'story', 'case', 'scenario', 'situation'
            ]
            
            example_count = sum(1 for phrase in example_indicators if phrase in content.lower())
            
            # Check for emotional words
            emotional_words = [
                'amazing', 'incredible', 'fantastic', 'wonderful', 'exciting',
                'important', 'crucial', 'essential', 'vital', 'significant'
            ]
            
            emotional_count = sum(1 for word in emotional_words if word in content.lower())
            
            # Calculate engagement score
            word_count = len(content.split())
            if word_count == 0:
                return 0.5
            
            engagement_score = min(1.0, (
                (questions * 2 + exclamations + quotes * 0.5) / word_count * 100 +
                interactive_count / word_count * 50 +
                example_count / word_count * 30 +
                emotional_count / word_count * 20
            ) / 100)
            
            return engagement_score
            
        except Exception as e:
            logger.error(f"Failed to analyze engagement: {str(e)}")
            return 0.5
    
    def _calculate_quality_score(
        self,
        readability_score: float,
        coherence_score: float,
        engagement_score: float,
        content_type: str
    ) -> float:
        """Calculate overall quality score."""
        try:
            # Weight factors based on content type
            if content_type == "technical":
                weights = {'readability': 0.4, 'coherence': 0.5, 'engagement': 0.1}
            elif content_type == "creative":
                weights = {'readability': 0.3, 'coherence': 0.3, 'engagement': 0.4}
            elif content_type == "educational":
                weights = {'readability': 0.4, 'coherence': 0.4, 'engagement': 0.2}
            else:  # general
                weights = {'readability': 0.35, 'coherence': 0.35, 'engagement': 0.3}
            
            quality_score = (
                readability_score * weights['readability'] +
                coherence_score * weights['coherence'] +
                engagement_score * weights['engagement']
            )
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {str(e)}")
            return 0.5
    
    async def _generate_metrics(self, content: str) -> Dict[str, Any]:
        """Generate detailed content metrics."""
        try:
            words = content.split()
            sentences = self._split_sentences(content)
            paragraphs = content.split('\n\n')
            
            # Basic metrics
            metrics = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'character_count': len(content),
                'character_count_no_spaces': len(content.replace(' ', '')),
                'average_words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'average_sentences_per_paragraph': len(sentences) / len(paragraphs) if paragraphs else 0
            }
            
            # Vocabulary metrics
            unique_words = set(word.lower() for word in words)
            metrics['unique_word_count'] = len(unique_words)
            metrics['vocabulary_diversity'] = len(unique_words) / len(words) if words else 0
            
            # Readability metrics
            if words and sentences:
                avg_syllables = self._calculate_avg_syllables(words)
                metrics['average_syllables_per_word'] = avg_syllables
                metrics['flesch_reading_ease'] = 206.835 - (1.015 * metrics['average_words_per_sentence']) - (84.6 * avg_syllables)
            
            # Content structure metrics
            metrics['question_count'] = content.count('?')
            metrics['exclamation_count'] = content.count('!')
            metrics['quote_count'] = content.count('"') + content.count("'")
            metrics['list_count'] = content.count('-') + content.count('*') + content.count('1.')
            
            # Language metrics
            metrics['uppercase_ratio'] = sum(1 for c in content if c.isupper()) / len(content) if content else 0
            metrics['digit_ratio'] = sum(1 for c in content if c.isdigit()) / len(content) if content else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to generate metrics: {str(e)}")
            return {}
    
    async def _generate_suggestions(
        self,
        content: str,
        readability_score: float,
        coherence_score: float,
        engagement_score: float
    ) -> List[str]:
        """Generate content improvement suggestions."""
        try:
            suggestions = []
            
            # Readability suggestions
            if readability_score < 0.6:
                suggestions.append("Consider using shorter sentences and simpler words to improve readability")
            
            if readability_score > 0.9:
                suggestions.append("Content might be too simple - consider adding more sophisticated vocabulary")
            
            # Coherence suggestions
            if coherence_score < 0.6:
                suggestions.append("Add transition words and phrases to improve flow between ideas")
                suggestions.append("Ensure logical progression from one point to the next")
            
            # Engagement suggestions
            if engagement_score < 0.5:
                suggestions.append("Add questions to engage the reader")
                suggestions.append("Include examples, stories, or case studies")
                suggestions.append("Use more active voice and direct language")
            
            # Content-specific suggestions
            word_count = len(content.split())
            if word_count < 100:
                suggestions.append("Consider expanding the content with more details and examples")
            elif word_count > 2000:
                suggestions.append("Consider breaking the content into smaller, more digestible sections")
            
            # Structure suggestions
            sentences = self._split_sentences(content)
            if sentences:
                avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if avg_length > 25:
                    suggestions.append("Some sentences are very long - consider breaking them down")
                elif avg_length < 8:
                    suggestions.append("Consider combining some short sentences for better flow")
            
            # Paragraph suggestions
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                avg_paragraph_length = word_count / len(paragraphs)
                if avg_paragraph_length > 200:
                    suggestions.append("Some paragraphs are very long - consider breaking them down")
                elif avg_paragraph_length < 50:
                    suggestions.append("Consider combining short paragraphs for better structure")
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
            return ["Unable to generate suggestions"]
    
    async def compare_content(
        self,
        content1: str,
        content2: str
    ) -> Dict[str, Any]:
        """Compare two pieces of content."""
        try:
            analysis1 = await self.analyze_content(content1)
            analysis2 = await self.analyze_content(content2)
            
            comparison = {
                'content1': {
                    'quality_score': analysis1.quality_score,
                    'readability_score': analysis1.readability_score,
                    'coherence_score': analysis1.coherence_score,
                    'engagement_score': analysis1.engagement_score
                },
                'content2': {
                    'quality_score': analysis2.quality_score,
                    'readability_score': analysis2.readability_score,
                    'coherence_score': analysis2.coherence_score,
                    'engagement_score': analysis2.engagement_score
                },
                'differences': {
                    'quality_difference': analysis2.quality_score - analysis1.quality_score,
                    'readability_difference': analysis2.readability_score - analysis1.readability_score,
                    'coherence_difference': analysis2.coherence_score - analysis1.coherence_score,
                    'engagement_difference': analysis2.engagement_score - analysis1.engagement_score
                },
                'better_content': 'content2' if analysis2.quality_score > analysis1.quality_score else 'content1'
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare content: {str(e)}")
            return {}
    
    async def get_quality_level(self, quality_score: float) -> str:
        """Get quality level description."""
        try:
            if quality_score >= self.quality_thresholds['excellent']:
                return 'excellent'
            elif quality_score >= self.quality_thresholds['good']:
                return 'good'
            elif quality_score >= self.quality_thresholds['fair']:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Failed to get quality level: {str(e)}")
            return 'unknown'
    
    async def cleanup(self):
        """Cleanup content analyzer."""
        try:
            # Clear cache
            self.analysis_cache.clear()
            
            logger.info("Content Analyzer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Content Analyzer: {str(e)}")











