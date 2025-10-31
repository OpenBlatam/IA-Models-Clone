"""
Content analysis service with functional approach.
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from textstat import flesch_reading_ease

from ..core.exceptions import ValidationError
from ..models.schemas import ContentAnalysisRequest, ContentAnalysisResponse
from ..core.logging import get_logger

logger = get_logger(__name__)


async def analyze_content(request: ContentAnalysisRequest) -> ContentAnalysisResponse:
    """Analyze content using multiple analysis techniques."""
    start_time = datetime.utcnow()
    
    try:
        # Basic text analysis
        basic_analysis = await _perform_basic_analysis(request.content)
        
        # Sentiment analysis
        sentiment_analysis = await _perform_sentiment_analysis(request.content)
        
        # Readability analysis
        readability_analysis = await _perform_readability_analysis(request.content)
        
        # Keyword extraction
        keyword_analysis = await _perform_keyword_analysis(request.content)
        
        # Language detection
        language_analysis = await _perform_language_analysis(request.content)
        
        # Combine all analyses
        analysis_results = {
            "basic": basic_analysis,
            "sentiment": sentiment_analysis,
            "readability": readability_analysis,
            "keywords": keyword_analysis,
            "language": language_analysis,
            "metadata": request.metadata or {}
        }
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ContentAnalysisResponse(
            content=request.content,
            model_version=request.model_version,
            word_count=basic_analysis["word_count"],
            character_count=basic_analysis["character_count"],
            analysis_results=analysis_results,
            systems_used={
                "basic_analysis": True,
                "sentiment_analysis": True,
                "readability_analysis": True,
                "keyword_analysis": True,
                "language_analysis": True
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing content: {e}")
        raise ValidationError(f"Content analysis failed: {e}")


async def _perform_basic_analysis(content: str) -> Dict[str, Any]:
    """Perform basic text analysis."""
    try:
        # Word count
        words = re.findall(r'\b\w+\b', content.lower())
        word_count = len(words)
        
        # Character count
        character_count = len(content)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Unique words
        unique_words = len(set(words))
        
        return {
            "word_count": word_count,
            "character_count": character_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "unique_words": unique_words,
            "vocabulary_diversity": round(unique_words / word_count, 3) if word_count > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error in basic analysis: {e}")
        return {"error": str(e)}


async def _perform_sentiment_analysis(content: str) -> Dict[str, Any]:
    """Perform sentiment analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Simple sentiment word lists
        positive_words = {
            "good", "great", "excellent", "awesome", "positive", "happy", "joy", "love",
            "amazing", "wonderful", "fantastic", "brilliant", "outstanding", "perfect",
            "beautiful", "nice", "best", "better", "improve", "success", "win", "victory"
        }
        
        negative_words = {
            "bad", "terrible", "poor", "negative", "awful", "horrible", "hate", "angry",
            "sad", "disappointed", "frustrated", "worst", "worse", "fail", "failure",
            "problem", "issue", "error", "wrong", "broken", "damaged", "ugly"
        }
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment_label = "positive"
        elif sentiment_score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "total_sentiment_words": total_sentiment_words
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"error": str(e)}


async def _perform_readability_analysis(content: str) -> Dict[str, Any]:
    """Perform readability analysis."""
    try:
        # Flesch Reading Ease Score
        flesch_score = flesch_reading_ease(content)
        
        # Determine readability level
        if flesch_score >= 90:
            readability_level = "Very Easy"
        elif flesch_score >= 80:
            readability_level = "Easy"
        elif flesch_score >= 70:
            readability_level = "Fairly Easy"
        elif flesch_score >= 60:
            readability_level = "Standard"
        elif flesch_score >= 50:
            readability_level = "Fairly Difficult"
        elif flesch_score >= 30:
            readability_level = "Difficult"
        else:
            readability_level = "Very Difficult"
        
        # Calculate average syllables per word (approximation)
        words = re.findall(r'\b\w+\b', content.lower())
        total_syllables = sum(_count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / len(words) if words else 0
        
        return {
            "flesch_reading_ease": round(flesch_score, 2),
            "readability_level": readability_level,
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "recommended_audience": _get_recommended_audience(flesch_score)
        }
        
    except Exception as e:
        logger.error(f"Error in readability analysis: {e}")
        return {"error": str(e)}


async def _perform_keyword_analysis(content: str) -> Dict[str, Any]:
    """Perform keyword extraction and analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_frequency = {}
        for word in filtered_words:
            word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, count in sorted_words[:10]]
        
        # Calculate keyword density
        total_words = len(words)
        keyword_density = {}
        for word, count in word_frequency.items():
            density = (count / total_words) * 100
            keyword_density[word] = round(density, 2)
        
        return {
            "top_keywords": top_keywords,
            "keyword_frequency": dict(sorted_words[:20]),
            "keyword_density": keyword_density,
            "total_unique_keywords": len(word_frequency),
            "keyword_richness": round(len(word_frequency) / total_words, 3) if total_words > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error in keyword analysis: {e}")
        return {"error": str(e)}


async def _perform_language_analysis(content: str) -> Dict[str, Any]:
    """Perform language detection and analysis."""
    try:
        # Simple language detection based on common words
        english_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        spanish_words = {"el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le"}
        french_words = {"le", "la", "de", "et", "à", "un", "il", "que", "ne", "se", "ce", "pas", "pour", "sur"}
        
        words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Count language-specific words
        english_count = len(words.intersection(english_words))
        spanish_count = len(words.intersection(spanish_words))
        french_count = len(words.intersection(french_words))
        
        # Determine most likely language
        language_scores = {
            "english": english_count,
            "spanish": spanish_count,
            "french": french_count
        }
        
        detected_language = max(language_scores, key=language_scores.get)
        confidence = language_scores[detected_language] / max(1, len(words) * 0.1)
        
        return {
            "detected_language": detected_language,
            "confidence": round(min(confidence, 1.0), 3),
            "language_scores": language_scores,
            "total_words_analyzed": len(words)
        }
        
    except Exception as e:
        logger.error(f"Error in language analysis: {e}")
        return {"error": str(e)}


def _count_syllables(word: str) -> int:
    """Count syllables in a word (approximation)."""
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


def _get_recommended_audience(flesch_score: float) -> str:
    """Get recommended audience based on Flesch score."""
    if flesch_score >= 80:
        return "General audience, children"
    elif flesch_score >= 60:
        return "General audience, teenagers"
    elif flesch_score >= 40:
        return "Adults, some education required"
    else:
        return "Specialists, advanced education required"




