"""
Advanced content analysis service with ML capabilities and parallel processing.
"""

import asyncio
import re
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index
from ..core.cache import cached, invalidate_analysis_cache
from ..core.metrics import track_performance, record_analysis_metrics
from ..core.logging import get_logger
from ..models.schemas import ContentAnalysisRequest, ContentAnalysisResponse

logger = get_logger(__name__)


class AdvancedAnalysisEngine:
    """Advanced analysis engine with ML capabilities."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Pre-trained models (simplified for demo)
        self.sentiment_model = self._load_sentiment_model()
        self.topic_model = self._load_topic_model()
        self.language_model = self._load_language_model()
    
    def _load_sentiment_model(self) -> Dict[str, float]:
        """Load sentiment analysis model (simplified)."""
        return {
            "positive": {
                "good": 0.8, "great": 0.9, "excellent": 0.95, "awesome": 0.9,
                "amazing": 0.9, "wonderful": 0.85, "fantastic": 0.9, "brilliant": 0.9,
                "outstanding": 0.9, "perfect": 0.95, "beautiful": 0.8, "nice": 0.7,
                "best": 0.8, "better": 0.7, "improve": 0.6, "success": 0.8,
                "win": 0.8, "victory": 0.9, "love": 0.9, "happy": 0.8, "joy": 0.8
            },
            "negative": {
                "bad": -0.8, "terrible": -0.9, "poor": -0.7, "awful": -0.9,
                "horrible": -0.9, "hate": -0.9, "angry": -0.8, "sad": -0.7,
                "disappointed": -0.7, "frustrated": -0.8, "worst": -0.9, "worse": -0.8,
                "fail": -0.8, "failure": -0.8, "problem": -0.6, "issue": -0.6,
                "error": -0.7, "wrong": -0.7, "broken": -0.8, "damaged": -0.8, "ugly": -0.6
            }
        }
    
    def _load_topic_model(self) -> Dict[str, List[str]]:
        """Load topic classification model (simplified)."""
        return {
            "technology": ["computer", "software", "hardware", "internet", "digital", "ai", "machine", "data", "algorithm", "code"],
            "business": ["company", "market", "sales", "profit", "revenue", "customer", "product", "service", "management", "strategy"],
            "science": ["research", "study", "experiment", "theory", "hypothesis", "analysis", "discovery", "innovation", "method", "result"],
            "health": ["medical", "health", "doctor", "patient", "treatment", "disease", "medicine", "hospital", "therapy", "care"],
            "education": ["school", "student", "teacher", "learning", "education", "course", "study", "knowledge", "skill", "training"]
        }
    
    def _load_language_model(self) -> Dict[str, Dict[str, float]]:
        """Load language detection model (simplified)."""
        return {
            "english": {
                "the": 0.1, "and": 0.08, "or": 0.05, "but": 0.04, "in": 0.06, "on": 0.04, "at": 0.03, "to": 0.07, "for": 0.05, "of": 0.08
            },
            "spanish": {
                "el": 0.08, "la": 0.07, "de": 0.1, "que": 0.06, "y": 0.05, "a": 0.06, "en": 0.05, "un": 0.04, "es": 0.04, "se": 0.03
            },
            "french": {
                "le": 0.07, "la": 0.06, "de": 0.09, "et": 0.05, "à": 0.04, "un": 0.04, "il": 0.03, "que": 0.03, "ne": 0.02, "se": 0.02
            }
        }


# Global analysis engine
_analysis_engine = AdvancedAnalysisEngine()


@track_performance("advanced_analysis")
@cached(ttl=3600, tags=["analysis", "content"])
async def analyze_content_advanced(request: ContentAnalysisRequest) -> ContentAnalysisResponse:
    """Perform advanced content analysis with ML capabilities."""
    start_time = datetime.utcnow()
    
    try:
        # Create content hash for caching
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        
        # Run analyses in parallel
        tasks = [
            _perform_advanced_basic_analysis(request.content),
            _perform_ml_sentiment_analysis(request.content),
            _perform_advanced_readability_analysis(request.content),
            _perform_topic_classification(request.content),
            _perform_advanced_keyword_analysis(request.content),
            _perform_advanced_language_analysis(request.content),
            _perform_style_analysis(request.content),
            _perform_complexity_analysis(request.content),
            _perform_quality_assessment(request.content)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        analysis_results = {
            "basic": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "sentiment": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "readability": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
            "topics": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            "keywords": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])},
            "language": results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])},
            "style": results[6] if not isinstance(results[6], Exception) else {"error": str(results[6])},
            "complexity": results[7] if not isinstance(results[7], Exception) else {"error": str(results[7])},
            "quality": results[8] if not isinstance(results[8], Exception) else {"error": str(results[8])},
            "metadata": {
                "content_hash": content_hash,
                "analysis_version": "2.0.0",
                "timestamp": start_time.isoformat()
            }
        }
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Record metrics
        await record_analysis_metrics("advanced_analysis", True, processing_time)
        
        return ContentAnalysisResponse(
            content=request.content,
            model_version=request.model_version,
            word_count=analysis_results["basic"].get("word_count", 0),
            character_count=analysis_results["basic"].get("character_count", 0),
            analysis_results=analysis_results,
            systems_used={
                "advanced_basic_analysis": True,
                "ml_sentiment_analysis": True,
                "advanced_readability_analysis": True,
                "topic_classification": True,
                "advanced_keyword_analysis": True,
                "advanced_language_analysis": True,
                "style_analysis": True,
                "complexity_analysis": True,
                "quality_assessment": True
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await record_analysis_metrics("advanced_analysis", False, processing_time)
        logger.error(f"Advanced analysis failed: {e}")
        raise


async def _perform_advanced_basic_analysis(content: str) -> Dict[str, Any]:
    """Perform advanced basic text analysis."""
    try:
        # Word analysis
        words = re.findall(r'\b\w+\b', content.lower())
        word_count = len(words)
        
        # Character analysis
        character_count = len(content)
        char_count_no_spaces = len(content.replace(' ', ''))
        
        # Sentence analysis
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Statistical analysis
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_paragraph_length = sentence_count / paragraph_count if paragraph_count > 0 else 0
        
        # Vocabulary analysis
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Word frequency distribution
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(10)
        
        # Text density
        text_density = char_count_no_spaces / character_count if character_count > 0 else 0
        
        return {
            "word_count": word_count,
            "character_count": character_count,
            "character_count_no_spaces": char_count_no_spaces,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_paragraph_length": round(avg_paragraph_length, 2),
            "unique_words": unique_words,
            "vocabulary_diversity": round(vocabulary_diversity, 3),
            "most_common_words": most_common_words,
            "text_density": round(text_density, 3)
        }
        
    except Exception as e:
        logger.error(f"Advanced basic analysis failed: {e}")
        return {"error": str(e)}


async def _perform_ml_sentiment_analysis(content: str) -> Dict[str, Any]:
    """Perform ML-based sentiment analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        model = _analysis_engine.sentiment_model
        
        # Calculate sentiment scores
        positive_score = 0
        negative_score = 0
        positive_words_found = []
        negative_words_found = []
        
        for word in words:
            if word in model["positive"]:
                positive_score += model["positive"][word]
                positive_words_found.append(word)
            elif word in model["negative"]:
                negative_score += abs(model["negative"][word])
                negative_words_found.append(word)
        
        # Normalize scores
        total_sentiment_words = len(positive_words_found) + len(negative_words_found)
        if total_sentiment_words > 0:
            normalized_positive = positive_score / total_sentiment_words
            normalized_negative = negative_score / total_sentiment_words
            sentiment_score = normalized_positive - normalized_negative
        else:
            sentiment_score = 0
        
        # Determine sentiment label and confidence
        if sentiment_score > 0.2:
            sentiment_label = "positive"
            confidence = min(sentiment_score, 1.0)
        elif sentiment_score < -0.2:
            sentiment_label = "negative"
            confidence = min(abs(sentiment_score), 1.0)
        else:
            sentiment_label = "neutral"
            confidence = 1.0 - abs(sentiment_score)
        
        # Emotional analysis
        emotions = _analyze_emotions(words)
        
        return {
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "confidence": round(confidence, 3),
            "positive_score": round(positive_score, 3),
            "negative_score": round(negative_score, 3),
            "positive_words_found": positive_words_found[:10],
            "negative_words_found": negative_words_found[:10],
            "total_sentiment_words": total_sentiment_words,
            "emotions": emotions
        }
        
    except Exception as e:
        logger.error(f"ML sentiment analysis failed: {e}")
        return {"error": str(e)}


async def _perform_advanced_readability_analysis(content: str) -> Dict[str, Any]:
    """Perform advanced readability analysis."""
    try:
        # Multiple readability formulas
        flesch_ease = flesch_reading_ease(content)
        flesch_grade = flesch_kincaid_grade(content)
        gunning_fog_score = gunning_fog(content)
        smog_score = smog_index(content)
        
        # Average readability
        avg_readability = (flesch_ease + (100 - flesch_grade * 10) + (100 - gunning_fog_score * 10) + (100 - smog_score * 10)) / 4
        
        # Readability level
        if avg_readability >= 80:
            readability_level = "Very Easy"
            target_audience = "Elementary school"
        elif avg_readability >= 60:
            readability_level = "Easy"
            target_audience = "Middle school"
        elif avg_readability >= 40:
            readability_level = "Moderate"
            target_audience = "High school"
        elif avg_readability >= 20:
            readability_level = "Difficult"
            target_audience = "College"
        else:
            readability_level = "Very Difficult"
            target_audience = "Graduate level"
        
        # Text complexity indicators
        words = re.findall(r'\b\w+\b', content.lower())
        complex_words = [word for word in words if len(word) > 6 or _count_syllables(word) > 2]
        complexity_ratio = len(complex_words) / len(words) if words else 0
        
        return {
            "flesch_reading_ease": round(flesch_ease, 2),
            "flesch_kincaid_grade": round(flesch_grade, 2),
            "gunning_fog": round(gunning_fog_score, 2),
            "smog_index": round(smog_score, 2),
            "average_readability": round(avg_readability, 2),
            "readability_level": readability_level,
            "target_audience": target_audience,
            "complexity_ratio": round(complexity_ratio, 3),
            "complex_words_count": len(complex_words)
        }
        
    except Exception as e:
        logger.error(f"Advanced readability analysis failed: {e}")
        return {"error": str(e)}


async def _perform_topic_classification(content: str) -> Dict[str, Any]:
    """Perform topic classification."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        model = _analysis_engine.topic_model
        
        topic_scores = {}
        for topic, keywords in model.items():
            score = sum(1 for word in words if word in keywords)
            topic_scores[topic] = score
        
        # Normalize scores
        total_score = sum(topic_scores.values())
        if total_score > 0:
            topic_scores = {topic: score / total_score for topic, score in topic_scores.items()}
        
        # Get primary topic
        primary_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "unknown"
        confidence = topic_scores.get(primary_topic, 0)
        
        return {
            "primary_topic": primary_topic,
            "confidence": round(confidence, 3),
            "topic_scores": {topic: round(score, 3) for topic, score in topic_scores.items()},
            "all_topics": list(topic_scores.keys())
        }
        
    except Exception as e:
        logger.error(f"Topic classification failed: {e}")
        return {"error": str(e)}


async def _perform_advanced_keyword_analysis(content: str) -> Dict[str, Any]:
    """Perform advanced keyword analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Remove stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # TF-IDF calculation (simplified)
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)
        
        # Calculate TF-IDF scores
        tfidf_scores = {}
        for word, freq in word_freq.items():
            tf = freq / total_words
            # Simplified IDF (in real implementation, would use document corpus)
            idf = math.log(total_words / freq) if freq > 0 else 0
            tfidf_scores[word] = tf * idf
        
        # Get top keywords by TF-IDF
        top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Keyword density
        keyword_density = {word: (freq / total_words) * 100 for word, freq in word_freq.items()}
        
        # N-gram analysis
        bigrams = _extract_ngrams(filtered_words, 2)
        trigrams = _extract_ngrams(filtered_words, 3)
        
        return {
            "top_keywords_tfidf": [(word, round(score, 4)) for word, score in top_keywords],
            "keyword_frequency": dict(word_freq.most_common(20)),
            "keyword_density": {word: round(density, 2) for word, density in keyword_density.items()},
            "bigrams": bigrams[:10],
            "trigrams": trigrams[:10],
            "total_unique_keywords": len(word_freq),
            "keyword_richness": round(len(word_freq) / total_words, 3) if total_words > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Advanced keyword analysis failed: {e}")
        return {"error": str(e)}


async def _perform_advanced_language_analysis(content: str) -> Dict[str, Any]:
    """Perform advanced language analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        model = _analysis_engine.language_model
        
        # Calculate language probabilities
        language_scores = {}
        for language, word_probs in model.items():
            score = 0
            for word in words:
                if word in word_probs:
                    score += word_probs[word]
            language_scores[language] = score
        
        # Get most likely language
        detected_language = max(language_scores, key=language_scores.get) if language_scores else "unknown"
        confidence = language_scores.get(detected_language, 0)
        
        # Text statistics
        total_words = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # Character analysis
        char_freq = Counter(content.lower())
        most_common_chars = char_freq.most_common(10)
        
        return {
            "detected_language": detected_language,
            "confidence": round(confidence, 3),
            "language_scores": {lang: round(score, 3) for lang, score in language_scores.items()},
            "total_words": total_words,
            "unique_words": unique_words,
            "avg_word_length": round(avg_word_length, 2),
            "most_common_characters": most_common_chars,
            "character_diversity": len(char_freq)
        }
        
    except Exception as e:
        logger.error(f"Advanced language analysis failed: {e}")
        return {"error": str(e)}


async def _perform_style_analysis(content: str) -> Dict[str, Any]:
    """Perform writing style analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Sentence structure analysis
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        sentence_variation = max(sentence_lengths) - min(sentence_lengths) if sentence_lengths else 0
        
        # Punctuation analysis
        punctuation_count = len(re.findall(r'[.!?,:;]', content))
        punctuation_density = punctuation_count / len(content) if content else 0
        
        # Capitalization analysis
        capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', content))
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', content))
        
        # Contraction analysis
        contractions = len(re.findall(r"\b\w+'\w+\b", content))
        
        # Passive voice detection (simplified)
        passive_indicators = ["was", "were", "been", "being", "get", "got", "getting"]
        passive_count = sum(1 for word in words if word in passive_indicators)
        passive_ratio = passive_count / len(words) if words else 0
        
        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "sentence_variation": sentence_variation,
            "punctuation_density": round(punctuation_density, 4),
            "capitalized_words": capitalized_words,
            "all_caps_words": all_caps_words,
            "contractions": contractions,
            "passive_voice_ratio": round(passive_ratio, 3),
            "writing_style": _classify_writing_style(avg_sentence_length, passive_ratio, punctuation_density)
        }
        
    except Exception as e:
        logger.error(f"Style analysis failed: {e}")
        return {"error": str(e)}


async def _perform_complexity_analysis(content: str) -> Dict[str, Any]:
    """Perform text complexity analysis."""
    try:
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Lexical complexity
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        # Syntactic complexity
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Morphological complexity
        complex_words = [word for word in words if len(word) > 6 or _count_syllables(word) > 2]
        morphological_complexity = len(complex_words) / len(words) if words else 0
        
        # Semantic complexity (simplified)
        semantic_complexity = _calculate_semantic_complexity(words)
        
        # Overall complexity score
        complexity_score = (
            lexical_diversity * 0.3 +
            min(avg_sentence_length / 20, 1) * 0.3 +
            morphological_complexity * 0.2 +
            semantic_complexity * 0.2
        )
        
        return {
            "lexical_diversity": round(lexical_diversity, 3),
            "syntactic_complexity": round(avg_sentence_length, 2),
            "morphological_complexity": round(morphological_complexity, 3),
            "semantic_complexity": round(semantic_complexity, 3),
            "overall_complexity": round(complexity_score, 3),
            "complexity_level": _get_complexity_level(complexity_score)
        }
        
    except Exception as e:
        logger.error(f"Complexity analysis failed: {e}")
        return {"error": str(e)}


async def _perform_quality_assessment(content: str) -> Dict[str, Any]:
    """Perform overall quality assessment."""
    try:
        # Get basic metrics
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Quality indicators
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability score
        try:
            readability = flesch_reading_ease(content)
        except:
            readability = 50  # Default middle score
        
        # Quality scoring
        length_score = min(word_count / 100, 1)  # Prefer 100+ words
        readability_score = readability / 100
        structure_score = min(sentence_count / 5, 1)  # Prefer 5+ sentences
        
        # Overall quality score
        quality_score = (length_score * 0.3 + readability_score * 0.4 + structure_score * 0.3)
        
        # Quality level
        if quality_score >= 0.8:
            quality_level = "Excellent"
        elif quality_score >= 0.6:
            quality_level = "Good"
        elif quality_score >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        # Recommendations
        recommendations = []
        if word_count < 50:
            recommendations.append("Consider expanding content for better analysis")
        if readability < 30:
            recommendations.append("Improve readability for broader audience")
        if sentence_count < 3:
            recommendations.append("Add more sentences for better structure")
        
        return {
            "quality_score": round(quality_score, 3),
            "quality_level": quality_level,
            "length_score": round(length_score, 3),
            "readability_score": round(readability_score, 3),
            "structure_score": round(structure_score, 3),
            "recommendations": recommendations,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2)
        }
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return {"error": str(e)}


# Helper functions
def _analyze_emotions(words: List[str]) -> Dict[str, int]:
    """Analyze emotional content."""
    emotions = {
        "joy": ["happy", "joy", "excited", "cheerful", "delighted"],
        "sadness": ["sad", "depressed", "melancholy", "gloomy", "sorrowful"],
        "anger": ["angry", "mad", "furious", "irritated", "annoyed"],
        "fear": ["afraid", "scared", "terrified", "worried", "anxious"],
        "surprise": ["surprised", "amazed", "shocked", "astonished", "stunned"]
    }
    
    emotion_counts = {}
    for emotion, keywords in emotions.items():
        count = sum(1 for word in words if word in keywords)
        emotion_counts[emotion] = count
    
    return emotion_counts


def _extract_ngrams(words: List[str], n: int) -> List[Tuple[str, int]]:
    """Extract n-grams from words."""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams).most_common(10)


def _count_syllables(word: str) -> int:
    """Count syllables in a word."""
    word = word.lower()
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count)


def _classify_writing_style(avg_sentence_length: float, passive_ratio: float, punctuation_density: float) -> str:
    """Classify writing style."""
    if avg_sentence_length > 20 and passive_ratio > 0.1:
        return "Academic"
    elif avg_sentence_length < 10 and punctuation_density > 0.05:
        return "Conversational"
    elif avg_sentence_length > 15:
        return "Formal"
    else:
        return "Casual"


def _calculate_semantic_complexity(words: List[str]) -> float:
    """Calculate semantic complexity (simplified)."""
    # This is a simplified version - in reality would use word embeddings
    complex_words = [word for word in words if len(word) > 6]
    return len(complex_words) / len(words) if words else 0


def _get_complexity_level(complexity_score: float) -> str:
    """Get complexity level description."""
    if complexity_score >= 0.8:
        return "Very Complex"
    elif complexity_score >= 0.6:
        return "Complex"
    elif complexity_score >= 0.4:
        return "Moderate"
    elif complexity_score >= 0.2:
        return "Simple"
    else:
        return "Very Simple"


