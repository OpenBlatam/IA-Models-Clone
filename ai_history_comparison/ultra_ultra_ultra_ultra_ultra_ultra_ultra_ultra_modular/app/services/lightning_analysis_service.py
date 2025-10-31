"""
Lightning-fast analysis service with extreme speed optimizations.
"""

import asyncio
import time
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache, partial
import re
from collections import Counter, defaultdict

from ..core.speed_optimization import ultra_fast, cpu_optimized, io_optimized, vectorized, cached_ultra_fast
from ..core.cache import cached, invalidate_analysis_cache
from ..core.metrics import track_performance, record_analysis_metrics
from ..core.logging import get_logger
from ..models.schemas import ContentAnalysisRequest, ContentAnalysisResponse

logger = get_logger(__name__)


class LightningAnalysisEngine:
    """Lightning-fast analysis engine with extreme speed optimizations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._init_lightning_pools()
        self._init_precompiled_functions()
        self._init_vectorized_operations()
    
    def _init_lightning_pools(self):
        """Initialize lightning-fast pools."""
        # Ultra-fast thread pool
        self.lightning_thread_pool = ThreadPoolExecutor(
            max_workers=min(128, mp.cpu_count() * 8),
            thread_name_prefix="lightning_worker"
        )
        
        # Process pool for CPU-intensive tasks
        self.lightning_process_pool = ProcessPoolExecutor(
            max_workers=min(32, mp.cpu_count() * 2)
        )
        
        # I/O pool for async operations
        self.lightning_io_pool = ThreadPoolExecutor(
            max_workers=min(256, mp.cpu_count() * 16),
            thread_name_prefix="lightning_io_worker"
        )
    
    def _init_precompiled_functions(self):
        """Initialize pre-compiled functions for maximum speed."""
        # Pre-compile regex patterns
        self.word_pattern = re.compile(r'\b\w+\b')
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        self.punctuation_pattern = re.compile(r'[.!?,:;]')
        self.capitalized_pattern = re.compile(r'\b[A-Z][a-z]+\b')
        self.all_caps_pattern = re.compile(r'\b[A-Z]{2,}\b')
        self.contraction_pattern = re.compile(r"\b\w+'\w+\b")
        
        # Pre-compile stop words set
        self.stop_words = frozenset({
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        })
        
        # Pre-compile sentiment models
        self.positive_words = frozenset({
            "good", "great", "excellent", "awesome", "amazing", "wonderful", "fantastic",
            "brilliant", "outstanding", "perfect", "beautiful", "nice", "best", "better",
            "improve", "success", "win", "victory", "love", "happy", "joy"
        })
        
        self.negative_words = frozenset({
            "bad", "terrible", "poor", "awful", "horrible", "hate", "angry", "sad",
            "disappointed", "frustrated", "worst", "worse", "fail", "failure", "problem",
            "issue", "error", "wrong", "broken", "damaged", "ugly"
        })
        
        # Pre-compile topic models
        self.topic_keywords = {
            "technology": frozenset({"computer", "software", "hardware", "internet", "digital", "ai", "machine", "data", "algorithm", "code"}),
            "business": frozenset({"company", "market", "sales", "profit", "revenue", "customer", "product", "service", "management", "strategy"}),
            "science": frozenset({"research", "study", "experiment", "theory", "hypothesis", "analysis", "discovery", "innovation", "method", "result"}),
            "health": frozenset({"medical", "health", "doctor", "patient", "treatment", "disease", "medicine", "hospital", "therapy", "care"}),
            "education": frozenset({"school", "student", "teacher", "learning", "education", "course", "study", "knowledge", "skill", "training"})
        }
    
    def _init_vectorized_operations(self):
        """Initialize vectorized operations for maximum speed."""
        # NumPy arrays for fast operations
        self._init_numpy_arrays()
        
        # Vectorized functions
        self._init_vectorized_functions()
    
    def _init_numpy_arrays(self):
        """Initialize NumPy arrays for fast operations."""
        # Pre-allocate arrays for common operations
        self.word_lengths_array = np.zeros(10000, dtype=np.int32)
        self.similarity_array = np.zeros(1000, dtype=np.float32)
        self.metrics_array = np.zeros(100, dtype=np.float32)
    
    def _init_vectorized_functions(self):
        """Initialize vectorized functions."""
        # Vectorized text processing
        self.vectorized_word_count = np.vectorize(len)
        self.vectorized_similarity = np.vectorize(self._fast_similarity)
        self.vectorized_metrics = np.vectorize(self._fast_metrics)
    
    async def shutdown(self):
        """Shutdown all pools."""
        self.lightning_thread_pool.shutdown(wait=True)
        self.lightning_process_pool.shutdown(wait=True)
        self.lightning_io_pool.shutdown(wait=True)
    
    @staticmethod
    def _fast_similarity(x: float) -> float:
        """Fast similarity calculation."""
        return x * 0.5  # Placeholder for fast calculation
    
    @staticmethod
    def _fast_metrics(x: float) -> float:
        """Fast metrics calculation."""
        return x * 0.1  # Placeholder for fast calculation


# Global analysis engine
_lightning_engine = LightningAnalysisEngine()


@track_performance("lightning_analysis")
@cached(ttl=14400, tags=["analysis", "content", "lightning"])
async def analyze_content_lightning_fast(request: ContentAnalysisRequest) -> ContentAnalysisResponse:
    """Perform lightning-fast content analysis with extreme speed optimizations."""
    start_time = time.perf_counter()
    
    try:
        # Create content hash for caching
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        
        # Pre-validate content
        if not request.content or len(request.content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        # Run analyses in parallel with lightning pools
        tasks = [
            _perform_lightning_basic_analysis(request.content),
            _perform_lightning_sentiment_analysis(request.content),
            _perform_lightning_readability_analysis(request.content),
            _perform_lightning_topic_classification(request.content),
            _perform_lightning_keyword_analysis(request.content),
            _perform_lightning_language_analysis(request.content),
            _perform_lightning_style_analysis(request.content),
            _perform_lightning_complexity_analysis(request.content),
            _perform_lightning_quality_assessment(request.content),
            _perform_lightning_performance_analysis(request.content)
        ]
        
        # Execute with ultra-fast timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=10.0  # Ultra-fast timeout
        )
        
        # Combine results efficiently
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
            "performance": results[9] if not isinstance(results[9], Exception) else {"error": str(results[9])},
            "metadata": {
                "content_hash": content_hash,
                "analysis_version": "4.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "optimization_level": "lightning"
            }
        }
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        # Record metrics
        await record_analysis_metrics("lightning_analysis", True, processing_time)
        
        return ContentAnalysisResponse(
            content=request.content,
            model_version=request.model_version,
            word_count=analysis_results["basic"].get("word_count", 0),
            character_count=analysis_results["basic"].get("character_count", 0),
            analysis_results=analysis_results,
            systems_used={
                "lightning_basic_analysis": True,
                "lightning_sentiment_analysis": True,
                "lightning_readability_analysis": True,
                "lightning_topic_classification": True,
                "lightning_keyword_analysis": True,
                "lightning_language_analysis": True,
                "lightning_style_analysis": True,
                "lightning_complexity_analysis": True,
                "lightning_quality_assessment": True,
                "lightning_performance_analysis": True
            },
            processing_time=processing_time
        )
        
    except asyncio.TimeoutError:
        processing_time = time.perf_counter() - start_time
        await record_analysis_metrics("lightning_analysis", False, processing_time)
        logger.error("Lightning analysis timed out")
        raise HTTPException(status_code=408, detail="Lightning analysis timed out")
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        await record_analysis_metrics("lightning_analysis", False, processing_time)
        logger.error(f"Lightning analysis failed: {e}")
        raise


@ultra_fast
@cached_ultra_fast(ttl=7200, maxsize=2000)
async def _perform_lightning_basic_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast basic text analysis."""
    try:
        # Use pre-compiled regex patterns for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        sentences = [s.strip() for s in _lightning_engine.sentence_pattern.split(content) if s.strip()]
        paragraphs = [p.strip() for p in _lightning_engine.paragraph_pattern.split(content) if p.strip()]
        
        # Calculate metrics efficiently
        word_count = len(words)
        character_count = len(content)
        char_count_no_spaces = len(content.replace(' ', ''))
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        # Statistical analysis with optimized calculations
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_paragraph_length = sentence_count / paragraph_count if paragraph_count > 0 else 0
        
        # Vocabulary analysis with set operations
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Word frequency with Counter (optimized)
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
            "text_density": round(text_density, 3),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning basic analysis failed: {e}")
        return {"error": str(e)}


@cpu_optimized
@cached_ultra_fast(ttl=3600, maxsize=5000)
async def _perform_lightning_sentiment_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast sentiment analysis."""
    try:
        # Use pre-compiled word sets for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        
        # Calculate sentiment scores with optimized set operations
        positive_words_found = words & _lightning_engine.positive_words
        negative_words_found = words & _lightning_engine.negative_words
        
        positive_score = len(positive_words_found)
        negative_score = len(negative_words_found)
        
        # Normalize scores efficiently
        total_sentiment_words = positive_score + negative_score
        if total_sentiment_words > 0:
            sentiment_score = (positive_score - negative_score) / total_sentiment_words
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
        
        # Emotional analysis with optimized lookup
        emotions = _analyze_emotions_lightning(words)
        
        return {
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "confidence": round(confidence, 3),
            "positive_score": positive_score,
            "negative_score": negative_score,
            "positive_words_found": list(positive_words_found)[:10],
            "negative_words_found": list(negative_words_found)[:10],
            "total_sentiment_words": total_sentiment_words,
            "emotions": emotions,
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning sentiment analysis failed: {e}")
        return {"error": str(e)}


@ultra_fast
@cached_ultra_fast(ttl=7200, maxsize=1000)
async def _perform_lightning_readability_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast readability analysis."""
    try:
        # Use optimized readability calculations
        from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index
        
        # Calculate readability scores
        flesch_ease = flesch_reading_ease(content)
        flesch_grade = flesch_kincaid_grade(content)
        gunning_fog_score = gunning_fog(content)
        smog_score = smog_index(content)
        
        # Average readability with optimized calculation
        avg_readability = (flesch_ease + (100 - flesch_grade * 10) + 
                          (100 - gunning_fog_score * 10) + (100 - smog_score * 10)) / 4
        
        # Readability level with optimized lookup
        readability_level, target_audience = _get_readability_level_lightning(avg_readability)
        
        # Text complexity indicators
        words = _lightning_engine.word_pattern.findall(content.lower())
        complex_words = [word for word in words if len(word) > 6 or _count_syllables_lightning(word) > 2]
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
            "complex_words_count": len(complex_words),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning readability analysis failed: {e}")
        return {"error": str(e)}


@cpu_optimized
@cached_ultra_fast(ttl=3600, maxsize=2000)
async def _perform_lightning_topic_classification(content: str) -> Dict[str, Any]:
    """Perform lightning-fast topic classification."""
    try:
        # Use pre-compiled topic model for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        
        # Calculate topic scores with optimized operations
        topic_scores = {}
        for topic, keywords in _lightning_engine.topic_keywords.items():
            score = len(words & keywords)
            topic_scores[topic] = score
        
        # Normalize scores efficiently
        total_score = sum(topic_scores.values())
        if total_score > 0:
            topic_scores = {topic: score / total_score for topic, score in topic_scores.items()}
        
        # Get primary topic with optimized lookup
        primary_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "unknown"
        confidence = topic_scores.get(primary_topic, 0)
        
        return {
            "primary_topic": primary_topic,
            "confidence": round(confidence, 3),
            "topic_scores": {topic: round(score, 3) for topic, score in topic_scores.items()},
            "all_topics": list(topic_scores.keys()),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning topic classification failed: {e}")
        return {"error": str(e)}


@ultra_fast
@cached_ultra_fast(ttl=3600, maxsize=2000)
async def _perform_lightning_keyword_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast keyword analysis."""
    try:
        # Use pre-compiled patterns for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        
        # Filter words efficiently with pre-compiled stop words
        filtered_words = [word for word in words if word not in _lightning_engine.stop_words and len(word) > 2]
        
        # TF-IDF calculation with optimized operations
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)
        
        # Calculate TF-IDF scores efficiently
        tfidf_scores = {}
        for word, freq in word_freq.items():
            tf = freq / total_words
            idf = np.log(total_words / freq) if freq > 0 else 0
            tfidf_scores[word] = tf * idf
        
        # Get top keywords with optimized sorting
        top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Keyword density with optimized calculation
        keyword_density = {word: (freq / total_words) * 100 for word, freq in word_freq.items()}
        
        # N-gram analysis with optimized extraction
        bigrams = _extract_ngrams_lightning(filtered_words, 2)
        trigrams = _extract_ngrams_lightning(filtered_words, 3)
        
        return {
            "top_keywords_tfidf": [(word, round(score, 4)) for word, score in top_keywords],
            "keyword_frequency": dict(word_freq.most_common(20)),
            "keyword_density": {word: round(density, 2) for word, density in keyword_density.items()},
            "bigrams": bigrams[:10],
            "trigrams": trigrams[:10],
            "total_unique_keywords": len(word_freq),
            "keyword_richness": round(len(word_freq) / total_words, 3) if total_words > 0 else 0,
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning keyword analysis failed: {e}")
        return {"error": str(e)}


@ultra_fast
@cached_ultra_fast(ttl=7200, maxsize=1000)
async def _perform_lightning_language_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast language analysis."""
    try:
        # Use pre-compiled language model for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        
        # Pre-compiled language models
        language_models = {
            "english": frozenset({"the", "and", "or", "but", "in", "on", "at", "to", "for", "of"}),
            "spanish": frozenset({"el", "la", "de", "que", "y", "a", "en", "un", "es", "se"}),
            "french": frozenset({"le", "la", "de", "et", "à", "un", "il", "que", "ne", "se"})
        }
        
        # Calculate language probabilities with optimized operations
        language_scores = {}
        for language, word_set in language_models.items():
            score = len(words & word_set)
            language_scores[language] = score
        
        # Get most likely language with optimized lookup
        detected_language = max(language_scores, key=language_scores.get) if language_scores else "unknown"
        confidence = language_scores.get(detected_language, 0)
        
        # Text statistics with optimized calculations
        total_words = len(words)
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # Character analysis with optimized operations
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
            "character_diversity": len(char_freq),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning language analysis failed: {e}")
        return {"error": str(e)}


@ultra_fast
@cached_ultra_fast(ttl=3600, maxsize=2000)
async def _perform_lightning_style_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast style analysis."""
    try:
        # Use pre-compiled patterns for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        sentences = [s.strip() for s in _lightning_engine.sentence_pattern.split(content) if s.strip()]
        
        # Sentence structure analysis with optimized calculations
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        sentence_variation = max(sentence_lengths) - min(sentence_lengths) if sentence_lengths else 0
        
        # Punctuation analysis with optimized operations
        punctuation_count = len(_lightning_engine.punctuation_pattern.findall(content))
        punctuation_density = punctuation_count / len(content) if content else 0
        
        # Capitalization analysis with optimized operations
        capitalized_words = len(_lightning_engine.capitalized_pattern.findall(content))
        all_caps_words = len(_lightning_engine.all_caps_pattern.findall(content))
        
        # Contraction analysis
        contractions = len(_lightning_engine.contraction_pattern.findall(content))
        
        # Passive voice detection with optimized lookup
        passive_indicators = frozenset({"was", "were", "been", "being", "get", "got", "getting"})
        passive_count = len(words & passive_indicators)
        passive_ratio = passive_count / len(words) if words else 0
        
        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "sentence_variation": sentence_variation,
            "punctuation_density": round(punctuation_density, 4),
            "capitalized_words": capitalized_words,
            "all_caps_words": all_caps_words,
            "contractions": contractions,
            "passive_voice_ratio": round(passive_ratio, 3),
            "writing_style": _classify_writing_style_lightning(avg_sentence_length, passive_ratio, punctuation_density),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning style analysis failed: {e}")
        return {"error": str(e)}


@cpu_optimized
@cached_ultra_fast(ttl=3600, maxsize=2000)
async def _perform_lightning_complexity_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast complexity analysis."""
    try:
        # Use pre-compiled patterns for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        sentences = [s.strip() for s in _lightning_engine.sentence_pattern.split(content) if s.strip()]
        
        # Lexical complexity with optimized calculations
        unique_words = len(set(words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        # Syntactic complexity
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Morphological complexity with optimized operations
        complex_words = [word for word in words if len(word) > 6 or _count_syllables_lightning(word) > 2]
        morphological_complexity = len(complex_words) / len(words) if words else 0
        
        # Semantic complexity with optimized calculation
        semantic_complexity = _calculate_semantic_complexity_lightning(words)
        
        # Overall complexity score with optimized calculation
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
            "complexity_level": _get_complexity_level_lightning(complexity_score),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning complexity analysis failed: {e}")
        return {"error": str(e)}


@ultra_fast
@cached_ultra_fast(ttl=3600, maxsize=2000)
async def _perform_lightning_quality_assessment(content: str) -> Dict[str, Any]:
    """Perform lightning-fast quality assessment."""
    try:
        # Use pre-compiled patterns for maximum speed
        words = _lightning_engine.word_pattern.findall(content.lower())
        sentences = [s.strip() for s in _lightning_engine.sentence_pattern.split(content) if s.strip()]
        
        # Quality indicators with optimized calculations
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Readability score with optimized calculation
        try:
            from textstat import flesch_reading_ease
            readability = flesch_reading_ease(content)
        except:
            readability = 50  # Default middle score
        
        # Quality scoring with optimized calculations
        length_score = min(word_count / 100, 1)  # Prefer 100+ words
        readability_score = readability / 100
        structure_score = min(sentence_count / 5, 1)  # Prefer 5+ sentences
        
        # Overall quality score with optimized calculation
        quality_score = (length_score * 0.3 + readability_score * 0.4 + structure_score * 0.3)
        
        # Quality level with optimized lookup
        quality_level = _get_quality_level_lightning(quality_score)
        
        # Recommendations with optimized generation
        recommendations = _generate_recommendations_lightning(word_count, readability, sentence_count)
        
        return {
            "quality_score": round(quality_score, 3),
            "quality_level": quality_level,
            "length_score": round(length_score, 3),
            "readability_score": round(readability_score, 3),
            "structure_score": round(structure_score, 3),
            "recommendations": recommendations,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning quality assessment failed: {e}")
        return {"error": str(e)}


@ultra_fast
@cached_ultra_fast(ttl=3600, maxsize=2000)
async def _perform_lightning_performance_analysis(content: str) -> Dict[str, Any]:
    """Perform lightning-fast performance analysis."""
    try:
        # Performance metrics with optimized calculations
        content_size = len(content)
        word_count = len(content.split())
        
        # Processing efficiency metrics
        processing_efficiency = min(word_count / 1000, 1.0)  # Normalize to 1000 words
        memory_efficiency = min(content_size / 10000, 1.0)  # Normalize to 10KB
        
        # Optimization recommendations
        optimizations = []
        if content_size > 50000:
            optimizations.append("Consider content chunking for large texts")
        if word_count > 10000:
            optimizations.append("Consider parallel processing for large documents")
        if content_size < 100:
            optimizations.append("Content may be too short for comprehensive analysis")
        
        return {
            "content_size": content_size,
            "word_count": word_count,
            "processing_efficiency": round(processing_efficiency, 3),
            "memory_efficiency": round(memory_efficiency, 3),
            "optimization_recommendations": optimizations,
            "performance_level": "lightning_fast",
            "optimization_level": "lightning"
        }
        
    except Exception as e:
        logger.error(f"Lightning performance analysis failed: {e}")
        return {"error": str(e)}


# Lightning-fast helper functions
@lru_cache(maxsize=10000)
def _analyze_emotions_lightning(words: tuple) -> Dict[str, int]:
    """Analyze emotional content with caching."""
    emotions = {
        "joy": frozenset({"happy", "joy", "excited", "cheerful", "delighted"}),
        "sadness": frozenset({"sad", "depressed", "melancholy", "gloomy", "sorrowful"}),
        "anger": frozenset({"angry", "mad", "furious", "irritated", "annoyed"}),
        "fear": frozenset({"afraid", "scared", "terrified", "worried", "anxious"}),
        "surprise": frozenset({"surprised", "amazed", "shocked", "astonished", "stunned"})
    }
    
    emotion_counts = {}
    for emotion, keywords in emotions.items():
        count = len(set(words) & keywords)
        emotion_counts[emotion] = count
    
    return emotion_counts


@lru_cache(maxsize=10000)
def _extract_ngrams_lightning(words: tuple, n: int) -> List[Tuple[str, int]]:
    """Extract n-grams with caching."""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams).most_common(10)


@lru_cache(maxsize=10000)
def _count_syllables_lightning(word: str) -> int:
    """Count syllables with caching."""
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


@lru_cache(maxsize=1000)
def _get_readability_level_lightning(avg_readability: float) -> Tuple[str, str]:
    """Get readability level with caching."""
    if avg_readability >= 80:
        return "Very Easy", "Elementary school"
    elif avg_readability >= 60:
        return "Easy", "Middle school"
    elif avg_readability >= 40:
        return "Moderate", "High school"
    elif avg_readability >= 20:
        return "Difficult", "College"
    else:
        return "Very Difficult", "Graduate level"


@lru_cache(maxsize=1000)
def _classify_writing_style_lightning(avg_sentence_length: float, passive_ratio: float, punctuation_density: float) -> str:
    """Classify writing style with caching."""
    if avg_sentence_length > 20 and passive_ratio > 0.1:
        return "Academic"
    elif avg_sentence_length < 10 and punctuation_density > 0.05:
        return "Conversational"
    elif avg_sentence_length > 15:
        return "Formal"
    else:
        return "Casual"


@lru_cache(maxsize=1000)
def _get_complexity_level_lightning(complexity_score: float) -> str:
    """Get complexity level with caching."""
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


@lru_cache(maxsize=1000)
def _get_quality_level_lightning(quality_score: float) -> str:
    """Get quality level with caching."""
    if quality_score >= 0.8:
        return "Excellent"
    elif quality_score >= 0.6:
        return "Good"
    elif quality_score >= 0.4:
        return "Fair"
    else:
        return "Poor"


@lru_cache(maxsize=1000)
def _generate_recommendations_lightning(word_count: int, readability: float, sentence_count: int) -> List[str]:
    """Generate recommendations with caching."""
    recommendations = []
    if word_count < 50:
        recommendations.append("Consider expanding content for better analysis")
    if readability < 30:
        recommendations.append("Improve readability for broader audience")
    if sentence_count < 3:
        recommendations.append("Add more sentences for better structure")
    return recommendations


@lru_cache(maxsize=1000)
def _calculate_semantic_complexity_lightning(words: tuple) -> float:
    """Calculate semantic complexity with caching."""
    complex_words = [word for word in words if len(word) > 6]
    return len(complex_words) / len(words) if words else 0


