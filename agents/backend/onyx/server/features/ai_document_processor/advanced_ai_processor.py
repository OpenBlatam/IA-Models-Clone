"""
Advanced Real AI Document Processor
Enhanced AI capabilities with caching, monitoring, and advanced features
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import time

# Advanced NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from real_ai_processor import RealAIDocumentProcessor

logger = logging.getLogger(__name__)

class AdvancedAIProcessor(RealAIDocumentProcessor):
    """Advanced AI Document Processor with enhanced capabilities"""
    
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.redis_client = None
        self.sentence_model = None
        self.vectorizer = None
        self.processing_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0,
            "error_count": 0
        }
        
    async def initialize(self):
        """Initialize advanced AI processor"""
        await super().initialize()
        
        # Initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis connected successfully")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.redis_client = None
        
        # Initialize sentence transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded")
            except Exception as e:
                logger.warning(f"Sentence transformer not available: {e}")
        
        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            logger.info("TF-IDF vectorizer initialized")
        
        logger.info("Advanced AI processor initialized successfully")
    
    async def process_document_advanced(self, document_text: str, task: str = "analyze", 
                                       use_cache: bool = True) -> Dict[str, Any]:
        """Process document with advanced features"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(document_text, task)
            
            # Check cache first
            if use_cache:
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    self.processing_stats["cache_hits"] += 1
                    return cached_result
                self.processing_stats["cache_misses"] += 1
            
            # Process document
            result = await self.process_document(document_text, task)
            
            # Add advanced features
            if task == "analyze":
                result.update(await self._add_advanced_analysis(document_text))
            
            # Add similarity analysis if requested
            if task == "similarity" and self.sentence_model:
                result.update(await self._analyze_similarity(document_text))
            
            # Add topic modeling if requested
            if task == "topics" and self.vectorizer:
                result.update(await self._analyze_topics(document_text))
            
            # Cache result
            if use_cache:
                await self._save_to_cache(cache_key, result)
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            result["processing_time"] = processing_time
            result["cache_used"] = use_cache
            
            return result
            
        except Exception as e:
            self.processing_stats["error_count"] += 1
            logger.error(f"Error in advanced processing: {e}")
            return {
                "error": str(e),
                "status": "error",
                "processing_time": time.time() - start_time
            }
    
    async def _add_advanced_analysis(self, text: str) -> Dict[str, Any]:
        """Add advanced analysis features"""
        advanced_features = {}
        
        # Text complexity analysis
        advanced_features["complexity"] = await self._analyze_complexity(text)
        
        # Readability analysis
        advanced_features["readability"] = await self._analyze_readability(text)
        
        # Language patterns
        advanced_features["language_patterns"] = await self._analyze_language_patterns(text)
        
        # Text quality metrics
        advanced_features["quality_metrics"] = await self._analyze_quality_metrics(text)
        
        return {"advanced_analysis": advanced_features}
    
    async def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity"""
        words = text.split()
        sentences = text.split('.')
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Average sentence length
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
        
        # Syllable count estimation
        syllable_count = sum([self._count_syllables(word) for word in words])
        avg_syllables_per_word = syllable_count / len(words) if words else 0
        
        # Complexity score (0-100)
        complexity_score = min(100, (avg_word_length * 2 + avg_sentence_length * 0.5 + avg_syllables_per_word * 10))
        
        return {
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "complexity_score": round(complexity_score, 2),
            "complexity_level": self._get_complexity_level(complexity_score)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word"""
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
    
    def _get_complexity_level(self, score: float) -> str:
        """Get complexity level based on score"""
        if score < 30:
            return "Simple"
        elif score < 60:
            return "Moderate"
        elif score < 80:
            return "Complex"
        else:
            return "Very Complex"
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return {"readability_score": 0, "level": "Unknown"}
        
        # Flesch Reading Ease Score
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = sum([self._count_syllables(word) for word in words]) / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        return {
            "flesch_score": round(flesch_score, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            "readability_level": self._get_readability_level(flesch_score)
        }
    
    def _get_readability_level(self, score: float) -> str:
        """Get readability level based on Flesch score"""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    async def _analyze_language_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze language patterns"""
        words = text.lower().split()
        
        # Word frequency
        word_freq = {}
        for word in words:
            word = word.strip('.,!?;:"()[]{}')
            if word:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Most common words
        most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Vocabulary diversity (unique words / total words)
        vocabulary_diversity = len(set(words)) / len(words) if words else 0
        
        # Word length distribution
        word_lengths = [len(word) for word in words]
        
        return {
            "vocabulary_diversity": round(vocabulary_diversity, 3),
            "most_common_words": most_common,
            "avg_word_length": round(np.mean(word_lengths), 2),
            "word_length_std": round(np.std(word_lengths), 2),
            "unique_words": len(set(words)),
            "total_words": len(words)
        }
    
    async def _analyze_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Analyze text quality metrics"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        # Text density (words per sentence)
        text_density = len(words) / len(sentences) if sentences else 0
        
        # Repetition analysis
        word_freq = {}
        for word in words:
            word = word.lower().strip('.,!?;:"()[]{}')
            if word:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate repetition score
        repetition_score = sum([count for count in word_freq.values() if count > 1]) / len(words) if words else 0
        
        # Paragraph analysis
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        avg_paragraph_length = len(words) / len(paragraphs) if paragraphs else 0
        
        return {
            "text_density": round(text_density, 2),
            "repetition_score": round(repetition_score, 3),
            "avg_paragraph_length": round(avg_paragraph_length, 2),
            "num_paragraphs": len(paragraphs),
            "quality_score": round((1 - repetition_score) * 100, 2)
        }
    
    async def _analyze_similarity(self, text: str) -> Dict[str, Any]:
        """Analyze text similarity using embeddings"""
        if not self.sentence_model:
            return {"error": "Sentence transformer not available"}
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([text])
            
            # Calculate similarity with common texts
            common_texts = [
                "This is a business document with formal language.",
                "This is a casual conversation between friends.",
                "This is a technical manual with specific instructions.",
                "This is a creative story with descriptive language.",
                "This is a news article with factual information."
            ]
            
            common_embeddings = self.sentence_model.encode(common_texts)
            similarities = cosine_similarity(embeddings, common_embeddings)[0]
            
            # Find most similar text type
            most_similar_idx = np.argmax(similarities)
            most_similar_text = common_texts[most_similar_idx]
            similarity_score = similarities[most_similar_idx]
            
            return {
                "similarity_analysis": {
                    "most_similar_type": most_similar_text,
                    "similarity_score": round(float(similarity_score), 3),
                    "all_similarities": [round(float(sim), 3) for sim in similarities]
                }
            }
            
        except Exception as e:
            return {"error": f"Similarity analysis failed: {str(e)}"}
    
    async def _analyze_topics(self, text: str) -> Dict[str, Any]:
        """Analyze topics using TF-IDF"""
        if not self.vectorizer:
            return {"error": "TF-IDF vectorizer not available"}
        
        try:
            # Fit and transform text
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top terms
            scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(scores)[-10:][::-1]
            top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return {
                "topic_analysis": {
                    "top_terms": top_terms,
                    "num_features": len(feature_names),
                    "max_tfidf_score": round(float(np.max(scores)), 3)
                }
            }
            
        except Exception as e:
            return {"error": f"Topic analysis failed: {str(e)}"}
    
    def _generate_cache_key(self, text: str, task: str) -> str:
        """Generate cache key for text and task"""
        content = f"{text}_{task}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"ai_cache:{cache_key}")
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to memory cache
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if datetime.now() < cached_item["expires"]:
                return cached_item["data"]
            else:
                del self.cache[cache_key]
        
        return None
    
    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save result to cache"""
        expires = datetime.now() + timedelta(seconds=self.cache_ttl)
        
        # Try Redis first
        if self.redis_client:
            try:
                cache_data = json.dumps(data)
                self.redis_client.setex(f"ai_cache:{cache_key}", self.cache_ttl, cache_data)
                return
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fallback to memory cache
        self.cache[cache_key] = {
            "data": data,
            "expires": expires
        }
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["total_requests"] += 1
        
        # Update average processing time
        total_requests = self.processing_stats["total_requests"]
        current_avg = self.processing_stats["average_processing_time"]
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.processing_stats["average_processing_time"] = round(new_avg, 3)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "stats": self.processing_stats.copy(),
            "cache_size": len(self.cache),
            "redis_available": self.redis_client is not None,
            "sentence_transformer_available": self.sentence_model is not None,
            "vectorizer_available": self.vectorizer is not None
        }
    
    async def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        
        if self.redis_client:
            try:
                # Clear all cache keys
                keys = self.redis_client.keys("ai_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")
        
        logger.info("Cache cleared successfully")

# Global instance
advanced_ai_processor = AdvancedAIProcessor()

async def initialize_advanced_ai_processor():
    """Initialize the advanced AI processor"""
    try:
        await advanced_ai_processor.initialize()
        logger.info("Advanced AI processor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing advanced AI processor: {e}")
        raise













