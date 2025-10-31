"""
Enhanced Practical AI Document Processor
Real, working AI document processing with advanced features
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import os
import time
from pathlib import Path

# Real, working libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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

logger = logging.getLogger(__name__)

class EnhancedAIProcessor:
    """Enhanced AI Document Processor with real, working features"""
    
    def __init__(self):
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.classifier = None
        self.summarizer = None
        self.qa_pipeline = None
        self.lemmatizer = None
        self.vectorizer = None
        self.redis_client = None
        self.initialized = False
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0,
            "error_count": 0
        }
        
        # Cache for results
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize the enhanced AI processor"""
        try:
            logger.info("Initializing enhanced AI processor...")
            
            # Initialize NLTK if available
            if NLTK_AVAILABLE:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('wordnet', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    self.lemmatizer = WordNetLemmatizer()
                    logger.info("NLTK components initialized")
                except Exception as e:
                    logger.warning(f"NLTK initialization failed: {e}")
            
            # Initialize spaCy if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy model not found, using basic processing")
            
            # Initialize transformers if available
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.classifier = pipeline("text-classification", 
                                             model="distilbert-base-uncased-finetuned-sst-2-english")
                    self.summarizer = pipeline("summarization", 
                                             model="facebook/bart-large-cnn")
                    self.qa_pipeline = pipeline("question-answering", 
                                              model="distilbert-base-cased-distilled-squad")
                    logger.info("Transformers models loaded successfully")
                except Exception as e:
                    logger.warning(f"Transformers initialization failed: {e}")
            
            # Initialize scikit-learn if available
            if SKLEARN_AVAILABLE:
                try:
                    self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    logger.info("TF-IDF vectorizer initialized")
                except Exception as e:
                    logger.warning(f"TF-IDF vectorizer initialization failed: {e}")
            
            # Initialize Redis if available
            if REDIS_AVAILABLE:
                try:
                    self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                    self.redis_client.ping()
                    logger.info("Redis connected successfully")
                except Exception as e:
                    logger.warning(f"Redis not available: {e}")
                    self.redis_client = None
            
            self.initialized = True
            logger.info("Enhanced AI processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced AI processor: {e}")
            raise
    
    async def process_text_enhanced(self, text: str, task: str = "analyze", 
                                  use_cache: bool = True) -> Dict[str, Any]:
        """Process text with enhanced AI capabilities"""
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(text, task)
            
            # Check cache first
            if use_cache:
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
                self.stats["cache_misses"] += 1
            
            # Process text
            result = await self._process_text_core(text, task)
            
            # Add enhanced features
            if task == "analyze":
                result.update(await self._add_enhanced_analysis(text))
            
            # Add similarity analysis if requested
            if task == "similarity" and self.vectorizer:
                result.update(await self._analyze_similarity(text))
            
            # Add topic analysis if requested
            if task == "topics" and self.vectorizer:
                result.update(await self._analyze_topics(text))
            
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
            self.stats["error_count"] += 1
            logger.error(f"Error in enhanced processing: {e}")
            return {
                "error": str(e),
                "status": "error",
                "processing_time": time.time() - start_time
            }
    
    async def _process_text_core(self, text: str, task: str) -> Dict[str, Any]:
        """Core text processing functionality"""
        result = {
            "text_id": self._generate_text_id(text),
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "status": "success"
        }
        
        # Basic text analysis (always available)
        result.update(self._analyze_text_basic(text))
        
        # NLP analysis if available
        if self.nlp_model:
            result.update(await self._analyze_text_nlp(text))
        
        # Sentiment analysis if available
        if self.sentiment_analyzer:
            result.update(self._analyze_sentiment(text))
        
        # Classification if available
        if self.classifier and task == "classify":
            result.update(await self._classify_text(text))
        
        # Summarization if available
        if self.summarizer and task == "summarize":
            result.update(await self._summarize_text(text))
        
        # Question answering if available
        if self.qa_pipeline and task == "qa":
            result.update(await self._prepare_qa_context(text))
        
        return result
    
    async def _add_enhanced_analysis(self, text: str) -> Dict[str, Any]:
        """Add enhanced analysis features"""
        enhanced_features = {}
        
        # Text complexity analysis
        enhanced_features["complexity"] = self._analyze_complexity(text)
        
        # Readability analysis
        enhanced_features["readability"] = self._analyze_readability(text)
        
        # Language patterns
        enhanced_features["language_patterns"] = self._analyze_language_patterns(text)
        
        # Text quality metrics
        enhanced_features["quality_metrics"] = self._analyze_quality_metrics(text)
        
        # Keyword analysis
        enhanced_features["keyword_analysis"] = self._analyze_keywords_advanced(text)
        
        return {"enhanced_analysis": enhanced_features}
    
    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not words or not sentences:
            return {"complexity_score": 0, "level": "Unknown"}
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable count estimation
        syllable_count = sum([self._count_syllables(word) for word in words])
        avg_syllables_per_word = syllable_count / len(words)
        
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
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
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
    
    def _analyze_language_patterns(self, text: str) -> Dict[str, Any]:
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
            "avg_word_length": round(sum(word_lengths) / len(word_lengths), 2) if word_lengths else 0,
            "unique_words": len(set(words)),
            "total_words": len(words)
        }
    
    def _analyze_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Analyze text quality metrics"""
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
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
    
    def _analyze_keywords_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced keyword analysis"""
        if not self.nlp_model:
            return {"error": "NLP model not available"}
        
        doc = self.nlp_model(text)
        
        # Extract keywords from entities and noun phrases
        keywords = []
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                keywords.append(ent.text)
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to 3 words max
                keywords.append(chunk.text)
        
        # Remove duplicates
        unique_keywords = list(set(keywords))
        
        return {
            "keywords": unique_keywords[:10],  # Top 10
            "total_keywords": len(unique_keywords),
            "keyword_density": len(unique_keywords) / len(text.split()) if text.split() else 0
        }
    
    async def _analyze_similarity(self, text: str) -> Dict[str, Any]:
        """Analyze text similarity using TF-IDF"""
        if not self.vectorizer:
            return {"error": "TF-IDF vectorizer not available"}
        
        try:
            # Fit and transform text
            tfidf_matrix = self.vectorizer.fit_transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top terms
            scores = tfidf_matrix.toarray()[0]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
            top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return {
                "similarity_analysis": {
                    "top_terms": top_terms,
                    "num_features": len(feature_names),
                    "max_tfidf_score": round(float(max(scores)), 3)
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
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
            top_terms = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return {
                "topic_analysis": {
                    "top_terms": top_terms,
                    "num_features": len(feature_names),
                    "max_tfidf_score": round(float(max(scores)), 3)
                }
            }
            
        except Exception as e:
            return {"error": f"Topic analysis failed: {str(e)}"}
    
    def _generate_text_id(self, text: str) -> str:
        """Generate unique text ID"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
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
        self.stats["total_requests"] += 1
        
        # Update average processing time
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_processing_time"]
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.stats["average_processing_time"] = round(new_avg, 3)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "stats": self.stats.copy(),
            "cache_size": len(self.cache),
            "redis_available": self.redis_client is not None,
            "models_loaded": {
                "spacy": self.nlp_model is not None,
                "nltk": self.sentiment_analyzer is not None,
                "transformers_classifier": self.classifier is not None,
                "transformers_summarizer": self.summarizer is not None,
                "transformers_qa": self.qa_pipeline is not None,
                "tfidf_vectorizer": self.vectorizer is not None
            }
        }
    
    async def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        
        if self.redis_client:
            try:
                keys = self.redis_client.keys("ai_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")
        
        logger.info("Cache cleared successfully")

# Global instance
enhanced_ai_processor = EnhancedAIProcessor()

async def initialize_enhanced_ai_processor():
    """Initialize the enhanced AI processor"""
    try:
        await enhanced_ai_processor.initialize()
        logger.info("Enhanced AI processor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing enhanced AI processor: {e}")
        raise













