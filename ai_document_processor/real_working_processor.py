"""
Real Working AI Document Processor
A completely functional AI document processing system using only real, working technologies
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import time
import os

# Real, working libraries that actually exist
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
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealWorkingProcessor:
    """Real Working AI Document Processor using only functional technologies"""
    
    def __init__(self):
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.classifier = None
        self.summarizer = None
        self.initialized = False
        
        # Real performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0,
            "start_time": time.time()
        }
        
    async def initialize(self):
        """Initialize the real working processor"""
        try:
            logger.info("Initializing real working AI processor...")
            
            # Initialize NLTK if available
            if NLTK_AVAILABLE:
                try:
                    nltk.download('vader_lexicon', quiet=True)
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    logger.info("NLTK sentiment analyzer initialized")
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
                    logger.info("Transformers models loaded successfully")
                except Exception as e:
                    logger.warning(f"Transformers initialization failed: {e}")
            
            self.initialized = True
            logger.info("Real working AI processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI processor: {e}")
            raise
    
    async def process_text(self, text: str, task: str = "analyze") -> Dict[str, Any]:
        """Process text with real working AI capabilities"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing text with task: {task}")
            
            result = {
                "text_id": self._generate_text_id(text),
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "status": "success"
            }
            
            # Basic text analysis (always works)
            result.update(self._analyze_text_basic(text))
            
            # NLP analysis if available
            if self.nlp_model:
                result.update(await self._analyze_text_nlp(text))
            
            # Sentiment analysis if available
            if self.sentiment_analyzer:
                result.update(self._analyze_sentiment(text))
            
            # Classification if available and requested
            if self.classifier and task == "classify":
                result.update(await self._classify_text(text))
            
            # Summarization if available and requested
            if self.summarizer and task == "summarize":
                result.update(await self._summarize_text(text))
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time, True)
            
            result["processing_time"] = processing_time
            result["models_used"] = self._get_models_used()
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            logger.error(f"Error processing text: {e}")
            return {
                "error": str(e),
                "status": "error",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_text_basic(self, text: str) -> Dict[str, Any]:
        """Basic text analysis that always works"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Reading time estimate (200 words per minute)
        reading_time = word_count / 200
        
        # Basic readability score
        readability_score = self._calculate_readability_score(word_count, sentence_count, avg_word_length)
        
        return {
            "basic_analysis": {
                "character_count": char_count,
                "character_count_no_spaces": char_count_no_spaces,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_word_length": round(avg_word_length, 2),
                "average_sentence_length": round(avg_sentence_length, 2),
                "reading_time_minutes": round(reading_time, 2),
                "readability_score": round(readability_score, 2)
            }
        }
    
    def _calculate_readability_score(self, word_count: int, sentence_count: int, avg_word_length: float) -> float:
        """Calculate a basic readability score"""
        if word_count == 0 or sentence_count == 0:
            return 0
        
        # Simple readability formula
        avg_sentence_length = word_count / sentence_count
        readability = 100 - (avg_sentence_length * 0.5) - (avg_word_length * 2)
        return max(0, min(100, readability))
    
    async def _analyze_text_nlp(self, text: str) -> Dict[str, Any]:
        """Advanced NLP analysis using spaCy"""
        if not self.nlp_model:
            return {}
        
        doc = self.nlp_model(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Part-of-speech distribution
        pos_tags = {}
        for token in doc:
            if token.pos_ not in pos_tags:
                pos_tags[token.pos_] = 0
            pos_tags[token.pos_] += 1
        
        # Noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        return {
            "nlp_analysis": {
                "entities": entities,
                "pos_distribution": pos_tags,
                "noun_phrases": noun_phrases,
                "sentences": [sent.text for sent in doc.sents]
            }
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis using NLTK"""
        if not self.sentiment_analyzer:
            return {}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine sentiment label
        compound_score = scores['compound']
        if compound_score >= 0.05:
            sentiment_label = "positive"
        elif compound_score <= -0.05:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "sentiment_analysis": {
                "compound_score": round(compound_score, 3),
                "positive_score": round(scores['pos'], 3),
                "negative_score": round(scores['neg'], 3),
                "neutral_score": round(scores['neu'], 3),
                "sentiment_label": sentiment_label
            }
        }
    
    async def _classify_text(self, text: str) -> Dict[str, Any]:
        """Text classification using transformers"""
        if not self.classifier:
            return {}
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.classifier(text)
            
            return {
                "classification": {
                    "label": result[0]['label'],
                    "confidence": round(result[0]['score'], 3)
                }
            }
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return {"classification": {"error": str(e)}}
    
    async def _summarize_text(self, text: str) -> Dict[str, Any]:
        """Text summarization using transformers"""
        if not self.summarizer:
            return {}
        
        try:
            # Truncate text if too long
            max_length = 1024
            if len(text) > max_length:
                text = text[:max_length]
            
            summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)
            
            return {
                "summary": {
                    "text": summary[0]['summary_text'],
                    "original_length": len(text),
                    "summary_length": len(summary[0]['summary_text'])
                }
            }
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return {"summary": {"error": str(e)}}
    
    def extract_keywords(self, text: str, top_n: int = 10) -> Dict[str, Any]:
        """Extract keywords from text"""
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
        
        # Remove duplicates and get top N
        unique_keywords = list(set(keywords))
        top_keywords = unique_keywords[:top_n]
        
        return {
            "keywords": top_keywords,
            "total_keywords": len(unique_keywords),
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of the text using simple heuristics"""
        # Simple language detection based on common words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le']
        french_words = ['le', 'la', 'de', 'et', 'à', 'un', 'il', 'que', 'ne', 'se', 'ce', 'pas', 'pour', 'par']
        
        text_lower = text.lower()
        
        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        french_count = sum(1 for word in french_words if word in text_lower)
        
        scores = {
            'english': english_count,
            'spanish': spanish_count,
            'french': french_count
        }
        
        detected_language = max(scores, key=scores.get)
        confidence = scores[detected_language] / max(1, sum(scores.values()))
        
        return {
            "language": detected_language,
            "confidence": round(confidence, 3),
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_text_id(self, text: str) -> str:
        """Generate unique text ID"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _get_models_used(self) -> Dict[str, bool]:
        """Get which models are being used"""
        return {
            "spacy": self.nlp_model is not None,
            "nltk": self.sentiment_analyzer is not None,
            "transformers_classifier": self.classifier is not None,
            "transformers_summarizer": self.summarizer is not None
        }
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update average processing time
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_processing_time"]
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.stats["average_processing_time"] = round(new_avg, 3)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "success_rate": round(self.stats["successful_requests"] / max(1, self.stats["total_requests"]) * 100, 2),
            "models_loaded": self._get_models_used()
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities"""
        return {
            "basic_analysis": True,
            "nlp_analysis": self.nlp_model is not None,
            "sentiment_analysis": self.sentiment_analyzer is not None,
            "text_classification": self.classifier is not None,
            "text_summarization": self.summarizer is not None,
            "keyword_extraction": self.nlp_model is not None,
            "language_detection": True,
            "models_loaded": self._get_models_used()
        }

# Global instance
real_working_processor = RealWorkingProcessor()

async def initialize_real_working_processor():
    """Initialize the real working processor"""
    try:
        await real_working_processor.initialize()
        logger.info("Real working processor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing real working processor: {e}")
        raise













