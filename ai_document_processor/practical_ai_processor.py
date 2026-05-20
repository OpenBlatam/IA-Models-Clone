"""
Practical AI Document Processor
A real, working AI document processing system using only proven technologies
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import os

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

class PracticalAIProcessor:
    """Practical AI Document Processor using only real, working technologies"""
    
    def __init__(self):
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.classifier = None
        self.summarizer = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the practical AI processor"""
        try:
            logger.info("Initializing practical AI processor...")
            
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
            logger.info("Practical AI processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI processor: {e}")
            raise
    
    async def process_text(self, text: str, task: str = "analyze") -> Dict[str, Any]:
        """Process text with practical AI capabilities"""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Processing text with task: {task}")
            
            results = {
                "text_id": self._generate_text_id(text),
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "status": "success"
            }
            
            # Basic text analysis (always available)
            results.update(self._analyze_text_basic(text))
            
            # NLP analysis if available
            if self.nlp_model:
                results.update(await self._analyze_text_nlp(text))
            
            # Sentiment analysis if available
            if self.sentiment_analyzer:
                results.update(self._analyze_sentiment(text))
            
            # Classification if available
            if self.classifier and task == "classify":
                results.update(await self._classify_text(text))
            
            # Summarization if available
            if self.summarizer and task == "summarize":
                results.update(await self._summarize_text(text))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_text_id(self, text: str) -> str:
        """Generate unique text ID"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _analyze_text_basic(self, text: str) -> Dict[str, Any]:
        """Basic text analysis without external libraries"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Remove empty sentences
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
        
        return {
            "basic_analysis": {
                "character_count": char_count,
                "character_count_no_spaces": char_count_no_spaces,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_word_length": round(avg_word_length, 2),
                "average_sentence_length": round(avg_sentence_length, 2),
                "reading_time_minutes": round(reading_time, 2)
            }
        }
    
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
            "models_loaded": {
                "spacy": self.nlp_model is not None,
                "nltk": self.sentiment_analyzer is not None,
                "transformers_classifier": self.classifier is not None,
                "transformers_summarizer": self.summarizer is not None
            }
        }

# Global instance
practical_ai_processor = PracticalAIProcessor()

async def initialize_practical_ai_processor():
    """Initialize the practical AI processor"""
    try:
        await practical_ai_processor.initialize()
        logger.info("Practical AI processor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing practical AI processor: {e}")
        raise













