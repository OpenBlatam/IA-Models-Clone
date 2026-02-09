"""
Real AI Document Processor
A practical, functional AI document processing system using real technologies
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
from pathlib import Path
import hashlib

# Real NLP libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealAIDocumentProcessor:
    """Real AI Document Processor using actual technologies"""
    
    def __init__(self):
        self.nlp_model = None
        self.sentiment_analyzer = None
        self.classifier = None
        self.summarizer = None
        self.qa_pipeline = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize real AI models"""
        try:
            logger.info("Initializing real AI document processor...")
            
            # Initialize spaCy if available
            if SPACY_AVAILABLE:
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy model not found, using basic processing")
            
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
            
            self.initialized = True
            logger.info("Real AI document processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI processor: {e}")
            raise
    
    async def process_document(self, document_text: str, task: str = "analyze") -> Dict[str, Any]:
        """Process document with real AI capabilities"""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Processing document with task: {task}")
            
            results = {
                "document_id": self._generate_document_id(document_text),
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "status": "success"
            }
            
            # Basic text analysis
            results.update(await self._analyze_text_basic(document_text))
            
            # NLP analysis if available
            if self.nlp_model:
                results.update(await self._analyze_text_nlp(document_text))
            
            # Sentiment analysis if available
            if self.sentiment_analyzer:
                results.update(await self._analyze_sentiment(document_text))
            
            # Classification if available
            if self.classifier:
                results.update(await self._classify_text(document_text))
            
            # Summarization if requested
            if task == "summarize" and self.summarizer:
                results.update(await self._summarize_text(document_text))
            
            # Question answering if requested
            if task == "qa" and self.qa_pipeline:
                results.update(await self._prepare_qa_context(document_text))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_document_id(self, text: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    async def _analyze_text_basic(self, text: str) -> Dict[str, Any]:
        """Basic text analysis without external libraries"""
        return {
            "basic_analysis": {
                "character_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(re.split(r'[.!?]+', text)),
                "paragraph_count": len(text.split('\n\n')),
                "average_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
                "reading_time_estimate": len(text.split()) / 200  # words per minute
            }
        }
    
    async def _analyze_text_nlp(self, text: str) -> Dict[str, Any]:
        """Advanced NLP analysis using spaCy"""
        if not self.nlp_model:
            return {}
        
        doc = self.nlp_model(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        pos_tags = {}
        for token in doc:
            if token.pos_ not in pos_tags:
                pos_tags[token.pos_] = 0
            pos_tags[token.pos_] += 1
        
        return {
            "nlp_analysis": {
                "entities": entities,
                "pos_distribution": pos_tags,
                "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
                "sentences": [sent.text for sent in doc.sents]
            }
        }
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Sentiment analysis using NLTK"""
        if not self.sentiment_analyzer:
            return {}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        return {
            "sentiment_analysis": {
                "compound_score": scores['compound'],
                "positive_score": scores['pos'],
                "negative_score": scores['neg'],
                "neutral_score": scores['neu'],
                "sentiment_label": self._get_sentiment_label(scores['compound'])
            }
        }
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert compound score to sentiment label"""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    async def _classify_text(self, text: str) -> Dict[str, Any]:
        """Text classification using transformers"""
        if not self.classifier:
            return {}
        
        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = self.classifier(text)
        
        return {
            "classification": {
                "label": result[0]['label'],
                "confidence": result[0]['score']
            }
        }
    
    async def _summarize_text(self, text: str) -> Dict[str, Any]:
        """Text summarization using transformers"""
        if not self.summarizer:
            return {}
        
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
    
    async def _prepare_qa_context(self, text: str) -> Dict[str, Any]:
        """Prepare context for question answering"""
        if not self.qa_pipeline:
            return {}
        
        return {
            "qa_context": {
                "context_available": True,
                "context_length": len(text),
                "ready_for_questions": True
            }
        }
    
    async def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """Answer a question about the document"""
        if not self.qa_pipeline:
            return {"error": "QA pipeline not available"}
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def extract_keywords(self, text: str, top_n: int = 10) -> Dict[str, Any]:
        """Extract keywords from text"""
        if not self.nlp_model:
            return {"error": "NLP model not available"}
        
        doc = self.nlp_model(text)
        
        # Extract noun phrases and named entities
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
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of the text"""
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
            "confidence": confidence,
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }

# Global instance
real_ai_processor = RealAIDocumentProcessor()

async def initialize_real_ai_processor():
    """Initialize the real AI document processor"""
    try:
        await real_ai_processor.initialize()
        logger.info("Real AI document processor initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing real AI processor: {e}")
        raise













