"""
NLP Processor for text analysis and understanding.
"""

import logging
from typing import Dict, List, Optional, Any
import re

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..config.maintenance_config import NLPConfig

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    NLP processor for maintenance text analysis using spaCy and transformers.
    """
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.nlp = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        if self.config.use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                logger.warning("Spanish spaCy model not found. Install with: python -m spacy download es_core_news_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Using English spaCy model as fallback")
                except OSError:
                    logger.warning("spaCy models not available")
        
        if self.config.use_transformers and TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment"
                )
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text with NLP to extract information.
        
        Args:
            text: Input text to process
        
        Returns:
            Dictionary with NLP analysis results
        """
        result = {
            "original_text": text,
            "entities": [],
            "keywords": [],
            "sentiment": None,
            "maintenance_terms": []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            if self.config.enable_ner:
                entities = []
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
                result["entities"] = [e["text"] for e in entities]
            
            if self.config.enable_keyword_extraction:
                keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
                result["keywords"] = keywords[:10]
        
        if self.sentiment_analyzer and self.config.enable_sentiment:
            try:
                sentiment_result = self.sentiment_analyzer(text[:512])
                result["sentiment"] = sentiment_result[0] if sentiment_result else None
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        result["maintenance_terms"] = self._extract_maintenance_terms(text)
        
        return result
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text synchronously.
        
        Args:
            text: Input text
        
        Returns:
            List of extracted entity texts
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text synchronously.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
        
        Returns:
            List of extracted keywords
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        return keywords[:top_n]
    
    def analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment of text synchronously.
        
        Args:
            text: Input text
        
        Returns:
            Sentiment analysis result or None
        """
        if not self.sentiment_analyzer:
            return None
        
        try:
            sentiment_result = self.sentiment_analyzer(text[:512])
            return sentiment_result[0] if sentiment_result else None
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return None
    
    async def extract_keywords_async(self, text: str) -> Dict[str, List[str]]:
        """
        Extract keywords from text asynchronously.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with extracted keywords
        """
        result = await self.process_text(text)
        return {
            "keywords": result.get("keywords", []),
            "entities": result.get("entities", []),
            "maintenance_terms": result.get("maintenance_terms", [])
        }
    
    def _extract_maintenance_terms(self, text: str) -> List[str]:
        """Extract maintenance-related terms from text."""
        maintenance_keywords = [
            "mantenimiento", "reparación", "diagnóstico", "calibración",
            "lubricación", "reemplazo", "inspección", "ajuste", "limpieza",
            "robot", "máquina", "sensor", "actuador", "motor", "engranaje",
            "batería", "circuito", "software", "firmware"
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in maintenance_keywords if term in text_lower]
        return found_terms
    
    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify user intent from text synchronously.
        
        Args:
            text: Input text
        
        Returns:
            Intent classification
        """
        intents = {
            "question": ["cómo", "qué", "por qué", "cuándo", "dónde"],
            "procedure": ["procedimiento", "pasos", "instrucciones", "cómo hacer"],
            "diagnosis": ["problema", "error", "falla", "no funciona", "síntoma"],
            "schedule": ["programa", "calendario", "cuándo", "frecuencia", "intervalo"]
        }
        
        text_lower = text.lower()
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_intents.append(intent)
        
        return {
            "intents": detected_intents,
            "primary_intent": detected_intents[0] if detected_intents else "general"
        }
    
    async def classify_intent_async(self, text: str) -> Dict[str, Any]:
        """
        Classify user intent from text asynchronously.
        
        Args:
            text: Input text
        
        Returns:
            Intent classification
        """
        return self.classify_intent(text)
    
    async def close(self):
        """Cleanup resources."""
        pass
