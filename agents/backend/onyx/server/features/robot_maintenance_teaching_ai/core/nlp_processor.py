"""
NLP Processor for maintenance text analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from ..config.maintenance_config import NLPConfig

logger = logging.getLogger(__name__)


class MaintenanceNLPProcessor:
    """
    NLP processor for analyzing maintenance-related text using spaCy and transformers.
    """
    
    def __init__(self, config: Optional[NLPConfig] = None):
        self.config = config or NLPConfig()
        self.nlp = None
        self.sentiment_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            self.nlp = spacy.load(self.config.model_name)
            logger.info(f"Loaded spaCy model: {self.config.model_name}")
        except OSError:
            logger.warning(f"spaCy model {self.config.model_name} not found. "
                         "Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", 
                          self.config.model_name], check=True)
            self.nlp = spacy.load(self.config.model_name)
        
        if self.config.use_transformer and self.config.enable_sentiment:
            try:
                device = 0 if torch.cuda.is_available() else -1
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.config.transformer_model,
                    device=device
                )
                logger.info(f"Loaded transformer model: {self.config.transformer_model}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.sentiment_analyzer = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from maintenance text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with entity types and values
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {
            "components": [],
            "tools": [],
            "problems": [],
            "procedures": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                entities["components"].append(ent.text)
            elif ent.label_ in ["EVENT", "MISC"]:
                entities["problems"].append(ent.text)
        
        return entities
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
        
        Returns:
            List of (keyword, score) tuples
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.pos_ in ["NOUN", "PROPN", "ADJ"] and
                len(token.text) > 2):
                keywords.append((token.text, token.rank))
        
        keywords.sort(key=lambda x: x[1])
        return keywords[:top_n]
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of maintenance text.
        
        Args:
            text: Input text
        
        Returns:
            Sentiment analysis results
        """
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                return {
                    "label": result["label"],
                    "score": result["score"]
                }
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
        
        return {"label": "NEUTRAL", "score": 0.5}
    
    def find_similar_procedures(
        self,
        query: str,
        procedures: List[str],
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar maintenance procedures.
        
        Args:
            query: Query text
            procedures: List of procedure descriptions
            threshold: Similarity threshold
        
        Returns:
            List of (procedure, similarity_score) tuples
        """
        if not self.nlp:
            return []
        
        threshold = threshold or self.config.similarity_threshold
        query_doc = self.nlp(query)
        
        similarities = []
        for procedure in procedures:
            proc_doc = self.nlp(procedure)
            similarity = query_doc.similarity(proc_doc)
            if similarity >= threshold:
                similarities.append((procedure, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def extract_maintenance_steps(self, text: str) -> List[str]:
        """
        Extract maintenance steps from text.
        
        Args:
            text: Input text containing procedure
        
        Returns:
            List of maintenance steps
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        steps = []
        
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in 
                   ["paso", "step", "instrucción", "procedimiento"]):
                steps.append(sent.text.strip())
        
        return steps
    
    def process_maintenance_query(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Process a maintenance query with full NLP pipeline.
        
        Args:
            query: User query
        
        Returns:
            Complete NLP analysis
        """
        return {
            "entities": self.extract_entities(query),
            "keywords": self.extract_keywords(query),
            "sentiment": self.analyze_sentiment(query),
            "steps": self.extract_maintenance_steps(query)
        }
    
    def extract_intent(self, text: str) -> Dict[str, Any]:
        """
        Extract user intent from maintenance query.
        
        Args:
            text: Input text
        
        Returns:
            Intent classification with confidence
        """
        if not self.nlp:
            return {"intent": "unknown", "confidence": 0.0}
        
        text_lower = text.lower()
        
        # Intent patterns
        intents = {
            "diagnosis": ["diagnosticar", "problema", "error", "fallo", "avería", "síntoma"],
            "procedure": ["procedimiento", "cómo", "pasos", "instrucciones", "proceso"],
            "explanation": ["qué es", "explicar", "definir", "significa", "función"],
            "schedule": ["programa", "calendario", "frecuencia", "cuándo", "cuando"],
            "question": ["pregunta", "duda", "consultar", "información"]
        }
        
        scores = {}
        for intent, keywords in intents.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[intent] = score / len(keywords)
        
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent]
            return {
                "intent": best_intent,
                "confidence": confidence,
                "all_scores": scores
            }
        
        return {"intent": "unknown", "confidence": 0.0}
    
    def extract_urgency(self, text: str) -> Dict[str, Any]:
        """
        Extract urgency level from maintenance text.
        
        Args:
            text: Input text
        
        Returns:
            Urgency classification
        """
        if not self.nlp:
            return {"urgency": "normal", "level": 2}
        
        text_lower = text.lower()
        
        # Urgency indicators
        urgent_keywords = ["urgente", "inmediato", "emergencia", "crítico", "grave", "fallo"]
        normal_keywords = ["normal", "rutinario", "programado", "preventivo"]
        low_keywords = ["futuro", "planificar", "eventual"]
        
        urgent_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        normal_count = sum(1 for keyword in normal_keywords if keyword in text_lower)
        low_count = sum(1 for keyword in low_keywords if keyword in text_lower)
        
        if urgent_count > 0:
            return {"urgency": "urgent", "level": 3, "confidence": min(1.0, urgent_count / 2)}
        elif low_count > 0:
            return {"urgency": "low", "level": 1, "confidence": min(1.0, low_count / 2)}
        else:
            return {"urgency": "normal", "level": 2, "confidence": 0.5}
    
    def extract_component_mentions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mentioned robot components from text.
        
        Args:
            text: Input text
        
        Returns:
            List of mentioned components with context
        """
        if not self.nlp:
            return []
        
        # Common robot components
        components = [
            "motor", "engranaje", "reductor", "junta", "rodamiento", "sensor",
            "actuador", "controlador", "batería", "cable", "conector", "válvula",
            "bomba", "filtro", "lubricante", "freno", "embrague"
        ]
        
        doc = self.nlp(text)
        mentioned = []
        
        for token in doc:
            token_lower = token.text.lower()
            for component in components:
                if component in token_lower or token_lower in component:
                    mentioned.append({
                        "component": component,
                        "mention": token.text,
                        "context": token.sent.text if token.sent else "",
                        "position": token.i
                    })
                    break
        
        return mentioned






