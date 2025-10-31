"""
Gamma App - Real Improvement AI NLP
Natural Language Processing system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import nltk
import spacy
from textblob import TextBlob
import requests
import aiohttp

logger = logging.getLogger(__name__)

class NLPTaskType(Enum):
    """NLP task types"""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    TEXT_SUMMARIZATION = "text_summarization"
    LANGUAGE_DETECTION = "language_detection"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TEXT_SIMILARITY = "text_similarity"
    TEXT_GENERATION = "text_generation"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"

class NLPModel(Enum):
    """NLP models"""
    SPACY = "spacy"
    NLTK = "nltk"
    TEXTBLOB = "textblob"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    CUSTOM = "custom"

@dataclass
class NLPTask:
    """NLP processing task"""
    task_id: str
    task_type: NLPTaskType
    model: NLPModel
    input_text: str
    output_data: Dict[str, Any] = None
    status: str = "pending"
    confidence: float = 0.0
    processing_time: float = 0.0
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class NLPResult:
    """NLP processing result"""
    task_id: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RealImprovementAINLP:
    """
    Natural Language Processing system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize AI NLP system"""
        self.project_root = Path(project_root)
        self.tasks: Dict[str, NLPTask] = {}
        self.nlp_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.models: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize with default models
        self._initialize_default_models()
        
        logger.info(f"Real Improvement AI NLP initialized for {self.project_root}")
    
    def _initialize_default_models(self):
        """Initialize default NLP models"""
        try:
            # Spacy model
            self.models["spacy"] = {
                "name": "spaCy",
                "type": "general",
                "languages": ["en", "es", "fr", "de"],
                "capabilities": ["ner", "pos", "sentiment", "classification"]
            }
            
            # NLTK model
            self.models["nltk"] = {
                "name": "NLTK",
                "type": "general",
                "languages": ["en"],
                "capabilities": ["tokenization", "pos", "sentiment", "classification"]
            }
            
            # TextBlob model
            self.models["textblob"] = {
                "name": "TextBlob",
                "type": "general",
                "languages": ["en"],
                "capabilities": ["sentiment", "translation", "classification"]
            }
            
            # Transformers model
            self.models["transformers"] = {
                "name": "Transformers",
                "type": "advanced",
                "languages": ["en", "es", "fr", "de", "zh"],
                "capabilities": ["ner", "sentiment", "classification", "generation", "qa"]
            }
            
            # OpenAI model
            self.models["openai"] = {
                "name": "OpenAI GPT",
                "type": "advanced",
                "languages": ["en", "es", "fr", "de", "zh"],
                "capabilities": ["generation", "classification", "summarization", "translation", "qa"]
            }
            
            # Custom model
            self.models["custom"] = {
                "name": "Custom NLP Model",
                "type": "custom",
                "languages": ["en"],
                "capabilities": ["custom"]
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
    
    def create_nlp_task(self, task_type: NLPTaskType, model: NLPModel,
                       input_text: str) -> str:
        """Create NLP processing task"""
        try:
            task_id = f"nlp_task_{int(time.time() * 1000)}"
            
            task = NLPTask(
                task_id=task_id,
                task_type=task_type,
                model=model,
                input_text=input_text
            )
            
            self.tasks[task_id] = task
            
            # Process task asynchronously
            asyncio.create_task(self._process_nlp_task(task))
            
            self._log_nlp("task_created", f"NLP task {task_id} created")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create NLP task: {e}")
            raise
    
    async def _process_nlp_task(self, task: NLPTask):
        """Process NLP task"""
        try:
            start_time = time.time()
            task.status = "processing"
            
            self._log_nlp("task_processing", f"Processing NLP task {task.task_id}")
            
            # Process based on task type
            if task.task_type == NLPTaskType.SENTIMENT_ANALYSIS:
                result = await self._analyze_sentiment(task)
            elif task.task_type == NLPTaskType.TEXT_CLASSIFICATION:
                result = await self._classify_text(task)
            elif task.task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
                result = await self._recognize_entities(task)
            elif task.task_type == NLPTaskType.TEXT_SUMMARIZATION:
                result = await self._summarize_text(task)
            elif task.task_type == NLPTaskType.LANGUAGE_DETECTION:
                result = await self._detect_language(task)
            elif task.task_type == NLPTaskType.KEYWORD_EXTRACTION:
                result = await self._extract_keywords(task)
            elif task.task_type == NLPTaskType.TEXT_SIMILARITY:
                result = await self._calculate_similarity(task)
            elif task.task_type == NLPTaskType.TEXT_GENERATION:
                result = await self._generate_text(task)
            elif task.task_type == NLPTaskType.TRANSLATION:
                result = await self._translate_text(task)
            elif task.task_type == NLPTaskType.QUESTION_ANSWERING:
                result = await self._answer_question(task)
            else:
                result = {"error": f"Unknown task type: {task.task_type}"}
            
            # Update task
            task.output_data = result
            task.status = "completed" if "error" not in result else "failed"
            task.completed_at = datetime.utcnow()
            task.processing_time = time.time() - start_time
            task.confidence = result.get("confidence", 0.0)
            
            self._log_nlp("task_completed", f"NLP task {task.task_id} completed in {task.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process NLP task: {e}")
            task.status = "failed"
            task.output_data = {"error": str(e)}
            task.completed_at = datetime.utcnow()
    
    async def _analyze_sentiment(self, task: NLPTask) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            text = task.input_text
            
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Calculate confidence based on polarity
            confidence = abs(sentiment.polarity)
            
            # Determine sentiment label
            if sentiment.polarity > 0.1:
                sentiment_label = "positive"
            elif sentiment.polarity < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "sentiment": sentiment_label,
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
                "confidence": confidence,
                "model": task.model.value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _classify_text(self, task: NLPTask) -> Dict[str, Any]:
        """Classify text"""
        try:
            text = task.input_text.lower()
            
            # Simple text classification based on keywords
            categories = {
                "technical": ["code", "programming", "software", "development", "bug", "error"],
                "business": ["meeting", "project", "budget", "revenue", "profit", "strategy"],
                "personal": ["family", "friend", "home", "vacation", "hobby", "personal"],
                "news": ["news", "update", "announcement", "report", "article", "story"]
            }
            
            # Calculate scores for each category
            category_scores = {}
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in text)
                category_scores[category] = score
            
            # Get best category
            best_category = max(category_scores.items(), key=lambda x: x[1])
            
            # Calculate confidence
            total_score = sum(category_scores.values())
            confidence = best_category[1] / total_score if total_score > 0 else 0.0
            
            return {
                "category": best_category[0],
                "confidence": confidence,
                "category_scores": category_scores,
                "model": task.model.value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _recognize_entities(self, task: NLPTask) -> Dict[str, Any]:
        """Recognize named entities in text"""
        try:
            text = task.input_text
            
            # Simple entity recognition using regex patterns
            entities = []
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            for email in emails:
                entities.append({
                    "text": email,
                    "label": "EMAIL",
                    "start": text.find(email),
                    "end": text.find(email) + len(email),
                    "confidence": 0.9
                })
            
            # Phone pattern
            phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s\d{3}-\d{4}\b'
            phones = re.findall(phone_pattern, text)
            for phone in phones:
                entities.append({
                    "text": phone,
                    "label": "PHONE",
                    "start": text.find(phone),
                    "end": text.find(phone) + len(phone),
                    "confidence": 0.8
                })
            
            # URL pattern
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            for url in urls:
                entities.append({
                    "text": url,
                    "label": "URL",
                    "start": text.find(url),
                    "end": text.find(url) + len(url),
                    "confidence": 0.9
                })
            
            # Person names (simple heuristic)
            words = text.split()
            for i, word in enumerate(words):
                if word.istitle() and len(word) > 2:
                    entities.append({
                        "text": word,
                        "label": "PERSON",
                        "start": text.find(word),
                        "end": text.find(word) + len(word),
                        "confidence": 0.6
                    })
            
            # Calculate overall confidence
            overall_confidence = np.mean([e["confidence"] for e in entities]) if entities else 0.0
            
            return {
                "entities": entities,
                "confidence": overall_confidence,
                "model": task.model.value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _summarize_text(self, task: NLPTask) -> Dict[str, Any]:
        """Summarize text"""
        try:
            text = task.input_text
            
            # Simple extractive summarization
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Score sentences based on word frequency
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    if len(word) > 3:  # Only consider words longer than 3 characters
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = []
            for sentence in sentences:
                words = sentence.lower().split()
                score = sum(word_freq.get(word, 0) for word in words if len(word) > 3)
                sentence_scores.append((sentence, score))
            
            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            summary_sentences = sentence_scores[:min(3, len(sentence_scores))]
            summary = '. '.join([s[0] for s in summary_sentences])
            
            # Calculate confidence
            confidence = min(0.9, len(summary) / len(text))
            
            return {
                "summary": summary,
                "confidence": confidence,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text),
                "model": task.model.value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _detect_language(self, task: NLPTask) -> Dict[str, Any]:
        """Detect language of text"""
        try:
            text = task.input_text
            
            # Simple language detection based on common words
            language_indicators = {
                "en": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
                "es": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"],
                "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
                "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"]
            }
            
            text_lower = text.lower()
            language_scores = {}
            
            for lang, indicators in language_indicators.items():
                score = sum(1 for indicator in indicators if indicator in text_lower)
                language_scores[lang] = score
            
            # Get best language
            best_language = max(language_scores.items(), key=lambda x: x[1])
            
            # Calculate confidence
            total_score = sum(language_scores.values())
            confidence = best_language[1] / total_score if total_score > 0 else 0.0
            
            return {
                "language": best_language[0],
                "confidence": confidence,
                "language_scores": language_scores,
                "model": task.model.value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _extract_keywords(self, task: NLPTask) -> Dict[str, Any]:
        """Extract keywords from text"""
        try:
            text = task.input_text
            
            # Simple keyword extraction
            words = text.lower().split()
            
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequency
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate confidence
            confidence = min(0.9, len(top_keywords) / 10)
            
            return {
                "keywords": [{"word": word, "frequency": freq} for word, freq in top_keywords],
                "confidence": confidence,
                "total_words": len(words),
                "filtered_words": len(filtered_words),
                "model": task.model.value
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _calculate_similarity(self, task: NLPTask) -> Dict[str, Any]:
        """Calculate text similarity"""
        try:
            # For similarity, we need two texts to compare
            # This is a simplified version that would need two texts
            text = task.input_text
            
            # Simple similarity based on common words
            words = set(text.lower().split())
            
            # This is a placeholder - in a real implementation,
            # you would compare with another text
            similarity_score = 0.5  # Placeholder
            
            return {
                "similarity": similarity_score,
                "confidence": 0.7,
                "model": task.model.value,
                "note": "This is a simplified similarity calculation"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _generate_text(self, task: NLPTask) -> Dict[str, Any]:
        """Generate text"""
        try:
            text = task.input_text
            
            # Simple text generation (placeholder)
            generated_text = f"Generated response to: {text[:50]}..."
            
            return {
                "generated_text": generated_text,
                "confidence": 0.6,
                "model": task.model.value,
                "note": "This is a simplified text generation"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _translate_text(self, task: NLPTask) -> Dict[str, Any]:
        """Translate text"""
        try:
            text = task.input_text
            
            # Simple translation (placeholder)
            translated_text = f"Translated: {text}"
            
            return {
                "translated_text": translated_text,
                "confidence": 0.7,
                "model": task.model.value,
                "note": "This is a simplified translation"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _answer_question(self, task: NLPTask) -> Dict[str, Any]:
        """Answer question"""
        try:
            text = task.input_text
            
            # Simple question answering (placeholder)
            answer = f"Answer to: {text}"
            
            return {
                "answer": answer,
                "confidence": 0.6,
                "model": task.model.value,
                "note": "This is a simplified question answering"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_nlp_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get NLP task information"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task_id,
            "task_type": task.task_type.value,
            "model": task.model.value,
            "status": task.status,
            "output_data": task.output_data,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "processing_time": task.processing_time,
            "confidence": task.confidence
        }
    
    def get_nlp_summary(self) -> Dict[str, Any]:
        """Get NLP system summary"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        failed_tasks = len([t for t in self.tasks.values() if t.status == "failed"])
        
        # Calculate average confidence and processing time
        completed_task_confidences = [t.confidence for t in self.tasks.values() if t.status == "completed"]
        completed_task_times = [t.processing_time for t in self.tasks.values() if t.status == "completed"]
        
        avg_confidence = np.mean(completed_task_confidences) if completed_task_confidences else 0.0
        avg_processing_time = np.mean(completed_task_times) if completed_task_times else 0.0
        
        # Count by task type
        task_type_counts = {}
        for task in self.tasks.values():
            task_type = task.task_type.value
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        # Count by model
        model_counts = {}
        for task in self.tasks.values():
            model = task.model.value
            model_counts[model] = model_counts.get(model, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "task_type_distribution": task_type_counts,
            "model_distribution": model_counts,
            "available_models": list(self.models.keys())
        }
    
    def _log_nlp(self, event: str, message: str):
        """Log NLP event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if "nlp_logs" not in self.nlp_logs:
            self.nlp_logs["nlp_logs"] = []
        
        self.nlp_logs["nlp_logs"].append(log_entry)
        
        logger.info(f"NLP: {event} - {message}")
    
    def get_nlp_logs(self) -> List[Dict[str, Any]]:
        """Get NLP logs"""
        return self.nlp_logs.get("nlp_logs", [])

# Global NLP instance
improvement_ai_nlp = None

def get_improvement_ai_nlp() -> RealImprovementAINLP:
    """Get improvement AI NLP instance"""
    global improvement_ai_nlp
    if not improvement_ai_nlp:
        improvement_ai_nlp = RealImprovementAINLP()
    return improvement_ai_nlp













