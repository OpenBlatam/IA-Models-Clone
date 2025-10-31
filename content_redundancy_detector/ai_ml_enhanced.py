"""
Enhanced AI/ML Engine for Advanced Content Analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import json

# AI/ML Libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import nltk
from textblob import TextBlob
from langdetect import detect, LangDetectException
import gensim
from gensim import corpora, models
import torch

from config import settings

logger = logging.getLogger(__name__)


class AIMLEngine:
    """Enhanced AI/ML Engine for advanced content analysis"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.nlp = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize AI/ML models and components"""
        if self.initialized:
            return
            
        try:
            logger.info("Initializing AI/ML Engine...")
            
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("Downloading NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            
            # Initialize sentence transformer
            self.models['sentence_transformer'] = SentenceTransformer(settings.embedding_model)
            
            # Initialize sentiment analysis pipeline
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Initialize text summarization
            self.models['summarizer'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            # Initialize TF-IDF vectorizer
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize topic modeling
            self.models['lda'] = LatentDirichletAllocation(
                n_components=10,
                random_state=42,
                max_iter=10
            )
            
            self.initialized = True
            logger.info("AI/ML Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI/ML Engine: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using advanced models"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use transformer-based sentiment analysis
            results = self.models['sentiment'](text)
            
            # Extract sentiment scores
            sentiment_scores = {}
            for result in results[0]:
                sentiment_scores[result['label']] = result['score']
            
            # Determine dominant sentiment
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            # Also use TextBlob for comparison
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            return {
                "dominant_sentiment": dominant_sentiment,
                "sentiment_scores": sentiment_scores,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "confidence": sentiment_scores[dominant_sentiment],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text"""
        try:
            language = detect(text)
            confidence = 1.0  # langdetect doesn't provide confidence scores
            
            return {
                "language": language,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except LangDetectException:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def extract_topics(self, texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
        """Extract topics from a collection of texts using LDA"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                doc = self.nlp(text)
                tokens = [token.lemma_.lower() for token in doc 
                         if not token.is_stop and not token.is_punct and token.is_alpha]
                processed_texts.append(' '.join(tokens))
            
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizers['tfidf'].fit_transform(processed_texts)
            
            # Fit LDA model
            lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            lda_model.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = self.vectorizers['tfidf'].get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            return {
                "topics": topics,
                "num_topics": num_topics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate semantic similarity using sentence transformers"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Generate embeddings
            embeddings = self.models['sentence_transformer'].encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return {
                "similarity_score": float(similarity),
                "similarity_percentage": float(similarity * 100),
                "method": "sentence_transformer",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def detect_plagiarism(self, text: str, reference_texts: List[str], threshold: float = 0.8) -> Dict[str, Any]:
        """Detect potential plagiarism by comparing with reference texts"""
        if not self.initialized:
            await self.initialize()
        
        try:
            similarities = []
            
            for i, ref_text in enumerate(reference_texts):
                similarity_result = await self.calculate_semantic_similarity(text, ref_text)
                if "similarity_score" in similarity_result:
                    similarities.append({
                        "reference_index": i,
                        "similarity_score": similarity_result["similarity_score"],
                        "is_plagiarized": similarity_result["similarity_score"] >= threshold
                    })
            
            # Find highest similarity
            max_similarity = max(similarities, key=lambda x: x["similarity_score"]) if similarities else None
            
            return {
                "is_plagiarized": max_similarity["is_plagiarized"] if max_similarity else False,
                "max_similarity": max_similarity["similarity_score"] if max_similarity else 0.0,
                "similarities": similarities,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in plagiarism detection: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text using spaCy"""
        if not self.initialized:
            await self.initialize()
        
        try:
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "description": spacy.explain(ent.label_)
                })
            
            return {
                "entities": entities,
                "entity_count": len(entities),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_summary(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """Generate text summary using BART model"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Truncate text if too long
            if len(text) > 1024:
                text = text[:1024]
            
            summary = self.models['summarizer'](text, max_length=max_length, min_length=30, do_sample=False)
            
            return {
                "summary": summary[0]["summary_text"],
                "original_length": len(text),
                "summary_length": len(summary[0]["summary_text"]),
                "compression_ratio": len(summary[0]["summary_text"]) / len(text),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_readability(self, text: str) -> Dict[str, Any]:
        """Advanced readability analysis using multiple metrics"""
        try:
            blob = TextBlob(text)
            
            # Basic metrics
            sentences = blob.sentences
            words = blob.words
            characters = len(text)
            
            # Calculate readability metrics
            avg_sentence_length = sum(len(sentence.words) for sentence in sentences) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Flesch Reading Ease (simplified)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            
            # Grade level estimation
            grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_word_length) - 15.59
            
            return {
                "flesch_score": flesch_score,
                "grade_level": grade_level,
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "sentence_count": len(sentences),
                "word_count": len(words),
                "character_count": characters,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in readability analysis: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all AI/ML features"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Run all analyses in parallel
            tasks = [
                self.analyze_sentiment(text),
                self.detect_language(text),
                self.extract_entities(text),
                self.generate_summary(text),
                self.analyze_readability(text)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "text_length": len(text),
                "sentiment": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "language": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "entities": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "summary": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "readability": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])},
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global AI/ML Engine instance
ai_ml_engine = AIMLEngine()
















