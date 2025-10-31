"""
Advanced Natural Language Processing System
==========================================

This module provides advanced NLP capabilities for AI model analysis including:
- Text preprocessing and cleaning
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Dependency parsing
- Semantic similarity analysis
- Topic modeling and clustering
- Text summarization
- Language detection and translation
- Advanced text analytics
- Content generation analysis
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import os
import nltk
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import langdetect
from googletrans import Translator
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import our AI history analyzer components
from .ai_history_analyzer import (
    AIHistoryAnalyzer, ModelType, PerformanceMetric,
    get_ai_history_analyzer
)
from .config import get_ai_history_config

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass


@dataclass
class TextPreprocessing:
    """Text preprocessing result"""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    lemmatized_tokens: List[str]
    stop_words_removed: int
    text_length: int
    word_count: int
    sentence_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NamedEntityRecognition:
    """Named Entity Recognition result"""
    text: str
    entities: List[Dict[str, Any]]
    entity_types: Dict[str, int]
    entity_mentions: Dict[str, List[str]]
    confidence_scores: List[float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SemanticAnalysis:
    """Semantic analysis result"""
    text: str
    embeddings: List[float]
    semantic_similarity: Dict[str, float]
    topic_scores: Dict[str, float]
    key_phrases: List[str]
    semantic_clusters: List[int]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TopicModeling:
    """Topic modeling result"""
    texts: List[str]
    topics: List[Dict[str, Any]]
    topic_distributions: List[Dict[str, float]]
    coherence_score: float
    perplexity_score: float
    optimal_topics: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TextSummarization:
    """Text summarization result"""
    original_text: str
    summary: str
    summary_ratio: float
    compression_ratio: float
    key_sentences: List[str]
    summary_score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LanguageAnalysis:
    """Language analysis result"""
    text: str
    detected_language: str
    language_confidence: float
    translation_available: bool
    translated_text: str = None
    language_features: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.language_features is None:
            self.language_features = {}
        if self.metadata is None:
            self.metadata = {}


class AdvancedNLPProcessor:
    """Advanced Natural Language Processing system"""
    
    def __init__(self, model_storage_path: str = "nlp_models"):
        self.model_storage_path = model_storage_path
        self.analyzer = get_ai_history_analyzer()
        self.config = get_ai_history_config()
        
        # NLP models and tools
        self.nlp_models: Dict[str, Any] = {}
        self.embedding_models: Dict[str, Any] = {}
        self.summarization_models: Dict[str, Any] = {}
        
        # spaCy model
        self.nlp = None
        
        # Sentence transformer
        self.sentence_transformer = None
        
        # Translation
        self.translator = None
        
        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Text corpora for analysis
        self.text_corpora: Dict[str, List[str]] = {}
        
        # Ensure model storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and tools"""
        try:
            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy English model loaded")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Initialize sentence transformer
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {str(e)}")
                self.sentence_transformer = None
            
            # Initialize translation
            try:
                self.translator = Translator()
                logger.info("Translation service initialized")
            except Exception as e:
                logger.warning(f"Could not initialize translator: {str(e)}")
                self.translator = None
            
            # Initialize summarization models
            try:
                self.summarization_models["extractive"] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    return_all_scores=True
                )
                logger.info("Summarization model loaded")
            except Exception as e:
                logger.warning(f"Could not load summarization model: {str(e)}")
                self.summarization_models["extractive"] = None
            
            logger.info("NLP models initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
    
    async def preprocess_text(self, text: str, advanced: bool = True) -> TextPreprocessing:
        """Advanced text preprocessing"""
        try:
            if not text or not text.strip():
                return TextPreprocessing(
                    original_text=text,
                    cleaned_text="",
                    tokens=[],
                    lemmatized_tokens=[],
                    stop_words_removed=0,
                    text_length=0,
                    word_count=0,
                    sentence_count=0
                )
            
            original_text = text
            
            # Basic cleaning
            cleaned_text = self._clean_text(text)
            
            # Tokenization
            if self.nlp:
                doc = self.nlp(cleaned_text)
                tokens = [token.text for token in doc if not token.is_space]
                lemmatized_tokens = [token.lemma_ for token in doc if not token.is_space and not token.is_stop]
                stop_words_removed = len(tokens) - len(lemmatized_tokens)
            else:
                # Fallback to NLTK
                tokens = nltk.word_tokenize(cleaned_text)
                lemmatized_tokens = [token.lower() for token in tokens if token.isalpha()]
                stop_words_removed = len(tokens) - len(lemmatized_tokens)
            
            # Calculate statistics
            text_length = len(cleaned_text)
            word_count = len(tokens)
            sentence_count = len(nltk.sent_tokenize(cleaned_text))
            
            return TextPreprocessing(
                original_text=original_text,
                cleaned_text=cleaned_text,
                tokens=tokens,
                lemmatized_tokens=lemmatized_tokens,
                stop_words_removed=stop_words_removed,
                text_length=text_length,
                word_count=word_count,
                sentence_count=sentence_count,
                metadata={
                    "preprocessing_method": "spacy" if self.nlp else "nltk",
                    "advanced_preprocessing": advanced
                }
            )
        
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return TextPreprocessing(
                original_text=text,
                cleaned_text=text,
                tokens=[],
                lemmatized_tokens=[],
                stop_words_removed=0,
                text_length=len(text),
                word_count=0,
                sentence_count=0,
                metadata={"error": str(e)}
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove extra punctuation
            text = re.sub(r'[\.]{2,}', '.', text)
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            
            return text.strip()
        
        except Exception:
            return text
    
    async def extract_named_entities(self, text: str) -> NamedEntityRecognition:
        """Extract named entities from text"""
        try:
            if not text or not text.strip():
                return NamedEntityRecognition(
                    text=text,
                    entities=[],
                    entity_types={},
                    entity_mentions={},
                    confidence_scores=[]
                )
            
            entities = []
            entity_types = {}
            entity_mentions = defaultdict(list)
            confidence_scores = []
            
            if self.nlp:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": spacy.explain(ent.label_)
                    }
                    entities.append(entity_info)
                    
                    # Count entity types
                    entity_types[ent.label_] = entity_types.get(ent.label_, 0) + 1
                    
                    # Group entity mentions
                    entity_mentions[ent.label_].append(ent.text)
                    
                    # Estimate confidence (spaCy doesn't provide confidence scores)
                    confidence_scores.append(0.8)  # Default confidence
            
            else:
                # Fallback to NLTK
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    pos_tags = nltk.pos_tag(words)
                    chunks = nltk.ne_chunk(pos_tags)
                    
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity_info = {
                                "text": ' '.join(c[0] for c in chunk),
                                "label": chunk.label(),
                                "start": 0,  # NLTK doesn't provide positions
                                "end": 0,
                                "description": "Named entity"
                            }
                            entities.append(entity_info)
                            
                            entity_types[chunk.label()] = entity_types.get(chunk.label(), 0) + 1
                            entity_mentions[chunk.label()].append(entity_info["text"])
                            confidence_scores.append(0.6)  # Lower confidence for NLTK
            
            return NamedEntityRecognition(
                text=text,
                entities=entities,
                entity_types=dict(entity_types),
                entity_mentions=dict(entity_mentions),
                confidence_scores=confidence_scores,
                metadata={
                    "ner_method": "spacy" if self.nlp else "nltk",
                    "entity_count": len(entities)
                }
            )
        
        except Exception as e:
            logger.error(f"Error extracting named entities: {str(e)}")
            return NamedEntityRecognition(
                text=text,
                entities=[],
                entity_types={},
                entity_mentions={},
                confidence_scores=[],
                metadata={"error": str(e)}
            )
    
    async def analyze_semantics(self, text: str, reference_texts: List[str] = None) -> SemanticAnalysis:
        """Analyze semantic properties of text"""
        try:
            if not text or not text.strip():
                return SemanticAnalysis(
                    text=text,
                    embeddings=[],
                    semantic_similarity={},
                    topic_scores={},
                    key_phrases=[],
                    semantic_clusters=[]
                )
            
            # Generate embeddings
            embeddings = []
            if self.sentence_transformer:
                embeddings = self.sentence_transformer.encode([text])[0].tolist()
            else:
                # Fallback to TF-IDF
                vectorizer = TfidfVectorizer(max_features=100)
                embeddings = vectorizer.fit_transform([text]).toarray()[0].tolist()
            
            # Calculate semantic similarity with reference texts
            semantic_similarity = {}
            if reference_texts:
                if self.sentence_transformer:
                    ref_embeddings = self.sentence_transformer.encode(reference_texts)
                    text_embedding = self.sentence_transformer.encode([text])
                    
                    similarities = cosine_similarity(text_embedding, ref_embeddings)[0]
                    for i, ref_text in enumerate(reference_texts):
                        semantic_similarity[ref_text[:50]] = float(similarities[i])
                else:
                    # Fallback to TF-IDF similarity
                    vectorizer = TfidfVectorizer(max_features=100)
                    all_texts = [text] + reference_texts
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
                    
                    for i, ref_text in enumerate(reference_texts):
                        semantic_similarity[ref_text[:50]] = float(similarities[i])
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(text)
            
            # Calculate topic scores (simplified)
            topic_scores = self._calculate_topic_scores(text)
            
            # Semantic clustering (simplified)
            semantic_clusters = [0]  # Single cluster for now
            
            return SemanticAnalysis(
                text=text,
                embeddings=embeddings,
                semantic_similarity=semantic_similarity,
                topic_scores=topic_scores,
                key_phrases=key_phrases,
                semantic_clusters=semantic_clusters,
                metadata={
                    "embedding_dimension": len(embeddings),
                    "reference_texts_count": len(reference_texts) if reference_texts else 0,
                    "key_phrases_count": len(key_phrases)
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing semantics: {str(e)}")
            return SemanticAnalysis(
                text=text,
                embeddings=[],
                semantic_similarity={},
                topic_scores={},
                key_phrases=[],
                semantic_clusters=[],
                metadata={"error": str(e)}
            )
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        try:
            if not self.nlp:
                return []
            
            doc = self.nlp(text)
            key_phrases = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    key_phrases.append(chunk.text)
            
            # Extract named entities as key phrases
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    key_phrases.append(ent.text)
            
            # Remove duplicates and sort by frequency
            phrase_counts = Counter(key_phrases)
            return [phrase for phrase, count in phrase_counts.most_common(10)]
        
        except Exception:
            return []
    
    def _calculate_topic_scores(self, text: str) -> Dict[str, float]:
        """Calculate topic scores for text"""
        try:
            # Simplified topic scoring based on keywords
            topics = {
                "technology": ["computer", "software", "hardware", "digital", "internet", "ai", "machine learning"],
                "business": ["company", "business", "market", "sales", "profit", "revenue", "customer"],
                "science": ["research", "study", "experiment", "data", "analysis", "scientific", "theory"],
                "health": ["health", "medical", "doctor", "patient", "treatment", "disease", "medicine"],
                "education": ["school", "university", "student", "teacher", "learning", "education", "course"]
            }
            
            text_lower = text.lower()
            topic_scores = {}
            
            for topic, keywords in topics.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                topic_scores[topic] = score / len(keywords)
            
            return topic_scores
        
        except Exception:
            return {}
    
    async def perform_topic_modeling(self, texts: List[str], num_topics: int = 5) -> TopicModeling:
        """Perform topic modeling on a collection of texts"""
        try:
            if not texts or len(texts) < 2:
                return TopicModeling(
                    texts=texts,
                    topics=[],
                    topic_distributions=[],
                    coherence_score=0.0,
                    perplexity_score=0.0,
                    optimal_topics=0
                )
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                preprocessing = await self.preprocess_text(text)
                processed_texts.append(' '.join(preprocessing.lemmatized_tokens))
            
            # Vectorize texts
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            
            # Perform LDA
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_scores = [topic[i] for i in top_words_idx]
                
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "scores": top_scores,
                    "description": f"Topic {topic_idx + 1}"
                })
            
            # Calculate topic distributions for each text
            topic_distributions = []
            doc_topic_probs = lda.transform(tfidf_matrix)
            
            for i, probs in enumerate(doc_topic_probs):
                distribution = {f"topic_{j}": float(prob) for j, prob in enumerate(probs)}
                topic_distributions.append(distribution)
            
            # Calculate coherence score (simplified)
            coherence_score = self._calculate_coherence_score(topics, processed_texts)
            
            # Calculate perplexity
            perplexity_score = lda.perplexity(tfidf_matrix)
            
            return TopicModeling(
                texts=texts,
                topics=topics,
                topic_distributions=topic_distributions,
                coherence_score=coherence_score,
                perplexity_score=perplexity_score,
                optimal_topics=num_topics,
                metadata={
                    "text_count": len(texts),
                    "vocabulary_size": len(feature_names),
                    "model_type": "LDA"
                }
            )
        
        except Exception as e:
            logger.error(f"Error performing topic modeling: {str(e)}")
            return TopicModeling(
                texts=texts,
                topics=[],
                topic_distributions=[],
                coherence_score=0.0,
                perplexity_score=0.0,
                optimal_topics=0,
                metadata={"error": str(e)}
            )
    
    def _calculate_coherence_score(self, topics: List[Dict], texts: List[str]) -> float:
        """Calculate topic coherence score (simplified)"""
        try:
            # Simplified coherence calculation
            total_coherence = 0.0
            
            for topic in topics:
                words = topic["words"][:5]  # Top 5 words
                word_pairs = [(words[i], words[j]) for i in range(len(words)) for j in range(i+1, len(words))]
                
                coherence = 0.0
                for word1, word2 in word_pairs:
                    # Count co-occurrences
                    co_occurrences = sum(1 for text in texts if word1 in text.lower() and word2 in text.lower())
                    total_texts = len(texts)
                    if total_texts > 0:
                        coherence += co_occurrences / total_texts
                
                if word_pairs:
                    coherence /= len(word_pairs)
                
                total_coherence += coherence
            
            return total_coherence / len(topics) if topics else 0.0
        
        except Exception:
            return 0.0
    
    async def summarize_text(self, text: str, max_length: int = 150) -> TextSummarization:
        """Summarize text using extractive and abstractive methods"""
        try:
            if not text or not text.strip():
                return TextSummarization(
                    original_text=text,
                    summary="",
                    summary_ratio=0.0,
                    compression_ratio=0.0,
                    key_sentences=[],
                    summary_score=0.0
                )
            
            # Method 1: Extractive summarization
            extractive_summary = self._extractive_summarization(text, max_length)
            
            # Method 2: Abstractive summarization (if model available)
            abstractive_summary = ""
            if self.summarization_models.get("extractive"):
                try:
                    # Truncate text if too long
                    if len(text) > 1000:
                        text = text[:1000]
                    
                    summary_result = self.summarization_models["extractive"](text, max_length=max_length, min_length=30)
                    abstractive_summary = summary_result[0]["summary_text"]
                except Exception as e:
                    logger.warning(f"Error in abstractive summarization: {str(e)}")
            
            # Choose best summary
            if abstractive_summary and len(abstractive_summary) > len(extractive_summary):
                final_summary = abstractive_summary
                method = "abstractive"
            else:
                final_summary = extractive_summary
                method = "extractive"
            
            # Calculate metrics
            summary_ratio = len(final_summary) / len(text) if text else 0.0
            compression_ratio = 1.0 - summary_ratio
            
            # Extract key sentences
            key_sentences = self._extract_key_sentences(text)
            
            # Calculate summary score
            summary_score = self._calculate_summary_score(text, final_summary)
            
            return TextSummarization(
                original_text=text,
                summary=final_summary,
                summary_ratio=summary_ratio,
                compression_ratio=compression_ratio,
                key_sentences=key_sentences,
                summary_score=summary_score,
                metadata={
                    "method": method,
                    "original_length": len(text),
                    "summary_length": len(final_summary)
                }
            )
        
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return TextSummarization(
                original_text=text,
                summary="",
                summary_ratio=0.0,
                compression_ratio=0.0,
                key_sentences=[],
                summary_score=0.0,
                metadata={"error": str(e)}
            )
    
    def _extractive_summarization(self, text: str, max_length: int) -> str:
        """Extractive summarization using sentence scoring"""
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 2:
                return text
            
            # Score sentences
            sentence_scores = []
            for sentence in sentences:
                score = self._score_sentence(sentence, text)
                sentence_scores.append((sentence, score))
            
            # Sort by score and select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            summary_sentences = []
            current_length = 0
            
            for sentence, score in sentence_scores:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            return ' '.join(summary_sentences)
        
        except Exception:
            return text[:max_length]
    
    def _score_sentence(self, sentence: str, full_text: str) -> float:
        """Score a sentence for summarization"""
        try:
            score = 0.0
            
            # Word frequency scoring
            words = nltk.word_tokenize(sentence.lower())
            word_freq = Counter(nltk.word_tokenize(full_text.lower()))
            
            for word in words:
                if word.isalpha():
                    score += word_freq.get(word, 0)
            
            # Position scoring (first and last sentences get higher scores)
            sentences = nltk.sent_tokenize(full_text)
            if sentence in sentences:
                position = sentences.index(sentence)
                if position == 0 or position == len(sentences) - 1:
                    score *= 1.5
            
            # Length scoring (medium length sentences preferred)
            length_score = 1.0 - abs(len(words) - 15) / 15  # Optimal around 15 words
            score *= length_score
            
            return score
        
        except Exception:
            return 0.0
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """Extract key sentences from text"""
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 3:
                return sentences
            
            # Score and select top sentences
            sentence_scores = [(sentence, self._score_sentence(sentence, text)) for sentence in sentences]
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 3 sentences
            return [sentence for sentence, score in sentence_scores[:3]]
        
        except Exception:
            return []
    
    def _calculate_summary_score(self, original: str, summary: str) -> float:
        """Calculate quality score for summary"""
        try:
            if not summary:
                return 0.0
            
            # Coverage score (how much of original is covered)
            original_words = set(nltk.word_tokenize(original.lower()))
            summary_words = set(nltk.word_tokenize(summary.lower()))
            coverage = len(original_words.intersection(summary_words)) / len(original_words) if original_words else 0.0
            
            # Compression score (appropriate length)
            compression = len(summary) / len(original) if original else 0.0
            compression_score = 1.0 - abs(compression - 0.3) / 0.3  # Optimal around 30%
            
            # Coherence score (simplified)
            coherence_score = 0.8  # Default value
            
            return (coverage * 0.4 + compression_score * 0.3 + coherence_score * 0.3)
        
        except Exception:
            return 0.0
    
    async def analyze_language(self, text: str) -> LanguageAnalysis:
        """Analyze language properties of text"""
        try:
            if not text or not text.strip():
                return LanguageAnalysis(
                    text=text,
                    detected_language="unknown",
                    language_confidence=0.0,
                    translation_available=False
                )
            
            # Detect language
            try:
                detected_language = langdetect.detect(text)
                language_confidence = 0.8  # langdetect doesn't provide confidence
            except Exception:
                detected_language = "unknown"
                language_confidence = 0.0
            
            # Check if translation is available
            translation_available = self.translator is not None and detected_language != "en"
            
            # Translate if needed and available
            translated_text = None
            if translation_available and detected_language != "en":
                try:
                    translation = self.translator.translate(text, dest='en')
                    translated_text = translation.text
                except Exception as e:
                    logger.warning(f"Translation failed: {str(e)}")
            
            # Analyze language features
            language_features = self._analyze_language_features(text)
            
            return LanguageAnalysis(
                text=text,
                detected_language=detected_language,
                language_confidence=language_confidence,
                translation_available=translation_available,
                translated_text=translated_text,
                language_features=language_features,
                metadata={
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
        
        except Exception as e:
            logger.error(f"Error analyzing language: {str(e)}")
            return LanguageAnalysis(
                text=text,
                detected_language="unknown",
                language_confidence=0.0,
                translation_available=False,
                metadata={"error": str(e)}
            )
    
    def _analyze_language_features(self, text: str) -> Dict[str, Any]:
        """Analyze language features"""
        try:
            features = {}
            
            # Readability metrics
            blob = TextBlob(text)
            features["polarity"] = blob.sentiment.polarity
            features["subjectivity"] = blob.sentiment.subjectivity
            
            # Text statistics
            words = text.split()
            sentences = nltk.sent_tokenize(text)
            
            features["avg_word_length"] = np.mean([len(word) for word in words]) if words else 0
            features["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
            features["syllable_count"] = sum(self._count_syllables(word) for word in words)
            
            # Flesch Reading Ease (simplified)
            if sentences and words:
                features["flesch_score"] = 206.835 - (1.015 * features["avg_sentence_length"]) - (84.6 * features["syllable_count"] / len(words))
            else:
                features["flesch_score"] = 0
            
            return features
        
        except Exception:
            return {}
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        try:
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent 'e'
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        except Exception:
            return 1
    
    async def comprehensive_nlp_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis"""
        try:
            # Run all analyses in parallel
            preprocessing_task = asyncio.create_task(self.preprocess_text(text))
            ner_task = asyncio.create_task(self.extract_named_entities(text))
            semantic_task = asyncio.create_task(self.analyze_semantics(text))
            summarization_task = asyncio.create_task(self.summarize_text(text))
            language_task = asyncio.create_task(self.analyze_language(text))
            
            # Wait for all analyses to complete
            preprocessing_result = await preprocessing_task
            ner_result = await ner_task
            semantic_result = await semantic_task
            summarization_result = await summarization_task
            language_result = await language_task
            
            # Combine results
            comprehensive_result = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "preprocessing": asdict(preprocessing_result),
                "named_entities": asdict(ner_result),
                "semantics": asdict(semantic_result),
                "summarization": asdict(summarization_result),
                "language": asdict(language_result),
                "overall_quality_score": self._calculate_nlp_quality_score(
                    preprocessing_result, ner_result, semantic_result, 
                    summarization_result, language_result
                )
            }
            
            # Store in history
            self.analysis_history.append(comprehensive_result)
            
            return comprehensive_result
        
        except Exception as e:
            logger.error(f"Error in comprehensive NLP analysis: {str(e)}")
            return {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _calculate_nlp_quality_score(self, preprocessing, ner, semantic, summarization, language) -> float:
        """Calculate overall NLP quality score"""
        try:
            # Weighted combination of all scores
            weights = {
                "preprocessing": 0.2,
                "ner": 0.2,
                "semantic": 0.2,
                "summarization": 0.2,
                "language": 0.2
            }
            
            # Calculate individual scores
            preprocessing_score = min(preprocessing.word_count / 100, 1.0)  # Normalize word count
            ner_score = len(ner.entities) / 10 if ner.entities else 0.0  # Normalize entity count
            semantic_score = len(semantic.key_phrases) / 5 if semantic.key_phrases else 0.0  # Normalize key phrases
            summarization_score = summarization.summary_score
            language_score = language.language_confidence
            
            overall_score = (
                weights["preprocessing"] * preprocessing_score +
                weights["ner"] * ner_score +
                weights["semantic"] * semantic_score +
                weights["summarization"] * summarization_score +
                weights["language"] * language_score
            )
            
            return overall_score
        
        except Exception:
            return 0.5
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get NLP analysis history"""
        return self.analysis_history[-limit:]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get NLP analysis statistics"""
        if not self.analysis_history:
            return {}
        
        try:
            return {
                "total_analyses": len(self.analysis_history),
                "average_quality_score": np.mean([analysis.get("overall_quality_score", 0.5) for analysis in self.analysis_history]),
                "languages_detected": list(set([analysis["language"]["detected_language"] for analysis in self.analysis_history])),
                "average_entity_count": np.mean([len(analysis["named_entities"]["entities"]) for analysis in self.analysis_history]),
                "average_summary_ratio": np.mean([analysis["summarization"]["summary_ratio"] for analysis in self.analysis_history])
            }
        
        except Exception as e:
            logger.error(f"Error calculating NLP analysis statistics: {str(e)}")
            return {}


# Global NLP processor instance
_nlp_processor: Optional[AdvancedNLPProcessor] = None


def get_nlp_processor(model_storage_path: str = "nlp_models") -> AdvancedNLPProcessor:
    """Get or create global NLP processor"""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = AdvancedNLPProcessor(model_storage_path)
    return _nlp_processor


# Example usage
async def main():
    """Example usage of advanced NLP processor"""
    nlp_processor = get_nlp_processor()
    
    # Test text
    test_text = "Artificial intelligence is transforming the way we work and live. Machine learning algorithms can now process vast amounts of data to make predictions and decisions. Companies like Google, Microsoft, and OpenAI are leading the development of advanced AI systems."
    
    # Comprehensive analysis
    result = await nlp_processor.comprehensive_nlp_analysis(test_text)
    
    print("Comprehensive NLP Analysis Result:")
    print(f"Word Count: {result['preprocessing']['word_count']}")
    print(f"Named Entities: {len(result['named_entities']['entities'])}")
    print(f"Key Phrases: {result['semantics']['key_phrases']}")
    print(f"Summary: {result['summarization']['summary']}")
    print(f"Detected Language: {result['language']['detected_language']}")
    print(f"Overall Quality Score: {result['overall_quality_score']:.3f}")
    
    # Get statistics
    stats = nlp_processor.get_analysis_statistics()
    print(f"\nNLP Analysis Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())

























