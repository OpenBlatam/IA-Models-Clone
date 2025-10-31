"""
NLP System for Business Agents
============================

A comprehensive Natural Language Processing system for the Business Agents platform.
Provides text analysis, sentiment analysis, entity extraction, topic modeling, and more.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
import json
import re
from dataclasses import dataclass

# Core NLP libraries
import spacy
import nltk
from textblob import TextBlob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Advanced NLP libraries
import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    pipeline, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)

# Custom imports
from .config import config
from .exceptions import NLPProcessingError, ModelLoadError, TextProcessingError

# Configure logging
logger = logging.getLogger(__name__)

class Language(str, Enum):
    """Supported languages for NLP processing."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"

class SentimentType(str, Enum):
    """Sentiment classification types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class EntityType(str, Enum):
    """Named entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"
    MONEY = "MONEY"
    DATE = "DATE"
    TIME = "TIME"
    PERCENT = "PERCENT"
    QUANTITY = "QUANTITY"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    CUSTOM = "CUSTOM"

@dataclass
class TextAnalysisResult:
    """Result of text analysis."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability_score: float
    word_count: int
    sentence_count: int
    processing_time: float
    timestamp: datetime

@dataclass
class Entity:
    """Named entity representation."""
    text: str
    label: str
    start: int
    end: int
    confidence: float

@dataclass
class Topic:
    """Topic representation."""
    id: int
    words: List[str]
    weights: List[float]
    coherence_score: float

class NLPSystem:
    """Comprehensive NLP system for business text processing."""
    
    def __init__(self):
        """Initialize the NLP system."""
        self.models = {}
        self.pipelines = {}
        self.vectorizers = {}
        self.classifiers = {}
        self.is_initialized = False
        
        # Model configurations
        self.model_configs = {
            "sentiment": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "task": "sentiment-analysis"
            },
            "ner": {
                "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "task": "ner"
            },
            "classification": {
                "model_name": "microsoft/DialoGPT-medium",
                "task": "text-classification"
            },
            "summarization": {
                "model_name": "facebook/bart-large-cnn",
                "task": "summarization"
            },
            "translation": {
                "model_name": "Helsinki-NLP/opus-mt-en-es",
                "task": "translation"
            }
        }
        
        # Initialize NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk_data = [
                'punkt', 'stopwords', 'averaged_perceptron_tagger',
                'vader_lexicon', 'wordnet', 'omw-1.4'
            ]
            for data in nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                except LookupError:
                    nltk.download(data, quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    async def initialize(self):
        """Initialize the NLP system with all required models."""
        try:
            logger.info("Initializing NLP system...")
            
            # Load spaCy models
            await self._load_spacy_models()
            
            # Load transformer models
            await self._load_transformer_models()
            
            # Initialize vectorizers
            self._initialize_vectorizers()
            
            # Initialize classifiers
            await self._initialize_classifiers()
            
            self.is_initialized = True
            logger.info("NLP system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP system: {e}")
            raise ModelLoadError(f"NLP system initialization failed: {e}")
    
    async def _load_spacy_models(self):
        """Load spaCy language models."""
        try:
            # Load English model
            self.models['en'] = spacy.load("en_core_web_sm")
            
            # Load other language models if available
            for lang in ['es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'ru', 'ar']:
                try:
                    model_name = f"{lang}_core_news_sm"
                    self.models[lang] = spacy.load(model_name)
                except OSError:
                    logger.warning(f"spaCy model for {lang} not available")
                    
        except Exception as e:
            logger.error(f"Failed to load spaCy models: {e}")
            raise ModelLoadError(f"spaCy model loading failed: {e}")
    
    async def _load_transformer_models(self):
        """Load transformer models for various NLP tasks."""
        try:
            for task, config in self.model_configs.items():
                try:
                    if config["task"] == "sentiment-analysis":
                        self.pipelines[task] = pipeline(
                            config["task"],
                            model=config["model_name"],
                            return_all_scores=True
                        )
                    else:
                        self.pipelines[task] = pipeline(
                            config["task"],
                            model=config["model_name"]
                        )
                    logger.info(f"Loaded {task} model: {config['model_name']}")
                except Exception as e:
                    logger.warning(f"Failed to load {task} model: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load transformer models: {e}")
            raise ModelLoadError(f"Transformer model loading failed: {e}")
    
    def _initialize_vectorizers(self):
        """Initialize text vectorizers."""
        try:
            # TF-IDF vectorizer
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            # Count vectorizer
            self.vectorizers['count'] = CountVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            logger.info("Vectorizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vectorizers: {e}")
            raise ModelLoadError(f"Vectorizer initialization failed: {e}")
    
    async def _initialize_classifiers(self):
        """Initialize text classifiers."""
        try:
            # Initialize classifiers for different tasks
            self.classifiers['sentiment'] = MultinomialNB()
            self.classifiers['topic'] = LatentDirichletAllocation(
                n_components=10,
                random_state=42,
                max_iter=100
            )
            self.classifiers['clustering'] = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            logger.info("Classifiers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize classifiers: {e}")
            raise ModelLoadError(f"Classifier initialization failed: {e}")
    
    async def analyze_text(self, text: str, language: str = "en") -> TextAnalysisResult:
        """Perform comprehensive text analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Basic text statistics
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Language detection
            detected_language = await self.detect_language(text)
            
            # Sentiment analysis
            sentiment = await self.analyze_sentiment(text, language)
            
            # Entity extraction
            entities = await self.extract_entities(text, language)
            
            # Keyword extraction
            keywords = await self.extract_keywords(text, language)
            
            # Topic modeling
            topics = await self.extract_topics([text], language)
            
            # Readability score
            readability_score = await self.calculate_readability(text, language)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TextAnalysisResult(
                text=text,
                language=detected_language,
                sentiment=sentiment,
                entities=entities,
                keywords=keywords,
                topics=topics,
                readability_score=readability_score,
                word_count=word_count,
                sentence_count=sentence_count,
                processing_time=processing_time,
                timestamp=start_time
            )
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise NLPProcessingError(f"Text analysis failed: {e}")
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            if 'sentiment' in self.pipelines:
                # Use transformer model for language detection
                blob = TextBlob(text)
                return blob.detect_language()
            else:
                # Fallback to simple heuristics
                return self._simple_language_detection(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"  # Default to English
    
    def _simple_language_detection(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        # Basic heuristics for language detection
        if re.search(r'[ñáéíóúü]', text.lower()):
            return "es"  # Spanish
        elif re.search(r'[àâäéèêëïîôöùûüÿç]', text.lower()):
            return "fr"  # French
        elif re.search(r'[äöüß]', text.lower()):
            return "de"  # German
        elif re.search(r'[àèéìíîòóùú]', text.lower()):
            return "it"  # Italian
        elif re.search(r'[ãõç]', text.lower()):
            return "pt"  # Portuguese
        elif re.search(r'[一-龯]', text):
            return "zh"  # Chinese
        elif re.search(r'[ひらがなカタカナ]', text):
            return "ja"  # Japanese
        elif re.search(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text):
            return "ko"  # Korean
        elif re.search(r'[а-яё]', text.lower()):
            return "ru"  # Russian
        elif re.search(r'[ا-ي]', text):
            return "ar"  # Arabic
        else:
            return "en"  # Default to English
    
    async def analyze_sentiment(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Analyze sentiment of the text."""
        try:
            results = {}
            
            # Use transformer model if available
            if 'sentiment' in self.pipelines:
                try:
                    sentiment_result = self.pipelines['sentiment'](text)
                    results['transformer'] = sentiment_result
                except Exception as e:
                    logger.warning(f"Transformer sentiment analysis failed: {e}")
            
            # Use TextBlob for sentiment analysis
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Classify sentiment
                if polarity > 0.1:
                    sentiment_label = "positive"
                elif polarity < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                results['textblob'] = {
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'sentiment': sentiment_label
                }
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis failed: {e}")
            
            # Use spaCy for additional analysis
            try:
                if language in self.models:
                    doc = self.models[language](text)
                    # Extract sentiment-related features
                    sentiment_words = [token.text for token in doc if token.pos_ == 'ADJ']
                    results['spacy'] = {
                        'sentiment_words': sentiment_words,
                        'word_count': len(sentiment_words)
                    }
            except Exception as e:
                logger.warning(f"spaCy sentiment analysis failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise NLPProcessingError(f"Sentiment analysis failed: {e}")
    
    async def extract_entities(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            entities = []
            
            # Use transformer NER model if available
            if 'ner' in self.pipelines:
                try:
                    ner_results = self.pipelines['ner'](text)
                    for entity in ner_results:
                        entities.append({
                            'text': entity['word'],
                            'label': entity['entity'],
                            'start': entity.get('start', 0),
                            'end': entity.get('end', len(entity['word'])),
                            'confidence': entity.get('score', 0.0)
                        })
                except Exception as e:
                    logger.warning(f"Transformer NER failed: {e}")
            
            # Use spaCy for entity extraction
            try:
                if language in self.models:
                    doc = self.models[language](text)
                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0  # spaCy doesn't provide confidence scores
                        })
            except Exception as e:
                logger.warning(f"spaCy NER failed: {e}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise NLPProcessingError(f"Entity extraction failed: {e}")
    
    async def extract_keywords(self, text: str, language: str = "en", top_k: int = 10) -> List[str]:
        """Extract keywords from text."""
        try:
            keywords = []
            
            # Use TF-IDF for keyword extraction
            try:
                vectorizer = self.vectorizers['tfidf']
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                # Get top keywords
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                keywords = [kw[0] for kw in keyword_scores[:top_k]]
                
            except Exception as e:
                logger.warning(f"TF-IDF keyword extraction failed: {e}")
            
            # Use spaCy for additional keyword extraction
            try:
                if language in self.models:
                    doc = self.models[language](text)
                    # Extract nouns and adjectives as keywords
                    spacy_keywords = [
                        token.lemma_.lower() 
                        for token in doc 
                        if token.pos_ in ['NOUN', 'ADJ'] 
                        and not token.is_stop 
                        and not token.is_punct
                        and len(token.text) > 2
                    ]
                    keywords.extend(spacy_keywords[:top_k])
                    
            except Exception as e:
                logger.warning(f"spaCy keyword extraction failed: {e}")
            
            # Remove duplicates and return top keywords
            unique_keywords = list(dict.fromkeys(keywords))
            return unique_keywords[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise NLPProcessingError(f"Keyword extraction failed: {e}")
    
    async def extract_topics(self, texts: List[str], language: str = "en", n_topics: int = 5) -> List[Dict[str, Any]]:
        """Extract topics from a collection of texts."""
        try:
            topics = []
            
            if not texts:
                return topics
            
            # Prepare texts for topic modeling
            processed_texts = []
            for text in texts:
                if language in self.models:
                    doc = self.models[language](text)
                    processed_text = " ".join([
                        token.lemma_.lower() 
                        for token in doc 
                        if not token.is_stop 
                        and not token.is_punct
                        and len(token.text) > 2
                    ])
                    processed_texts.append(processed_text)
                else:
                    processed_texts.append(text.lower())
            
            # Use LDA for topic modeling
            try:
                vectorizer = self.vectorizers['tfidf']
                tfidf_matrix = vectorizer.fit_transform(processed_texts)
                
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=100
                )
                lda.fit(tfidf_matrix)
                
                feature_names = vectorizer.get_feature_names_out()
                
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    top_weights = [topic[i] for i in top_words_idx]
                    
                    topics.append({
                        'id': topic_idx,
                        'words': top_words,
                        'weights': top_weights,
                        'coherence_score': 0.0  # Would need additional calculation
                    })
                    
            except Exception as e:
                logger.warning(f"LDA topic modeling failed: {e}")
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            raise NLPProcessingError(f"Topic extraction failed: {e}")
    
    async def calculate_readability(self, text: str, language: str = "en") -> float:
        """Calculate readability score for the text."""
        try:
            # Simple readability calculation based on sentence and word counts
            sentences = re.split(r'[.!?]+', text)
            words = text.split()
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Average words per sentence
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Average syllables per word (simplified)
            syllables = 0
            for word in words:
                syllables += self._count_syllables(word)
            avg_syllables_per_word = syllables / len(words)
            
            # Simple readability score (0-100, higher is more readable)
            readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            
            return max(0, min(100, readability))
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
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
    
    async def classify_text(self, text: str, categories: List[str], language: str = "en") -> Dict[str, float]:
        """Classify text into predefined categories."""
        try:
            # This would require training data and a trained classifier
            # For now, return a placeholder implementation
            results = {}
            for category in categories:
                # Simple keyword-based classification
                category_keywords = self._get_category_keywords(category)
                score = self._calculate_category_score(text, category_keywords)
                results[category] = score
            
            return results
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            raise NLPProcessingError(f"Text classification failed: {e}")
    
    def _get_category_keywords(self, category: str) -> List[str]:
        """Get keywords for a category."""
        category_keywords = {
            'business': ['company', 'revenue', 'profit', 'market', 'customer', 'sales'],
            'technical': ['software', 'code', 'system', 'database', 'api', 'development'],
            'marketing': ['campaign', 'advertising', 'brand', 'promotion', 'social media'],
            'finance': ['budget', 'investment', 'financial', 'cost', 'expense', 'revenue'],
            'legal': ['contract', 'agreement', 'law', 'compliance', 'regulation', 'terms']
        }
        return category_keywords.get(category.lower(), [])
    
    def _calculate_category_score(self, text: str, keywords: List[str]) -> float:
        """Calculate score for a category based on keyword presence."""
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return matches / len(keywords) if keywords else 0.0
    
    async def summarize_text(self, text: str, max_length: int = 150, language: str = "en") -> str:
        """Summarize text using extractive or abstractive methods."""
        try:
            # Use transformer summarization if available
            if 'summarization' in self.pipelines:
                try:
                    summary = self.pipelines['summarization'](
                        text, 
                        max_length=max_length,
                        min_length=30,
                        do_sample=False
                    )
                    return summary[0]['summary_text']
                except Exception as e:
                    logger.warning(f"Transformer summarization failed: {e}")
            
            # Fallback to extractive summarization
            return await self._extractive_summarization(text, max_length, language)
            
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            raise NLPProcessingError(f"Text summarization failed: {e}")
    
    async def _extractive_summarization(self, text: str, max_length: int, language: str) -> str:
        """Extractive summarization using sentence scoring."""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 1:
                return text
            
            # Score sentences based on word frequency
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    if word not in word_freq:
                        word_freq[word] = 0
                    word_freq[word] += 1
            
            # Score each sentence
            sentence_scores = []
            for sentence in sentences:
                words = sentence.lower().split()
                score = sum(word_freq.get(word, 0) for word in words)
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
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return text[:max_length]  # Fallback to truncation
    
    async def translate_text(self, text: str, target_language: str, source_language: str = "en") -> str:
        """Translate text to target language."""
        try:
            if 'translation' in self.pipelines:
                try:
                    translation = self.pipelines['translation'](text)
                    return translation[0]['translation_text']
                except Exception as e:
                    logger.warning(f"Transformer translation failed: {e}")
            
            # Fallback to simple translation (placeholder)
            return f"[Translated to {target_language}]: {text}"
            
        except Exception as e:
            logger.error(f"Text translation failed: {e}")
            raise NLPProcessingError(f"Text translation failed: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get NLP system health status."""
        try:
            health = {
                'initialized': self.is_initialized,
                'models_loaded': len(self.models),
                'pipelines_loaded': len(self.pipelines),
                'vectorizers_loaded': len(self.vectorizers),
                'classifiers_loaded': len(self.classifiers),
                'available_languages': list(self.models.keys()),
                'available_tasks': list(self.pipelines.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

# Global NLP system instance
nlp_system = NLPSystem()












