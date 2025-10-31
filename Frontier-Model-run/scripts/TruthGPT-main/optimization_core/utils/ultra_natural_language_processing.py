"""
Ultra-Advanced Natural Language Processing Module
=================================================

This module provides advanced NLP capabilities for TruthGPT models,
including text generation, sentiment analysis, named entity recognition, and language understanding.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings
import threading
import queue
import asyncio
from abc import ABC, abstractmethod
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForCausalLM
from transformers import pipeline, set_seed
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class NLPTask(Enum):
    """NLP tasks."""
    TEXT_GENERATION = "text_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"
    TEXT_SUMMARIZATION = "text_summarization"
    MACHINE_TRANSLATION = "machine_translation"
    TEXT_SIMILARITY = "text_similarity"
    LANGUAGE_MODELING = "language_modeling"
    TEXT_EMBEDDING = "text_embedding"

class LanguageModel(Enum):
    """Language models."""
    GPT2 = "gpt2"
    GPT3 = "gpt3"
    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    ALBERT = "albert"
    ELECTRA = "electra"
    DEBERTA = "deberta"
    T5 = "t5"
    BART = "bart"
    PEGASUS = "pegasus"
    CUSTOM = "custom"

class EmbeddingModel(Enum):
    """Embedding models."""
    WORD2VEC = "word2vec"
    GLOVE = "glove"
    FASTTEXT = "fasttext"
    BERT_EMBEDDINGS = "bert_embeddings"
    SENTENCE_BERT = "sentence_bert"
    UNIVERSAL_SENTENCE_ENCODER = "universal_sentence_encoder"
    CUSTOM_EMBEDDINGS = "custom_embeddings"

@dataclass
class NLPConfig:
    """Configuration for NLP."""
    task: NLPTask = NLPTask.TEXT_GENERATION
    model_name: str = "gpt2"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    num_beams: int = 4
    early_stopping: bool = True
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./nlp_results"

class TextProcessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        return sent_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                  lemmatize: bool = True) -> List[str]:
        """Complete text preprocessing pipeline."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return tokens

class TextGenerator:
    """Text generation using language models."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self, model_name: Optional[str] = None):
        """Load language model for text generation."""
        model_name = model_name or self.config.model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            logger.info(f"Loaded text generation model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_text(self, prompt: str, max_length: Optional[int] = None,
                     temperature: Optional[float] = None, top_p: Optional[float] = None,
                     top_k: Optional[int] = None, num_return_sequences: int = 1) -> List[str]:
        """Generate text from prompt."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=self.config.early_stopping
            )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def continue_text(self, text: str, max_new_tokens: int = 50) -> str:
        """Continue existing text."""
        generated_texts = self.generate_text(text, max_length=len(text.split()) + max_new_tokens)
        return generated_texts[0] if generated_texts else text

class SentimentAnalyzer:
    """Sentiment analysis using pre-trained models."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Load sentiment analysis model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded sentiment analysis model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model {model_name}: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Map class to sentiment
        sentiment_labels = ["negative", "neutral", "positive"]
        sentiment = sentiment_labels[predicted_class]
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0][0].item(),
                'neutral': probabilities[0][1].item(),
                'positive': probabilities[0][2].item()
            }
        }
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts."""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results

class NamedEntityRecognizer:
    """Named Entity Recognition using spaCy and transformers."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spacy_model = None
        self.transformer_model = None
        self.tokenizer = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_spacy_model(self, model_name: str = "en_core_web_sm"):
        """Load spaCy NER model."""
        try:
            self.spacy_model = spacy.load(model_name)
            logger.info(f"Loaded spaCy NER model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model {model_name}: {e}")
            # Try to download the model
            try:
                spacy.cli.download(model_name)
                self.spacy_model = spacy.load(model_name)
                logger.info(f"Downloaded and loaded spaCy NER model: {model_name}")
            except Exception as e2:
                logger.error(f"Failed to download spaCy model: {e2}")
                raise
    
    def load_transformer_model(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """Load transformer NER model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.transformer_model.to(self.device)
            logger.info(f"Loaded transformer NER model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load transformer NER model {model_name}: {e}")
            raise
    
    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy."""
        if self.spacy_model is None:
            raise ValueError("spaCy model not loaded. Call load_spacy_model() first.")
        
        doc = self.spacy_model(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0  # spaCy doesn't provide confidence scores
            })
        
        return entities
    
    def extract_entities_transformer(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using transformer model."""
        if self.transformer_model is None or self.tokenizer is None:
            raise ValueError("Transformer model not loaded. Call load_transformer_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.transformer_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Get tokens and labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = predictions[0].cpu().numpy()
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if token.startswith('##'):
                token = token[2:]  # Remove ## prefix
            
            # Map label to entity type
            label_name = self.transformer_model.config.id2label[label]
            
            if label_name.startswith('B-'):  # Beginning of entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': label_name[2:],
                    'start': i,
                    'end': i,
                    'confidence': 1.0
                }
            elif label_name.startswith('I-'):  # Inside entity
                if current_entity and current_entity['label'] == label_name[2:]:
                    current_entity['text'] += token
                    current_entity['end'] = i
            else:  # Outside entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def extract_entities(self, text: str, use_transformer: bool = True) -> List[Dict[str, Any]]:
        """Extract named entities using specified method."""
        if use_transformer:
            return self.extract_entities_transformer(text)
        else:
            return self.extract_entities_spacy(text)

class TextEmbedder:
    """Text embedding using various models."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.word2vec_model = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_bert_model(self, model_name: str = "bert-base-uncased"):
        """Load BERT model for embeddings."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded BERT embedding model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model {model_name}: {e}")
            raise
    
    def load_word2vec_model(self, model_path: Optional[str] = None):
        """Load Word2Vec model."""
        try:
            if model_path:
                self.word2vec_model = Word2Vec.load(model_path)
            else:
                # Create a simple Word2Vec model for demonstration
                sentences = [
                    ["this", "is", "a", "sample", "sentence"],
                    ["another", "example", "sentence", "for", "training"],
                    ["word2vec", "is", "useful", "for", "embeddings"]
                ]
                self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            
            logger.info("Loaded Word2Vec model")
            
        except Exception as e:
            logger.error(f"Failed to load Word2Vec model: {e}")
            raise
    
    def get_bert_embeddings(self, text: str, pooling_strategy: str = "mean") -> np.ndarray:
        """Get BERT embeddings for text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model not loaded. Call load_bert_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                              padding=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Apply pooling strategy
        if pooling_strategy == "mean":
            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
        elif pooling_strategy == "cls":
            # CLS token pooling
            pooled_embeddings = embeddings[:, 0, :]
        else:
            # Default to mean pooling
            pooled_embeddings = embeddings.mean(dim=1)
        
        return pooled_embeddings.cpu().numpy()
    
    def get_word2vec_embeddings(self, text: str) -> np.ndarray:
        """Get Word2Vec embeddings for text."""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not loaded. Call load_word2vec_model() first.")
        
        # Tokenize text
        tokens = text.lower().split()
        
        # Get embeddings for each token
        embeddings = []
        for token in tokens:
            if token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
        
        if not embeddings:
            # Return zero vector if no tokens found
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average embeddings
        return np.mean(embeddings, axis=0)
    
    def get_embeddings(self, text: str, method: str = "bert") -> np.ndarray:
        """Get embeddings using specified method."""
        if method == "bert":
            return self.get_bert_embeddings(text)
        elif method == "word2vec":
            return self.get_word2vec_embeddings(text)
        else:
            raise ValueError(f"Unsupported embedding method: {method}")

class QuestionAnswerer:
    """Question answering using transformer models."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_model(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """Load question answering model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded QA model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load QA model {model_name}: {e}")
            raise
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer a question given a context."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(question, context, return_tensors="pt", 
                              truncation=True, padding=True, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # Get answer span
        start_idx = torch.argmax(start_scores, dim=1).item()
        end_idx = torch.argmax(end_scores, dim=1).item()
        
        # Convert tokens back to text
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        answer_tokens = tokens[start_idx:end_idx + 1]
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)
        
        # Calculate confidence
        start_confidence = torch.softmax(start_scores, dim=1)[0][start_idx].item()
        end_confidence = torch.softmax(end_scores, dim=1)[0][end_idx].item()
        confidence = (start_confidence + end_confidence) / 2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'start_position': start_idx,
            'end_position': end_idx
        }

class NLPManager:
    """Main manager for NLP tasks."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.generators = {}
        self.sentiment_analyzers = {}
        self.ner_models = {}
        self.embedders = {}
        self.qa_models = {}
        self.processing_history = deque(maxlen=1000)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_text_generator(self, generator_id: str, model_name: Optional[str] = None) -> TextGenerator:
        """Create text generator."""
        generator = TextGenerator(self.config)
        generator.load_model(model_name)
        self.generators[generator_id] = generator
        return generator
    
    def create_sentiment_analyzer(self, analyzer_id: str, model_name: Optional[str] = None) -> SentimentAnalyzer:
        """Create sentiment analyzer."""
        analyzer = SentimentAnalyzer(self.config)
        analyzer.load_model(model_name)
        self.sentiment_analyzers[analyzer_id] = analyzer
        return analyzer
    
    def create_ner_model(self, ner_id: str, use_transformer: bool = True) -> NamedEntityRecognizer:
        """Create NER model."""
        ner = NamedEntityRecognizer(self.config)
        if use_transformer:
            ner.load_transformer_model()
        else:
            ner.load_spacy_model()
        self.ner_models[ner_id] = ner
        return ner
    
    def create_text_embedder(self, embedder_id: str, method: str = "bert") -> TextEmbedder:
        """Create text embedder."""
        embedder = TextEmbedder(self.config)
        if method == "bert":
            embedder.load_bert_model()
        elif method == "word2vec":
            embedder.load_word2vec_model()
        self.embedders[embedder_id] = embedder
        return embedder
    
    def create_qa_model(self, qa_id: str, model_name: Optional[str] = None) -> QuestionAnswerer:
        """Create question answering model."""
        qa = QuestionAnswerer(self.config)
        qa.load_model(model_name)
        self.qa_models[qa_id] = qa
        return qa
    
    def generate_text(self, generator_id: str, prompt: str, **kwargs) -> List[str]:
        """Generate text using specified generator."""
        if generator_id not in self.generators:
            raise ValueError(f"Generator {generator_id} not found")
        
        generator = self.generators[generator_id]
        generated_texts = generator.generate_text(prompt, **kwargs)
        
        # Record processing
        self.processing_history.append({
            'task': 'text_generation',
            'generator_id': generator_id,
            'prompt_length': len(prompt),
            'generated_length': len(generated_texts[0]) if generated_texts else 0,
            'timestamp': time.time()
        })
        
        return generated_texts
    
    def analyze_sentiment(self, analyzer_id: str, text: str) -> Dict[str, Any]:
        """Analyze sentiment using specified analyzer."""
        if analyzer_id not in self.sentiment_analyzers:
            raise ValueError(f"Sentiment analyzer {analyzer_id} not found")
        
        analyzer = self.sentiment_analyzers[analyzer_id]
        result = analyzer.analyze_sentiment(text)
        
        # Record processing
        self.processing_history.append({
            'task': 'sentiment_analysis',
            'analyzer_id': analyzer_id,
            'text_length': len(text),
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'timestamp': time.time()
        })
        
        return result
    
    def extract_entities(self, ner_id: str, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using specified NER model."""
        if ner_id not in self.ner_models:
            raise ValueError(f"NER model {ner_id} not found")
        
        ner = self.ner_models[ner_id]
        entities = ner.extract_entities(text)
        
        # Record processing
        self.processing_history.append({
            'task': 'named_entity_recognition',
            'ner_id': ner_id,
            'text_length': len(text),
            'num_entities': len(entities),
            'timestamp': time.time()
        })
        
        return entities
    
    def get_text_embeddings(self, embedder_id: str, text: str, method: str = "bert") -> np.ndarray:
        """Get text embeddings using specified embedder."""
        if embedder_id not in self.embedders:
            raise ValueError(f"Embedder {embedder_id} not found")
        
        embedder = self.embedders[embedder_id]
        embeddings = embedder.get_embeddings(text, method)
        
        # Record processing
        self.processing_history.append({
            'task': 'text_embedding',
            'embedder_id': embedder_id,
            'text_length': len(text),
            'embedding_dim': len(embeddings),
            'timestamp': time.time()
        })
        
        return embeddings
    
    def answer_question(self, qa_id: str, question: str, context: str) -> Dict[str, Any]:
        """Answer question using specified QA model."""
        if qa_id not in self.qa_models:
            raise ValueError(f"QA model {qa_id} not found")
        
        qa = self.qa_models[qa_id]
        result = qa.answer_question(question, context)
        
        # Record processing
        self.processing_history.append({
            'task': 'question_answering',
            'qa_id': qa_id,
            'question_length': len(question),
            'context_length': len(context),
            'answer_length': len(result['answer']),
            'confidence': result['confidence'],
            'timestamp': time.time()
        })
        
        return result
    
    def get_nlp_statistics(self) -> Dict[str, Any]:
        """Get NLP statistics."""
        return {
            'task': self.config.task.value,
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'num_generators': len(self.generators),
            'num_sentiment_analyzers': len(self.sentiment_analyzers),
            'num_ner_models': len(self.ner_models),
            'num_embedders': len(self.embedders),
            'num_qa_models': len(self.qa_models),
            'processing_history_size': len(self.processing_history),
            'config': {
                'temperature': self.config.temperature,
                'top_p': self.config.top_p,
                'top_k': self.config.top_k,
                'repetition_penalty': self.config.repetition_penalty
            }
        }

# Factory functions
def create_nlp_config(task: NLPTask = NLPTask.TEXT_GENERATION,
                     model_name: str = "gpt2",
                     **kwargs) -> NLPConfig:
    """Create NLP configuration."""
    return NLPConfig(
        task=task,
        model_name=model_name,
        **kwargs
    )

def create_text_processor() -> TextProcessor:
    """Create text processor."""
    return TextProcessor()

def create_text_generator(config: NLPConfig) -> TextGenerator:
    """Create text generator."""
    return TextGenerator(config)

def create_sentiment_analyzer(config: NLPConfig) -> SentimentAnalyzer:
    """Create sentiment analyzer."""
    return SentimentAnalyzer(config)

def create_named_entity_recognizer(config: NLPConfig) -> NamedEntityRecognizer:
    """Create named entity recognizer."""
    return NamedEntityRecognizer(config)

def create_text_embedder(config: NLPConfig) -> TextEmbedder:
    """Create text embedder."""
    return TextEmbedder(config)

def create_question_answerer(config: NLPConfig) -> QuestionAnswerer:
    """Create question answerer."""
    return QuestionAnswerer(config)

def create_nlp_manager(config: Optional[NLPConfig] = None) -> NLPManager:
    """Create NLP manager."""
    if config is None:
        config = create_nlp_config()
    return NLPManager(config)

# Example usage
def example_nlp():
    """Example of NLP functionality."""
    # Create configuration
    config = create_nlp_config(
        task=NLPTask.TEXT_GENERATION,
        model_name="gpt2",
        max_length=100
    )
    
    # Create manager
    manager = create_nlp_manager(config)
    
    # Create text generator
    generator = manager.create_text_generator("gpt2_generator")
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    generated_texts = manager.generate_text("gpt2_generator", prompt, max_length=50)
    print(f"Generated text: {generated_texts[0]}")
    
    # Create sentiment analyzer
    analyzer = manager.create_sentiment_analyzer("sentiment_analyzer")
    
    # Analyze sentiment
    text = "I love this new AI technology!"
    sentiment_result = manager.analyze_sentiment("sentiment_analyzer", text)
    print(f"Sentiment: {sentiment_result['sentiment']} (confidence: {sentiment_result['confidence']:.2f})")
    
    # Get statistics
    stats = manager.get_nlp_statistics()
    print(f"Statistics: {stats}")
    
    return generated_texts, sentiment_result

if __name__ == "__main__":
    # Run example
    example_nlp()
