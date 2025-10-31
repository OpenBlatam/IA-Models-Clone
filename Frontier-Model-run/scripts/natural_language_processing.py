#!/usr/bin/env python3
"""
Advanced Natural Language Processing System for Frontier Model Training
Provides comprehensive NLP algorithms, text processing, and language understanding capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import pipeline, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
import nltk
import spacy
import textblob
import gensim
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models import LdaModel, LsiModel, HdpModel
import sentence_transformers
from sentence_transformers import SentenceTransformer
import openai
import anthropic
import cohere
import joblib
import pickle
from collections import defaultdict, Counter
import re
import string
import warnings
warnings.filterwarnings('ignore')

console = Console()

class NLPTask(Enum):
    """NLP tasks."""
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    PART_OF_SPEECH_TAGGING = "part_of_speech_tagging"
    QUESTION_ANSWERING = "question_answering"
    TEXT_SUMMARIZATION = "text_summarization"
    MACHINE_TRANSLATION = "machine_translation"
    TEXT_GENERATION = "text_generation"
    TEXT_SIMILARITY = "text_similarity"
    TEXT_CLUSTERING = "text_clustering"
    TOPIC_MODELING = "topic_modeling"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TEXT_PREPROCESSING = "text_preprocessing"
    LANGUAGE_DETECTION = "language_detection"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"

class NLPModel(Enum):
    """NLP models."""
    # Transformer models
    BERT_BASE = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    ROBERTA_BASE = "roberta-base"
    ROBERTA_LARGE = "roberta-large"
    DISTILBERT = "distilbert-base-uncased"
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    T5_BASE = "t5-base"
    T5_LARGE = "t5-large"
    BART_BASE = "facebook/bart-base"
    BART_LARGE = "facebook/bart-large"
    
    # Specialized models
    SENTIMENT_BERT = "nlptown/bert-base-multilingual-uncased-sentiment"
    NER_BERT = "dbmdz/bert-large-cased-finetuned-conll03-english"
    QA_BERT = "deepset/roberta-base-squad2"
    SUMMARIZATION_BART = "facebook/bart-large-cnn"
    TRANSLATION_T5 = "t5-base"
    
    # Embedding models
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    UNIVERSAL_SENTENCE_ENCODER = "sentence-transformers/all-mpnet-base-v2"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"

class TextPreprocessing(Enum):
    """Text preprocessing operations."""
    TOKENIZATION = "tokenization"
    LOWER_CASING = "lower_casing"
    REMOVE_PUNCTUATION = "remove_punctuation"
    REMOVE_NUMBERS = "remove_numbers"
    REMOVE_STOPWORDS = "remove_stopwords"
    LEMMATIZATION = "lemmatization"
    STEMMING = "stemming"
    SPELLING_CORRECTION = "spelling_correction"
    REMOVE_HTML = "remove_html"
    REMOVE_URLS = "remove_urls"
    REMOVE_EMAILS = "remove_emails"
    REMOVE_SPECIAL_CHARS = "remove_special_chars"
    NORMALIZE_WHITESPACE = "normalize_whitespace"
    REMOVE_EMOJIS = "remove_emojis"
    EXPAND_CONTRACTIONS = "expand_contractions"

class EmbeddingMethod(Enum):
    """Embedding methods."""
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    BERT = "bert"
    ROBERTA = "roberta"
    SENTENCE_BERT = "sentence_bert"
    UNIVERSAL_SENTENCE_ENCODER = "universal_sentence_encoder"
    ELMO = "elmo"
    GPT = "gpt"
    T5 = "t5"

@dataclass
class NLPConfig:
    """NLP configuration."""
    task: NLPTask = NLPTask.TEXT_CLASSIFICATION
    model: NLPModel = NLPModel.BERT_BASE
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    num_labels: int = 2
    pretrained: bool = True
    preprocessing_steps: List[TextPreprocessing] = None
    embedding_method: EmbeddingMethod = EmbeddingMethod.BERT
    enable_data_augmentation: bool = True
    enable_early_stopping: bool = True
    enable_gradient_accumulation: bool = True
    enable_mixed_precision: bool = True
    enable_model_checkpointing: bool = True
    enable_tensorboard_logging: bool = True
    enable_visualization: bool = True
    enable_multilingual: bool = True
    device: str = "auto"

@dataclass
class TextData:
    """Text data container."""
    text_id: str
    text: str
    label: Optional[Any] = None
    metadata: Dict[str, Any] = None
    processed_text: Optional[str] = None
    embeddings: Optional[np.ndarray] = None
    tokens: Optional[List[str]] = None

@dataclass
class NLPModelResult:
    """NLP model result."""
    result_id: str
    task: NLPTask
    model: NLPModel
    performance_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    model_state: Dict[str, Any]
    created_at: datetime

class TextPreprocessor:
    """Text preprocessing engine."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
        
        # Initialize stopwords
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
    
    def preprocess_text(self, text: str, steps: List[TextPreprocessing] = None) -> str:
        """Preprocess text with specified steps."""
        if steps is None:
            steps = self.config.preprocessing_steps or [
                TextPreprocessing.LOWER_CASING,
                TextPreprocessing.REMOVE_PUNCTUATION,
                TextPreprocessing.REMOVE_STOPWORDS,
                TextPreprocessing.LEMMATIZATION
            ]
        
        processed_text = text
        
        for step in steps:
            if step == TextPreprocessing.LOWER_CASING:
                processed_text = processed_text.lower()
            elif step == TextPreprocessing.REMOVE_PUNCTUATION:
                processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
            elif step == TextPreprocessing.REMOVE_NUMBERS:
                processed_text = re.sub(r'\d+', '', processed_text)
            elif step == TextPreprocessing.REMOVE_STOPWORDS:
                processed_text = self._remove_stopwords(processed_text)
            elif step == TextPreprocessing.LEMMATIZATION:
                processed_text = self._lemmatize_text(processed_text)
            elif step == TextPreprocessing.STEMMING:
                processed_text = self._stem_text(processed_text)
            elif step == TextPreprocessing.REMOVE_HTML:
                processed_text = re.sub(r'<[^>]+>', '', processed_text)
            elif step == TextPreprocessing.REMOVE_URLS:
                processed_text = re.sub(r'http\S+|www\S+|https\S+', '', processed_text)
            elif step == TextPreprocessing.REMOVE_EMAILS:
                processed_text = re.sub(r'\S+@\S+', '', processed_text)
            elif step == TextPreprocessing.REMOVE_SPECIAL_CHARS:
                processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
            elif step == TextPreprocessing.NORMALIZE_WHITESPACE:
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            elif step == TextPreprocessing.REMOVE_EMOJIS:
                processed_text = self._remove_emojis(processed_text)
            elif step == TextPreprocessing.EXPAND_CONTRACTIONS:
                processed_text = self._expand_contractions(processed_text)
        
        return processed_text
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize text using spaCy."""
        if self.nlp:
            doc = self.nlp(text)
            return ' '.join([token.lemma_ for token in doc])
        else:
            # Fallback to simple lemmatization
            return text
    
    def _stem_text(self, text: str) -> str:
        """Stem text using NLTK."""
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            words = text.split()
            stemmed_words = [stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        except:
            return text
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'m": " am",
            "'s": " is",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text."""
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except:
            return text.split()

class EmbeddingGenerator:
    """Text embedding generator."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models."""
        try:
            if self.config.embedding_method == EmbeddingMethod.BERT:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.value)
                self.model = AutoModel.from_pretrained(self.config.model.value)
                self.model.to(self.device)
                self.model.eval()
            elif self.config.embedding_method == EmbeddingMethod.SENTENCE_BERT:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            elif self.config.embedding_method == EmbeddingMethod.WORD2VEC:
                # Will be loaded when needed
                pass
            elif self.config.embedding_method == EmbeddingMethod.FASTTEXT:
                # Will be loaded when needed
                pass
            
            console.print(f"[green]{self.config.embedding_method.value} model initialized[/green]")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        console.print(f"[blue]Generating embeddings using {self.config.embedding_method.value}...[/blue]")
        
        try:
            if self.config.embedding_method == EmbeddingMethod.BERT:
                return self._generate_bert_embeddings(texts)
            elif self.config.embedding_method == EmbeddingMethod.SENTENCE_BERT:
                return self._generate_sentence_bert_embeddings(texts)
            elif self.config.embedding_method == EmbeddingMethod.WORD2VEC:
                return self._generate_word2vec_embeddings(texts)
            elif self.config.embedding_method == EmbeddingMethod.FASTTEXT:
                return self._generate_fasttext_embeddings(texts)
            else:
                return self._generate_bert_embeddings(texts)
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return np.random.randn(len(texts), 768)  # Fallback embeddings
    
    def _generate_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings."""
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def _generate_sentence_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate Sentence-BERT embeddings."""
        return self.model.encode(texts)
    
    def _generate_word2vec_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate Word2Vec embeddings."""
        # Simplified Word2Vec implementation
        from gensim.models import Word2Vec
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec model
        model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        
        # Generate embeddings
        embeddings = []
        for text in tokenized_texts:
            if text:
                # Average word embeddings
                word_embeddings = [model.wv[word] for word in text if word in model.wv]
                if word_embeddings:
                    embedding = np.mean(word_embeddings, axis=0)
                else:
                    embedding = np.zeros(100)
            else:
                embedding = np.zeros(100)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _generate_fasttext_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate FastText embeddings."""
        # Simplified FastText implementation
        from gensim.models import FastText
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train FastText model
        model = FastText(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
        
        # Generate embeddings
        embeddings = []
        for text in tokenized_texts:
            if text:
                # Average word embeddings
                word_embeddings = [model.wv[word] for word in text if word in model.wv]
                if word_embeddings:
                    embedding = np.mean(word_embeddings, axis=0)
                else:
                    embedding = np.zeros(100)
            else:
                embedding = np.zeros(100)
            embeddings.append(embedding)
        
        return np.array(embeddings)

class NLPModelFactory:
    """NLP model factory."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_model(self) -> Any:
        """Create NLP model."""
        console.print(f"[blue]Creating {self.config.model.value} model for {self.config.task.value}...[/blue]")
        
        try:
            if self.config.task == NLPTask.TEXT_CLASSIFICATION:
                return self._create_classification_model()
            elif self.config.task == NLPTask.SENTIMENT_ANALYSIS:
                return self._create_sentiment_model()
            elif self.config.task == NLPTask.NAMED_ENTITY_RECOGNITION:
                return self._create_ner_model()
            elif self.config.task == NLPTask.QUESTION_ANSWERING:
                return self._create_qa_model()
            elif self.config.task == NLPTask.TEXT_SUMMARIZATION:
                return self._create_summarization_model()
            elif self.config.task == NLPTask.TEXT_GENERATION:
                return self._create_generation_model()
            else:
                return self._create_classification_model()
                
        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            return self._create_fallback_model()
    
    def _create_classification_model(self) -> Any:
        """Create text classification model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.value,
            num_labels=self.config.num_labels
        )
        return model.to(self.device)
    
    def _create_sentiment_model(self) -> Any:
        """Create sentiment analysis model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.value,
            num_labels=3  # Positive, Negative, Neutral
        )
        return model.to(self.device)
    
    def _create_ner_model(self) -> Any:
        """Create named entity recognition model."""
        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model.value,
            num_labels=9  # Common NER labels
        )
        return model.to(self.device)
    
    def _create_qa_model(self) -> Any:
        """Create question answering model."""
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.config.model.value
        )
        return model.to(self.device)
    
    def _create_summarization_model(self) -> Any:
        """Create text summarization model."""
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.value
        )
        return model.to(self.device)
    
    def _create_generation_model(self) -> Any:
        """Create text generation model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.value
        )
        return model.to(self.device)
    
    def _create_fallback_model(self) -> Any:
        """Create fallback model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.config.num_labels
        )
        return model.to(self.device)

class NLPTrainer:
    """NLP training engine."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def train_model(self, model: Any, train_dataset: Dataset, 
                   val_dataset: Dataset) -> Dict[str, Any]:
        """Train NLP model."""
        console.print(f"[blue]Training {self.config.model.value} model...[/blue]")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.value)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./nlp_results',
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        eval_results = trainer.evaluate()
        
        return {
            'training_history': trainer.state.log_history,
            'eval_results': eval_results,
            'best_model_path': training_args.output_dir
        }

class NLPSystem:
    """Main NLP system."""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.model_factory = NLPModelFactory(config)
        self.trainer = NLPTrainer(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.nlp_results: Dict[str, NLPModelResult] = {}
    
    def _init_database(self) -> str:
        """Initialize NLP database."""
        db_path = Path("./natural_language_processing.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nlp_models (
                    model_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    training_history TEXT NOT NULL,
                    model_state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_nlp_experiment(self, texts: List[str], labels: List[int] = None) -> NLPModelResult:
        """Run complete NLP experiment."""
        console.print(f"[blue]Starting NLP experiment with {self.config.task.value}...[/blue]")
        
        start_time = time.time()
        result_id = f"nlp_exp_{int(time.time())}"
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed_text = self.text_preprocessor.preprocess_text(text)
            processed_texts.append(processed_text)
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(processed_texts)
        
        # Create model
        model = self.model_factory.create_model()
        
        # Create datasets
        train_dataset, val_dataset = self._create_datasets(processed_texts, labels)
        
        # Train model
        training_result = self.trainer.train_model(model, train_dataset, val_dataset)
        
        # Evaluate model
        performance_metrics = self._evaluate_model(model, val_dataset)
        
        # Create NLP result
        nlp_result = NLPModelResult(
            result_id=result_id,
            task=self.config.task,
            model=self.config.model,
            performance_metrics=performance_metrics,
            training_history=training_result.get('training_history', {}),
            model_state={
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
                'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 768
            },
            created_at=datetime.now()
        )
        
        # Store result
        self.nlp_results[result_id] = nlp_result
        
        # Save to database
        self._save_nlp_result(nlp_result)
        
        experiment_time = time.time() - start_time
        console.print(f"[green]NLP experiment completed in {experiment_time:.2f} seconds[/green]")
        console.print(f"[blue]Final accuracy: {performance_metrics.get('accuracy', 0):.4f}[/blue]")
        
        return nlp_result
    
    def _create_datasets(self, texts: List[str], labels: List[int]) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels or [0] * len(texts)
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        # Split data
        split_idx = int(0.8 * len(texts))
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx] if labels else None
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:] if labels else None
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.value)
        
        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, tokenizer, self.config.max_length)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer, self.config.max_length)
        
        return train_dataset, val_dataset
    
    def _evaluate_model(self, model: Any, dataset: Dataset) -> Dict[str, float]:
        """Evaluate model performance."""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for item in dataset:
                input_ids = item['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)
                labels = item['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _save_nlp_result(self, result: NLPModelResult):
        """Save NLP result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO nlp_models 
                (model_id, task, model_name, performance_metrics,
                 training_history, model_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.task.value,
                result.model.value,
                json.dumps(result.performance_metrics),
                json.dumps(result.training_history),
                json.dumps(result.model_state),
                result.created_at.isoformat()
            ))
    
    def visualize_nlp_results(self, result: NLPModelResult, 
                            output_path: str = None) -> str:
        """Visualize NLP training results."""
        if output_path is None:
            output_path = f"nlp_training_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        performance_metrics = result.performance_metrics
        metric_names = list(performance_metrics.keys())
        metric_values = list(performance_metrics.values())
        
        axes[0, 0].bar(metric_names, metric_values)
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Model information
        model_state = result.model_state
        info_names = list(model_state.keys())
        info_values = list(model_state.values())
        
        axes[0, 1].bar(info_names, info_values)
        axes[0, 1].set_title('Model Information')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training history (simplified)
        if result.training_history:
            epochs = list(range(len(result.training_history)))
            losses = [h.get('train_loss', 0) for h in result.training_history]
            
            axes[1, 0].plot(epochs, losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Task and model info
        task_model_info = {
            'Task': len(result.task.value),
            'Model': len(result.model.value),
            'Parameters': result.model_state.get('num_parameters', 0) // 1000000,  # In millions
            'Size (MB)': result.model_state.get('model_size_mb', 0)
        }
        
        info_names = list(task_model_info.keys())
        info_values = list(task_model_info.values())
        
        axes[1, 1].bar(info_names, info_values)
        axes[1, 1].set_title('Task and Model Info')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]NLP visualization saved: {output_path}[/green]")
        return output_path
    
    def get_nlp_summary(self) -> Dict[str, Any]:
        """Get NLP system summary."""
        if not self.nlp_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.nlp_results)
        
        # Calculate average metrics
        accuracies = [result.performance_metrics.get('accuracy', 0) for result in self.nlp_results.values()]
        f1_scores = [result.performance_metrics.get('f1_score', 0) for result in self.nlp_results.values()]
        
        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)
        
        # Best performing experiment
        best_result = max(self.nlp_results.values(), 
                         key=lambda x: x.performance_metrics.get('accuracy', 0))
        
        return {
            'total_experiments': total_experiments,
            'average_accuracy': avg_accuracy,
            'average_f1_score': avg_f1,
            'best_accuracy': best_result.performance_metrics.get('accuracy', 0),
            'best_experiment_id': best_result.result_id,
            'tasks_used': list(set(result.task.value for result in self.nlp_results.values())),
            'models_used': list(set(result.model.value for result in self.nlp_results.values()))
        }

def main():
    """Main function for NLP CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Natural Language Processing System")
    parser.add_argument("--task", type=str,
                       choices=["text_classification", "sentiment_analysis", "named_entity_recognition", "question_answering"],
                       default="text_classification", help="NLP task")
    parser.add_argument("--model", type=str,
                       choices=["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
                       default="bert-base-uncased", help="NLP model")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--num-labels", type=int, default=2,
                       help="Number of labels")
    parser.add_argument("--embedding-method", type=str,
                       choices=["bert", "sentence_bert", "word2vec", "fasttext"],
                       default="bert", help="Embedding method")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create NLP configuration
    config = NLPConfig(
        task=NLPTask(args.task),
        model=NLPModel(args.model),
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_labels=args.num_labels,
        embedding_method=EmbeddingMethod(args.embedding_method),
        device=args.device
    )
    
    # Create NLP system
    nlp_system = NLPSystem(config)
    
    # Create sample data
    sample_texts = [
        "This is a great product!",
        "I love this movie.",
        "The service was terrible.",
        "Amazing experience!",
        "Not worth the money.",
        "Highly recommended!",
        "Poor quality product.",
        "Excellent customer service.",
        "Waste of time.",
        "Outstanding performance!"
    ]
    
    sample_labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1: positive, 0: negative
    
    # Run NLP experiment
    result = nlp_system.run_nlp_experiment(sample_texts, sample_labels)
    
    # Show results
    console.print(f"[green]NLP experiment completed[/green]")
    console.print(f"[blue]Task: {result.task.value}[/blue]")
    console.print(f"[blue]Model: {result.model.value}[/blue]")
    console.print(f"[blue]Final accuracy: {result.performance_metrics.get('accuracy', 0):.4f}[/blue]")
    
    # Create visualization
    nlp_system.visualize_nlp_results(result)
    
    # Show summary
    summary = nlp_system.get_nlp_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
