"""
Natural Language Processing Engine for Export IA
Advanced NLP with transformers, embeddings, and language models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import json
import random
from pathlib import Path
from collections import defaultdict, deque
import copy
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, pipeline, BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
)
import sentence_transformers
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class NLPConfig:
    """Configuration for natural language processing"""
    # Model types
    model_type: str = "transformer"  # transformer, bert, gpt, t5, custom
    
    # Transformer parameters
    transformer_model: str = "bert-base-uncased"  # bert-base-uncased, roberta-base, distilbert-base-uncased
    transformer_max_length: int = 512
    transformer_padding: str = "max_length"
    transformer_truncation: bool = True
    
    # BERT parameters
    bert_model: str = "bert-base-uncased"
    bert_layers: List[int] = None  # Which layers to use for embeddings
    
    # GPT parameters
    gpt_model: str = "gpt2"
    gpt_max_length: int = 1024
    gpt_temperature: float = 1.0
    gpt_top_p: float = 0.9
    gpt_top_k: int = 50
    gpt_repetition_penalty: float = 1.0
    
    # T5 parameters
    t5_model: str = "t5-base"
    t5_max_length: int = 512
    t5_num_beams: int = 4
    
    # Sentence Transformers parameters
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    sentence_transformer_dimension: int = 384
    
    # Text preprocessing
    enable_preprocessing: bool = True
    remove_stopwords: bool = True
    enable_stemming: bool = True
    enable_lemmatization: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True
    
    # Language detection
    enable_language_detection: bool = True
    supported_languages: List[str] = None  # ["en", "es", "fr", "de", "it"]
    
    # Sentiment analysis
    enable_sentiment_analysis: bool = True
    sentiment_model: str = "vader"  # vader, transformer, custom
    
    # Named Entity Recognition
    enable_ner: bool = True
    ner_model: str = "spacy"  # spacy, transformer, custom
    
    # Text classification
    enable_text_classification: bool = True
    classification_model: str = "transformer"
    classification_labels: List[str] = None
    
    # Question Answering
    enable_qa: bool = True
    qa_model: str = "transformer"
    
    # Text summarization
    enable_summarization: bool = True
    summarization_model: str = "t5"
    summarization_max_length: int = 150
    summarization_min_length: int = 30
    
    # Text generation
    enable_text_generation: bool = True
    generation_model: str = "gpt"
    generation_max_length: int = 100
    
    # Embeddings
    enable_embeddings: bool = True
    embedding_model: str = "sentence_transformer"
    embedding_dimension: int = 384
    
    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass
            
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            
    def preprocess(self, text: str) -> str:
        """Preprocess text"""
        
        if not self.config.enable_preprocessing:
            return text
            
        # Convert to lowercase
        if self.config.lowercase:
            text = text.lower()
            
        # Remove punctuation
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        # Remove stopwords
        if self.config.remove_stopwords:
            words = word_tokenize(text)
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
            
        # Stemming
        if self.config.enable_stemming:
            words = word_tokenize(text)
            words = [self.stemmer.stem(word) for word in words]
            text = ' '.join(words)
            
        # Lemmatization
        if self.config.enable_lemmatization:
            words = word_tokenize(text)
            words = [self.lemmatizer.lemmatize(word) for word in words]
            text = ' '.join(words)
            
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        
        return word_tokenize(text)
        
    def sent_tokenize(self, text: str) -> List[str]:
        """Sentence tokenization"""
        
        return sent_tokenize(text)

class SentimentAnalyzer:
    """Sentiment analysis using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.sentiment_model == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
        elif config.sentiment_model == "transformer":
            self.model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            raise ValueError(f"Unsupported sentiment model: {config.sentiment_model}")
            
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment"""
        
        if self.config.sentiment_model == "vader":
            scores = self.analyzer.polarity_scores(text)
            
            # Determine sentiment
            if scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'sentiment': sentiment,
                'scores': scores,
                'confidence': abs(scores['compound'])
            }
            
        elif self.config.sentiment_model == "transformer":
            result = self.model(text)[0]
            
            return {
                'sentiment': result['label'].lower(),
                'confidence': result['score'],
                'scores': {result['label'].lower(): result['score']}
            }

class NamedEntityRecognizer:
    """Named Entity Recognition using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.ner_model == "spacy":
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
        elif config.ner_model == "transformer":
            self.model = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            raise ValueError(f"Unsupported NER model: {config.ner_model}")
            
    def recognize(self, text: str) -> List[Dict[str, Any]]:
        """Recognize named entities"""
        
        entities = []
        
        if self.config.ner_model == "spacy" and self.nlp:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                }
                entities.append(entity)
                
        elif self.config.ner_model == "transformer":
            results = self.model(text)
            
            for result in results:
                entity = {
                    'text': result['word'],
                    'label': result['entity'],
                    'start': result['start'],
                    'end': result['end'],
                    'confidence': result['score']
                }
                entities.append(entity)
                
        return entities

class TextClassifier:
    """Text classification using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.classification_model == "transformer":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-emotion"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cardiffnlp/twitter-roberta-base-emotion"
            )
        else:
            raise ValueError(f"Unsupported classification model: {config.classification_model}")
            
    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.transformer_max_length,
            padding=self.config.transformer_padding,
            truncation=self.config.transformer_truncation
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
        # Get predictions
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        
        # Get class labels
        if hasattr(self.model.config, 'id2label'):
            class_label = self.model.config.id2label[predicted_class_id]
        else:
            class_label = f"class_{predicted_class_id}"
            
        return {
            'class_id': predicted_class_id,
            'class_label': class_label,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }

class QuestionAnswerer:
    """Question Answering using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.qa_model == "transformer":
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                "distilbert-base-cased-distilled-squad"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-cased-distilled-squad"
            )
        else:
            raise ValueError(f"Unsupported QA model: {config.qa_model}")
            
    def answer(self, question: str, context: str) -> Dict[str, Any]:
        """Answer question based on context"""
        
        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=self.config.transformer_max_length,
            padding=self.config.transformer_padding,
            truncation=self.config.transformer_truncation
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
        # Get answer span
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()
        
        # Extract answer
        input_ids = inputs['input_ids'][0]
        answer_tokens = input_ids[start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence
        start_confidence = F.softmax(start_logits, dim=-1)[0, start_idx].item()
        end_confidence = F.softmax(end_logits, dim=-1)[0, end_idx].item()
        confidence = (start_confidence + end_confidence) / 2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'start_position': start_idx,
            'end_position': end_idx
        }

class TextSummarizer:
    """Text summarization using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.summarization_model == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        else:
            raise ValueError(f"Unsupported summarization model: {config.summarization_model}")
            
    def summarize(self, text: str) -> Dict[str, Any]:
        """Summarize text"""
        
        # Prepare input
        input_text = f"summarize: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config.transformer_max_length,
            padding=self.config.transformer_padding,
            truncation=self.config.transformer_truncation
        )
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.summarization_max_length,
                min_length=self.config.summarization_min_length,
                num_beams=self.config.t5_num_beams,
                early_stopping=True
            )
            
        # Decode summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(text)
        }

class TextGenerator:
    """Text generation using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.generation_model == "gpt":
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            raise ValueError(f"Unsupported generation model: {config.generation_model}")
            
    def generate(self, prompt: str, max_length: int = None) -> Dict[str, Any]:
        """Generate text from prompt"""
        
        if max_length is None:
            max_length = self.config.generation_max_length
            
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=self.config.gpt_temperature,
                top_p=self.config.gpt_top_p,
                top_k=self.config.gpt_top_k,
                repetition_penalty=self.config.gpt_repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'prompt': prompt,
            'generated_length': len(generated_text) - len(prompt)
        }

class EmbeddingGenerator:
    """Generate embeddings using various models"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        if config.embedding_model == "sentence_transformer":
            self.model = SentenceTransformer(config.sentence_transformer_model)
        elif config.embedding_model == "bert":
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise ValueError(f"Unsupported embedding model: {config.embedding_model}")
            
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for texts"""
        
        if isinstance(texts, str):
            texts = [texts]
            
        if self.config.embedding_model == "sentence_transformer":
            embeddings = self.model.encode(texts)
        elif self.config.embedding_model == "bert":
            embeddings = []
            
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.config.transformer_max_length,
                    padding=self.config.transformer_padding,
                    truncation=self.config.transformer_truncation
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(embedding)
                    
            embeddings = np.array(embeddings)
            
        return embeddings
        
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        
        embeddings = self.generate_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return similarity

class NaturalLanguageProcessingEngine:
    """Main Natural Language Processing Engine"""
    
    def __init__(self, config: NLPConfig):
        self.config = config
        
        # Initialize components
        self.preprocessor = TextPreprocessor(config)
        
        if config.enable_sentiment_analysis:
            self.sentiment_analyzer = SentimentAnalyzer(config)
            
        if config.enable_ner:
            self.ner = NamedEntityRecognizer(config)
            
        if config.enable_text_classification:
            self.classifier = TextClassifier(config)
            
        if config.enable_qa:
            self.qa = QuestionAnswerer(config)
            
        if config.enable_summarization:
            self.summarizer = TextSummarizer(config)
            
        if config.enable_text_generation:
            self.generator = TextGenerator(config)
            
        if config.enable_embeddings:
            self.embedding_generator = EmbeddingGenerator(config)
            
        # Results storage
        self.results = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
    def process_text(self, text: str, tasks: List[str] = None) -> Dict[str, Any]:
        """Process text with specified tasks"""
        
        if tasks is None:
            tasks = ['preprocessing', 'sentiment', 'ner', 'classification', 'embeddings']
            
        start_time = time.time()
        results = {}
        
        # Preprocessing
        if 'preprocessing' in tasks:
            results['preprocessed_text'] = self.preprocessor.preprocess(text)
            results['tokens'] = self.preprocessor.tokenize(text)
            results['sentences'] = self.preprocessor.sent_tokenize(text)
            
        # Sentiment analysis
        if 'sentiment' in tasks and self.config.enable_sentiment_analysis:
            results['sentiment'] = self.sentiment_analyzer.analyze(text)
            
        # Named Entity Recognition
        if 'ner' in tasks and self.config.enable_ner:
            results['entities'] = self.ner.recognize(text)
            
        # Text classification
        if 'classification' in tasks and self.config.enable_text_classification:
            results['classification'] = self.classifier.classify(text)
            
        # Question Answering
        if 'qa' in tasks and self.config.enable_qa:
            # For QA, we need a context - using the text itself as context
            results['qa'] = self.qa.answer("What is this text about?", text)
            
        # Text summarization
        if 'summarization' in tasks and self.config.enable_summarization:
            results['summarization'] = self.summarizer.summarize(text)
            
        # Text generation
        if 'generation' in tasks and self.config.enable_text_generation:
            results['generation'] = self.generator.generate(text)
            
        # Embeddings
        if 'embeddings' in tasks and self.config.enable_embeddings:
            results['embeddings'] = self.embedding_generator.generate_embeddings(text)
            
        # Add processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        # Store results
        self.results['text_processing'].append(results)
        
        return results
        
    def process_batch(self, texts: List[str], tasks: List[str] = None) -> List[Dict[str, Any]]:
        """Process batch of texts"""
        
        results = []
        
        for text in texts:
            result = self.process_text(text, tasks)
            results.append(result)
            
        return results
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        metrics = {
            'total_texts_processed': sum(len(results) for results in self.results.values()),
            'average_processing_time': 0.0,
            'enabled_features': []
        }
        
        # Add enabled features
        if self.config.enable_sentiment_analysis:
            metrics['enabled_features'].append('sentiment_analysis')
        if self.config.enable_ner:
            metrics['enabled_features'].append('ner')
        if self.config.enable_text_classification:
            metrics['enabled_features'].append('text_classification')
        if self.config.enable_qa:
            metrics['enabled_features'].append('question_answering')
        if self.config.enable_summarization:
            metrics['enabled_features'].append('summarization')
        if self.config.enable_text_generation:
            metrics['enabled_features'].append('text_generation')
        if self.config.enable_embeddings:
            metrics['enabled_features'].append('embeddings')
            
        # Calculate average processing time
        all_times = []
        for results in self.results.values():
            for result in results:
                if 'processing_time' in result:
                    all_times.append(result['processing_time'])
                    
        if all_times:
            metrics['average_processing_time'] = np.mean(all_times)
            
        return metrics
        
    def save_results(self, filepath: str):
        """Save results to file"""
        
        results_data = {
            'results': dict(self.results),
            'performance_metrics': self.get_performance_metrics(),
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, default=str)
            
    def load_results(self, filepath: str):
        """Load results from file"""
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
            
        self.results = defaultdict(list, results_data['results'])

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test NLP engine
    print("Testing Natural Language Processing Engine...")
    
    # Create config
    config = NLPConfig(
        model_type="transformer",
        transformer_model="bert-base-uncased",
        enable_sentiment_analysis=True,
        enable_ner=True,
        enable_text_classification=True,
        enable_embeddings=True
    )
    
    # Create engine
    nlp_engine = NaturalLanguageProcessingEngine(config)
    
    # Test text processing
    test_text = "This is a great day! I love using AI for natural language processing."
    
    print("Testing text processing...")
    results = nlp_engine.process_text(test_text)
    print(f"Processing results: {list(results.keys())}")
    
    # Test batch processing
    print("Testing batch processing...")
    batch_texts = [
        "I love this product!",
        "This is terrible.",
        "The weather is nice today."
    ]
    batch_results = nlp_engine.process_batch(batch_texts)
    print(f"Batch results: {len(batch_results)} texts processed")
    
    # Get performance metrics
    metrics = nlp_engine.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    print("\nNatural Language Processing engine initialized successfully!")
























