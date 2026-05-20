from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
from sentence_transformers import SentenceTransformer
import spacy
from spacy.language import Language
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import textblob
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
import json
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from typing import Any, List, Dict, Optional
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering,
    pipeline, TextIteratorStreamer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPModelType(Enum):
    BERT = "bert"
    GPT = "gpt"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    SPACY = "spacy"
    NLTK = "nltk"
    TEXTBLOB = "textblob"
    GENSIM = "gensim"

@dataclass
class TextAnalysisResult:
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    summary: str
    embeddings: np.ndarray
    language: str
    confidence: float

@dataclass
class ProcessingConfig:
    max_length: int = 512
    batch_size: int = 8
    use_gpu: bool = True
    cache_embeddings: bool = True
    parallel_processing: bool = True

class OptimizedNLPService:
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 config: ProcessingConfig = None):
        
        self.device = device
        self.config = config or ProcessingConfig()
        
        # Initialize models
        self._initialize_models()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        # Performance monitoring
        self.performance_stats = {
            'total_texts_processed': 0,
            'total_processing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"OptimizedNLPService initialized on {device}")

    def _initialize_models(self) -> Any:
        """Initialize optimized NLP models"""
        try:
            # BERT models for various tasks
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
            
            if self.config.use_gpu and torch.cuda.is_available():
                self.bert_model = self.bert_model.to(self.device)
                self.bert_model.eval()
            
            # Sentiment analysis
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Named Entity Recognition
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Question Answering
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Sentence embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            if self.config.use_gpu and torch.cuda.is_available():
                self.sentence_transformer = self.sentence_transformer.to(self.device)
            
            # SpaCy for advanced NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found, downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # NLTK setup
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')
                nltk.download('averaged_perceptron_tagger')
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            
            # Gensim models
            self.word2vec_model = None
            self.doc2vec_model = None
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            raise

    async def analyze_text(self, text: str) -> TextAnalysisResult:
        """Comprehensive text analysis with parallel processing"""
        start_time = time.time()
        
        try:
            # Process different aspects in parallel
            tasks = [
                self._analyze_sentiment(text),
                self._extract_entities(text),
                self._extract_keywords(text),
                self._generate_summary(text),
                self._get_embeddings(text),
                self._detect_language(text)
            ]
            
            sentiment, entities, keywords, summary, embeddings, language = await asyncio.gather(*tasks)
            
            # Calculate confidence based on model outputs
            confidence = self._calculate_confidence(sentiment, entities, keywords)
            
            result = TextAnalysisResult(
                sentiment=sentiment,
                entities=entities,
                keywords=keywords,
                summary=summary,
                embeddings=embeddings,
                language=language,
                confidence=confidence
            )
            
            # Update performance stats
            self._update_performance_stats(time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            raise

    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with multiple models"""
        try:
            # Process in thread pool
            sentiment_result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.sentiment_pipeline(text)
            )
            
            # Additional sentiment analysis with TextBlob
            blob = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: TextBlob(text)
            )
            
            return {
                'roberta': sentiment_result[0],
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'error': str(e)}

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities"""
        try:
            # BERT NER
            ner_result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.ner_pipeline(text)
            )
            
            # SpaCy NER
            doc = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.nlp(text)
            )
            
            spacy_entities = [
                {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                for ent in doc.ents
            ]
            
            return {
                'bert_ner': ner_result,
                'spacy_ner': spacy_entities
            }
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []

    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using multiple methods"""
        try:
            # NLTK keyword extraction
            tokens = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: word_tokenize(text.lower())
            )
            
            # Remove stop words and lemmatize
            keywords = []
            for token in tokens:
                if token.isalnum() and token not in self.stop_words:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    keywords.append(lemmatized)
            
            # SpaCy keyword extraction
            doc = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.nlp(text)
            )
            
            spacy_keywords = [
                token.text for token in doc
                if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'ADJ', 'VERB']
            ]
            
            # Combine and deduplicate
            all_keywords = list(set(keywords + spacy_keywords))
            
            # Return top keywords by frequency
            keyword_freq = {}
            for keyword in all_keywords:
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
            
            return sorted(keyword_freq.keys(), key=lambda x: keyword_freq[x], reverse=True)[:10]
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return []

    async def _generate_summary(self, text: str) -> str:
        """Generate text summary"""
        try:
            # Simple extractive summarization
            sentences = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: sent_tokenize(text)
            )
            
            if len(sentences) <= 3:
                return text
            
            # Score sentences by word frequency
            word_freq = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                for word in words:
                    if word.isalnum() and word not in self.stop_words:
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                score = sum(word_freq.get(word, 0) for word in words if word.isalnum())
                sentence_scores[sentence] = score
            
            # Select top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            summary = ' '.join([sentence for sentence, score in top_sentences])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return text[:200] + "..." if len(text) > 200 else text

    async def _get_embeddings(self, text: str) -> np.ndarray:
        """Get text embeddings with caching"""
        try:
            # Check cache first
            if self.config.cache_embeddings and text in self.embedding_cache:
                self.performance_stats['cache_hits'] += 1
                return self.embedding_cache[text]
            
            self.performance_stats['cache_misses'] += 1
            
            # Generate embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.sentence_transformer.encode(text)
            )
            
            # Cache the result
            if self.config.cache_embeddings:
                self.embedding_cache[text] = embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.zeros(384)  # Default embedding size

    async def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            # Simple language detection using TextBlob
            blob = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: TextBlob(text)
            )
            
            # TextBlob doesn't have language detection, so we'll use a simple heuristic
            # This is a basic implementation - in production, use a proper language detection library
            
            # Check for common patterns
            if re.search(r'[áéíóúñ]', text, re.IGNORECASE):
                return 'es'
            elif re.search(r'[àâäéèêëïîôöùûüÿç]', text, re.IGNORECASE):
                return 'fr'
            elif re.search(r'[äöüß]', text, re.IGNORECASE):
                return 'de'
            else:
                return 'en'
                
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return 'en'

    def _calculate_confidence(self, sentiment: Dict, entities: List, keywords: List) -> float:
        """Calculate confidence score for analysis"""
        try:
            confidence = 0.0
            
            # Sentiment confidence
            if 'roberta' in sentiment:
                confidence += 0.3
            
            # Entity confidence
            if entities and len(entities) > 0:
                confidence += 0.3
            
            # Keyword confidence
            if keywords and len(keywords) > 0:
                confidence += 0.2
            
            # Text length confidence
            confidence += min(0.2, len(str(sentiment)) / 1000)
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    async def batch_analyze(self, texts: List[str]) -> List[TextAnalysisResult]:
        """Analyze multiple texts in batch"""
        try:
            tasks = [self.analyze_text(text) for text in texts]
            results = await asyncio.gather(*tasks)
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            raise

    async def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer questions using QA model"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.qa_pipeline(question=question, context=context)
            )
            
            return {
                'answer': result['answer'],
                'confidence': result['score'],
                'start': result['start'],
                'end': result['end']
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            return {'error': str(e)}

    def _update_performance_stats(self, processing_time: float):
        """Update performance monitoring statistics"""
        self.performance_stats['total_processing_time'] += processing_time
        self.performance_stats['total_texts_processed'] += 1

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.performance_stats.copy()

    def clear_cache(self) -> Any:
        """Clear embedding cache"""
        self.embedding_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    async def close(self) -> Any:
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.clear_cache()
        logger.info("OptimizedNLPService closed")

# Usage example
async def main():
    
    """main function."""
nlp_service = OptimizedNLPService(
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=ProcessingConfig(
            max_length=512,
            batch_size=8,
            use_gpu=True,
            cache_embeddings=True,
            parallel_processing=True
        )
    )
    
    try:
        # Analyze text
        text = "I love this amazing product! It's absolutely fantastic and works perfectly."
        result = await nlp_service.analyze_text(text)
        
        print(f"Sentiment: {result.sentiment}")
        print(f"Keywords: {result.keywords}")
        print(f"Summary: {result.summary}")
        print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence}")
        
        # Batch analysis
        texts = [
            "This is great!",
            "I hate this product.",
            "The weather is nice today."
        ]
        
        batch_results = await nlp_service.batch_analyze(texts)
        print(f"Batch processed {len(batch_results)} texts")
        
        # Question answering
        context = "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."
        question = "Where is the Eiffel Tower located?"
        
        qa_result = await nlp_service.answer_question(question, context)
        print(f"Q&A: {qa_result}")
        
        # Performance stats
        stats = nlp_service.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    finally:
        await nlp_service.close()

match __name__:
    case "__main__":
    asyncio.run(main()) 