"""
Advanced NLP System
==================

Sistema NLP avanzado con capacidades adicionales y optimizaciones.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import torch
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core NLP imports
import spacy
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Advanced libraries
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

logger = logging.getLogger(__name__)

class AdvancedNLPConfig:
    """Configuración del sistema NLP avanzado."""
    
    def __init__(self):
        self.max_workers = mp.cpu_count() * 2
        self.batch_size = 32
        self.max_concurrent = 50
        self.memory_limit_gb = 32.0
        self.cache_size_mb = 16384
        self.gpu_memory_fraction = 0.9
        self.mixed_precision = True

@dataclass
class AdvancedNLPResult:
    """Resultado del sistema NLP avanzado."""
    text: str
    language: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[Dict[str, Any]]
    readability: Dict[str, Any]
    advanced_features: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class AdvancedNLPSystem:
    """Sistema NLP avanzado."""
    
    def __init__(self, config: AdvancedNLPConfig = None):
        """Initialize advanced NLP system."""
        self.config = config or AdvancedNLPConfig()
        self.is_initialized = False
        
        # Advanced components
        self.models = {}
        self.pipelines = {}
        self.vectorizers = {}
        self.embeddings = {}
        self.ml_models = {}
        
        # Performance optimization
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        # GPU optimization
        self.gpu_available = torch.cuda.is_available()
        self.gpu_device = "cuda" if self.gpu_available else "cpu"
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0,
            'average_quality_score': 0.0,
            'average_confidence_score': 0.0,
            'error_count': 0
        }
        
        # Background tasks
        self._background_tasks = []
        self._running = False
    
    async def initialize(self):
        """Initialize advanced NLP system."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info("Initializing Advanced NLP System...")
            
            # Load advanced models
            await self._load_advanced_models()
            
            # Initialize advanced features
            await self._initialize_advanced_features()
            
            # Start background optimization
            await self._start_background_optimization()
            
            # Warm up models
            await self._warm_up_models()
            
            self.is_initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"Advanced NLP System initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced NLP System: {e}")
            raise
    
    async def _load_advanced_models(self):
        """Load advanced models."""
        try:
            # Load spaCy models
            await self._load_spacy_advanced()
            
            # Load transformer models
            await self._load_transformers_advanced()
            
            # Load sentence transformers
            await self._load_sentence_transformers_advanced()
            
            # Initialize advanced vectorizers
            self._initialize_advanced_vectorizers()
            
            # Load advanced analysis models
            await self._load_advanced_analysis_models()
            
        except Exception as e:
            logger.error(f"Advanced model loading failed: {e}")
            raise
    
    async def _load_spacy_advanced(self):
        """Load spaCy models with advanced features."""
        try:
            spacy.prefer_gpu() if self.gpu_available else None
            
            models_to_load = {
                'en': 'en_core_web_lg',
                'es': 'es_core_news_lg',
                'fr': 'fr_core_news_lg',
                'de': 'de_core_news_lg'
            }
            
            for lang, model_name in models_to_load.items():
                try:
                    self.models[f'spacy_{lang}'] = spacy.load(
                        model_name,
                        disable=[]
                    )
                    logger.info(f"Loaded advanced spaCy model: {model_name}")
                except OSError:
                    logger.warning(f"spaCy model {model_name} not available")
                    
        except Exception as e:
            logger.error(f"spaCy advanced loading failed: {e}")
    
    async def _load_transformers_advanced(self):
        """Load transformer models with advanced features."""
        try:
            device = self.gpu_device if self.gpu_available else "cpu"
            
            model_configs = {
                'sentiment': {
                    'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'task': 'sentiment-analysis',
                    'return_all_scores': True
                },
                'ner': {
                    'model': 'xlm-roberta-large-finetuned-conll03-english',
                    'task': 'ner',
                    'aggregation_strategy': 'max'
                },
                'classification': {
                    'model': 'microsoft/DialoGPT-large',
                    'task': 'text-classification'
                }
            }
            
            for task, config in model_configs.items():
                try:
                    pipeline_config = {
                        'device': 0 if device == 'cuda' else -1,
                        'batch_size': self.config.batch_size,
                        'max_length': 1024,
                        'truncation': True,
                        'padding': True
                    }
                    
                    self.pipelines[task] = pipeline(
                        config['task'],
                        model=config['model'],
                        **pipeline_config
                    )
                    
                    logger.info(f"Loaded advanced {task} model: {config['model']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {task} model: {e}")
                    
        except Exception as e:
            logger.error(f"Transformer advanced loading failed: {e}")
    
    async def _load_sentence_transformers_advanced(self):
        """Load sentence transformers with advanced features."""
        try:
            model_name = 'all-mpnet-base-v2'
            
            self.embeddings['sentence_transformer'] = SentenceTransformer(
                model_name,
                device=self.gpu_device,
                cache_folder='./advanced_nlp_cache'
            )
            
            logger.info(f"Loaded advanced sentence transformer: {model_name}")
            
        except Exception as e:
            logger.error(f"Sentence transformer advanced loading failed: {e}")
    
    def _initialize_advanced_vectorizers(self):
        """Initialize advanced vectorizers."""
        try:
            # TF-IDF with advanced features
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=50000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                lowercase=True,
                strip_accents='unicode',
                dtype=np.float64
            )
            
            # LDA for topic modeling
            self.vectorizers['lda'] = LatentDirichletAllocation(
                n_components=20,
                random_state=42,
                max_iter=200
            )
            
            logger.info("Initialized advanced vectorizers")
            
        except Exception as e:
            logger.error(f"Advanced vectorizer initialization failed: {e}")
    
    async def _load_advanced_analysis_models(self):
        """Load advanced analysis models."""
        try:
            # VADER sentiment analyzer
            self.models['vader'] = SentimentIntensityAnalyzer()
            
            # TextBlob for additional analysis
            self.models['textblob'] = TextBlob
            
            # NLTK components
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK download failed: {e}")
            
            self.models['lemmatizer'] = WordNetLemmatizer()
            self.models['stopwords'] = set(stopwords.words('english'))
            
            logger.info("Loaded advanced analysis models")
            
        except Exception as e:
            logger.error(f"Advanced analysis model loading failed: {e}")
    
    async def _initialize_advanced_features(self):
        """Initialize advanced features."""
        try:
            # Initialize ML models
            self.ml_models['classification'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_models['regression'] = LogisticRegression(random_state=42, max_iter=1000)
            
            logger.info("Initialized advanced features")
            
        except Exception as e:
            logger.error(f"Advanced feature initialization failed: {e}")
    
    async def _start_background_optimization(self):
        """Start background optimization tasks."""
        self._running = True
        
        # Background optimization task
        opt_task = asyncio.create_task(self._optimization_loop())
        self._background_tasks.append(opt_task)
        
        logger.info("Background optimization tasks started")
    
    async def _optimization_loop(self):
        """Background optimization."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Optimize system
                await self._optimize_system()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization error: {e}")
    
    async def _optimize_system(self):
        """Optimize system performance."""
        try:
            # This would implement system optimization
            logger.info("Optimizing system performance")
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
    
    async def _warm_up_models(self):
        """Warm up models with advanced features."""
        try:
            warm_up_text = "This is a comprehensive warm-up text for advanced performance validation."
            
            # Warm up spaCy models
            for lang in ['en', 'es', 'fr', 'de']:
                if f'spacy_{lang}' in self.models:
                    doc = self.models[f'spacy_{lang}'](warm_up_text)
                    _ = [ent for ent in doc.ents]
                    _ = [token for token in doc]
                    _ = [chunk for chunk in doc.noun_chunks]
            
            # Warm up transformer pipelines
            for task, pipeline_obj in self.pipelines.items():
                try:
                    _ = pipeline_obj(warm_up_text)
                except Exception as e:
                    logger.warning(f"Warm-up failed for {task}: {e}")
            
            # Warm up sentence transformer
            if 'sentence_transformer' in self.embeddings:
                _ = self.embeddings['sentence_transformer'].encode([warm_up_text])
            
            logger.info("Models warmed up with advanced features")
            
        except Exception as e:
            logger.error(f"Model warm-up with advanced features failed: {e}")
    
    async def analyze_advanced(
        self,
        text: str,
        language: str = "en",
        use_cache: bool = True,
        advanced_features: bool = True
    ) -> AdvancedNLPResult:
        """Perform advanced text analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_hit = False
            if use_cache:
                cache_key = self._generate_cache_key(text, language)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    cache_hit = True
                    self.stats['cache_hits'] += 1
                    return cached_result
                else:
                    self.stats['cache_misses'] += 1
            
            # Perform advanced analysis
            result = await self._comprehensive_advanced_analysis(
                text, language, advanced_features
            )
            
            # Create result
            processing_time = time.time() - start_time
            result = AdvancedNLPResult(
                text=text,
                language=language,
                sentiment=result.get('sentiment', {}),
                entities=result.get('entities', []),
                keywords=result.get('keywords', []),
                topics=result.get('topics', []),
                readability=result.get('readability', {}),
                advanced_features=result.get('advanced_features', {}),
                quality_score=result.get('quality_score', 0.0),
                confidence_score=result.get('confidence_score', 0.0),
                processing_time=processing_time,
                cache_hit=cache_hit,
                timestamp=datetime.now()
            )
            
            # Cache result
            if use_cache and not cache_hit:
                self.cache[cache_key] = result
            
            # Update statistics
            self._update_stats(processing_time, result.quality_score, result.confidence_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced analysis failed: {e}")
            raise
    
    async def _comprehensive_advanced_analysis(
        self,
        text: str,
        language: str,
        advanced_features: bool
    ) -> Dict[str, Any]:
        """Perform comprehensive advanced analysis."""
        try:
            # Perform basic analyses
            sentiment = await self._analyze_sentiment_advanced(text, language)
            entities = await self._extract_entities_advanced(text, language)
            keywords = await self._extract_keywords_advanced(text, language)
            topics = await self._extract_topics_advanced(text, language)
            readability = await self._analyze_readability_advanced(text, language)
            
            # Advanced features
            advanced_feat = {}
            if advanced_features:
                advanced_feat = await self._perform_advanced_features(text, language)
            
            # Quality assessment
            quality_score = await self._assess_advanced_quality(
                sentiment, entities, keywords, topics, readability, advanced_feat
            )
            
            # Confidence assessment
            confidence_score = await self._calculate_advanced_confidence(
                quality_score, advanced_feat
            )
            
            return {
                'sentiment': sentiment,
                'entities': entities,
                'keywords': keywords,
                'topics': topics,
                'readability': readability,
                'advanced_features': advanced_feat,
                'quality_score': quality_score,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"Comprehensive advanced analysis failed: {e}")
            return {}
    
    async def _analyze_sentiment_advanced(self, text: str, language: str) -> Dict[str, Any]:
        """Advanced sentiment analysis."""
        try:
            results = {}
            
            # Use transformer model
            if 'sentiment' in self.pipelines:
                try:
                    sentiment_result = self.pipelines['sentiment'](text)
                    results['transformer'] = sentiment_result
                except Exception as e:
                    logger.warning(f"Transformer sentiment failed: {e}")
            
            # Use VADER for additional analysis
            try:
                vader_scores = self.models['vader'].polarity_scores(text)
                results['vader'] = vader_scores
            except Exception as e:
                logger.warning(f"VADER sentiment failed: {e}")
            
            # Use TextBlob
            try:
                blob = self.models['textblob'](text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                logger.warning(f"TextBlob sentiment failed: {e}")
            
            # Ensemble result
            ensemble_result = self._ensemble_sentiment_advanced(results)
            results['ensemble'] = ensemble_result
            
            return results
            
        except Exception as e:
            logger.error(f"Advanced sentiment analysis failed: {e}")
            return {}
    
    async def _extract_entities_advanced(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Advanced entity extraction."""
        try:
            entities = []
            
            # Use spaCy with advanced features
            if f'spacy_{language}' in self.models:
                try:
                    doc = self.models[f'spacy_{language}'](text)
                    for ent in doc.ents:
                        entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0,
                            'method': 'spacy',
                            'description': spacy.explain(ent.label_)
                        })
                except Exception as e:
                    logger.warning(f"spaCy NER failed: {e}")
            
            # Use transformer NER
            if 'ner' in self.pipelines:
                try:
                    ner_results = self.pipelines['ner'](text)
                    for entity in ner_results:
                        entities.append({
                            'text': entity['word'],
                            'label': entity['entity_group'],
                            'start': entity.get('start', 0),
                            'end': entity.get('end', len(entity['word'])),
                            'confidence': entity.get('score', 0.0),
                            'method': 'transformer'
                        })
                except Exception as e:
                    logger.warning(f"Transformer NER failed: {e}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Advanced entity extraction failed: {e}")
            return []
    
    async def _extract_keywords_advanced(self, text: str) -> List[str]:
        """Advanced keyword extraction."""
        try:
            keywords = []
            
            # Use TF-IDF with advanced features
            try:
                vectorizer = self.vectorizers['tfidf']
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                keywords = [kw[0] for kw in keyword_scores[:30]]
                
            except Exception as e:
                logger.warning(f"TF-IDF keyword extraction failed: {e}")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Advanced keyword extraction failed: {e}")
            return []
    
    async def _extract_topics_advanced(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Advanced topic extraction."""
        try:
            topics = []
            
            # Use LDA for advanced topic modeling
            try:
                vectorizer = self.vectorizers['tfidf']
                lda = self.vectorizers['lda']
                
                # Fit LDA
                tfidf_matrix = vectorizer.fit_transform([text])
                lda.fit(tfidf_matrix)
                
                # Get topics
                feature_names = vectorizer.get_feature_names_out()
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-15:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        'id': topic_idx,
                        'words': top_words,
                        'weights': topic[top_words_idx].tolist(),
                        'coherence_score': 0.0
                    })
                
            except Exception as e:
                logger.warning(f"LDA topic extraction failed: {e}")
            
            return topics
            
        except Exception as e:
            logger.error(f"Advanced topic extraction failed: {e}")
            return []
    
    async def _analyze_readability_advanced(self, text: str, language: str) -> Dict[str, Any]:
        """Advanced readability analysis."""
        try:
            scores = {}
            
            # Flesch Reading Ease
            try:
                scores['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            except:
                scores['flesch_reading_ease'] = 0.0
            
            # Flesch-Kincaid Grade Level
            try:
                scores['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            except:
                scores['flesch_kincaid_grade'] = 0.0
            
            # Gunning Fog Index
            try:
                scores['gunning_fog'] = textstat.gunning_fog(text)
            except:
                scores['gunning_fog'] = 0.0
            
            # SMOG Index
            try:
                scores['smog'] = textstat.smog_index(text)
            except:
                scores['smog'] = 0.0
            
            # Automated Readability Index
            try:
                scores['ari'] = textstat.automated_readability_index(text)
            except:
                scores['ari'] = 0.0
            
            # Overall readability level
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            if avg_score >= 80:
                level = "Very Easy"
            elif avg_score >= 60:
                level = "Easy"
            elif avg_score >= 40:
                level = "Moderate"
            elif avg_score >= 20:
                level = "Difficult"
            else:
                level = "Very Difficult"
            
            scores['overall_level'] = level
            scores['average_score'] = avg_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Advanced readability analysis failed: {e}")
            return {'average_score': 0.0, 'overall_level': 'Unknown'}
    
    async def _perform_advanced_features(self, text: str, language: str) -> Dict[str, Any]:
        """Perform advanced features."""
        try:
            features = {}
            
            # Text complexity analysis
            features['complexity'] = await self._analyze_text_complexity(text)
            
            # Language detection
            features['language_detection'] = await self._detect_language(text)
            
            # Text classification
            features['classification'] = await self._classify_text(text)
            
            # Text similarity
            features['similarity'] = await self._calculate_similarity(text)
            
            return features
            
        except Exception as e:
            logger.error(f"Advanced features failed: {e}")
            return {}
    
    async def _analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity."""
        try:
            complexity = {
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'character_count': len(text),
                'average_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
                'average_sentence_length': len(text.split()) / len(text.split('.')) if text.split('.') else 0
            }
            
            return complexity
            
        except Exception as e:
            logger.error(f"Text complexity analysis failed: {e}")
            return {}
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language."""
        try:
            # Simple language detection based on common words
            languages = {
                'en': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
                'es': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le'],
                'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'],
                'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for lang, words in languages.items():
                score = sum(1 for word in words if word in text_lower)
                scores[lang] = score
            
            detected_language = max(scores, key=scores.get) if scores else 'en'
            confidence = max(scores.values()) / sum(scores.values()) if sum(scores.values()) > 0 else 0
            
            return {
                'detected_language': detected_language,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {'detected_language': 'en', 'confidence': 0.0, 'scores': {}}
    
    async def _classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text."""
        try:
            # Simple text classification based on keywords
            categories = {
                'business': ['company', 'profit', 'revenue', 'market', 'investment', 'financial'],
                'technology': ['software', 'computer', 'digital', 'internet', 'data', 'system'],
                'science': ['research', 'study', 'experiment', 'theory', 'hypothesis', 'analysis'],
                'news': ['report', 'news', 'update', 'announcement', 'statement', 'press']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[category] = score
            
            predicted_category = max(scores, key=scores.get) if scores else 'general'
            confidence = max(scores.values()) / sum(scores.values()) if sum(scores.values()) > 0 else 0
            
            return {
                'predicted_category': predicted_category,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            return {'predicted_category': 'general', 'confidence': 0.0, 'scores': {}}
    
    async def _calculate_similarity(self, text: str) -> Dict[str, Any]:
        """Calculate text similarity."""
        try:
            # Simple similarity calculation
            similarity = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'unique_words': len(set(text.lower().split())),
                'vocabulary_diversity': len(set(text.lower().split())) / len(text.split()) if text.split() else 0
            }
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {}
    
    async def _assess_advanced_quality(
        self,
        sentiment: Dict[str, Any],
        entities: List[Dict[str, Any]],
        keywords: List[str],
        topics: List[Dict[str, Any]],
        readability: Dict[str, Any],
        advanced_features: Dict[str, Any]
    ) -> float:
        """Assess advanced quality of analysis results."""
        try:
            quality_score = 0.0
            total_weight = 0.0
            
            # Basic analysis quality (60%)
            basic_weight = 0.6
            basic_quality = 0.0
            
            # Sentiment quality
            if sentiment and 'ensemble' in sentiment:
                sentiment_quality = sentiment['ensemble'].get('confidence', 0)
                basic_quality += sentiment_quality * 0.25
            
            # Entity quality
            if entities:
                entity_quality = min(1.0, len(entities) / 20)
                basic_quality += entity_quality * 0.25
            
            # Keyword quality
            if keywords:
                keyword_quality = min(1.0, len(keywords) / 30)
                basic_quality += keyword_quality * 0.25
            
            # Readability quality
            if readability and 'average_score' in readability:
                readability_quality = readability['average_score'] / 100
                basic_quality += readability_quality * 0.25
            
            quality_score += basic_quality * basic_weight
            total_weight += basic_weight
            
            # Advanced features quality (40%)
            advanced_weight = 0.4
            advanced_quality = 0.0
            
            # Advanced features quality
            if advanced_features:
                advanced_quality += min(1.0, len(advanced_features) / 5) * 0.5
                advanced_quality += min(1.0, advanced_features.get('complexity', {}).get('word_count', 0) / 100) * 0.5
            
            quality_score += advanced_quality * advanced_weight
            total_weight += advanced_weight
            
            return quality_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Advanced quality assessment failed: {e}")
            return 0.0
    
    async def _calculate_advanced_confidence(
        self,
        quality_score: float,
        advanced_features: Dict[str, Any]
    ) -> float:
        """Calculate advanced confidence score."""
        try:
            confidence_score = quality_score  # Start with quality score
            
            # Boost confidence based on advanced features
            if advanced_features:
                feature_count = len(advanced_features)
                if feature_count > 0:
                    feature_confidence = min(1.0, feature_count / 5)
                    confidence_score = (confidence_score + feature_confidence) / 2
            
            return min(1.0, max(0.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Advanced confidence calculation failed: {e}")
            return quality_score
    
    def _ensemble_sentiment_advanced(self, sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple sentiment analysis results with advanced features."""
        try:
            scores = []
            weights = []
            
            # Collect scores from different methods with weights
            for method, result in sentiment_results.items():
                if method == 'vader' and 'compound' in result:
                    scores.append(result['compound'])
                    weights.append(0.3)  # VADER weight
                elif method == 'textblob' and 'polarity' in result:
                    scores.append(result['polarity'])
                    weights.append(0.2)  # TextBlob weight
                elif method == 'transformer' and isinstance(result, list) and result:
                    # Extract score from transformer result
                    transformer_score = result[0].get('score', 0) if result[0].get('label') == 'POSITIVE' else -result[0].get('score', 0)
                    scores.append(transformer_score)
                    weights.append(0.5)  # Transformer weight (highest)
            
            if scores and weights:
                # Weighted average
                weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
                total_weight = sum(weights)
                avg_score = weighted_sum / total_weight
                
                # Calculate confidence based on agreement
                if len(scores) > 1:
                    variance = np.var(scores)
                    confidence = max(0, 1 - variance)
                else:
                    confidence = 0.5
                
                if avg_score > 0.1:
                    sentiment = 'positive'
                elif avg_score < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'score': avg_score,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'method_count': len(scores),
                    'variance': np.var(scores) if len(scores) > 1 else 0
                }
            else:
                return {'score': 0, 'sentiment': 'neutral', 'confidence': 0, 'method_count': 0, 'variance': 0}
                
        except Exception as e:
            logger.error(f"Ensemble sentiment calculation failed: {e}")
            return {'score': 0, 'sentiment': 'neutral', 'confidence': 0, 'method_count': 0, 'variance': 0}
    
    def _generate_cache_key(self, text: str, language: str) -> str:
        """Generate cache key for advanced analysis."""
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"advanced:{language}:{content_hash}"
    
    def _update_stats(self, processing_time: float, quality_score: float, confidence_score: float):
        """Update advanced statistics."""
        self.stats['requests_processed'] += 1
        
        # Update average processing time
        if self.stats['average_processing_time'] == 0:
            self.stats['average_processing_time'] = processing_time
        else:
            self.stats['average_processing_time'] = (
                self.stats['average_processing_time'] * 0.9 + processing_time * 0.1
            )
        
        # Update quality scores
        if quality_score > 0:
            self.stats['average_quality_score'] = (
                self.stats['average_quality_score'] * 0.9 + quality_score * 0.1
            )
        
        # Update confidence scores
        if confidence_score > 0:
            self.stats['average_confidence_score'] = (
                self.stats['average_confidence_score'] * 0.9 + confidence_score * 0.1
            )
    
    async def batch_analyze_advanced(
        self,
        texts: List[str],
        language: str = "en",
        use_cache: bool = True,
        advanced_features: bool = True
    ) -> List[AdvancedNLPResult]:
        """Perform advanced batch analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Process in parallel batches
            batch_size = min(self.config.batch_size, len(texts))
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    self.analyze_advanced(
                        text=text,
                        language=language,
                        use_cache=use_cache,
                        advanced_features=advanced_features
                    )
                    for text in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch analysis error for text {i + j}: {result}")
                        results.append(AdvancedNLPResult(
                            text=batch[j],
                            language=language,
                            sentiment={},
                            entities=[],
                            keywords=[],
                            topics=[],
                            readability={},
                            advanced_features={},
                            quality_score=0.0,
                            confidence_score=0.0,
                            processing_time=0,
                            cache_hit=False,
                            timestamp=datetime.now()
                        ))
                    else:
                        results.append(result)
            
            return results
                
        except Exception as e:
            logger.error(f"Advanced batch analysis failed: {e}")
            raise
    
    async def get_advanced_status(self) -> Dict[str, Any]:
        """Get advanced system status."""
        try:
            # System status
            system_status = {
                'initialized': self.is_initialized,
                'advanced_mode': True,
                'gpu_available': self.gpu_available,
                'gpu_device': self.gpu_device,
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size
            }
            
            # Performance statistics
            performance_stats = {
                'requests_processed': self.stats['requests_processed'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                ),
                'average_processing_time': self.stats['average_processing_time'],
                'error_count': self.stats['error_count'],
                'success_rate': (
                    (self.stats['requests_processed'] - self.stats['error_count']) / self.stats['requests_processed']
                    if self.stats['requests_processed'] > 0 else 0
                )
            }
            
            # Advanced statistics
            advanced_stats = {
                'average_quality_score': self.stats['average_quality_score'],
                'average_confidence_score': self.stats['average_confidence_score'],
                'advanced_features_enabled': True
            }
            
            # Cache status
            cache_status = {
                'size': len(self.cache),
                'hit_rate': (
                    self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                )
            }
            
            # Memory status
            memory_status = {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            }
            
            return {
                'system': system_status,
                'performance': performance_stats,
                'advanced': advanced_stats,
                'cache': cache_status,
                'memory': memory_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get advanced status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def shutdown(self):
        """Shutdown advanced NLP system."""
        try:
            logger.info("Shutting down Advanced NLP System...")
            
            # Stop background tasks
            self._running = False
            for task in self._background_tasks:
                task.cancel()
            
            # Shutdown executors
            self.executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            logger.info("Advanced NLP System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Global advanced NLP system instance
advanced_nlp_system = AdvancedNLPSystem()