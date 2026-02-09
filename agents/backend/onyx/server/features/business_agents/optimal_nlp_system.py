"""
Optimal NLP System
==================

Sistema NLP óptimo con las mejores prácticas, máximo rendimiento
y optimizaciones avanzadas para producción.
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from functools import lru_cache
import gc
import psutil
import torch
from contextlib import asynccontextmanager

# Core NLP imports
import spacy
import nltk
from textblob import TextBlob
from transformers import (
    AutoTokenizer, AutoModel, pipeline,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Advanced imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Custom imports
from .nlp_cache import IntelligentNLPCache
from .nlp_metrics import NLPMonitoringSystem
from .nlp_trends import NLPTrendAnalyzer

logger = logging.getLogger(__name__)

class OptimizationLevel(str, Enum):
    """Niveles de optimización."""
    MINIMAL = "minimal"      # Mínimo uso de recursos
    BALANCED = "balanced"    # Equilibrio rendimiento/recursos
    MAXIMUM = "maximum"      # Máximo rendimiento
    ULTRA = "ultra"          # Rendimiento extremo

class ProcessingMode(str, Enum):
    """Modos de procesamiento."""
    CPU_ONLY = "cpu_only"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

@dataclass
class OptimalConfig:
    """Configuración óptima del sistema."""
    # Performance settings
    optimization_level: OptimizationLevel = OptimizationLevel.MAXIMUM
    processing_mode: ProcessingMode = ProcessingMode.GPU_ACCELERATED
    max_workers: int = mp.cpu_count()
    batch_size: int = 64
    max_concurrent: int = 100
    
    # Memory optimization
    memory_limit_gb: float = 8.0
    cache_size_mb: int = 2048
    model_cache_size: int = 10
    
    # GPU optimization
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Caching
    enable_smart_cache: bool = True
    cache_compression: bool = True
    cache_ttl: int = 3600
    
    # Monitoring
    enable_metrics: bool = True
    enable_profiling: bool = True
    metrics_interval: int = 30
    
    # Quality
    quality_threshold: float = 0.8
    confidence_threshold: float = 0.9
    enable_ensemble: bool = True

class OptimalNLPSystem:
    """Sistema NLP óptimo con máximo rendimiento."""
    
    def __init__(self, config: OptimalConfig = None):
        """Initialize optimal NLP system."""
        self.config = config or OptimalConfig()
        self.is_initialized = False
        
        # Core components
        self.models = {}
        self.pipelines = {}
        self.vectorizers = {}
        self.embeddings = {}
        
        # Optimization components
        self.cache = IntelligentNLPCache(
            max_size=10000,
            max_memory_mb=self.config.cache_size_mb,
            default_ttl=self.config.cache_ttl,
            enable_compression=self.config.cache_compression
        )
        
        self.monitoring = NLPMonitoringSystem()
        self.trend_analyzer = NLPTrendAnalyzer()
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        # Memory management
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        self.model_cache = ModelCache(self.config.model_cache_size)
        
        # GPU optimization
        self.gpu_available = torch.cuda.is_available()
        self.gpu_device = "cuda" if self.gpu_available else "cpu"
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.quality_assessor = QualityAssessor()
        
        # Background tasks
        self._background_tasks = []
        self._running = False
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_processing_time': 0.0,
            'quality_scores': deque(maxlen=1000),
            'error_count': 0
        }
    
    async def initialize(self):
        """Initialize optimal NLP system with maximum performance."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info("Initializing Optimal NLP System...")
            
            # Initialize memory monitoring
            await self.memory_monitor.start()
            
            # Load optimized models
            await self._load_optimized_models()
            
            # Initialize cache
            await self.cache.start()
            
            # Initialize monitoring
            await self.monitoring.start_monitoring()
            
            # Start background optimization
            await self._start_background_optimization()
            
            # Warm up models
            await self._warm_up_models()
            
            self.is_initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"Optimal NLP System initialized in {init_time:.2f}s")
            
            # Record initialization metrics
            await self.monitoring.record_request(
                task="system_initialization",
                processing_time=init_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Optimal NLP System: {e}")
            raise
    
    async def _load_optimized_models(self):
        """Load models with optimal configuration."""
        try:
            # Load spaCy models with optimization
            await self._load_spacy_optimized()
            
            # Load transformer models with optimization
            await self._load_transformers_optimized()
            
            # Load sentence transformers
            await self._load_sentence_transformers_optimized()
            
            # Initialize vectorizers
            self._initialize_vectorizers_optimized()
            
            # Load quality assessment models
            await self._load_quality_models()
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    async def _load_spacy_optimized(self):
        """Load spaCy models with optimization."""
        try:
            # Load with optimization
            spacy.prefer_gpu() if self.gpu_available else None
            
            # Load core models
            models_to_load = {
                'en': 'en_core_web_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm'
            }
            
            for lang, model_name in models_to_load.items():
                try:
                    self.models[f'spacy_{lang}'] = spacy.load(
                        model_name,
                        disable=['parser'] if self.config.optimization_level == OptimizationLevel.MINIMAL else []
                    )
                    logger.info(f"Loaded optimized spaCy model: {model_name}")
                except OSError:
                    logger.warning(f"spaCy model {model_name} not available")
                    
        except Exception as e:
            logger.error(f"spaCy optimization failed: {e}")
    
    async def _load_transformers_optimized(self):
        """Load transformer models with optimization."""
        try:
            # Configure for optimal performance
            device = self.gpu_device if self.gpu_available else "cpu"
            
            # Model configurations for different optimization levels
            model_configs = {
                OptimizationLevel.MINIMAL: {
                    'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
                    'ner': 'dbmdz/bert-base-cased-finetuned-conll03-english',
                    'classification': 'distilbert-base-uncased'
                },
                OptimizationLevel.BALANCED: {
                    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'ner': 'dbmdz/bert-large-cased-finetuned-conll03-english',
                    'classification': 'microsoft/DialoGPT-medium'
                },
                OptimizationLevel.MAXIMUM: {
                    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'ner': 'xlm-roberta-large-finetuned-conll03-english',
                    'classification': 'microsoft/DialoGPT-medium'
                },
                OptimizationLevel.ULTRA: {
                    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                    'ner': 'xlm-roberta-large-finetuned-conll03-english',
                    'classification': 'microsoft/DialoGPT-medium'
                }
            }
            
            config = model_configs[self.config.optimization_level]
            
            # Load models with optimization
            for task, model_name in config.items():
                try:
                    pipeline_config = {
                        'device': 0 if device == 'cuda' else -1,
                        'batch_size': self.config.batch_size,
                        'max_length': 512 if self.config.optimization_level == OptimizationLevel.MINIMAL else 1024
                    }
                    
                    if task == 'sentiment':
                        self.pipelines[task] = pipeline(
                            "sentiment-analysis",
                            model=model_name,
                            return_all_scores=True,
                            **pipeline_config
                        )
                    else:
                        self.pipelines[task] = pipeline(
                            task,
                            model=model_name,
                            **pipeline_config
                        )
                    
                    logger.info(f"Loaded optimized {task} model: {model_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {task} model: {e}")
                    
        except Exception as e:
            logger.error(f"Transformer optimization failed: {e}")
    
    async def _load_sentence_transformers_optimized(self):
        """Load sentence transformers with optimization."""
        try:
            # Choose model based on optimization level
            model_name = {
                OptimizationLevel.MINIMAL: 'all-MiniLM-L6-v2',
                OptimizationLevel.BALANCED: 'all-mpnet-base-v2',
                OptimizationLevel.MAXIMUM: 'all-mpnet-base-v2',
                OptimizationLevel.ULTRA: 'all-mpnet-base-v2'
            }[self.config.optimization_level]
            
            self.embeddings['sentence_transformer'] = SentenceTransformer(
                model_name,
                device=self.gpu_device,
                cache_folder='./model_cache'
            )
            
            logger.info(f"Loaded optimized sentence transformer: {model_name}")
            
        except Exception as e:
            logger.error(f"Sentence transformer optimization failed: {e}")
    
    def _initialize_vectorizers_optimized(self):
        """Initialize vectorizers with optimization."""
        try:
            # TF-IDF with optimization
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000 if self.config.optimization_level == OptimizationLevel.MINIMAL else 10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=True,
                strip_accents='unicode',
                dtype=np.float32  # Use float32 for memory efficiency
            )
            
            logger.info("Initialized optimized vectorizers")
            
        except Exception as e:
            logger.error(f"Vectorizer optimization failed: {e}")
    
    async def _load_quality_models(self):
        """Load quality assessment models."""
        try:
            # VADER sentiment analyzer
            self.models['vader'] = SentimentIntensityAnalyzer()
            
            # TextBlob for additional analysis
            self.models['textblob'] = TextBlob
            
            logger.info("Loaded quality assessment models")
            
        except Exception as e:
            logger.error(f"Quality model loading failed: {e}")
    
    async def _warm_up_models(self):
        """Warm up models for optimal performance."""
        try:
            warm_up_text = "This is a warm-up text for optimal performance."
            
            # Warm up spaCy models
            for lang in ['en', 'es', 'fr', 'de']:
                if f'spacy_{lang}' in self.models:
                    doc = self.models[f'spacy_{lang}'](warm_up_text)
                    _ = [ent for ent in doc.ents]
            
            # Warm up transformer pipelines
            for task, pipeline_obj in self.pipelines.items():
                try:
                    _ = pipeline_obj(warm_up_text)
                except Exception as e:
                    logger.warning(f"Warm-up failed for {task}: {e}")
            
            # Warm up sentence transformer
            if 'sentence_transformer' in self.embeddings:
                _ = self.embeddings['sentence_transformer'].encode([warm_up_text])
            
            logger.info("Models warmed up successfully")
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
    
    async def _start_background_optimization(self):
        """Start background optimization tasks."""
        self._running = True
        
        # Memory optimization task
        memory_task = asyncio.create_task(self._memory_optimization_loop())
        self._background_tasks.append(memory_task)
        
        # Cache optimization task
        cache_task = asyncio.create_task(self._cache_optimization_loop())
        self._background_tasks.append(cache_task)
        
        # Performance monitoring task
        perf_task = asyncio.create_task(self._performance_monitoring_loop())
        self._background_tasks.append(perf_task)
        
        # Model optimization task
        model_task = asyncio.create_task(self._model_optimization_loop())
        self._background_tasks.append(model_task)
        
        logger.info("Background optimization tasks started")
    
    async def _memory_optimization_loop(self):
        """Background memory optimization."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Check memory usage
                memory_usage = psutil.virtual_memory().percent
                
                if memory_usage > 80:
                    # Trigger garbage collection
                    gc.collect()
                    
                    # Clear model cache if needed
                    if memory_usage > 90:
                        await self.model_cache.clear_old_models()
                
                # Record memory metrics
                await self.monitoring.record_metric_value(
                    metric_name="memory_usage",
                    value=memory_usage,
                    metadata={'source': 'memory_optimization'}
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory optimization error: {e}")
    
    async def _cache_optimization_loop(self):
        """Background cache optimization."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Optimize cache
                await self.cache.optimize()
                
                # Record cache metrics
                cache_stats = self.cache.get_stats()
                await self.monitoring.record_quality_metrics(
                    task="cache_optimization",
                    accuracy=cache_stats['hit_rate']
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Update performance statistics
                self.performance_tracker.update_stats(self.stats)
                
                # Record performance metrics
                await self.monitoring.record_metric_value(
                    metric_name="processing_time",
                    value=self.stats['average_processing_time'],
                    metadata={'source': 'performance_monitoring'}
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _model_optimization_loop(self):
        """Background model optimization."""
        while self._running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Optimize model loading
                await self.model_cache.optimize()
                
                # Clear unused models
                await self.model_cache.clear_unused_models()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Model optimization error: {e}")
    
    async def analyze_text_optimal(
        self,
        text: str,
        language: str = "en",
        tasks: List[str] = None,
        use_cache: bool = True,
        quality_check: bool = True,
        parallel_processing: bool = True
    ) -> Dict[str, Any]:
        """Perform optimal text analysis with maximum performance."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_hit = False
            if use_cache:
                cache_key = self._generate_cache_key(text, language, tasks)
                cached_result = await self.cache.get(
                    text=text,
                    task="optimal_analysis",
                    language=language
                )
                if cached_result:
                    cache_hit = True
                    self.stats['cache_hits'] += 1
                    return cached_result
                else:
                    self.stats['cache_misses'] += 1
            
            # Determine tasks to perform
            if tasks is None:
                tasks = ['sentiment', 'entities', 'keywords', 'readability']
            
            # Perform analysis with optimal strategy
            if parallel_processing and len(tasks) > 1:
                result = await self._parallel_analysis(text, language, tasks)
            else:
                result = await self._sequential_analysis(text, language, tasks)
            
            # Quality assessment
            if quality_check:
                quality_score = await self.quality_assessor.assess_quality(result)
                result['quality_score'] = quality_score
                result['quality_assessment'] = await self.quality_assessor.get_assessment(result)
            
            # Add metadata
            result['processing_time'] = time.time() - start_time
            result['cache_hit'] = cache_hit
            result['timestamp'] = datetime.now().isoformat()
            result['optimization_level'] = self.config.optimization_level.value
            
            # Cache result
            if use_cache and not cache_hit:
                await self.cache.set(
                    text=text,
                    task="optimal_analysis",
                    value=result,
                    language=language,
                    ttl=self.config.cache_ttl
                )
            
            # Update statistics
            self._update_stats(result['processing_time'], quality_score if quality_check else 0)
            
            # Record metrics
            await self.monitoring.record_request(
                task="optimal_analysis",
                processing_time=result['processing_time'],
                success=True,
                language=language,
                text_length=len(text)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['error_count'] += 1
            
            await self.monitoring.record_request(
                task="optimal_analysis",
                processing_time=processing_time,
                success=False,
                error_type=type(e).__name__,
                language=language,
                text_length=len(text)
            )
            
            logger.error(f"Optimal analysis failed: {e}")
            raise
    
    async def _parallel_analysis(self, text: str, language: str, tasks: List[str]) -> Dict[str, Any]:
        """Perform parallel analysis for maximum performance."""
        try:
            # Create tasks for parallel execution
            analysis_tasks = []
            
            for task in tasks:
                if task == 'sentiment':
                    analysis_tasks.append(self._analyze_sentiment_optimal(text, language))
                elif task == 'entities':
                    analysis_tasks.append(self._extract_entities_optimal(text, language))
                elif task == 'keywords':
                    analysis_tasks.append(self._extract_keywords_optimal(text, language))
                elif task == 'readability':
                    analysis_tasks.append(self._analyze_readability_optimal(text, language))
                elif task == 'topics':
                    analysis_tasks.append(self._extract_topics_optimal(text, language))
                elif task == 'embeddings':
                    analysis_tasks.append(self._get_embeddings_optimal(text, language))
            
            # Execute tasks in parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            combined_result = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Task {tasks[i]} failed: {result}")
                    combined_result[tasks[i]] = None
                else:
                    combined_result[tasks[i]] = result
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Parallel analysis failed: {e}")
            raise
    
    async def _sequential_analysis(self, text: str, language: str, tasks: List[str]) -> Dict[str, Any]:
        """Perform sequential analysis."""
        result = {}
        
        for task in tasks:
            try:
                if task == 'sentiment':
                    result['sentiment'] = await self._analyze_sentiment_optimal(text, language)
                elif task == 'entities':
                    result['entities'] = await self._extract_entities_optimal(text, language)
                elif task == 'keywords':
                    result['keywords'] = await self._extract_keywords_optimal(text, language)
                elif task == 'readability':
                    result['readability'] = await self._analyze_readability_optimal(text, language)
                elif task == 'topics':
                    result['topics'] = await self._extract_topics_optimal(text, language)
                elif task == 'embeddings':
                    result['embeddings'] = await self._get_embeddings_optimal(text, language)
            except Exception as e:
                logger.warning(f"Task {task} failed: {e}")
                result[task] = None
        
        return result
    
    async def _analyze_sentiment_optimal(self, text: str, language: str) -> Dict[str, Any]:
        """Optimal sentiment analysis."""
        try:
            results = {}
            
            # Use transformer model if available
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
            ensemble_result = self._ensemble_sentiment(results)
            results['ensemble'] = ensemble_result
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {}
    
    async def _extract_entities_optimal(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Optimal entity extraction."""
        try:
            entities = []
            
            # Use spaCy if available
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
                            'method': 'spacy'
                        })
                except Exception as e:
                    logger.warning(f"spaCy NER failed: {e}")
            
            # Use transformer NER if available
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
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_keywords_optimal(self, text: str, language: str) -> List[str]:
        """Optimal keyword extraction."""
        try:
            keywords = []
            
            # Use TF-IDF
            try:
                vectorizer = self.vectorizers['tfidf']
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                keyword_scores = list(zip(feature_names, scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                keywords = [kw[0] for kw in keyword_scores[:20]]
                
            except Exception as e:
                logger.warning(f"TF-IDF keyword extraction failed: {e}")
            
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []
    
    async def _analyze_readability_optimal(self, text: str, language: str) -> Dict[str, float]:
        """Optimal readability analysis."""
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
            logger.error(f"Readability analysis failed: {e}")
            return {'average_score': 0.0, 'overall_level': 'Unknown'}
    
    async def _extract_topics_optimal(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Optimal topic extraction."""
        try:
            # Simple topic extraction for single text
            # In practice, you'd use LDA or BERTopic for multiple texts
            topics = []
            
            # Extract key phrases as topics
            keywords = await self._extract_keywords_optimal(text, language)
            if keywords:
                topics.append({
                    'id': 0,
                    'words': keywords[:10],
                    'weights': [1.0] * len(keywords[:10]),
                    'coherence_score': 0.0
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []
    
    async def _get_embeddings_optimal(self, text: str, language: str) -> Dict[str, Any]:
        """Optimal embedding generation."""
        try:
            if 'sentence_transformer' in self.embeddings:
                embeddings = self.embeddings['sentence_transformer'].encode(text)
                return {
                    'embeddings': embeddings.tolist(),
                    'dimension': len(embeddings),
                    'model': 'sentence_transformer'
                }
            else:
                return {'embeddings': [], 'dimension': 0, 'model': 'none'}
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {'embeddings': [], 'dimension': 0, 'model': 'none'}
    
    def _ensemble_sentiment(self, sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple sentiment analysis results."""
        try:
            scores = []
            
            # Collect scores from different methods
            for method, result in sentiment_results.items():
                if method == 'vader' and 'compound' in result:
                    scores.append(result['compound'])
                elif method == 'textblob' and 'polarity' in result:
                    scores.append(result['polarity'])
                elif method == 'transformer' and isinstance(result, list) and result:
                    # Extract score from transformer result
                    transformer_score = result[0].get('score', 0) if result[0].get('label') == 'POSITIVE' else -result[0].get('score', 0)
                    scores.append(transformer_score)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.1:
                    sentiment = 'positive'
                elif avg_score < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'score': avg_score,
                    'sentiment': sentiment,
                    'confidence': 1 - abs(avg_score)
                }
            else:
                return {'score': 0, 'sentiment': 'neutral', 'confidence': 0}
                
        except Exception as e:
            logger.error(f"Ensemble sentiment calculation failed: {e}")
            return {'score': 0, 'sentiment': 'neutral', 'confidence': 0}
    
    def _generate_cache_key(self, text: str, language: str, tasks: List[str]) -> str:
        """Generate cache key for text and tasks."""
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        tasks_str = '_'.join(sorted(tasks)) if tasks else 'all'
        return f"{language}:{tasks_str}:{content_hash}"
    
    def _update_stats(self, processing_time: float, quality_score: float):
        """Update system statistics."""
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
            self.stats['quality_scores'].append(quality_score)
    
    async def batch_analyze_optimal(
        self,
        texts: List[str],
        language: str = "en",
        tasks: List[str] = None,
        use_cache: bool = True,
        parallel_processing: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform optimal batch analysis."""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            if parallel_processing:
                # Process in parallel batches
                batch_size = min(self.config.batch_size, len(texts))
                results = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    # Process batch concurrently
                    batch_tasks = [
                        self.analyze_text_optimal(
                            text=text,
                            language=language,
                            tasks=tasks,
                            use_cache=use_cache,
                            parallel_processing=False  # Avoid nested parallelism
                        )
                        for text in batch
                    ]
                    
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Handle results
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Batch analysis error for text {i + j}: {result}")
                            results.append({'error': str(result), 'text': batch[j]})
                        else:
                            results.append(result)
                
                return results
            else:
                # Sequential processing
                results = []
                for text in texts:
                    try:
                        result = await self.analyze_text_optimal(
                            text=text,
                            language=language,
                            tasks=tasks,
                            use_cache=use_cache,
                            parallel_processing=False
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Sequential analysis error: {e}")
                        results.append({'error': str(e), 'text': text})
                
                return results
                
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise
    
    async def get_optimal_status(self) -> Dict[str, Any]:
        """Get comprehensive optimal system status."""
        try:
            # System status
            system_status = {
                'initialized': self.is_initialized,
                'optimization_level': self.config.optimization_level.value,
                'processing_mode': self.config.processing_mode.value,
                'gpu_available': self.gpu_available,
                'gpu_device': self.gpu_device
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
            
            # Quality statistics
            quality_stats = {}
            if self.stats['quality_scores']:
                quality_scores = list(self.stats['quality_scores'])
                quality_stats = {
                    'average_quality': sum(quality_scores) / len(quality_scores),
                    'min_quality': min(quality_scores),
                    'max_quality': max(quality_scores),
                    'quality_samples': len(quality_scores)
                }
            
            # Memory status
            memory_status = await self.memory_monitor.get_status()
            
            # Cache status
            cache_status = self.cache.get_stats()
            
            return {
                'system': system_status,
                'performance': performance_stats,
                'quality': quality_stats,
                'memory': memory_status,
                'cache': cache_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimal status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def shutdown(self):
        """Shutdown optimal NLP system."""
        try:
            logger.info("Shutting down Optimal NLP System...")
            
            # Stop background tasks
            self._running = False
            for task in self._background_tasks:
                task.cancel()
            
            # Stop components
            await self.cache.stop()
            await self.monitoring.stop_monitoring()
            await self.memory_monitor.stop()
            
            # Shutdown executors
            self.executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            # Clear model cache
            await self.model_cache.clear_all()
            
            logger.info("Optimal NLP System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Supporting classes for optimization

class MemoryMonitor:
    """Memory monitoring and optimization."""
    
    def __init__(self, limit_gb: float):
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self._running = False
        self._task = None
    
    async def start(self):
        """Start memory monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop memory monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
    
    async def _monitor_loop(self):
        """Background memory monitoring."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                memory_usage = psutil.virtual_memory()
                if memory_usage.percent > 90:
                    # Trigger garbage collection
                    gc.collect()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get memory status."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'limit_gb': self.limit_bytes / (1024**3)
        }

class ModelCache:
    """Model caching for optimal performance."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    async def get_model(self, model_name: str):
        """Get cached model."""
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            return self.cache[model_name]
        return None
    
    async def cache_model(self, model_name: str, model):
        """Cache model."""
        if len(self.cache) >= self.max_size:
            await self.clear_old_models()
        
        self.cache[model_name] = model
        self.access_times[model_name] = time.time()
    
    async def clear_old_models(self):
        """Clear least recently used models."""
        if len(self.cache) > self.max_size // 2:
            # Remove oldest models
            sorted_models = sorted(self.access_times.items(), key=lambda x: x[1])
            for model_name, _ in sorted_models[:len(sorted_models)//2]:
                if model_name in self.cache:
                    del self.cache[model_name]
                    del self.access_times[model_name]
    
    async def clear_unused_models(self):
        """Clear unused models."""
        current_time = time.time()
        unused_threshold = 3600  # 1 hour
        
        unused_models = [
            name for name, access_time in self.access_times.items()
            if current_time - access_time > unused_threshold
        ]
        
        for model_name in unused_models:
            if model_name in self.cache:
                del self.cache[model_name]
                del self.access_times[model_name]
    
    async def clear_all(self):
        """Clear all cached models."""
        self.cache.clear()
        self.access_times.clear()
    
    async def optimize(self):
        """Optimize model cache."""
        await self.clear_unused_models()

class PerformanceTracker:
    """Performance tracking and optimization."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.thresholds = {
            'processing_time': 5.0,  # seconds
            'memory_usage': 80.0,    # percent
            'error_rate': 5.0        # percent
        }
    
    def update_stats(self, stats: Dict[str, Any]):
        """Update performance statistics."""
        self.metrics['processing_time'].append(stats.get('average_processing_time', 0))
        self.metrics['requests_processed'].append(stats.get('requests_processed', 0))
        self.metrics['error_count'].append(stats.get('error_count', 0))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[metric] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': self._calculate_trend(values)
                }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'stable'
        
        recent = values[-5:] if len(values) >= 5 else values
        if len(recent) < 2:
            return 'stable'
        
        if recent[-1] > recent[0]:
            return 'increasing'
        elif recent[-1] < recent[0]:
            return 'decreasing'
        else:
            return 'stable'

class QualityAssessor:
    """Quality assessment for NLP results."""
    
    def __init__(self):
        self.quality_weights = {
            'sentiment': 0.3,
            'entities': 0.2,
            'keywords': 0.2,
            'readability': 0.2,
            'completeness': 0.1
        }
    
    async def assess_quality(self, result: Dict[str, Any]) -> float:
        """Assess overall quality of analysis result."""
        try:
            quality_score = 0.0
            total_weight = 0.0
            
            # Sentiment quality
            if 'sentiment' in result and result['sentiment']:
                sentiment_quality = self._assess_sentiment_quality(result['sentiment'])
                quality_score += sentiment_quality * self.quality_weights['sentiment']
                total_weight += self.quality_weights['sentiment']
            
            # Entity quality
            if 'entities' in result and result['entities']:
                entity_quality = self._assess_entity_quality(result['entities'])
                quality_score += entity_quality * self.quality_weights['entities']
                total_weight += self.quality_weights['entities']
            
            # Keyword quality
            if 'keywords' in result and result['keywords']:
                keyword_quality = self._assess_keyword_quality(result['keywords'])
                quality_score += keyword_quality * self.quality_weights['keywords']
                total_weight += self.quality_weights['keywords']
            
            # Readability quality
            if 'readability' in result and result['readability']:
                readability_quality = self._assess_readability_quality(result['readability'])
                quality_score += readability_quality * self.quality_weights['readability']
                total_weight += self.quality_weights['readability']
            
            # Completeness
            completeness = self._assess_completeness(result)
            quality_score += completeness * self.quality_weights['completeness']
            total_weight += self.quality_weights['completeness']
            
            return quality_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0
    
    def _assess_sentiment_quality(self, sentiment: Dict[str, Any]) -> float:
        """Assess sentiment analysis quality."""
        try:
            if 'ensemble' in sentiment and sentiment['ensemble']:
                confidence = sentiment['ensemble'].get('confidence', 0)
                return min(1.0, confidence)
            return 0.5  # Default quality
        except:
            return 0.0
    
    def _assess_entity_quality(self, entities: List[Dict[str, Any]]) -> float:
        """Assess entity extraction quality."""
        try:
            if not entities:
                return 0.0
            
            # Check for confidence scores
            confidences = [e.get('confidence', 0) for e in entities if 'confidence' in e]
            if confidences:
                return sum(confidences) / len(confidences)
            
            # Quality based on number of entities
            entity_count = len(entities)
            return min(1.0, entity_count / 10)  # Normalize to 0-1
            
        except:
            return 0.0
    
    def _assess_keyword_quality(self, keywords: List[str]) -> float:
        """Assess keyword extraction quality."""
        try:
            if not keywords:
                return 0.0
            
            # Quality based on number of keywords
            keyword_count = len(keywords)
            return min(1.0, keyword_count / 15)  # Normalize to 0-1
            
        except:
            return 0.0
    
    def _assess_readability_quality(self, readability: Dict[str, Any]) -> float:
        """Assess readability analysis quality."""
        try:
            if 'average_score' in readability:
                score = readability['average_score']
                # Normalize to 0-1 (assuming 0-100 scale)
                return min(1.0, max(0.0, score / 100))
            return 0.5  # Default quality
        except:
            return 0.0
    
    def _assess_completeness(self, result: Dict[str, Any]) -> float:
        """Assess result completeness."""
        try:
            expected_fields = ['sentiment', 'entities', 'keywords', 'readability']
            present_fields = sum(1 for field in expected_fields if field in result and result[field])
            return present_fields / len(expected_fields)
        except:
            return 0.0
    
    async def get_assessment(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed quality assessment."""
        try:
            assessment = {
                'overall_quality': await self.assess_quality(result),
                'sentiment_quality': self._assess_sentiment_quality(result.get('sentiment', {})),
                'entity_quality': self._assess_entity_quality(result.get('entities', [])),
                'keyword_quality': self._assess_keyword_quality(result.get('keywords', [])),
                'readability_quality': self._assess_readability_quality(result.get('readability', {})),
                'completeness': self._assess_completeness(result)
            }
            
            # Generate recommendations
            recommendations = []
            if assessment['overall_quality'] < 0.7:
                recommendations.append("Consider improving text quality for better analysis")
            if assessment['sentiment_quality'] < 0.5:
                recommendations.append("Sentiment analysis may need improvement")
            if assessment['entity_quality'] < 0.5:
                recommendations.append("Entity extraction quality could be enhanced")
            if assessment['keyword_quality'] < 0.5:
                recommendations.append("Keyword extraction could be improved")
            if assessment['readability_quality'] < 0.5:
                recommendations.append("Text readability could be enhanced")
            
            assessment['recommendations'] = recommendations
            
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment generation failed: {e}")
            return {'overall_quality': 0.0, 'recommendations': ['Assessment failed']}

# Global optimal NLP system instance
optimal_nlp_system = OptimalNLPSystem()












