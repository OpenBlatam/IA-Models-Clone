"""
Ultra Fast NLP System
=====================

Sistema NLP ultra-rápido con optimizaciones extremas
para máximo rendimiento y velocidad.
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import numpy as np
import torch
import psutil
import gc
from functools import lru_cache
import pickle
import gzip
from contextlib import asynccontextmanager

# Ultra-fast imports
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

logger = logging.getLogger(__name__)

class UltraFastConfig:
    """Configuración ultra-rápida del sistema."""
    
    def __init__(self):
        # Performance settings
        self.max_workers = mp.cpu_count() * 2
        self.batch_size = 128
        self.max_concurrent = 200
        
        # Memory optimization
        self.memory_limit_gb = 16.0
        self.cache_size_mb = 8192
        self.model_cache_size = 50
        
        # GPU optimization
        self.gpu_memory_fraction = 0.9
        self.mixed_precision = True
        self.gradient_checkpointing = True
        
        # Caching
        self.enable_smart_cache = True
        self.cache_compression = True
        self.cache_ttl = 7200  # 2 hours
        
        # Ultra-fast settings
        self.ultra_fast_mode = True
        self.skip_quality_check = True
        self.minimal_analysis = True
        self.parallel_processing = True
        self.gpu_acceleration = True

@dataclass
class UltraFastResult:
    """Resultado ultra-rápido."""
    text: str
    sentiment: Dict[str, Any]
    entities: List[Dict[str, Any]]
    keywords: List[str]
    processing_time: float
    cache_hit: bool
    timestamp: datetime

class UltraFastCache:
    """Caché ultra-rápido con optimizaciones extremas."""
    
    def __init__(self, max_size: int = 50000, max_memory_mb: int = 8192):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.access_times = {}
        self.memory_usage = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result ultra-fast."""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 7200):
        """Set cached result ultra-fast."""
        with self._lock:
            # Check memory limit
            if self.memory_usage > self.max_memory_mb * 1024 * 1024:
                self._evict_oldest()
            
            # Compress if needed
            if self.cache_compression and len(str(value)) > 1000:
                value = gzip.compress(pickle.dumps(value))
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.memory_usage += len(str(value))
    
    def _evict_oldest(self):
        """Evict oldest entries ultra-fast."""
        if not self.access_times:
            return
        
        # Remove 20% of oldest entries
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        to_remove = sorted_items[:len(sorted_items) // 5]
        
        for key, _ in to_remove:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'memory_usage_mb': self.memory_usage / (1024 * 1024),
            'hit_rate': 0.0  # Will be calculated by system
        }

class UltraFastNLP:
    """Sistema NLP ultra-rápido con optimizaciones extremas."""
    
    def __init__(self, config: UltraFastConfig = None):
        """Initialize ultra-fast NLP system."""
        self.config = config or UltraFastConfig()
        self.is_initialized = False
        
        # Ultra-fast components
        self.models = {}
        self.pipelines = {}
        self.cache = UltraFastCache(
            max_size=50000,
            max_memory_mb=self.config.cache_size_mb
        )
        
        # Performance optimization
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
            'total_processing_time': 0.0
        }
        
        # Background optimization
        self._background_tasks = []
        self._running = False
    
    async def initialize(self):
        """Initialize ultra-fast NLP system."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info("Initializing Ultra-Fast NLP System...")
            
            # Load ultra-fast models
            await self._load_ultra_fast_models()
            
            # Start background optimization
            await self._start_background_optimization()
            
            self.is_initialized = True
            init_time = time.time() - start_time
            
            logger.info(f"Ultra-Fast NLP System initialized in {init_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ultra-Fast NLP System: {e}")
            raise
    
    async def _load_ultra_fast_models(self):
        """Load models with ultra-fast optimization."""
        try:
            # Load ultra-fast spaCy model
            if self.config.ultra_fast_mode:
                # Use smallest model for speed
                self.models['spacy_en'] = spacy.load(
                    'en_core_web_sm',
                    disable=['parser', 'ner']  # Disable heavy components
                )
            else:
                self.models['spacy_en'] = spacy.load('en_core_web_sm')
            
            # Load ultra-fast transformer models
            if self.gpu_available:
                device = 0
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            else:
                device = -1
            
            # Ultra-fast sentiment analysis
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device,
                batch_size=self.config.batch_size,
                return_all_scores=True
            )
            
            # Ultra-fast NER
            self.pipelines['ner'] = pipeline(
                "ner",
                model="dbmdz/bert-base-cased-finetuned-conll03-english",
                device=device,
                batch_size=self.config.batch_size,
                aggregation_strategy="simple"
            )
            
            # Ultra-fast sentence transformer
            self.models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',  # Smallest model
                device=self.gpu_device,
                cache_folder='./ultra_fast_cache'
            )
            
            # Ultra-fast VADER
            self.models['vader'] = SentimentIntensityAnalyzer()
            
            logger.info("Ultra-fast models loaded successfully")
            
        except Exception as e:
            logger.error(f"Ultra-fast model loading failed: {e}")
            raise
    
    async def _start_background_optimization(self):
        """Start background optimization tasks."""
        self._running = True
        
        # Memory optimization task
        memory_task = asyncio.create_task(self._memory_optimization_loop())
        self._background_tasks.append(memory_task)
        
        # Cache optimization task
        cache_task = asyncio.create_task(self._cache_optimization_loop())
        self._background_tasks.append(cache_task)
        
        logger.info("Background optimization tasks started")
    
    async def _memory_optimization_loop(self):
        """Background memory optimization."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Every 10 seconds
                
                # Check memory usage
                memory_usage = psutil.virtual_memory().percent
                
                if memory_usage > 85:
                    # Aggressive garbage collection
                    gc.collect()
                    
                    # Clear unused models
                    if memory_usage > 95:
                        await self._clear_unused_models()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory optimization error: {e}")
    
    async def _cache_optimization_loop(self):
        """Background cache optimization."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Optimize cache
                self.cache._evict_oldest()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    async def _clear_unused_models(self):
        """Clear unused models to free memory."""
        try:
            # Clear model cache if needed
            if hasattr(self, 'model_cache'):
                self.model_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Model clearing failed: {e}")
    
    async def analyze_ultra_fast(
        self,
        text: str,
        language: str = "en",
        use_cache: bool = True,
        parallel_processing: bool = True
    ) -> UltraFastResult:
        """Perform ultra-fast text analysis."""
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
            
            # Perform ultra-fast analysis
            if parallel_processing:
                result = await self._parallel_ultra_fast_analysis(text, language)
            else:
                result = await self._sequential_ultra_fast_analysis(text, language)
            
            # Create result
            processing_time = time.time() - start_time
            result = UltraFastResult(
                text=text,
                sentiment=result.get('sentiment', {}),
                entities=result.get('entities', []),
                keywords=result.get('keywords', []),
                processing_time=processing_time,
                cache_hit=cache_hit,
                timestamp=datetime.now()
            )
            
            # Cache result
            if use_cache and not cache_hit:
                self.cache.set(cache_key, result, ttl=self.config.cache_ttl)
            
            # Update statistics
            self._update_stats(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Ultra-fast analysis failed: {e}")
            raise
    
    async def _parallel_ultra_fast_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """Perform parallel ultra-fast analysis."""
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Sentiment analysis
            tasks.append(self._analyze_sentiment_ultra_fast(text))
            
            # Entity extraction
            tasks.append(self._extract_entities_ultra_fast(text, language))
            
            # Keyword extraction
            tasks.append(self._extract_keywords_ultra_fast(text))
            
            # Execute tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_result = {}
            task_names = ['sentiment', 'entities', 'keywords']
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Task {task_names[i]} failed: {result}")
                    combined_result[task_names[i]] = {}
                else:
                    combined_result[task_names[i]] = result
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Parallel ultra-fast analysis failed: {e}")
            return {}
    
    async def _sequential_ultra_fast_analysis(self, text: str, language: str) -> Dict[str, Any]:
        """Perform sequential ultra-fast analysis."""
        result = {}
        
        try:
            # Sentiment analysis
            result['sentiment'] = await self._analyze_sentiment_ultra_fast(text)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            result['sentiment'] = {}
        
        try:
            # Entity extraction
            result['entities'] = await self._extract_entities_ultra_fast(text, language)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            result['entities'] = []
        
        try:
            # Keyword extraction
            result['keywords'] = await self._extract_keywords_ultra_fast(text)
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            result['keywords'] = []
        
        return result
    
    async def _analyze_sentiment_ultra_fast(self, text: str) -> Dict[str, Any]:
        """Ultra-fast sentiment analysis."""
        try:
            # Use VADER for speed
            vader_scores = self.models['vader'].polarity_scores(text)
            
            # Use transformer if available and not too slow
            transformer_result = None
            if len(text) < 500:  # Only for short texts
                try:
                    transformer_result = self.pipelines['sentiment'](text)
                except Exception:
                    pass
            
            # Combine results
            result = {
                'vader': vader_scores,
                'transformer': transformer_result,
                'ensemble': {
                    'score': vader_scores['compound'],
                    'sentiment': 'positive' if vader_scores['compound'] > 0.1 else 'negative' if vader_scores['compound'] < -0.1 else 'neutral'
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ultra-fast sentiment analysis failed: {e}")
            return {}
    
    async def _extract_entities_ultra_fast(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Ultra-fast entity extraction."""
        try:
            entities = []
            
            # Use spaCy for speed
            if 'spacy_en' in self.models and language == 'en':
                doc = self.models['spacy_en'](text)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 1.0
                    })
            
            # Use transformer NER for short texts only
            if len(text) < 300 and 'ner' in self.pipelines:
                try:
                    ner_results = self.pipelines['ner'](text)
                    for entity in ner_results:
                        entities.append({
                            'text': entity['word'],
                            'label': entity['entity_group'],
                            'start': entity.get('start', 0),
                            'end': entity.get('end', len(entity['word'])),
                            'confidence': entity.get('score', 0.0)
                        })
                except Exception:
                    pass
            
            return entities
            
        except Exception as e:
            logger.error(f"Ultra-fast entity extraction failed: {e}")
            return []
    
    async def _extract_keywords_ultra_fast(self, text: str) -> List[str]:
        """Ultra-fast keyword extraction."""
        try:
            # Simple keyword extraction for speed
            words = text.lower().split()
            
            # Filter out common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            # Extract keywords
            keywords = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Count frequency
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Return top keywords
            return [word for word, freq in sorted_keywords[:10]]
            
        except Exception as e:
            logger.error(f"Ultra-fast keyword extraction failed: {e}")
            return []
    
    def _generate_cache_key(self, text: str, language: str) -> str:
        """Generate cache key ultra-fast."""
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
        return f"{language}:{content_hash}"
    
    def _update_stats(self, processing_time: float):
        """Update statistics ultra-fast."""
        self.stats['requests_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # Update average processing time
        if self.stats['requests_processed'] > 0:
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['requests_processed']
            )
    
    async def batch_analyze_ultra_fast(
        self,
        texts: List[str],
        language: str = "en",
        use_cache: bool = True,
        parallel_processing: bool = True
    ) -> List[UltraFastResult]:
        """Perform ultra-fast batch analysis."""
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
                        self.analyze_ultra_fast(
                            text=text,
                            language=language,
                            use_cache=use_cache,
                            parallel_processing=False
                        )
                        for text in batch
                    ]
                    
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Handle results
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Batch analysis error for text {i + j}: {result}")
                            results.append(UltraFastResult(
                                text=batch[j],
                                sentiment={},
                                entities=[],
                                keywords=[],
                                processing_time=0,
                                cache_hit=False,
                                timestamp=datetime.now()
                            ))
                        else:
                            results.append(result)
                
                return results
            else:
                # Sequential processing
                results = []
                for text in texts:
                    try:
                        result = await self.analyze_ultra_fast(
                            text=text,
                            language=language,
                            use_cache=use_cache,
                            parallel_processing=False
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Sequential analysis error: {e}")
                        results.append(UltraFastResult(
                            text=text,
                            sentiment={},
                            entities=[],
                            keywords=[],
                            processing_time=0,
                            cache_hit=False,
                            timestamp=datetime.now()
                        ))
                
                return results
                
        except Exception as e:
            logger.error(f"Ultra-fast batch analysis failed: {e}")
            raise
    
    async def get_ultra_fast_status(self) -> Dict[str, Any]:
        """Get ultra-fast system status."""
        try:
            # System status
            system_status = {
                'initialized': self.is_initialized,
                'ultra_fast_mode': self.config.ultra_fast_mode,
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
                'total_processing_time': self.stats['total_processing_time']
            }
            
            # Cache status
            cache_status = self.cache.get_stats()
            
            # Memory status
            memory_status = {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            }
            
            return {
                'system': system_status,
                'performance': performance_stats,
                'cache': cache_status,
                'memory': memory_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get ultra-fast status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def shutdown(self):
        """Shutdown ultra-fast NLP system."""
        try:
            logger.info("Shutting down Ultra-Fast NLP System...")
            
            # Stop background tasks
            self._running = False
            for task in self._background_tasks:
                task.cancel()
            
            # Shutdown executors
            self.executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            logger.info("Ultra-Fast NLP System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

# Global ultra-fast NLP system instance
ultra_fast_nlp = UltraFastNLP()












