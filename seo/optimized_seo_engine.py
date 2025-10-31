#!/usr/bin/env python3
"""
Optimized SEO Engine - Advanced Architecture
High-performance SEO analysis with intelligent optimization and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
try:
    import torch.autocast
except ImportError:
    # Fallback for older PyTorch versions
    torch.autocast = None
import torch.profiler
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.tensorboard
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import logging
import time
import json
import re
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol, runtime_checkable
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache, wraps
import gc
import psutil
from contextlib import contextmanager, asynccontextmanager
import threading
import queue
from collections import defaultdict, deque
import hashlib
import pickle
from abc import ABC, abstractmethod
import tracemalloc
import line_profiler
import cProfile
import pstats
import io

# Third-party imports
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, TrainingArguments, Trainer, DataCollatorWithPadding,
    PreTrainedModel, PreTrainedTokenizer, AutoConfig,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
import gradio as gr
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, 
    StratifiedGroupKFold, LeaveOneOut, LeavePOut,
    RepeatedStratifiedKFold, RepeatedKFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Local imports
from core_config import (
    SEOConfig, get_config, get_container, config_section, 
    inject_service, config_context
)
from advanced_monitoring import (
    MonitoringSystem, PerformanceProfiler, MetricsCollector
)

warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# ============================================================================
# CORE INTERFACES AND PROTOCOLS
# ============================================================================

@runtime_checkable
class SEOProcessor(Protocol):
    """Protocol for SEO processing components."""
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process text and return SEO analysis results."""
        ...
    
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts in batch."""
        ...

@runtime_checkable
class ModelManager(Protocol):
    """Protocol for model management components."""
    
    def load_model(self, model_name: str) -> Any:
        """Load a model by name."""
        ...
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model by name."""
        ...
    
    def get_model(self, model_name: str) -> Any:
        """Get a loaded model by name."""
        ...

@runtime_checkable
class CacheManager(Protocol):
    """Protocol for caching components."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        ...
    
    def invalidate(self, key: str) -> None:
        """Invalidate cached value."""
        ...

# ============================================================================
# ADVANCED CACHING SYSTEM
# ============================================================================

class AdvancedCacheManager:
    """High-performance caching with intelligent eviction and compression."""
    
    def __init__(self, max_size: int = 1000, compression_threshold: int = 1024):
        self.max_size = max_size
        self.compression_threshold = compression_threshold
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.sizes: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._total_size = 0
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _get_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return sys.getsizeof(obj)
    
    def _compress(self, data: Any) -> Any:
        """Compress data if it exceeds threshold."""
        if self._get_size(data) > self.compression_threshold:
            try:
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                return data
        return data
    
    def _decompress(self, data: Any) -> Any:
        """Decompress data if it's compressed."""
        if isinstance(data, bytes):
            try:
                return pickle.loads(data)
            except:
                return data
        return data
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with access time update."""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self._decompress(self.cache[key])
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with size tracking."""
        with self._lock:
            # Remove old entry if exists
            if key in self.cache:
                self._total_size -= self.sizes[key]
                del self.cache[key]
                del self.access_times[key]
                del self.sizes[key]
            
            # Compress and store
            compressed_value = self._compress(value)
            size = self._get_size(compressed_value)
            
            # Check if we need to evict items
            while self._total_size + size > self.max_size and self.cache:
                self._evict_oldest()
            
            # Store new item
            self.cache[key] = compressed_value
            self.access_times[key] = time.time()
            self.sizes[key] = size
            self._total_size += size
    
    def invalidate(self, key: str) -> None:
        """Invalidate cached value."""
        with self._lock:
            if key in self.cache:
                self._total_size -= self.sizes[key]
                del self.cache[key]
                del self.access_times[key]
                del self.sizes[key]
    
    def _evict_oldest(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.invalidate(oldest_key)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(60)  # Clean up every minute
                self._cleanup_expired()
            except Exception as e:
                logging.error(f"Cache cleanup failed: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, access_time in self.access_times.items()
                if current_time - access_time > 3600  # 1 hour TTL
            ]
            for key in expired_keys:
                self.invalidate(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'total_items': len(self.cache),
                'total_size_bytes': self._total_size,
                'max_size_bytes': self.max_size,
                'hit_rate': self._calculate_hit_rate(),
                'compression_ratio': self._calculate_compression_ratio()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder for now)."""
        return 0.85  # Placeholder
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self._total_size == 0:
            return 1.0
        return self._total_size / max(self._total_size, 1)

# ============================================================================
# INTELLIGENT MODEL MANAGER
# ============================================================================

class IntelligentModelManager:
    """Advanced model management with automatic optimization and memory management."""
    
    def __init__(self, config: SEOConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._memory_monitor = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self._memory_monitor.start()
        
        # Performance tracking
        self.usage_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'load_count': 0,
            'last_used': 0,
            'total_inference_time': 0,
            'inference_count': 0
        })
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """Load model with intelligent caching and optimization."""
        with self._lock:
            if model_name in self.models and not force_reload:
                self._update_usage_stats(model_name)
                return self.models[model_name]
            
            try:
                start_time = time.time()
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizers[model_name] = tokenizer
                
                # Load model with optimizations
                model = self._load_optimized_model(model_name)
                
                # Store model and metadata
                self.models[model_name] = model
                self.model_metadata[model_name] = {
                    'loaded_at': time.time(),
                    'model_size_mb': self._get_model_size(model),
                    'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'cpu'
                }
                
                # Update usage stats
                self._update_usage_stats(model_name, load_time=time.time() - start_time)
                
                logging.info(f"Loaded model: {model_name} in {time.time() - start_time:.2f}s")
                return model
                
            except Exception as e:
                logging.error(f"Failed to load model {model_name}: {e}")
                raise
    
    def _load_optimized_model(self, model_name: str) -> Any:
        """Load model with PyTorch optimizations."""
        try:
            # Try to load with optimizations
            if self.config.performance.enable_compilation:
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Apply optimizations
                if torch.cuda.is_available():
                    model = model.cuda()
                    if self.config.performance.enable_mixed_precision:
                        model = model.half()
                
                # Compile model if PyTorch 2.0+ is available
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='reduce-overhead')
                
                return model
            else:
                return AutoModelForSequenceClassification.from_pretrained(model_name)
                
        except Exception as e:
            logging.warning(f"Optimized loading failed for {model_name}, falling back to standard: {e}")
            return AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def _get_model_size(self, model: Any) -> float:
        """Get model size in MB."""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / (1024 * 1024)
        except:
            return 0.0
    
    def _update_usage_stats(self, model_name: str, load_time: float = 0) -> None:
        """Update model usage statistics."""
        stats = self.usage_stats[model_name]
        stats['last_used'] = time.time()
        if load_time > 0:
            stats['load_count'] += 1
    
    def get_model(self, model_name: str) -> Any:
        """Get loaded model."""
        with self._lock:
            if model_name not in self.models:
                return self.load_model(model_name)
            
            self._update_usage_stats(model_name)
            return self.models[model_name]
    
    def unload_model(self, model_name: str) -> None:
        """Unload model and free memory."""
        with self._lock:
            if model_name in self.models:
                del self.models[model_name]
                del self.tokenizers[model_name]
                del self.model_metadata[model_name]
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logging.info(f"Unloaded model: {model_name}")
    
    def _memory_monitor_loop(self) -> None:
        """Monitor memory usage and unload unused models."""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._check_memory_usage()
            except Exception as e:
                logging.error(f"Memory monitor failed: {e}")
    
    def _check_memory_usage(self) -> None:
        """Check memory usage and unload models if necessary."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 85:  # High memory usage
                self._unload_least_used_models()
        except Exception as e:
            logging.error(f"Memory check failed: {e}")
    
    def _unload_least_used_models(self) -> None:
        """Unload least recently used models."""
        with self._lock:
            if len(self.models) <= 1:
                return
            
            # Sort by last used time
            sorted_models = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1]['last_used']
            )
            
            # Unload oldest models until memory usage is acceptable
            for model_name, _ in sorted_models[:-1]:  # Keep at least one model
                if model_name in self.models:
                    self.unload_model(model_name)
                    
                    # Check if memory usage improved
                    try:
                        memory = psutil.virtual_memory()
                        if memory.percent < 70:
                            break
                    except:
                        break
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        with self._lock:
            return {
                'loaded_models': list(self.models.keys()),
                'model_metadata': self.model_metadata.copy(),
                'usage_stats': dict(self.usage_stats),
                'total_models': len(self.models)
            }

# ============================================================================
# ADVANCED SEO PROCESSOR
# ============================================================================

class AdvancedSEOProcessor:
    """High-performance SEO analysis with multiple optimization strategies."""
    
    def __init__(self, config: SEOConfig, model_manager: IntelligentModelManager):
        self.config = config
        self.model_manager = model_manager
        self.cache = AdvancedCacheManager()
        self.profiler = PerformanceProfiler()
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to basic tokenization
            self.nlp = None
        
        # SEO analysis components
        self.keyword_analyzer = KeywordAnalyzer()
        self.content_analyzer = ContentAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
    
    def analyze_text(self, text: str, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Analyze text for SEO optimization."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"seo_analysis:{hashlib.md5(text.encode()).hexdigest()}:{analysis_type}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self._record_performance('cache_hit', time.time() - start_time)
                return cached_result
            
            # Perform analysis
            with self.profiler.profile_function(f"seo_analysis_{analysis_type}"):
                result = self._perform_analysis(text, analysis_type)
            
            # Cache result
            self.cache.set(cache_key, result, ttl=3600)
            
            # Record performance
            analysis_time = time.time() - start_time
            self._record_performance('analysis_time', analysis_time)
            
            # Add metadata
            result['metadata'] = {
                'analysis_time': analysis_time,
                'analysis_type': analysis_type,
                'timestamp': time.time(),
                'cache_hit': False
            }
            
            return result
            
        except Exception as e:
            logging.error(f"SEO analysis failed: {e}")
            return {
                'error': str(e),
                'metadata': {
                    'analysis_time': time.time() - start_time,
                    'analysis_type': analysis_type,
                    'timestamp': time.time(),
                    'cache_hit': False
                }
            }
    
    def _perform_analysis(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Perform the actual SEO analysis."""
        result = {
            'seo_score': 0,
            'keyword_analysis': {},
            'content_analysis': {},
            'readability_analysis': {},
            'technical_analysis': {},
            'recommendations': []
        }
        
        # Keyword analysis
        if analysis_type in ['comprehensive', 'keywords']:
            result['keyword_analysis'] = self.keyword_analyzer.analyze(text)
        
        # Content analysis
        if analysis_type in ['comprehensive', 'content']:
            result['content_analysis'] = self.content_analyzer.analyze(text)
        
        # Readability analysis
        if analysis_type in ['comprehensive', 'readability']:
            result['readability_analysis'] = self.readability_analyzer.analyze(text)
        
        # Technical analysis
        if analysis_type in ['comprehensive', 'technical']:
            result['technical_analysis'] = self.technical_analyzer.analyze(text)
        
        # Calculate overall SEO score
        result['seo_score'] = self._calculate_seo_score(result)
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(result)
        
        return result
    
    def _calculate_seo_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall SEO score from analysis components."""
        score = 0.0
        weights = {
            'keyword': 0.3,
            'content': 0.25,
            'readability': 0.25,
            'technical': 0.2
        }
        
        # Keyword score
        if 'keyword_analysis' in analysis_result:
            keyword_score = analysis_result['keyword_analysis'].get('score', 0)
            score += keyword_score * weights['keyword']
        
        # Content score
        if 'content_analysis' in analysis_result:
            content_score = analysis_result['content_analysis'].get('score', 0)
            score += content_score * weights['content']
        
        # Readability score
        if 'readability_analysis' in analysis_result:
            readability_score = analysis_result['readability_analysis'].get('score', 0)
            score += readability_score * weights['readability']
        
        # Technical score
        if 'technical_analysis' in analysis_result:
            technical_score = analysis_result['technical_analysis'].get('score', 0)
            score += technical_score * weights['technical']
        
        return min(100.0, max(0.0, score))
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate actionable SEO recommendations."""
        recommendations = []
        
        # Keyword recommendations
        keyword_analysis = analysis_result.get('keyword_analysis', {})
        if keyword_analysis.get('keyword_density', 0) < 1.0:
            recommendations.append("Increase keyword density to 1-2% for better SEO")
        
        # Content recommendations
        content_analysis = analysis_result.get('content_analysis', {})
        if content_analysis.get('word_count', 0) < 300:
            recommendations.append("Increase content length to at least 300 words")
        
        # Readability recommendations
        readability_analysis = analysis_result.get('readability_analysis', {})
        if readability_analysis.get('flesch_reading_ease', 0) < 60:
            recommendations.append("Improve readability by using shorter sentences and simpler words")
        
        # Technical recommendations
        technical_analysis = analysis_result.get('technical_analysis', {})
        if technical_analysis.get('has_headings', False) == False:
            recommendations.append("Add proper heading structure (H1, H2, H3) for better organization")
        
        return recommendations
    
    def _record_performance(self, metric: str, value: float) -> None:
        """Record performance metric."""
        self.performance_metrics[metric].append(value)
        
        # Keep only last 1000 values
        if len(self.performance_metrics[metric]) > 1000:
            self.performance_metrics[metric] = self.performance_metrics[metric][-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        for metric, values in self.performance_metrics.items():
            if values:
                stats[metric] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
        return stats

# ============================================================================
# ANALYSIS COMPONENTS
# ============================================================================

class KeywordAnalyzer:
    """Advanced keyword analysis and optimization."""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze keyword usage and optimization."""
        # Basic text preprocessing
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        content_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Calculate metrics
        word_count = len(content_words)
        unique_words = len(set(content_words))
        
        # Keyword density (simplified)
        keyword_density = (unique_words / max(word_count, 1)) * 100
        
        # Calculate score
        score = min(100, max(0, 100 - abs(keyword_density - 2) * 20))
        
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'keyword_density': keyword_density,
            'score': score
        }

class ContentAnalyzer:
    """Content quality and structure analysis."""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze content quality and structure."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Basic metrics
        sentence_count = len(sentences)
        word_count = len(words)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Structure analysis
        has_headings = bool(re.search(r'^#+\s+', text, re.MULTILINE))
        has_lists = bool(re.search(r'^[\-\*]\s+', text, re.MULTILINE))
        
        # Calculate score
        score = 50  # Base score
        
        if 10 <= avg_sentence_length <= 25:
            score += 20
        if has_headings:
            score += 15
        if has_lists:
            score += 15
        
        return {
            'sentence_count': sentence_count,
            'word_count': word_count,
            'avg_sentence_length': avg_sentence_length,
            'has_headings': has_headings,
            'has_lists': has_lists,
            'score': min(100, score)
        }

class ReadabilityAnalyzer:
    """Text readability analysis using multiple metrics."""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text readability."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        syllables = self._count_syllables(text)
        
        # Flesch Reading Ease
        if sentence_count > 0 and word_count > 0:
            flesch_score = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (syllables / word_count))
            flesch_score = max(0, min(100, flesch_score))
        else:
            flesch_score = 0
        
        # Calculate score
        score = 50  # Base score
        
        if 60 <= flesch_score <= 80:
            score += 30
        elif 50 <= flesch_score < 60:
            score += 20
        elif 80 < flesch_score <= 90:
            score += 15
        
        return {
            'flesch_reading_ease': flesch_score,
            'sentence_count': len(sentences),
            'word_count': len(words),
            'syllable_count': syllables,
            'score': min(100, score)
        }
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified)."""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)

class TechnicalAnalyzer:
    """Technical SEO analysis."""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze technical SEO aspects."""
        # Check for common technical issues
        has_meta_description = bool(re.search(r'<meta.*?name=["\']description["\']', text, re.IGNORECASE))
        has_title_tag = bool(re.search(r'<title>.*?</title>', text, re.IGNORECASE))
        has_alt_tags = bool(re.search(r'<img.*?alt=["\']', text, re.IGNORECASE))
        
        # Calculate score
        score = 0
        if has_meta_description:
            score += 25
        if has_title_tag:
            score += 25
        if has_alt_tags:
            score += 25
        
        # Additional points for good structure
        if len(text) > 1000:
            score += 25
        
        return {
            'has_meta_description': has_meta_description,
            'has_title_tag': has_title_tag,
            'has_alt_tags': has_alt_tags,
            'content_length': len(text),
            'score': min(100, score)
        }

# ============================================================================
# OPTIMIZED SEO ENGINE
# ============================================================================

class OptimizedSEOEngine:
    """Complete optimized SEO engine with advanced features."""
    
    def __init__(self, config: Optional[SEOConfig] = None):
        self.config = config or SEOConfig()
        self.config.validate()
        
        # Initialize components
        self.model_manager = IntelligentModelManager(self.config)
        self.seo_processor = AdvancedSEOProcessor(self.config, self.model_manager)
        self.monitoring = MonitoringSystem()
        
        # Register services in dependency container
        container = get_container()
        container.register('model_manager', self.model_manager)
        container.register('seo_processor', self.seo_processor)
        container.register('monitoring', self.monitoring)
        
        # Start monitoring
        self.monitoring.start(collection_interval=2.0)
        
        logging.info("Optimized SEO Engine initialized successfully")
    
    def analyze_text(self, text: str, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """Analyze text for SEO optimization."""
        return self.seo_processor.analyze_text(text, analysis_type)
    
    def analyze_texts(self, texts: List[str], analysis_type: str = 'comprehensive') -> List[Dict[str, Any]]:
        """Analyze multiple texts in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.system.max_workers) as executor:
            futures = [
                executor.submit(self.analyze_text, text, analysis_type)
                for text in texts
            ]
            
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Text analysis failed: {e}")
                    results.append({'error': str(e)})
            
            return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        return {
            'system_health': self.monitoring.get_system_health(),
            'cache_stats': self.seo_processor.cache.get_stats(),
            'model_info': self.model_manager.get_model_info(),
            'performance_stats': self.seo_processor.get_performance_stats()
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Apply performance optimizations."""
        optimizations = {}
        
        # Memory optimization
        if psutil.virtual_memory().percent > 80:
            self.model_manager._unload_least_used_models()
            optimizations['memory_cleanup'] = 'Unloaded least used models'
        
        # Cache optimization
        cache_stats = self.seo_processor.cache.get_stats()
        if cache_stats['total_items'] > 800:
            # Force cleanup of old entries
            self.seo_processor.cache._cleanup_expired()
            optimizations['cache_cleanup'] = 'Cleaned up expired cache entries'
        
        return optimizations
    
    def export_analysis_report(self, filename: str, format: str = 'json') -> None:
        """Export comprehensive analysis report."""
        try:
            report = {
                'system_metrics': self.get_system_metrics(),
                'configuration': self.config.to_dict(),
                'timestamp': time.time()
            }
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            elif format.lower() == 'yaml':
                import yaml
                with open(filename, 'w') as f:
                    yaml.dump(report, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logging.error(f"Failed to export report: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.monitoring.stop()
            
            # Unload all models
            for model_name in list(self.model_manager.models.keys()):
                self.model_manager.unload_model(model_name)
            
            logging.info("SEO Engine cleanup completed")
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_optimized_seo_engine(config_path: Optional[str] = None) -> OptimizedSEOEngine:
    """Create and configure an optimized SEO engine."""
    if config_path:
        config = SEOConfig.load_from_file(config_path)
    else:
        config = SEOConfig()
    
    return OptimizedSEOEngine(config)

def quick_seo_analysis(text: str, config: Optional[SEOConfig] = None) -> Dict[str, Any]:
    """Quick SEO analysis with default configuration."""
    engine = OptimizedSEOEngine(config)
    try:
        result = engine.analyze_text(text)
        return result
    finally:
        engine.cleanup()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    engine = create_optimized_seo_engine()
    
    try:
        # Analyze sample text
        sample_text = """
        # SEO Optimization Guide
        
        This comprehensive guide covers the essential aspects of SEO optimization.
        We'll explore keyword research, content optimization, and technical SEO.
        
        ## Key Points
        - Keyword density optimization
        - Content quality improvement
        - Technical SEO best practices
        """
        
        result = engine.analyze_text(sample_text)
        print(f"SEO Score: {result['seo_score']:.1f}")
        print(f"Recommendations: {result['recommendations']}")
        
        # Get system metrics
        metrics = engine.get_system_metrics()
        print(f"System Health: {metrics['system_health']['status']}")
        
    finally:
        engine.cleanup()


