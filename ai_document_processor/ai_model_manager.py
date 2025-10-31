#!/usr/bin/env python3
"""
AI Model Manager - Advanced AI Document Processor
===============================================

Intelligent AI model management system for dynamic loading, caching, and optimization.
"""

import asyncio
import time
import logging
import json
import hashlib
import weakref
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    type: str  # 'llm', 'embedding', 'vision', 'audio', 'nlp'
    provider: str  # 'openai', 'anthropic', 'huggingface', 'local'
    model_id: str
    version: str
    size_mb: float
    memory_usage_mb: float
    load_time: float
    last_used: datetime
    usage_count: int = 0
    accuracy_score: Optional[float] = None
    speed_score: Optional[float] = None
    cost_per_token: Optional[float] = None
    max_tokens: Optional[int] = None
    supported_languages: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelRequest:
    """Request for model usage."""
    model_name: str
    task_type: str
    input_data: Any
    options: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    timeout: float = 30.0
    callback: Optional[Callable] = None

class ModelCache:
    """Intelligent model caching system."""
    
    def __init__(self, max_size_mb: float = 8192, max_models: int = 10):
        self.max_size_mb = max_size_mb
        self.max_models = max_models
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.access_times: Dict[str, datetime] = {}
        self.current_size_mb = 0.0
        self._lock = threading.RLock()
    
    def add_model(self, name: str, model: Any, info: ModelInfo):
        """Add a model to the cache."""
        with self._lock:
            # Remove existing model if present
            if name in self.models:
                self.remove_model(name)
            
            # Check if we need to evict models
            while (self.current_size_mb + info.memory_usage_mb > self.max_size_mb or 
                   len(self.models) >= self.max_models):
                self._evict_least_recently_used()
            
            # Add the model
            self.models[name] = model
            self.model_info[name] = info
            self.access_times[name] = datetime.utcnow()
            self.current_size_mb += info.memory_usage_mb
            
            logger.info(f"Model {name} added to cache (size: {info.memory_usage_mb:.1f}MB)")
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a model from the cache."""
        with self._lock:
            if name in self.models:
                self.access_times[name] = datetime.utcnow()
                self.model_info[name].usage_count += 1
                self.model_info[name].last_used = datetime.utcnow()
                return self.models[name]
            return None
    
    def remove_model(self, name: str):
        """Remove a model from the cache."""
        with self._lock:
            if name in self.models:
                info = self.model_info[name]
                del self.models[name]
                del self.model_info[name]
                del self.access_times[name]
                self.current_size_mb -= info.memory_usage_mb
                
                # Force garbage collection
                gc.collect()
                
                logger.info(f"Model {name} removed from cache")
    
    def _evict_least_recently_used(self):
        """Evict the least recently used model."""
        if not self.access_times:
            return
        
        lru_name = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.remove_model(lru_name)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_models": len(self.models),
                "current_size_mb": self.current_size_mb,
                "max_size_mb": self.max_size_mb,
                "max_models": self.max_models,
                "models": list(self.models.keys()),
                "utilization": (self.current_size_mb / self.max_size_mb) * 100
            }

class AIModelManager:
    """Advanced AI model manager with intelligent caching and optimization."""
    
    def __init__(self):
        self.model_cache = ModelCache()
        self.model_registry: Dict[str, Type] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.loading_models: Dict[str, asyncio.Task] = {}
        self.model_metrics: Dict[str, List[float]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Model providers
        self.providers = {
            'openai': self._load_openai_model,
            'anthropic': self._load_anthropic_model,
            'huggingface': self._load_huggingface_model,
            'local': self._load_local_model
        }
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default model configurations."""
        default_models = {
            'gpt-4-turbo': {
                'type': 'llm',
                'provider': 'openai',
                'model_id': 'gpt-4-turbo',
                'version': '1.0',
                'size_mb': 0,  # API model
                'max_tokens': 128000,
                'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                'capabilities': ['text_generation', 'conversation', 'summarization', 'translation', 'qa']
            },
            'claude-3-opus': {
                'type': 'llm',
                'provider': 'anthropic',
                'model_id': 'claude-3-opus-20240229',
                'version': '1.0',
                'size_mb': 0,  # API model
                'max_tokens': 200000,
                'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                'capabilities': ['text_generation', 'conversation', 'summarization', 'translation', 'qa', 'analysis']
            },
            'text-embedding-3-large': {
                'type': 'embedding',
                'provider': 'openai',
                'model_id': 'text-embedding-3-large',
                'version': '1.0',
                'size_mb': 0,  # API model
                'max_tokens': 8191,
                'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                'capabilities': ['text_embedding', 'similarity_search', 'clustering']
            },
            'all-MiniLM-L6-v2': {
                'type': 'embedding',
                'provider': 'huggingface',
                'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
                'version': '1.0',
                'size_mb': 90,
                'max_tokens': 512,
                'supported_languages': ['en'],
                'capabilities': ['text_embedding', 'similarity_search', 'clustering']
            },
            'whisper-base': {
                'type': 'audio',
                'provider': 'openai',
                'model_id': 'whisper-1',
                'version': '1.0',
                'size_mb': 0,  # API model
                'max_tokens': 0,
                'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
                'capabilities': ['speech_to_text', 'transcription', 'translation']
            }
        }
        
        for name, config in default_models.items():
            self.model_configs[name] = config
    
    async def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """Load a model with intelligent caching."""
        with self._lock:
            # Check if model is already loaded
            if not force_reload and model_name in self.model_cache.models:
                logger.info(f"Model {model_name} already loaded, returning cached version")
                return self.model_cache.get_model(model_name)
            
            # Check if model is currently loading
            if model_name in self.loading_models:
                logger.info(f"Model {model_name} is already loading, waiting...")
                return await self.loading_models[model_name]
            
            # Start loading the model
            loading_task = asyncio.create_task(self._load_model_async(model_name))
            self.loading_models[model_name] = loading_task
            
            try:
                model = await loading_task
                return model
            finally:
                # Remove from loading models
                if model_name in self.loading_models:
                    del self.loading_models[model_name]
    
    async def _load_model_async(self, model_name: str) -> Any:
        """Asynchronously load a model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        provider = config['provider']
        
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"Loading model {model_name} from provider {provider}")
        start_time = time.time()
        
        try:
            # Load the model
            model = await self.providers[provider](config)
            
            # Calculate load time
            load_time = time.time() - start_time
            
            # Create model info
            model_info = ModelInfo(
                name=model_name,
                type=config['type'],
                provider=provider,
                model_id=config['model_id'],
                version=config['version'],
                size_mb=config['size_mb'],
                memory_usage_mb=self._estimate_memory_usage(model, config),
                load_time=load_time,
                last_used=datetime.utcnow(),
                max_tokens=config.get('max_tokens'),
                supported_languages=config.get('supported_languages', []),
                capabilities=config.get('capabilities', []),
                metadata=config.get('metadata', {})
            )
            
            # Add to cache
            self.model_cache.add_model(model_name, model, model_info)
            
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def _load_openai_model(self, config: Dict[str, Any]) -> Any:
        """Load an OpenAI model."""
        try:
            import openai
            # For API models, we return a client instance
            return openai
        except ImportError:
            raise ImportError("OpenAI library not installed")
    
    async def _load_anthropic_model(self, config: Dict[str, Any]) -> Any:
        """Load an Anthropic model."""
        try:
            import anthropic
            # For API models, we return a client instance
            return anthropic
        except ImportError:
            raise ImportError("Anthropic library not installed")
    
    async def _load_huggingface_model(self, config: Dict[str, Any]) -> Any:
        """Load a Hugging Face model."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModel
            
            model_id = config['model_id']
            model_type = config['type']
            
            if model_type == 'embedding':
                # Load sentence transformer
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_id)
            elif model_type == 'llm':
                # Load text generation pipeline
                model = pipeline("text-generation", model=model_id)
            elif model_type == 'nlp':
                # Load NLP pipeline
                model = pipeline("sentiment-analysis", model=model_id)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            return model
            
        except ImportError:
            raise ImportError("Transformers library not installed")
    
    async def _load_local_model(self, config: Dict[str, Any]) -> Any:
        """Load a local model."""
        # This would load models from local files
        # Implementation depends on the specific model format
        raise NotImplementedError("Local model loading not implemented yet")
    
    def _estimate_memory_usage(self, model: Any, config: Dict[str, Any]) -> float:
        """Estimate memory usage of a model."""
        if config['provider'] in ['openai', 'anthropic']:
            # API models don't use local memory
            return 0.0
        
        # For local models, estimate based on size
        size_mb = config.get('size_mb', 0)
        if size_mb > 0:
            # Add overhead for model loading
            return size_mb * 1.5
        
        # Default estimation
        return 100.0
    
    async def process_request(self, request: ModelRequest) -> Any:
        """Process a model request."""
        model_name = request.model_name
        
        # Load model if not cached
        model = await self.load_model(model_name)
        
        # Get model info
        model_info = self.model_cache.model_info.get(model_name)
        if not model_info:
            raise ValueError(f"Model info not found for {model_name}")
        
        # Process the request
        start_time = time.time()
        try:
            result = await self._process_with_model(model, model_info, request)
            
            # Record metrics
            processing_time = time.time() - start_time
            self._record_metrics(model_name, processing_time, True)
            
            return result
            
        except Exception as e:
            # Record error metrics
            processing_time = time.time() - start_time
            self._record_metrics(model_name, processing_time, False)
            raise
    
    async def _process_with_model(self, model: Any, model_info: ModelInfo, request: ModelRequest) -> Any:
        """Process request with specific model."""
        task_type = request.task_type
        input_data = request.input_data
        options = request.options
        
        if model_info.provider == 'openai':
            return await self._process_openai_request(model, model_info, task_type, input_data, options)
        elif model_info.provider == 'anthropic':
            return await self._process_anthropic_request(model, model_info, task_type, input_data, options)
        elif model_info.provider == 'huggingface':
            return await self._process_huggingface_request(model, model_info, task_type, input_data, options)
        else:
            raise ValueError(f"Unsupported provider: {model_info.provider}")
    
    async def _process_openai_request(self, model: Any, model_info: ModelInfo, task_type: str, input_data: Any, options: Dict[str, Any]) -> Any:
        """Process OpenAI request."""
        if task_type == 'text_generation':
            response = await model.ChatCompletion.acreate(
                model=model_info.model_id,
                messages=[{"role": "user", "content": str(input_data)}],
                max_tokens=options.get('max_tokens', 1000),
                temperature=options.get('temperature', 0.7)
            )
            return response.choices[0].message.content
        
        elif task_type == 'embedding':
            response = await model.Embedding.acreate(
                model=model_info.model_id,
                input=str(input_data)
            )
            return response.data[0].embedding
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _process_anthropic_request(self, model: Any, model_info: ModelInfo, task_type: str, input_data: Any, options: Dict[str, Any]) -> Any:
        """Process Anthropic request."""
        if task_type == 'text_generation':
            response = await model.messages.create(
                model=model_info.model_id,
                max_tokens=options.get('max_tokens', 1000),
                messages=[{"role": "user", "content": str(input_data)}]
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _process_huggingface_request(self, model: Any, model_info: ModelInfo, task_type: str, input_data: Any, options: Dict[str, Any]) -> Any:
        """Process Hugging Face request."""
        if task_type == 'embedding':
            # For sentence transformers
            if hasattr(model, 'encode'):
                return model.encode(str(input_data))
            else:
                raise ValueError("Model does not support embedding")
        
        elif task_type == 'text_generation':
            # For text generation pipeline
            if hasattr(model, '__call__'):
                result = model(str(input_data), max_length=options.get('max_tokens', 100))
                return result[0]['generated_text']
            else:
                raise ValueError("Model does not support text generation")
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _record_metrics(self, model_name: str, processing_time: float, success: bool):
        """Record model performance metrics."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = []
        
        self.model_metrics[model_name].append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'success': success
        })
        
        # Keep only last 100 metrics per model
        if len(self.model_metrics[model_name]) > 100:
            self.model_metrics[model_name] = self.model_metrics[model_name][-100:]
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        cache_stats = self.model_cache.get_cache_stats()
        
        # Calculate performance metrics
        performance_stats = {}
        for model_name, metrics in self.model_metrics.items():
            if metrics:
                avg_time = sum(m['processing_time'] for m in metrics) / len(metrics)
                success_rate = sum(1 for m in metrics if m['success']) / len(metrics)
                performance_stats[model_name] = {
                    'avg_processing_time': avg_time,
                    'success_rate': success_rate,
                    'total_requests': len(metrics)
                }
        
        return {
            'cache': cache_stats,
            'performance': performance_stats,
            'loaded_models': list(self.model_cache.models.keys()),
            'available_models': list(self.model_configs.keys())
        }
    
    def display_model_dashboard(self):
        """Display model management dashboard."""
        stats = self.get_model_stats()
        
        # Cache statistics table
        cache_table = Table(title="Model Cache Statistics")
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", style="green")
        
        cache = stats['cache']
        cache_table.add_row("Total Models", str(cache['total_models']))
        cache_table.add_row("Current Size", f"{cache['current_size_mb']:.1f} MB")
        cache_table.add_row("Max Size", f"{cache['max_size_mb']:.1f} MB")
        cache_table.add_row("Utilization", f"{cache['utilization']:.1f}%")
        cache_table.add_row("Loaded Models", ", ".join(cache['models']))
        
        console.print(cache_table)
        
        # Performance statistics table
        if stats['performance']:
            perf_table = Table(title="Model Performance Statistics")
            perf_table.add_column("Model", style="cyan")
            perf_table.add_column("Avg Time (s)", style="green")
            perf_table.add_column("Success Rate", style="yellow")
            perf_table.add_column("Total Requests", style="magenta")
            
            for model_name, perf in stats['performance'].items():
                perf_table.add_row(
                    model_name,
                    f"{perf['avg_processing_time']:.3f}",
                    f"{perf['success_rate']:.1%}",
                    str(perf['total_requests'])
                )
            
            console.print(perf_table)
    
    def cleanup(self):
        """Cleanup resources."""
        self.model_cache.models.clear()
        self.model_cache.model_info.clear()
        self.model_cache.access_times.clear()
        self.model_cache.current_size_mb = 0.0
        
        # Cancel any loading tasks
        for task in self.loading_models.values():
            task.cancel()
        self.loading_models.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Force garbage collection
        gc.collect()
        
        logger.info("AI Model Manager cleanup completed")

# Global model manager instance
model_manager = AIModelManager()

# Utility functions
async def load_model(model_name: str, force_reload: bool = False) -> Any:
    """Load a model using the global model manager."""
    return await model_manager.load_model(model_name, force_reload)

async def process_model_request(request: ModelRequest) -> Any:
    """Process a model request using the global model manager."""
    return await model_manager.process_request(request)

def get_model_stats() -> Dict[str, Any]:
    """Get model statistics from the global model manager."""
    return model_manager.get_model_stats()

def display_model_dashboard():
    """Display model dashboard from the global model manager."""
    model_manager.display_model_dashboard()

def cleanup_model_manager():
    """Cleanup the global model manager."""
    model_manager.cleanup()

if __name__ == "__main__":
    # Example usage
    async def main():
        # Load a model
        model = await load_model('gpt-4-turbo')
        print(f"Model loaded: {type(model)}")
        
        # Process a request
        request = ModelRequest(
            model_name='gpt-4-turbo',
            task_type='text_generation',
            input_data='Hello, how are you?',
            options={'max_tokens': 100}
        )
        
        result = await process_model_request(request)
        print(f"Result: {result}")
        
        # Display dashboard
        display_model_dashboard()
        
        # Cleanup
        cleanup_model_manager()
    
    asyncio.run(main())
















