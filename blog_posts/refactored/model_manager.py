from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache
import weakref
from .config import NLPConfig, ModelType
    from transformers import (
    import torch
    from torch.quantization import quantize_dynamic
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    import spacy
from typing import Any, List, Dict, Optional
"""
Gestor de modelos ML ultra-optimizado y refactorizado.
"""



# Importaciones condicionales organizadas
try:
        AutoTokenizer, AutoModel, pipeline,
        DistilBertTokenizer, DistilBertModel
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Información de un modelo cargado."""
    name: str
    type: str
    model: Any
    tokenizer: Optional[Any] = None
    load_time_ms: float = 0.0
    memory_usage_mb: Optional[float] = None
    is_quantized: bool = False
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0

class ModelRegistry:
    """Registro de configuraciones de modelos disponibles."""
    
    LIGHTWEIGHT_MODELS = {
        'sentiment': {
            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'task': 'sentiment-analysis',
            'max_length': 512,
            'memory_mb': 250
        },
        'embedding': {
            'model_name': 'all-MiniLM-L6-v2',
            'task': 'sentence-similarity',
            'max_length': 256,
            'memory_mb': 80
        },
        'spacy': {
            'model_name': 'en_core_web_sm',
            'task': 'nlp',
            'max_length': 1000000,
            'memory_mb': 50
        }
    }
    
    STANDARD_MODELS = {
        'sentiment': {
            'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'task': 'sentiment-analysis',
            'max_length': 512,
            'memory_mb': 500
        },
        'embedding': {
            'model_name': 'all-MiniLM-L12-v2',
            'task': 'sentence-similarity',
            'max_length': 512,
            'memory_mb': 120
        },
        'spacy': {
            'model_name': 'en_core_web_md',
            'task': 'nlp',
            'max_length': 1000000,
            'memory_mb': 150
        }
    }
    
    ADVANCED_MODELS = {
        'sentiment': {
            'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
            'task': 'sentiment-analysis',
            'max_length': 512,
            'memory_mb': 800
        },
        'embedding': {
            'model_name': 'all-mpnet-base-v2',
            'task': 'sentence-similarity',
            'max_length': 512,
            'memory_mb': 400
        },
        'spacy': {
            'model_name': 'en_core_web_lg',
            'task': 'nlp',
            'max_length': 1000000,
            'memory_mb': 750
        }
    }
    
    @classmethod
    def get_model_config(cls, model_type: str, tier: ModelType) -> Optional[Dict[str, Any]]:
        """Obtener configuración de modelo según tipo y tier."""
        tier_mapping = {
            ModelType.LIGHTWEIGHT: cls.LIGHTWEIGHT_MODELS,
            ModelType.STANDARD: cls.STANDARD_MODELS,
            ModelType.ADVANCED: cls.ADVANCED_MODELS
        }
        
        models = tier_mapping.get(tier, cls.LIGHTWEIGHT_MODELS)
        return models.get(model_type)

class ModelLoader:
    """Cargador especializado de modelos."""
    
    def __init__(self, config: NLPConfig, executor: ThreadPoolExecutor):
        
    """__init__ function."""
self.config = config
        self.executor = executor
        self.logger = logging.getLogger(f"{__name__}.ModelLoader")
    
    async def load_transformers_model(self, model_config: Dict[str, Any]) -> Optional[ModelInfo]:
        """Cargar modelo de transformers."""
        if not TORCH_AVAILABLE:
            return None
        
        model_name = model_config['model_name']
        task = model_config['task']
        
        try:
            start_time = time.time()
            
            # Cargar en executor para no bloquear
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                self._load_transformers_sync,
                model_name,
                task
            )
            
            load_time_ms = (time.time() - start_time) * 1000
            
            model_info = ModelInfo(
                name=model_name,
                type='transformers',
                model=model,
                load_time_ms=load_time_ms
            )
            
            # Quantizar si está habilitado
            if self.config.models.quantization_enabled:
                model_info = await self._quantize_model(model_info)
            
            self.logger.info(f"Transformers model {model_name} loaded in {load_time_ms:.2f}ms")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load transformers model {model_name}: {e}")
            return None
    
    def _load_transformers_sync(self, model_name: str, task: str):
        """Cargar modelo transformers síncronamente."""
        if task == 'sentiment-analysis':
            return pipeline(
                task,
                model=model_name,
                return_all_scores=True,
                device=-1  # CPU para compatibilidad
            )
        else:
            return pipeline(task, model=model_name, device=-1)
    
    async def load_sentence_transformers_model(self, model_config: Dict[str, Any]) -> Optional[ModelInfo]:
        """Cargar modelo de sentence transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        model_name = model_config['model_name']
        
        try:
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                SentenceTransformer,
                model_name
            )
            
            load_time_ms = (time.time() - start_time) * 1000
            
            model_info = ModelInfo(
                name=model_name,
                type='sentence_transformers',
                model=model,
                load_time_ms=load_time_ms
            )
            
            self.logger.info(f"SentenceTransformers model {model_name} loaded in {load_time_ms:.2f}ms")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformers model {model_name}: {e}")
            return None
    
    async def load_spacy_model(self, model_config: Dict[str, Any]) -> Optional[ModelInfo]:
        """Cargar modelo de spaCy."""
        if not SPACY_AVAILABLE:
            return None
        
        model_name = model_config['model_name']
        
        try:
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                spacy.load,
                model_name
            )
            
            load_time_ms = (time.time() - start_time) * 1000
            
            model_info = ModelInfo(
                name=model_name,
                type='spacy',
                model=model,
                load_time_ms=load_time_ms
            )
            
            self.logger.info(f"spaCy model {model_name} loaded in {load_time_ms:.2f}ms")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model {model_name}: {e}")
            return None
    
    async def _quantize_model(self, model_info: ModelInfo) -> ModelInfo:
        """Quantizar modelo para reducir memoria."""
        if not TORCH_AVAILABLE or model_info.type != 'transformers':
            return model_info
        
        try:
            loop = asyncio.get_event_loop()
            quantized_model = await loop.run_in_executor(
                self.executor,
                self._quantize_sync,
                model_info.model
            )
            
            model_info.model = quantized_model
            model_info.is_quantized = True
            self.logger.info(f"Model {model_info.name} quantized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to quantize model {model_info.name}: {e}")
        
        return model_info
    
    def _quantize_sync(self, model) -> Any:
        """Quantizar modelo síncronamente."""
        if hasattr(model, 'model'):
            quantized = quantize_dynamic(
                model.model,
                {nn.Linear},
                dtype=torch.qint8
            )
            model.model = quantized
        return model

class ModelManager:
    """Gestor principal de modelos ML refactorizado."""
    
    def __init__(self, config: NLPConfig):
        
    """__init__ function."""
self.config = config
        self.models: Dict[str, ModelInfo] = {}
        self.loading_locks: Dict[str, asyncio.Lock] = {}
        self.model_cache = LRUCache(maxsize=config.models.cache_size)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelLoader")
        self.loader = ModelLoader(config, self.executor)
        self.registry = ModelRegistry()
        self.stats = {
            'models_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_load_time_ms': 0.0
        }
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
    
    async def initialize(self) -> Any:
        """Inicializar gestor de modelos."""
        if self._initialized:
            return
        
        self.logger.info("Initializing ModelManager...")
        
        # Pre-cargar modelos críticos si está habilitado
        if self.config.models.preload_critical:
            await self._preload_critical_models()
        
        self._initialized = True
        self.logger.info("ModelManager initialized successfully")
    
    async def get_model(self, model_type: str, required: bool = True) -> Optional[ModelInfo]:
        """
        Obtener modelo con lazy loading y cache.
        
        Args:
            model_type: Tipo de modelo ('sentiment', 'embedding', 'spacy')
            required: Si es True, intentará cargar el modelo si no existe
            
        Returns:
            ModelInfo o None si no está disponible
        """
        cache_key = f"{model_type}:{self.config.models.type.value}"
        
        # Cache hit
        if cache_key in self.model_cache:
            model_info = self.model_cache[cache_key]
            model_info.last_used = time.time()
            model_info.usage_count += 1
            self.stats['cache_hits'] += 1
            self.logger.debug(f"Cache hit for model: {cache_key}")
            return model_info
        
        # Cache miss
        self.stats['cache_misses'] += 1
        
        if not required:
            return None
        
        # Lock para evitar cargas duplicadas
        if cache_key not in self.loading_locks:
            self.loading_locks[cache_key] = asyncio.Lock()
        
        async with self.loading_locks[cache_key]:
            # Double-check después del lock
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
            
            # Cargar modelo
            model_info = await self._load_model(model_type)
            
            if model_info:
                self.model_cache[cache_key] = model_info
                self.stats['models_loaded'] += 1
                self.stats['total_load_time_ms'] += model_info.load_time_ms
                self.logger.info(f"Model {cache_key} loaded and cached")
            
            return model_info
    
    async def _load_model(self, model_type: str) -> Optional[ModelInfo]:
        """Cargar modelo específico."""
        # Obtener configuración del modelo
        model_config = self.registry.get_model_config(model_type, self.config.models.type)
        
        if not model_config:
            self.logger.warning(f"No model config found for {model_type}")
            return None
        
        # Intentar cargar según el tipo
        if model_type == 'sentiment' and TORCH_AVAILABLE:
            return await self.loader.load_transformers_model(model_config)
        elif model_type == 'embedding' and SENTENCE_TRANSFORMERS_AVAILABLE:
            return await self.loader.load_sentence_transformers_model(model_config)
        elif model_type == 'spacy' and SPACY_AVAILABLE:
            return await self.loader.load_spacy_model(model_config)
        else:
            self.logger.warning(f"Model type {model_type} not available or supported")
            return None
    
    async def _preload_critical_models(self) -> Any:
        """Pre-cargar modelos críticos."""
        critical_models = ['sentiment'] # Solo los más usados
        
        if self.config.models.type != ModelType.LIGHTWEIGHT:
            critical_models.extend(['embedding'])
        
        tasks = []
        for model_type in critical_models:
            task = asyncio.create_task(self.get_model(model_type, required=False))
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            loaded_count = sum(1 for r in results if isinstance(r, ModelInfo))
            self.logger.info(f"Pre-loaded {loaded_count}/{len(critical_models)} critical models")
        except Exception as e:
            self.logger.warning(f"Error pre-loading models: {e}")
    
    def is_model_available(self, model_type: str) -> bool:
        """Verificar si un tipo de modelo está disponible."""
        model_config = self.registry.get_model_config(model_type, self.config.models.type)
        if not model_config:
            return False
        
        if model_type == 'sentiment':
            return TORCH_AVAILABLE
        elif model_type == 'embedding':
            return SENTENCE_TRANSFORMERS_AVAILABLE
        elif model_type == 'spacy':
            return SPACY_AVAILABLE
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del gestor."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'models_loaded': self.stats['models_loaded'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': hit_rate,
            'total_load_time_ms': self.stats['total_load_time_ms'],
            'avg_load_time_ms': (
                self.stats['total_load_time_ms'] / self.stats['models_loaded']
                if self.stats['models_loaded'] > 0 else 0
            ),
            'cached_models': len(self.model_cache),
            'model_tier': self.config.models.type.value
        }
    
    async def cleanup(self) -> Any:
        """Limpiar recursos."""
        self.logger.info("Cleaning up ModelManager...")
        
        # Cerrar executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Limpiar cache
        self.model_cache.clear()
        self.models.clear()
        
        self.logger.info("ModelManager cleanup completed") 