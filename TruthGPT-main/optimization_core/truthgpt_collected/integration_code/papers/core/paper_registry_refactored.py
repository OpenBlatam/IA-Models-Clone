#!/usr/bin/env python3
"""
Paper Registry Refactored - Versión Refactorizada y Optimizada
===============================================================

Refactorización completa:
- Usa MetadataExtractor unificado
- Elimina duplicación
- Mejor organización
- Thread-safe
- Cache persistente
"""

import importlib.util
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, asdict
from collections import OrderedDict
import logging
import time
import hashlib
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

try:
    from .metadata_extractor import MetadataExtractor, PaperMetadata
except (ImportError, ValueError):
    from metadata_extractor import MetadataExtractor, PaperMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperModule:
    """Módulo cargado de un paper."""
    metadata: PaperMetadata
    config_class: Type
    module_class: Type
    module: Any = None
    loaded: bool = False
    load_time: float = 0.0
    error: Optional[str] = None
    cache_key: Optional[str] = None


class PaperRegistryRefactored:
    """
    Registry refactorizado y optimizado.
    
    Mejoras:
    - Usa MetadataExtractor unificado
    - Thread-safe
    - Cache persistente
    - Pre-carga inteligente
    - Búsqueda avanzada
    """
    
    # Categorías de papers
    CATEGORY_DIRS = {
        'research': 'research',
        'architecture': 'architecture',
        'inference': 'inference',
        'memory': 'memory',
        'redundancy': 'redundancy',
        'techniques': 'techniques',
        'code': 'code',
        'best': 'best'
    }
    
    def __init__(
        self,
        papers_base_dir: Optional[Path] = None,
        enable_disk_cache: bool = True,
        preload_popular: bool = True,
        max_cache_size: int = 100
    ):
        if papers_base_dir is None:
            papers_base_dir = Path(__file__).parent.parent
        
        self.papers_base_dir = Path(papers_base_dir)
        self.registry: Dict[str, PaperMetadata] = {}
        self.loaded_modules: Dict[str, PaperModule] = OrderedDict()
        self.enable_disk_cache = enable_disk_cache
        self.preload_popular = preload_popular
        self.max_cache_size = max_cache_size
        self.cache_dir = self.papers_base_dir / ".paper_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        
        # Estadísticas
        self.stats = {
            'total_papers': 0,
            'loaded_papers': 0,
            'failed_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_load_time': 0.0
        }
        
        # Cargar metadata cache
        self._load_metadata_cache()
        
        # Descubrir papers
        self._discover_papers()
        
        # Pre-cargar papers populares
        if self.preload_popular:
            self._preload_popular_papers()
    
    def _load_metadata_cache(self):
        """Carga metadata desde cache."""
        metadata_file = self.cache_dir / "metadata_cache.json"
        self._file_mtimes = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    cached_data = json.load(f)
                    for paper_id, data in cached_data.items():
                        # Support for old format (only stats)
                        if 'module_path' not in data:
                            if paper_id in self.registry:
                                self.registry[paper_id].usage_count = data.get('usage_count', 0)
                                self.registry[paper_id].load_count = data.get('load_count', 0)
                                self.registry[paper_id].error_count = data.get('error_count', 0)
                                self.registry[paper_id].last_used = data.get('last_used')
                            continue
                            
                        # Full metadata load
                        if '_mtime' in data:
                            self._file_mtimes[paper_id] = data.pop('_mtime')
                        
                        data['module_path'] = Path(data['module_path'])
                        
                        # Handle potential new/missing dataclass fields gracefully
                        try:
                            self.registry[paper_id] = PaperMetadata(**data)
                        except TypeError:
                            pass # Let _discover_papers rebuild it if fields mismatch
                            
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")
                self._file_mtimes = {}
    
    def _save_metadata_cache(self):
        """Guarda metadata en cache."""
        metadata_file = self.cache_dir / "metadata_cache.json"
        try:
            data = {}
            for paper_id, meta in self.registry.items():
                meta_dict = asdict(meta)
                meta_dict['module_path'] = str(meta_dict['module_path'])
                try:
                    meta_dict['_mtime'] = meta.module_path.stat().st_mtime
                except Exception:
                    meta_dict['_mtime'] = 0
                data[paper_id] = meta_dict
                
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")
    
    def _discover_papers(self):
        """Descubre papers usando MetadataExtractor de forma optimizada con cache."""
        logger.info(f"🔍 Discovering papers in {self.papers_base_dir}")
        
        changed = False
        current_ids = set()
        
        for category, dir_name in self.CATEGORY_DIRS.items():
            category_path = self.papers_base_dir / dir_name
            if not category_path.exists():
                continue
            
            for paper_file in category_path.glob("paper_*.py"):
                try:
                    paper_id = MetadataExtractor._extract_paper_id(paper_file)
                    current_ids.add(paper_id)
                    mtime = paper_file.stat().st_mtime
                    
                    if paper_id in self.registry and getattr(self, '_file_mtimes', {}).get(paper_id) == mtime:
                        continue
                        
                    metadata = MetadataExtractor.extract_from_file(paper_file, category)
                    if metadata:
                        if paper_id in self.registry:
                            old_meta = self.registry[paper_id]
                            metadata.usage_count = old_meta.usage_count
                            metadata.load_count = old_meta.load_count
                            metadata.error_count = old_meta.error_count
                            metadata.last_used = old_meta.last_used
                        
                        self.registry[metadata.paper_id] = metadata
                        if not hasattr(self, '_file_mtimes'):
                            self._file_mtimes = {}
                        self._file_mtimes[metadata.paper_id] = mtime
                        changed = True
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to register {paper_file.name}: {e}")
        
        to_remove = [pid for pid in self.registry if pid not in current_ids]
        for pid in to_remove:
            del self.registry[pid]
            if getattr(self, '_file_mtimes', {}).get(pid):
                del self._file_mtimes[pid]
            changed = True
            
        self.stats['total_papers'] = len(self.registry)
        
        if changed or not (self.cache_dir / "metadata_cache.json").exists():
            self._save_metadata_cache()
            
        logger.info(f"✅ Discovered {len(self.registry)} papers")
    
    def _preload_popular_papers(self, top_n: int = 5):
        """Pre-carga papers más usados."""
        if len(self.registry) == 0:
            return
        
        start_time = time.time()
        logger.info(f"📦 Pre-loading top {top_n} popular papers...")
        
        sorted_papers = sorted(
            self.registry.items(),
            key=lambda x: (x[1].usage_count, x[1].load_count),
            reverse=True
        )[:top_n]
        
        with ThreadPoolExecutor(max_workers=min(top_n, 4)) as executor:
            futures = {
                executor.submit(self._load_paper_internal, paper_id): paper_id
                for paper_id, _ in sorted_papers
            }
            
            for future in futures:
                try:
                    future.result()
                except Exception:
                    pass
        
        logger.info(f"✅ Pre-loaded {len(sorted_papers)} papers in {time.time() - start_time:.3f}s")
    
    def _load_paper_internal(self, paper_id: str, force_reload: bool = False) -> Optional[PaperModule]:
        """Carga interna de paper (thread-safe)."""
        with self._lock:
            if paper_id not in self.registry:
                return None
            
            # Verificar cache
            if not force_reload and paper_id in self.loaded_modules:
                self.loaded_modules.move_to_end(paper_id)  # LRU
                self.stats['cache_hits'] += 1
                return self.loaded_modules[paper_id]
            
            self.stats['cache_misses'] += 1
            metadata = self.registry[paper_id]
            start_time = time.time()
            
            try:
                # Cargar módulo
                spec = importlib.util.spec_from_file_location(
                    metadata.module_name,
                    metadata.module_path
                )
                module = importlib.util.module_from_spec(spec)
                
                module_dir = str(metadata.module_path.parent)
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                
                spec.loader.exec_module(module)
                
                # Obtener clases
                config_class = getattr(module, metadata.config_class, None) if metadata.config_class else None
                module_class = getattr(module, metadata.module_class, None) if metadata.module_class else None
                
                # Auto-descubrir si no se encontraron
                if not config_class or not module_class:
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            if 'Config' in attr_name and not config_class:
                                config_class = attr
                            elif 'Module' in attr_name and not module_class:
                                module_class = attr
                
                if not config_class or not module_class:
                    raise ValueError(f"Could not find Config or Module class")
                
                load_time = time.time() - start_time
                
                paper_module = PaperModule(
                    metadata=metadata,
                    config_class=config_class,
                    module_class=module_class,
                    module=module,
                    loaded=True,
                    load_time=load_time,
                    cache_key=self._generate_cache_key(paper_id)
                )
                
                # Guardar en cache
                self.loaded_modules[paper_id] = paper_module
                self._evict_if_needed()
                
                # Actualizar estadísticas
                metadata.load_count += 1
                metadata.last_used = time.time()
                self.stats['loaded_papers'] += 1
                self.stats['total_load_time'] += load_time
                
                return paper_module
                
            except Exception as e:
                load_time = time.time() - start_time
                metadata.error_count += 1
                
                paper_module = PaperModule(
                    metadata=metadata,
                    config_class=None,
                    module_class=None,
                    loaded=False,
                    load_time=load_time,
                    error=str(e)
                )
                
                self.loaded_modules[paper_id] = paper_module
                self.stats['failed_loads'] += 1
                
                return paper_module
    
    def _evict_if_needed(self):
        """Evicta módulos del cache (LRU)."""
        while len(self.loaded_modules) > self.max_cache_size:
            oldest_key = next(iter(self.loaded_modules))
            del self.loaded_modules[oldest_key]
    
    def _generate_cache_key(self, paper_id: str) -> str:
        """Genera clave de cache."""
        metadata = self.registry[paper_id]
        file_hash = hashlib.md5(metadata.module_path.read_bytes()).hexdigest()
        return f"{paper_id}_{file_hash[:8]}"
    
    @lru_cache(maxsize=128)
    def load_paper(self, paper_id: str, force_reload: bool = False) -> Optional[PaperModule]:
        """Carga un paper con retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._load_paper_internal(paper_id, force_reload)
                if result and result.loaded:
                    result.metadata.usage_count += 1
                    result.metadata.last_used = time.time()
                    return result
                elif attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    logger.error(f"Failed to load {paper_id} after {max_retries} attempts: {e}")
        
        return None
    
    def search_papers(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_speedup: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        max_memory_impact: Optional[str] = None
    ) -> List[PaperMetadata]:
        """Búsqueda avanzada de papers."""
        results = list(self.registry.values())
        
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.paper_id.lower() or
                   query_lower in p.paper_name.lower() or
                   any(query_lower in tech.lower() for tech in p.key_techniques)
            ]
        
        if category:
            results = [p for p in results if p.category == category]
        
        if min_speedup:
            results = [p for p in results if p.speedup and p.speedup >= min_speedup]
        
        if min_accuracy:
            results = [p for p in results if p.accuracy_improvement and p.accuracy_improvement >= min_accuracy]
        
        if max_memory_impact:
            impact_order = {'low': 1, 'medium': 2, 'high': 3}
            max_order = impact_order.get(max_memory_impact, 3)
            results = [p for p in results if impact_order.get(p.memory_impact, 2) <= max_order]
        
        return sorted(results, key=lambda x: (x.usage_count, x.load_count), reverse=True)
    
    def list_papers(self, category: Optional[str] = None) -> List[PaperMetadata]:
        """Lista papers."""
        papers = list(self.registry.values())
        if category:
            papers = [p for p in papers if p.category == category]
        return sorted(papers, key=lambda x: x.paper_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        return {
            **self.stats,
            'avg_load_time': (
                self.stats['total_load_time'] / self.stats['loaded_papers']
                if self.stats['loaded_papers'] > 0 else 0.0
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0.0
            ),
            'cache_size': len(self.loaded_modules),
            'most_used_papers': [
                {'paper_id': p.paper_id, 'usage_count': p.usage_count}
                for p in sorted(self.registry.values(), key=lambda x: x.usage_count, reverse=True)[:5]
            ]
        }
    
    def clear_cache(self):
        """Limpia el cache."""
        with self._lock:
            self.loaded_modules.clear()
            self.load_paper.cache_clear()
            logger.info("✅ Cache cleared")
    
    def save_state(self):
        """Guarda estado."""
        self._save_metadata_cache()


# Singleton
_global_registry: Optional[PaperRegistryRefactored] = None
_registry_lock = threading.Lock()


def get_registry(papers_base_dir: Optional[Path] = None, **kwargs) -> PaperRegistryRefactored:
    """Obtiene el registry global (thread-safe singleton)."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = PaperRegistryRefactored(papers_base_dir, **kwargs)
    return _global_registry


