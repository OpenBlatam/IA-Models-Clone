"""
Routing Performance Optimizations
==================================

Optimizaciones avanzadas de rendimiento: JIT compilation, inference optimization,
parallel processing, y caching agresivo.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
import time
import functools
from collections import OrderedDict
import hashlib

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.jit import script, trace
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Performance optimizations will be limited.")


class JITModelOptimizer:
    """Optimizador de modelos usando JIT compilation."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.compiled_models: Dict[str, Any] = {}
    
    def compile_model(
        self,
        model: nn.Module,
        model_name: str,
        example_input: Optional[torch.Tensor] = None,
        optimize: bool = True
    ) -> nn.Module:
        """
        Compilar modelo con JIT para mejor rendimiento.
        
        Args:
            model: Modelo PyTorch
            model_name: Nombre del modelo
            example_input: Input de ejemplo para tracing
            optimize: Habilitar optimizaciones
        
        Returns:
            Modelo compilado
        """
        if not TORCH_AVAILABLE:
            return model
        
        if model_name in self.compiled_models:
            return self.compiled_models[model_name]
        
        try:
            model.eval()
            model = model.to(self.device)
            
            if example_input is not None:
                example_input = example_input.to(self.device)
                # Tracing
                with torch.no_grad():
                    traced_model = trace(model, example_input)
                    if optimize:
                        traced_model = torch.jit.optimize_for_inference(traced_model)
                compiled = traced_model
            else:
                # Scripting
                compiled = script(model)
                if optimize:
                    compiled = torch.jit.optimize_for_inference(compiled)
            
            self.compiled_models[model_name] = compiled
            logger.info(f"Model {model_name} compiled with JIT")
            return compiled
        
        except Exception as e:
            logger.warning(f"JIT compilation failed for {model_name}: {e}, using original model")
            return model


class FastInferenceEngine:
    """Motor de inferencia optimizado."""
    
    def __init__(self, device: Optional[str] = None, batch_size: int = 64):
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.cache: OrderedDict = OrderedDict()
        self.cache_max_size = 1000
    
    def batch_predict(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Predicción en batch optimizada.
        
        Args:
            model: Modelo PyTorch
            inputs: Lista de tensores de entrada
            use_cache: Usar cache de predicciones
        
        Returns:
            Lista de predicciones
        """
        if not TORCH_AVAILABLE:
            return []
        
        model.eval()
        model = model.to(self.device)
        
        # Verificar cache (fuera de no_grad para operaciones de cache)
        if use_cache:
            cached_results = []
            uncached_inputs = []
            uncached_indices = []
            
            for i, inp in enumerate(inputs):
                cache_key = self._make_cache_key(inp)
                if cache_key in self.cache:
                    cached_results.append((i, self.cache[cache_key]))
                    # Mover al final (LRU)
                    self.cache.move_to_end(cache_key)
                else:
                    uncached_inputs.append(inp)
                    uncached_indices.append(i)
            
            if not uncached_inputs:
                # Todo estaba en cache
                results = [None] * len(inputs)
                for idx, result in cached_results:
                    results[idx] = result
                return results
        else:
            uncached_inputs = inputs
            uncached_indices = list(range(len(inputs)))
            cached_results = []
        
        # Procesar en batches con no_grad
        all_predictions = []
        with torch.no_grad():
            for i in range(0, len(uncached_inputs), self.batch_size):
                batch = uncached_inputs[i:i + self.batch_size]
                batch_tensor = torch.stack(batch).to(self.device)
                
                # Inferencia con autocast
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    predictions = model(batch_tensor)
                
                predictions_np = predictions.cpu().numpy()
                all_predictions.extend(predictions_np)
        
        # Guardar en cache (fuera de no_grad)
        if use_cache:
            for j, pred in enumerate(all_predictions):
                if j < len(uncached_inputs):
                    inp = uncached_inputs[j]
                    cache_key = self._make_cache_key(inp)
                    self._add_to_cache(cache_key, pred)
        
        # Combinar resultados
        results = [None] * len(inputs)
        for idx, result in cached_results:
            results[idx] = result
        
        pred_idx = 0
        for orig_idx in uncached_indices:
            if pred_idx < len(all_predictions):
                results[orig_idx] = all_predictions[pred_idx]
                pred_idx += 1
        
        return results
    
    def _make_cache_key(self, tensor: torch.Tensor) -> str:
        """Crear clave de cache desde tensor."""
        # Hash del contenido del tensor
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def _add_to_cache(self, key: str, value: np.ndarray):
        """Agregar a cache con LRU."""
        if len(self.cache) >= self.cache_max_size:
            self.cache.popitem(last=False)  # Eliminar el más antiguo
        self.cache[key] = value


class ParallelRouteProcessor:
    """Procesador paralelo de rutas."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers
        if self.num_workers is None:
            try:
                import multiprocessing as mp
                self.num_workers = min(mp.cpu_count(), 8)
            except:
                self.num_workers = 4
    
    def process_parallel(
        self,
        route_func: Callable,
        route_args: List[Tuple],
        chunk_size: int = 100
    ) -> List[Any]:
        """
        Procesar rutas en paralelo.
        
        Args:
            route_func: Función para calcular rutas
            route_args: Lista de argumentos para cada ruta
            chunk_size: Tamaño de chunk para procesamiento
        
        Returns:
            Lista de resultados
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        
        # Usar ThreadPoolExecutor para evitar problemas de serialización
        results = [None] * len(route_args)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit todas las tareas
            future_to_idx = {
                executor.submit(route_func, *args): idx
                for idx, args in enumerate(route_args)
            }
            
            # Recoger resultados
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Error processing route {idx}: {e}")
                    results[idx] = None
        
        return results


class VectorizedOperations:
    """Operaciones vectorizadas optimizadas."""
    
    @staticmethod
    def batch_distance_matrix(
        positions: np.ndarray,
        chunk_size: int = 1000
    ) -> np.ndarray:
        """
        Calcular matriz de distancias en batch optimizado.
        
        Args:
            positions: Array de posiciones [N, 3]
            chunk_size: Tamaño de chunk para memoria
        
        Returns:
            Matriz de distancias [N, N]
        """
        N = len(positions)
        distances = np.zeros((N, N), dtype=np.float32)
        
        # Procesar en chunks para ahorrar memoria
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_i = positions[i:end_i]
            
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                chunk_j = positions[j:end_j]
                
                # Calcular distancias entre chunks
                diff = chunk_i[:, np.newaxis, :] - chunk_j[np.newaxis, :, :]
                chunk_dist = np.linalg.norm(diff, axis=2)
                distances[i:end_i, j:end_j] = chunk_dist
        
        return distances
    
    @staticmethod
    def fast_path_search(
        graph: Dict[str, Dict[str, float]],
        start: str,
        end: str,
        max_depth: int = 100
    ) -> Optional[List[str]]:
        """
        Búsqueda de ruta rápida usando BFS optimizado.
        
        Args:
            graph: Grafo como diccionario
            start: Nodo inicio
            end: Nodo fin
            max_depth: Profundidad máxima
        
        Returns:
            Ruta o None
        """
        if start == end:
            return [start]
        
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue and len(queue[0][1]) < max_depth:
            current, path = queue.popleft()
            
            if current not in graph:
                continue
            
            for neighbor in graph[current]:
                if neighbor == end:
                    return path + [end]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None


class ModelQuantization:
    """Cuantización de modelos para inferencia más rápida."""
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        method: str = "int8"
    ) -> nn.Module:
        """
        Cuantizar modelo para inferencia más rápida.
        
        Args:
            model: Modelo PyTorch
            method: Método de cuantización ('int8', 'dynamic')
        
        Returns:
            Modelo cuantizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            if method == "int8":
                # Cuantización estática (requiere calibración)
                model.eval()
                quantized = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                return quantized
            elif method == "dynamic":
                # Cuantización dinámica
                model.eval()
                quantized = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
                )
                return quantized
            else:
                logger.warning(f"Unknown quantization method: {method}")
                return model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, using original model")
            return model


class FastCache:
    """Cache ultra-rápido con múltiples niveles."""
    
    def __init__(self, max_size: int = 10000):
        self.l1_cache: OrderedDict = OrderedDict()  # Cache en memoria
        self.l1_max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener del cache."""
        if key in self.l1_cache:
            self.hits += 1
            # Mover al final (LRU)
            value = self.l1_cache.pop(key)
            self.l1_cache[key] = value
            return value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Guardar en cache."""
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)
        else:
            if len(self.l1_cache) >= self.l1_max_size:
                self.l1_cache.popitem(last=False)
            self.l1_cache[key] = value
    
    def get_stats(self) -> Dict[str, float]:
        """Obtener estadísticas de cache."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.l1_cache)
        }


@functools.lru_cache(maxsize=1000)
def cached_path_calculation(
    graph_hash: str,
    start: str,
    end: str,
    strategy: str
) -> Optional[Tuple]:
    """Función de cálculo de ruta con cache LRU."""
    # Esta función será implementada por el router
    return None


class PerformanceProfiler:
    """Profiler de rendimiento."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
    
    def time_function(self, func_name: str):
        """Decorator para medir tiempo de funciones."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if func_name not in self.timings:
                    self.timings[func_name] = []
                self.timings[func_name].append(elapsed)
                
                return result
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtener estadísticas de timing."""
        stats = {}
        for func_name, times in self.timings.items():
            if times:
                stats[func_name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99),
                    'count': len(times)
                }
        return stats

