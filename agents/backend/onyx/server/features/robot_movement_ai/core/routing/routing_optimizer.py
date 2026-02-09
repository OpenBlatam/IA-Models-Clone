"""
Routing Performance Optimizer
============================

Optimizaciones de rendimiento para routing: caching, batch processing, GPU acceleration.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from functools import lru_cache
from collections import OrderedDict
import hashlib
import time

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FastPathCache:
    """Cache de alta velocidad para rutas calculadas con optimizaciones."""
    
    def __init__(self, max_size: int = 10000, ttl: float = 3600.0):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps: Dict[str, float] = {}
        # Pre-allocate para mejor rendimiento
        self._access_count: Dict[str, int] = {}
    
    def _make_key(self, start: str, end: str, strategy: str, graph_hash: str) -> str:
        """Crear clave única para cache."""
        return hashlib.md5(f"{start}_{end}_{strategy}_{graph_hash}".encode()).hexdigest()
    
    def get(self, start: str, end: str, strategy: str, graph_hash: str) -> Optional[Tuple]:
        """Obtener ruta del cache (optimizado)."""
        key = self._make_key(start, end, strategy, graph_hash)
        
        if key not in self.cache:
            return None
        
        # Verificar TTL (solo si es necesario)
        timestamp = self.timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            if key in self._access_count:
                del self._access_count[key]
            return None
        
        # Mover al final (LRU) - optimizado
        result = self.cache.pop(key)
        self.cache[key] = result
        self._access_count[key] = self._access_count.get(key, 0) + 1
        return result
    
    def put(self, start: str, end: str, strategy: str, graph_hash: str, result: Tuple):
        """Guardar ruta en cache (optimizado)."""
        key = self._make_key(start, end, strategy, graph_hash)
        
        if len(self.cache) >= self.max_size:
            # Eliminar el menos accedido (LFU) o el más antiguo (LRU)
            if self._access_count:
                # Encontrar clave menos accedida
                least_accessed = min(self._access_count.items(), key=lambda x: x[1])[0]
                del self.cache[least_accessed]
                del self.timestamps[least_accessed]
                del self._access_count[least_accessed]
            else:
                # Fallback a LRU
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
        
        self.cache[key] = result
        self.timestamps[key] = time.time()
        self._access_count[key] = 0
    
    def clear(self):
        """Limpiar cache."""
        self.cache.clear()
        self.timestamps.clear()


class BatchRouteProcessor:
    """Procesador de rutas en batch para mejor rendimiento (optimizado)."""
    
    def __init__(self, device: Optional[str] = None, batch_size: int = 64):
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
    
    def process_batch(
        self,
        routes: List[Dict[str, Any]],
        graph: Dict[str, Dict[str, Dict[str, float]]],
        strategy_func: callable
    ) -> List[Tuple]:
        """Procesar múltiples rutas en batch."""
        if not routes:
            return []
        
        # Vectorizar operaciones cuando sea posible
        if TORCH_AVAILABLE and len(routes) > 10:
            return self._process_batch_gpu(routes, graph, strategy_func)
        else:
            return self._process_batch_cpu(routes, graph, strategy_func)
    
    def _process_batch_cpu(
        self,
        routes: List[Dict[str, Any]],
        graph: Dict[str, Dict[str, Dict[str, float]]],
        strategy_func: callable
    ) -> List[Tuple]:
        """Procesar batch en CPU."""
        results = []
        for route in routes:
            try:
                result = strategy_func(
                    graph,
                    route['start'],
                    route['end'],
                    route.get('nodes', {}),
                    route.get('edges', {})
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error procesando ruta: {e}")
                results.append((None, 0.0, 0.0, 0.0, 0.0))
        return results
    
    def _process_batch_gpu(
        self,
        routes: List[Dict[str, Any]],
        graph: Dict[str, Dict[str, Dict[str, float]]],
        strategy_func: callable
    ) -> List[Tuple]:
        """Procesar batch en GPU cuando sea posible (optimizado)."""
        # Agrupar rutas en batches para mejor eficiencia
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        num_workers = min(len(routes), mp.cpu_count(), 8)
        results = [None] * len(routes)
        
        # Procesar en chunks para mejor balanceo de carga
        chunk_size = max(1, len(routes) // num_workers)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit en chunks
            future_to_idx = {}
            for i in range(0, len(routes), chunk_size):
                chunk = routes[i:i + chunk_size]
                for j, route in enumerate(chunk):
                    idx = i + j
                    future = executor.submit(
                        strategy_func,
                        graph,
                        route['start'],
                        route['end'],
                        route.get('nodes', {}),
                        route.get('edges', {})
                    )
                    future_to_idx[future] = idx
            
            # Recoger resultados conforme se completan
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Error processing route {idx}: {e}")
                    results[idx] = (None, 0.0, 0.0, 0.0, 0.0)
        
        return results


class GraphHashCalculator:
    """Calculador rápido de hash para grafos."""
    
    @staticmethod
    def calculate_hash(
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> str:
        """Calcular hash del grafo para cache."""
        # Crear representación serializable
        node_repr = sorted([(nid, str(node)) for nid, node in nodes.items()])
        edge_repr = sorted([(eid, str(edge)) for eid, edge in edges.items()])
        
        combined = f"{node_repr}_{edge_repr}"
        return hashlib.md5(combined.encode()).hexdigest()


class VectorizedPathOperations:
    """Operaciones vectorizadas para mejor rendimiento."""
    
    @staticmethod
    def batch_dijkstra(
        graphs: List[Dict[str, Dict[str, Dict[str, float]]]],
        starts: List[str],
        ends: List[str],
        weight: str = "distance"
    ) -> List[Tuple]:
        """Ejecutar Dijkstra en batch vectorizado."""
        import heapq
        
        results = []
        for graph, start, end in zip(graphs, starts, ends):
            distances = {node: float('inf') for node in graph}
            distances[start] = 0.0
            previous = {node: None for node in graph}
            queue = [(0.0, start)]
            
            while queue:
                current_dist, current = heapq.heappop(queue)
                
                if current == end:
                    break
                
                if current_dist > distances[current]:
                    continue
                
                if current not in graph:
                    continue
                
                for neighbor, edge_data in graph[current].items():
                    edge_weight = edge_data.get(weight, float('inf'))
                    new_dist = current_dist + edge_weight
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(queue, (new_dist, neighbor))
            
            # Reconstruir ruta
            path = []
            current = end
            while current is not None:
                path.insert(0, current)
                current = previous[current]
            
            if not path or path[0] != start:
                results.append((None, 0.0, 0.0, 0.0))
            else:
                # Calcular métricas
                total_distance = 0.0
                total_time = 0.0
                total_cost = 0.0
                
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    if from_node in graph and to_node in graph[from_node]:
                        edge_data = graph[from_node][to_node]
                        total_distance += edge_data.get("distance", 0.0)
                        total_time += edge_data.get("time", 0.0)
                        total_cost += edge_data.get("cost", 0.0)
                
                results.append((path, total_distance, total_time, total_cost))
        
        return results
    
    @staticmethod
    def vectorized_distance_matrix(
        positions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calcular matriz de distancias vectorizada."""
        if not positions:
            return np.array([])
        
        pos_array = np.array(list(positions.values()))
        node_ids = list(positions.keys())
        
        # Calcular distancias euclidianas vectorizadas
        diff = pos_array[:, np.newaxis, :] - pos_array[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        return distances, node_ids


class ModelInferenceOptimizer:
    """Optimizador de inferencia de modelos ML."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.model_cache: Dict[str, Any] = {}
    
    def optimize_model(self, model: Any) -> Any:
        """Optimizar modelo para inferencia."""
        if not TORCH_AVAILABLE:
            return model
        
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device)
            model.eval()
            
            # Compilar con torch.jit si es posible
            try:
                if hasattr(torch.jit, 'script'):
                    model = torch.jit.script(model)
            except:
                pass
            
            # Habilitar optimizaciones de inferencia
            if hasattr(torch, 'inference_mode'):
                torch.set_grad_enabled(False)
        
        return model
    
    def batch_inference(
        self,
        model: Any,
        inputs: List[torch.Tensor],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """Inferencia en batch optimizada."""
        if not TORCH_AVAILABLE:
            return []
        
        results = []
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                batch_tensor = torch.stack(batch).to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    outputs = model(batch_tensor)
                
                results.extend(outputs.cpu())
        
        return results


class RoutePrecomputation:
    """Precomputación de rutas comunes."""
    
    def __init__(self, graph: Dict[str, Dict[str, Dict[str, float]]]):
        self.graph = graph
        self.shortest_paths: Dict[Tuple[str, str], List[str]] = {}
        self.distance_matrix: Optional[np.ndarray] = None
        self.node_ids: List[str] = None
    
    def precompute_all_pairs(self, max_nodes: int = 100):
        """Precomputar todas las rutas entre pares de nodos."""
        nodes = list(self.graph.keys())
        if len(nodes) > max_nodes:
            logger.warning(f"Grafo muy grande ({len(nodes)} nodos), saltando precomputación")
            return
        
        self.node_ids = nodes
        n = len(nodes)
        self.distance_matrix = np.full((n, n), np.inf)
        
        # Floyd-Warshall para todas las rutas más cortas
        for i, node_i in enumerate(nodes):
            self.distance_matrix[i, i] = 0.0
            if node_i in self.graph:
                for node_j, edge_data in self.graph[node_i].items():
                    if node_j in nodes:
                        j = nodes.index(node_j)
                        self.distance_matrix[i, j] = edge_data.get("distance", np.inf)
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.distance_matrix[i, k] + self.distance_matrix[k, j] < self.distance_matrix[i, j]:
                        self.distance_matrix[i, j] = self.distance_matrix[i, k] + self.distance_matrix[k, j]
    
    def get_precomputed_distance(self, start: str, end: str) -> Optional[float]:
        """Obtener distancia precomputada."""
        if self.distance_matrix is None or self.node_ids is None:
            return None
        
        if start not in self.node_ids or end not in self.node_ids:
            return None
        
        i = self.node_ids.index(start)
        j = self.node_ids.index(end)
        
        dist = self.distance_matrix[i, j]
        return dist if dist != np.inf else None


class PerformanceMonitor:
    """Monitor de rendimiento para routing."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'route_time': [],
            'cache_hits': [],
            'cache_misses': [],
            'batch_size': []
        }
    
    def record_route_time(self, time_taken: float):
        """Registrar tiempo de cálculo de ruta."""
        self.metrics['route_time'].append(time_taken)
    
    def record_cache_event(self, hit: bool):
        """Registrar evento de cache."""
        if hit:
            self.metrics['cache_hits'].append(1.0)
        else:
            self.metrics['cache_misses'].append(1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento."""
        stats = {}
        
        if self.metrics['route_time']:
            stats['avg_route_time'] = np.mean(self.metrics['route_time'])
            stats['p95_route_time'] = np.percentile(self.metrics['route_time'], 95)
            stats['p99_route_time'] = np.percentile(self.metrics['route_time'], 99)
        
        total_cache = len(self.metrics['cache_hits']) + len(self.metrics['cache_misses'])
        if total_cache > 0:
            stats['cache_hit_rate'] = len(self.metrics['cache_hits']) / total_cache
        
        return stats
    
    def reset(self):
        """Resetear métricas."""
        for key in self.metrics:
            self.metrics[key].clear()

