"""
Routing Ultra-Fast Optimizations
=================================

Optimizaciones ultra-rápidas para máximo rendimiento.
Incluye: GPU acceleration, JIT compilation, memory pooling, async processing.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import threading
import queue
import time

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False


class MemoryPool:
    """Pool de memoria para reutilizar tensores y evitar allocaciones."""
    
    def __init__(self, max_size: int = 100):
        """
        Inicializar pool de memoria.
        
        Args:
            max_size: Tamaño máximo del pool
        """
        self.max_size = max_size
        self.pools: Dict[Tuple[int, ...], deque] = {}
        self.lock = threading.Lock()
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
        """
        Obtener tensor del pool o crear uno nuevo.
        
        Args:
            shape: Forma del tensor
            dtype: Tipo de datos
            device: Dispositivo
        
        Returns:
            Tensor del pool o nuevo
        """
        key = (shape, dtype, device)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].popleft()
                tensor.zero_()  # Limpiar
                return tensor
        
        # Crear nuevo tensor
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """
        Devolver tensor al pool.
        
        Args:
            tensor: Tensor a devolver
        """
        if tensor.numel() == 0:
            return
        
        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
        
        with self.lock:
            if key not in self.pools:
                self.pools[key] = deque(maxlen=self.max_size)
            
            if len(self.pools[key]) < self.max_size:
                self.pools[key].append(tensor.detach())


class AsyncRouteProcessor:
    """Procesador asíncrono de rutas con queue."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        """
        Inicializar procesador asíncrono.
        
        Args:
            max_workers: Número máximo de workers
            queue_size: Tamaño de la queue
        """
        self.max_workers = max_workers
        self.queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if CONCURRENT_AVAILABLE else None
        self.results: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def submit_route(self, route_id: str, route_func: callable, *args, **kwargs):
        """
        Enviar ruta para procesamiento asíncrono.
        
        Args:
            route_id: ID único de la ruta
            route_func: Función para calcular ruta
            *args: Argumentos posicionales
            **kwargs: Argumentos keyword
        """
        if not self.executor:
            # Fallback síncrono
            result = route_func(*args, **kwargs)
            with self.lock:
                self.results[route_id] = result
            return
        
        future = self.executor.submit(route_func, *args, **kwargs)
        
        def callback(f):
            try:
                result = f.result()
                with self.lock:
                    self.results[route_id] = result
            except Exception as e:
                logger.error(f"Error processing route {route_id}: {e}")
                with self.lock:
                    self.results[route_id] = None
        
        future.add_done_callback(callback)
    
    def get_result(self, route_id: str, timeout: float = 5.0) -> Optional[Any]:
        """
        Obtener resultado de ruta.
        
        Args:
            route_id: ID de la ruta
            timeout: Timeout en segundos
        
        Returns:
            Resultado o None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if route_id in self.results:
                    result = self.results.pop(route_id)
                    return result
            time.sleep(0.01)  # Pequeño delay
        
        return None


class GPUAccelerator:
    """Acelerador GPU con optimizaciones avanzadas."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Inicializar acelerador GPU.
        
        Args:
            device: Dispositivo GPU (auto-detect si None)
        """
        if not TORCH_AVAILABLE:
            self.device = "cpu"
            self.available = False
            return
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.available = self.device != "cpu" and torch.cuda.is_available()
        
        if self.available:
            # Optimizaciones de GPU
            torch.backends.cudnn.benchmark = True  # Optimizar convoluciones
            torch.backends.cudnn.deterministic = False  # Más rápido
            torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32
            torch.backends.cudnn.allow_tf32 = True
        
        self.memory_pool = MemoryPool() if TORCH_AVAILABLE else None
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mover tensor a GPU."""
        if self.available:
            return tensor.to(self.device, non_blocking=True)
        return tensor
    
    def batch_to_device(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Mover batch a GPU."""
        if self.available:
            return [t.to(self.device, non_blocking=True) for t in batch]
        return batch


class FastJITCompiler:
    """Compilador JIT ultra-rápido con caching."""
    
    def __init__(self):
        """Inicializar compilador JIT."""
        self.compiled_models: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def compile_model(
        self,
        model: nn.Module,
        model_id: str,
        example_input: torch.Tensor,
        optimize: bool = True
    ) -> nn.Module:
        """
        Compilar modelo con JIT.
        
        Args:
            model: Modelo PyTorch
            model_id: ID único del modelo
            example_input: Input de ejemplo
            optimize: Optimizar modelo
        
        Returns:
            Modelo compilado
        """
        if not TORCH_AVAILABLE:
            return model
        
        with self.lock:
            if model_id in self.compiled_models:
                return self.compiled_models[model_id]
        
        try:
            # Script mode (más rápido)
            model.eval()
            traced_model = torch.jit.trace(model, example_input)
            
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            with self.lock:
                self.compiled_models[model_id] = traced_model
            
            logger.info(f"Model {model_id} compiled with JIT")
            return traced_model
        except Exception as e:
            logger.warning(f"JIT compilation failed for {model_id}: {e}, using original model")
            return model


class VectorizedRouteCalculator:
    """Calculadora vectorizada de rutas usando NumPy."""
    
    def __init__(self):
        """Inicializar calculadora vectorizada."""
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, vectorization disabled")
    
    def batch_calculate_distances(
        self,
        positions: np.ndarray,
        path_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Calcular distancias en batch (vectorizado).
        
        Args:
            positions: Array de posiciones (N, 3)
            path_pairs: Lista de pares (start_idx, end_idx)
        
        Returns:
            Array de distancias
        """
        if not NUMPY_AVAILABLE:
            return np.array([])
        
        if len(path_pairs) == 0:
            return np.array([])
        
        # Vectorizar cálculo de distancias
        pairs = np.array(path_pairs)
        start_pos = positions[pairs[:, 0]]
        end_pos = positions[pairs[:, 1]]
        
        # Calcular distancias euclidianas
        distances = np.linalg.norm(end_pos - start_pos, axis=1)
        
        return distances
    
    def batch_calculate_paths(
        self,
        graph_matrix: np.ndarray,
        start_indices: np.ndarray,
        end_indices: np.ndarray
    ) -> List[List[int]]:
        """
        Calcular múltiples rutas en batch usando matrices.
        
        Args:
            graph_matrix: Matriz de adyacencia (N, N)
            start_indices: Índices de inicio
            end_indices: Índices de fin
        
        Returns:
            Lista de rutas
        """
        if not NUMPY_AVAILABLE:
            return []
        
        # Usar Floyd-Warshall para precomputar todas las distancias
        n = graph_matrix.shape[0]
        dist = graph_matrix.copy()
        next_node = np.full((n, n), -1, dtype=np.int32)
        
        for i in range(n):
            for j in range(n):
                if i != j and dist[i, j] != np.inf:
                    next_node[i, j] = j
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        next_node[i, j] = next_node[i, k]
        
        # Reconstruir rutas
        paths = []
        for start, end in zip(start_indices, end_indices):
            if dist[start, end] == np.inf:
                paths.append([])
                continue
            
            path = [start]
            current = start
            while current != end:
                current = next_node[current, end]
                if current == -1:
                    break
                path.append(current)
            
            paths.append(path)
        
        return paths


class PrecomputationEngine:
    """Motor de precomputación agresiva."""
    
    def __init__(self, max_nodes: int = 1000):
        """
        Inicializar motor de precomputación.
        
        Args:
            max_nodes: Número máximo de nodos para precomputar
        """
        self.max_nodes = max_nodes
        self.distance_matrix: Optional[np.ndarray] = None
        self.path_matrix: Optional[np.ndarray] = None
        self.node_indices: Dict[str, int] = {}
        self.computed = False
    
    def precompute_all_pairs(self, nodes: Dict[str, Any], edges: Dict[str, Any]):
        """
        Precomputar todas las distancias entre pares de nodos.
        
        Args:
            nodes: Diccionario de nodos
            edges: Diccionario de aristas
        """
        if len(nodes) > self.max_nodes:
            logger.warning(f"Too many nodes ({len(nodes)}), skipping precomputation")
            return
        
        if not NUMPY_AVAILABLE:
            return
        
        n = len(nodes)
        self.node_indices = {node_id: i for i, node_id in enumerate(nodes.keys())}
        
        # Inicializar matriz de distancias
        self.distance_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(self.distance_matrix, 0.0)
        
        self.path_matrix = np.full((n, n), -1, dtype=np.int32)
        
        # Llenar con aristas
        for edge in edges.values():
            from_idx = self.node_indices.get(edge.from_node)
            to_idx = self.node_indices.get(edge.to_node)
            
            if from_idx is not None and to_idx is not None:
                dist = edge.distance
                if dist < self.distance_matrix[from_idx, to_idx]:
                    self.distance_matrix[from_idx, to_idx] = dist
                    self.path_matrix[from_idx, to_idx] = to_idx
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.distance_matrix[i, k] + self.distance_matrix[k, j] < self.distance_matrix[i, j]:
                        self.distance_matrix[i, j] = self.distance_matrix[i, k] + self.distance_matrix[k, j]
                        self.path_matrix[i, j] = self.path_matrix[i, k]
        
        self.computed = True
        logger.info(f"Precomputed all-pairs shortest paths for {n} nodes")
    
    def get_distance(self, start_node: str, end_node: str) -> Optional[float]:
        """
        Obtener distancia precomputada.
        
        Args:
            start_node: Nodo inicio
            end_node: Nodo fin
        
        Returns:
            Distancia o None
        """
        if not self.computed:
            return None
        
        start_idx = self.node_indices.get(start_node)
        end_idx = self.node_indices.get(end_node)
        
        if start_idx is None or end_idx is None:
            return None
        
        dist = self.distance_matrix[start_idx, end_idx]
        return float(dist) if dist != np.inf else None
    
    def get_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """
        Obtener ruta precomputada.
        
        Args:
            start_node: Nodo inicio
            end_node: Nodo fin
        
        Returns:
            Lista de IDs de nodos o None
        """
        if not self.computed:
            return None
        
        start_idx = self.node_indices.get(start_node)
        end_idx = self.node_indices.get(end_node)
        
        if start_idx is None or end_idx is None:
            return None
        
        if self.distance_matrix[start_idx, end_idx] == np.inf:
            return None
        
        # Reconstruir ruta
        path_indices = [start_idx]
        current = start_idx
        
        while current != end_idx:
            next_idx = self.path_matrix[current, end_idx]
            if next_idx == -1:
                return None
            path_indices.append(next_idx)
            current = next_idx
        
        # Convertir índices a IDs de nodos
        reverse_indices = {v: k for k, v in self.node_indices.items()}
        return [reverse_indices[i] for i in path_indices]


class UltraFastRouter:
    """Router ultra-rápido con todas las optimizaciones."""
    
    def __init__(self, use_gpu: bool = True, enable_precomputation: bool = True):
        """
        Inicializar router ultra-rápido.
        
        Args:
            use_gpu: Usar GPU si está disponible
            enable_precomputation: Habilitar precomputación
        """
        self.gpu_accelerator = GPUAccelerator() if TORCH_AVAILABLE else None
        self.jit_compiler = FastJITCompiler() if TORCH_AVAILABLE else None
        self.vectorized_calc = VectorizedRouteCalculator() if NUMPY_AVAILABLE else None
        self.precomputation = PrecomputationEngine() if enable_precomputation and NUMPY_AVAILABLE else None
        self.async_processor = AsyncRouteProcessor() if CONCURRENT_AVAILABLE else None
    
    def optimize_model(self, model: nn.Module, model_id: str, example_input: torch.Tensor) -> nn.Module:
        """
        Optimizar modelo con todas las técnicas.
        
        Args:
            model: Modelo PyTorch
            model_id: ID del modelo
            example_input: Input de ejemplo
        
        Returns:
            Modelo optimizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        # Mover a GPU
        if self.gpu_accelerator and self.gpu_accelerator.available:
            model = model.to(self.gpu_accelerator.device)
            example_input = example_input.to(self.gpu_accelerator.device)
        
        # Compilar con JIT
        if self.jit_compiler:
            model = self.jit_compiler.compile_model(model, model_id, example_input)
        
        return model
    
    def precompute_graph(self, nodes: Dict[str, Any], edges: Dict[str, Any]):
        """Precomputar grafo completo."""
        if self.precomputation:
            self.precomputation.precompute_all_pairs(nodes, edges)

