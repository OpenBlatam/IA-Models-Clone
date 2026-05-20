"""
Índice Vectorial
================

Índice vectorial para búsqueda rápida usando FAISS.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
import pickle

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS no disponible. Instalar con: pip install faiss-cpu o faiss-gpu")

logger = logging.getLogger(__name__)


class VectorIndex:
    """Índice vectorial para búsqueda rápida."""
    
    def __init__(
        self,
        dimension: int,
        use_gpu: bool = False,
        index_type: str = "flat"  # flat, ivf, hnsw
    ):
        """
        Inicializar índice vectorial.
        
        Args:
            dimension: Dimensión de los vectores
            use_gpu: Usar GPU
            index_type: Tipo de índice (flat, ivf, hnsw)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS no está disponible. Instalar con: pip install faiss-cpu")
        
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index_type = index_type
        
        # Crear índice
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # IVF con 100 clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.nprobe = 10
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Mover a GPU si está disponible
        if use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Índice movido a GPU")
            except Exception as e:
                logger.warning(f"No se pudo mover a GPU: {str(e)}")
        
        self._ids: List[int] = []
        self._is_trained = False
        
        logger.info(f"Índice vectorial creado: {index_type}, dim={dimension}")
    
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """
        Agregar vectores al índice.
        
        Args:
            vectors: Array de vectores (n, dimension)
            ids: IDs opcionales
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Dimensión incorrecta: esperado {self.dimension}, obtenido {vectors.shape[1]}")
        
        # Normalizar vectores
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        
        # Entrenar si es necesario (IVF)
        if self.index_type == "ivf" and not self._is_trained:
            self.index.train(vectors)
            self._is_trained = True
        
        # Agregar
        self.index.add(vectors)
        
        # Guardar IDs
        if ids is None:
            start_id = len(self._ids)
            ids = list(range(start_id, start_id + len(vectors)))
        self._ids.extend(ids)
        
        logger.info(f"Agregados {len(vectors)} vectores al índice")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Buscar vectores similares.
        
        Args:
            query_vector: Vector de consulta (1, dimension) o (dimension,)
            k: Número de resultados
        
        Returns:
            (distancias, índices)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        # Convertir índices a IDs
        result_ids = [self._ids[idx] for idx in indices[0] if idx < len(self._ids)]
        
        return distances[0][:len(result_ids)], np.array(result_ids)
    
    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int = 10
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Buscar múltiples queries."""
        results = []
        for query in query_vectors:
            dist, ids = self.search(query, k)
            results.append((dist, ids))
        return results
    
    def save(self, path: str):
        """Guardar índice."""
        faiss.write_index(self.index, path)
        # Guardar IDs
        with open(f"{path}.ids", 'wb') as f:
            pickle.dump(self._ids, f)
        logger.info(f"Índice guardado en {path}")
    
    def load(self, path: str):
        """Cargar índice."""
        self.index = faiss.read_index(path)
        # Cargar IDs
        with open(f"{path}.ids", 'rb') as f:
            self._ids = pickle.load(f)
        logger.info(f"Índice cargado desde {path}")
    
    def get_size(self) -> int:
        """Obtener número de vectores en el índice."""
        return self.index.ntotal
    
    def clear(self):
        """Limpiar índice."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.nprobe = 10
            self._is_trained = False
        self._ids.clear()




