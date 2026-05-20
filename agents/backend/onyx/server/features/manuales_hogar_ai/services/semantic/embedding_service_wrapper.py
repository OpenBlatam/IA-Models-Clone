"""
Embedding Service Wrapper
========================

Wrapper especializado para servicio de embeddings.
"""

from typing import List, Dict, Any, Optional
from ...core.base.service_base import BaseService
from ...ml.embeddings.embedding_service import EmbeddingService
from ...ml.config.ml_config import get_ml_config


class EmbeddingServiceWrapper(BaseService):
    """Wrapper para servicio de embeddings."""
    
    def __init__(self):
        """Inicializar wrapper."""
        super().__init__(logger_name=__name__)
        config = get_ml_config()
        self.service = EmbeddingService(
            model_name=config.embedding_model,
            device=config.device,
            use_cache=True,
            cache_size=2000
        )
        self.device = config.device
    
    def get_embedding_dimension(self) -> int:
        """Obtener dimensión de embeddings."""
        return self.service.get_embedding_dimension()
    
    def encode(self, text: str) -> List[float]:
        """
        Codificar texto a embedding.
        
        Args:
            text: Texto a codificar
        
        Returns:
            Embedding como lista de floats
        """
        return self.service.encode(text)
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 64
    ) -> List[List[float]]:
        """
        Codificar múltiples textos.
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño de batch
        
        Returns:
            Lista de embeddings
        """
        return self.service.encode(texts, batch_size=batch_size)
    
    def find_similar(
        self,
        query: str,
        texts: List[str],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[tuple]:
        """
        Encontrar textos similares.
        
        Args:
            query: Query de búsqueda
            texts: Lista de textos
            top_k: Número de resultados
            threshold: Umbral de similitud
        
        Returns:
            Lista de tuplas (idx, text, similarity)
        """
        return self.service.find_similar(query, texts, top_k, threshold)

