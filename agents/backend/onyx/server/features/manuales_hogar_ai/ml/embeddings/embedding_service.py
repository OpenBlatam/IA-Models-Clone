"""
Servicio de Embeddings
======================

Servicio para generar embeddings semánticos usando sentence-transformers.
"""

import logging
import torch
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ..optimizations.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Servicio para generar y comparar embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Inicializar servicio de embeddings.
        
        Args:
            model_name: Nombre del modelo de embeddings
            device: Dispositivo (cuda/cpu)
            use_cache: Usar caché de embeddings
            cache_size: Tamaño del caché
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache
        
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Modelo cargado. Dimensión: {self.embedding_dim}")
            
            # Inicializar caché
            if use_cache:
                self.cache = EmbeddingCache(max_size=cache_size)
                logger.info(f"Caché de embeddings inicializado (tamaño: {cache_size})")
            else:
                self.cache = None
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,  # Aumentado para mejor rendimiento
        show_progress: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generar embeddings para textos (con caché).
        
        Args:
            texts: Texto o lista de textos
            batch_size: Tamaño de batch
            show_progress: Mostrar progreso
        
        Returns:
            Array de embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                single_text = True
            else:
                single_text = False
            
            # Verificar caché
            if self.cache:
                cached_embeddings, texts_to_encode = self.cache.get_batch(texts)
                
                if len(texts_to_encode) == 0:
                    # Todo en caché
                    result = np.array(cached_embeddings)
                    return result[0] if single_text else result
                
                # Generar embeddings para textos no cacheados
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=normalize_embeddings
                )
                
                # Guardar en caché
                self.cache.set_batch(texts_to_encode, new_embeddings)
                
                # Combinar resultados
                result = []
                new_idx = 0
                for i, cached_emb in enumerate(cached_embeddings):
                    if cached_emb is not None:
                        result.append(cached_emb)
                    else:
                        result.append(new_embeddings[new_idx])
                        new_idx += 1
                
                result = np.array(result)
            else:
                # Sin caché
                result = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=normalize_embeddings
                )
            
            return result[0] if single_text else result
        
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
            raise
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calcular similitud coseno entre dos textos.
        
        Args:
            text1: Primer texto
            text2: Segundo texto
        
        Returns:
            Similitud (0-1)
        """
        try:
            embeddings = self.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculando similitud: {str(e)}")
            return 0.0
    
    def find_similar(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[tuple]:
        """
        Encontrar textos similares.
        
        Args:
            query: Texto de consulta
            texts: Lista de textos a buscar
            top_k: Número de resultados
            threshold: Umbral mínimo de similitud
        
        Returns:
            Lista de (índice, texto, similitud)
        """
        try:
            # Generar embeddings
            query_embedding = self.encode(query)
            text_embeddings = self.encode(texts)
            
            # Calcular similitudes
            similarities = cosine_similarity(query_embedding.reshape(1, -1), text_embeddings)[0]
            
            # Obtener top k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim >= threshold:
                    results.append((int(idx), texts[idx], sim))
            
            return results
        
        except Exception as e:
            logger.error(f"Error buscando similares: {str(e)}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Obtener dimensión de embeddings."""
        return self.embedding_dim
    
    def get_cache_stats(self) -> dict:
        """Obtener estadísticas del caché."""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}
    
    def clear_cache(self):
        """Limpiar caché."""
        if self.cache:
            self.cache.clear()

