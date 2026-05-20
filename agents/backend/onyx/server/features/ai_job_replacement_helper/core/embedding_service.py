"""
Embedding Service - Servicio de embeddings
============================================

Sistema para generar y gestionar embeddings de texto usando modelos de transformers.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Embedding:
    """Embedding de texto"""
    text: str
    vector: List[float]
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimilarityResult:
    """Resultado de similitud"""
    text: str
    similarity: float
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingService:
    """Servicio de embeddings"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.embeddings_cache: Dict[str, Embedding] = {}
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        logger.info("EmbeddingService initialized")
    
    async def generate_embedding(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Embedding:
        """Generar embedding de texto"""
        # En producción, esto usaría sentence-transformers
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(model_name or self.model_name)
        # vector = model.encode(text).tolist()
        
        # Por ahora, simulamos
        cache_key = f"{model_name or self.model_name}:{text}"
        
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Generar embedding simulado (normalizado)
        import random
        random.seed(hash(text))
        vector = [random.random() - 0.5 for _ in range(self.embedding_dimension)]
        norm = sum(x**2 for x in vector) ** 0.5
        vector = [x / norm for x in vector] if norm > 0 else vector
        
        embedding = Embedding(
            text=text,
            vector=vector,
            model=model_name or self.model_name,
            dimension=self.embedding_dimension,
        )
        
        self.embeddings_cache[cache_key] = embedding
        
        return embedding
    
    async def generate_batch_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        batch_size: int = 32
    ) -> List[Embedding]:
        """Generar embeddings en batch"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = [await self.generate_embedding(text, model_name) for text in batch]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calcular similitud coseno"""
        if len(embedding1) != len(embedding2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(x**2 for x in embedding1) ** 0.5
        norm2 = sum(x**2 for x in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def find_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """Encontrar textos similares"""
        query_embedding = await self.generate_embedding(query_text)
        candidate_embeddings = await self.generate_batch_embeddings(candidate_texts)
        
        results = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(
                query_embedding.vector,
                candidate_embedding.vector
            )
            
            if similarity >= threshold:
                results.append(SimilarityResult(
                    text=candidate_texts[i],
                    similarity=similarity,
                    index=i,
                ))
        
        # Ordenar por similitud
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        return results[:top_k]
    
    async def cluster_embeddings(
        self,
        texts: List[str],
        num_clusters: int = 5
    ) -> Dict[str, Any]:
        """Agrupar embeddings en clusters"""
        embeddings = await self.generate_batch_embeddings(texts)
        vectors = np.array([e.vector for e in embeddings])
        
        # En producción, usaría KMeans de scikit-learn
        # Por ahora, simulamos clustering
        
        # Asignación simple (simulado)
        clusters = {}
        for i, text in enumerate(texts):
            cluster_id = i % num_clusters
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append({
                "text": text,
                "index": i,
            })
        
        return {
            "num_clusters": num_clusters,
            "clusters": clusters,
            "total_texts": len(texts),
        }
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de embeddings"""
        return {
            "cached_embeddings": len(self.embeddings_cache),
            "model_name": self.model_name,
            "dimension": self.embedding_dimension,
        }




