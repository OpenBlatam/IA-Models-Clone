"""
Embedding Manager - Gestor de embeddings y similitud semántica
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Gestor de embeddings y similitud semántica."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._initialized = False
        self.model = None
        self.embedding_cache = {}
        self.cache_file = "embeddings_cache.pkl"
        self.max_cache_size = 10000
    
    async def initialize(self):
        """Inicializar el gestor de embeddings."""
        if not self._initialized:
            try:
                logger.info(f"Inicializando EmbeddingManager con modelo: {self.model_name}")
                
                # Cargar modelo de sentence transformers
                self.model = SentenceTransformer(self.model_name)
                
                # Cargar cache si existe
                await self._load_cache()
                
                self._initialized = True
                logger.info("EmbeddingManager inicializado exitosamente")
                
            except Exception as e:
                logger.error(f"Error al inicializar EmbeddingManager: {e}")
                raise
    
    async def shutdown(self):
        """Cerrar el gestor de embeddings."""
        if self._initialized:
            try:
                # Guardar cache
                await self._save_cache()
                
                # Limpiar modelo
                if self.model:
                    del self.model
                    self.model = None
                
                self._initialized = False
                logger.info("EmbeddingManager cerrado")
                
            except Exception as e:
                logger.error(f"Error al cerrar EmbeddingManager: {e}")
    
    async def _load_cache(self):
        """Cargar cache de embeddings."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Cache de embeddings cargado: {len(self.embedding_cache)} entradas")
        except Exception as e:
            logger.warning(f"No se pudo cargar el cache de embeddings: {e}")
            self.embedding_cache = {}
    
    async def _save_cache(self):
        """Guardar cache de embeddings."""
        try:
            # Limpiar cache si es muy grande
            if len(self.embedding_cache) > self.max_cache_size:
                # Mantener solo las entradas más recientes
                sorted_items = sorted(self.embedding_cache.items(), key=lambda x: x[1]['timestamp'], reverse=True)
                self.embedding_cache = dict(sorted_items[:self.max_cache_size])
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Cache de embeddings guardado: {len(self.embedding_cache)} entradas")
        except Exception as e:
            logger.warning(f"No se pudo guardar el cache de embeddings: {e}")
    
    def _generate_cache_key(self, text: str) -> str:
        """Generar clave de cache para texto."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    async def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Obtener embedding para un texto."""
        if not self._initialized:
            await self.initialize()
        
        # Verificar cache
        if use_cache:
            cache_key = self._generate_cache_key(text)
            if cache_key in self.embedding_cache:
                return np.array(self.embedding_cache[cache_key]['embedding'])
        
        try:
            # Generar embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Guardar en cache
            if use_cache:
                cache_key = self._generate_cache_key(text)
                self.embedding_cache[cache_key] = {
                    'embedding': embedding.tolist(),
                    'timestamp': datetime.now(),
                    'text': text[:100]  # Guardar solo los primeros 100 caracteres
                }
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error al generar embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Obtener embeddings para múltiples textos."""
        if not self._initialized:
            await self.initialize()
        
        embeddings = []
        
        try:
            # Procesar en lotes
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error al generar embeddings en lote: {e}")
            raise
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similitud coseno entre dos textos."""
        try:
            # Obtener embeddings
            embedding1 = await self.get_embedding(text1)
            embedding2 = await self.get_embedding(text2)
            
            # Calcular similitud coseno
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error al calcular similitud: {e}")
            raise
    
    async def find_most_similar(self, query_text: str, candidate_texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Encontrar textos más similares a una consulta."""
        try:
            # Obtener embedding de la consulta
            query_embedding = await self.get_embedding(query_text)
            
            # Obtener embeddings de candidatos
            candidate_embeddings = await self.get_embeddings_batch(candidate_texts)
            
            # Calcular similitudes
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
                similarities.append({
                    "text": candidate_texts[i],
                    "similarity": float(similarity),
                    "index": i
                })
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error al encontrar textos similares: {e}")
            raise
    
    async def cluster_texts(self, texts: List[str], n_clusters: int = 3) -> Dict[str, Any]:
        """Agrupar textos por similitud semántica."""
        try:
            from sklearn.cluster import KMeans
            
            # Obtener embeddings
            embeddings = await self.get_embeddings_batch(texts)
            embeddings_array = np.array(embeddings)
            
            # Aplicar clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Organizar resultados
            clusters = {}
            for i, (text, label) in enumerate(zip(texts, cluster_labels)):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    "text": text,
                    "index": i
                })
            
            return {
                "clusters": clusters,
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en clustering de textos: {e}")
            raise
    
    async def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Búsqueda semántica en documentos."""
        try:
            # Encontrar documentos más similares
            similar_docs = await self.find_most_similar(query, documents, top_k)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            raise
    
    async def get_text_representation(self, text: str) -> Dict[str, Any]:
        """Obtener representación completa de un texto."""
        try:
            # Obtener embedding
            embedding = await self.get_embedding(text)
            
            # Calcular estadísticas del embedding
            embedding_stats = {
                "dimension": len(embedding),
                "mean": float(np.mean(embedding)),
                "std": float(np.std(embedding)),
                "min": float(np.min(embedding)),
                "max": float(np.max(embedding)),
                "norm": float(np.linalg.norm(embedding))
            }
            
            return {
                "text": text,
                "embedding": embedding.tolist(),
                "embedding_stats": embedding_stats,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener representación de texto: {e}")
            raise
    
    async def compare_texts(self, texts: List[str]) -> Dict[str, Any]:
        """Comparar múltiples textos y obtener matriz de similitud."""
        try:
            # Obtener embeddings
            embeddings = await self.get_embeddings_batch(texts)
            embeddings_array = np.array(embeddings)
            
            # Calcular matriz de similitud
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Crear resultados detallados
            comparisons = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    comparisons.append({
                        "text1": texts[i],
                        "text2": texts[j],
                        "similarity": float(similarity_matrix[i][j]),
                        "index1": i,
                        "index2": j
                    })
            
            # Ordenar por similitud
            comparisons.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "texts": texts,
                "similarity_matrix": similarity_matrix.tolist(),
                "pairwise_comparisons": comparisons,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al comparar textos: {e}")
            raise
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        return {
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.max_cache_size,
            "cache_usage_percentage": (len(self.embedding_cache) / self.max_cache_size) * 100,
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def clear_cache(self):
        """Limpiar cache de embeddings."""
        self.embedding_cache.clear()
        logger.info("Cache de embeddings limpiado")
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del gestor de embeddings."""
        try:
            return {
                "status": "healthy" if self._initialized else "unhealthy",
                "initialized": self._initialized,
                "model_name": self.model_name,
                "model_loaded": self.model is not None,
                "cache_size": len(self.embedding_cache),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




