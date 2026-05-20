"""
Embeddings - Sistema de embeddings para búsqueda semántica
==========================================================

Sistema de embeddings para búsqueda semántica de comandos y tareas.
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Almacén de embeddings para búsqueda semántica"""
    
    def __init__(self, storage_path: str = "./data/embeddings.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._embeddings: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict] = {}
        self._model = None
        self._initialized = False
        
    async def initialize(self):
        """Inicializar modelo de embeddings"""
        if self._initialized:
            return
        
        try:
            # Intentar cargar modelo local
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ SentenceTransformer loaded")
            except ImportError:
                try:
                    from transformers import AutoTokenizer, AutoModel
                    import torch
                    
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._model = AutoModel.from_pretrained(model_name)
                    self._model.eval()
                    logger.info("✅ Transformers model loaded")
                except ImportError:
                    logger.warning("No embedding models available")
                    self._model = None
            
            # Cargar embeddings guardados
            await self.load()
            
            self._initialized = True
        except Exception as e:
            logger.warning(f"Could not initialize embeddings: {e}")
            self._model = None
    
    async def embed(self, text: str) -> Optional[List[float]]:
        """Generar embedding para un texto"""
        if not self._initialized:
            await self.initialize()
        
        if not self._model:
            return None
        
        try:
            if hasattr(self._model, 'encode'):
                # SentenceTransformer
                embedding = self._model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            else:
                # Transformers
                import torch
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def add(self, key: str, text: str, metadata: Optional[Dict] = None):
        """Agregar embedding"""
        embedding = await self.embed(text)
        if embedding:
            self._embeddings[key] = embedding
            self._metadata[key] = {
                "text": text,
                **(metadata or {})
            }
            await self.save()
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[str, float, Dict]]:
        """Buscar embeddings similares"""
        if not self._initialized:
            await self.initialize()
        
        query_embedding = await self.embed(query)
        if not query_embedding:
            return []
        
        if not self._embeddings:
            return []
        
        # Calcular similitud coseno
        similarities = []
        for key, embedding in self._embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((key, similarity, self._metadata.get(key, {})))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcular similitud coseno"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def save(self):
        """Guardar embeddings"""
        try:
            data = {
                "embeddings": self._embeddings,
                "metadata": self._metadata
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    async def load(self):
        """Cargar embeddings"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._embeddings = data.get("embeddings", {})
                    self._metadata = data.get("metadata", {})
                logger.info(f"✅ Loaded {len(self._embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")
    
    async def remove(self, key: str):
        """Eliminar embedding"""
        if key in self._embeddings:
            del self._embeddings[key]
        if key in self._metadata:
            del self._metadata[key]
        await self.save()
    
    def clear(self):
        """Limpiar todos los embeddings"""
        self._embeddings.clear()
        self._metadata.clear()


