"""
Document Semantic Search - Búsqueda Semántica Avanzada
=======================================================

Búsqueda semántica de documentos usando embeddings.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado de búsqueda semántica."""
    document_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    highlights: List[str] = field(default_factory=list)


@dataclass
class SemanticIndex:
    """Índice semántico de documentos."""
    index_id: str
    document_count: int
    created_at: datetime = field(default_factory=datetime.now)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


class SemanticSearchEngine:
    """Motor de búsqueda semántica."""
    
    def __init__(self, analyzer):
        """Inicializar motor."""
        self.analyzer = analyzer
        self.indexes: Dict[str, SemanticIndex] = {}
        self.document_store: Dict[str, Dict[str, Any]] = {}
    
    async def create_index(
        self,
        index_id: str,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> SemanticIndex:
        """
        Crear índice semántico.
        
        Args:
            index_id: ID del índice
            documents: Lista de (doc_id, content, metadata)
        
        Returns:
            SemanticIndex creado
        """
        index = SemanticIndex(
            index_id=index_id,
            document_count=len(documents)
        )
        
        # Generar embeddings para cada documento
        for doc_id, content, metadata in documents:
            try:
                # Usar generador de embeddings del analizador
                if hasattr(self.analyzer, 'embedding_generator'):
                    embedding = await self.analyzer.embedding_generator.generate_embedding(content)
                    index.embeddings[doc_id] = embedding
                    
                    # Guardar documento
                    self.document_store[doc_id] = {
                        "content": content,
                        "metadata": metadata or {},
                        "index_id": index_id
                    }
            except Exception as e:
                logger.error(f"Error generando embedding para {doc_id}: {e}")
        
        self.indexes[index_id] = index
        logger.info(f"Índice semántico creado: {index_id} con {len(documents)} documentos")
        
        return index
    
    async def search(
        self,
        index_id: str,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Buscar documentos semánticamente similares.
        
        Args:
            index_id: ID del índice
            query: Consulta de búsqueda
            top_k: Número de resultados
            threshold: Umbral de similitud mínimo
        
        Returns:
            Lista de SearchResult ordenados por score
        """
        if index_id not in self.indexes:
            raise ValueError(f"Índice {index_id} no encontrado")
        
        index = self.indexes[index_id]
        
        # Generar embedding de la consulta
        if hasattr(self.analyzer, 'embedding_generator'):
            query_embedding = await self.analyzer.embedding_generator.generate_embedding(query)
        else:
            # Fallback: embedding simple
            query_embedding = np.random.rand(768)  # Placeholder
        
        # Calcular similitud con todos los documentos
        results = []
        
        for doc_id, doc_embedding in index.embeddings.items():
            # Calcular similitud coseno
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= threshold:
                doc_data = self.document_store.get(doc_id, {})
                results.append(SearchResult(
                    document_id=doc_id,
                    score=similarity,
                    content=doc_data.get("content", ""),
                    metadata=doc_data.get("metadata", {}),
                    highlights=self._extract_highlights(query, doc_data.get("content", ""))
                ))
        
        # Ordenar por score descendente
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcular similitud coseno."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _extract_highlights(self, query: str, content: str, max_highlights: int = 3) -> List[str]:
        """Extraer highlights de la consulta en el contenido."""
        highlights = []
        query_words = query.lower().split()
        
        sentences = content.split('. ')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            if matches > 0:
                highlights.append(sentence.strip())
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    def get_index(self, index_id: str) -> Optional[SemanticIndex]:
        """Obtener índice."""
        return self.indexes.get(index_id)
    
    def list_indexes(self) -> List[str]:
        """Listar índices."""
        return list(self.indexes.keys())


__all__ = [
    "SemanticSearchEngine",
    "SemanticIndex",
    "SearchResult"
]



