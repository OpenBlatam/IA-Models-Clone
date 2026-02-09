"""
Búsqueda Semántica Avanzada
============================

Sistema de búsqueda semántica con índices vectoriales y búsqueda híbrida.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .embedding_generator import EmbeddingGenerator
from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado de búsqueda"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class SemanticSearchEngine:
    """
    Motor de búsqueda semántica avanzado
    
    Proporciona:
    - Búsqueda semántica usando embeddings
    - Índices vectoriales en memoria
    - Búsqueda híbrida (semántica + keyword)
    - Filtrado por metadata
    - Ranking inteligente
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar motor de búsqueda
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        self.embedding_generator = analyzer.embedding_generator
        
        # Índice de documentos
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata_index: Dict[str, List[str]] = {}
        
        logger.info("SemanticSearchEngine inicializado")
    
    def index_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Indexar documento
        
        Args:
            document_id: ID único del documento
            content: Contenido del documento
            metadata: Metadata adicional
        """
        # Generar embedding
        embedding = self.embedding_generator.generate_embeddings([content])
        if isinstance(embedding, list):
            embedding = embedding[0]
        
        # Almacenar documento
        self.documents[document_id] = {
            "content": content,
            "metadata": metadata or {},
            "indexed_at": datetime.now().isoformat()
        }
        
        # Almacenar embedding
        if isinstance(embedding, np.ndarray):
            self.embeddings[document_id] = embedding
        else:
            self.embeddings[document_id] = np.array(embedding)
        
        # Indexar metadata
        if metadata:
            for key, value in metadata.items():
                if key not in self.metadata_index:
                    self.metadata_index[key] = {}
                if value not in self.metadata_index[key]:
                    self.metadata_index[key][value] = []
                self.metadata_index[key][value].append(document_id)
        
        logger.debug(f"Documento indexado: {document_id}")
    
    def remove_document(self, document_id: str):
        """Remover documento del índice"""
        if document_id in self.documents:
            del self.documents[document_id]
        
        if document_id in self.embeddings:
            del self.embeddings[document_id]
        
        # Remover de metadata index
        for key in self.metadata_index:
            for value in self.metadata_index[key]:
                if document_id in self.metadata_index[key][value]:
                    self.metadata_index[key][value].remove(document_id)
        
        logger.debug(f"Documento removido: {document_id}")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[SearchResult]:
        """
        Buscar documentos
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a retornar
            filters: Filtros por metadata
            use_hybrid: Usar búsqueda híbrida (semántica + keyword)
        
        Returns:
            Lista de SearchResult ordenados por relevancia
        """
        # Generar embedding de la consulta
        query_embedding = self.embedding_generator.generate_embeddings([query])
        if isinstance(query_embedding, list):
            query_embedding = query_embedding[0]
        
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Aplicar filtros
        candidate_ids = set(self.documents.keys())
        if filters:
            for key, value in filters.items():
                if key in self.metadata_index and value in self.metadata_index[key]:
                    filtered_ids = set(self.metadata_index[key][value])
                    candidate_ids = candidate_ids & filtered_ids
        
        # Calcular similitudes
        results = []
        for doc_id in candidate_ids:
            if doc_id not in self.embeddings:
                continue
            
            doc_embedding = self.embeddings[doc_id]
            
            # Similitud coseno
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            # Búsqueda híbrida: combinar con keyword matching
            if use_hybrid:
                keyword_score = self._keyword_match_score(query, self.documents[doc_id]["content"])
                # Combinar scores (70% semántico, 30% keyword)
                final_score = 0.7 * similarity + 0.3 * keyword_score
            else:
                final_score = similarity
            
            results.append({
                "document_id": doc_id,
                "score": float(final_score),
                "similarity": float(similarity)
            })
        
        # Ordenar por score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Crear SearchResult objects
        search_results = []
        for result in results[:top_k]:
            doc_id = result["document_id"]
            doc = self.documents[doc_id]
            
            # Generar highlights
            highlights = self._generate_highlights(query, doc["content"])
            
            search_results.append(SearchResult(
                document_id=doc_id,
                content=doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"],
                score=result["score"],
                metadata=doc["metadata"],
                highlights=highlights
            ))
        
        return search_results
    
    def _keyword_match_score(self, query: str, content: str) -> float:
        """Calcular score de keyword matching"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        matches = query_words & content_words
        if not query_words:
            return 0.0
        
        return len(matches) / len(query_words)
    
    def _generate_highlights(self, query: str, content: str, max_highlights: int = 3) -> List[str]:
        """Generar highlights de la búsqueda"""
        query_words = query.lower().split()
        sentences = content.split('.')
        
        highlights = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for word in query_words if word in sentence_lower)
            if matches > 0:
                highlights.append(sentence.strip())
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del índice"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "metadata_fields": list(self.metadata_index.keys()),
            "index_size_mb": sum(
                emb.nbytes for emb in self.embeddings.values()
            ) / (1024 * 1024) if self.embeddings else 0.0
        }
















