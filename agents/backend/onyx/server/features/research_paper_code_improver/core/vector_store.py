"""
Vector Store - Almacenamiento vectorial para papers
====================================================
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from .core_utils import get_logger, ensure_dir
from .optional_imports import get_chromadb, get_sentence_transformers

logger = get_logger(__name__)

# Check optional imports
_chromadb = get_chromadb()
_sentence_transformers = get_sentence_transformers()

CHROMA_AVAILABLE = _chromadb is not None
SENTENCE_TRANSFORMERS_AVAILABLE = _sentence_transformers is not None


class VectorStore:
    """
    Almacenamiento vectorial para papers usando ChromaDB.
    Permite búsqueda semántica de papers relevantes.
    """
    
    def __init__(self, collection_name: str = "research_papers", persist_dir: str = "data/vector_db"):
        """
        Inicializar vector store.
        
        Args:
            collection_name: Nombre de la colección
            persist_dir: Directorio para persistencia
        """
        self.collection_name = collection_name
        self.persist_dir = ensure_dir(persist_dir)
        
        self.client = None
        self.collection = None
        self.embedder = None
        
        if CHROMA_AVAILABLE:
            self._initialize_chroma()
        else:
            logger.warning("ChromaDB no disponible, usando almacenamiento en memoria")
            self._documents = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedder inicializado: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"No se pudo cargar embedder: {e}")
    
    def _initialize_chroma(self):
        """Inicializar ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                persist_directory=str(self.persist_dir),
                anonymized_telemetry=False
            ))
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Research papers for code improvement"}
            )
            
            logger.info(f"ChromaDB inicializado: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            self._documents = {}
    
    def add_paper(self, paper_id: str, paper_data: Dict[str, Any]) -> bool:
        """
        Agrega un paper al vector store.
        
        Args:
            paper_id: ID único del paper
            paper_data: Datos del paper
            
        Returns:
            True si se agregó exitosamente
        """
        try:
            # Generar embedding del contenido
            content = self._extract_searchable_content(paper_data)
            
            if not self.embedder:
                logger.warning("Embedder no disponible, no se puede indexar")
                return False
            
            embedding = self.embedder.encode(content).tolist()
            
            # Metadata
            metadata = {
                "title": paper_data.get("title", ""),
                "authors": ", ".join(paper_data.get("authors", [])),
                "abstract": paper_data.get("abstract", "")[:500],
                "source": paper_data.get("source", ""),
            }
            
            if self.collection:
                # Usar ChromaDB
                self.collection.add(
                    ids=[paper_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata]
                )
                logger.info(f"Paper agregado a ChromaDB: {paper_id}")
            else:
                # Almacenamiento en memoria
                self._documents[paper_id] = {
                    "embedding": embedding,
                    "content": content,
                    "metadata": metadata,
                    "paper_data": paper_data
                }
                logger.info(f"Paper agregado a memoria: {paper_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error agregando paper: {e}")
            return False
    
    def search_relevant_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca papers relevantes para una consulta.
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de papers relevantes con scores
        """
        try:
            if not self.embedder:
                logger.warning("Embedder no disponible")
                return []
            
            # Generar embedding de la consulta
            query_embedding = self.embedder.encode(query).tolist()
            
            if self.collection:
                # Búsqueda en ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                papers = []
                if results["ids"] and len(results["ids"][0]) > 0:
                    for i, paper_id in enumerate(results["ids"][0]):
                        papers.append({
                            "paper_id": paper_id,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "content": results["documents"][0][i] if results["documents"] else "",
                            "distance": results["distances"][0][i] if results["distances"] else 0.0
                        })
                
                return papers
            else:
                # Búsqueda en memoria
                return self._search_memory(query_embedding, top_k)
                
        except Exception as e:
            logger.error(f"Error buscando papers: {e}")
            return []
    
    def _search_memory(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Búsqueda en memoria usando similitud coseno"""
        import math
        
        results = []
        query_vec = np.array(query_embedding)
        
        for paper_id, doc in self._documents.items():
            doc_vec = np.array(doc["embedding"])
            
            # Similitud coseno
            dot_product = np.dot(query_vec, doc_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_doc = np.linalg.norm(doc_vec)
            
            if norm_query > 0 and norm_doc > 0:
                similarity = dot_product / (norm_query * norm_doc)
                results.append({
                    "paper_id": paper_id,
                    "metadata": doc["metadata"],
                    "content": doc["content"],
                    "distance": 1 - similarity  # Convertir a distancia
                })
        
        # Ordenar por distancia (menor es mejor)
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]
    
    def _extract_searchable_content(self, paper_data: Dict[str, Any]) -> str:
        """
        Extrae contenido buscable del paper.
        
        Args:
            paper_data: Datos del paper
            
        Returns:
            Contenido concatenado para búsqueda
        """
        parts = []
        
        # Título
        if paper_data.get("title"):
            parts.append(f"Title: {paper_data['title']}")
        
        # Abstract
        if paper_data.get("abstract"):
            parts.append(f"Abstract: {paper_data['abstract']}")
        
        # Secciones relevantes
        sections = paper_data.get("sections", [])
        for section in sections:
            if section.get("type") in ["methodology", "implementation", "results"]:
                parts.append(f"{section.get('title', '')}: {section.get('content', '')[:1000]}")
        
        # Contenido completo (limitado)
        if paper_data.get("content"):
            content = paper_data["content"][:2000]  # Limitar tamaño
            parts.append(content)
        
        return "\n\n".join(parts)
    
    def get_paper_count(self) -> int:
        """Obtiene el número de papers indexados"""
        if self.collection:
            return self.collection.count()
        else:
            return len(self._documents)
    
    def delete_paper(self, paper_id: str) -> bool:
        """Elimina un paper del vector store"""
        try:
            if self.collection:
                self.collection.delete(ids=[paper_id])
            else:
                if paper_id in self._documents:
                    del self._documents[paper_id]
            
            logger.info(f"Paper eliminado: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Error eliminando paper: {e}")
            return False




