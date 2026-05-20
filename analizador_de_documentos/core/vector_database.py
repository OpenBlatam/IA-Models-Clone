"""
Integración con Bases de Datos Vectoriales
===========================================

Sistema para integrar con bases de datos vectoriales para búsqueda escalable.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Documento vectorial"""
    id: str
    embedding: np.ndarray
    content: str
    metadata: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class VectorDatabase:
    """
    Interfaz para bases de datos vectoriales
    
    Soporta múltiples backends:
    - Pinecone
    - Weaviate
    - Chroma
    - Qdrant
    - Milvus
    - In-memory (fallback)
    """
    
    def __init__(self, backend: str = "memory", **backend_config):
        """
        Inicializar base de datos vectorial
        
        Args:
            backend: Backend a usar (pinecone, weaviate, chroma, qdrant, milvus, memory)
            **backend_config: Configuración del backend
        """
        self.backend = backend
        self.db = None
        self._initialize_backend(backend_config)
        logger.info(f"VectorDatabase inicializado con backend: {backend}")
    
    def _initialize_backend(self, config: Dict[str, Any]):
        """Inicializar backend específico"""
        if self.backend == "pinecone":
            self._init_pinecone(config)
        elif self.backend == "weaviate":
            self._init_weaviate(config)
        elif self.backend == "chroma":
            self._init_chroma(config)
        elif self.backend == "qdrant":
            self._init_qdrant(config)
        elif self.backend == "milvus":
            self._init_milvus(config)
        else:
            # Memory fallback
            self._init_memory()
    
    def _init_pinecone(self, config: Dict[str, Any]):
        """Inicializar Pinecone"""
        try:
            import pinecone
            api_key = config.get("api_key") or os.getenv("PINECONE_API_KEY")
            environment = config.get("environment") or os.getenv("PINECONE_ENVIRONMENT")
            
            pinecone.init(api_key=api_key, environment=environment)
            index_name = config.get("index_name", "document-analyzer")
            self.db = pinecone.Index(index_name)
            logger.info("Pinecone inicializado")
        except ImportError:
            logger.warning("Pinecone no instalado, usando memoria")
            self._init_memory()
        except Exception as e:
            logger.warning(f"Error inicializando Pinecone: {e}, usando memoria")
            self._init_memory()
    
    def _init_weaviate(self, config: Dict[str, Any]):
        """Inicializar Weaviate"""
        try:
            import weaviate
            url = config.get("url", "http://localhost:8080")
            self.db = weaviate.Client(url)
            logger.info("Weaviate inicializado")
        except ImportError:
            logger.warning("Weaviate no instalado, usando memoria")
            self._init_memory()
        except Exception as e:
            logger.warning(f"Error inicializando Weaviate: {e}, usando memoria")
            self._init_memory()
    
    def _init_chroma(self, config: Dict[str, Any]):
        """Inicializar Chroma"""
        try:
            import chromadb
            persist_directory = config.get("persist_directory", "./chroma_db")
            self.client = chromadb.Client()
            self.db = self.client.create_collection(
                name=config.get("collection_name", "documents"),
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Chroma inicializado")
        except ImportError:
            logger.warning("Chroma no instalado, usando memoria")
            self._init_memory()
        except Exception as e:
            logger.warning(f"Error inicializando Chroma: {e}, usando memoria")
            self._init_memory()
    
    def _init_qdrant(self, config: Dict[str, Any]):
        """Inicializar Qdrant"""
        try:
            from qdrant_client import QdrantClient
            url = config.get("url", "http://localhost:6333")
            self.client = QdrantClient(url=url)
            collection_name = config.get("collection_name", "documents")
            self.db = collection_name
            logger.info("Qdrant inicializado")
        except ImportError:
            logger.warning("Qdrant no instalado, usando memoria")
            self._init_memory()
        except Exception as e:
            logger.warning(f"Error inicializando Qdrant: {e}, usando memoria")
            self._init_memory()
    
    def _init_milvus(self, config: Dict[str, Any]):
        """Inicializar Milvus"""
        try:
            from pymilvus import connections, Collection
            host = config.get("host", "localhost")
            port = config.get("port", 19530)
            connections.connect(host=host, port=port)
            collection_name = config.get("collection_name", "documents")
            self.db = Collection(collection_name)
            logger.info("Milvus inicializado")
        except ImportError:
            logger.warning("Milvus no instalado, usando memoria")
            self._init_memory()
        except Exception as e:
            logger.warning(f"Error inicializando Milvus: {e}, usando memoria")
            self._init_memory()
    
    def _init_memory(self):
        """Inicializar almacenamiento en memoria"""
        self.documents: Dict[str, VectorDocument] = {}
        self.db = "memory"
        logger.info("Usando almacenamiento en memoria")
    
    async def add_document(
        self,
        document_id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Agregar documento a la base de datos"""
        if self.db == "memory":
            self.documents[document_id] = VectorDocument(
                id=document_id,
                embedding=embedding,
                content=content,
                metadata=metadata or {}
            )
        elif self.backend == "pinecone":
            self.db.upsert([(document_id, embedding.tolist(), metadata)])
        elif self.backend == "chroma":
            self.db.add(
                ids=[document_id],
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[metadata or {}]
            )
        # Otros backends seguirían aquí
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Buscar documentos similares"""
        if self.db == "memory":
            return self._search_memory(query_embedding, top_k, filters)
        elif self.backend == "pinecone":
            results = self.db.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        elif self.backend == "chroma":
            results = self.db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            return results
        return []
    
    def _search_memory(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Búsqueda en memoria"""
        results = []
        for doc_id, doc in self.documents.items():
            # Aplicar filtros
            if filters:
                match = True
                for key, value in filters.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calcular similitud
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            
            results.append({
                "id": doc_id,
                "score": float(similarity),
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        # Ordenar y retornar top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def delete_document(self, document_id: str):
        """Eliminar documento"""
        if self.db == "memory":
            if document_id in self.documents:
                del self.documents[document_id]
        elif self.backend == "pinecone":
            self.db.delete(ids=[document_id])
        elif self.backend == "chroma":
            self.db.delete(ids=[document_id])


# Importar os
import os
















