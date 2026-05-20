"""
Elasticsearch Client - Búsqueda y análisis
===========================================

Cliente para Elasticsearch para búsquedas avanzadas y análisis.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Elasticsearch client (lazy initialization)
_es_client: Optional[Any] = None


def get_elasticsearch_client() -> Optional[Any]:
    """Obtener cliente de Elasticsearch."""
    global _es_client
    
    if _es_client is not None:
        return _es_client
    
    try:
        from elasticsearch import Elasticsearch
        from elasticsearch.helpers import bulk
        
        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        es_port = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
        es_url = os.getenv("ELASTICSEARCH_URL", f"http://{es_host}:{es_port}")
        
        _es_client = Elasticsearch(
            [es_url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        # Test connection
        if _es_client.ping():
            logger.info(f"Elasticsearch connected to {es_url}")
        else:
            logger.warning("Elasticsearch connection failed")
            _es_client = None
        
        return _es_client
    
    except ImportError:
        logger.warning("elasticsearch not installed, search disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to connect to Elasticsearch: {e}")
        return None


class ElasticsearchService:
    """Servicio para operaciones con Elasticsearch."""
    
    def __init__(self, index_prefix: str = "cursor-agent"):
        """
        Inicializar servicio.
        
        Args:
            index_prefix: Prefijo para índices.
        """
        self.index_prefix = index_prefix
        self.client = get_elasticsearch_client()
    
    def create_index(self, index_name: str, mapping: Optional[Dict[str, Any]] = None) -> bool:
        """
        Crear índice.
        
        Args:
            index_name: Nombre del índice.
            mapping: Mapping del índice.
        
        Returns:
            True si se creó exitosamente.
        """
        if not self.client:
            return False
        
        full_index = f"{self.index_prefix}-{index_name}"
        
        try:
            if not self.client.indices.exists(index=full_index):
                body = {}
                if mapping:
                    body["mappings"] = mapping
                
                self.client.indices.create(index=full_index, body=body)
                logger.info(f"Index {full_index} created")
            return True
        except Exception as e:
            logger.error(f"Failed to create index {full_index}: {e}")
            return False
    
    def index_document(self, index_name: str, document: Dict[str, Any], doc_id: Optional[str] = None) -> bool:
        """
        Indexar documento.
        
        Args:
            index_name: Nombre del índice.
            document: Documento a indexar.
            doc_id: ID del documento (opcional).
        
        Returns:
            True si se indexó exitosamente.
        """
        if not self.client:
            return False
        
        full_index = f"{self.index_prefix}-{index_name}"
        
        try:
            self.client.index(
                index=full_index,
                id=doc_id,
                document=document
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return False
    
    def search(self, index_name: str, query: Dict[str, Any], size: int = 10) -> List[Dict[str, Any]]:
        """
        Buscar documentos.
        
        Args:
            index_name: Nombre del índice.
            query: Query de búsqueda.
            size: Número de resultados.
        
        Returns:
            Lista de documentos encontrados.
        """
        if not self.client:
            return []
        
        full_index = f"{self.index_prefix}-{index_name}"
        
        try:
            response = self.client.search(
                index=full_index,
                body={"query": query},
                size=size
            )
            
            return [
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"]
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_tasks(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Buscar tareas.
        
        Args:
            query: Texto de búsqueda.
            filters: Filtros adicionales.
        
        Returns:
            Lista de tareas encontradas.
        """
        search_query = {
            "bool": {
                "should": [
                    {"match": {"command": {"query": query, "boost": 2}}},
                    {"match": {"result": {"query": query}}},
                ],
                "must": []
            }
        }
        
        if filters:
            for key, value in filters.items():
                search_query["bool"]["must"].append({"term": {key: value}})
        
        return self.search("tasks", search_query)
    
    def delete_document(self, index_name: str, doc_id: str) -> bool:
        """Eliminar documento."""
        if not self.client:
            return False
        
        full_index = f"{self.index_prefix}-{index_name}"
        
        try:
            self.client.delete(index=full_index, id=doc_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False


def get_elasticsearch_service() -> ElasticsearchService:
    """Obtener servicio de Elasticsearch."""
    return ElasticsearchService()




