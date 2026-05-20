"""
Search Engine Integration - Integración con motores de búsqueda
==============================================================

Integración con motores de búsqueda para:
- Elasticsearch
- Full-text search
- Analytics
- Aggregations
"""

import logging
from typing import Optional, Dict, Any, List, Protocol
from abc import ABC, abstractmethod

from .types import DatabaseKey, DatabaseValue, JSONDict

logger = logging.getLogger(__name__)


class SearchEngine(ABC):
    """Interfaz para motores de búsqueda"""
    
    @abstractmethod
    async def index(
        self,
        index_name: str,
        document_id: str,
        document: Dict[str, Any]
    ) -> bool: ...
    
    @abstractmethod
    async def search(
        self,
        index_name: str,
        query: Dict[str, Any],
        size: int = 10,
        from_: int = 0
    ) -> List[Dict[str, Any]]: ...
    
    @abstractmethod
    async def delete(
        self,
        index_name: str,
        document_id: str
    ) -> bool: ...
    
    @abstractmethod
    async def bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]]
    ) -> bool: ...


class ElasticsearchClient(SearchEngine):
    """Cliente para Elasticsearch"""
    
    def __init__(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False
    ) -> None:
        self.hosts = hosts
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self._client: Optional[Any] = None
    
    def _get_client(self) -> Any:
        """Obtiene cliente de Elasticsearch"""
        if self._client is None:
            try:
                from elasticsearch import AsyncElasticsearch
                
                client_config = {
                    "hosts": self.hosts,
                    "use_ssl": self.use_ssl
                }
                
                if self.username and self.password:
                    client_config["http_auth"] = (self.username, self.password)
                
                self._client = AsyncElasticsearch(**client_config)
            except ImportError:
                logger.error(
                    "elasticsearch not available. "
                    "Install with: pip install elasticsearch[async]"
                )
                raise
        return self._client
    
    async def index(
        self,
        index_name: str,
        document_id: str,
        document: Dict[str, Any]
    ) -> bool:
        """Indexa un documento"""
        try:
            client = self._get_client()
            response = await client.index(
                index=index_name,
                id=document_id,
                document=document
            )
            return response.get("result") in ["created", "updated"]
        except Exception as e:
            logger.error(f"Elasticsearch index error: {e}")
            return False
    
    async def search(
        self,
        index_name: str,
        query: Dict[str, Any],
        size: int = 10,
        from_: int = 0
    ) -> List[Dict[str, Any]]:
        """Busca documentos"""
        try:
            client = self._get_client()
            response = await client.search(
                index=index_name,
                body={"query": query},
                size=size,
                from_=from_
            )
            
            hits = response.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}")
            return []
    
    async def delete(
        self,
        index_name: str,
        document_id: str
    ) -> bool:
        """Elimina un documento"""
        try:
            client = self._get_client()
            response = await client.delete(
                index=index_name,
                id=document_id
            )
            return response.get("result") == "deleted"
        except Exception as e:
            logger.error(f"Elasticsearch delete error: {e}")
            return False
    
    async def bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Indexa múltiples documentos"""
        try:
            client = self._get_client()
            
            actions = []
            for doc in documents:
                doc_id = doc.get("id") or doc.get("_id")
                source = {k: v for k, v in doc.items() if k not in ["id", "_id"]}
                
                action = {
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": source
                }
                actions.append(action)
            
            from elasticsearch.helpers import async_bulk
            success, failed = await async_bulk(client, actions)
            
            return len(failed) == 0
        except Exception as e:
            logger.error(f"Elasticsearch bulk index error: {e}")
            return False
    
    async def create_index(
        self,
        index_name: str,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Crea un índice"""
        try:
            client = self._get_client()
            
            body: Dict[str, Any] = {}
            if mappings:
                body["mappings"] = mappings
            if settings:
                body["settings"] = settings
            
            await client.indices.create(index=index_name, body=body)
            return True
        except Exception as e:
            logger.error(f"Elasticsearch create index error: {e}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Elimina un índice"""
        try:
            client = self._get_client()
            await client.indices.delete(index=index_name)
            return True
        except Exception as e:
            logger.error(f"Elasticsearch delete index error: {e}")
            return False
    
    async def get_aggregations(
        self,
        index_name: str,
        query: Dict[str, Any],
        aggregations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Obtiene agregaciones"""
        try:
            client = self._get_client()
            response = await client.search(
                index=index_name,
                body={
                    "query": query,
                    "aggs": aggregations
                },
                size=0
            )
            return response.get("aggregations", {})
        except Exception as e:
            logger.error(f"Elasticsearch aggregations error: {e}")
            return {}


def get_search_engine(
    engine_type: str = "elasticsearch",
    **kwargs: Any
) -> Optional[SearchEngine]:
    """
    Obtiene cliente de motor de búsqueda.
    
    Args:
        engine_type: Tipo de motor (elasticsearch)
        **kwargs: Configuración específica
    
    Returns:
        Cliente de motor de búsqueda
    """
    if engine_type.lower() == "elasticsearch":
        hosts = kwargs.get("hosts", ["localhost:9200"])
        if isinstance(hosts, str):
            hosts = [hosts]
        return ElasticsearchClient(
            hosts=hosts,
            username=kwargs.get("username"),
            password=kwargs.get("password"),
            use_ssl=kwargs.get("use_ssl", False)
        )
    else:
        logger.warning(f"Search engine type {engine_type} not implemented")
        return None















