"""
Elasticsearch Client - Cliente para búsquedas y análisis
========================================================

Soporta:
- Búsqueda full-text
- Agregaciones
- Análisis de datos
- Indexación de documentos
"""

import logging
from typing import Dict, Optional, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Cliente para Elasticsearch"""
    
    def __init__(self, hosts: List[str] = None, index_prefix: str = "3d_prototype_"):
        self.hosts = hosts or ["http://localhost:9200"]
        self.index_prefix = index_prefix
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura cliente de Elasticsearch"""
        try:
            from elasticsearch import Elasticsearch
            
            self.client = Elasticsearch(
                hosts=self.hosts,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Verificar conexión
            if self.client.ping():
                logger.info("Elasticsearch client configured and connected")
            else:
                logger.warning("Elasticsearch client configured but connection failed")
        except ImportError:
            logger.warning("elasticsearch not available. Install with: pip install elasticsearch")
        except Exception as e:
            logger.error(f"Failed to setup Elasticsearch: {e}")
    
    def create_index(self, index_name: str, mappings: Optional[Dict] = None) -> bool:
        """Crea un índice en Elasticsearch"""
        if not self.client:
            return False
        
        full_index_name = f"{self.index_prefix}{index_name}"
        
        try:
            if not self.client.indices.exists(index=full_index_name):
                body = {}
                if mappings:
                    body["mappings"] = mappings
                
                self.client.indices.create(index=full_index_name, body=body)
                logger.info(f"Index created: {full_index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def index_document(self, index_name: str, doc_id: str, document: Dict) -> bool:
        """Indexa un documento"""
        if not self.client:
            return False
        
        full_index_name = f"{self.index_prefix}{index_name}"
        
        try:
            document["indexed_at"] = datetime.utcnow().isoformat()
            self.client.index(
                index=full_index_name,
                id=doc_id,
                body=document
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return False
    
    def search(self, index_name: str, query: Dict, size: int = 10) -> List[Dict]:
        """Realiza una búsqueda"""
        if not self.client:
            return []
        
        full_index_name = f"{self.index_prefix}{index_name}"
        
        try:
            response = self.client.search(
                index=full_index_name,
                body={"query": query},
                size=size
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_full_text(self, index_name: str, text: str, fields: List[str] = None) -> List[Dict]:
        """Búsqueda full-text"""
        if not self.client:
            return []
        
        query = {
            "multi_match": {
                "query": text,
                "fields": fields or ["_all"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        return self.search(index_name, query)
    
    def aggregate(self, index_name: str, aggregations: Dict) -> Dict:
        """Realiza agregaciones"""
        if not self.client:
            return {}
        
        full_index_name = f"{self.index_prefix}{index_name}"
        
        try:
            response = self.client.search(
                index=full_index_name,
                body={
                    "size": 0,
                    "aggs": aggregations
                }
            )
            
            return response.get("aggregations", {})
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {}
    
    def delete_document(self, index_name: str, doc_id: str) -> bool:
        """Elimina un documento"""
        if not self.client:
            return False
        
        full_index_name = f"{self.index_prefix}{index_name}"
        
        try:
            self.client.delete(index=full_index_name, id=doc_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def bulk_index(self, index_name: str, documents: List[Dict]) -> bool:
        """Indexa múltiples documentos en bulk"""
        if not self.client:
            return False
        
        full_index_name = f"{self.index_prefix}{index_name}"
        
        try:
            from elasticsearch.helpers import bulk
            
            actions = [
                {
                    "_index": full_index_name,
                    "_id": doc.get("id"),
                    "_source": doc
                }
                for doc in documents
            ]
            
            bulk(self.client, actions)
            return True
        except Exception as e:
            logger.error(f"Bulk index failed: {e}")
            return False


# Instancia global
elasticsearch_client = ElasticsearchClient()




