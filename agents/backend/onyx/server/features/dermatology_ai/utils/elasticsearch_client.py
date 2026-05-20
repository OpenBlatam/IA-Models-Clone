"""
Elasticsearch Client for Search and Analytics
Optimized for read-heavy workloads and full-text search
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from elasticsearch import AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    logger.warning("Elasticsearch not available. Install with: pip install elasticsearch")


class ElasticsearchClient:
    """
    Elasticsearch client for search and analytics.
    Optimized for read-heavy workloads.
    """
    
    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.hosts = hosts or [os.getenv("ELASTICSEARCH_HOST", "localhost:9200")]
        self.cloud_id = cloud_id or os.getenv("ELASTICSEARCH_CLOUD_ID")
        self.api_key = api_key or os.getenv("ELASTICSEARCH_API_KEY")
        self.username = username or os.getenv("ELASTICSEARCH_USERNAME")
        self.password = password or os.getenv("ELASTICSEARCH_PASSWORD")
        self.client: Optional[AsyncElasticsearch] = None
        self.connected = False
    
    async def connect(self):
        """Connect to Elasticsearch"""
        if not ELASTICSEARCH_AVAILABLE:
            logger.warning("Elasticsearch library not installed")
            return
        
        try:
            # Build connection parameters
            if self.cloud_id:
                # Elastic Cloud
                self.client = AsyncElasticsearch(
                    cloud_id=self.cloud_id,
                    api_key=self.api_key,
                    request_timeout=30
                )
            else:
                # Self-hosted
                http_auth = None
                if self.username and self.password:
                    http_auth = (self.username, self.password)
                
                self.client = AsyncElasticsearch(
                    hosts=self.hosts,
                    http_auth=http_auth,
                    api_key=self.api_key,
                    request_timeout=30
                )
            
            # Test connection
            await self.client.ping()
            self.connected = True
            logger.info("✅ Connected to Elasticsearch")
            
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Elasticsearch"""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("Disconnected from Elasticsearch")
    
    async def create_index(
        self,
        index_name: str,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None
    ):
        """Create Elasticsearch index"""
        if not self.connected or not self.client:
            await self.connect()
        
        if await self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists")
            return
        
        body = {}
        if mappings:
            body["mappings"] = mappings
        if settings:
            body["settings"] = settings
        
        await self.client.indices.create(index=index_name, body=body)
        logger.info(f"Created index: {index_name}")
    
    async def index_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """Index a document"""
        if not self.connected or not self.client:
            await self.connect()
        
        response = await self.client.index(
            index=index_name,
            id=document_id,
            document=document
        )
        return response["_id"]
    
    async def bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]]
    ) -> int:
        """Bulk index documents"""
        if not self.connected or not self.client:
            await self.connect()
        
        actions = [
            {
                "_index": index_name,
                "_id": doc.get("id"),
                "_source": doc
            }
            for doc in documents
        ]
        
        success, failed = await async_bulk(self.client, actions)
        logger.info(f"Bulk indexed {success} documents, {len(failed)} failed")
        return success
    
    async def search(
        self,
        index_name: str,
        query: Dict[str, Any],
        size: int = 10,
        from_: int = 0
    ) -> Dict[str, Any]:
        """Search documents"""
        if not self.connected or not self.client:
            await self.connect()
        
        response = await self.client.search(
            index=index_name,
            body={"query": query},
            size=size,
            from_=from_
        )
        
        return {
            "total": response["hits"]["total"]["value"],
            "hits": [hit["_source"] for hit in response["hits"]["hits"]],
            "took": response["took"]
        }
    
    async def full_text_search(
        self,
        index_name: str,
        text: str,
        fields: List[str] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """Full-text search"""
        fields = fields or ["_all"]
        
        query = {
            "multi_match": {
                "query": text,
                "fields": fields,
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        return await self.search(index_name, query, size=size)
    
    async def delete_document(
        self,
        index_name: str,
        document_id: str
    ) -> bool:
        """Delete a document"""
        if not self.connected or not self.client:
            await self.connect()
        
        try:
            await self.client.delete(index=index_name, id=document_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    async def get_document(
        self,
        index_name: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        if not self.connected or not self.client:
            await self.connect()
        
        try:
            response = await self.client.get(index=index_name, id=document_id)
            return response["_source"]
        except Exception:
            return None
    
    async def update_document(
        self,
        index_name: str,
        document_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a document"""
        if not self.connected or not self.client:
            await self.connect()
        
        try:
            await self.client.update(
                index=index_name,
                id=document_id,
                body={"doc": updates}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False


def get_elasticsearch_client() -> Optional[ElasticsearchClient]:
    """Get Elasticsearch client from environment"""
    if not ELASTICSEARCH_AVAILABLE:
        return None
    
    return ElasticsearchClient(
        hosts=os.getenv("ELASTICSEARCH_HOSTS", "localhost:9200").split(","),
        cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        username=os.getenv("ELASTICSEARCH_USERNAME"),
        password=os.getenv("ELASTICSEARCH_PASSWORD"),
    )















