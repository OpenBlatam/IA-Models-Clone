"""
Search Service
==============

Advanced search service with full-text search, faceted search, and AI-powered search.
"""

from __future__ import annotations
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import elasticsearch
from elasticsearch import AsyncElasticsearch
import whoosh
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC, BOOLEAN
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import *
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)


class SearchBackend(str, Enum):
    """Search backend enumeration"""
    ELASTICSEARCH = "elasticsearch"
    WHOOSH = "whoosh"
    SOLR = "solr"
    LUCENE = "lucene"


class SearchType(str, Enum):
    """Search type enumeration"""
    FULL_TEXT = "full_text"
    FACETED = "faceted"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    VECTOR = "vector"
    BOOLEAN = "boolean"


class SearchOperator(str, Enum):
    """Search operator enumeration"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class SearchConfig:
    """Search configuration"""
    backend: SearchBackend = SearchBackend.ELASTICSEARCH
    elasticsearch_url: str = "http://localhost:9200"
    whoosh_index_path: str = "./search_index"
    solr_url: str = "http://localhost:8983/solr"
    max_results: int = 100
    default_operator: SearchOperator = SearchOperator.AND
    enable_highlighting: bool = True
    enable_suggestions: bool = True
    enable_facets: bool = True
    enable_ai_search: bool = True
    ai_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    language: str = "en"
    stop_words: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Search result representation"""
    id: str
    title: str
    content: str
    score: float
    highlights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    facets: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Search query representation"""
    query: str
    search_type: SearchType = SearchType.FULL_TEXT
    filters: Dict[str, Any] = field(default_factory=dict)
    facets: List[str] = field(default_factory=list)
    sort: List[Tuple[str, str]] = field(default_factory=list)
    limit: int = 100
    offset: int = 0
    operator: SearchOperator = SearchOperator.AND
    fuzzy: bool = False
    highlight: bool = True


@dataclass
class SearchIndex:
    """Search index representation"""
    name: str
    fields: Dict[str, str]
    settings: Dict[str, Any] = field(default_factory=dict)
    mappings: Dict[str, Any] = field(default_factory=dict)


class SearchService:
    """Advanced search service with multiple backends"""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self._is_running = False
        self._search_backend = None
        self._ai_model = None
        self._initialize_backend()
        self._initialize_ai()
    
    def _initialize_backend(self):
        """Initialize search backend"""
        try:
            if self.config.backend == SearchBackend.ELASTICSEARCH:
                self._search_backend = ElasticsearchBackend(self.config)
            
            elif self.config.backend == SearchBackend.WHOOSH:
                self._search_backend = WhooshBackend(self.config)
            
            elif self.config.backend == SearchBackend.SOLR:
                self._search_backend = SolrBackend(self.config)
            
            else:
                raise ValueError(f"Unsupported search backend: {self.config.backend}")
            
            logger.info(f"Search backend initialized: {self.config.backend.value}")
        
        except Exception as e:
            logger.error(f"Failed to initialize search backend: {e}")
            raise
    
    def _initialize_ai(self):
        """Initialize AI model for semantic search"""
        try:
            if self.config.enable_ai_search:
                # In real implementation, load sentence transformer model
                logger.info("AI search model initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize AI model: {e}")
    
    async def start(self):
        """Start the search service"""
        if self._is_running:
            return
        
        try:
            await self._search_backend.initialize()
            self._is_running = True
            logger.info("Search service started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start search service: {e}")
            raise
    
    async def stop(self):
        """Stop the search service"""
        if not self._is_running:
            return
        
        try:
            await self._search_backend.cleanup()
            self._is_running = False
            logger.info("Search service stopped")
        
        except Exception as e:
            logger.error(f"Error stopping search service: {e}")
    
    @measure_performance
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search"""
        try:
            # Preprocess query
            processed_query = await self._preprocess_query(query)
            
            # Perform search based on type
            if query.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(processed_query)
            elif query.search_type == SearchType.VECTOR:
                results = await self._vector_search(processed_query)
            else:
                results = await self._search_backend.search(processed_query)
            
            # Post-process results
            processed_results = await self._postprocess_results(results, query)
            
            logger.info(f"Search completed: {len(processed_results)} results")
            return processed_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def index_document(
        self,
        index_name: str,
        document_id: str,
        document: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Index document"""
        try:
            # Preprocess document
            processed_document = await self._preprocess_document(document)
            
            # Add metadata
            if metadata:
                processed_document.update(metadata)
            
            # Index document
            await self._search_backend.index_document(index_name, document_id, processed_document)
            
            logger.info(f"Document indexed: {document_id}")
            return True
        
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return False
    
    async def delete_document(self, index_name: str, document_id: str) -> bool:
        """Delete document from index"""
        try:
            await self._search_backend.delete_document(index_name, document_id)
            logger.info(f"Document deleted: {document_id}")
            return True
        
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
    
    async def create_index(self, index_config: SearchIndex) -> bool:
        """Create search index"""
        try:
            await self._search_backend.create_index(index_config)
            logger.info(f"Index created: {index_config.name}")
            return True
        
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete search index"""
        try:
            await self._search_backend.delete_index(index_name)
            logger.info(f"Index deleted: {index_name}")
            return True
        
        except Exception as e:
            logger.error(f"Index deletion failed: {e}")
            return False
    
    async def get_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """Get search suggestions"""
        try:
            if self.config.enable_suggestions:
                return await self._search_backend.get_suggestions(query, limit)
            return []
        
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    async def get_facets(self, index_name: str, query: str, facet_fields: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """Get search facets"""
        try:
            if self.config.enable_facets:
                return await self._search_backend.get_facets(index_name, query, facet_fields)
            return {}
        
        except Exception as e:
            logger.error(f"Failed to get facets: {e}")
            return {}
    
    async def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess search query"""
        # Tokenize and clean query
        if self.config.language == "en":
            tokens = word_tokenize(query.query.lower())
            # Remove stop words
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            # Stem tokens
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
            query.query = " ".join(tokens)
        
        elif self.config.language == "zh":
            # Chinese tokenization
            tokens = list(jieba.cut(query.query))
            query.query = " ".join(tokens)
        
        return query
    
    async def _preprocess_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess document for indexing"""
        processed_doc = {}
        
        for field, value in document.items():
            if isinstance(value, str):
                # Clean and normalize text
                processed_doc[field] = value.strip().lower()
            else:
                processed_doc[field] = value
        
        return processed_doc
    
    async def _postprocess_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Post-process search results"""
        # Add highlights if requested
        if query.highlight and self.config.enable_highlighting:
            for result in results:
                result.highlights = await self._generate_highlights(result, query.query)
        
        # Add facets if requested
        if query.facets and self.config.enable_facets:
            for result in results:
                result.facets = await self.get_facets("default", query.query, query.facets)
        
        return results
    
    async def _generate_highlights(self, result: SearchResult, query: str) -> List[str]:
        """Generate highlights for search result"""
        # Simple highlighting implementation
        highlights = []
        query_terms = query.lower().split()
        
        for term in query_terms:
            if term in result.content.lower():
                # Find context around the term
                start = result.content.lower().find(term)
                if start != -1:
                    context_start = max(0, start - 50)
                    context_end = min(len(result.content), start + len(term) + 50)
                    context = result.content[context_start:context_end]
                    highlights.append(f"...{context}...")
        
        return highlights
    
    async def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search using AI"""
        try:
            # Generate embeddings for query
            query_embedding = await self._generate_embedding(query.query)
            
            # Search for similar documents
            results = await self._search_backend.vector_search(query_embedding, query.limit)
            
            return results
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _vector_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform vector search"""
        try:
            # Generate embeddings for query
            query_embedding = await self._generate_embedding(query.query)
            
            # Search for similar documents
            results = await self._search_backend.vector_search(query_embedding, query.limit)
            
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate text embedding"""
        # In real implementation, use sentence transformer model
        # For now, return dummy embedding
        return [0.1] * 384  # Typical embedding dimension
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        try:
            return await self._search_backend.get_stats()
        
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}


class ElasticsearchBackend:
    """Elasticsearch search backend"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.client = AsyncElasticsearch([config.elasticsearch_url])
    
    async def initialize(self):
        """Initialize Elasticsearch backend"""
        try:
            # Test connection
            await self.client.ping()
            logger.info("Elasticsearch backend initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup Elasticsearch backend"""
        await self.client.close()
        logger.info("Elasticsearch backend cleanup completed")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform Elasticsearch search"""
        try:
            # Build Elasticsearch query
            es_query = self._build_elasticsearch_query(query)
            
            # Execute search
            response = await self.client.search(
                index="*",
                body=es_query,
                size=query.limit,
                from_=query.offset
            )
            
            # Convert results
            results = []
            for hit in response['hits']['hits']:
                result = SearchResult(
                    id=hit['_id'],
                    title=hit['_source'].get('title', ''),
                    content=hit['_source'].get('content', ''),
                    score=hit['_score'],
                    metadata=hit['_source']
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def _build_elasticsearch_query(self, query: SearchQuery) -> Dict[str, Any]:
        """Build Elasticsearch query"""
        if query.search_type == SearchType.FULL_TEXT:
            es_query = {
                "query": {
                    "multi_match": {
                        "query": query.query,
                        "fields": ["title^2", "content"],
                        "operator": query.operator.value
                    }
                }
            }
        elif query.search_type == SearchType.FUZZY:
            es_query = {
                "query": {
                    "fuzzy": {
                        "content": {
                            "value": query.query,
                            "fuzziness": "AUTO"
                        }
                    }
                }
            }
        else:
            es_query = {
                "query": {
                    "match": {
                        "content": query.query
                    }
                }
            }
        
        # Add filters
        if query.filters:
            es_query["query"] = {
                "bool": {
                    "must": [es_query["query"]],
                    "filter": []
                }
            }
            
            for field, value in query.filters.items():
                es_query["query"]["bool"]["filter"].append({
                    "term": {field: value}
                })
        
        # Add sorting
        if query.sort:
            es_query["sort"] = [
                {field: {"order": order}} for field, order in query.sort
            ]
        
        return es_query
    
    async def index_document(self, index_name: str, document_id: str, document: Dict[str, Any]):
        """Index document in Elasticsearch"""
        await self.client.index(
            index=index_name,
            id=document_id,
            body=document
        )
    
    async def delete_document(self, index_name: str, document_id: str):
        """Delete document from Elasticsearch"""
        await self.client.delete(
            index=index_name,
            id=document_id
        )
    
    async def create_index(self, index_config: SearchIndex):
        """Create Elasticsearch index"""
        await self.client.indices.create(
            index=index_config.name,
            body={
                "settings": index_config.settings,
                "mappings": index_config.mappings
            }
        )
    
    async def delete_index(self, index_name: str):
        """Delete Elasticsearch index"""
        await self.client.indices.delete(index=index_name)
    
    async def get_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """Get search suggestions from Elasticsearch"""
        try:
            response = await self.client.search(
                index="*",
                body={
                    "suggest": {
                        "suggestion": {
                            "prefix": query,
                            "completion": {
                                "field": "suggest"
                            }
                        }
                    }
                },
                size=limit
            )
            
            suggestions = []
            for suggestion in response['suggest']['suggestion'][0]['options']:
                suggestions.append(suggestion['text'])
            
            return suggestions
        
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    async def get_facets(self, index_name: str, query: str, facet_fields: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """Get search facets from Elasticsearch"""
        try:
            es_query = {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "aggs": {}
            }
            
            for field in facet_fields:
                es_query["aggs"][field] = {
                    "terms": {
                        "field": field,
                        "size": 10
                    }
                }
            
            response = await self.client.search(
                index=index_name,
                body=es_query,
                size=0
            )
            
            facets = {}
            for field in facet_fields:
                if field in response['aggregations']:
                    facets[field] = [
                        (bucket['key'], bucket['doc_count'])
                        for bucket in response['aggregations'][field]['buckets']
                    ]
            
            return facets
        
        except Exception as e:
            logger.error(f"Failed to get facets: {e}")
            return {}
    
    async def vector_search(self, embedding: List[float], limit: int = 100) -> List[SearchResult]:
        """Perform vector search in Elasticsearch"""
        try:
            response = await self.client.search(
                index="*",
                body={
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": embedding}
                            }
                        }
                    }
                },
                size=limit
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = SearchResult(
                    id=hit['_id'],
                    title=hit['_source'].get('title', ''),
                    content=hit['_source'].get('content', ''),
                    score=hit['_score'],
                    metadata=hit['_source']
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Elasticsearch statistics"""
        try:
            stats = await self.client.indices.stats()
            return {
                "backend": "elasticsearch",
                "indices": len(stats['indices']),
                "total_documents": sum(
                    index_stats['total']['docs']['count']
                    for index_stats in stats['indices'].values()
                ),
                "total_size": sum(
                    index_stats['total']['store']['size_in_bytes']
                    for index_stats in stats['indices'].values()
                )
            }
        
        except Exception as e:
            logger.error(f"Failed to get Elasticsearch stats: {e}")
            return {}


class WhooshBackend:
    """Whoosh search backend"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.index_path = Path(config.whoosh_index_path)
        self.index = None
    
    async def initialize(self):
        """Initialize Whoosh backend"""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Create schema
            schema = Schema(
                id=ID(stored=True, unique=True),
                title=TEXT(stored=True),
                content=TEXT(stored=True),
                created_at=DATETIME(stored=True),
                metadata=TEXT(stored=True)
            )
            
            # Create or open index
            if not index.exists_in(str(self.index_path)):
                self.index = index.create_in(str(self.index_path), schema)
            else:
                self.index = index.open_dir(str(self.index_path))
            
            logger.info("Whoosh backend initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize Whoosh: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup Whoosh backend"""
        if self.index:
            self.index.close()
        logger.info("Whoosh backend cleanup completed")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform Whoosh search"""
        try:
            with self.index.searcher() as searcher:
                # Build query
                if query.search_type == SearchType.FULL_TEXT:
                    whoosh_query = MultifieldParser(
                        ["title", "content"],
                        self.index.schema
                    ).parse(query.query)
                else:
                    whoosh_query = QueryParser(
                        "content",
                        self.index.schema
                    ).parse(query.query)
                
                # Execute search
                results = searcher.search(
                    whoosh_query,
                    limit=query.limit
                )
                
                # Convert results
                search_results = []
                for hit in results:
                    result = SearchResult(
                        id=hit['id'],
                        title=hit.get('title', ''),
                        content=hit.get('content', ''),
                        score=hit.score,
                        metadata=json.loads(hit.get('metadata', '{}'))
                    )
                    search_results.append(result)
                
                return search_results
        
        except Exception as e:
            logger.error(f"Whoosh search failed: {e}")
            return []
    
    async def index_document(self, index_name: str, document_id: str, document: Dict[str, Any]):
        """Index document in Whoosh"""
        with self.index.writer() as writer:
            writer.update_document(
                id=document_id,
                title=document.get('title', ''),
                content=document.get('content', ''),
                created_at=DateTimeHelpers.now_utc(),
                metadata=json.dumps(document)
            )
    
    async def delete_document(self, index_name: str, document_id: str):
        """Delete document from Whoosh"""
        with self.index.writer() as writer:
            writer.delete_by_term('id', document_id)
    
    async def create_index(self, index_config: SearchIndex):
        """Create Whoosh index"""
        # Whoosh indexes are created automatically
        pass
    
    async def delete_index(self, index_name: str):
        """Delete Whoosh index"""
        import shutil
        shutil.rmtree(str(self.index_path))
    
    async def get_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """Get search suggestions from Whoosh"""
        # Whoosh doesn't have built-in suggestions
        return []
    
    async def get_facets(self, index_name: str, query: str, facet_fields: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """Get search facets from Whoosh"""
        # Whoosh doesn't have built-in faceting
        return {}
    
    async def vector_search(self, embedding: List[float], limit: int = 100) -> List[SearchResult]:
        """Perform vector search in Whoosh"""
        # Whoosh doesn't support vector search natively
        return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Whoosh statistics"""
        try:
            with self.index.searcher() as searcher:
                return {
                    "backend": "whoosh",
                    "total_documents": searcher.doc_count(),
                    "index_path": str(self.index_path)
                }
        
        except Exception as e:
            logger.error(f"Failed to get Whoosh stats: {e}")
            return {}


class SolrBackend:
    """Apache Solr search backend"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.solr_url = config.solr_url
    
    async def initialize(self):
        """Initialize Solr backend"""
        logger.info("Solr backend initialized")
    
    async def cleanup(self):
        """Cleanup Solr backend"""
        logger.info("Solr backend cleanup completed")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform Solr search"""
        # In real implementation, use pysolr or similar
        return []
    
    async def index_document(self, index_name: str, document_id: str, document: Dict[str, Any]):
        """Index document in Solr"""
        pass
    
    async def delete_document(self, index_name: str, document_id: str):
        """Delete document from Solr"""
        pass
    
    async def create_index(self, index_config: SearchIndex):
        """Create Solr index"""
        pass
    
    async def delete_index(self, index_name: str):
        """Delete Solr index"""
        pass
    
    async def get_suggestions(self, query: str, limit: int = 10) -> List[str]:
        """Get search suggestions from Solr"""
        return []
    
    async def get_facets(self, index_name: str, query: str, facet_fields: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """Get search facets from Solr"""
        return {}
    
    async def vector_search(self, embedding: List[float], limit: int = 100) -> List[SearchResult]:
        """Perform vector search in Solr"""
        return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Solr statistics"""
        return {
            "backend": "solr",
            "solr_url": self.solr_url
        }


# Global search service
search_service = SearchService()


# Utility functions
async def start_search_service():
    """Start the search service"""
    await search_service.start()


async def stop_search_service():
    """Stop the search service"""
    await search_service.stop()


async def search_documents(query: str, search_type: SearchType = SearchType.FULL_TEXT, limit: int = 100) -> List[SearchResult]:
    """Search documents"""
    search_query = SearchQuery(
        query=query,
        search_type=search_type,
        limit=limit
    )
    return await search_service.search(search_query)


async def index_document(
    index_name: str,
    document_id: str,
    document: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Index document"""
    return await search_service.index_document(index_name, document_id, document, metadata)


async def delete_document(index_name: str, document_id: str) -> bool:
    """Delete document from index"""
    return await search_service.delete_document(index_name, document_id)


async def get_search_suggestions(query: str, limit: int = 10) -> List[str]:
    """Get search suggestions"""
    return await search_service.get_suggestions(query, limit)


async def get_search_facets(index_name: str, query: str, facet_fields: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Get search facets"""
    return await search_service.get_facets(index_name, query, facet_fields)


async def get_search_stats() -> Dict[str, Any]:
    """Get search statistics"""
    return await search_service.get_search_stats()


