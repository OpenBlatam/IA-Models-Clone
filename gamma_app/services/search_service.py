"""
Gamma App - Search Service
Advanced search and indexing service
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path
import redis
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Search types"""
    CONTENT = "content"
    USER = "user"
    FILE = "file"
    METADATA = "metadata"
    FULL_TEXT = "full_text"

class SearchOperator(Enum):
    """Search operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"
    FUZZY = "FUZZY"

@dataclass
class SearchQuery:
    """Search query"""
    query: str
    search_type: SearchType = SearchType.FULL_TEXT
    operator: SearchOperator = SearchOperator.AND
    filters: Dict[str, Any] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    limit: int = 100
    offset: int = 0
    highlight: bool = True
    fuzzy_threshold: float = 0.8

@dataclass
class SearchResult:
    """Search result"""
    id: str
    title: str
    content: str
    type: str
    score: float
    metadata: Dict[str, Any]
    highlighted_content: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class IndexDocument:
    """Index document"""
    id: str
    title: str
    content: str
    type: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = None

class SearchIndex:
    """Search index"""
    
    def __init__(self, name: str):
        self.name = name
        self.documents: Dict[str, IndexDocument] = {}
        self.inverted_index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.total_documents = 0
    
    def add_document(self, document: IndexDocument):
        """Add document to index"""
        try:
            # Remove existing document if present
            if document.id in self.documents:
                self.remove_document(document.id)
            
            # Add document
            self.documents[document.id] = document
            self.total_documents += 1
            
            # Index terms
            self._index_terms(document)
            
        except Exception as e:
            logger.error(f"Error adding document to index: {e}")
    
    def remove_document(self, document_id: str):
        """Remove document from index"""
        try:
            if document_id not in self.documents:
                return
            
            document = self.documents[document_id]
            
            # Remove from inverted index
            self._remove_terms(document)
            
            # Remove document
            del self.documents[document_id]
            self.total_documents -= 1
            
        except Exception as e:
            logger.error(f"Error removing document from index: {e}")
    
    def _index_terms(self, document: IndexDocument):
        """Index document terms"""
        try:
            # Combine title and content
            text = f"{document.title} {document.content}"
            
            # Tokenize and process text
            terms = self._tokenize(text)
            
            # Add to inverted index
            for position, term in enumerate(terms):
                self.inverted_index[term][document.id].append(position)
                self.document_frequency[term] += 1
                
        except Exception as e:
            logger.error(f"Error indexing terms: {e}")
    
    def _remove_terms(self, document: IndexDocument):
        """Remove document terms from index"""
        try:
            # Combine title and content
            text = f"{document.title} {document.content}"
            
            # Tokenize and process text
            terms = self._tokenize(text)
            
            # Remove from inverted index
            for term in terms:
                if document.id in self.inverted_index[term]:
                    del self.inverted_index[term][document.id]
                    self.document_frequency[term] -= 1
                    
                    if self.document_frequency[term] <= 0:
                        del self.document_frequency[term]
                        
        except Exception as e:
            logger.error(f"Error removing terms: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and split
            tokens = re.findall(r'\b\w+\b', text)
            
            # Remove stop words
            stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with'
            }
            
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search index"""
        try:
            # Tokenize query
            query_terms = self._tokenize(query.query)
            
            if not query_terms:
                return []
            
            # Get document scores
            document_scores = self._calculate_scores(query_terms, query.operator)
            
            # Apply filters
            if query.filters:
                document_scores = self._apply_filters(document_scores, query.filters)
            
            # Sort results
            sorted_docs = self._sort_results(document_scores, query.sort_by, query.sort_order)
            
            # Create results
            results = []
            for doc_id, score in sorted_docs[query.offset:query.offset + query.limit]:
                if doc_id in self.documents:
                    document = self.documents[doc_id]
                    
                    # Create result
                    result = SearchResult(
                        id=document.id,
                        title=document.title,
                        content=document.content,
                        type=document.type,
                        score=score,
                        metadata=document.metadata,
                        created_at=document.created_at,
                        updated_at=document.updated_at
                    )
                    
                    # Add highlighting if requested
                    if query.highlight:
                        result.highlighted_content = self._highlight_terms(
                            document.content, query_terms
                        )
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []
    
    def _calculate_scores(
        self,
        query_terms: List[str],
        operator: SearchOperator
    ) -> Dict[str, float]:
        """Calculate document scores"""
        try:
            document_scores = defaultdict(float)
            
            if operator == SearchOperator.AND:
                # All terms must be present
                for term in query_terms:
                    if term in self.inverted_index:
                        for doc_id in self.inverted_index[term]:
                            document_scores[doc_id] += 1
                
                # Only include documents that have all terms
                document_scores = {
                    doc_id: score for doc_id, score in document_scores.items()
                    if score == len(query_terms)
                }
                
            elif operator == SearchOperator.OR:
                # Any term can be present
                for term in query_terms:
                    if term in self.inverted_index:
                        for doc_id in self.inverted_index[term]:
                            # Use TF-IDF scoring
                            tf = len(self.inverted_index[term][doc_id])
                            idf = self._calculate_idf(term)
                            document_scores[doc_id] += tf * idf
                            
            elif operator == SearchOperator.NOT:
                # Exclude documents with terms
                all_docs = set(self.documents.keys())
                excluded_docs = set()
                
                for term in query_terms:
                    if term in self.inverted_index:
                        excluded_docs.update(self.inverted_index[term].keys())
                
                for doc_id in all_docs - excluded_docs:
                    document_scores[doc_id] = 1.0
                    
            elif operator == SearchOperator.PHRASE:
                # Exact phrase matching
                phrase = " ".join(query_terms)
                for doc_id, document in self.documents.items():
                    if phrase.lower() in document.content.lower():
                        document_scores[doc_id] = 1.0
                        
            elif operator == SearchOperator.FUZZY:
                # Fuzzy matching
                for term in query_terms:
                    for index_term in self.inverted_index:
                        if self._calculate_similarity(term, index_term) > 0.8:
                            for doc_id in self.inverted_index[index_term]:
                                document_scores[doc_id] += 1
            
            return dict(document_scores)
            
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            return {}
    
    def _calculate_idf(self, term: str) -> float:
        """Calculate IDF for term"""
        try:
            if term not in self.document_frequency or self.total_documents == 0:
                return 0.0
            
            return 1.0 + (self.total_documents / self.document_frequency[term])
            
        except Exception as e:
            logger.error(f"Error calculating IDF: {e}")
            return 0.0
    
    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity between two terms"""
        try:
            # Simple Levenshtein distance-based similarity
            if len(term1) == 0 or len(term2) == 0:
                return 0.0
            
            # Calculate edit distance
            distance = self._levenshtein_distance(term1, term2)
            max_len = max(len(term1), len(term2))
            
            return 1.0 - (distance / max_len)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        try:
            if len(s1) < len(s2):
                return self._levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
            
        except Exception as e:
            logger.error(f"Error calculating Levenshtein distance: {e}")
            return 0
    
    def _apply_filters(
        self,
        document_scores: Dict[str, float],
        filters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply filters to search results"""
        try:
            filtered_scores = {}
            
            for doc_id, score in document_scores.items():
                if doc_id not in self.documents:
                    continue
                
                document = self.documents[doc_id]
                
                # Apply filters
                include = True
                for filter_key, filter_value in filters.items():
                    if filter_key in document.metadata:
                        if document.metadata[filter_key] != filter_value:
                            include = False
                            break
                    elif filter_key == 'type':
                        if document.type != filter_value:
                            include = False
                            break
                    elif filter_key == 'tags':
                        if not any(tag in (document.tags or []) for tag in filter_value):
                            include = False
                            break
                
                if include:
                    filtered_scores[doc_id] = score
            
            return filtered_scores
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return document_scores
    
    def _sort_results(
        self,
        document_scores: Dict[str, float],
        sort_by: str,
        sort_order: str
    ) -> List[Tuple[str, float]]:
        """Sort search results"""
        try:
            if sort_by == "relevance":
                # Sort by score
                sorted_docs = sorted(
                    document_scores.items(),
                    key=lambda x: x[1],
                    reverse=(sort_order == "desc")
                )
            elif sort_by == "date":
                # Sort by creation date
                sorted_docs = sorted(
                    document_scores.items(),
                    key=lambda x: self.documents[x[0]].created_at,
                    reverse=(sort_order == "desc")
                )
            elif sort_by == "title":
                # Sort by title
                sorted_docs = sorted(
                    document_scores.items(),
                    key=lambda x: self.documents[x[0]].title,
                    reverse=(sort_order == "desc")
                )
            else:
                # Default to relevance
                sorted_docs = sorted(
                    document_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            
            return sorted_docs
            
        except Exception as e:
            logger.error(f"Error sorting results: {e}")
            return list(document_scores.items())
    
    def _highlight_terms(self, content: str, terms: List[str]) -> str:
        """Highlight search terms in content"""
        try:
            highlighted = content
            
            for term in terms:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(
                    f'<mark>{term}</mark>',
                    highlighted
                )
            
            return highlighted
            
        except Exception as e:
            logger.error(f"Error highlighting terms: {e}")
            return content

class SearchService:
    """Advanced search service"""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.indexes: Dict[str, SearchIndex] = {}
        self.default_index = "default"
        
        # Initialize default index
        self.indexes[self.default_index] = SearchIndex(self.default_index)
        
        # Load indexes from Redis
        self._load_indexes()
    
    def _load_indexes(self):
        """Load indexes from Redis"""
        try:
            # Load indexes from Redis
            if self.redis_client:
                for index_name in self.indexes.keys():
                    index_data = self.redis_client.get(f"search_index:{index_name}")
                    if index_data:
                        self.indexes[index_name] = json.loads(index_data)
                        logger.info(f"Loaded index: {index_name}")
            else:
                # Initialize default index structure
                for index_name in self.indexes.keys():
                    self.indexes[index_name] = {
                        "documents": {},
                        "inverted_index": {},
                        "metadata": {"created": datetime.now().isoformat()}
                    }
                logger.info("Initialized default indexes")
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            # Fallback to default indexes
            for index_name in self.indexes.keys():
                self.indexes[index_name] = {
                    "documents": {},
                    "inverted_index": {},
                    "metadata": {"created": datetime.now().isoformat()}
                }
    
    def _save_indexes(self):
        """Save indexes to Redis"""
        try:
            # Save indexes to Redis
            if self.redis_client:
                for index_name, index_data in self.indexes.items():
                    self.redis_client.set(
                        f"search_index:{index_name}",
                        json.dumps(index_data),
                        ex=3600  # Expire in 1 hour
                    )
                logger.info("Saved indexes to Redis")
            else:
                # Save to local file as fallback
                indexes_file = Path("data/search_indexes.json")
                indexes_file.parent.mkdir(exist_ok=True)
                with open(indexes_file, 'w') as f:
                    json.dump(self.indexes, f, indent=2)
                logger.info("Saved indexes to local file")
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")
    
    async def create_index(self, name: str) -> bool:
        """Create new search index"""
        try:
            if name in self.indexes:
                return False
            
            self.indexes[name] = SearchIndex(name)
            self._save_indexes()
            
            logger.info(f"Created search index: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    async def delete_index(self, name: str) -> bool:
        """Delete search index"""
        try:
            if name not in self.indexes:
                return False
            
            del self.indexes[name]
            self._save_indexes()
            
            logger.info(f"Deleted search index: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    async def add_document(
        self,
        document: IndexDocument,
        index_name: str = None
    ) -> bool:
        """Add document to search index"""
        try:
            if index_name is None:
                index_name = self.default_index
            
            if index_name not in self.indexes:
                await self.create_index(index_name)
            
            self.indexes[index_name].add_document(document)
            self._save_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    async def remove_document(
        self,
        document_id: str,
        index_name: str = None
    ) -> bool:
        """Remove document from search index"""
        try:
            if index_name is None:
                index_name = self.default_index
            
            if index_name not in self.indexes:
                return False
            
            self.indexes[index_name].remove_document(document_id)
            self._save_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False
    
    async def search(
        self,
        query: SearchQuery,
        index_name: str = None
    ) -> List[SearchResult]:
        """Search documents"""
        try:
            if index_name is None:
                index_name = self.default_index
            
            if index_name not in self.indexes:
                return []
            
            return self.indexes[index_name].search(query)
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    async def search_multiple(
        self,
        query: SearchQuery,
        index_names: List[str]
    ) -> List[SearchResult]:
        """Search multiple indexes"""
        try:
            all_results = []
            
            for index_name in index_names:
                if index_name in self.indexes:
                    results = self.indexes[index_name].search(query)
                    all_results.extend(results)
            
            # Sort by score
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply limit
            return all_results[:query.limit]
            
        except Exception as e:
            logger.error(f"Error searching multiple indexes: {e}")
            return []
    
    async def get_suggestions(
        self,
        query: str,
        limit: int = 10
    ) -> List[str]:
        """Get search suggestions"""
        try:
            suggestions = set()
            
            for index in self.indexes.values():
                # Get terms that start with query
                for term in index.inverted_index:
                    if term.startswith(query.lower()):
                        suggestions.add(term)
                
                # Get terms that contain query
                for term in index.inverted_index:
                    if query.lower() in term and term not in suggestions:
                        suggestions.add(term)
            
            # Sort by frequency
            suggestions = sorted(
                suggestions,
                key=lambda x: self._get_term_frequency(x),
                reverse=True
            )
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []
    
    def _get_term_frequency(self, term: str) -> int:
        """Get term frequency across all indexes"""
        try:
            total_frequency = 0
            for index in self.indexes.values():
                total_frequency += index.document_frequency.get(term, 0)
            return total_frequency
        except Exception as e:
            logger.error(f"Error getting term frequency: {e}")
            return 0
    
    async def get_index_stats(self, index_name: str = None) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            if index_name is None:
                index_name = self.default_index
            
            if index_name not in self.indexes:
                return {}
            
            index = self.indexes[index_name]
            
            return {
                'name': index.name,
                'total_documents': index.total_documents,
                'total_terms': len(index.inverted_index),
                'average_terms_per_document': len(index.inverted_index) / max(index.total_documents, 1),
                'most_frequent_terms': sorted(
                    index.document_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    async def rebuild_index(self, index_name: str = None) -> bool:
        """Rebuild search index"""
        try:
            if index_name is None:
                index_name = self.default_index
            
            if index_name not in self.indexes:
                return False
            
            # Clear existing index
            self.indexes[index_name] = SearchIndex(index_name)
            
            # Rebuild from data source
            # This would typically reload from database
            self._save_indexes()
            
            logger.info(f"Rebuilt search index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False




