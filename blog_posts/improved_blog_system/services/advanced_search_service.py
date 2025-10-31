"""
Advanced Search Service for intelligent search with AI and semantic capabilities
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from sentence_transformers import SentenceTransformer

from ..models.database import SearchQuery, SearchResult, SearchIndex, SearchAnalytics
from ..core.exceptions import DatabaseError, ValidationError


class SearchType(Enum):
    """Search types."""
    TEXT = "text"
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"
    BOOLEAN = "boolean"
    ADVANCED = "advanced"


class SearchScope(Enum):
    """Search scope."""
    TITLE = "title"
    CONTENT = "content"
    TAGS = "tags"
    AUTHOR = "author"
    ALL = "all"


@dataclass
class SearchFilter:
    """Search filter configuration."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, contains, starts_with, ends_with
    value: Any
    boost: float = 1.0


@dataclass
class SearchFacet:
    """Search facet configuration."""
    field: str
    size: int = 10
    min_count: int = 1


class AdvancedSearchService:
    """Service for advanced search operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.sentence_model = None
        self.nlp = None
        self.stemmer = PorterStemmer()
        self.search_index = {}
        self.search_cache = {}
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP models."""
        try:
            # Initialize sentence transformer for semantic search
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize spaCy for advanced NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback to basic model
                self.nlp = None
            
            # Download NLTK data
            try:
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
            except:
                pass
                
        except Exception as e:
            print(f"Warning: Could not initialize NLP models: {e}")
    
    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.TEXT,
        scope: SearchScope = SearchScope.ALL,
        filters: Optional[List[SearchFilter]] = None,
        facets: Optional[List[SearchFacet]] = None,
        sort: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform advanced search."""
        try:
            # Log search query
            await self._log_search_query(query, search_type, scope, user_id)
            
            # Check cache first
            cache_key = self._generate_cache_key(query, search_type, scope, filters, page, page_size)
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if datetime.now() - cached_result["timestamp"] < timedelta(minutes=5):
                    return cached_result["data"]
            
            # Perform search based on type
            if search_type == SearchType.TEXT:
                results = await self._text_search(query, scope, filters, sort, page, page_size)
            elif search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query, scope, filters, sort, page, page_size)
            elif search_type == SearchType.FUZZY:
                results = await self._fuzzy_search(query, scope, filters, sort, page, page_size)
            elif search_type == SearchType.BOOLEAN:
                results = await self._boolean_search(query, scope, filters, sort, page, page_size)
            elif search_type == SearchType.ADVANCED:
                results = await self._advanced_search(query, scope, filters, sort, page, page_size)
            else:
                raise ValidationError(f"Unsupported search type: {search_type}")
            
            # Apply facets if requested
            if facets:
                results["facets"] = await self._calculate_facets(query, facets, filters)
            
            # Cache results
            self.search_cache[cache_key] = {
                "data": results,
                "timestamp": datetime.now()
            }
            
            # Update search analytics
            await self._update_search_analytics(query, len(results.get("results", [])))
            
            return results
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform search: {str(e)}")
    
    async def _text_search(
        self,
        query: str,
        scope: SearchScope,
        filters: Optional[List[SearchFilter]],
        sort: Optional[str],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """Perform text-based search."""
        try:
            # Build SQL query
            sql_query = self._build_base_query(scope)
            
            # Add text search conditions
            if scope == SearchScope.ALL:
                sql_query = sql_query.where(
                    or_(
                        text("title ILIKE :query"),
                        text("content ILIKE :query"),
                        text("tags ILIKE :query"),
                        text("author ILIKE :query")
                    )
                )
            elif scope == SearchScope.TITLE:
                sql_query = sql_query.where(text("title ILIKE :query"))
            elif scope == SearchScope.CONTENT:
                sql_query = sql_query.where(text("content ILIKE :query"))
            elif scope == SearchScope.TAGS:
                sql_query = sql_query.where(text("tags ILIKE :query"))
            elif scope == SearchScope.AUTHOR:
                sql_query = sql_query.where(text("author ILIKE :query"))
            
            # Add filters
            if filters:
                sql_query = self._apply_filters(sql_query, filters)
            
            # Add sorting
            if sort:
                sql_query = self._apply_sorting(sql_query, sort)
            else:
                sql_query = sql_query.order_by(desc("created_at"))
            
            # Add pagination
            offset = (page - 1) * page_size
            sql_query = sql_query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.session.execute(sql_query, {"query": f"%{query}%"})
            posts = result.scalars().all()
            
            # Get total count
            count_query = self._build_count_query(scope, filters)
            if scope == SearchScope.ALL:
                count_query = count_query.where(
                    or_(
                        text("title ILIKE :query"),
                        text("content ILIKE :query"),
                        text("tags ILIKE :query"),
                        text("author ILIKE :query")
                    )
                )
            else:
                count_query = count_query.where(text(f"{scope.value} ILIKE :query"))
            
            if filters:
                count_query = self._apply_filters(count_query, filters)
            
            count_result = await self.session.execute(count_query, {"query": f"%{query}%"})
            total_count = count_result.scalar()
            
            return {
                "results": [self._format_post_result(post) for post in posts],
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size,
                "search_type": "text",
                "query": query
            }
            
        except Exception as e:
            raise DatabaseError(f"Text search failed: {str(e)}")
    
    async def _semantic_search(
        self,
        query: str,
        scope: SearchScope,
        filters: Optional[List[SearchFilter]],
        sort: Optional[str],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """Perform semantic search using embeddings."""
        try:
            if not self.sentence_model:
                # Fallback to text search
                return await self._text_search(query, scope, filters, sort, page, page_size)
            
            # Generate query embedding
            query_embedding = self.sentence_model.encode([query])
            
            # Get all posts for similarity calculation
            all_posts_query = self._build_base_query(scope)
            if filters:
                all_posts_query = self._apply_filters(all_posts_query, filters)
            
            all_posts_result = await self.session.execute(all_posts_query)
            all_posts = all_posts_result.scalars().all()
            
            if not all_posts:
                return {
                    "results": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "search_type": "semantic",
                    "query": query
                }
            
            # Calculate similarities
            similarities = []
            for post in all_posts:
                # Combine title and content for embedding
                post_text = f"{post.title} {post.content}"
                post_embedding = self.sentence_model.encode([post_text])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, post_embedding)[0][0]
                similarities.append((post, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply pagination
            offset = (page - 1) * page_size
            paginated_results = similarities[offset:offset + page_size]
            
            return {
                "results": [
                    {
                        **self._format_post_result(post),
                        "similarity_score": float(similarity)
                    }
                    for post, similarity in paginated_results
                ],
                "total": len(similarities),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(similarities) + page_size - 1) // page_size,
                "search_type": "semantic",
                "query": query
            }
            
        except Exception as e:
            raise DatabaseError(f"Semantic search failed: {str(e)}")
    
    async def _fuzzy_search(
        self,
        query: str,
        scope: SearchScope,
        filters: Optional[List[SearchFilter]],
        sort: Optional[str],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """Perform fuzzy search with typo tolerance."""
        try:
            # Generate fuzzy variations of the query
            fuzzy_queries = self._generate_fuzzy_queries(query)
            
            # Build search conditions for fuzzy queries
            conditions = []
            for fuzzy_query in fuzzy_queries:
                if scope == SearchScope.ALL:
                    conditions.append(
                        or_(
                            text("title ILIKE :fuzzy_query"),
                            text("content ILIKE :fuzzy_query"),
                            text("tags ILIKE :fuzzy_query"),
                            text("author ILIKE :fuzzy_query")
                        )
                    )
                else:
                    conditions.append(text(f"{scope.value} ILIKE :fuzzy_query"))
            
            # Build main query
            sql_query = self._build_base_query(scope)
            sql_query = sql_query.where(or_(*conditions))
            
            # Add filters
            if filters:
                sql_query = self._apply_filters(sql_query, filters)
            
            # Add sorting
            if sort:
                sql_query = self._apply_sorting(sql_query, sort)
            else:
                sql_query = sql_query.order_by(desc("created_at"))
            
            # Add pagination
            offset = (page - 1) * page_size
            sql_query = sql_query.offset(offset).limit(page_size)
            
            # Execute query with all fuzzy variations
            all_results = set()
            for fuzzy_query in fuzzy_queries:
                result = await self.session.execute(sql_query, {"fuzzy_query": f"%{fuzzy_query}%"})
                posts = result.scalars().all()
                all_results.update(posts)
            
            # Convert to list and apply pagination
            results_list = list(all_results)
            total_count = len(results_list)
            
            # Sort results
            if sort:
                results_list = self._sort_results(results_list, sort)
            
            # Apply pagination
            paginated_results = results_list[offset:offset + page_size]
            
            return {
                "results": [self._format_post_result(post) for post in paginated_results],
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size,
                "search_type": "fuzzy",
                "query": query,
                "fuzzy_queries": fuzzy_queries
            }
            
        except Exception as e:
            raise DatabaseError(f"Fuzzy search failed: {str(e)}")
    
    async def _boolean_search(
        self,
        query: str,
        scope: SearchScope,
        filters: Optional[List[SearchFilter]],
        sort: Optional[str],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """Perform boolean search with AND, OR, NOT operators."""
        try:
            # Parse boolean query
            parsed_query = self._parse_boolean_query(query)
            
            # Build search conditions
            conditions = self._build_boolean_conditions(parsed_query, scope)
            
            if not conditions:
                return {
                    "results": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "search_type": "boolean",
                    "query": query
                }
            
            # Build main query
            sql_query = self._build_base_query(scope)
            sql_query = sql_query.where(and_(*conditions))
            
            # Add filters
            if filters:
                sql_query = self._apply_filters(sql_query, filters)
            
            # Add sorting
            if sort:
                sql_query = self._apply_sorting(sql_query, sort)
            else:
                sql_query = sql_query.order_by(desc("created_at"))
            
            # Add pagination
            offset = (page - 1) * page_size
            sql_query = sql_query.offset(offset).limit(page_size)
            
            # Execute query
            result = await self.session.execute(sql_query)
            posts = result.scalars().all()
            
            # Get total count
            count_query = self._build_count_query(scope, filters)
            count_query = count_query.where(and_(*conditions))
            
            count_result = await self.session.execute(count_query)
            total_count = count_result.scalar()
            
            return {
                "results": [self._format_post_result(post) for post in posts],
                "total": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size,
                "search_type": "boolean",
                "query": query,
                "parsed_query": parsed_query
            }
            
        except Exception as e:
            raise DatabaseError(f"Boolean search failed: {str(e)}")
    
    async def _advanced_search(
        self,
        query: str,
        scope: SearchScope,
        filters: Optional[List[SearchFilter]],
        sort: Optional[str],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """Perform advanced search combining multiple techniques."""
        try:
            # Combine text and semantic search
            text_results = await self._text_search(query, scope, filters, sort, page, page_size)
            semantic_results = await self._semantic_search(query, scope, filters, sort, page, page_size)
            
            # Merge and rank results
            merged_results = self._merge_search_results(text_results, semantic_results)
            
            # Apply pagination to merged results
            offset = (page - 1) * page_size
            paginated_results = merged_results[offset:offset + page_size]
            
            return {
                "results": paginated_results,
                "total": len(merged_results),
                "page": page,
                "page_size": page_size,
                "total_pages": (len(merged_results) + page_size - 1) // page_size,
                "search_type": "advanced",
                "query": query,
                "text_results_count": len(text_results.get("results", [])),
                "semantic_results_count": len(semantic_results.get("results", []))
            }
            
        except Exception as e:
            raise DatabaseError(f"Advanced search failed: {str(e)}")
    
    def _build_base_query(self, scope: SearchScope):
        """Build base SQL query."""
        # This would be implemented based on your actual database models
        # For now, returning a placeholder
        return select("*").select_from("blog_posts")
    
    def _build_count_query(self, scope: SearchScope, filters: Optional[List[SearchFilter]]):
        """Build count query."""
        # This would be implemented based on your actual database models
        return select(func.count("*")).select_from("blog_posts")
    
    def _apply_filters(self, query, filters: List[SearchFilter]):
        """Apply search filters to query."""
        for filter_obj in filters:
            if filter_obj.operator == "eq":
                query = query.where(text(f"{filter_obj.field} = :filter_value"))
            elif filter_obj.operator == "ne":
                query = query.where(text(f"{filter_obj.field} != :filter_value"))
            elif filter_obj.operator == "gt":
                query = query.where(text(f"{filter_obj.field} > :filter_value"))
            elif filter_obj.operator == "lt":
                query = query.where(text(f"{filter_obj.field} < :filter_value"))
            elif filter_obj.operator == "gte":
                query = query.where(text(f"{filter_obj.field} >= :filter_value"))
            elif filter_obj.operator == "lte":
                query = query.where(text(f"{filter_obj.field} <= :filter_value"))
            elif filter_obj.operator == "in":
                query = query.where(text(f"{filter_obj.field} IN :filter_value"))
            elif filter_obj.operator == "contains":
                query = query.where(text(f"{filter_obj.field} ILIKE :filter_value"))
            elif filter_obj.operator == "starts_with":
                query = query.where(text(f"{filter_obj.field} ILIKE :filter_value"))
            elif filter_obj.operator == "ends_with":
                query = query.where(text(f"{filter_obj.field} ILIKE :filter_value"))
        
        return query
    
    def _apply_sorting(self, query, sort: str):
        """Apply sorting to query."""
        if sort.startswith("-"):
            field = sort[1:]
            return query.order_by(desc(text(field)))
        else:
            return query.order_by(text(sort))
    
    def _format_post_result(self, post) -> Dict[str, Any]:
        """Format post result for response."""
        return {
            "id": post.id,
            "title": post.title,
            "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
            "author": post.author,
            "tags": post.tags,
            "created_at": post.created_at.isoformat(),
            "updated_at": post.updated_at.isoformat(),
            "view_count": post.view_count,
            "like_count": post.like_count
        }
    
    def _generate_fuzzy_queries(self, query: str) -> List[str]:
        """Generate fuzzy variations of the query."""
        fuzzy_queries = [query]
        
        # Add common typos
        words = query.split()
        for i, word in enumerate(words):
            if len(word) > 3:
                # Character substitution
                for j in range(len(word)):
                    for char in "abcdefghijklmnopqrstuvwxyz":
                        if char != word[j]:
                            new_word = word[:j] + char + word[j+1:]
                            new_query = " ".join(words[:i] + [new_word] + words[i+1:])
                            fuzzy_queries.append(new_query)
                
                # Character insertion
                for j in range(len(word) + 1):
                    for char in "abcdefghijklmnopqrstuvwxyz":
                        new_word = word[:j] + char + word[j:]
                        new_query = " ".join(words[:i] + [new_word] + words[i+1:])
                        fuzzy_queries.append(new_query)
                
                # Character deletion
                for j in range(len(word)):
                    new_word = word[:j] + word[j+1:]
                    if new_word:
                        new_query = " ".join(words[:i] + [new_word] + words[i+1:])
                        fuzzy_queries.append(new_query)
        
        return list(set(fuzzy_queries))[:10]  # Limit to 10 variations
    
    def _parse_boolean_query(self, query: str) -> Dict[str, Any]:
        """Parse boolean query into structured format."""
        # Simple boolean query parser
        # This is a basic implementation - could be enhanced
        tokens = query.split()
        parsed = {
            "must": [],
            "should": [],
            "must_not": []
        }
        
        current_list = "must"
        for token in tokens:
            if token.upper() == "AND":
                current_list = "must"
            elif token.upper() == "OR":
                current_list = "should"
            elif token.upper() == "NOT":
                current_list = "must_not"
            else:
                parsed[current_list].append(token)
        
        return parsed
    
    def _build_boolean_conditions(self, parsed_query: Dict[str, Any], scope: SearchScope) -> List:
        """Build SQL conditions from parsed boolean query."""
        conditions = []
        
        # Must conditions (AND)
        for term in parsed_query["must"]:
            if scope == SearchScope.ALL:
                conditions.append(
                    or_(
                        text("title ILIKE :term"),
                        text("content ILIKE :term"),
                        text("tags ILIKE :term"),
                        text("author ILIKE :term")
                    )
                )
            else:
                conditions.append(text(f"{scope.value} ILIKE :term"))
        
        # Should conditions (OR) - would need more complex logic
        # For now, treating as must conditions
        
        # Must not conditions (NOT)
        for term in parsed_query["must_not"]:
            if scope == SearchScope.ALL:
                conditions.append(
                    and_(
                        ~text("title ILIKE :term"),
                        ~text("content ILIKE :term"),
                        ~text("tags ILIKE :term"),
                        ~text("author ILIKE :term")
                    )
                )
            else:
                conditions.append(~text(f"{scope.value} ILIKE :term"))
        
        return conditions
    
    def _merge_search_results(self, text_results: Dict, semantic_results: Dict) -> List[Dict]:
        """Merge and rank search results from different search types."""
        # Combine results and remove duplicates
        all_results = {}
        
        # Add text search results
        for result in text_results.get("results", []):
            post_id = result["id"]
            all_results[post_id] = {
                **result,
                "text_score": 1.0,
                "semantic_score": 0.0
            }
        
        # Add semantic search results
        for result in semantic_results.get("results", []):
            post_id = result["id"]
            if post_id in all_results:
                all_results[post_id]["semantic_score"] = result.get("similarity_score", 0.0)
            else:
                all_results[post_id] = {
                    **result,
                    "text_score": 0.0,
                    "semantic_score": result.get("similarity_score", 0.0)
                }
        
        # Calculate combined score and sort
        for post_id, result in all_results.items():
            result["combined_score"] = (
                result["text_score"] * 0.6 + 
                result["semantic_score"] * 0.4
            )
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        return sorted_results
    
    def _sort_results(self, results: List, sort: str) -> List:
        """Sort results based on sort parameter."""
        if sort.startswith("-"):
            field = sort[1:]
            reverse = True
        else:
            field = sort
            reverse = False
        
        try:
            return sorted(results, key=lambda x: getattr(x, field, 0), reverse=reverse)
        except:
            return results
    
    async def _calculate_facets(
        self,
        query: str,
        facets: List[SearchFacet],
        filters: Optional[List[SearchFilter]]
    ) -> Dict[str, Any]:
        """Calculate search facets."""
        facet_results = {}
        
        for facet in facets:
            # This would be implemented based on your database structure
            # For now, returning placeholder data
            facet_results[facet.field] = {
                "buckets": [
                    {"key": f"value_{i}", "count": i * 10}
                    for i in range(1, min(facet.size + 1, 6))
                ],
                "total": facet.size
            }
        
        return facet_results
    
    def _generate_cache_key(
        self,
        query: str,
        search_type: SearchType,
        scope: SearchScope,
        filters: Optional[List[SearchFilter]],
        page: int,
        page_size: int
    ) -> str:
        """Generate cache key for search results."""
        key_data = {
            "query": query,
            "search_type": search_type.value,
            "scope": scope.value,
            "filters": [f"{f.field}:{f.operator}:{f.value}" for f in (filters or [])],
            "page": page,
            "page_size": page_size
        }
        return f"search:{hash(str(key_data))}"
    
    async def _log_search_query(
        self,
        query: str,
        search_type: SearchType,
        scope: SearchScope,
        user_id: Optional[str]
    ):
        """Log search query for analytics."""
        try:
            search_query = SearchQuery(
                query=query,
                search_type=search_type.value,
                scope=scope.value,
                user_id=user_id,
                timestamp=datetime.utcnow()
            )
            
            self.session.add(search_query)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            # Don't raise error for logging failures
    
    async def _update_search_analytics(self, query: str, result_count: int):
        """Update search analytics."""
        try:
            # This would update search analytics in the database
            # For now, just a placeholder
            pass
            
        except Exception as e:
            # Don't raise error for analytics failures
            pass
    
    async def get_search_suggestions(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get search suggestions based on query."""
        try:
            # Get popular search queries that start with the input
            suggestions_query = select(SearchQuery.query).where(
                and_(
                    SearchQuery.query.ilike(f"{query}%"),
                    SearchQuery.timestamp >= datetime.utcnow() - timedelta(days=30)
                )
            ).group_by(SearchQuery.query).order_by(
                func.count(SearchQuery.id).desc()
            ).limit(limit)
            
            result = await self.session.execute(suggestions_query)
            suggestions = [row[0] for row in result.fetchall()]
            
            return {
                "suggestions": suggestions,
                "query": query,
                "total": len(suggestions)
            }
            
        except Exception as e:
            return {
                "suggestions": [],
                "query": query,
                "total": 0,
                "error": str(e)
            }
    
    async def get_search_analytics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get search analytics."""
        try:
            # Get search statistics
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total searches
            total_searches_query = select(func.count(SearchQuery.id)).where(
                SearchQuery.timestamp >= start_date
            )
            total_searches_result = await self.session.execute(total_searches_query)
            total_searches = total_searches_result.scalar()
            
            # Searches by type
            searches_by_type_query = select(
                SearchQuery.search_type,
                func.count(SearchQuery.id).label('count')
            ).where(
                SearchQuery.timestamp >= start_date
            ).group_by(SearchQuery.search_type)
            
            searches_by_type_result = await self.session.execute(searches_by_type_query)
            searches_by_type = {row[0]: row[1] for row in searches_by_type_result}
            
            # Top queries
            top_queries_query = select(
                SearchQuery.query,
                func.count(SearchQuery.id).label('count')
            ).where(
                SearchQuery.timestamp >= start_date
            ).group_by(SearchQuery.query).order_by(
                func.count(SearchQuery.id).desc()
            ).limit(10)
            
            top_queries_result = await self.session.execute(top_queries_query)
            top_queries = [{"query": row[0], "count": row[1]} for row in top_queries_result]
            
            return {
                "total_searches": total_searches,
                "searches_by_type": searches_by_type,
                "top_queries": top_queries,
                "period_days": days
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get search analytics: {str(e)}")
    
    async def rebuild_search_index(self) -> Dict[str, Any]:
        """Rebuild search index."""
        try:
            # Clear existing index
            self.search_index.clear()
            
            # Get all posts
            posts_query = select("*").select_from("blog_posts")
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.fetchall()
            
            # Build index
            for post in posts:
                # Create searchable text
                searchable_text = f"{post.title} {post.content} {post.tags} {post.author}"
                
                # Generate TF-IDF features
                if hasattr(self, 'vectorizer'):
                    try:
                        features = self.vectorizer.fit_transform([searchable_text])
                        self.search_index[post.id] = {
                            "text": searchable_text,
                            "features": features.toarray()[0].tolist(),
                            "title": post.title,
                            "content": post.content,
                            "tags": post.tags,
                            "author": post.author
                        }
                    except:
                        # Fallback to simple text indexing
                        self.search_index[post.id] = {
                            "text": searchable_text,
                            "title": post.title,
                            "content": post.content,
                            "tags": post.tags,
                            "author": post.author
                        }
            
            return {
                "success": True,
                "indexed_documents": len(self.search_index),
                "message": "Search index rebuilt successfully"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to rebuild search index: {str(e)}")
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search system statistics."""
        try:
            # Get total searches
            total_searches_query = select(func.count(SearchQuery.id))
            total_searches_result = await self.session.execute(total_searches_query)
            total_searches = total_searches_result.scalar()
            
            # Get searches today
            today = datetime.utcnow().date()
            today_searches_query = select(func.count(SearchQuery.id)).where(
                func.date(SearchQuery.timestamp) == today
            )
            today_searches_result = await self.session.execute(today_searches_query)
            today_searches = today_searches_result.scalar()
            
            # Get unique users
            unique_users_query = select(func.count(func.distinct(SearchQuery.user_id)))
            unique_users_result = await self.session.execute(unique_users_query)
            unique_users = unique_users_result.scalar()
            
            return {
                "total_searches": total_searches,
                "today_searches": today_searches,
                "unique_users": unique_users,
                "indexed_documents": len(self.search_index),
                "cache_size": len(self.search_cache),
                "nlp_models_loaded": {
                    "sentence_transformer": self.sentence_model is not None,
                    "spacy": self.nlp is not None
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get search stats: {str(e)}")