"""
Advanced search service with full-text search and filtering
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.orm import selectinload

from ..models.database import BlogPost, User, Comment, Tag, Category
from ..models.schemas import PostStatus, PostCategory
from ..core.exceptions import DatabaseError
from ..utils.pagination import create_paginated_response
from ..models.schemas import PaginationParams


class SearchService:
    """Service for advanced search operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def full_text_search(
        self,
        query: str,
        pagination: PaginationParams,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Perform full-text search on blog posts."""
        try:
            # Build base query
            base_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    or_(
                        BlogPost.title.ilike(f"%{query}%"),
                        BlogPost.content.ilike(f"%{query}%"),
                        BlogPost.excerpt.ilike(f"%{query}%"),
                        BlogPost.tags.contains([query])
                    )
                )
            )
            
            # Apply additional filters
            if filters:
                base_query = self._apply_filters(base_query, filters)
            
            # Get total count
            count_query = select(func.count(BlogPost.id))
            count_query = count_query.where(base_query.whereclause)
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            base_query = base_query.order_by(desc(BlogPost.created_at))
            base_query = base_query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await self.session.execute(base_query)
            posts = result.scalars().all()
            
            # Format results
            search_results = []
            for post in posts:
                # Calculate relevance score (simplified)
                relevance_score = self._calculate_relevance_score(post, query)
                
                search_results.append({
                    "id": post.id,
                    "uuid": str(post.uuid),
                    "title": post.title,
                    "slug": post.slug,
                    "excerpt": post.excerpt,
                    "author_id": str(post.author_id),
                    "category": post.category,
                    "tags": post.tags,
                    "view_count": post.view_count,
                    "like_count": post.like_count,
                    "comment_count": post.comment_count,
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "relevance_score": relevance_score,
                    "matched_fields": self._get_matched_fields(post, query)
                })
            
            # Sort by relevance score
            search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return search_results, total
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform full-text search: {str(e)}")
    
    async def advanced_search(
        self,
        search_params: Dict[str, Any],
        pagination: PaginationParams
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Perform advanced search with multiple criteria."""
        try:
            # Build query
            query = select(BlogPost).where(BlogPost.status == PostStatus.PUBLISHED.value)
            
            # Text search
            if search_params.get("query"):
                text_query = search_params["query"]
                query = query.where(
                    or_(
                        BlogPost.title.ilike(f"%{text_query}%"),
                        BlogPost.content.ilike(f"%{text_query}%"),
                        BlogPost.excerpt.ilike(f"%{text_query}%")
                    )
                )
            
            # Category filter
            if search_params.get("category"):
                query = query.where(BlogPost.category == search_params["category"])
            
            # Tags filter
            if search_params.get("tags"):
                for tag in search_params["tags"]:
                    query = query.where(BlogPost.tags.contains([tag]))
            
            # Author filter
            if search_params.get("author_id"):
                query = query.where(BlogPost.author_id == search_params["author_id"])
            
            # Date range filter
            if search_params.get("date_from"):
                query = query.where(BlogPost.created_at >= search_params["date_from"])
            
            if search_params.get("date_to"):
                query = query.where(BlogPost.created_at <= search_params["date_to"])
            
            # View count filter
            if search_params.get("min_views"):
                query = query.where(BlogPost.view_count >= search_params["min_views"])
            
            if search_params.get("max_views"):
                query = query.where(BlogPost.view_count <= search_params["max_views"])
            
            # Like count filter
            if search_params.get("min_likes"):
                query = query.where(BlogPost.like_count >= search_params["min_likes"])
            
            # Reading time filter
            if search_params.get("max_reading_time"):
                query = query.where(
                    BlogPost.reading_time_minutes <= search_params["max_reading_time"]
                )
            
            # Get total count
            count_query = select(func.count(BlogPost.id))
            count_query = count_query.where(query.whereclause)
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Apply sorting
            sort_by = search_params.get("sort_by", "created_at")
            sort_order = search_params.get("sort_order", "desc")
            
            if sort_by == "relevance" and search_params.get("query"):
                # For relevance sorting, we'll sort by title match first
                query = query.order_by(
                    desc(BlogPost.title.ilike(f"%{search_params['query']}%")),
                    desc(BlogPost.created_at)
                )
            else:
                sort_field = getattr(BlogPost, sort_by, BlogPost.created_at)
                if sort_order == "asc":
                    query = query.order_by(asc(sort_field))
                else:
                    query = query.order_by(desc(sort_field))
            
            # Apply pagination
            query = query.offset(pagination.offset).limit(pagination.size)
            
            # Execute query
            result = await self.session.execute(query)
            posts = result.scalars().all()
            
            # Format results
            search_results = []
            for post in posts:
                search_results.append({
                    "id": post.id,
                    "uuid": str(post.uuid),
                    "title": post.title,
                    "slug": post.slug,
                    "excerpt": post.excerpt,
                    "author_id": str(post.author_id),
                    "category": post.category,
                    "tags": post.tags,
                    "view_count": post.view_count,
                    "like_count": post.like_count,
                    "comment_count": post.comment_count,
                    "reading_time_minutes": post.reading_time_minutes,
                    "created_at": post.created_at,
                    "published_at": post.published_at,
                    "featured_image_url": post.featured_image_url
                })
            
            return search_results, total
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform advanced search: {str(e)}")
    
    async def search_suggestions(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get search suggestions based on query."""
        try:
            suggestions = []
            
            # Get title suggestions
            title_query = select(BlogPost.title).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.title.ilike(f"%{query}%")
                )
            ).limit(limit)
            
            title_result = await self.session.execute(title_query)
            titles = title_result.scalars().all()
            
            for title in titles:
                suggestions.append({
                    "type": "title",
                    "text": title,
                    "category": "Blog Post"
                })
            
            # Get tag suggestions
            tag_query = select(BlogPost.tags).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    func.jsonb_path_exists(BlogPost.tags, f'$[*] ? (@ like_regex "{query}" flag "i")')
                )
            ).limit(limit)
            
            tag_result = await self.session.execute(tag_query)
            tag_lists = tag_result.scalars().all()
            
            unique_tags = set()
            for tag_list in tag_lists:
                if tag_list:
                    for tag in tag_list:
                        if query.lower() in tag.lower():
                            unique_tags.add(tag)
            
            for tag in list(unique_tags)[:limit]:
                suggestions.append({
                    "type": "tag",
                    "text": tag,
                    "category": "Tag"
                })
            
            # Get category suggestions
            category_query = select(BlogPost.category).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.category.ilike(f"%{query}%")
                )
            ).distinct().limit(limit)
            
            category_result = await self.session.execute(category_query)
            categories = category_result.scalars().all()
            
            for category in categories:
                suggestions.append({
                    "type": "category",
                    "text": category,
                    "category": "Category"
                })
            
            return suggestions[:limit]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get search suggestions: {str(e)}")
    
    async def get_popular_searches(
        self,
        limit: int = 10,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get popular search terms (mock implementation)."""
        try:
            # In a real implementation, you would track search queries
            # For now, we'll return popular tags and categories
            
            since_date = datetime.now() - timedelta(days=days)
            
            # Get popular tags
            tag_query = select(
                func.jsonb_array_elements_text(BlogPost.tags).label('tag'),
                func.count().label('count')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.created_at >= since_date
                )
            ).group_by('tag').order_by(desc('count')).limit(limit)
            
            tag_result = await self.session.execute(tag_query)
            popular_tags = tag_result.all()
            
            # Get popular categories
            category_query = select(
                BlogPost.category,
                func.count(BlogPost.id).label('count')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.created_at >= since_date
                )
            ).group_by(BlogPost.category).order_by(desc('count')).limit(limit)
            
            category_result = await self.session.execute(category_query)
            popular_categories = category_result.all()
            
            popular_searches = []
            
            for tag, count in popular_tags:
                popular_searches.append({
                    "term": tag,
                    "type": "tag",
                    "count": count
                })
            
            for category, count in popular_categories:
                popular_searches.append({
                    "term": category,
                    "type": "category",
                    "count": count
                })
            
            # Sort by count and return top results
            popular_searches.sort(key=lambda x: x["count"], reverse=True)
            return popular_searches[:limit]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get popular searches: {str(e)}")
    
    async def get_search_analytics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get search analytics (mock implementation)."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            # Get total posts in period
            total_posts_query = select(func.count(BlogPost.id)).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.created_at >= since_date
                )
            )
            total_posts_result = await self.session.execute(total_posts_query)
            total_posts = total_posts_result.scalar()
            
            # Get posts by category
            category_query = select(
                BlogPost.category,
                func.count(BlogPost.id).label('count')
            ).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.created_at >= since_date
                )
            ).group_by(BlogPost.category).order_by(desc('count'))
            
            category_result = await self.session.execute(category_query)
            posts_by_category = dict(category_result.all())
            
            # Get most viewed posts
            popular_posts_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.created_at >= since_date
                )
            ).order_by(desc(BlogPost.view_count)).limit(10)
            
            popular_posts_result = await self.session.execute(popular_posts_query)
            popular_posts = popular_posts_result.scalars().all()
            
            return {
                "period_days": days,
                "total_posts": total_posts,
                "posts_by_category": posts_by_category,
                "popular_posts": [
                    {
                        "id": post.id,
                        "title": post.title,
                        "view_count": post.view_count,
                        "category": post.category
                    }
                    for post in popular_posts
                ]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get search analytics: {str(e)}")
    
    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply additional filters to the query."""
        if filters.get("category"):
            query = query.where(BlogPost.category == filters["category"])
        
        if filters.get("tags"):
            for tag in filters["tags"]:
                query = query.where(BlogPost.tags.contains([tag]))
        
        if filters.get("author_id"):
            query = query.where(BlogPost.author_id == filters["author_id"])
        
        if filters.get("date_from"):
            query = query.where(BlogPost.created_at >= filters["date_from"])
        
        if filters.get("date_to"):
            query = query.where(BlogPost.created_at <= filters["date_to"])
        
        return query
    
    def _calculate_relevance_score(self, post: BlogPost, query: str) -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        query_lower = query.lower()
        
        # Title match (highest weight)
        if query_lower in post.title.lower():
            score += 3.0
        
        # Content match
        if query_lower in post.content.lower():
            score += 1.0
        
        # Excerpt match
        if post.excerpt and query_lower in post.excerpt.lower():
            score += 2.0
        
        # Tag match
        if post.tags:
            for tag in post.tags:
                if query_lower in tag.lower():
                    score += 1.5
        
        # Boost score for popular posts
        score += min(post.view_count / 1000, 1.0)
        score += min(post.like_count / 100, 0.5)
        
        return score
    
    def _get_matched_fields(self, post: BlogPost, query: str) -> List[str]:
        """Get list of fields that matched the query."""
        matched_fields = []
        query_lower = query.lower()
        
        if query_lower in post.title.lower():
            matched_fields.append("title")
        
        if query_lower in post.content.lower():
            matched_fields.append("content")
        
        if post.excerpt and query_lower in post.excerpt.lower():
            matched_fields.append("excerpt")
        
        if post.tags:
            for tag in post.tags:
                if query_lower in tag.lower():
                    matched_fields.append("tags")
                    break
        
        return matched_fields






























