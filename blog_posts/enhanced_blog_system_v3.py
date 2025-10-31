"""
🚀 ENHANCED BLOG SYSTEM V3
===========================

Advanced blog system with enterprise features:
- Full-text search with Elasticsearch
- Real-time analytics and metrics
- AI-powered content analysis
- WebSocket notifications
- Advanced pagination and filtering
- Content recommendation engine
- SEO optimization
- Social media integration
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import traceback

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.background import BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# Database and ORM
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, update, delete, text, func, desc, asc
from sqlalchemy.pool import QueuePool

# Search and analytics
try:
    from elasticsearch import AsyncElasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

# AI and ML
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Caching and performance
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from cachetools import TTLCache, LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

# Monitoring and metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Performance optimization
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    EVENT_LOOP = "uvloop"
except ImportError:
    EVENT_LOOP = "asyncio"

# JSON optimization
try:
    import orjson
    JSON_LIB = "orjson"
except ImportError:
    JSON_LIB = "json"

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class SearchConfig(BaseModel):
    """Search configuration."""
    elasticsearch_url: Optional[str] = Field(default="http://localhost:9200")
    enable_full_text_search: bool = Field(default=True)
    enable_semantic_search: bool = Field(default=False)
    search_analytics: bool = Field(default=True)

class AnalyticsConfig(BaseModel):
    """Analytics configuration."""
    enable_real_time_analytics: bool = Field(default=True)
    enable_user_tracking: bool = Field(default=True)
    enable_content_analytics: bool = Field(default=True)
    analytics_retention_days: int = Field(default=90)

class AIConfig(BaseModel):
    """AI configuration."""
    enable_content_analysis: bool = Field(default=True)
    enable_recommendations: bool = Field(default=True)
    enable_sentiment_analysis: bool = Field(default=True)
    enable_keyword_extraction: bool = Field(default=True)

class NotificationConfig(BaseModel):
    """Notification configuration."""
    enable_websocket_notifications: bool = Field(default=True)
    enable_email_notifications: bool = Field(default=False)
    enable_push_notifications: bool = Field(default=False)

class EnhancedConfig(BaseModel):
    """Enhanced configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    debug: bool = Field(default=False)

# ============================================================================
# ENHANCED MODELS
# ============================================================================

class BlogPostModel(Base):
    """Enhanced database model for blog posts."""
    __tablename__ = "blog_posts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(nullable=False, index=True)
    content: Mapped[str] = mapped_column(nullable=False)
    excerpt: Mapped[Optional[str]] = mapped_column(nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(nullable=True)  # JSON string
    category: Mapped[Optional[str]] = mapped_column(nullable=True, index=True)
    author: Mapped[Optional[str]] = mapped_column(nullable=True, index=True)
    is_published: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[float] = mapped_column(default=lambda: time.time(), index=True)
    updated_at: Mapped[float] = mapped_column(default=lambda: time.time())
    published_at: Mapped[Optional[float]] = mapped_column(nullable=True, index=True)
    views: Mapped[int] = mapped_column(default=0, index=True)
    likes: Mapped[int] = mapped_column(default=0)
    shares: Mapped[int] = mapped_column(default=0)
    comments_count: Mapped[int] = mapped_column(default=0)
    reading_time: Mapped[Optional[int]] = mapped_column(nullable=True)
    seo_title: Mapped[Optional[str]] = mapped_column(nullable=True)
    seo_description: Mapped[Optional[str]] = mapped_column(nullable=True)
    seo_keywords: Mapped[Optional[str]] = mapped_column(nullable=True)
    featured_image: Mapped[Optional[str]] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(default="draft")  # draft, published, archived

class AnalyticsModel(Base):
    """Analytics tracking model."""
    __tablename__ = "analytics"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(nullable=False, index=True)  # view, like, share, comment
    user_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(nullable=True)
    referrer: Mapped[Optional[str]] = mapped_column(nullable=True)
    timestamp: Mapped[float] = mapped_column(default=lambda: time.time(), index=True)
    session_id: Mapped[Optional[str]] = mapped_column(nullable=True)
    country: Mapped[Optional[str]] = mapped_column(nullable=True)
    city: Mapped[Optional[str]] = mapped_column(nullable=True)

class SearchResult(BaseModel):
    """Search result model."""
    id: int
    title: str
    excerpt: Optional[str] = None
    score: float
    highlights: Optional[List[str]] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: float
    views: int

class AnalyticsEvent(BaseModel):
    """Analytics event model."""
    post_id: int
    event_type: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    referrer: Optional[str] = None
    session_id: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None

class ContentAnalysis(BaseModel):
    """AI content analysis model."""
    sentiment_score: float
    readability_score: float
    keyword_density: Dict[str, float]
    topic_categories: List[str]
    reading_time_minutes: int
    content_quality_score: float
    seo_score: float
    engagement_prediction: float

# ============================================================================
# ENHANCED SERVICES
# ============================================================================

class SearchService:
    """Advanced search service with Elasticsearch."""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.es = None
        if ELASTICSEARCH_AVAILABLE and config.elasticsearch_url:
            self.es = AsyncElasticsearch([config.elasticsearch_url])
    
    async def index_post(self, post: BlogPost) -> None:
        """Index a blog post for search."""
        if not self.es:
            return
        
        doc = {
            "title": post.title,
            "content": post.content,
            "excerpt": post.excerpt,
            "tags": post.tags or [],
            "category": post.category,
            "author": post.author,
            "created_at": post.created_at,
            "views": post.views,
            "status": post.status
        }
        
        await self.es.index(
            index="blog_posts",
            id=post.id,
            body=doc
        )
    
    async def search_posts(self, query: str, limit: int = 20, offset: int = 0) -> List[SearchResult]:
        """Search blog posts."""
        if not self.es:
            return []
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content^2", "excerpt", "tags", "category"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {"fragment_size": 150, "number_of_fragments": 3},
                    "excerpt": {}
                }
            },
            "sort": [{"_score": {"order": "desc"}}, {"created_at": {"order": "desc"}}],
            "from": offset,
            "size": limit
        }
        
        response = await self.es.search(
            index="blog_posts",
            body=search_body
        )
        
        results = []
        for hit in response["hits"]["hits"]:
            highlights = []
            if "highlight" in hit:
                for field, fragments in hit["highlight"].items():
                    highlights.extend(fragments)
            
            results.append(SearchResult(
                id=hit["_id"],
                title=hit["_source"]["title"],
                excerpt=hit["_source"].get("excerpt"),
                score=hit["_score"],
                highlights=highlights,
                category=hit["_source"].get("category"),
                tags=hit["_source"].get("tags", []),
                created_at=hit["_source"]["created_at"],
                views=hit["_source"]["views"]
            ))
        
        return results

class AnalyticsService:
    """Real-time analytics service."""
    
    def __init__(self, db_manager: DatabaseManager, config: AnalyticsConfig):
        self.db_manager = db_manager
        self.config = config
        self.real_time_stats = {}
    
    async def track_event(self, event: AnalyticsEvent) -> None:
        """Track an analytics event."""
        async with self.db_manager.get_session() as session:
            analytics_record = AnalyticsModel(
                post_id=event.post_id,
                event_type=event.event_type,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                referrer=event.referrer,
                session_id=event.session_id,
                country=event.country,
                city=event.city
            )
            session.add(analytics_record)
            await session.commit()
        
        # Update real-time stats
        key = f"stats:{event.post_id}:{event.event_type}"
        self.real_time_stats[key] = self.real_time_stats.get(key, 0) + 1
    
    async def get_post_analytics(self, post_id: int, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific post."""
        async with self.db_manager.get_session() as session:
            # Get event counts
            stmt = select(
                AnalyticsModel.event_type,
                func.count(AnalyticsModel.id).label("count")
            ).where(
                AnalyticsModel.post_id == post_id,
                AnalyticsModel.timestamp >= time.time() - (days * 24 * 3600)
            ).group_by(AnalyticsModel.event_type)
            
            result = await session.execute(stmt)
            event_counts = {row.event_type: row.count for row in result}
            
            # Get daily trends
            stmt = select(
                func.date(func.datetime(AnalyticsModel.timestamp, 'unixepoch')).label("date"),
                func.count(AnalyticsModel.id).label("count")
            ).where(
                AnalyticsModel.post_id == post_id,
                AnalyticsModel.timestamp >= time.time() - (days * 24 * 3600)
            ).group_by(
                func.date(func.datetime(AnalyticsModel.timestamp, 'unixepoch'))
            ).order_by(desc("date"))
            
            result = await session.execute(stmt)
            daily_trends = [{"date": row.date, "count": row.count} for row in result]
            
            return {
                "event_counts": event_counts,
                "daily_trends": daily_trends,
                "real_time_stats": self.real_time_stats
            }

class AIService:
    """AI-powered content analysis service."""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.vectorizer = None
        if ML_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
    
    def analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content using AI/ML."""
        if not ML_AVAILABLE:
            return ContentAnalysis(
                sentiment_score=0.5,
                readability_score=0.7,
                keyword_density={},
                topic_categories=[],
                reading_time_minutes=len(content.split()) // 200,
                content_quality_score=0.8,
                seo_score=0.7,
                engagement_prediction=0.6
            )
        
        # Sentiment analysis (simplified)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "best"]
        negative_words = ["bad", "terrible", "awful", "worst", "horrible", "disappointing"]
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        sentiment_score = (positive_count - negative_count) / max(total_words, 1) + 0.5
        
        # Readability score (simplified)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        readability_score = max(0, 1 - (avg_sentence_length - 15) / 30)
        
        # Keyword density
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        keyword_density = {word: freq / total_words for word, freq in 
                          sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]}
        
        # Reading time
        reading_time_minutes = max(1, total_words // 200)
        
        # Content quality score
        content_quality_score = min(1.0, (len(content) / 1000) * 0.3 + 
                                  (len(sentences) / 10) * 0.3 + 
                                  (len(set(words)) / len(words)) * 0.4)
        
        # SEO score
        seo_score = min(1.0, (len(content) / 300) * 0.4 + 
                       (len(sentences) / 5) * 0.3 + 
                       (len(keyword_density) / 5) * 0.3)
        
        # Engagement prediction
        engagement_prediction = (sentiment_score * 0.3 + 
                               readability_score * 0.3 + 
                               content_quality_score * 0.4)
        
        return ContentAnalysis(
            sentiment_score=sentiment_score,
            readability_score=readability_score,
            keyword_density=keyword_density,
            topic_categories=["technology", "business"],  # Simplified
            reading_time_minutes=reading_time_minutes,
            content_quality_score=content_quality_score,
            seo_score=seo_score,
            engagement_prediction=engagement_prediction
        )
    
    def get_recommendations(self, post_id: int, posts: List[BlogPost], limit: int = 5) -> List[BlogPost]:
        """Get content recommendations."""
        if not ML_AVAILABLE or not self.vectorizer:
            return posts[:limit]
        
        # Create content matrix
        content_matrix = [f"{post.title} {post.content}" for post in posts]
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(content_matrix)
        
        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Get recommendations for the target post
        target_idx = next((i for i, post in enumerate(posts) if post.id == post_id), 0)
        similar_indices = similarities[target_idx].argsort()[::-1][1:limit+1]
        
        return [posts[i] for i in similar_indices if i != target_idx]

class NotificationService:
    """Real-time notification service."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except WebSocketDisconnect:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def notify_post_created(self, post: BlogPost):
        """Notify about new post creation."""
        await self.broadcast({
            "type": "post_created",
            "post_id": post.id,
            "title": post.title,
            "author": post.author,
            "timestamp": time.time()
        })
    
    async def notify_post_updated(self, post: BlogPost):
        """Notify about post update."""
        await self.broadcast({
            "type": "post_updated",
            "post_id": post.id,
            "title": post.title,
            "timestamp": time.time()
        })

# ============================================================================
# ENHANCED BLOG SERVICE
# ============================================================================

class EnhancedBlogService:
    """Enhanced blog service with advanced features."""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager,
                 search_service: SearchService, analytics_service: AnalyticsService,
                 ai_service: AIService, notification_service: NotificationService):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.search_service = search_service
        self.analytics_service = analytics_service
        self.ai_service = ai_service
        self.notification_service = notification_service
    
    async def list_posts(self, limit: int = 100, offset: int = 0, 
                        category: Optional[str] = None, author: Optional[str] = None,
                        status: Optional[str] = None, sort_by: str = "created_at",
                        sort_order: str = "desc") -> Tuple[List[BlogPost], int]:
        """List posts with advanced filtering and pagination."""
        cache_key = f"posts:list:{limit}:{offset}:{category}:{author}:{status}:{sort_by}:{sort_order}"
        
        # Try cache first
        cached_result = await self.cache_manager.get("posts", "list", cache_key)
        if cached_result:
            return cached_result["posts"], cached_result["total"]
        
        async with self.db_manager.get_session() as session:
            # Build query
            query = select(BlogPostModel)
            
            # Apply filters
            if category:
                query = query.where(BlogPostModel.category == category)
            if author:
                query = query.where(BlogPostModel.author == author)
            if status:
                query = query.where(BlogPostModel.status == status)
            
            # Get total count
            count_query = select(func.count(BlogPostModel.id)).select_from(query.subquery())
            total_result = await session.execute(count_query)
            total = total_result.scalar()
            
            # Apply sorting
            if sort_by == "created_at":
                query = query.order_by(desc(BlogPostModel.created_at) if sort_order == "desc" 
                                     else asc(BlogPostModel.created_at))
            elif sort_by == "views":
                query = query.order_by(desc(BlogPostModel.views) if sort_order == "desc" 
                                     else asc(BlogPostModel.views))
            elif sort_by == "title":
                query = query.order_by(desc(BlogPostModel.title) if sort_order == "desc" 
                                     else asc(BlogPostModel.title))
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            result = await session.execute(query)
            posts = []
            
            for row in result.scalars():
                post_dict = {
                    "id": row.id,
                    "title": row.title,
                    "content": row.content,
                    "excerpt": row.excerpt,
                    "tags": json.loads(row.tags) if row.tags else [],
                    "category": row.category,
                    "author": row.author,
                    "is_published": row.is_published,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "published_at": row.published_at,
                    "views": row.views,
                    "likes": row.likes,
                    "shares": row.shares,
                    "comments_count": row.comments_count,
                    "reading_time": row.reading_time,
                    "seo_title": row.seo_title,
                    "seo_description": row.seo_description,
                    "seo_keywords": row.seo_keywords,
                    "featured_image": row.featured_image,
                    "status": row.status
                }
                posts.append(BlogPost(**post_dict))
            
            # Cache result
            await self.cache_manager.set("posts", "list", cache_key, 
                                       {"posts": posts, "total": total}, ttl=300)
            
            return posts, total
    
    async def search_posts(self, query: str, limit: int = 20, offset: int = 0) -> List[SearchResult]:
        """Search posts using Elasticsearch."""
        return await self.search_service.search_posts(query, limit, offset)
    
    async def get_post_analytics(self, post_id: int, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a post."""
        return await self.analytics_service.get_post_analytics(post_id, days)
    
    async def analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content using AI."""
        return self.ai_service.analyze_content(content)
    
    async def get_recommendations(self, post_id: int, limit: int = 5) -> List[BlogPost]:
        """Get content recommendations."""
        posts, _ = await self.list_posts(limit=100)
        return self.ai_service.get_recommendations(post_id, posts, limit)

# ============================================================================
# ENHANCED FASTAPI APPLICATION
# ============================================================================

class EnhancedBlogSystem:
    """Enhanced blog system with advanced features."""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.app = FastAPI(
            title="Enhanced Blog System",
            description="Advanced blog system with search, analytics, and AI features",
            version="3.0.0",
            debug=config.debug
        )
        
        # Initialize components
        self.db_manager = DatabaseManager(config.database)
        self.cache_manager = CacheManager(config.cache)
        self.search_service = SearchService(config.search)
        self.analytics_service = AnalyticsService(self.db_manager, config.analytics)
        self.ai_service = AIService(config.ai)
        self.notification_service = NotificationService(config.notifications)
        self.blog_service = EnhancedBlogService(
            self.db_manager, self.cache_manager, self.search_service,
            self.analytics_service, self.ai_service, self.notification_service
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup exception handlers
        self._setup_exception_handlers()
    
    def _setup_middleware(self):
        """Setup application middleware."""
        # Performance middleware
        self.app.add_middleware(PerformanceMiddleware)
        
        # Rate limiting
        if self.config.performance.rate_limit_requests > 0:
            self.app.add_middleware(RateLimitMiddleware, self.config.performance.rate_limit_requests)
        
        # CORS
        if self.config.performance.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Gzip compression
        if self.config.performance.enable_gzip:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup enhanced API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Enhanced health check."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "3.0.0",
                "features": {
                    "search": ELASTICSEARCH_AVAILABLE,
                    "ai": ML_AVAILABLE,
                    "cache": REDIS_AVAILABLE,
                    "monitoring": PSUTIL_AVAILABLE
                }
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Enhanced metrics endpoint."""
            metrics = {
                "timestamp": time.time(),
                "system": {}
            }
            
            if PSUTIL_AVAILABLE:
                metrics["system"] = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            
            return metrics
        
        @self.app.get("/posts", response_model=Dict[str, Any])
        async def list_posts(
            limit: int = 100,
            offset: int = 0,
            category: Optional[str] = None,
            author: Optional[str] = None,
            status: Optional[str] = None,
            sort_by: str = "created_at",
            sort_order: str = "desc"
        ):
            """Enhanced list posts with filtering and pagination."""
            posts, total = await self.blog_service.list_posts(
                limit, offset, category, author, status, sort_by, sort_order
            )
            
            return {
                "posts": posts,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        
        @self.app.get("/posts/search")
        async def search_posts(query: str, limit: int = 20, offset: int = 0):
            """Search posts."""
            results = await self.blog_service.search_posts(query, limit, offset)
            return {
                "results": results,
                "query": query,
                "total": len(results)
            }
        
        @self.app.get("/posts/{post_id}/analytics")
        async def get_post_analytics(post_id: int, days: int = 30):
            """Get post analytics."""
            analytics = await self.blog_service.get_post_analytics(post_id, days)
            return analytics
        
        @self.app.post("/posts/{post_id}/track")
        async def track_post_event(post_id: int, event: AnalyticsEvent):
            """Track post event."""
            event.post_id = post_id
            await self.analytics_service.track_event(event)
            return {"status": "tracked"}
        
        @self.app.post("/content/analyze")
        async def analyze_content(content: str):
            """Analyze content using AI."""
            analysis = await self.blog_service.analyze_content(content)
            return analysis
        
        @self.app.get("/posts/{post_id}/recommendations")
        async def get_recommendations(post_id: int, limit: int = 5):
            """Get content recommendations."""
            recommendations = await self.blog_service.get_recommendations(post_id, limit)
            return recommendations
        
        @self.app.websocket("/ws/notifications")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time notifications."""
            await self.notification_service.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.notification_service.disconnect(websocket)
    
    def _setup_exception_handlers(self):
        """Setup exception handlers."""
        
        @self.app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            return JSONResponse(
                status_code=422,
                content={"detail": "Validation error", "errors": exc.errors()}
            )
        
        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    async def startup(self):
        """Enhanced startup."""
        # Create tables
        await self.db_manager.create_tables()
        
        # Initialize search index
        if ELASTICSEARCH_AVAILABLE and self.config.search.enable_full_text_search:
            try:
                await self.search_service.es.indices.create(
                    index="blog_posts",
                    ignore=400  # Index already exists
                )
            except Exception as e:
                logging.warning(f"Could not create search index: {e}")
    
    async def shutdown(self):
        """Enhanced shutdown."""
        # Close database connections
        if hasattr(self.db_manager, 'engine'):
            await self.db_manager.engine.dispose()
        
        # Close search connections
        if hasattr(self.search_service, 'es') and self.search_service.es:
            await self.search_service.es.close()
        
        # Close cache connections
        if hasattr(self.cache_manager, 'redis') and self.cache_manager.redis:
            await self.cache_manager.redis.close()

def create_enhanced_blog_system(config: EnhancedConfig = None) -> EnhancedBlogSystem:
    """Create enhanced blog system."""
    if config is None:
        config = EnhancedConfig()
    
    system = EnhancedBlogSystem(config)
    
    @system.app.on_event("startup")
    async def startup_event():
        await system.startup()
    
    @system.app.on_event("shutdown")
    async def shutdown_event():
        await system.shutdown()
    
    return system

if __name__ == "__main__":
    import uvicorn
    
    # Create and run enhanced system
    system = create_enhanced_blog_system()
    
    uvicorn.run(
        system.app,
        host="0.0.0.0",
        port=8000,
        loop=EVENT_LOOP
    ) 
 
 