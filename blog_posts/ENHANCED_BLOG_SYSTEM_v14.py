"""
Enhanced Blog System v14.0.0 - OPTIMIZED ARCHITECTURE
A high-performance, scalable blog system with advanced features
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import uuid
import json

# Core dependencies
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# API and web framework
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
import uvicorn

# Database
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

# Caching and performance
import redis
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Monitoring and logging
import structlog
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# AI/ML components
from transformers import AutoTokenizer, AutoModel
import sentence_transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Search and indexing
from elasticsearch import Elasticsearch
import whoosh
from whoosh.fields import Schema, TEXT, DATETIME, NUMERIC
from whoosh.index import create_in
from whoosh.qparser import QueryParser

# Security
import bcrypt
import jwt
from cryptography.fernet import Fernet

# Configuration
from pydantic_settings import BaseSettings
from typing import Optional

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
BLOG_POSTS_CREATED = Counter('blog_posts_created_total', 'Total blog posts created')
BLOG_POSTS_READ = Counter('blog_posts_read_total', 'Total blog posts read')
BLOG_POSTS_UPDATED = Counter('blog_posts_updated_total', 'Total blog posts updated')
BLOG_POSTS_DELETED = Counter('blog_posts_deleted_total', 'Total blog posts deleted')
BLOG_POSTS_SEARCH = Counter('blog_posts_search_total', 'Total blog posts searches')
BLOG_POSTS_PROCESSING_TIME = Histogram('blog_posts_processing_seconds', 'Time spent processing blog posts')
BLOG_POSTS_ACTIVE = Gauge('blog_posts_active', 'Active blog posts')

# Configuration
class BlogSystemConfig(BaseSettings):
    """Configuration for the enhanced blog system"""
    
    # Database
    database_url: str = "postgresql://user:password@localhost/blog_db"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # AI/ML
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_sequence_length: int = 512
    embedding_dimension: int = 384
    
    # Search
    elasticsearch_url: str = "http://localhost:9200"
    search_index_name: str = "blog_posts"
    
    # Performance
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_requests: int = 100
    batch_size: int = 32
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"

# Enums
class PostStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    SCHEDULED = "scheduled"

class PostCategory(Enum):
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    BUSINESS = "business"
    LIFESTYLE = "lifestyle"
    TRAVEL = "travel"
    FOOD = "food"
    HEALTH = "health"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"

class SearchType(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

# Database Models
Base = declarative_base()

class BlogPost(Base):
    """Enhanced blog post model with advanced features"""
    __tablename__ = "blog_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4, index=True)
    title = Column(String(500), nullable=False, index=True)
    slug = Column(String(500), unique=True, nullable=False, index=True)
    content = Column(Text, nullable=False)
    excerpt = Column(Text, nullable=True)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default=PostStatus.DRAFT.value, index=True)
    category = Column(String(50), default=PostCategory.OTHER.value, index=True)
    tags = Column(JSONB, default=list)
    metadata = Column(JSONB, default=dict)
    
    # SEO and analytics
    seo_title = Column(String(500), nullable=True)
    seo_description = Column(Text, nullable=True)
    seo_keywords = Column(JSONB, default=list)
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    published_at = Column(DateTime(timezone=True), nullable=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    
    # AI/ML features
    embedding = Column(JSONB, nullable=True)
    sentiment_score = Column(Integer, nullable=True)
    readability_score = Column(Integer, nullable=True)
    topic_tags = Column(JSONB, default=list)
    
    # Relationships
    author = relationship("User", back_populates="blog_posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="post", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_blog_posts_status_published', 'status', 'published_at'),
        Index('idx_blog_posts_category_status', 'category', 'status'),
        Index('idx_blog_posts_author_status', 'author_id', 'status'),
    )

class Comment(Base):
    """Blog post comments"""
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("blog_posts.id"), nullable=False)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("comments.id"), nullable=True)
    content = Column(Text, nullable=False)
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    post = relationship("BlogPost", back_populates="comments")
    author = relationship("User")
    replies = relationship("Comment", back_populates="parent")
    parent = relationship("Comment", remote_side=[id], back_populates="replies")

class Like(Base):
    """Blog post likes"""
    __tablename__ = "likes"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("blog_posts.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    post = relationship("BlogPost", back_populates="likes")
    user = relationship("User")

# Pydantic Models
class BlogPostBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    excerpt: Optional[str] = None
    category: PostCategory = PostCategory.OTHER
    tags: List[str] = Field(default_factory=list)
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_keywords: List[str] = Field(default_factory=list)
    scheduled_at: Optional[datetime] = None

class BlogPostCreate(BlogPostBase):
    pass

class BlogPostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    excerpt: Optional[str] = None
    status: Optional[PostStatus] = None
    category: Optional[PostCategory] = None
    tags: Optional[List[str]] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_keywords: Optional[List[str]] = None
    scheduled_at: Optional[datetime] = None

class BlogPostResponse(BlogPostBase):
    id: int
    uuid: str
    slug: str
    author_id: str
    status: PostStatus
    view_count: int
    like_count: int
    share_count: int
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime]
    scheduled_at: Optional[datetime]
    sentiment_score: Optional[int]
    readability_score: Optional[int]
    topic_tags: List[str]
    
    class Config:
        from_attributes = True

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    search_type: SearchType = SearchType.HYBRID
    category: Optional[PostCategory] = None
    tags: Optional[List[str]] = None
    author_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

class SearchResponse(BaseModel):
    results: List[BlogPostResponse]
    total: int
    query: str
    search_type: SearchType
    processing_time: float

# AI/ML Components
class ContentAnalyzer:
    """Analyzes blog post content for various metrics"""
    
    def __init__(self, config: BlogSystemConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.sentence_model = sentence_transformers.SentenceTransformer(config.model_name)
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding for text"""
        embedding = self.sentence_model.encode(text)
        return embedding.tolist()
    
    def analyze_sentiment(self, text: str) -> int:
        """Analyze sentiment of text (simplified implementation)"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return -1
        else:
            return 0
    
    def calculate_readability(self, text: str) -> int:
        """Calculate Flesch Reading Ease score"""
        sentences = text.split('.')
        words = text.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0, min(100, int(score)))
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topic tags from text"""
        # Simplified topic extraction
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            'technology': ['tech', 'software', 'programming', 'ai', 'machine learning'],
            'business': ['business', 'startup', 'entrepreneur', 'marketing', 'finance'],
            'science': ['science', 'research', 'study', 'discovery', 'experiment'],
            'health': ['health', 'fitness', 'wellness', 'medical', 'nutrition'],
            'travel': ['travel', 'trip', 'vacation', 'destination', 'adventure']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics

# Search Engine
class SearchEngine:
    """Advanced search engine with multiple search types"""
    
    def __init__(self, config: BlogSystemConfig):
        self.config = config
        self.es_client = Elasticsearch([config.elasticsearch_url])
        self.content_analyzer = ContentAnalyzer(config)
        
    async def search_posts(self, request: SearchRequest) -> SearchResponse:
        """Search blog posts with various strategies"""
        start_time = time.time()
        
        if request.search_type == SearchType.SEMANTIC:
            results = await self._semantic_search(request)
        elif request.search_type == SearchType.FUZZY:
            results = await self._fuzzy_search(request)
        elif request.search_type == SearchType.EXACT:
            results = await self._exact_search(request)
        else:  # HYBRID
            results = await self._hybrid_search(request)
        
        processing_time = time.time() - start_time
        BLOG_POSTS_SEARCH.inc()
        
        return SearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            search_type=request.search_type,
            processing_time=processing_time
        )
    
    async def _semantic_search(self, request: SearchRequest) -> List[BlogPostResponse]:
        """Semantic search using embeddings"""
        query_embedding = self.content_analyzer.generate_embedding(request.query)
        
        # Search in Elasticsearch with vector similarity
        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": request.limit,
            "from": request.offset
        }
        
        response = self.es_client.search(
            index=self.config.search_index_name,
            body=search_body
        )
        
        # Convert to BlogPostResponse objects
        results = []
        for hit in response['hits']['hits']:
            # This would need to be implemented with actual database queries
            pass
        
        return results
    
    async def _fuzzy_search(self, request: SearchRequest) -> List[BlogPostResponse]:
        """Fuzzy search for typos and variations"""
        search_body = {
            "query": {
                "multi_match": {
                    "query": request.query,
                    "fields": ["title^2", "content", "excerpt"],
                    "fuzziness": "AUTO",
                    "type": "best_fields"
                }
            },
            "size": request.limit,
            "from": request.offset
        }
        
        response = self.es_client.search(
            index=self.config.search_index_name,
            body=search_body
        )
        
        # Convert to BlogPostResponse objects
        results = []
        for hit in response['hits']['hits']:
            # This would need to be implemented with actual database queries
            pass
        
        return results
    
    async def _exact_search(self, request: SearchRequest) -> List[BlogPostResponse]:
        """Exact match search"""
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"match_phrase": {"content": request.query}}
                    ]
                }
            },
            "size": request.limit,
            "from": request.offset
        }
        
        response = self.es_client.search(
            index=self.config.search_index_name,
            body=search_body
        )
        
        # Convert to BlogPostResponse objects
        results = []
        for hit in response['hits']['hits']:
            # This would need to be implemented with actual database queries
            pass
        
        return results
    
    async def _hybrid_search(self, request: SearchRequest) -> List[BlogPostResponse]:
        """Combines semantic and keyword search"""
        # Combine semantic and fuzzy search results
        semantic_results = await self._semantic_search(request)
        fuzzy_results = await self._fuzzy_search(request)
        
        # Merge and rank results
        all_results = semantic_results + fuzzy_results
        # Remove duplicates and re-rank
        unique_results = list({post.id: post for post in all_results}.values())
        
        return unique_results[:request.limit]

# Cache Manager
class CacheManager:
    """Redis-based caching for improved performance"""
    
    def __init__(self, config: BlogSystemConfig):
        self.config = config
        self.redis_client = redis.Redis.from_url(config.redis_url)
        self.ttl = config.cache_ttl
    
    async def get_cached_post(self, post_id: int) -> Optional[BlogPostResponse]:
        """Get cached blog post"""
        key = f"blog_post:{post_id}"
        cached_data = self.redis_client.get(key)
        
        if cached_data:
            return BlogPostResponse.parse_raw(cached_data)
        return None
    
    async def cache_post(self, post: BlogPostResponse) -> None:
        """Cache blog post"""
        key = f"blog_post:{post.id}"
        self.redis_client.setex(
            key,
            self.ttl,
            post.json()
        )
    
    async def invalidate_post_cache(self, post_id: int) -> None:
        """Invalidate cached post"""
        key = f"blog_post:{post_id}"
        self.redis_client.delete(key)
    
    async def get_cached_search_results(self, query: str, params: dict) -> Optional[SearchResponse]:
        """Get cached search results"""
        key = f"search:{hash(query + str(params))}"
        cached_data = self.redis_client.get(key)
        
        if cached_data:
            return SearchResponse.parse_raw(cached_data)
        return None
    
    async def cache_search_results(self, query: str, params: dict, results: SearchResponse) -> None:
        """Cache search results"""
        key = f"search:{hash(query + str(params))}"
        self.redis_client.setex(
            key,
            self.ttl // 2,  # Shorter TTL for search results
            results.json()
        )

# Blog Service
class BlogService:
    """Main blog service with business logic"""
    
    def __init__(self, config: BlogSystemConfig):
        self.config = config
        self.engine = create_engine(config.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.content_analyzer = ContentAnalyzer(config)
        self.search_engine = SearchEngine(config)
        self.cache_manager = CacheManager(config)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_db(self) -> Session:
        """Get database session"""
        db = self.SessionLocal()
        try:
            return db
        except Exception:
            db.close()
            raise
    
    async def create_post(self, post_data: BlogPostCreate, author_id: str) -> BlogPostResponse:
        """Create a new blog post with AI analysis"""
        start_time = time.time()
        
        db = self.get_db()
        try:
            # Generate slug
            slug = self._generate_slug(post_data.title)
            
            # Analyze content
            embedding = self.content_analyzer.generate_embedding(post_data.content)
            sentiment_score = self.content_analyzer.analyze_sentiment(post_data.content)
            readability_score = self.content_analyzer.calculate_readability(post_data.content)
            topic_tags = self.content_analyzer.extract_topics(post_data.content)
            
            # Create post
            db_post = BlogPost(
                title=post_data.title,
                slug=slug,
                content=post_data.content,
                excerpt=post_data.excerpt,
                author_id=author_id,
                category=post_data.category,
                tags=post_data.tags,
                seo_title=post_data.seo_title,
                seo_description=post_data.seo_description,
                seo_keywords=post_data.seo_keywords,
                scheduled_at=post_data.scheduled_at,
                embedding=embedding,
                sentiment_score=sentiment_score,
                readability_score=readability_score,
                topic_tags=topic_tags
            )
            
            db.add(db_post)
            db.commit()
            db.refresh(db_post)
            
            # Convert to response
            response = BlogPostResponse.from_orm(db_post)
            
            # Cache the post
            await self.cache_manager.cache_post(response)
            
            # Update metrics
            BLOG_POSTS_CREATED.inc()
            BLOG_POSTS_PROCESSING_TIME.observe(time.time() - start_time)
            
            return response
            
        finally:
            db.close()
    
    async def get_post(self, post_id: int) -> Optional[BlogPostResponse]:
        """Get blog post with caching"""
        # Try cache first
        cached_post = await self.cache_manager.get_cached_post(post_id)
        if cached_post:
            BLOG_POSTS_READ.inc()
            return cached_post
        
        # Get from database
        db = self.get_db()
        try:
            db_post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
            if not db_post:
                return None
            
            response = BlogPostResponse.from_orm(db_post)
            
            # Cache the post
            await self.cache_manager.cache_post(response)
            
            BLOG_POSTS_READ.inc()
            return response
            
        finally:
            db.close()
    
    async def update_post(self, post_id: int, post_data: BlogPostUpdate, author_id: str) -> Optional[BlogPostResponse]:
        """Update blog post"""
        start_time = time.time()
        
        db = self.get_db()
        try:
            db_post = db.query(BlogPost).filter(
                BlogPost.id == post_id,
                BlogPost.author_id == author_id
            ).first()
            
            if not db_post:
                return None
            
            # Update fields
            update_data = post_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_post, field, value)
            
            # Re-analyze content if content changed
            if post_data.content:
                embedding = self.content_analyzer.generate_embedding(post_data.content)
                sentiment_score = self.content_analyzer.analyze_sentiment(post_data.content)
                readability_score = self.content_analyzer.calculate_readability(post_data.content)
                topic_tags = self.content_analyzer.extract_topics(post_data.content)
                
                db_post.embedding = embedding
                db_post.sentiment_score = sentiment_score
                db_post.readability_score = readability_score
                db_post.topic_tags = topic_tags
            
            db.commit()
            db.refresh(db_post)
            
            # Convert to response
            response = BlogPostResponse.from_orm(db_post)
            
            # Update cache
            await self.cache_manager.cache_post(response)
            
            # Update metrics
            BLOG_POSTS_UPDATED.inc()
            BLOG_POSTS_PROCESSING_TIME.observe(time.time() - start_time)
            
            return response
            
        finally:
            db.close()
    
    async def delete_post(self, post_id: int, author_id: str) -> bool:
        """Delete blog post"""
        db = self.get_db()
        try:
            db_post = db.query(BlogPost).filter(
                BlogPost.id == post_id,
                BlogPost.author_id == author_id
            ).first()
            
            if not db_post:
                return False
            
            db.delete(db_post)
            db.commit()
            
            # Invalidate cache
            await self.cache_manager.invalidate_post_cache(post_id)
            
            # Update metrics
            BLOG_POSTS_DELETED.inc()
            
            return True
            
        finally:
            db.close()
    
    async def search_posts(self, request: SearchRequest) -> SearchResponse:
        """Search blog posts"""
        # Try cache first
        cache_params = {
            'search_type': request.search_type.value,
            'category': request.category.value if request.category else None,
            'limit': request.limit,
            'offset': request.offset
        }
        
        cached_results = await self.cache_manager.get_cached_search_results(
            request.query, cache_params
        )
        if cached_results:
            return cached_results
        
        # Perform search
        results = await self.search_engine.search_posts(request)
        
        # Cache results
        await self.cache_manager.cache_search_results(
            request.query, cache_params, results
        )
        
        return results
    
    def _generate_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title"""
        import re
        import unicodedata
        
        # Normalize unicode
        slug = unicodedata.normalize('NFKD', title)
        
        # Convert to lowercase and replace spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', slug.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        return slug

# FastAPI Application
app = FastAPI(
    title="Enhanced Blog System v14.0.0",
    description="A high-performance, scalable blog system with advanced features",
    version="14.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
config = BlogSystemConfig()
blog_service = BlogService(config)

# Security
security = HTTPBearer()

# Dependency
def get_current_user(token: str = Depends(security)):
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(token.credentials, config.secret_key, algorithms=[config.algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Routes
@app.post("/posts/", response_model=BlogPostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post_data: BlogPostCreate,
    current_user: str = Depends(get_current_user)
):
    """Create a new blog post"""
    return await blog_service.create_post(post_data, current_user)

@app.get("/posts/{post_id}", response_model=BlogPostResponse)
async def get_post(post_id: int):
    """Get a blog post by ID"""
    post = await blog_service.get_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.put("/posts/{post_id}", response_model=BlogPostResponse)
async def update_post(
    post_id: int,
    post_data: BlogPostUpdate,
    current_user: str = Depends(get_current_user)
):
    """Update a blog post"""
    post = await blog_service.update_post(post_id, post_data, current_user)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: int,
    current_user: str = Depends(get_current_user)
):
    """Delete a blog post"""
    success = await blog_service.delete_post(post_id, current_user)
    if not success:
        raise HTTPException(status_code=404, detail="Post not found")

@app.post("/search/", response_model=SearchResponse)
async def search_posts(request: SearchRequest):
    """Search blog posts"""
    return await blog_service.search_posts(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "14.0.0"}

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "posts_created": BLOG_POSTS_CREATED._value.get(),
        "posts_read": BLOG_POSTS_READ._value.get(),
        "posts_updated": BLOG_POSTS_UPDATED._value.get(),
        "posts_deleted": BLOG_POSTS_DELETED._value.get(),
        "searches_performed": BLOG_POSTS_SEARCH._value.get(),
    }

# Initialize Sentry if configured
if config.sentry_dsn:
    sentry_sdk.init(
        dsn=config.sentry_dsn,
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
    )

if __name__ == "__main__":
    uvicorn.run(
        "ENHANCED_BLOG_SYSTEM_v14:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 