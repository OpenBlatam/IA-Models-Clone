from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timezone
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import ORJSONResponse
import uvicorn
from pydantic import BaseModel, Field
import httpx
import aiohttp
import asyncpg
import aioredis
from aiocache import cached
from aiocache.serializers import PickleSerializer
import aiofiles
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from circuitbreaker import circuit
import tenacity
import uvloop
import orjson
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
LinkedIn Posts - Dedicated Async Operations Implementation
========================================================

Comprehensive implementation demonstrating dedicated async functions
for database and external API operations with proper connection management,
error handling, and performance optimization.
"""


# FastAPI and async imports

# Async HTTP client

# Database and caching

# File operations

# Monitoring and metrics

# Circuit breaker and retry

# Performance optimization

# Configure uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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
DB_OPERATION_DURATION = Histogram('db_operation_duration_seconds', 'Database operation duration', ['operation'])
DB_OPERATION_ERRORS = Counter('db_operation_errors_total', 'Database operation errors', ['operation'])
API_OPERATION_DURATION = Histogram('api_operation_duration_seconds', 'API operation duration', ['operation'])
API_OPERATION_ERRORS = Counter('api_operation_errors_total', 'API operation errors', ['operation'])
CONNECTION_POOL_SIZE = Gauge('connection_pool_size', 'Connection pool size', ['type'])

# Pydantic models
class PostData(BaseModel):
    content: str = Field(..., min_length=10, max_length=3000)
    post_type: str = Field(default="educational", regex="^(educational|promotional|personal|industry)$")
    tone: str = Field(default="professional", regex="^(professional|casual|enthusiastic|thoughtful)$")
    target_audience: str = Field(default="general", regex="^(general|executives|developers|marketers)$")
    user_id: str = Field(..., description="User ID")
    hashtags: Optional[List[str]] = Field(default_factory=list)
    call_to_action: Optional[str] = None

class PostUpdate(BaseModel):
    content: Optional[str] = None
    post_type: Optional[str] = None
    tone: Optional[str] = None
    target_audience: Optional[str] = None
    hashtags: Optional[List[str]] = None
    call_to_action: Optional[str] = None

class LinkedInPostRequest(BaseModel):
    post_data: PostData
    access_token: str = Field(..., description="LinkedIn access token")
    publish_immediately: bool = Field(default=True, description="Publish immediately to LinkedIn")

# Custom exceptions
class DatabaseError(Exception):
    """Base exception for database operations"""
    pass

class ConnectionError(DatabaseError):
    """Exception for database connection issues"""
    pass

class QueryError(DatabaseError):
    """Exception for database query errors"""
    pass

class APIError(Exception):
    """Base exception for API operations"""
    pass

class RateLimitError(APIError):
    """Exception for rate limiting"""
    pass

class AuthenticationError(APIError):
    """Exception for authentication failures"""
    pass

class TimeoutError(APIError):
    """Exception for timeout errors"""
    pass

# Metrics decorators
def track_database_operation(operation_name: str):
    """Decorator to track database operation metrics"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                DB_OPERATION_DURATION.labels(operation_name).observe(duration)
                return result
            except Exception as e:
                DB_OPERATION_ERRORS.labels(operation_name).inc()
                logger.error(f"Database operation {operation_name} failed", error=str(e))
                raise
        return wrapper
    return decorator

def track_api_operation(operation_name: str):
    """Decorator to track API operation metrics"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                API_OPERATION_DURATION.labels(operation_name).observe(duration)
                return result
            except Exception as e:
                API_OPERATION_ERRORS.labels(operation_name).inc()
                logger.error(f"API operation {operation_name} failed", error=str(e))
                raise
        return wrapper
    return decorator

# Database Connection Pool
class DatabaseConnectionPool:
    """Manages database connection pool with proper lifecycle management"""
    
    def __init__(self, database_url: str, pool_size: int = 20, min_size: int = 5):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.min_size = min_size
        self.pool = None
    
    async def initialize(self) -> Any:
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.pool_size,
                command_timeout=30,
                server_settings={
                    'application_name': 'linkedin_posts_app'
                }
            )
            
            CONNECTION_POOL_SIZE.labels('database').set(self.pool_size)
            logger.info(f"Database connection pool initialized with {self.pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise ConnectionError(f"Database connection pool initialization failed: {e}")
    
    async def close(self) -> Any:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def get_connection(self) -> Optional[Dict[str, Any]]:
        """Get connection from pool"""
        if not self.pool:
            raise ConnectionError("Database connection pool not initialized")
        return await self.pool.acquire()
    
    async def release_connection(self, conn) -> Any:
        """Release connection back to pool"""
        if self.pool:
            await self.pool.release(conn)
    
    async def health_check(self) -> bool:
        """Check if connection pool is healthy"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Dedicated Database Operations
class LinkedInPostsDatabase:
    """Dedicated database operations for LinkedIn posts"""
    
    def __init__(self, connection_pool: DatabaseConnectionPool):
        
    """__init__ function."""
self.pool = connection_pool
    
    @track_database_operation("create_post")
    async def create_post(self, post_data: PostData) -> str:
        """Dedicated function for creating a post"""
        async with self.pool.get_connection() as conn:
            try:
                query = """
                    INSERT INTO linkedin_posts (
                        id, content, post_type, tone, target_audience, 
                        user_id, hashtags, call_to_action, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """
                post_id = str(uuid.uuid4())
                result = await conn.fetchval(
                    query,
                    post_id,
                    post_data.content,
                    post_data.post_type,
                    post_data.tone,
                    post_data.target_audience,
                    post_data.user_id,
                    json.dumps(post_data.hashtags) if post_data.hashtags else None,
                    post_data.call_to_action,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc)
                )
                
                logger.info(f"Post created successfully", post_id=post_id)
                return result
                
            except Exception as e:
                logger.error(f"Failed to create post: {e}")
                raise QueryError(f"Failed to create post: {e}")
    
    @track_database_operation("get_post_by_id")
    @cached(ttl=300, serializer=PickleSerializer())
    async def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Dedicated function for retrieving a post by ID"""
        async with self.pool.get_connection() as conn:
            try:
                query = """
                    SELECT 
                        id, content, post_type, tone, target_audience,
                        user_id, hashtags, call_to_action, 
                        sentiment_score, readability_score, engagement_prediction,
                        views_count, likes_count, comments_count, shares_count,
                        created_at, updated_at, status
                    FROM linkedin_posts 
                    WHERE id = $1
                """
                result = await conn.fetchrow(query, post_id)
                
                if result:
                    post_dict = dict(result)
                    # Parse JSON fields
                    if post_dict.get('hashtags'):
                        post_dict['hashtags'] = json.loads(post_dict['hashtags'])
                    return post_dict
                return None
                
            except Exception as e:
                logger.error(f"Failed to get post by ID: {e}")
                raise QueryError(f"Failed to get post by ID: {e}")
    
    @track_database_operation("update_post")
    async def update_post(self, post_id: str, updates: PostUpdate) -> bool:
        """Dedicated function for updating a post"""
        async with self.pool.get_connection() as conn:
            try:
                # Build dynamic update query
                set_clauses = []
                values = []
                param_count = 1
                
                update_dict = updates.dict(exclude_unset=True)
                
                for key, value in update_dict.items():
                    if key == 'hashtags' and value is not None:
                        set_clauses.append(f"{key} = ${param_count}")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ${param_count}")
                        values.append(value)
                    param_count += 1
                
                if not set_clauses:
                    return False
                
                # Add updated_at timestamp
                set_clauses.append(f"updated_at = ${param_count}")
                values.append(datetime.now(timezone.utc))
                param_count += 1
                
                values.append(post_id)
                query = f"""
                    UPDATE linkedin_posts 
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_count}
                """
                
                result = await conn.execute(query, *values)
                success = result == "UPDATE 1"
                
                if success:
                    logger.info(f"Post updated successfully", post_id=post_id)
                else:
                    logger.warning(f"Post not found for update", post_id=post_id)
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to update post: {e}")
                raise QueryError(f"Failed to update post: {e}")
    
    @track_database_operation("delete_post")
    async def delete_post(self, post_id: str) -> bool:
        """Dedicated function for deleting a post"""
        async with self.pool.get_connection() as conn:
            try:
                query = "DELETE FROM linkedin_posts WHERE id = $1"
                result = await conn.execute(query, post_id)
                success = result == "DELETE 1"
                
                if success:
                    logger.info(f"Post deleted successfully", post_id=post_id)
                else:
                    logger.warning(f"Post not found for deletion", post_id=post_id)
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to delete post: {e}")
                raise QueryError(f"Failed to delete post: {e}")
    
    @track_database_operation("get_posts_by_user")
    async def get_posts_by_user(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Dedicated function for retrieving posts by user with pagination"""
        async with self.pool.get_connection() as conn:
            try:
                query = """
                    SELECT 
                        id, content, post_type, tone, target_audience,
                        user_id, hashtags, call_to_action, 
                        sentiment_score, readability_score, engagement_prediction,
                        views_count, likes_count, comments_count, shares_count,
                        created_at, updated_at, status
                    FROM linkedin_posts 
                    WHERE user_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                """
                results = await conn.fetch(query, user_id, limit, offset)
                
                posts = []
                for row in results:
                    post_dict = dict(row)
                    if post_dict.get('hashtags'):
                        post_dict['hashtags'] = json.loads(post_dict['hashtags'])
                    posts.append(post_dict)
                
                logger.info(f"Retrieved {len(posts)} posts for user", user_id=user_id)
                return posts
                
            except Exception as e:
                logger.error(f"Failed to get posts by user: {e}")
                raise QueryError(f"Failed to get posts by user: {e}")
    
    @track_database_operation("search_posts")
    async def search_posts(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Dedicated function for searching posts"""
        async with self.pool.get_connection() as conn:
            try:
                query = """
                    SELECT 
                        id, content, post_type, tone, target_audience,
                        user_id, hashtags, call_to_action, 
                        sentiment_score, readability_score, engagement_prediction,
                        views_count, likes_count, comments_count, shares_count,
                        created_at, updated_at, status
                    FROM linkedin_posts 
                    WHERE content ILIKE $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                """
                results = await conn.fetch(query, f"%{search_term}%", limit)
                
                posts = []
                for row in results:
                    post_dict = dict(row)
                    if post_dict.get('hashtags'):
                        post_dict['hashtags'] = json.loads(post_dict['hashtags'])
                    posts.append(post_dict)
                
                logger.info(f"Search returned {len(posts)} posts", search_term=search_term)
                return posts
                
            except Exception as e:
                logger.error(f"Failed to search posts: {e}")
                raise QueryError(f"Failed to search posts: {e}")
    
    @track_database_operation("get_post_analytics")
    async def get_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """Dedicated function for retrieving post analytics"""
        async with self.pool.get_connection() as conn:
            try:
                query = """
                    SELECT 
                        sentiment_score,
                        readability_score,
                        engagement_prediction,
                        views_count,
                        likes_count,
                        comments_count,
                        shares_count,
                        created_at,
                        updated_at
                    FROM linkedin_posts 
                    WHERE id = $1
                """
                result = await conn.fetchrow(query, post_id)
                
                if result:
                    analytics = dict(result)
                    logger.info(f"Retrieved analytics for post", post_id=post_id)
                    return analytics
                else:
                    logger.warning(f"Post not found for analytics", post_id=post_id)
                    return {}
                
            except Exception as e:
                logger.error(f"Failed to get post analytics: {e}")
                raise QueryError(f"Failed to get post analytics: {e}")
    
    @track_database_operation("batch_create_posts")
    async def batch_create_posts(self, posts_data: List[PostData]) -> List[str]:
        """Dedicated function for batch creating posts"""
        async with self.pool.get_connection() as conn:
            try:
                async with conn.transaction():
                    post_ids = []
                    for post_data in posts_data:
                        post_id = str(uuid.uuid4())
                        query = """
                            INSERT INTO linkedin_posts (
                                id, content, post_type, tone, target_audience, 
                                user_id, hashtags, call_to_action, created_at, updated_at
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """
                        await conn.execute(
                            query,
                            post_id,
                            post_data.content,
                            post_data.post_type,
                            post_data.tone,
                            post_data.target_audience,
                            post_data.user_id,
                            json.dumps(post_data.hashtags) if post_data.hashtags else None,
                            post_data.call_to_action,
                            datetime.now(timezone.utc),
                            datetime.now(timezone.utc)
                        )
                        post_ids.append(post_id)
                    
                    logger.info(f"Batch created {len(post_ids)} posts")
                    return post_ids
                    
            except Exception as e:
                logger.error(f"Failed to batch create posts: {e}")
                raise QueryError(f"Failed to batch create posts: {e}")
    
    @track_database_operation("update_post_analytics")
    async def update_post_analytics(self, post_id: str, analytics: Dict[str, Any]) -> bool:
        """Dedicated function for updating post analytics"""
        async with self.pool.get_connection() as conn:
            try:
                query = """
                    UPDATE linkedin_posts 
                    SET 
                        sentiment_score = $2,
                        readability_score = $3,
                        engagement_prediction = $4,
                        views_count = $5,
                        likes_count = $6,
                        comments_count = $7,
                        shares_count = $8,
                        updated_at = $9
                    WHERE id = $1
                """
                result = await conn.execute(
                    query,
                    post_id,
                    analytics.get('sentiment_score'),
                    analytics.get('readability_score'),
                    analytics.get('engagement_prediction'),
                    analytics.get('views_count', 0),
                    analytics.get('likes_count', 0),
                    analytics.get('comments_count', 0),
                    analytics.get('shares_count', 0),
                    datetime.now(timezone.utc)
                )
                
                success = result == "UPDATE 1"
                if success:
                    logger.info(f"Post analytics updated", post_id=post_id)
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to update post analytics: {e}")
                raise QueryError(f"Failed to update post analytics: {e}")

# External API Session Management
class ExternalAPISession:
    """Manages HTTP sessions for external API calls"""
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        
    """__init__ function."""
self.base_url = base_url
        self.timeout = timeout
        self.session = None
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
    
    async def initialize(self) -> Any:
        """Initialize HTTP session"""
        try:
            self.session = aiohttp.ClientSession(
                base_url=self.base_url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=self.connector,
                headers={
                    'User-Agent': 'LinkedInPostsApp/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            )
            
            CONNECTION_POOL_SIZE.labels('api').set(100)
            logger.info(f"External API session initialized for {self.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize API session: {e}")
            raise ConnectionError(f"API session initialization failed: {e}")
    
    async def close(self) -> Any:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.info("External API session closed")
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self.close()

# Dedicated LinkedIn API Operations
class LinkedInAPI:
    """Dedicated LinkedIn API operations"""
    
    def __init__(self, session: ExternalAPISession):
        
    """__init__ function."""
self.session = session
    
    @track_api_operation("create_linkedin_post")
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def create_linkedin_post(self, post_data: PostData, access_token: str) -> Dict[str, Any]:
        """Dedicated function for creating a LinkedIn post via API"""
        try:
            # Prepare LinkedIn API payload
            payload = {
                "author": f"urn:li:person:{post_data.user_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": post_data.content
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }
            
            async with self.session.session.post(
                "/v2/ugcPosts",
                json=payload,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    logger.info(f"LinkedIn post created successfully", post_id=result.get('id'))
                    return result
                elif response.status == 429:
                    raise RateLimitError("LinkedIn API rate limit exceeded")
                elif response.status == 401:
                    raise AuthenticationError("LinkedIn API authentication failed")
                else:
                    error_text = await response.text()
                    raise APIError(f"LinkedIn API error: {response.status} - {error_text}")
                    
        except Exception as e:
            if isinstance(e, (RateLimitError, AuthenticationError, APIError)):
                raise
            logger.error(f"Error creating LinkedIn post: {e}")
            raise APIError(f"Error creating LinkedIn post: {e}")
    
    @track_api_operation("get_linkedin_profile")
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def get_linkedin_profile(self, user_id: str, access_token: str) -> Dict[str, Any]:
        """Dedicated function for retrieving LinkedIn profile"""
        try:
            async with self.session.session.get(
                f"/v2/people/{user_id}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"LinkedIn profile retrieved", user_id=user_id)
                    return result
                elif response.status == 401:
                    raise AuthenticationError("LinkedIn API authentication failed")
                else:
                    error_text = await response.text()
                    raise APIError(f"LinkedIn API error: {response.status} - {error_text}")
                    
        except Exception as e:
            if isinstance(e, (AuthenticationError, APIError)):
                raise
            logger.error(f"Error getting LinkedIn profile: {e}")
            raise APIError(f"Error getting LinkedIn profile: {e}")
    
    @track_api_operation("get_linkedin_analytics")
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def get_linkedin_analytics(self, post_id: str, access_token: str) -> Dict[str, Any]:
        """Dedicated function for retrieving LinkedIn post analytics"""
        try:
            async with self.session.session.get(
                f"/v2/socialMetrics/{post_id}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0"
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"LinkedIn analytics retrieved", post_id=post_id)
                    return result
                elif response.status == 404:
                    logger.warning(f"LinkedIn post not found for analytics", post_id=post_id)
                    return {}
                else:
                    error_text = await response.text()
                    raise APIError(f"LinkedIn API error: {response.status} - {error_text}")
                    
        except Exception as e:
            if isinstance(e, APIError):
                raise
            logger.error(f"Error getting LinkedIn analytics: {e}")
            raise APIError(f"Error getting LinkedIn analytics: {e}")

# Dedicated AI Analysis API Operations
class AIAnalysisAPI:
    """Dedicated AI analysis API operations"""
    
    def __init__(self, session: ExternalAPISession):
        
    """__init__ function."""
self.session = session
    
    @track_api_operation("analyze_sentiment")
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((TimeoutError, RateLimitError))
    )
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Dedicated function for sentiment analysis"""
        try:
            async with self.session.session.post(
                "/analyze/sentiment",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Sentiment analysis completed", text_length=len(text))
                    return result
                elif response.status == 429:
                    raise RateLimitError("AI API rate limit exceeded")
                else:
                    error_text = await response.text()
                    raise APIError(f"AI API error: {response.status} - {error_text}")
                    
        except Exception as e:
            if isinstance(e, (RateLimitError, APIError)):
                raise
            logger.error(f"Error analyzing sentiment: {e}")
            raise APIError(f"Error analyzing sentiment: {e}")
    
    @track_api_operation("generate_hashtags")
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((TimeoutError, RateLimitError))
    )
    async def generate_hashtags(self, content: str) -> List[str]:
        """Dedicated function for generating hashtags"""
        try:
            async with self.session.session.post(
                "/generate/hashtags",
                json={"content": content},
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    hashtags = result.get("hashtags", [])
                    logger.info(f"Hashtags generated", count=len(hashtags))
                    return hashtags
                elif response.status == 429:
                    raise RateLimitError("AI API rate limit exceeded")
                else:
                    error_text = await response.text()
                    raise APIError(f"AI API error: {response.status} - {error_text}")
                    
        except Exception as e:
            if isinstance(e, (RateLimitError, APIError)):
                raise
            logger.error(f"Error generating hashtags: {e}")
            raise APIError(f"Error generating hashtags: {e}")
    
    @track_api_operation("optimize_content")
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((TimeoutError, RateLimitError))
    )
    async def optimize_content(self, content: str, optimization_type: str) -> str:
        """Dedicated function for content optimization"""
        try:
            async with self.session.session.post(
                "/optimize/content",
                json={
                    "content": content,
                    "optimization_type": optimization_type
                },
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    optimized_content = result.get("optimized_content", content)
                    logger.info(f"Content optimized", optimization_type=optimization_type)
                    return optimized_content
                elif response.status == 429:
                    raise RateLimitError("AI API rate limit exceeded")
                else:
                    error_text = await response.text()
                    raise APIError(f"AI API error: {response.status} - {error_text}")
                    
        except Exception as e:
            if isinstance(e, (RateLimitError, APIError)):
                raise
            logger.error(f"Error optimizing content: {e}")
            raise APIError(f"Error optimizing content: {e}")

# Dedicated Notification API Operations
class NotificationAPI:
    """Dedicated notification API operations"""
    
    def __init__(self, session: ExternalAPISession):
        
    """__init__ function."""
self.session = session
    
    @track_api_operation("send_email_notification")
    async def send_email_notification(self, user_email: str, subject: str, content: str) -> bool:
        """Dedicated function for sending email notifications"""
        try:
            async with self.session.session.post(
                "/notifications/email",
                json={
                    "to": user_email,
                    "subject": subject,
                    "content": content
                },
                headers={"Content-Type": "application/json"}
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Email notification sent", user_email=user_email)
                else:
                    logger.error(f"Failed to send email notification", status=response.status)
                return success
                
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    @track_api_operation("send_push_notification")
    async def send_push_notification(self, user_id: str, title: str, message: str) -> bool:
        """Dedicated function for sending push notifications"""
        try:
            async with self.session.session.post(
                "/notifications/push",
                json={
                    "user_id": user_id,
                    "title": title,
                    "message": message
                },
                headers={"Content-Type": "application/json"}
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Push notification sent", user_id=user_id)
                else:
                    logger.error(f"Failed to send push notification", status=response.status)
                return success
                
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False

# Main Service Orchestrator
class LinkedInPostsService:
    """Main service orchestrator using dedicated async operations"""
    
    def __init__(self, database: LinkedInPostsDatabase, linkedin_api: LinkedInAPI, 
                 ai_api: AIAnalysisAPI, notification_api: NotificationAPI):
        
    """__init__ function."""
self.database = database
        self.linkedin_api = linkedin_api
        self.ai_api = ai_api
        self.notification_api = notification_api
    
    async def create_post_with_analysis(self, request: LinkedInPostRequest) -> Dict[str, Any]:
        """Create post with AI analysis and LinkedIn publishing"""
        try:
            # Step 1: Create post in database
            post_id = await self.database.create_post(request.post_data)
            
            # Step 2: Perform AI analysis in parallel
            analysis_tasks = [
                self.ai_api.analyze_sentiment(request.post_data.content),
                self.ai_api.generate_hashtags(request.post_data.content)
            ]
            
            sentiment_result, hashtags = await asyncio.gather(*analysis_tasks)
            
            # Step 3: Update post with analysis results
            analytics = {
                'sentiment_score': sentiment_result.get('sentiment_score', 0.0),
                'readability_score': sentiment_result.get('readability_score', 0.0),
                'engagement_prediction': sentiment_result.get('engagement_prediction', 0.0)
            }
            
            await self.database.update_post_analytics(post_id, analytics)
            
            # Step 4: Publish to LinkedIn if requested
            linkedin_result = None
            if request.publish_immediately:
                try:
                    linkedin_result = await self.linkedin_api.create_linkedin_post(
                        request.post_data, 
                        request.access_token
                    )
                except Exception as e:
                    logger.error(f"Failed to publish to LinkedIn: {e}")
            
            # Step 5: Send notifications
            notification_tasks = [
                self.notification_api.send_email_notification(
                    f"user-{request.post_data.user_id}@example.com",
                    "Post Created Successfully",
                    f"Your post '{request.post_data.content[:50]}...' has been created."
                ),
                self.notification_api.send_push_notification(
                    request.post_data.user_id,
                    "Post Created",
                    "Your LinkedIn post has been created successfully!"
                )
            ]
            
            # Fire and forget notifications
            asyncio.create_task(asyncio.gather(*notification_tasks, return_exceptions=True))
            
            # Return comprehensive result
            return {
                'post_id': post_id,
                'database_status': 'created',
                'analytics': analytics,
                'hashtags': hashtags,
                'linkedin_status': 'published' if linkedin_result else 'failed',
                'linkedin_post_id': linkedin_result.get('id') if linkedin_result else None
            }
            
        except Exception as e:
            logger.error(f"Failed to create post with analysis: {e}")
            raise
    
    async def get_post_with_analytics(self, post_id: str) -> Dict[str, Any]:
        """Get post with comprehensive analytics"""
        try:
            # Get post data
            post_data = await self.database.get_post_by_id(post_id)
            if not post_data:
                raise HTTPException(status_code=404, detail="Post not found")
            
            # Get analytics
            analytics = await self.database.get_post_analytics(post_id)
            
            return {
                'post': post_data,
                'analytics': analytics
            }
            
        except Exception as e:
            logger.error(f"Failed to get post with analytics: {e}")
            raise
    
    async def update_post_with_optimization(self, post_id: str, updates: PostUpdate, 
                                          optimization_type: str = "engagement") -> Dict[str, Any]:
        """Update post with AI optimization"""
        try:
            # Get current post
            current_post = await self.database.get_post_by_id(post_id)
            if not current_post:
                raise HTTPException(status_code=404, detail="Post not found")
            
            # Optimize content if provided
            if updates.content:
                optimized_content = await self.ai_api.optimize_content(
                    updates.content, 
                    optimization_type
                )
                updates.content = optimized_content
            
            # Update post
            success = await self.database.update_post(post_id, updates)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update post")
            
            # Get updated post
            updated_post = await self.database.get_post_by_id(post_id)
            
            return {
                'post_id': post_id,
                'update_status': 'success',
                'updated_post': updated_post,
                'optimization_applied': updates.content is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to update post with optimization: {e}")
            raise

# FastAPI Application
class DedicatedAsyncOperationsAPI:
    """FastAPI application with dedicated async operations"""
    
    def __init__(self) -> Any:
        self.app = FastAPI(
            title="LinkedIn Posts - Dedicated Async Operations API",
            description="High-performance LinkedIn posts API with dedicated async operations",
            version="4.0.0"
        )
        
        # Initialize components
        self.db_pool = DatabaseConnectionPool("postgresql://user:pass@localhost/linkedin_posts")
        self.linkedin_session = ExternalAPISession("https://api.linkedin.com")
        self.ai_session = ExternalAPISession("https://api.ai-service.com")
        self.notification_session = ExternalAPISession("https://api.notifications.com")
        
        self.database = LinkedInPostsDatabase(self.db_pool)
        self.linkedin_api = LinkedInAPI(self.linkedin_session)
        self.ai_api = AIAnalysisAPI(self.ai_session)
        self.notification_api = NotificationAPI(self.notification_session)
        
        self.service = LinkedInPostsService(
            self.database, 
            self.linkedin_api, 
            self.ai_api, 
            self.notification_api
        )
        
        self._setup_routes()
        self._setup_events()
    
    def _setup_routes(self) -> Any:
        """Setup API routes"""
        
        @self.app.post("/api/v1/posts")
        async def create_post(request: LinkedInPostRequest):
            """Create a post with AI analysis and LinkedIn publishing"""
            try:
                result = await self.service.create_post_with_analysis(request)
                return result
            except Exception as e:
                logger.error(f"Failed to create post: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/posts/{post_id}")
        async def get_post(post_id: str):
            """Get a post with analytics"""
            try:
                result = await self.service.get_post_with_analytics(post_id)
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get post: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/api/v1/posts/{post_id}")
        async def update_post(post_id: str, updates: PostUpdate, 
                            optimization_type: str = "engagement"):
            """Update a post with AI optimization"""
            try:
                result = await self.service.update_post_with_optimization(
                    post_id, updates, optimization_type
                )
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update post: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/posts/user/{user_id}")
        async def get_user_posts(user_id: str, limit: int = 10, offset: int = 0):
            """Get posts by user with pagination"""
            try:
                posts = await self.database.get_posts_by_user(user_id, limit, offset)
                return {"posts": posts, "count": len(posts)}
            except Exception as e:
                logger.error(f"Failed to get user posts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/posts/search")
        async def search_posts(q: str, limit: int = 10):
            """Search posts"""
            try:
                posts = await self.database.search_posts(q, limit)
                return {"posts": posts, "count": len(posts), "query": q}
            except Exception as e:
                logger.error(f"Failed to search posts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                db_healthy = await self.db_pool.health_check()
                return {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "database": "connected" if db_healthy else "disconnected",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
    
    def _setup_events(self) -> Any:
        """Setup application events"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize on startup"""
            await self.db_pool.initialize()
            await self.linkedin_session.initialize()
            await self.ai_session.initialize()
            await self.notification_session.initialize()
            logger.info("Dedicated Async Operations API started")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            await self.db_pool.close()
            await self.linkedin_session.close()
            await self.ai_session.close()
            await self.notification_session.close()
            logger.info("Dedicated Async Operations API shutdown")
    
    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the application"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            loop="uvloop",
            http="httptools",
            access_log=False
        )
        
        server = uvicorn.Server(config)
        await server.serve()

# Main execution
async def main():
    """Main function"""
    api = DedicatedAsyncOperationsAPI()
    await api.run()

match __name__:
    case "__main__":
    asyncio.run(main()) 