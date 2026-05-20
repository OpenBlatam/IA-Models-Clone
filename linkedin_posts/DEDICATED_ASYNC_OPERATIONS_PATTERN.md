# Dedicated Async Functions for Database and External API Operations

## Overview

Dedicated async functions for database and external API operations are essential for building maintainable, scalable, and high-performance applications. This guide covers how to design and implement specialized async functions that handle specific types of operations with proper error handling, connection management, and performance optimization.

## Key Principles

### 1. **Separation of Concerns**
- Dedicated functions for specific operation types
- Clear boundaries between database and API operations
- Single responsibility principle for each function

### 2. **Connection Management**
- Proper connection pooling for databases
- HTTP session management for external APIs
- Resource cleanup and lifecycle management

### 3. **Error Handling and Resilience**
- Specific error types for different operations
- Retry logic with exponential backoff
- Circuit breaker patterns for external services

### 4. **Performance Optimization**
- Connection reuse and pooling
- Batch operations where possible
- Caching strategies for frequently accessed data

## Database Operations Pattern

### 1. **Connection Pool Management**
```python
class DatabaseConnectionPool:
    def __init__(self, database_url: str, pool_size: int = 20):
        self.database_url = database_url
        self.pool_size = pool_size
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=self.pool_size,
            command_timeout=30
        )
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def get_connection(self):
        """Get connection from pool"""
        return await self.pool.acquire()
    
    async def release_connection(self, conn):
        """Release connection back to pool"""
        await self.pool.release(conn)
```

### 2. **Dedicated Database Functions**
```python
class LinkedInPostsDatabase:
    def __init__(self, connection_pool: DatabaseConnectionPool):
        self.pool = connection_pool
    
    async def create_post(self, post_data: Dict[str, Any]) -> str:
        """Dedicated function for creating a post"""
        async with self.pool.get_connection() as conn:
            query = """
                INSERT INTO linkedin_posts (id, content, post_type, tone, target_audience, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """
            post_id = str(uuid.uuid4())
            result = await conn.fetchval(
                query,
                post_id,
                post_data['content'],
                post_data['post_type'],
                post_data['tone'],
                post_data['target_audience'],
                datetime.utcnow()
            )
            return result
    
    async def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Dedicated function for retrieving a post by ID"""
        async with self.pool.get_connection() as conn:
            query = """
                SELECT * FROM linkedin_posts 
                WHERE id = $1
            """
            result = await conn.fetchrow(query, post_id)
            return dict(result) if result else None
    
    async def update_post(self, post_id: str, updates: Dict[str, Any]) -> bool:
        """Dedicated function for updating a post"""
        async with self.pool.get_connection() as conn:
            set_clauses = []
            values = []
            param_count = 1
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1
            
            values.append(post_id)
            query = f"""
                UPDATE linkedin_posts 
                SET {', '.join(set_clauses)}, updated_at = ${param_count}
                WHERE id = ${param_count + 1}
            """
            
            result = await conn.execute(query, *values, datetime.utcnow())
            return result == "UPDATE 1"
    
    async def delete_post(self, post_id: str) -> bool:
        """Dedicated function for deleting a post"""
        async with self.pool.get_connection() as conn:
            query = "DELETE FROM linkedin_posts WHERE id = $1"
            result = await conn.execute(query, post_id)
            return result == "DELETE 1"
    
    async def get_posts_by_user(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Dedicated function for retrieving posts by user with pagination"""
        async with self.pool.get_connection() as conn:
            query = """
                SELECT * FROM linkedin_posts 
                WHERE user_id = $1 
                ORDER BY created_at DESC 
                LIMIT $2 OFFSET $3
            """
            results = await conn.fetch(query, user_id, limit, offset)
            return [dict(row) for row in results]
    
    async def search_posts(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Dedicated function for searching posts"""
        async with self.pool.get_connection() as conn:
            query = """
                SELECT * FROM linkedin_posts 
                WHERE content ILIKE $1 
                ORDER BY created_at DESC 
                LIMIT $2
            """
            results = await conn.fetch(query, f"%{search_term}%", limit)
            return [dict(row) for row in results]
    
    async def get_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """Dedicated function for retrieving post analytics"""
        async with self.pool.get_connection() as conn:
            query = """
                SELECT 
                    sentiment_score,
                    readability_score,
                    engagement_prediction,
                    views_count,
                    likes_count,
                    comments_count,
                    shares_count
                FROM linkedin_posts 
                WHERE id = $1
            """
            result = await conn.fetchrow(query, post_id)
            return dict(result) if result else {}
    
    async def batch_create_posts(self, posts_data: List[Dict[str, Any]]) -> List[str]:
        """Dedicated function for batch creating posts"""
        async with self.pool.get_connection() as conn:
            async with conn.transaction():
                post_ids = []
                for post_data in posts_data:
                    post_id = str(uuid.uuid4())
                    query = """
                        INSERT INTO linkedin_posts (id, content, post_type, tone, target_audience, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """
                    await conn.execute(
                        query,
                        post_id,
                        post_data['content'],
                        post_data['post_type'],
                        post_data['tone'],
                        post_data['target_audience'],
                        datetime.utcnow()
                    )
                    post_ids.append(post_id)
                return post_ids
```

## External API Operations Pattern

### 1. **HTTP Session Management**
```python
class ExternalAPISession:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self.session = None
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300
        )
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=self.connector
        )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
```

### 2. **Dedicated API Functions**
```python
class LinkedInAPI:
    def __init__(self, session: ExternalAPISession):
        self.session = session
    
    async def create_linkedin_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated function for creating a LinkedIn post via API"""
        try:
            async with self.session.session.post(
                "/v2/ugcPosts",
                json={
                    "author": f"urn:li:person:{post_data['user_id']}",
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {
                                "text": post_data['content']
                            },
                            "shareMediaCategory": "NONE"
                        }
                    },
                    "visibility": {
                        "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                    }
                },
                headers={
                    "Authorization": f"Bearer {post_data['access_token']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 201:
                    return await response.json()
                else:
                    raise APIError(f"Failed to create LinkedIn post: {response.status}")
        except Exception as e:
            raise APIError(f"Error creating LinkedIn post: {e}")
    
    async def get_linkedin_profile(self, user_id: str, access_token: str) -> Dict[str, Any]:
        """Dedicated function for retrieving LinkedIn profile"""
        try:
            async with self.session.session.get(
                f"/v2/people/{user_id}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise APIError(f"Failed to get LinkedIn profile: {response.status}")
        except Exception as e:
            raise APIError(f"Error getting LinkedIn profile: {e}")
    
    async def get_linkedin_analytics(self, post_id: str, access_token: str) -> Dict[str, Any]:
        """Dedicated function for retrieving LinkedIn post analytics"""
        try:
            async with self.session.session.get(
                f"/v2/socialMetrics/{post_id}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise APIError(f"Failed to get LinkedIn analytics: {response.status}")
        except Exception as e:
            raise APIError(f"Error getting LinkedIn analytics: {e}")

class AIAnalysisAPI:
    def __init__(self, session: ExternalAPISession):
        self.session = session
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Dedicated function for sentiment analysis"""
        try:
            async with self.session.session.post(
                "/analyze/sentiment",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise APIError(f"Failed to analyze sentiment: {response.status}")
        except Exception as e:
            raise APIError(f"Error analyzing sentiment: {e}")
    
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
                    return result.get("hashtags", [])
                else:
                    raise APIError(f"Failed to generate hashtags: {response.status}")
        except Exception as e:
            raise APIError(f"Error generating hashtags: {e}")
    
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
                    return result.get("optimized_content", content)
                else:
                    raise APIError(f"Failed to optimize content: {response.status}")
        except Exception as e:
            raise APIError(f"Error optimizing content: {e}")

class NotificationAPI:
    def __init__(self, session: ExternalAPISession):
        self.session = session
    
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
                return response.status == 200
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
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
                return response.status == 200
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False
```

## Error Handling and Resilience

### 1. **Custom Error Types**
```python
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
```

### 2. **Retry Logic with Exponential Backoff**
```python
import tenacity

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((ConnectionError, TimeoutError))
)
async def resilient_database_operation(operation_func, *args, **kwargs):
    """Execute database operation with retry logic"""
    return await operation_func(*args, **kwargs)

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type((TimeoutError, RateLimitError))
)
async def resilient_api_operation(operation_func, *args, **kwargs):
    """Execute API operation with retry logic"""
    return await operation_func(*args, **kwargs)
```

### 3. **Circuit Breaker Pattern**
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def circuit_breaker_api_call(api_func, *args, **kwargs):
    """Execute API call with circuit breaker"""
    return await api_func(*args, **kwargs)
```

## Performance Optimization

### 1. **Caching Strategy**
```python
import aioredis
from aiocache import cached

class CachedDatabaseOperations:
    def __init__(self, database: LinkedInPostsDatabase, redis_client: aioredis.Redis):
        self.database = database
        self.redis = redis_client
    
    @cached(ttl=300)  # Cache for 5 minutes
    async def get_post_by_id_cached(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Cached version of get_post_by_id"""
        return await self.database.get_post_by_id(post_id)
    
    async def invalidate_post_cache(self, post_id: str):
        """Invalidate cache for a specific post"""
        cache_key = f"get_post_by_id_cached:{post_id}"
        await self.redis.delete(cache_key)
```

### 2. **Batch Operations**
```python
class BatchOperations:
    def __init__(self, database: LinkedInPostsDatabase, api: LinkedInAPI):
        self.database = database
        self.api = api
    
    async def batch_create_posts_with_api(self, posts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch create posts in database and LinkedIn API"""
        # Create posts in database
        post_ids = await self.database.batch_create_posts(posts_data)
        
        # Create posts in LinkedIn API concurrently
        api_tasks = []
        for i, post_data in enumerate(posts_data):
            post_data['post_id'] = post_ids[i]
            task = self.api.create_linkedin_post(post_data)
            api_tasks.append(task)
        
        api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
        
        # Combine results
        results = []
        for i, (post_id, api_result) in enumerate(zip(post_ids, api_results)):
            if isinstance(api_result, Exception):
                results.append({
                    'post_id': post_id,
                    'database_status': 'created',
                    'api_status': 'failed',
                    'api_error': str(api_result)
                })
            else:
                results.append({
                    'post_id': post_id,
                    'database_status': 'created',
                    'api_status': 'created',
                    'api_response': api_result
                })
        
        return results
```

## Monitoring and Observability

### 1. **Metrics Collection**
```python
from prometheus_client import Counter, Histogram

# Database metrics
DB_OPERATION_DURATION = Histogram('db_operation_duration_seconds', 'Database operation duration', ['operation'])
DB_OPERATION_ERRORS = Counter('db_operation_errors_total', 'Database operation errors', ['operation'])

# API metrics
API_OPERATION_DURATION = Histogram('api_operation_duration_seconds', 'API operation duration', ['operation'])
API_OPERATION_ERRORS = Counter('api_operation_errors_total', 'API operation errors', ['operation'])

def track_database_operation(operation_name: str):
    """Decorator to track database operation metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                DB_OPERATION_DURATION.labels(operation_name).observe(duration)
                return result
            except Exception as e:
                DB_OPERATION_ERRORS.labels(operation_name).inc()
                raise
        return wrapper
    return decorator

def track_api_operation(operation_name: str):
    """Decorator to track API operation metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                API_OPERATION_DURATION.labels(operation_name).observe(duration)
                return result
            except Exception as e:
                API_OPERATION_ERRORS.labels(operation_name).inc()
                raise
        return wrapper
    return decorator
```

## Testing Patterns

### 1. **Unit Testing**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_create_post_database_operation():
    """Test database create post operation"""
    # Mock connection pool
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.get_connection.return_value.__aenter__.return_value = mock_conn
    mock_conn.fetchval.return_value = "test-post-id"
    
    database = LinkedInPostsDatabase(mock_pool)
    post_data = {
        'content': 'Test post',
        'post_type': 'educational',
        'tone': 'professional',
        'target_audience': 'general'
    }
    
    result = await database.create_post(post_data)
    
    assert result == "test-post-id"
    mock_conn.fetchval.assert_called_once()

@pytest.mark.asyncio
async def test_linkedin_api_create_post():
    """Test LinkedIn API create post operation"""
    # Mock HTTP session
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 201
    mock_response.json = AsyncMock(return_value={'id': 'linkedin-post-id'})
    mock_session.post.return_value.__aenter__.return_value = mock_response
    
    api = LinkedInAPI(mock_session)
    post_data = {
        'user_id': 'test-user',
        'content': 'Test post',
        'access_token': 'test-token'
    }
    
    result = await api.create_linkedin_post(post_data)
    
    assert result['id'] == 'linkedin-post-id'
    mock_session.post.assert_called_once()
```

### 2. **Integration Testing**
```python
@pytest.mark.asyncio
async def test_full_post_creation_flow():
    """Test complete post creation flow with database and API"""
    # Setup test database and API
    database = LinkedInPostsDatabase(test_connection_pool)
    api = LinkedInAPI(test_api_session)
    
    post_data = {
        'content': 'Integration test post',
        'post_type': 'educational',
        'tone': 'professional',
        'target_audience': 'general',
        'user_id': 'test-user',
        'access_token': 'test-token'
    }
    
    # Create post in database
    post_id = await database.create_post(post_data)
    assert post_id is not None
    
    # Create post in LinkedIn API
    api_result = await api.create_linkedin_post(post_data)
    assert api_result is not None
    
    # Verify post exists in database
    retrieved_post = await database.get_post_by_id(post_id)
    assert retrieved_post is not None
    assert retrieved_post['content'] == post_data['content']
```

## Best Practices

### 1. **Function Design**
- Keep functions focused on a single responsibility
- Use descriptive function names that indicate the operation
- Implement proper error handling for each function
- Add comprehensive logging for debugging

### 2. **Connection Management**
- Always use connection pooling for databases
- Implement proper cleanup in context managers
- Handle connection timeouts gracefully
- Monitor connection pool health

### 3. **Error Handling**
- Use specific exception types for different error scenarios
- Implement retry logic for transient failures
- Use circuit breakers for external services
- Provide meaningful error messages

### 4. **Performance**
- Implement caching for frequently accessed data
- Use batch operations when possible
- Monitor operation performance with metrics
- Optimize queries and API calls

### 5. **Testing**
- Write unit tests for each dedicated function
- Use mocks for external dependencies
- Implement integration tests for complete flows
- Test error scenarios and edge cases

## Summary

1. **Create dedicated functions for specific database operations**
2. **Implement dedicated functions for external API calls**
3. **Use proper connection pooling and session management**
4. **Implement comprehensive error handling and retry logic**
5. **Add caching and performance optimizations**
6. **Monitor operations with metrics and logging**
7. **Write comprehensive tests for all functions**

By following these patterns, you can build robust, maintainable, and high-performance async functions for database and external API operations that are easy to test, debug, and scale. 