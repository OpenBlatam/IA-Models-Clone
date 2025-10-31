from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, AsyncIterator
from datetime import datetime, timezone, timedelta
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field, validator, ConfigDict
from .lazy_loader import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Specialized Lazy Loaders for HeyGen AI API
Specialized lazy loading implementations for different data types.
"""


    LazyLoadingManager, LazyLoadingConfig, LoadingStrategy, 
    DataSourceType, LoadingPriority, lazy_load, lazy_stream
)

logger = structlog.get_logger()

# =============================================================================
# Data Types
# =============================================================================

class DataType(Enum):
    """Data type enumeration."""
    VIDEOS = "videos"
    USERS = "users"
    ANALYTICS = "analytics"
    TEMPLATES = "templates"
    SEARCH_RESULTS = "search_results"
    NOTIFICATIONS = "notifications"
    FILES = "files"
    REPORTS = "reports"

class VideoStatus(str, Enum):
    """Video status for filtering."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class UserRole(str, Enum):
    """User roles for filtering."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"
    PREMIUM = "premium"

# =============================================================================
# Base Models
# =============================================================================

class LazyLoadingRequest(BaseModel):
    """Base request model for lazy loading."""
    data_type: DataType
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc")
    batch_size: int = Field(default=100, ge=1, le=1000)
    enable_caching: bool = Field(default=True)
    enable_streaming: bool = Field(default=True)

class LazyLoadingResponse(BaseModel):
    """Base response model for lazy loading."""
    success: bool
    message: str
    data: Optional[Any] = None
    total_count: Optional[int] = None
    has_more: bool = False
    next_cursor: Optional[str] = None
    loading_stats: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Video Lazy Loader
# =============================================================================

class VideoLazyLoader:
    """Specialized lazy loader for video data."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.manager = LazyLoadingManager(config)
    
    @lazy_load(strategy=LoadingStrategy.PAGINATION, batch_size=50, enable_caching=True)
    async def load_user_videos(self, user_id: UUID, status: Optional[VideoStatus] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Load user videos with lazy loading."""
        # Simulate database query
        filters = {"user_id": user_id}
        if status:
            filters["status"] = status
        
        # This would be replaced with actual database query
        videos = await self._query_videos(filters, limit=limit)
        return videos
    
    @lazy_stream(strategy=LoadingStrategy.STREAMING, batch_size=20, enable_caching=True)
    async def stream_user_videos(self, user_id: UUID, status: Optional[VideoStatus] = None) -> AsyncIterator[Dict[str, Any]]:
        """Stream user videos with lazy loading."""
        # Simulate streaming database query
        filters = {"user_id": user_id}
        if status:
            filters["status"] = status
        
        # This would be replaced with actual streaming database query
        async for video in self._stream_videos(filters):
            yield video
    
    @lazy_load(strategy=LoadingStrategy.CURSOR_BASED, batch_size=100, enable_caching=True)
    async def load_popular_videos(self, category: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """Load popular videos with cursor-based pagination."""
        # Simulate cursor-based query
        filters = {
            "category": category,
            "days": days,
            "sort_by": "views",
            "sort_order": "desc"
        }
        
        videos = await self._query_popular_videos(filters)
        return videos
    
    @lazy_load(strategy=LoadingStrategy.WINDOW_BASED, batch_size=200, enable_caching=True)
    async def load_video_analytics(self, video_id: UUID, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Load video analytics with window-based loading."""
        # Simulate analytics query
        filters = {
            "video_id": video_id,
            "start_date": start_date,
            "end_date": end_date
        }
        
        analytics = await self._query_video_analytics(filters)
        return analytics
    
    async def _query_videos(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Simulate video database query."""
        # Simulate database delay
        await asyncio.sleep(0.1)
        
        # Return mock data
        return [
            {
                "id": f"video_{i}",
                "title": f"Video {i}",
                "status": "completed",
                "duration": 120 + i,
                "views": 1000 + i * 100,
                "created_at": datetime.now(timezone.utc) - timedelta(days=i)
            }
            for i in range(min(limit, 50))
        ]
    
    async def _stream_videos(self, filters: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Simulate streaming video database query."""
        for i in range(100):
            # Simulate processing delay
            await asyncio.sleep(0.01)
            
            yield {
                "id": f"video_{i}",
                "title": f"Video {i}",
                "status": "completed",
                "duration": 120 + i,
                "views": 1000 + i * 100,
                "created_at": datetime.now(timezone.utc) - timedelta(days=i)
            }
    
    async def _query_popular_videos(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate popular videos query."""
        await asyncio.sleep(0.2)
        
        return [
            {
                "id": f"popular_video_{i}",
                "title": f"Popular Video {i}",
                "views": 10000 + i * 1000,
                "likes": 500 + i * 50,
                "category": filters.get("category", "general"),
                "created_at": datetime.now(timezone.utc) - timedelta(days=i)
            }
            for i in range(100)
        ]
    
    async def _query_video_analytics(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate video analytics query."""
        await asyncio.sleep(0.3)
        
        start_date = filters["start_date"]
        end_date = filters["end_date"]
        days = (end_date - start_date).days
        
        return [
            {
                "date": start_date + timedelta(days=i),
                "views": 100 + i * 10,
                "likes": 10 + i,
                "shares": 5 + i // 2,
                "watch_time": 1000 + i * 100
            }
            for i in range(min(days, 365))
        ]

# =============================================================================
# User Lazy Loader
# =============================================================================

class UserLazyLoader:
    """Specialized lazy loader for user data."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.manager = LazyLoadingManager(config)
    
    @lazy_load(strategy=LoadingStrategy.PAGINATION, batch_size=100, enable_caching=True)
    async def load_users(self, role: Optional[UserRole] = None, is_active: bool = True, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load users with lazy loading."""
        filters = {"is_active": is_active}
        if role:
            filters["role"] = role
        
        users = await self._query_users(filters, limit=limit)
        return users
    
    @lazy_stream(strategy=LoadingStrategy.STREAMING, batch_size=50, enable_caching=True)
    async def stream_users(self, role: Optional[UserRole] = None, is_active: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """Stream users with lazy loading."""
        filters = {"is_active": is_active}
        if role:
            filters["role"] = role
        
        async for user in self._stream_users(filters):
            yield user
    
    @lazy_load(strategy=LoadingStrategy.CURSOR_BASED, batch_size=200, enable_caching=True)
    async def load_user_analytics(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Load user analytics with cursor-based pagination."""
        filters = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        analytics = await self._query_user_analytics(filters)
        return analytics
    
    @lazy_load(strategy=LoadingStrategy.WINDOW_BASED, batch_size=500, enable_caching=True)
    async def load_user_sessions(self, user_id: UUID, days: int = 30) -> List[Dict[str, Any]]:
        """Load user sessions with window-based loading."""
        filters = {
            "user_id": user_id,
            "days": days
        }
        
        sessions = await self._query_user_sessions(filters)
        return sessions
    
    async def _query_users(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Simulate user database query."""
        await asyncio.sleep(0.1)
        
        return [
            {
                "id": f"user_{i}",
                "email": f"user{i}@example.com",
                "first_name": f"User{i}",
                "last_name": f"Last{i}",
                "role": filters.get("role", "user"),
                "is_active": filters.get("is_active", True),
                "created_at": datetime.now(timezone.utc) - timedelta(days=i),
                "last_login": datetime.now(timezone.utc) - timedelta(hours=i)
            }
            for i in range(min(limit, 200))
        ]
    
    async def _stream_users(self, filters: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Simulate streaming user database query."""
        for i in range(500):
            await asyncio.sleep(0.01)
            
            yield {
                "id": f"user_{i}",
                "email": f"user{i}@example.com",
                "first_name": f"User{i}",
                "last_name": f"Last{i}",
                "role": filters.get("role", "user"),
                "is_active": filters.get("is_active", True),
                "created_at": datetime.now(timezone.utc) - timedelta(days=i),
                "last_login": datetime.now(timezone.utc) - timedelta(hours=i)
            }
    
    async def _query_user_analytics(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate user analytics query."""
        await asyncio.sleep(0.2)
        
        start_date = filters["start_date"]
        end_date = filters["end_date"]
        days = (end_date - start_date).days
        
        return [
            {
                "date": start_date + timedelta(days=i),
                "new_users": 50 + i * 5,
                "active_users": 1000 + i * 10,
                "total_sessions": 5000 + i * 50,
                "avg_session_duration": 300 + i * 2
            }
            for i in range(min(days, 365))
        ]
    
    async def _query_user_sessions(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate user sessions query."""
        await asyncio.sleep(0.1)
        
        user_id = filters["user_id"]
        days = filters["days"]
        
        return [
            {
                "session_id": f"session_{i}",
                "user_id": user_id,
                "started_at": datetime.now(timezone.utc) - timedelta(hours=i),
                "ended_at": datetime.now(timezone.utc) - timedelta(hours=i-1),
                "duration": 3600 + i * 60,
                "ip_address": f"192.168.1.{i % 255}",
                "user_agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{i}"
            }
            for i in range(min(days * 24, 1000))
        ]

# =============================================================================
# Analytics Lazy Loader
# =============================================================================

class AnalyticsLazyLoader:
    """Specialized lazy loader for analytics data."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.manager = LazyLoadingManager(config)
    
    @lazy_load(strategy=LoadingStrategy.WINDOW_BASED, batch_size=1000, enable_caching=True)
    async def load_platform_analytics(self, start_date: datetime, end_date: datetime, metrics: List[str]) -> List[Dict[str, Any]]:
        """Load platform analytics with window-based loading."""
        filters = {
            "start_date": start_date,
            "end_date": end_date,
            "metrics": metrics
        }
        
        analytics = await self._query_platform_analytics(filters)
        return analytics
    
    @lazy_stream(strategy=LoadingStrategy.STREAMING, batch_size=100, enable_caching=True)
    async def stream_performance_metrics(self, component: str, time_range: str = "24h") -> AsyncIterator[Dict[str, Any]]:
        """Stream performance metrics with lazy loading."""
        filters = {
            "component": component,
            "time_range": time_range
        }
        
        async for metric in self._stream_performance_metrics(filters):
            yield metric
    
    @lazy_load(strategy=LoadingStrategy.CURSOR_BASED, batch_size=500, enable_caching=True)
    async def load_error_logs(self, severity: str = "error", limit: int = 1000) -> List[Dict[str, Any]]:
        """Load error logs with cursor-based pagination."""
        filters = {
            "severity": severity,
            "limit": limit
        }
        
        logs = await self._query_error_logs(filters)
        return logs
    
    @lazy_load(strategy=LoadingStrategy.VIRTUAL_SCROLLING, batch_size=200, enable_caching=True)
    async def load_usage_reports(self, report_type: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Load usage reports with virtual scrolling."""
        filters = {
            "report_type": report_type,
            "start_date": start_date,
            "end_date": end_date
        }
        
        reports = await self._query_usage_reports(filters)
        return reports
    
    async def _query_platform_analytics(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate platform analytics query."""
        await asyncio.sleep(0.5)
        
        start_date = filters["start_date"]
        end_date = filters["end_date"]
        metrics = filters["metrics"]
        days = (end_date - start_date).days
        
        return [
            {
                "date": start_date + timedelta(days=i),
                "total_users": 10000 + i * 100,
                "total_videos": 5000 + i * 50,
                "total_views": 100000 + i * 1000,
                "total_revenue": 10000 + i * 100,
                "avg_session_duration": 600 + i * 5,
                "bounce_rate": 0.3 - i * 0.001
            }
            for i in range(min(days, 365))
        ]
    
    async def _stream_performance_metrics(self, filters: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Simulate streaming performance metrics."""
        component = filters["component"]
        time_range = filters["time_range"]
        
        # Simulate real-time metrics
        for i in range(1440):  # 24 hours in minutes
            await asyncio.sleep(0.001)  # Simulate processing time
            
            yield {
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=i),
                "component": component,
                "cpu_usage": 50 + (i % 50),
                "memory_usage": 60 + (i % 40),
                "response_time": 100 + (i % 200),
                "error_rate": 0.01 + (i % 10) * 0.001,
                "throughput": 1000 + (i % 500)
            }
    
    async def _query_error_logs(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate error logs query."""
        await asyncio.sleep(0.3)
        
        severity = filters["severity"]
        limit = filters["limit"]
        
        return [
            {
                "id": f"log_{i}",
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=i),
                "severity": severity,
                "message": f"Error message {i}",
                "component": f"component_{i % 10}",
                "user_id": f"user_{i % 100}",
                "stack_trace": f"Stack trace for error {i}",
                "metadata": {"request_id": f"req_{i}", "ip": f"192.168.1.{i % 255}"}
            }
            for i in range(min(limit, 1000))
        ]
    
    async def _query_usage_reports(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate usage reports query."""
        await asyncio.sleep(0.4)
        
        report_type = filters["report_type"]
        start_date = filters["start_date"]
        end_date = filters["end_date"]
        days = (end_date - start_date).days
        
        return [
            {
                "id": f"report_{i}",
                "report_type": report_type,
                "date": start_date + timedelta(days=i),
                "user_id": f"user_{i % 1000}",
                "action": f"action_{i % 10}",
                "duration": 100 + i * 10,
                "data_usage": 1024 + i * 100,
                "cost": 1.0 + i * 0.1,
                "metadata": {"feature": f"feature_{i % 5}", "plan": f"plan_{i % 3}"}
            }
            for i in range(min(days * 10, 10000))
        ]

# =============================================================================
# Template Lazy Loader
# =============================================================================

class TemplateLazyLoader:
    """Specialized lazy loader for template data."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.manager = LazyLoadingManager(config)
    
    @lazy_load(strategy=LoadingStrategy.PAGINATION, batch_size=50, enable_caching=True)
    async def load_templates(self, category: Optional[str] = None, is_public: bool = True, limit: int = 500) -> List[Dict[str, Any]]:
        """Load templates with lazy loading."""
        filters = {"is_public": is_public}
        if category:
            filters["category"] = category
        
        templates = await self._query_templates(filters, limit=limit)
        return templates
    
    @lazy_stream(strategy=LoadingStrategy.STREAMING, batch_size=25, enable_caching=True)
    async def stream_templates(self, category: Optional[str] = None, is_public: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """Stream templates with lazy loading."""
        filters = {"is_public": is_public}
        if category:
            filters["category"] = category
        
        async for template in self._stream_templates(filters):
            yield template
    
    @lazy_load(strategy=LoadingStrategy.CURSOR_BASED, batch_size=100, enable_caching=True)
    async def load_template_assets(self, template_id: UUID) -> List[Dict[str, Any]]:
        """Load template assets with cursor-based pagination."""
        filters = {"template_id": template_id}
        
        assets = await self._query_template_assets(filters)
        return assets
    
    async def _query_templates(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Simulate template database query."""
        await asyncio.sleep(0.1)
        
        return [
            {
                "id": f"template_{i}",
                "name": f"Template {i}",
                "category": filters.get("category", "general"),
                "is_public": filters.get("is_public", True),
                "creator_id": f"user_{i % 100}",
                "usage_count": 100 + i * 10,
                "rating": 4.0 + (i % 10) * 0.1,
                "created_at": datetime.now(timezone.utc) - timedelta(days=i),
                "thumbnail_url": f"https://example.com/thumbnails/template_{i}.jpg"
            }
            for i in range(min(limit, 200))
        ]
    
    async def _stream_templates(self, filters: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Simulate streaming template database query."""
        for i in range(300):
            await asyncio.sleep(0.01)
            
            yield {
                "id": f"template_{i}",
                "name": f"Template {i}",
                "category": filters.get("category", "general"),
                "is_public": filters.get("is_public", True),
                "creator_id": f"user_{i % 100}",
                "usage_count": 100 + i * 10,
                "rating": 4.0 + (i % 10) * 0.1,
                "created_at": datetime.now(timezone.utc) - timedelta(days=i),
                "thumbnail_url": f"https://example.com/thumbnails/template_{i}.jpg"
            }
    
    async def _query_template_assets(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate template assets query."""
        await asyncio.sleep(0.2)
        
        template_id = filters["template_id"]
        
        return [
            {
                "id": f"asset_{i}",
                "template_id": template_id,
                "type": ["image", "video", "audio", "font"][i % 4],
                "name": f"Asset {i}",
                "url": f"https://example.com/assets/asset_{i}.{['jpg', 'mp4', 'mp3', 'ttf'][i % 4]}",
                "size": 1024 * (i + 1),
                "created_at": datetime.now(timezone.utc) - timedelta(hours=i)
            }
            for i in range(50)
        ]

# =============================================================================
# Search Results Lazy Loader
# =============================================================================

class SearchResultsLazyLoader:
    """Specialized lazy loader for search results."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.manager = LazyLoadingManager(config)
    
    @lazy_load(strategy=LoadingStrategy.PAGINATION, batch_size=20, enable_caching=True)
    async def load_search_results(self, query: str, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Load search results with lazy loading."""
        search_params = {
            "query": query,
            "filters": filters,
            "limit": limit
        }
        
        results = await self._query_search_results(search_params)
        return results
    
    @lazy_stream(strategy=LoadingStrategy.STREAMING, batch_size=10, enable_caching=True)
    async def stream_search_results(self, query: str, filters: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Stream search results with lazy loading."""
        search_params = {
            "query": query,
            "filters": filters
        }
        
        async for result in self._stream_search_results(search_params):
            yield result
    
    @lazy_load(strategy=LoadingStrategy.CURSOR_BASED, batch_size=50, enable_caching=True)
    async def load_suggestions(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Load search suggestions with cursor-based pagination."""
        search_params = {
            "query": query,
            "limit": limit
        }
        
        suggestions = await self._query_suggestions(search_params)
        return suggestions
    
    async def _query_search_results(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate search results query."""
        await asyncio.sleep(0.2)
        
        query = search_params["query"]
        filters = search_params.get("filters", {})
        limit = search_params.get("limit", 100)
        
        return [
            {
                "id": f"result_{i}",
                "type": ["video", "user", "template"][i % 3],
                "title": f"Search result {i} for '{query}'",
                "description": f"Description for result {i}",
                "score": 0.9 - (i * 0.01),
                "metadata": {
                    "category": filters.get("category", "all"),
                    "tags": [f"tag_{j}" for j in range(i % 5)],
                    "created_at": datetime.now(timezone.utc) - timedelta(days=i)
                }
            }
            for i in range(min(limit, 200))
        ]
    
    async def _stream_search_results(self, search_params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Simulate streaming search results."""
        query = search_params["query"]
        filters = search_params.get("filters", {})
        
        for i in range(100):
            await asyncio.sleep(0.01)
            
            yield {
                "id": f"result_{i}",
                "type": ["video", "user", "template"][i % 3],
                "title": f"Search result {i} for '{query}'",
                "description": f"Description for result {i}",
                "score": 0.9 - (i * 0.01),
                "metadata": {
                    "category": filters.get("category", "all"),
                    "tags": [f"tag_{j}" for j in range(i % 5)],
                    "created_at": datetime.now(timezone.utc) - timedelta(days=i)
                }
            }
    
    async def _query_suggestions(self, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate search suggestions query."""
        await asyncio.sleep(0.1)
        
        query = search_params["query"]
        limit = search_params.get("limit", 50)
        
        return [
            {
                "id": f"suggestion_{i}",
                "text": f"{query} suggestion {i}",
                "type": ["query", "autocomplete", "related"][i % 3],
                "score": 0.8 - (i * 0.01),
                "frequency": 100 - i
            }
            for i in range(min(limit, 50))
        ]

# =============================================================================
# Lazy Loading Factory
# =============================================================================

class LazyLoadingFactory:
    """Factory for creating specialized lazy loaders."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.loaders: Dict[DataType, Any] = {}
    
    def get_video_loader(self) -> VideoLazyLoader:
        """Get video lazy loader."""
        if DataType.VIDEOS not in self.loaders:
            self.loaders[DataType.VIDEOS] = VideoLazyLoader(self.config)
        return self.loaders[DataType.VIDEOS]
    
    def get_user_loader(self) -> UserLazyLoader:
        """Get user lazy loader."""
        if DataType.USERS not in self.loaders:
            self.loaders[DataType.USERS] = UserLazyLoader(self.config)
        return self.loaders[DataType.USERS]
    
    def get_analytics_loader(self) -> AnalyticsLazyLoader:
        """Get analytics lazy loader."""
        if DataType.ANALYTICS not in self.loaders:
            self.loaders[DataType.ANALYTICS] = AnalyticsLazyLoader(self.config)
        return self.loaders[DataType.ANALYTICS]
    
    def get_template_loader(self) -> TemplateLazyLoader:
        """Get template lazy loader."""
        if DataType.TEMPLATES not in self.loaders:
            self.loaders[DataType.TEMPLATES] = TemplateLazyLoader(self.config)
        return self.loaders[DataType.TEMPLATES]
    
    def get_search_loader(self) -> SearchResultsLazyLoader:
        """Get search results lazy loader."""
        if DataType.SEARCH_RESULTS not in self.loaders:
            self.loaders[DataType.SEARCH_RESULTS] = SearchResultsLazyLoader(self.config)
        return self.loaders[DataType.SEARCH_RESULTS]
    
    def get_all_loaders(self) -> Dict[DataType, Any]:
        """Get all specialized loaders."""
        return self.loaders

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_lazy_loading_factory() -> LazyLoadingFactory:
    """Dependency to get lazy loading factory."""
    config = LazyLoadingConfig(
        strategy=LoadingStrategy.STREAMING,
        batch_size=100,
        max_concurrent_batches=5,
        enable_caching=True,
        enable_prefetching=True,
        memory_limit_mb=500
    )
    return LazyLoadingFactory(config)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "DataType",
    "VideoStatus",
    "UserRole",
    "LazyLoadingRequest",
    "LazyLoadingResponse",
    "VideoLazyLoader",
    "UserLazyLoader",
    "AnalyticsLazyLoader",
    "TemplateLazyLoader",
    "SearchResultsLazyLoader",
    "LazyLoadingFactory",
    "get_lazy_loading_factory",
] 