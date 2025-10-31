"""
Advanced Library Enhancements for BUL API
=========================================

Modern, optimized libraries following best practices:
- Enhanced HTTP clients with connection pooling
- Advanced caching with multiple strategies
- Improved database operations with async support
- Enhanced logging with structured output
- Advanced validation with custom validators
- Performance monitoring and metrics
"""

import asyncio
import time
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic, AsyncGenerator
from functools import wraps, lru_cache, partial
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import re
from pathlib import Path

# Enhanced HTTP Client
import httpx
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import orjson
import ujson

# Enhanced Database
import asyncpg
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, Float
from sqlalchemy.dialects.postgresql import UUID

# Enhanced Validation
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt, EmailStr, HttpUrl
import email_validator
from marshmallow import Schema, fields, validate

# Enhanced Logging
import structlog
from loguru import logger
import sys

# Enhanced Caching
import redis
from redis.asyncio import Redis
import pickle
import zlib

# Enhanced Security
import bcrypt
import argon2
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

# Enhanced Performance
import psutil
import memory_profiler
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Type variables
T = TypeVar('T')
R = TypeVar('R')

# Enhanced HTTP Client
class AdvancedHTTPClient:
    """Enhanced HTTP client with connection pooling and retry logic"""
    
    def __init__(
        self,
        base_url: str = "",
        timeout: int = 30,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        
        # Connection limits
        self.limits = httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections
        )
        
        # Timeout configuration
        self.timeout_config = httpx.Timeout(
            connect=10.0,
            read=timeout,
            write=10.0,
            pool=5.0
        )
        
        # Client configuration
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=self.timeout_config,
            limits=self.limits,
            headers={
                "User-Agent": "BUL-API/3.0.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0.0
    
    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enhanced GET request with retry logic"""
        return await self._make_request("GET", url, params=params, headers=headers, timeout=timeout)
    
    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enhanced POST request with retry logic"""
        return await self._make_request("POST", url, data=data, json_data=json_data, headers=headers, timeout=timeout)
    
    async def put(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enhanced PUT request with retry logic"""
        return await self._make_request("PUT", url, data=data, json_data=json_data, headers=headers, timeout=timeout)
    
    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enhanced DELETE request with retry logic"""
        return await self._make_request("DELETE", url, headers=headers, timeout=timeout)
    
    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and metrics"""
        start_time = time.time()
        last_exception = None
        
        # Prepare request
        request_headers = headers or {}
        request_timeout = timeout or self.timeout
        
        for attempt in range(self.retry_attempts + 1):
            try:
                # Make request
                if method == "GET":
                    response = await self.client.get(url, params=params, headers=request_headers, timeout=request_timeout)
                elif method == "POST":
                    if json_data:
                        response = await self.client.post(url, json=json_data, headers=request_headers, timeout=request_timeout)
                    else:
                        response = await self.client.post(url, data=data, headers=request_headers, timeout=request_timeout)
                elif method == "PUT":
                    if json_data:
                        response = await self.client.put(url, json=json_data, headers=request_headers, timeout=request_timeout)
                    else:
                        response = await self.client.put(url, data=data, headers=request_headers, timeout=request_timeout)
                elif method == "DELETE":
                    response = await self.client.delete(url, headers=request_headers, timeout=request_timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Update metrics
                duration = time.time() - start_time
                self.request_count += 1
                self.total_duration += duration
                
                # Parse response
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {"content": response.text}
                
                return {
                    "status_code": response.status_code,
                    "data": response_data,
                    "headers": dict(response.headers),
                    "duration": duration,
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    await asyncio.sleep(delay)
                else:
                    self.error_count += 1
                    raise last_exception
        
        # This should never be reached
        raise last_exception
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_duration": self.total_duration,
            "avg_duration": self.total_duration / self.request_count if self.request_count > 0 else 0,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }

# Enhanced Database Client
class AdvancedDatabaseClient:
    """Enhanced database client with connection pooling and health monitoring"""
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        # Create async engine
        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=echo,
            pool_pre_ping=True
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Metrics
        self.query_count = 0
        self.error_count = 0
        self.total_duration = 0.0
        self.connection_count = 0
    
    async def get_session(self) -> AsyncSession:
        """Get database session with metrics"""
        self.connection_count += 1
        return self.session_factory()
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute query with metrics and error handling"""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                result = await session.execute(query, params or {})
                await session.commit()
                
                duration = time.time() - start_time
                self.query_count += 1
                self.total_duration += duration
                
                return result
                
        except Exception as e:
            duration = time.time() - start_time
            self.error_count += 1
            self.total_duration += duration
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()
            
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": duration,
                "pool_size": self.pool_size,
                "active_connections": self.connection_count,
                "query_count": self.query_count,
                "error_count": self.error_count,
                "avg_query_time": self.total_duration / self.query_count if self.query_count > 0 else 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    async def close(self):
        """Close database connections"""
        await self.engine.dispose()

# Enhanced Cache Client
class AdvancedCacheClient:
    """Enhanced cache client with multiple strategies and compression"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_connections: int = 20,
        compression: bool = True,
        default_ttl: int = 3600
    ):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.compression = compression
        self.default_ttl = default_ttl
        
        # Redis connection pool
        self.redis = Redis.from_url(
            redis_url,
            max_connections=max_connections,
            retry_on_timeout=True,
            decode_responses=True
        )
        
        # Metrics
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.delete_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with decompression"""
        try:
            value = await self.redis.get(key)
            if value is None:
                self.miss_count += 1
                return None
            
            # Decompress if needed
            if self.compression:
                value = zlib.decompress(value.encode('latin1')).decode('utf-8')
            
            # Deserialize
            try:
                data = orjson.loads(value)
            except (orjson.JSONDecodeError, ValueError):
                data = value
            
            self.hit_count += 1
            return data
            
        except Exception as e:
            self.miss_count += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with compression"""
        try:
            # Serialize
            if isinstance(value, (dict, list)):
                serialized = orjson.dumps(value)
            else:
                serialized = str(value).encode('utf-8')
            
            # Compress if enabled
            if self.compression:
                serialized = zlib.compress(serialized)
                serialized = serialized.decode('latin1')
            
            # Set in Redis
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, serialized)
            
            self.set_count += 1
            return True
            
        except Exception as e:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.redis.delete(key)
            self.delete_count += 1
            return result > 0
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            await self.redis.flushdb()
            return True
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        start_time = time.time()
        
        try:
            await self.redis.ping()
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": duration,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "set_count": self.set_count,
                "delete_count": self.delete_count,
                "hit_rate": self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    async def close(self):
        """Close cache connection"""
        await self.redis.close()

# Enhanced Logger
class AdvancedLogger:
    """Enhanced logger with structured output and performance monitoring"""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        format_type: str = "json",
        file_path: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        self.name = name
        self.level = level
        self.format_type = format_type
        self.file_path = file_path
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Configure structlog
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
                structlog.processors.JSONRenderer() if format_type == "json" else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Get logger
        self.logger = structlog.get_logger(name)
        
        # Configure loguru if file logging is enabled
        if file_path:
            logger.add(
                file_path,
                rotation=f"{max_file_size} bytes",
                retention=f"{backup_count} files",
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
                serialize=format_type == "json"
            )
        
        # Metrics
        self.log_count = 0
        self.error_count = 0
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log_count += 1
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log_count += 1
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log_count += 1
        self.error_count += 1
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log_count += 1
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log_count += 1
        self.error_count += 1
        self.logger.critical(message, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logger metrics"""
        return {
            "log_count": self.log_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.log_count if self.log_count > 0 else 0
        }

# Enhanced Validator
class AdvancedValidator:
    """Enhanced validator with custom validation rules"""
    
    def __init__(self):
        self.custom_validators: Dict[str, Callable] = {}
        self.validation_cache: Dict[str, bool] = {}
    
    def register_validator(self, name: str, validator_func: Callable[[Any], bool) -> None:
        """Register custom validator"""
        self.custom_validators[name] = validator_func
    
    def validate_email(self, email: str) -> bool:
        """Validate email address"""
        try:
            email_validator.validate_email(email)
            return True
        except email_validator.EmailNotValidError:
            return False
    
    def validate_url(self, url: str) -> bool:
        """Validate URL"""
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number"""
        pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        return bool(re.match(pattern, phone))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        issues = []
        score = 0
        
        # Length check
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        else:
            score += 1
        
        # Character variety checks
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        else:
            score += 1
        
        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        else:
            score += 1
        
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        else:
            score += 1
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        else:
            score += 1
        
        return {
            "is_valid": len(issues) == 0,
            "score": score,
            "issues": issues
        }
    
    def validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate data against JSON schema"""
        try:
            for key, expected_type in schema.items():
                if key not in data:
                    return False
                if not isinstance(data[key], expected_type):
                    return False
            return True
        except Exception:
            return False
    
    def validate_cached(self, key: str, validator_func: Callable[[Any], bool], data: Any) -> bool:
        """Validate with caching"""
        cache_key = f"{key}:{hash(str(data))}"
        
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        result = validator_func(data)
        self.validation_cache[cache_key] = result
        
        # Limit cache size
        if len(self.validation_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.validation_cache.keys())[:100]
            for key in oldest_keys:
                del self.validation_cache[key]
        
        return result

# Enhanced Security
class AdvancedSecurity:
    """Enhanced security with advanced encryption and hashing"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode('utf-8')
        self.fernet = Fernet(Fernet.generate_key())
        
        # PBKDF2 key derivation
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'bul_salt',
            iterations=100000,
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def hash_password_argon2(self, password: str) -> str:
        """Hash password with argon2"""
        return argon2.hash_password(password.encode('utf-8')).decode('utf-8')
    
    def verify_password_argon2(self, password: str, hashed_password: str) -> bool:
        """Verify password against argon2 hash"""
        try:
            return argon2.verify_password(hashed_password.encode('utf-8'), password.encode('utf-8'))
        except Exception:
            return False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data with Fernet"""
        return self.fernet.encrypt(data.encode('utf-8')).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data with Fernet"""
        return self.fernet.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    def generate_salt(self, length: int = 16) -> str:
        """Generate random salt"""
        return secrets.token_hex(length)

# Enhanced Performance Monitor
class AdvancedPerformanceMonitor:
    """Enhanced performance monitor with Prometheus metrics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Prometheus metrics
        self.request_counter = Counter(
            'bul_api_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'bul_api_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'bul_api_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'bul_api_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'bul_api_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Internal metrics
        self.metrics: Dict[str, Any] = {}
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """Record request metrics"""
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def update_system_metrics(self) -> None:
        """Update system metrics"""
        # Memory usage
        memory_info = psutil.virtual_memory()
        self.memory_usage.set(memory_info.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        self.update_system_metrics()
        
        return {
            "request_counter": self.request_counter._value._value,
            "request_duration": self.request_duration._sum._value,
            "active_connections": self.active_connections._value._value,
            "memory_usage": self.memory_usage._value._value,
            "cpu_usage": self.cpu_usage._value._value
        }

# Enhanced Data Processor
class AdvancedDataProcessor:
    """Enhanced data processor with advanced transformations"""
    
    def __init__(self):
        self.transformations: Dict[str, Callable] = {}
        self.filters: Dict[str, Callable] = {}
        self.aggregations: Dict[str, Callable] = {}
    
    def register_transformation(self, name: str, transform_func: Callable[[Any], Any]) -> None:
        """Register data transformation"""
        self.transformations[name] = transform_func
    
    def register_filter(self, name: str, filter_func: Callable[[Any], bool]) -> None:
        """Register data filter"""
        self.filters[name] = filter_func
    
    def register_aggregation(self, name: str, agg_func: Callable[[List[Any]], Any]) -> None:
        """Register data aggregation"""
        self.aggregations[name] = agg_func
    
    def transform_data(
        self,
        data: List[Dict[str, Any]],
        transformations: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Transform data using registered transformations"""
        result = []
        
        for item in data:
            transformed_item = {}
            
            for key, value in item.items():
                if key in transformations:
                    transform_name = transformations[key]
                    if transform_name in self.transformations:
                        transformed_item[key] = self.transformations[transform_name](value)
                    else:
                        transformed_item[key] = value
                else:
                    transformed_item[key] = value
            
            result.append(transformed_item)
        
        return result
    
    def filter_data(
        self,
        data: List[Dict[str, Any]],
        filters: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Filter data using registered filters"""
        result = []
        
        for item in data:
            include = True
            
            for key, filter_name in filters.items():
                if key in item and filter_name in self.filters:
                    if not self.filters[filter_name](item[key]):
                        include = False
                        break
            
            if include:
                result.append(item)
        
        return result
    
    def aggregate_data(
        self,
        data: List[Dict[str, Any]],
        group_by: str,
        aggregations: Dict[str, str]
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate data by key with custom aggregations"""
        grouped = {}
        
        for item in data:
            key = item.get(group_by)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        
        result = {}
        
        for key, items in grouped.items():
            aggregated = {}
            
            for field, agg_name in aggregations.items():
                values = [item.get(field) for item in items if field in item]
                if values and agg_name in self.aggregations:
                    aggregated[field] = self.aggregations[agg_name](values)
            
            result[key] = aggregated
        
        return result

# Factory functions for easy instantiation
def create_http_client(**kwargs) -> AdvancedHTTPClient:
    """Create enhanced HTTP client"""
    return AdvancedHTTPClient(**kwargs)

def create_database_client(**kwargs) -> AdvancedDatabaseClient:
    """Create enhanced database client"""
    return AdvancedDatabaseClient(**kwargs)

def create_cache_client(**kwargs) -> AdvancedCacheClient:
    """Create enhanced cache client"""
    return AdvancedCacheClient(**kwargs)

def create_logger(**kwargs) -> AdvancedLogger:
    """Create enhanced logger"""
    return AdvancedLogger(**kwargs)

def create_validator() -> AdvancedValidator:
    """Create enhanced validator"""
    return AdvancedValidator()

def create_security(secret_key: str) -> AdvancedSecurity:
    """Create enhanced security"""
    return AdvancedSecurity(secret_key)

def create_performance_monitor(**kwargs) -> AdvancedPerformanceMonitor:
    """Create enhanced performance monitor"""
    return AdvancedPerformanceMonitor(**kwargs)

def create_data_processor() -> AdvancedDataProcessor:
    """Create enhanced data processor"""
    return AdvancedDataProcessor()

# Export all enhanced libraries
__all__ = [
    # HTTP Client
    "AdvancedHTTPClient",
    "create_http_client",
    
    # Database Client
    "AdvancedDatabaseClient", 
    "create_database_client",
    
    # Cache Client
    "AdvancedCacheClient",
    "create_cache_client",
    
    # Logger
    "AdvancedLogger",
    "create_logger",
    
    # Validator
    "AdvancedValidator",
    "create_validator",
    
    # Security
    "AdvancedSecurity",
    "create_security",
    
    # Performance Monitor
    "AdvancedPerformanceMonitor",
    "create_performance_monitor",
    
    # Data Processor
    "AdvancedDataProcessor",
    "create_data_processor"
]












