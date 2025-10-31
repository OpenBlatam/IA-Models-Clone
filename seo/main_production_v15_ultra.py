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
import signal
import sys
import time
from datetime import datetime
import gc
import psutil
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union, Tuple, TypedDict, AsyncGenerator, Iterator, Generator
from dataclasses import dataclass, field
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import weakref
from contextlib import asynccontextmanager, contextmanager
import signal
import os
import sys
import uvloop
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.requests import Request
from pydantic import BaseModel, Field, validator, ConfigDict, computed_field
import orjson
import httpx
import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
import aioredis
import aiofiles
import aiohttp
import asyncio_mqtt
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import structlog
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import jwt
from passlib.context import CryptContext
import bcrypt
import secrets
import random
import hashlib
import hmac
import base64
import json
import yaml
import toml
import configparser
import argparse
import logging
import warnings
import traceback
import inspect
import functools
import operator
import itertools
import collections
import heapq
import bisect
import array
import mmap
import pickle
import marshal
import shelve
import sqlite3
import threading
import queue
import socket
import ssl
import urllib.parse
import urllib.request
import urllib.error
import email
import mimetypes
import tempfile
import shutil
import pathlib
import glob
import fnmatch
import zipfile
import tarfile
import gzip
import bz2
import lzma
import zlib
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, pipeline
import diffusers
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gradio as gr
import numpy as np
from tqdm import tqdm
import wandb
import tensorboard
from tensorboard import program
import psutil
import GPUtil
import nvidia_ml_py3
import py3nvml
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import weakref
from contextlib import asynccontextmanager, contextmanager
import signal
import os
import sys
                            import csv
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Ultra-Optimized SEO Service v15 - MAXIMUM PERFORMANCE
Latest Optimizations with Fastest Libraries 2024 - Complete Ultra Refactor
HTTP/3 Support, Ultra-Fast JSON, Advanced Caching, Maximum Performance
RORO Pattern Implementation for Clean Function Signatures
"""


# Ultra-fast imports with latest optimizations

# Deep Learning imports

# Performance monitoring

# Non-blocking operation optimizations

# Configure uvloop for maximum performance
if sys.platform != "win32":
    uvloop.install()

# Global thread and process pools for non-blocking operations
_thread_pool = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4))
_process_pool = ProcessPoolExecutor(max_workers=min(8, (os.cpu_count() or 1) * 2))

# Connection pools for external services
_http_connection_pool = None
_redis_connection_pool = None
_mongo_connection_pool = None

# Background task queue for non-blocking operations
_background_task_queue = asyncio.Queue(maxsize=1000)
_background_worker_task = None

# Performance monitoring for blocking operations
_blocking_operation_metrics = {
    'thread_pool_usage': 0,
    'process_pool_usage': 0,
    'background_tasks_pending': 0,
    'connection_pool_usage': 0
}

class NonBlockingOperationManager:
    """Manages non-blocking operations to prevent route blocking."""
    
    def __init__(self) -> Any:
        self.thread_pool = _thread_pool
        self.process_pool = _process_pool
        self.background_queue = _background_task_queue
        self.metrics = _blocking_operation_metrics
        self._running_tasks = weakref.WeakSet()
        self._semaphore = asyncio.Semaphore(100)  # Limit concurrent operations
        
    async def run_in_thread(self, func, *args, **kwargs) -> Any:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        """Run CPU-intensive operations in thread pool."""
        async with self._semaphore:
            self.metrics['thread_pool_usage'] += 1
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool, 
                    partial(func, *args, **kwargs)
                )
                return result
            finally:
                self.metrics['thread_pool_usage'] -= 1
    
    async def run_in_process(self, func, *args, **kwargs) -> Any:
        """Run CPU-intensive operations in process pool."""
        async with self._semaphore:
            self.metrics['process_pool_usage'] += 1
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_pool, 
                    partial(func, *args, **kwargs)
                )
                return result
            finally:
                self.metrics['process_pool_usage'] -= 1
    
    async def add_background_task(self, task_func, *args, **kwargs) -> Any:
        """Add task to background queue for non-blocking execution."""
        try:
            await self.background_queue.put((task_func, args, kwargs))
            self.metrics['background_tasks_pending'] += 1
        except asyncio.QueueFull:
            # If queue is full, run task immediately in thread pool
            return await self.run_in_thread(task_func, *args, **kwargs)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    async def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current non-blocking operation metrics."""
        return {
            'thread_pool_usage': self.metrics['thread_pool_usage'],
            'process_pool_usage': self.metrics['process_pool_usage'],
            'background_tasks_pending': self.metrics['background_tasks_pending'],
            'connection_pool_usage': self.metrics['connection_pool_usage'],
            'semaphore_available': self._semaphore._value,
            'queue_size': self.background_queue.qsize()
        }

# Global non-blocking operation manager
non_blocking_manager = NonBlockingOperationManager()

class ConnectionPoolManager:
    """Manages connection pools for external services."""
    
    def __init__(self) -> Any:
        self.http_pool = None
        self.redis_pool = None
        self.mongo_pool = None
        self._lock = asyncio.Lock()
    
    async async def get_http_pool(self) -> Optional[Dict[str, Any]]:
        """Get or create HTTP connection pool."""
        if self.http_pool is None:
            async with self._lock:
                if self.http_pool is None:
                    self.http_pool = httpx.AsyncClient(
                        timeout=httpx.Timeout(30.0),
                        limits=httpx.Limits(
                            max_keepalive_connections=50,
                            max_connections=200,
                            keepalive_expiry=30.0
                        ),
                        http2=True
                    )
        return self.http_pool
    
    async def get_redis_pool(self, redis_url: str):
        """Get or create Redis connection pool."""
        if self.redis_pool is None:
            async with self._lock:
                if self.redis_pool is None:
                    self.redis_pool = redis.from_url(
                        redis_url,
                        encoding="utf-8",
                        decode_responses=True,
                        max_connections=50,
                        retry_on_timeout=True,
                        socket_keepalive=True
                    )
        return self.redis_pool
    
    async def get_mongo_pool(self, mongo_url: str):
        """Get or create MongoDB connection pool."""
        if self.mongo_pool is None:
            async with self._lock:
                if self.mongo_pool is None:
                    self.mongo_pool = AsyncIOMotorClient(
                        mongo_url,
                        maxPoolSize=50,
                        minPoolSize=10,
                        maxIdleTimeMS=30000,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=10000,
                        socketTimeoutMS=20000
                    )
        return self.mongo_pool
    
    async def close_all(self) -> Any:
        """Close all connection pools."""
        if self.http_pool:
            await self.http_pool.aclose()
        if self.redis_pool:
            await self.redis_pool.close()
        if self.mongo_pool:
            self.mongo_pool.close()

# Global connection pool manager
connection_pool_manager = ConnectionPoolManager()

class AsyncTaskScheduler:
    """Schedules and manages background tasks."""
    
    def __init__(self) -> Any:
        self.tasks = weakref.WeakSet()
        self.scheduler_task = None
        self._running = False
    
    async def start(self) -> Any:
        """Start the background task scheduler."""
        if not self._running:
            self._running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self) -> Any:
        """Stop the background task scheduler."""
        self._running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self) -> Any:
        """Main scheduler loop for processing background tasks."""
        while self._running:
            try:
                # Process background tasks
                while not non_blocking_manager.background_queue.empty():
                    task_func, args, kwargs = await non_blocking_manager.background_queue.get()
                    task = asyncio.create_task(self._execute_background_task(task_func, args, kwargs))
                    self.tasks.add(task)
                    non_blocking_manager.metrics['background_tasks_pending'] -= 1
                
                # Clean up completed tasks
                completed_tasks = [task for task in self.tasks if task.done()]
                for task in completed_tasks:
                    self.tasks.discard(task)
                    try:
                        await task
                    except Exception as e:
                        logger.error("Background task failed", error=str(e))
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduler error", error=str(e))
                await asyncio.sleep(1)
    
    async def _execute_background_task(self, task_func, args, kwargs) -> Any:
        """Execute a background task with error handling."""
        try:
            if asyncio.iscoroutinefunction(task_func):
                await task_func(*args, **kwargs)
            else:
                await non_blocking_manager.run_in_thread(task_func, *args, **kwargs)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            logger.error("Background task execution failed", error=str(e))

# Global task scheduler
task_scheduler = AsyncTaskScheduler()

# Optimized utility functions for non-blocking operations
async def non_blocking_file_operation(func, *args, **kwargs) -> Any:
    """Execute file operations in thread pool to avoid blocking."""
    return await non_blocking_manager.run_in_thread(func, *args, **kwargs)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

async def non_blocking_cpu_operation(func, *args, **kwargs) -> Any:
    """Execute CPU-intensive operations in process pool."""
    return await non_blocking_manager.run_in_process(func, *args, **kwargs)

async def non_blocking_network_operation(func, *args, **kwargs) -> Any:
    """Execute network operations with connection pooling."""
    return await non_blocking_manager.run_in_thread(func, *args, **kwargs)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

def optimize_blocking_operations():
    """Apply optimizations to reduce blocking operations."""
    
    # Optimize JSON serialization
    def fast_json_dumps(obj) -> Any:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NAIVE_UTC)
    
    def fast_json_loads(data) -> Any:
        return orjson.loads(data)
    
    # Replace standard JSON functions with optimized versions
    json.dumps = fast_json_dumps
    json.loads = fast_json_loads
    
    # Optimize hashlib operations
    def fast_hash(data) -> Any:
        return hashlib.sha256(data.encode()).hexdigest()
    
    # Optimize URL parsing
    def fast_url_parse(url) -> Any:
        return urllib.parse.urlparse(url)
    
    # Optimize string operations
    def fast_string_join(iterable, separator='') -> Any:
        return separator.join(iterable)
    
    return {
        'json_dumps': fast_json_dumps,
        'json_loads': fast_json_loads,
        'fast_hash': fast_hash,
        'fast_url_parse': fast_url_parse,
        'fast_string_join': fast_string_join
    }

# Apply optimizations
optimized_functions = optimize_blocking_operations()

# ============================================================================
# DEDICATED ASYNC FUNCTIONS FOR DATABASE AND EXTERNAL API OPERATIONS
# ============================================================================

class AsyncDatabaseOperations:
    """Dedicated async functions for database operations."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, mongo_client: Optional[AsyncIOMotorClient] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.mongo_client = mongo_client
        self._connection_semaphore = asyncio.Semaphore(50)  # Limit concurrent DB connections
        
    async def store_seo_result(self, result: 'SEOResultModel', collection: str = "seo_results") -> bool:
        """Store SEO analysis result in database asynchronously."""
        async with self._connection_semaphore:
            try:
                if self.mongo_client:
                    # Store in MongoDB
                    db = self.mongo_client.seo_database
                    await db[collection].insert_one(result.model_dump(mode='json'))
                
                if self.redis_client:
                    # Cache in Redis with TTL
                    cache_key = f"seo_result:{result.url}"
                    await self.redis_client.setex(
                        cache_key, 
                        3600,  # 1 hour TTL
                        orjson.dumps(result.model_dump(mode='json'))
                    )
                
                return True
            except Exception as e:
                logger.error("Failed to store SEO result", error=str(e), url=result.url)
                return False
    
    async def retrieve_seo_result(self, url: str, collection: str = "seo_results") -> Optional['SEOResultModel']:
        """Retrieve SEO analysis result from database asynchronously."""
        async with self._connection_semaphore:
            try:
                # Try Redis cache first
                if self.redis_client:
                    cache_key = f"seo_result:{url}"
                    cached_data = await self.redis_client.get(cache_key)
                    if cached_data:
                        return SEOResultModel(**orjson.loads(cached_data))
                
                # Fallback to MongoDB
                if self.mongo_client:
                    db = self.mongo_client.seo_database
                    doc = await db[collection].find_one({"url": url})
                    if doc:
                        # Cache in Redis for future requests
                        if self.redis_client:
                            cache_key = f"seo_result:{url}"
                            await self.redis_client.setex(
                                cache_key,
                                3600,
                                orjson.dumps(doc)
                            )
                        return SEOResultModel(**doc)
                
                return None
            except Exception as e:
                logger.error("Failed to retrieve SEO result", error=str(e), url=url)
                return None
    
    async def store_bulk_results(self, results: List['SEOResultModel'], collection: str = "seo_results") -> int:
        """Store multiple SEO results in database asynchronously."""
        async with self._connection_semaphore:
            try:
                stored_count = 0
                
                if self.mongo_client:
                    # Bulk insert to MongoDB
                    db = self.mongo_client.seo_database
                    documents = [result.model_dump(mode='json') for result in results]
                    result = await db[collection].insert_many(documents)
                    stored_count = len(result.inserted_ids)
                
                if self.redis_client:
                    # Cache individual results in Redis
                    pipeline = self.redis_client.pipeline()
                    for result in results:
                        cache_key = f"seo_result:{result.url}"
                        pipeline.setex(
                            cache_key,
                            3600,
                            orjson.dumps(result.model_dump(mode='json'))
                        )
                    await pipeline.execute()
                
                return stored_count
            except Exception as e:
                logger.error("Failed to store bulk SEO results", error=str(e))
                return 0
    
    async def get_analysis_history(self, url: str, limit: int = 10, collection: str = "seo_results") -> List['SEOResultModel']:
        """Get analysis history for a URL asynchronously."""
        async with self._connection_semaphore:
            try:
                if self.mongo_client:
                    db = self.mongo_client.seo_database
                    cursor = db[collection].find(
                        {"url": url}
                    ).sort("timestamp", -1).limit(limit)
                    
                    results = []
                    async for doc in cursor:
                        results.append(SEOResultModel(**doc))
                    
                    return results
                
                return []
            except Exception as e:
                logger.error("Failed to get analysis history", error=str(e), url=url)
                return []
    
    async def delete_old_results(self, days_old: int = 30, collection: str = "seo_results") -> int:
        """Delete old SEO results asynchronously."""
        async with self._connection_semaphore:
            try:
                if self.mongo_client:
                    db = self.mongo_client.seo_database
                    cutoff_time = time.time() - (days_old * 24 * 3600)
                    
                    result = await db[collection].delete_many({
                        "timestamp": {"$lt": cutoff_time}
                    })
                    
                    return result.deleted_count
                
                return 0
            except Exception as e:
                logger.error("Failed to delete old results", error=str(e))
                return 0
    
    async def get_database_stats(self, collection: str = "seo_results") -> Dict[str, Any]:
        """Get database statistics asynchronously."""
        async with self._connection_semaphore:
            try:
                stats = {
                    "total_documents": 0,
                    "average_score": 0.0,
                    "recent_analyses": 0,
                    "cache_hit_rate": 0.0
                }
                
                if self.mongo_client:
                    db = self.mongo_client.seo_database
                    
                    # Get total document count
                    stats["total_documents"] = await db[collection].count_documents({})
                    
                    # Get average score
                    pipeline = [
                        {"$group": {"_id": None, "avg_score": {"$avg": "$score"}}}
                    ]
                    cursor = db[collection].aggregate(pipeline)
                    async for doc in cursor:
                        stats["average_score"] = doc.get("avg_score", 0.0)
                    
                    # Get recent analyses (last 24 hours)
                    recent_time = time.time() - 86400
                    stats["recent_analyses"] = await db[collection].count_documents({
                        "timestamp": {"$gte": recent_time}
                    })
                
                if self.redis_client:
                    # Get cache hit rate (simplified)
                    info = await self.redis_client.info("stats")
                    hits = int(info.get("keyspace_hits", 0))
                    misses = int(info.get("keyspace_misses", 0))
                    total = hits + misses
                    stats["cache_hit_rate"] = (hits / total * 100) if total > 0 else 0.0
                
                return stats
            except Exception as e:
                logger.error("Failed to get database stats", error=str(e))
                return {"error": str(e)}

class AsyncExternalAPIOperations:
    """Dedicated async functions for external API operations."""
    
    def __init__(self, http_client: httpx.AsyncClient):
        
    """__init__ function."""
self.http_client = http_client
        self._request_semaphore = asyncio.Semaphore(100)  # Limit concurrent API requests
        self._rate_limit_semaphore = asyncio.Semaphore(10)  # Rate limiting
        
    async async def fetch_page_content(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """Fetch page content from external URL asynchronously."""
        async with self._request_semaphore:
            try:
                headers = {
                    'User-Agent': 'SEO-Bot/1.0',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive'
                }
                
                response = await self.http_client.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                
                return {
                    'content': response.content,
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }
            except Exception as e:
                logger.error("Failed to fetch page content", error=str(e), url=url)
                return {
                    'content': None,
                    'status_code': None,
                    'headers': {},
                    'url': url,
                    'error': str(e)
                }
    
    async def check_url_accessibility(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Check if URL is accessible asynchronously."""
        async with self._request_semaphore:
            try:
                start_time = time.time()
                response = await self.http_client.head(url, timeout=timeout)
                response_time = (time.time() - start_time) * 1000
                
                return {
                    'accessible': True,
                    'status_code': response.status_code,
                    'response_time_ms': response_time,
                    'content_length': response.headers.get('content-length'),
                    'content_type': response.headers.get('content-type')
                }
            except Exception as e:
                return {
                    'accessible': False,
                    'error': str(e),
                    'response_time_ms': 0
                }
    
    async async def fetch_robots_txt(self, base_url: str) -> Dict[str, Any]:
        """Fetch robots.txt file asynchronously."""
        async with self._request_semaphore:
            try:
                robots_url = f"{base_url.rstrip('/')}/robots.txt"
                response = await self.http_client.get(robots_url, timeout=10)
                
                if response.status_code == 200:
                    return {
                        'found': True,
                        'content': response.text,
                        'status_code': response.status_code
                    }
                else:
                    return {
                        'found': False,
                        'status_code': response.status_code
                    }
            except Exception as e:
                return {
                    'found': False,
                    'error': str(e)
                }
    
    async async def fetch_sitemap(self, sitemap_url: str) -> Dict[str, Any]:
        """Fetch sitemap asynchronously."""
        async with self._request_semaphore:
            try:
                response = await self.http_client.get(sitemap_url, timeout=30)
                response.raise_for_status()
                
                return {
                    'found': True,
                    'content': response.text,
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type')
                }
            except Exception as e:
                return {
                    'found': False,
                    'error': str(e)
                }
    
    async async def check_social_media_apis(self, url: str) -> Dict[str, Any]:
        """Check social media sharing data asynchronously."""
        async with self._request_semaphore:
            try:
                # Facebook Open Graph
                fb_url = f"https://graph.facebook.com/?id={url}&fields=og_object"
                fb_response = await self.http_client.get(fb_url, timeout=10)
                
                # Twitter Card
                twitter_url = f"https://api.twitter.com/1.1/urls/card.json?url={url}"
                twitter_response = await self.http_client.get(twitter_url, timeout=10)
                
                return {
                    'facebook': fb_response.json() if fb_response.status_code == 200 else None,
                    'twitter': twitter_response.json() if twitter_response.status_code == 200 else None
                }
            except Exception as e:
                return {
                    'error': str(e)
                }
    
    async async def fetch_webpage_metadata(self, url: str) -> Dict[str, Any]:
        """Fetch comprehensive webpage metadata asynchronously."""
        async with self._request_semaphore:
            try:
                # Fetch page content
                page_data = await self.fetch_page_content(url)
                if not page_data.get('content'):
                    return {'error': 'Failed to fetch page content'}
                
                # Parse metadata in thread pool
                def extract_metadata(content, url) -> Any:
                    soup = BeautifulSoup(content, 'lxml')
                    
                    metadata = {
                        'title': None,
                        'description': None,
                        'keywords': None,
                        'author': None,
                        'robots': None,
                        'canonical': None,
                        'og_tags': {},
                        'twitter_tags': {},
                        'structured_data': []
                    }
                    
                    # Basic meta tags
                    title_tag = soup.find('title')
                    if title_tag:
                        metadata['title'] = title_tag.get_text().strip()
                    
                    for meta in soup.find_all('meta'):
                        name = meta.get('name', '').lower()
                        property_attr = meta.get('property', '').lower()
                        content = meta.get('content', '')
                        
                        if name == 'description':
                            metadata['description'] = content
                        elif name == 'keywords':
                            metadata['keywords'] = content
                        elif name == 'author':
                            metadata['author'] = content
                        elif name == 'robots':
                            metadata['robots'] = content
                        elif property_attr.startswith('og:'):
                            metadata['og_tags'][property_attr] = content
                        elif name.startswith('twitter:'):
                            metadata['twitter_tags'][name] = content
                    
                    # Canonical URL
                    canonical = soup.find('link', rel='canonical')
                    if canonical:
                        metadata['canonical'] = canonical.get('href')
                    
                    # Structured data
                    for script in soup.find_all('script', type='application/ld+json'):
                        try:
                            data = json.loads(script.string)
                            metadata['structured_data'].append(data)
                        except:
                            continue
                    
                    return metadata
                
                # Extract metadata in thread pool
                metadata = await non_blocking_manager.run_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    extract_metadata, page_data['content'], url
                )
                
                return {
                    'url': url,
                    'status_code': page_data['status_code'],
                    'metadata': metadata
                }
                
            except Exception as e:
                return {
                    'url': url,
                    'error': str(e)
                }
    
    async def batch_check_urls(self, urls: List[str], timeout: int = 10) -> List[Dict[str, Any]]:
        """Check multiple URLs for accessibility asynchronously."""
        async with self._rate_limit_semaphore:
            try:
                tasks = [self.check_url_accessibility(url, timeout) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append({
                            'url': urls[i],
                            'accessible': False,
                            'error': str(result)
                        })
                    else:
                        result['url'] = urls[i]
                        processed_results.append(result)
                
                return processed_results
            except Exception as e:
                logger.error("Failed to batch check URLs", error=str(e))
                return [{'url': url, 'accessible': False, 'error': str(e)} for url in urls]

class AsyncDataPersistenceOperations:
    """Dedicated async functions for data persistence operations."""
    
    def __init__(self, cache_manager: CacheManager, db_operations: AsyncDatabaseOperations):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.db_operations = db_operations
        self._persistence_semaphore = asyncio.Semaphore(20)  # Limit concurrent persistence operations
        
    async def persist_seo_analysis(self, result: 'SEOResultModel', cache_ttl: int = 3600) -> bool:
        """Persist SEO analysis result to cache and database asynchronously."""
        async with self._persistence_semaphore:
            try:
                # Store in cache
                cache_key = f"seo_analysis:{result.url}"
                await self.cache_manager.set(cache_key, result.model_dump(mode='json'), cache_ttl)
                
                # Store in database
                await self.db_operations.store_seo_result(result)
                
                return True
            except Exception as e:
                logger.error("Failed to persist SEO analysis", error=str(e), url=result.url)
                return False
    
    async def persist_bulk_analyses(self, results: List['SEOResultModel'], cache_ttl: int = 3600) -> int:
        """Persist multiple SEO analyses asynchronously."""
        async with self._persistence_semaphore:
            try:
                persisted_count = 0
                
                # Store in cache
                for result in results:
                    cache_key = f"seo_analysis:{result.url}"
                    await self.cache_manager.set(cache_key, result.model_dump(mode='json'), cache_ttl)
                
                # Store in database
                persisted_count = await self.db_operations.store_bulk_results(results)
                
                return persisted_count
            except Exception as e:
                logger.error("Failed to persist bulk analyses", error=str(e))
                return 0
    
    async def backup_analysis_data(self, collection: str = "seo_results") -> Dict[str, Any]:
        """Create backup of analysis data asynchronously."""
        async with self._persistence_semaphore:
            try:
                backup_info = {
                    'timestamp': time.time(),
                    'total_records': 0,
                    'backup_size_bytes': 0,
                    'success': False
                }
                
                if self.db_operations.mongo_client:
                    db = self.db_operations.mongo_client.seo_database
                    
                    # Get all documents
                    cursor = db[collection].find({})
                    documents = []
                    async for doc in cursor:
                        documents.append(doc)
                    
                    backup_info['total_records'] = len(documents)
                    
                    # Create backup file
                    backup_filename = f"seo_backup_{int(time.time())}.json"
                    backup_data = {
                        'timestamp': time.time(),
                        'collection': collection,
                        'documents': documents
                    }
                    
                    # Write backup file in thread pool
                    def write_backup_file(filename, data) -> Any:
                        with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            json.dump(data, f, default=str)
                        return os.path.getsize(filename)
                    
                    backup_size = await non_blocking_manager.run_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        write_backup_file, backup_filename, backup_data
                    )
                    
                    backup_info['backup_size_bytes'] = backup_size
                    backup_info['backup_file'] = backup_filename
                    backup_info['success'] = True
                
                return backup_info
            except Exception as e:
                logger.error("Failed to backup analysis data", error=str(e))
                return {
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                }
    
    async def restore_analysis_data(self, backup_file: str, collection: str = "seo_results") -> Dict[str, Any]:
        """Restore analysis data from backup asynchronously."""
        async with self._persistence_semaphore:
            try:
                restore_info = {
                    'timestamp': time.time(),
                    'restored_records': 0,
                    'success': False
                }
                
                # Read backup file in thread pool
                def read_backup_file(filename) -> Any:
                    with open(filename, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        return json.load(f)
                
                backup_data = await non_blocking_manager.run_in_thread(read_backup_file, backup_file)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                if self.db_operations.mongo_client:
                    db = self.db_operations.mongo_client.seo_database
                    
                    # Clear existing collection
                    await db[collection].delete_many({})
                    
                    # Insert backup documents
                    if backup_data.get('documents'):
                        result = await db[collection].insert_many(backup_data['documents'])
                        restore_info['restored_records'] = len(result.inserted_ids)
                        restore_info['success'] = True
                
                return restore_info
            except Exception as e:
                logger.error("Failed to restore analysis data", error=str(e))
                return {
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                }
    
    async def export_analysis_data(self, format: str = "json", collection: str = "seo_results") -> Dict[str, Any]:
        """Export analysis data in specified format asynchronously."""
        async with self._persistence_semaphore:
            try:
                export_info = {
                    'timestamp': time.time(),
                    'format': format,
                    'total_records': 0,
                    'export_size_bytes': 0,
                    'success': False
                }
                
                if self.db_operations.mongo_client:
                    db = self.db_operations.mongo_client.seo_database
                    
                    # Get all documents
                    cursor = db[collection].find({})
                    documents = []
                    async for doc in cursor:
                        documents.append(doc)
                    
                    export_info['total_records'] = len(documents)
                    
                    # Export in specified format
                    export_filename = f"seo_export_{int(time.time())}.{format}"
                    
                    if format == "json":
                        def write_json_export(filename, data) -> Any:
                            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                                json.dump(data, f, default=str, indent=2)
                            return os.path.getsize(filename)
                        
                        export_size = await non_blocking_manager.run_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            write_json_export, export_filename, documents
                        )
                    
                    elif format == "csv":
                        def write_csv_export(filename, data) -> Any:
                            if data:
                                fieldnames = data[0].keys()
                                with open(filename, 'w', newline='') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    writer.writeheader()
                                    writer.writerows(data)
                            return os.path.getsize(filename)
                        
                        export_size = await non_blocking_manager.run_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            write_csv_export, export_filename, documents
                        )
                    
                    else:
                        raise ValueError(f"Unsupported export format: {format}")
                    
                    export_info['export_size_bytes'] = export_size
                    export_info['export_file'] = export_filename
                    export_info['success'] = True
                
                return export_info
            except Exception as e:
                logger.error("Failed to export analysis data", error=str(e))
                return {
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                }

# ============================================================================
# END OF DEDICATED ASYNC FUNCTIONS
# ============================================================================

class CrawlParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., description="URL to crawl", min_length=1, max_length=2048)
    depth: int = Field(default=2, ge=1, le=5, description="Crawl depth")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries")
    user_agent: str = Field(default="SEO-Bot/1.0", description="User agent", max_length=256)

class CrawlResultModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., description="Crawled URL", min_length=1, max_length=2048)
    title: Optional[str] = Field(None, description="Page title", max_length=512)
    meta_tags: Dict[str, str] = Field(default_factory=dict, description="Meta tags", max_length=100)
    headings: Dict[str, List[str]] = Field(default_factory=dict, description="Headings", max_length=50)
    links: List[Dict[str, Any]] = Field(default_factory=list, description="Links", max_length=1000)
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Images", max_length=500)
    content_length: int = Field(default=0, ge=0, le=10000000, description="Content length")
    status_code: int = Field(..., ge=100, le=599, description="HTTP status code")
    error: Optional[str] = Field(None, description="Error message", max_length=1024)

    @validator('url')
    def validate_url(cls, v) -> bool:
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @computed_field
    @property
    def has_content(self) -> bool:
        """Check if the crawl has content."""
        return self.content_length > 0 and not self.error
    
    @computed_field
    @property
    def success(self) -> bool:
        """Check if the crawl was successful."""
        return 200 <= self.status_code < 400 and not self.error

class AnalysisParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    crawl_data: CrawlResultModel = Field(..., description="Crawl data")
    keywords: List[str] = Field(default_factory=list, max_items=100, description="Keywords")
    include_meta: bool = Field(default=True, description="Include meta analysis")
    include_links: bool = Field(default=True, description="Include link analysis")
    include_images: bool = Field(default=True, description="Include image analysis")

class AnalysisResultModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    title_score: float = Field(..., ge=0, le=100, description="Title score")
    description_score: float = Field(..., ge=0, le=100, description="Description score")
    headings_score: float = Field(..., ge=0, le=100, description="Headings score")
    keywords_score: float = Field(..., ge=0, le=100, description="Keywords score")
    links_score: float = Field(..., ge=0, le=100, description="Links score")
    images_score: float = Field(..., ge=0, le=100, description="Images score")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions", max_length=50)
    warnings: List[str] = Field(default_factory=list, description="Warnings", max_length=50)
    errors: List[str] = Field(default_factory=list, description="Errors", max_length=50)
    
    @computed_field
    @property
    def overall_score(self) -> float:
        """Calculate overall SEO score."""
        scores = [
            self.title_score,
            self.description_score,
            self.headings_score,
            self.keywords_score,
            self.links_score,
            self.images_score
        ]
        return sum(scores) / len(scores)
    
    @computed_field
    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return len(self.errors) > 0 or len(self.warnings) > 0
    
    @computed_field
    @property
    def issue_count(self) -> int:
        """Get total number of issues."""
        return len(self.errors) + len(self.warnings)

class PerformanceParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., description="URL to test", min_length=1, max_length=2048)
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout")
    follow_redirects: bool = Field(default=True, description="Follow redirects")

class PerformanceResultModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    load_time: Optional[float] = Field(None, ge=0, le=60, description="Load time")
    status_code: Optional[int] = Field(None, ge=100, le=599, description="Status code")
    content_length: int = Field(default=0, ge=0, le=10000000, description="Content length")
    headers: Dict[str, str] = Field(default_factory=dict, description="Headers", max_length=100)
    error: Optional[str] = Field(None, description="Error message", max_length=1024)
    
    @computed_field
    @property
    def is_fast(self) -> bool:
        """Check if the page loads fast."""
        return self.load_time is not None and self.load_time < 3.0
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """Check if the performance test was successful."""
        return self.status_code is not None and 200 <= self.status_code < 400 and not self.error
    
    @computed_field
    @property
    def performance_score(self) -> float:
        """Calculate performance score based on load time."""
        if self.load_time is None or self.error:
            return 0.0
        if self.load_time < 1.0:
            return 100.0
        elif self.load_time < 3.0:
            return 80.0
        elif self.load_time < 5.0:
            return 60.0
        else:
            return max(0.0, 100.0 - (self.load_time - 5.0) * 10)

class SEOParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., description="URL to analyze", min_length=1, max_length=2048)
    keywords: List[str] = Field(default_factory=list, max_items=100, description="Keywords")
    depth: int = Field(default=2, ge=1, le=5, description="Crawl depth")
    include_meta: bool = Field(default=True, description="Include meta analysis")
    include_links: bool = Field(default=True, description="Include link analysis")
    include_images: bool = Field(default=True, description="Include image analysis")
    include_performance: bool = Field(default=True, description="Include performance analysis")
    cache_ttl: int = Field(default=3600, ge=0, le=86400, description="Cache TTL")

    @validator('url')
    def validate_url(cls, v) -> bool:
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class SEOResultModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., description="Analyzed URL", min_length=1, max_length=2048)
    score: float = Field(..., ge=0, le=100, description="SEO score")
    title: Optional[str] = Field(None, max_length=512, description="Page title")
    description: Optional[str] = Field(None, max_length=1024, description="Meta description")
    keywords: List[str] = Field(default_factory=list, description="Keywords", max_length=100)
    headings: Dict[str, List[str]] = Field(default_factory=dict, description="Headings", max_length=50)
    links: List[Dict[str, Any]] = Field(default_factory=list, description="Links", max_length=1000)
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Images", max_length=500)
    meta_tags: Dict[str, str] = Field(default_factory=dict, description="Meta tags", max_length=100)
    performance: PerformanceResultModel = Field(default_factory=PerformanceResultModel, description="Performance data")
    errors: List[str] = Field(default_factory=list, description="Errors", max_length=50)
    warnings: List[str] = Field(default_factory=list, description="Warnings", max_length=50)
    suggestions: List[str] = Field(default_factory=list, description="Suggestions", max_length=50)
    timestamp: float = Field(default_factory=time.time, description="Timestamp")
    
    @computed_field
    @property
    def is_optimized(self) -> bool:
        """Check if the page is well optimized."""
        return self.score >= 80.0 and len(self.errors) == 0
    
    @computed_field
    @property
    def needs_improvement(self) -> bool:
        """Check if the page needs improvement."""
        return self.score < 60.0 or len(self.errors) > 0
    
    @computed_field
    @property
    def issue_summary(self) -> Dict[str, int]:
        """Get summary of issues."""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'suggestions': len(self.suggestions)
        }

class CacheParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    key: str = Field(..., description="Cache key", min_length=1, max_length=256)
    data: Dict[str, Any] = Field(..., description="Data to cache")
    ttl: int = Field(default=3600, ge=0, le=86400, description="TTL in seconds")

class CacheResultModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    success: bool = Field(..., description="Success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Cached data")
    error: Optional[str] = Field(None, description="Error message", max_length=1024)

class RateLimitParamsModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    client_id: str = Field(..., description="Client ID", min_length=1, max_length=128)
    max_requests: int = Field(default=100, ge=1, le=10000, description="Max requests")
    window: int = Field(default=60, ge=1, le=3600, description="Time window")

class RateLimitResultModel(BaseModel):
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    allowed: bool = Field(..., description="Request allowed")
    remaining: int = Field(..., ge=0, description="Remaining requests")
    reset_time: float = Field(..., description="Reset time")
    
    @computed_field
    @property
    def is_rate_limited(self) -> bool:
        """Check if the request is rate limited."""
        return not self.allowed
    
    @computed_field
    @property
    def time_until_reset(self) -> float:
        """Get time until rate limit resets."""
        return max(0.0, self.reset_time - time.time())

# Configuration
@dataclass
class Config:
    """Application configuration with validation."""
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8000, description="Port to bind")
    workers: int = Field(default=multiprocessing.cpu_count(), description="Number of workers")
    max_connections: int = Field(default=1000, description="Max connections")
    timeout: int = Field(default=30, description="Request timeout")
    rate_limit: int = Field(default=100, description="Rate limit per minute")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    mongo_url: str = Field(default="mongodb://localhost:27017", description="MongoDB URL")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="JWT secret")
    bcrypt_rounds: int = Field(default=12, description="Bcrypt rounds")
    
    @validator('port')
    def validate_port(cls, v) -> bool:
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('workers')
    def validate_workers(cls, v) -> bool:
        if v < 1:
            raise ValueError('Workers must be at least 1')
        return min(v, multiprocessing.cpu_count() * 2)

# Global configuration
config = Config()

# Structured logging
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

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

# Global state
class GlobalState:
    """Global application state."""
    def __init__(self) -> Any:
        self.redis_client: Optional[redis.Redis] = None
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.is_shutting_down = False
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

# Dependency Injection Container
class DependencyContainer:
    """FastAPI dependency injection container for managing shared resources."""
    
    def __init__(self) -> Any:
        self._cache_manager: Optional[CacheManager] = None
        self._static_cache: Optional[StaticDataCache] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._lazy_loader: Optional[LazyDataLoader] = None
        self._bulk_processor: Optional[BulkSEOProcessor] = None
        self._redis_client: Optional[redis.Redis] = None
        self._mongo_client: Optional[AsyncIOMotorClient] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._async_db_operations: Optional[AsyncDatabaseOperations] = None
        self._async_api_operations: Optional[AsyncExternalAPIOperations] = None
        self._async_persistence_operations: Optional[AsyncDataPersistenceOperations] = None
        self._config: Optional[Config] = None
        self._logger: Optional[structlog.BoundLogger] = None
        self._startup_time: float = time.time()
        self._request_count: int = 0
        self._error_count: int = 0
    
    @property
    def cache_manager(self) -> CacheManager:
        """Get cache manager instance."""
        if self._cache_manager is None:
            self._cache_manager = CacheManager()
        return self._cache_manager
    
    @property
    def static_cache(self) -> StaticDataCache:
        """Get static cache instance."""
        if self._static_cache is None:
            self._static_cache = StaticDataCache()
        return self._static_cache
    
    @property
    def rate_limiter(self) -> RateLimiter:
        """Get rate limiter instance."""
        if self._rate_limiter is None:
            self._rate_limiter = RateLimiter(config.rate_limit, 60)
        return self._rate_limiter
    
    @property
    def lazy_loader(self) -> LazyDataLoader:
        """Get lazy loader instance."""
        if self._lazy_loader is None:
            self._lazy_loader = LazyDataLoader(LazyLoadingConfig())
        return self._lazy_loader
    
    @property
    def bulk_processor(self) -> BulkSEOProcessor:
        """Get bulk processor instance."""
        if self._bulk_processor is None:
            self._bulk_processor = BulkSEOProcessor(LazyLoadingConfig())
        return self._bulk_processor
    
    @property
    def config(self) -> Config:
        """Get configuration instance."""
        if self._config is None:
            self._config = config
        return self._config
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance."""
        if self._logger is None:
            self._logger = structlog.get_logger()
        return self._logger
    
    @property
    def async_db_operations(self) -> AsyncDatabaseOperations:
        """Get async database operations instance."""
        if self._async_db_operations is None:
            self._async_db_operations = AsyncDatabaseOperations(
                redis_client=self._redis_client,
                mongo_client=self._mongo_client
            )
        return self._async_db_operations
    
    @property
    async def async_api_operations(self) -> AsyncExternalAPIOperations:
        """Get async external API operations instance."""
        if self._async_api_operations is None:
            if self._http_client is None:
                # Create HTTP client if not available
                self._http_client = httpx.AsyncClient(
                    timeout=30,
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    http2=True,
                    follow_redirects=True
                )
            self._async_api_operations = AsyncExternalAPIOperations(self._http_client)
        return self._async_api_operations
    
    @property
    def async_persistence_operations(self) -> AsyncDataPersistenceOperations:
        """Get async data persistence operations instance."""
        if self._async_persistence_operations is None:
            self._async_persistence_operations = AsyncDataPersistenceOperations(
                cache_manager=self.cache_manager,
                db_operations=self.async_db_operations
            )
        return self._async_persistence_operations
    
    async def increment_request_count(self) -> None:
        """Increment request counter."""
        self._request_count += 1
    
    def increment_error_count(self) -> None:
        """Increment error counter."""
        self._error_count += 1
    
    @property
    async def request_count(self) -> int:
        """Get total request count."""
        return self._request_count
    
    @property
    def error_count(self) -> int:
        """Get total error count."""
        return self._error_count
    
    @property
    def startup_time(self) -> float:
        """Get application startup time."""
        return self._startup_time
    
    @property
    def uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self._startup_time

# Performance Metrics Manager
class PerformanceMetricsManager:
    """Comprehensive performance metrics collection and analysis."""
    
    def __init__(self, max_history_size: int = 10000):
        
    """__init__ function."""
self.max_history_size = max_history_size
        self.lock = threading.Lock()
        
        # Response time tracking
        self.response_times = deque(maxlen=max_history_size)
        self.latency_times = deque(maxlen=max_history_size)
        self.processing_times = deque(maxlen=max_history_size)
        
        # Throughput tracking
        self.request_timestamps = deque(maxlen=max_history_size)
        self.request_sizes = deque(maxlen=max_history_size)
        self.response_sizes = deque(maxlen=max_history_size)
        
        # Error tracking
        self.error_count = 0
        self.timeout_count = 0
        self.total_requests = 0
        
        # Cache tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # System metrics
        self.system_metrics = deque(maxlen=1000)  # Keep last 1000 system snapshots
        
        # API endpoint metrics
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'response_times': deque(maxlen=1000),
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        
        # Performance thresholds
        self.thresholds = PerformanceThresholdsModel()
        
        # Start background monitoring
        self.monitoring_task = None
        self.is_monitoring = False
    
    def start_monitoring(self) -> Any:
        """Start background performance monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitor_performance())
    
    def stop_monitoring(self) -> Any:
        """Stop background performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
    
    async def _monitor_performance(self) -> Any:
        """Background task for continuous performance monitoring."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check performance thresholds
                await self._check_performance_thresholds()
                
                # Log performance summary
                await self._log_performance_summary()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error("Performance monitoring error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_system_metrics(self) -> Any:
        """Collect current system metrics."""
        try:
            process = psutil.Process()
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_bandwidth = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
            
            system_metric = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'network_bandwidth_mb': network_bandwidth
            }
            
            with self.lock:
                self.system_metrics.append(system_metric)
                
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
    
    async def _check_performance_thresholds(self) -> Any:
        """Check if performance metrics exceed thresholds."""
        try:
            current_metrics = self.get_current_metrics()
            
            # Check response time
            if current_metrics.response_time_ms > self.thresholds.response_time_critical_ms:
                logger.warning("Critical response time threshold exceeded", 
                             response_time=current_metrics.response_time_ms,
                             threshold=self.thresholds.response_time_critical_ms)
            
            elif current_metrics.response_time_ms > self.thresholds.response_time_warning_ms:
                logger.warning("Warning response time threshold exceeded",
                             response_time=current_metrics.response_time_ms,
                             threshold=self.thresholds.response_time_warning_ms)
            
            # Check error rate
            if current_metrics.error_rate > self.thresholds.error_rate_critical_percent:
                logger.error("Critical error rate threshold exceeded",
                           error_rate=current_metrics.error_rate,
                           threshold=self.thresholds.error_rate_critical_percent)
            
            elif current_metrics.error_rate > self.thresholds.error_rate_warning_percent:
                logger.warning("Warning error rate threshold exceeded",
                             error_rate=current_metrics.error_rate,
                             threshold=self.thresholds.error_rate_warning_percent)
            
            # Check CPU usage
            if current_metrics.cpu_usage_percent > self.thresholds.cpu_usage_critical_percent:
                logger.error("Critical CPU usage threshold exceeded",
                           cpu_usage=current_metrics.cpu_usage_percent,
                           threshold=self.thresholds.cpu_usage_critical_percent)
            
            elif current_metrics.cpu_usage_percent > self.thresholds.cpu_usage_warning_percent:
                logger.warning("Warning CPU usage threshold exceeded",
                             cpu_usage=current_metrics.cpu_usage_percent,
                             threshold=self.thresholds.cpu_usage_warning_percent)
            
            # Check memory usage
            if current_metrics.memory_usage_percent > self.thresholds.memory_usage_critical_percent:
                logger.error("Critical memory usage threshold exceeded",
                           memory_usage=current_metrics.memory_usage_percent,
                           threshold=self.thresholds.memory_usage_critical_percent)
            
            elif current_metrics.memory_usage_percent > self.thresholds.memory_usage_warning_percent:
                logger.warning("Warning memory usage threshold exceeded",
                             memory_usage=current_metrics.memory_usage_percent,
                             threshold=self.thresholds.memory_usage_warning_percent)
                
        except Exception as e:
            logger.error("Error checking performance thresholds", error=str(e))
    
    async def _log_performance_summary(self) -> Any:
        """Log periodic performance summary."""
        try:
            current_metrics = self.get_current_metrics()
            
            logger.info("Performance summary",
                       response_time_ms=current_metrics.response_time_ms,
                       requests_per_second=current_metrics.requests_per_second,
                       error_rate=current_metrics.error_rate,
                       cache_hit_rate=current_metrics.cache_hit_rate,
                       cpu_usage=current_metrics.cpu_usage_percent,
                       memory_usage=current_metrics.memory_usage_percent,
                       performance_score=current_metrics.performance_score)
                       
        except Exception as e:
            logger.error("Error logging performance summary", error=str(e))
    
    def record_request(self, api_metrics: APIMetricsModel):
        """Record API request metrics."""
        with self.lock:
            # Update total counts
            self.total_requests += 1
            
            # Record response time
            self.response_times.append(api_metrics.response_time_ms)
            
            # Record request/response sizes
            self.request_sizes.append(api_metrics.request_size_bytes)
            self.response_sizes.append(api_metrics.response_size_bytes)
            
            # Record timestamp for throughput calculation
            self.request_timestamps.append(api_metrics.request_start_time)
            
            # Record errors
            if api_metrics.is_error:
                self.error_count += 1
            
            if api_metrics.is_timeout:
                self.timeout_count += 1
            
            # Record cache hits/misses
            if api_metrics.is_cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Record endpoint-specific metrics
            endpoint_key = f"{api_metrics.method}:{api_metrics.endpoint}"
            self.endpoint_metrics[endpoint_key]['count'] += 1
            self.endpoint_metrics[endpoint_key]['response_times'].append(api_metrics.response_time_ms)
            
            if api_metrics.is_error:
                self.endpoint_metrics[endpoint_key]['errors'] += 1
            
            if api_metrics.is_cache_hit:
                self.endpoint_metrics[endpoint_key]['cache_hits'] += 1
            else:
                self.endpoint_metrics[endpoint_key]['cache_misses'] += 1
    
    def get_current_metrics(self) -> PerformanceMetricsModel:
        """Get current performance metrics."""
        with self.lock:
            # Calculate response time statistics
            response_time_ms = 0.0
            if self.response_times:
                response_time_ms = statistics.mean(self.response_times)
            
            # Calculate throughput
            requests_per_second = 0.0
            if len(self.request_timestamps) >= 2:
                time_window = self.request_timestamps[-1] - self.request_timestamps[0]
                if time_window > 0:
                    requests_per_second = len(self.request_timestamps) / time_window
            
            # Calculate throughput in MB/s
            throughput_mbps = 0.0
            if self.response_sizes and self.response_times:
                total_bytes = sum(self.response_sizes)
                total_time = sum(self.response_times) / 1000  # Convert to seconds
                if total_time > 0:
                    throughput_mbps = (total_bytes / 1024 / 1024) / total_time
            
            # Calculate error rates
            error_rate = 0.0
            if self.total_requests > 0:
                error_rate = (self.error_count / self.total_requests) * 100
            
            timeout_rate = 0.0
            if self.total_requests > 0:
                timeout_rate = (self.timeout_count / self.total_requests) * 100
            
            # Calculate cache hit rate
            cache_hit_rate = 0.0
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                cache_hit_rate = (self.cache_hits / total_cache_requests) * 100
            
            cache_miss_rate = 100.0 - cache_hit_rate
            
            # Get current system metrics
            cpu_usage_percent = 0.0
            memory_usage_mb = 0.0
            memory_usage_percent = 0.0
            network_bandwidth_mbps = 0.0
            
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                cpu_usage_percent = latest_system['cpu_percent']
                memory_usage_mb = latest_system['memory_mb']
                memory_usage_percent = latest_system['memory_percent']
                network_bandwidth_mbps = latest_system['network_bandwidth_mb']
            
            return PerformanceMetricsModel(
                response_time_ms=response_time_ms,
                latency_ms=response_time_ms * 0.3,  # Estimate 30% as network latency
                processing_time_ms=response_time_ms * 0.7,  # Estimate 70% as processing time
                requests_per_second=requests_per_second,
                throughput_mbps=throughput_mbps,
                cpu_usage_percent=cpu_usage_percent,
                memory_usage_mb=memory_usage_mb,
                memory_usage_percent=memory_usage_percent,
                cache_hit_rate=cache_hit_rate,
                cache_miss_rate=cache_miss_rate,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
                db_connection_pool_size=50,  # Default value
                db_query_time_ms=response_time_ms * 0.2,  # Estimate 20% as DB time
                network_bandwidth_mbps=network_bandwidth_mbps,
                network_latency_ms=response_time_ms * 0.3
            )
    
    def get_endpoint_metrics(self, endpoint: str = None) -> Dict[str, Any]:
        """Get metrics for specific endpoint or all endpoints."""
        with self.lock:
            if endpoint:
                if endpoint in self.endpoint_metrics:
                    metrics = self.endpoint_metrics[endpoint]
                    response_times = list(metrics['response_times'])
                    
                    return {
                        'endpoint': endpoint,
                        'total_requests': metrics['count'],
                        'average_response_time_ms': statistics.mean(response_times) if response_times else 0,
                        'min_response_time_ms': min(response_times) if response_times else 0,
                        'max_response_time_ms': max(response_times) if response_times else 0,
                        'error_count': metrics['errors'],
                        'error_rate': (metrics['errors'] / metrics['count'] * 100) if metrics['count'] > 0 else 0,
                        'cache_hits': metrics['cache_hits'],
                        'cache_misses': metrics['cache_misses'],
                        'cache_hit_rate': (metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) * 100) if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0
                    }
                else:
                    return {}
            else:
                # Return all endpoint metrics
                result = {}
                for endpoint_key, metrics in self.endpoint_metrics.items():
                    response_times = list(metrics['response_times'])
                    
                    result[endpoint_key] = {
                        'total_requests': metrics['count'],
                        'average_response_time_ms': statistics.mean(response_times) if response_times else 0,
                        'min_response_time_ms': min(response_times) if response_times else 0,
                        'max_response_time_ms': max(response_times) if response_times else 0,
                        'error_count': metrics['errors'],
                        'error_rate': (metrics['errors'] / metrics['count'] * 100) if metrics['count'] > 0 else 0,
                        'cache_hits': metrics['cache_hits'],
                        'cache_misses': metrics['cache_misses'],
                        'cache_hit_rate': (metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) * 100) if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0
                    }
                return result
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get current performance alerts based on thresholds."""
        alerts = []
        current_metrics = self.get_current_metrics()
        
        # Response time alerts
        if current_metrics.response_time_ms > self.thresholds.response_time_critical_ms:
            alerts.append({
                'level': 'critical',
                'metric': 'response_time',
                'value': current_metrics.response_time_ms,
                'threshold': self.thresholds.response_time_critical_ms,
                'message': f'Response time {current_metrics.response_time_ms:.2f}ms exceeds critical threshold {self.thresholds.response_time_critical_ms}ms'
            })
        elif current_metrics.response_time_ms > self.thresholds.response_time_warning_ms:
            alerts.append({
                'level': 'warning',
                'metric': 'response_time',
                'value': current_metrics.response_time_ms,
                'threshold': self.thresholds.response_time_warning_ms,
                'message': f'Response time {current_metrics.response_time_ms:.2f}ms exceeds warning threshold {self.thresholds.response_time_warning_ms}ms'
            })
        
        # Error rate alerts
        if current_metrics.error_rate > self.thresholds.error_rate_critical_percent:
            alerts.append({
                'level': 'critical',
                'metric': 'error_rate',
                'value': current_metrics.error_rate,
                'threshold': self.thresholds.error_rate_critical_percent,
                'message': f'Error rate {current_metrics.error_rate:.2f}% exceeds critical threshold {self.thresholds.error_rate_critical_percent}%'
            })
        elif current_metrics.error_rate > self.thresholds.error_rate_warning_percent:
            alerts.append({
                'level': 'warning',
                'metric': 'error_rate',
                'value': current_metrics.error_rate,
                'threshold': self.thresholds.error_rate_warning_percent,
                'message': f'Error rate {current_metrics.error_rate:.2f}% exceeds warning threshold {self.thresholds.error_rate_warning_percent}%'
            })
        
        # CPU usage alerts
        if current_metrics.cpu_usage_percent > self.thresholds.cpu_usage_critical_percent:
            alerts.append({
                'level': 'critical',
                'metric': 'cpu_usage',
                'value': current_metrics.cpu_usage_percent,
                'threshold': self.thresholds.cpu_usage_critical_percent,
                'message': f'CPU usage {current_metrics.cpu_usage_percent:.2f}% exceeds critical threshold {self.thresholds.cpu_usage_critical_percent}%'
            })
        elif current_metrics.cpu_usage_percent > self.thresholds.cpu_usage_warning_percent:
            alerts.append({
                'level': 'warning',
                'metric': 'cpu_usage',
                'value': current_metrics.cpu_usage_percent,
                'threshold': self.thresholds.cpu_usage_warning_percent,
                'message': f'CPU usage {current_metrics.cpu_usage_percent:.2f}% exceeds warning threshold {self.thresholds.cpu_usage_warning_percent}%'
            })
        
        # Memory usage alerts
        if current_metrics.memory_usage_percent > self.thresholds.memory_usage_critical_percent:
            alerts.append({
                'level': 'critical',
                'metric': 'memory_usage',
                'value': current_metrics.memory_usage_percent,
                'threshold': self.thresholds.memory_usage_critical_percent,
                'message': f'Memory usage {current_metrics.memory_usage_percent:.2f}% exceeds critical threshold {self.thresholds.memory_usage_critical_percent}%'
            })
        elif current_metrics.memory_usage_percent > self.thresholds.memory_usage_warning_percent:
            alerts.append({
                'level': 'warning',
                'metric': 'memory_usage',
                'value': current_metrics.memory_usage_percent,
                'threshold': self.thresholds.memory_usage_warning_percent,
                'message': f'Memory usage {current_metrics.memory_usage_percent:.2f}% exceeds warning threshold {self.thresholds.memory_usage_warning_percent}%'
            })
        
        return alerts
    
    def reset_metrics(self) -> Any:
        """Reset all performance metrics."""
        with self.lock:
            self.response_times.clear()
            self.latency_times.clear()
            self.processing_times.clear()
            self.request_timestamps.clear()
            self.request_sizes.clear()
            self.response_sizes.clear()
            self.system_metrics.clear()
            self.endpoint_metrics.clear()
            self.error_count = 0
            self.timeout_count = 0
            self.total_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0

# Global dependency container
container = DependencyContainer()

# Legacy global state for backward compatibility
state = GlobalState()

# Global performance metrics manager
performance_manager = PerformanceMetricsManager()

# Pydantic models
class SEORequest(BaseModel):
    """SEO analysis request model."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., description="URL to analyze", min_length=1, max_length=2048)
    keywords: List[str] = Field(default_factory=list, description="Keywords to check", max_items=100)
    depth: int = Field(default=2, ge=1, le=5, description="Crawl depth")
    include_meta: bool = Field(default=True, description="Include meta analysis")
    include_links: bool = Field(default=True, description="Include link analysis")
    include_images: bool = Field(default=True, description="Include image analysis")
    include_performance: bool = Field(default=True, description="Include performance analysis")
    
    @validator('url')
    def validate_url(cls, v) -> bool:
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class SEOResponse(BaseModel):
    """SEO analysis response model."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    url: str = Field(..., min_length=1, max_length=2048)
    score: float = Field(..., ge=0, le=100, description="SEO score")
    title: Optional[str] = Field(None, max_length=512)
    description: Optional[str] = Field(None, max_length=1024)
    keywords: List[str] = Field(default_factory=list, max_items=100)
    headings: Dict[str, List[str]] = Field(default_factory=dict, max_length=50)
    links: List[Dict[str, Any]] = Field(default_factory=list, max_items=1000)
    images: List[Dict[str, Any]] = Field(default_factory=list, max_items=500)
    meta_tags: Dict[str, str] = Field(default_factory=dict, max_length=100)
    performance: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list, max_items=50)
    warnings: List[str] = Field(default_factory=list, max_items=50)
    suggestions: List[str] = Field(default_factory=list, max_items=50)
    timestamp: float = Field(default_factory=time.time)
    
    @computed_field
    @property
    def is_optimized(self) -> bool:
        """Check if the page is well optimized."""
        return self.score >= 80.0 and len(self.errors) == 0
    
    @computed_field
    @property
    def needs_improvement(self) -> bool:
        """Check if the page needs improvement."""
        return self.score < 60.0 or len(self.errors) > 0
    
    @computed_field
    @property
    def issue_count(self) -> int:
        """Get total number of issues."""
        return len(self.errors) + len(self.warnings)

# Lazy Loading Models and Utilities
class LazyLoadingConfig(BaseModel):
    """Configuration for lazy loading operations."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    chunk_size: int = Field(default=100, ge=10, le=1000, description="Chunk size for lazy loading")
    max_items: int = Field(default=10000, ge=100, le=100000, description="Maximum items to load")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    enable_pagination: bool = Field(default=True, description="Enable pagination")
    cache_chunks: bool = Field(default=True, description="Cache individual chunks")
    compression_threshold: int = Field(default=1024, ge=100, le=10000, description="Compression threshold in bytes")

class PaginationParams(BaseModel):
    """Pagination parameters for lazy loading."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=50, ge=1, le=1000, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")

class LazyLoadResult(BaseModel):
    """Result wrapper for lazy loading operations."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Loaded data")
    total_count: int = Field(default=0, ge=0, description="Total number of items")
    page: int = Field(default=1, ge=1, description="Current page")
    page_size: int = Field(default=50, ge=1, description="Items per page")
    has_next: bool = Field(default=False, description="Has next page")
    has_previous: bool = Field(default=False, description="Has previous page")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
    
    @computed_field
    @property
    def is_empty(self) -> bool:
        """Check if the result is empty."""
        return len(self.data) == 0
    
    @computed_field
    @property
    def start_index(self) -> int:
        """Get the start index of current page."""
        return (self.page - 1) * self.page_size + 1
    
    @computed_field
    @property
    def end_index(self) -> int:
        """Get the end index of current page."""
        return min(self.page * self.page_size, self.total_count)

class BulkSEOParams(BaseModel):
    """Parameters for bulk SEO analysis with lazy loading."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    urls: List[str] = Field(default_factory=list, max_items=1000, description="URLs to analyze")
    keywords: List[str] = Field(default_factory=list, max_items=100, description="Keywords to check")
    depth: int = Field(default=2, ge=1, le=5, description="Crawl depth")
    include_meta: bool = Field(default=True, description="Include meta analysis")
    include_links: bool = Field(default=True, description="Include link analysis")
    include_images: bool = Field(default=True, description="Include image analysis")
    include_performance: bool = Field(default=True, description="Include performance analysis")
    lazy_loading: LazyLoadingConfig = Field(default_factory=LazyLoadingConfig, description="Lazy loading configuration")
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="Pagination parameters")

class BulkSEOResult(BaseModel):
    """Result for bulk SEO analysis with lazy loading."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    results: List[SEOResultModel] = Field(default_factory=list, description="SEO analysis results")
    total_urls: int = Field(default=0, ge=0, description="Total URLs processed")
    successful_analyses: int = Field(default=0, ge=0, description="Successful analyses")
    failed_analyses: int = Field(default=0, ge=0, description="Failed analyses")
    average_score: float = Field(default=0.0, ge=0, le=100, description="Average SEO score")
    processing_time: float = Field(default=0.0, ge=0, description="Total processing time")
    lazy_loading: LazyLoadingConfig = Field(default_factory=LazyLoadingConfig, description="Lazy loading configuration")
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="Pagination parameters")
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_urls == 0:
            return 0.0
        return (self.successful_analyses / self.total_urls) * 100
    
    @computed_field
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_urls == 0:
            return 0.0
        return (self.failed_analyses / self.total_urls) * 100

# Exception classes
class SEOException(Exception):
    """Base SEO exception."""
    pass

class URLValidationError(SEOException):
    """URL validation error."""
    pass

class CrawlError(SEOException):
    """Crawl error."""
    pass

class AnalysisError(SEOException):
    """Analysis error."""
    pass

# Performance Metrics Models
class PerformanceMetricsModel(BaseModel):
    """Comprehensive performance metrics model."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    # Response Time Metrics
    response_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    latency_ms: float = Field(..., ge=0, description="Network latency in milliseconds")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    
    # Throughput Metrics
    requests_per_second: float = Field(..., ge=0, description="Requests per second")
    throughput_mbps: float = Field(..., ge=0, description="Throughput in MB/s")
    
    # System Metrics
    cpu_usage_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage_mb: float = Field(..., ge=0, description="Memory usage in MB")
    memory_usage_percent: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    
    # Cache Metrics
    cache_hit_rate: float = Field(..., ge=0, le=100, description="Cache hit rate percentage")
    cache_miss_rate: float = Field(..., ge=0, le=100, description="Cache miss rate percentage")
    
    # Error Metrics
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage")
    timeout_rate: float = Field(..., ge=0, le=100, description="Timeout rate percentage")
    
    # Database Metrics
    db_connection_pool_size: int = Field(..., ge=0, description="Database connection pool size")
    db_query_time_ms: float = Field(..., ge=0, description="Database query time in milliseconds")
    
    # Network Metrics
    network_bandwidth_mbps: float = Field(..., ge=0, description="Network bandwidth in MB/s")
    network_latency_ms: float = Field(..., ge=0, description="Network latency in milliseconds")
    
    # Timestamp
    timestamp: float = Field(default_factory=time.time, description="Metrics timestamp")
    
    @computed_field
    @property
    def total_time_ms(self) -> float:
        """Total time including latency and processing."""
        return self.latency_ms + self.processing_time_ms
    
    @computed_field
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Base score starts at 100
        score = 100.0
        
        # Penalize slow response times
        if self.response_time_ms > 1000:  # > 1 second
            score -= 20
        elif self.response_time_ms > 500:  # > 500ms
            score -= 10
        elif self.response_time_ms > 200:  # > 200ms
            score -= 5
        
        # Penalize high error rates
        score -= self.error_rate * 0.5
        
        # Penalize high CPU usage
        if self.cpu_usage_percent > 80:
            score -= 15
        elif self.cpu_usage_percent > 60:
            score -= 10
        
        # Penalize high memory usage
        if self.memory_usage_percent > 80:
            score -= 15
        elif self.memory_usage_percent > 60:
            score -= 10
        
        # Bonus for good cache hit rate
        if self.cache_hit_rate > 80:
            score += 10
        elif self.cache_hit_rate > 60:
            score += 5
        
        return max(0.0, min(100.0, score))

class APIMetricsModel(BaseModel):
    """API-specific performance metrics."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field(..., description="HTTP method")
    status_code: int = Field(..., ge=100, le=599, description="HTTP status code")
    
    # Timing metrics
    request_start_time: float = Field(..., description="Request start timestamp")
    request_end_time: float = Field(..., description="Request end timestamp")
    response_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    
    # Request metrics
    request_size_bytes: int = Field(..., ge=0, description="Request size in bytes")
    response_size_bytes: int = Field(..., ge=0, description="Response size in bytes")
    
    # Performance indicators
    is_cache_hit: bool = Field(..., description="Whether request was served from cache")
    is_error: bool = Field(..., description="Whether request resulted in error")
    is_timeout: bool = Field(..., description="Whether request timed out")
    
    # Client information
    client_ip: str = Field(..., description="Client IP address")
    user_agent: str = Field(default="", description="User agent string")
    
    @computed_field
    @property
    def duration_ms(self) -> float:
        """Request duration in milliseconds."""
        return (self.request_end_time - self.request_start_time) * 1000
    
    @computed_field
    @property
    def throughput_kbps(self) -> float:
        """Throughput in KB/s."""
        if self.duration_ms > 0:
            return (self.response_size_bytes / 1024) / (self.duration_ms / 1000)
        return 0.0

class PerformanceThresholdsModel(BaseModel):
    """Performance thresholds for alerting."""
    model_config = ConfigDict(
        json_encoders={},
        validate_assignment=True,
        extra='forbid',
        frozen=True,
        use_enum_values=True,
        populate_by_name=True
    )
    
    # Response time thresholds
    response_time_warning_ms: float = Field(default=500, ge=0, description="Warning threshold for response time")
    response_time_critical_ms: float = Field(default=1000, ge=0, description="Critical threshold for response time")
    
    # Throughput thresholds
    min_throughput_rps: float = Field(default=10, ge=0, description="Minimum throughput in requests per second")
    target_throughput_rps: float = Field(default=100, ge=0, description="Target throughput in requests per second")
    
    # Error rate thresholds
    error_rate_warning_percent: float = Field(default=5, ge=0, le=100, description="Warning threshold for error rate")
    error_rate_critical_percent: float = Field(default=10, ge=0, le=100, description="Critical threshold for error rate")
    
    # System resource thresholds
    cpu_usage_warning_percent: float = Field(default=70, ge=0, le=100, description="Warning threshold for CPU usage")
    cpu_usage_critical_percent: float = Field(default=90, ge=0, le=100, description="Critical threshold for CPU usage")
    memory_usage_warning_percent: float = Field(default=80, ge=0, le=100, description="Warning threshold for memory usage")
    memory_usage_critical_percent: float = Field(default=95, ge=0, le=100, description="Critical threshold for memory usage")

class LazyLoadingError(SEOException):
    """Exception for lazy loading errors."""
    pass
    """Validate URL format."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def sanitize_url(url: str) -> str:
    """Sanitize URL for safe processing."""
    return urllib.parse.quote(url, safe=':/?=&')

def calculate_seo_score(metrics: Dict[str, Any]) -> float:
    """Calculate SEO score from metrics."""
    score = 0.0
    weights = {
        'title': 0.15,
        'description': 0.10,
        'headings': 0.10,
        'keywords': 0.10,
        'links': 0.10,
        'images': 0.05,
        'performance': 0.20,
        'accessibility': 0.10,
        'mobile_friendly': 0.10
    }
    
    for metric, weight in weights.items():
        if metric in metrics:
            score += metrics[metric] * weight
    
    return min(100.0, max(0.0, score))

# Advanced Caching System with Redis and In-Memory Fallback
class CacheManager:
    """Advanced caching system with Redis and in-memory fallback."""
    
    def __init__(self) -> Any:
        self.memory_cache = {}
        self.memory_ttl = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'redis_hits': 0,
            'memory_hits': 0
        }
    
    async def get(self, key: str, cache_type: str = 'auto') -> dict | None:
        """Get cached result with intelligent fallback."""
        if cache_type == 'memory' or not state.redis_client:
            return self._get_from_memory(key)
        
        # Try Redis first
        try:
            cached = await state.redis_client.get(key)
            if cached:
                self.cache_stats['redis_hits'] += 1
                self.cache_stats['hits'] += 1
                # Use orjson for ultra-fast deserialization
                return orjson.loads(cached)
        except Exception as e:
            logger.warning("Redis cache retrieval failed", error=str(e))
        
        # Fallback to memory cache
        result = self._get_from_memory(key)
        if result:
            self.cache_stats['memory_hits'] += 1
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
        
        return result
    
    def _get_from_memory(self, key: str) -> dict | None:
        """Get from in-memory cache."""
        if key in self.memory_cache:
            # Check TTL
            if key in self.memory_ttl and time.time() > self.memory_ttl[key]:
                del self.memory_cache[key]
                del self.memory_ttl[key]
                return None
            return self.memory_cache[key]
        return None
    
    async def set(self, key: str, data: dict, ttl: int = 3600, cache_type: str = 'auto') -> None:
        """Set cached result with intelligent storage."""
        # Always store in memory for fast access
        self.memory_cache[key] = data
        self.memory_ttl[key] = time.time() + ttl
        
        # Store in Redis if available
        if cache_type != 'memory' and state.redis_client:
            try:
                # Use orjson for ultra-fast serialization
                serialized_data = orjson.dumps(data)
                await state.redis_client.setex(key, ttl, serialized_data)
            except Exception as e:
                logger.warning("Redis cache storage failed", error=str(e))
    
    async def delete(self, key: str) -> None:
        """Delete cached result from both stores."""
        # Remove from memory
        self.memory_cache.pop(key, None)
        self.memory_ttl.pop(key, None)
        
        # Remove from Redis
        if state.redis_client:
            try:
                await state.redis_client.delete(key)
            except Exception as e:
                logger.warning("Redis cache deletion failed", error=str(e))
    
    async def clear(self, pattern: str = None) -> None:
        """Clear cache entries."""
        if pattern:
            # Clear matching keys from memory
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                self.memory_cache.pop(key, None)
                self.memory_ttl.pop(key, None)
            
            # Clear matching keys from Redis
            if state.redis_client:
                try:
                    keys = await state.redis_client.keys(pattern)
                    if keys:
                        await state.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning("Redis cache clear failed", error=str(e))
        else:
            # Clear all
            self.memory_cache.clear()
            self.memory_ttl.clear()
            if state.redis_client:
                try:
                    await state.redis_client.flushdb()
                except Exception as e:
                    logger.warning("Redis cache flush failed", error=str(e))
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            **self.cache_stats,
            'memory_size': len(self.memory_cache),
            'memory_keys': list(self.memory_cache.keys())[:10]  # First 10 keys
        }

# Global cache manager instance
cache_manager = CacheManager()

# Cache functions with advanced features
async def get_cached_result(key: str) -> dict | None:
    """Get cached result with intelligent fallback."""
    return await cache_manager.get(key)

async def set_cached_result(key: str, data: dict, ttl: int = 3600) -> None:
    """Set cached result with intelligent storage."""
    await cache_manager.set(key, data, ttl)

async def delete_cached_result(key: str) -> None:
    """Delete cached result."""
    await cache_manager.delete(key)

async def clear_cache(pattern: str = None) -> None:
    """Clear cache entries."""
    await cache_manager.clear(pattern)

def generate_cache_key(params: SEOParamsModel) -> str:
    """Generate cache key for URL and parameters with optimized serialization."""
    url = params.url
    other_params = params.model_dump(exclude={'url'}, mode='json')
    key_data = {
        'url': url,
        'params': sorted(other_params.items())
    }
    # Use orjson for ultra-fast serialization
    return f"seo_analysis:{hashlib.sha256(orjson.dumps(key_data)).hexdigest()}"

# Lazy Loading Utilities and Classes
class LazyDataLoader:
    """Lazy data loader for large datasets with chunking and streaming."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.chunk_cache = {}
        self.chunk_ttl = {}
    
    async def load_data_chunk(self, data_source: List[Any], chunk_index: int) -> List[Any]:
        """Load a specific chunk of data."""
        start_idx = chunk_index * self.config.chunk_size
        end_idx = min(start_idx + self.config.chunk_size, len(data_source))
        
        if start_idx >= len(data_source):
            return []
        
        return data_source[start_idx:end_idx]
    
    async def stream_data(self, data_source: List[Any]) -> AsyncGenerator[List[Any], None]:
        """Stream data in chunks."""
        total_chunks = (len(data_source) + self.config.chunk_size - 1) // self.config.chunk_size
        
        for chunk_index in range(total_chunks):
            chunk = await self.load_data_chunk(data_source, chunk_index)
            if chunk:
                yield chunk
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.001)
    
    async def get_paginated_data(self, data_source: List[Any], pagination: PaginationParams) -> LazyLoadResult:
        """Get paginated data with lazy loading."""
        start_idx = (pagination.page - 1) * pagination.page_size
        end_idx = min(start_idx + pagination.page_size, len(data_source))
        
        # Apply sorting if specified
        if pagination.sort_by:
            data_source = sorted(
                data_source,
                key=lambda x: getattr(x, pagination.sort_by, 0),
                reverse=(pagination.sort_order == "desc")
            )
        
        data = data_source[start_idx:end_idx] if start_idx < len(data_source) else []
        
        total_pages = (len(data_source) + pagination.page_size - 1) // pagination.page_size
        
        return LazyLoadResult(
            data=data,
            total_count=len(data_source),
            page=pagination.page,
            page_size=pagination.page_size,
            has_next=pagination.page < total_pages,
            has_previous=pagination.page > 1,
            total_pages=total_pages
        )

class LazyResponseGenerator:
    """Generate lazy responses for large datasets."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
    
    async def generate_streaming_response(self, data: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """Generate streaming JSON response."""
        yield "{\n"
        yield '"data": [\n'
        
        for i, item in enumerate(data):
            json_item = orjson.dumps(item).decode('utf-8')
            if i < len(data) - 1:
                yield f"{json_item},\n"
            else:
                yield f"{json_item}\n"
            
            # Yield control to event loop
            if i % 100 == 0:
                await asyncio.sleep(0.001)
        
        yield "],\n"
        yield f'"total_count": {len(data)},\n'
        yield f'"chunk_size": {self.config.chunk_size}\n'
        yield "}\n"
    
    async def generate_compressed_response(self, data: List[Dict[str, Any]]) -> bytes:
        """Generate compressed response for large datasets."""
        json_data = orjson.dumps(data)
        
        if len(json_data) > self.config.compression_threshold:
            return gzip.compress(json_data)
        else:
            return json_data

class BulkSEOProcessor:
    """Process bulk SEO analysis with lazy loading."""
    
    def __init__(self, config: LazyLoadingConfig):
        
    """__init__ function."""
self.config = config
        self.data_loader = LazyDataLoader(config)
        self.response_generator = LazyResponseGenerator(config)
    
    async def process_bulk_analysis(self, params: BulkSEOParams) -> AsyncGenerator[BulkSEOResult, None]:
        """Process bulk SEO analysis with lazy loading."""
        start_time = time.time()
        total_urls = len(params.urls)
        successful_analyses = 0
        failed_analyses = 0
        results = []
        
        # Process URLs in chunks
        async for url_chunk in self.data_loader.stream_data(params.urls):
            chunk_results = []
            
            # Process chunk in parallel
            tasks = []
            for url in url_chunk:
                task = self._analyze_single_url(url, params)
                tasks.append(task)
            
            # Execute tasks with concurrency limit
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in chunk_results:
                if isinstance(result, Exception):
                    failed_analyses += 1
                    logger.error("Bulk analysis failed", error=str(result))
                else:
                    successful_analyses += 1
                    results.append(result)
            
            # Yield intermediate results
            if params.lazy_loading.enable_streaming:
                yield BulkSEOResult(
                    results=results[-len(chunk_results):],  # Only new results
                    total_urls=total_urls,
                    successful_analyses=successful_analyses,
                    failed_analyses=failed_analyses,
                    average_score=sum(r.score for r in results) / len(results) if results else 0.0,
                    processing_time=time.time() - start_time,
                    lazy_loading=params.lazy_loading,
                    pagination=params.pagination
                )
        
        # Final result
        final_result = BulkSEOResult(
            results=results,
            total_urls=total_urls,
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            average_score=sum(r.score for r in results) / len(results) if results else 0.0,
            processing_time=time.time() - start_time,
            lazy_loading=params.lazy_loading,
            pagination=params.pagination
        )
        
        yield final_result
    
    async def _analyze_single_url(self, url: str, params: BulkSEOParams) -> SEOResultModel:
        """Analyze a single URL for bulk processing."""
        try:
            seo_params = SEOParamsModel(
                url=url,
                keywords=params.keywords,
                depth=params.depth,
                include_meta=params.include_meta,
                include_links=params.include_links,
                include_images=params.include_images,
                include_performance=params.include_performance,
                cache_ttl=3600
            )
            
            return await analyze_seo(seo_params)
        except Exception as e:
            logger.error("Single URL analysis failed", url=url, error=str(e))
            raise LazyLoadingError(f"Analysis failed for {url}: {str(e)}")

# SEO Service Class with Dependency Injection
class SEOService:
    """SEO analysis service with dependency injection."""
    
    def __init__(
        self,
        cache_manager: CacheManager,
        static_cache: StaticDataCache,
        http_client: httpx.AsyncClient,
        logger: structlog.BoundLogger
    ):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.static_cache = static_cache
        self.http_client = http_client
        self.logger = logger
    
    async def analyze_seo(self, params: SEOParamsModel) -> SEOResultModel:
        """Analyze SEO with dependency injection."""
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = generate_cache_key(params)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                self.logger.info("Cache hit", url=params.url)
                return SEOResultModel(**cached_result)
            
            # Perform analysis
            self.logger.info("Starting SEO analysis", url=params.url)
            
            # Parallel execution for I/O-bound tasks
            crawl_task = self._crawl_url(params)
            performance_task = self._analyze_performance(params)
            
            crawl_result, performance_result = await asyncio.gather(
                crawl_task, performance_task, return_exceptions=True
            )
            
            # Handle exceptions from parallel tasks
            if isinstance(crawl_result, Exception):
                self.logger.error("Crawl failed", url=params.url, error=str(crawl_result))
                crawl_result = CrawlResultModel(
                    url=params.url, 
                    status_code=500, 
                    error=str(crawl_result)
                )
            
            if isinstance(performance_result, Exception):
                self.logger.error("Performance analysis failed", url=params.url, error=str(performance_result))
                performance_result = PerformanceResultModel(error=str(performance_result))
            
            # Analyze content
            analysis_params = AnalysisParamsModel(
                crawl_data=crawl_result,
                keywords=params.keywords,
                include_meta=params.include_meta,
                include_links=params.include_links,
                include_images=params.include_images
            )
            content_analysis = await self._analyze_seo_content(analysis_params)
            
            # Calculate score
            metrics = {
                'title': content_analysis.title_score,
                'description': content_analysis.description_score,
                'headings': content_analysis.headings_score,
                'links': content_analysis.links_score,
                'images': content_analysis.images_score,
                'performance': 100.0 if (performance_result.load_time and performance_result.load_time < 3.0) else 50.0
            }
            
            overall_score = calculate_seo_score(metrics)
            
            # Build result
            result = SEOResultModel(
                url=params.url,
                score=overall_score,
                title=crawl_result.title,
                description=crawl_result.meta_tags.get('description'),
                keywords=crawl_result.meta_tags.get('keywords', '').split(',') if crawl_result.meta_tags.get('keywords') else [],
                headings=crawl_result.headings,
                links=crawl_result.links,
                images=crawl_result.images,
                meta_tags=crawl_result.meta_tags,
                performance=performance_result,
                warnings=content_analysis.warnings,
                suggestions=content_analysis.suggestions,
                errors=content_analysis.errors,
                timestamp=time.time()
            )
            
            # Cache result
            await self.cache_manager.set(cache_key, result.model_dump(mode='json'), params.cache_ttl)
            
            # Log performance metrics
            duration = time.perf_counter() - start_time
            self.logger.info(
                "SEO analysis completed",
                url=params.url,
                score=overall_score,
                duration=duration
            )
            
            return result
        
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error(
                "SEO analysis failed",
                url=params.url,
                error=str(e),
                duration=duration
            )
            return SEOResultModel(
                url=params.url,
                score=0.0,
                errors=[str(e)],
                timestamp=time.time()
            )
    
    async def _crawl_url(self, params: SEOParamsModel) -> CrawlResultModel:
        """Crawl URL with dependency injection."""
        crawl_params = CrawlParamsModel(
            url=params.url,
            depth=params.depth,
            timeout=30,
            max_retries=3,
            user_agent=self.static_cache.get_user_agent()
        )
        return await crawl_url(crawl_params)
    
    async def _analyze_performance(self, params: SEOParamsModel) -> PerformanceResultModel:
        """Analyze performance with dependency injection."""
        performance_params = PerformanceParamsModel(
            url=params.url,
            timeout=30,
            follow_redirects=True
        )
        return await analyze_performance(performance_params)
    
    async def _analyze_seo_content(self, params: AnalysisParamsModel) -> AnalysisResultModel:
        """Analyze SEO content with dependency injection."""
        return await analyze_seo_content(params)
    
    async def bulk_analyze(self, params: BulkSEOParams) -> AsyncGenerator[BulkSEOResult, None]:
        """Bulk analyze URLs with dependency injection."""
        bulk_processor = BulkSEOProcessor(LazyLoadingConfig())
        async for result in bulk_processor.process_bulk_analysis(params):
            yield result

# Global lazy loading instances (legacy)
lazy_loader = LazyDataLoader(LazyLoadingConfig())
bulk_processor = BulkSEOProcessor(LazyLoadingConfig())

# Static data caching
class StaticDataCache:
    """Cache for static and frequently accessed data."""
    
    def __init__(self) -> Any:
        self.seo_rules = {}
        self.keyword_scores = {}
        self.user_agents = [
            'Mozilla/5.0 (compatible; SEO-Bot/1.0)',
            'Mozilla/5.0 (compatible; Googlebot/2.1)',
            'Mozilla/5.0 (compatible; Bingbot/2.0)'
        ]
        self.common_keywords = [
            'seo', 'optimization', 'search', 'google', 'ranking',
            'keywords', 'meta', 'title', 'description', 'headings'
        ]
    
    async def get_seo_rules(self) -> dict:
        """Get SEO rules with caching."""
        if not self.seo_rules:
            self.seo_rules = {
                'title_length': {'min': 30, 'max': 60, 'optimal': 50},
                'description_length': {'min': 120, 'max': 160, 'optimal': 140},
                'h1_count': {'min': 1, 'max': 1, 'optimal': 1},
                'link_density': {'min': 0.1, 'max': 0.3, 'optimal': 0.2},
                'image_alt_ratio': {'min': 0.8, 'max': 1.0, 'optimal': 1.0},
                'load_time': {'min': 0, 'max': 3.0, 'optimal': 1.0}
            }
        return self.seo_rules
    
    async def get_keyword_score(self, keyword: str) -> float:
        """Get keyword importance score with caching."""
        if keyword not in self.keyword_scores:
            # Calculate keyword score based on common SEO factors
            score = 1.0
            if keyword.lower() in self.common_keywords:
                score = 0.8
            elif len(keyword) > 20:
                score = 0.6
            elif len(keyword) < 3:
                score = 0.4
            self.keyword_scores[keyword] = score
        return self.keyword_scores[keyword]
    
    def get_user_agent(self) -> str:
        """Get random user agent."""
        return random.choice(self.user_agents)
    
    async def preload_static_data(self) -> Any:
        """Preload static data into cache."""
        await self.get_seo_rules()
        # Preload common keywords
        for keyword in self.common_keywords:
            await self.get_keyword_score(keyword)

# Global static data cache instance
static_cache = StaticDataCache()

# Async Rate limiting with Redis backend
class RateLimiter:
    """Async rate limiter implementation with Redis backend."""
    
    def __init__(self, max_requests: int, window: int = 60):
        
    """__init__ function."""
self.max_requests = max_requests
        self.window = window
    
    async def check_rate_limit(self, client_id: str, max_requests: int | None = None, window: int | None = None) -> dict:
        """Check if request is allowed with async Redis operations."""
        if not state.redis_client:
            # Fallback to in-memory if Redis unavailable
            return await self._check_rate_limit_memory(client_id, max_requests, window)
        
        max_requests = max_requests or self.max_requests
        window = window or self.window
        
        try:
            # Use Redis for distributed rate limiting
            key = f"rate_limit:{client_id}"
            now = time.time()
            
            # Use Redis pipeline for atomic operations
            async with state.redis_client.pipeline() as pipe:
                # Remove old entries and get current count
                await pipe.zremrangebyscore(key, 0, now - window)
                await pipe.zcard(key)
                await pipe.zadd(key, {str(now): now})
                await pipe.expire(key, window)
                results = await pipe.execute()
            
            current_count = results[1] + 1  # +1 for current request
            
            if current_count > max_requests:
                return {
                    'allowed': False,
                    'remaining': 0,
                    'reset_time': now + window
                }
            
            return {
                'allowed': True,
                'remaining': max_requests - current_count,
                'reset_time': now + window
            }
            
        except Exception as e:
            logger.warning("Redis rate limiting failed, falling back to memory", error=str(e))
            return await self._check_rate_limit_memory(client_id, max_requests, window)
    
    async def _check_rate_limit_memory(self, client_id: str, max_requests: int | None = None, window: int | None = None) -> dict:
        """Fallback in-memory rate limiting."""
        max_requests = max_requests or self.max_requests
        window = window or self.window
        
        now = time.time()
        
        # Use thread-safe operations for in-memory storage
        if not hasattr(self, '_requests'):
            self._requests = collections.defaultdict(list)
        
        client_requests = self._requests[client_id]
        
        # Remove old requests
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < window]
        
        remaining = max(0, max_requests - len(client_requests))
        reset_time = now + window
        
        if len(client_requests) >= max_requests:
            return {
                'allowed': False,
                'remaining': remaining,
                'reset_time': reset_time
            }
        
        client_requests.append(now)
        return {
            'allowed': True,
            'remaining': remaining - 1,
            'reset_time': reset_time
        }

rate_limiter = RateLimiter(config.rate_limit)

# Performance-optimized SEO Analysis functions with caching
async def crawl_url(params: CrawlParamsModel) -> CrawlResultModel:
    """Crawl URL and extract content with non-blocking optimizations."""
    if not params.url: return CrawlResultModel(url="", error="URL required")
    
    # Check cache for crawl results using optimized hash
    crawl_cache_key = f"crawl:{optimized_functions['fast_hash'](params.url)}"
    cached_crawl = await get_cached_result(crawl_cache_key)
    if cached_crawl:
        logger.info("Returning cached crawl result", url=params.url)
        return CrawlResultModel(**cached_crawl)
    
    # Use connection pool for HTTP requests
    http_pool = await connection_pool_manager.get_http_pool()
    
    try:
        # Use cached user agent or generate random one
        user_agent = params.user_agent or static_cache.get_user_agent()
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }
        
        # Non-blocking HTTP request with connection pooling
        response = await http_pool.get(params.url, headers=headers, timeout=params.timeout)
        response.raise_for_status()
        
        # Parse HTML content in thread pool to avoid blocking
        def parse_html_content(content, url) -> Any:
            soup = BeautifulSoup(content, 'lxml')
            
            # Optimized extraction with list comprehensions
            title = soup.find('title')
            title_text = title.get_text().strip() if title else None
            
            # Batch meta tag extraction
            meta_tags = {
                meta.get('name') or meta.get('property'): meta.get('content')
                for meta in soup.find_all('meta')
                if meta.get('name') or meta.get('property')
            }
            
            # Optimized headings extraction
            headings = {
                f'h{i}': [h.get_text().strip() for h in soup.find_all(f'h{i}')]
                for i in range(1, 7)
            }
            
            # Optimized links extraction with filtering
            links = [
                {
                    'url': urllib.parse.urljoin(url, link.get('href')),
                    'text': link.get_text().strip(),
                    'title': link.get('title', '')
                }
                for link in soup.find_all('a', href=True)
                if link.get_text().strip()
            ]
            
            # Optimized images extraction
            images = [
                {
                    'src': urllib.parse.urljoin(url, img.get('src')),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                }
                for img in soup.find_all('img', src=True)
            ]
            
            return {
                'title': title_text,
                'meta_tags': meta_tags,
                'headings': headings,
                'links': links,
                'images': images
            }
        
        # Parse HTML in thread pool
        parsed_content = await non_blocking_manager.run_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            parse_html_content, response.content, params.url
        )
        
        result = CrawlResultModel(
            url=params.url,
            title=parsed_content['title'],
            meta_tags=parsed_content['meta_tags'],
            headings=parsed_content['headings'],
            links=parsed_content['links'],
            images=parsed_content['images'],
            content_length=len(response.content),
            status_code=response.status_code,
            error=None
        )
        
        # Cache the crawl result in background
        await non_blocking_manager.add_background_task(
            set_cached_result, crawl_cache_key, result.model_dump(mode='json'), 1800
        )
        
        return result
    
    except Exception as e:
        logger.error("Crawl failed", url=params.url, error=str(e))
        return CrawlResultModel(
            url=params.url,
            title=None,
            meta_tags={},
            headings={},
            links=[],
            images=[],
            content_length=0,
            status_code=None,
            error=str(e)
        )

async def analyze_seo_content(params: AnalysisParamsModel) -> AnalysisResultModel:
    """Analyze SEO content with non-blocking optimizations."""
    if not params.crawl_data: return AnalysisResultModel(errors=["No crawl data provided"])
    
    # Check cache for analysis results using optimized hash
    analysis_cache_key = f"analysis:{optimized_functions['fast_hash'](f'{params.crawl_data.url}:{str(params.keywords)}')}"
    cached_analysis = await get_cached_result(analysis_cache_key)
    if cached_analysis:
        logger.info("Returning cached analysis result", url=params.crawl_data.url)
        return AnalysisResultModel(**cached_analysis)
    
    # Get cached SEO rules
    seo_rules = await static_cache.get_seo_rules()
    
    # Pre-compute values for performance
    title = params.crawl_data.title
    description = params.crawl_data.meta_tags.get('description')
    headings = params.crawl_data.headings
    links = params.crawl_data.links
    images = params.crawl_data.images
    
    # Initialize with optimized defaults
    analysis = AnalysisResultModel(
        title_score=0.0,
        description_score=0.0,
        headings_score=0.0,
        keywords_score=0.0,
        links_score=0.0,
        images_score=0.0,
        suggestions=[],
        warnings=[],
        errors=[]
    )
    
    # Run SEO analysis in thread pool to avoid blocking
    def perform_seo_analysis(seo_rules, title, description, headings, links, images, include_links, include_images, url) -> Any:
        # Optimized title analysis with cached rules
        title_rules = seo_rules['title_length']
        title_score = 0.0
        title_warnings = []
        title_suggestions = []
        
        if not title:
            title_warnings.append("Missing title tag")
        elif len(title) < title_rules['min']:
            title_score = 50.0
            title_suggestions.append("Title is too short")
        elif len(title) > title_rules['max']:
            title_score = 70.0
            title_warnings.append("Title is too long")
        else:
            title_score = 100.0
        
        # Optimized description analysis with cached rules
        desc_rules = seo_rules['description_length']
        desc_score = 0.0
        desc_warnings = []
        desc_suggestions = []
        
        if not description:
            desc_warnings.append("Missing meta description")
        elif len(description) < desc_rules['min']:
            desc_score = 60.0
            desc_suggestions.append("Description is too short")
        elif len(description) > desc_rules['max']:
            desc_score = 80.0
            desc_warnings.append("Description is too long")
        else:
            desc_score = 100.0
        
        # Optimized headings analysis with cached rules
        h1_rules = seo_rules['h1_count']
        h1_tags = headings.get('h1', [])
        headings_score = 0.0
        headings_warnings = []
        
        if not h1_tags:
            headings_warnings.append("Missing H1 tag")
        elif len(h1_tags) > h1_rules['max']:
            headings_score = 70.0
            headings_warnings.append("Multiple H1 tags found")
        else:
            headings_score = 100.0
        
        # Optimized links analysis with caching
        links_score = 0.0
        links_suggestions = []
        if include_links and links:
            # Cache URL parsing for performance
            base_domain = optimized_functions['fast_url_parse'](url).netloc
            
            # Use list comprehension for better performance
            internal_links = [
                link for link in links 
                if optimized_functions['fast_url_parse'](link['url']).netloc == base_domain
            ]
            
            link_count = len(links)
            if link_count >= 10:
                links_score = 100.0
            elif link_count >= 5:
                links_score = 80.0
            else:
                links_score = 40.0
                links_suggestions.append("Add more internal links")
        
        # Optimized images analysis with cached rules
        images_score = 0.0
        images_suggestions = []
        if include_images and images:
            image_rules = seo_rules['image_alt_ratio']
            # Use sum() for better performance than len() of filtered list
            images_with_alt = sum(1 for img in images if img.get('alt'))
            alt_ratio = images_with_alt / len(images)
            images_score = alt_ratio * 100.0
            
            if alt_ratio < image_rules['min']:
                images_suggestions.append("Add alt text to images")
        
        return {
            'title_score': title_score,
            'description_score': desc_score,
            'headings_score': headings_score,
            'links_score': links_score,
            'images_score': images_score,
            'warnings': title_warnings + desc_warnings + headings_warnings,
            'suggestions': title_suggestions + desc_suggestions + links_suggestions + images_suggestions
        }
    
    # Run analysis in thread pool
    analysis_result = await non_blocking_manager.run_in_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        perform_seo_analysis,
        seo_rules, title, description, headings, links, images,
        params.include_links, params.include_images, params.crawl_data.url
    )
    
    # Update analysis object
    analysis.title_score = analysis_result['title_score']
    analysis.description_score = analysis_result['description_score']
    analysis.headings_score = analysis_result['headings_score']
    analysis.links_score = analysis_result['links_score']
    analysis.images_score = analysis_result['images_score']
    analysis.warnings = analysis_result['warnings']
    analysis.suggestions = analysis_result['suggestions']
    
    # Cache the analysis result in background
    await non_blocking_manager.add_background_task(
        set_cached_result, analysis_cache_key, analysis.model_dump(mode='json'), 3600
    )
    
    return analysis

async def analyze_performance(params: PerformanceParamsModel) -> PerformanceResultModel:
    """Analyze page performance with non-blocking optimizations."""
    if not params.url: return PerformanceResultModel(error="URL required")
    
    # Use connection pool for HTTP requests
    http_pool = await connection_pool_manager.get_http_pool()
    
    try:
        # High-precision timing with non-blocking request
        start_time = time.perf_counter()
        response = await http_pool.get(
            params.url, 
            timeout=params.timeout,
            follow_redirects=params.follow_redirects
        )
        load_time = time.perf_counter() - start_time
        
        # Extract headers in thread pool to avoid blocking
        def extract_headers(response_headers) -> Any:
            return dict(response_headers)
        
        headers = await non_blocking_manager.run_in_thread(extract_headers, response.headers)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return PerformanceResultModel(
            load_time=load_time,
            status_code=response.status_code,
            content_length=len(response.content),
            headers=headers,
            error=None
        )
    except Exception as e:
        logger.error("Performance analysis failed", url=params.url, error=str(e))
        return PerformanceResultModel(
            load_time=None,
            status_code=None,
            content_length=0,
            headers={},
            error=str(e)
        )

# Non-blocking Main SEO analysis function
async def analyze_seo(params: SEOParamsModel) -> SEOResultModel:
    """Main SEO analysis function with dedicated async operations."""
    if not params.url: return SEOResultModel(url="", errors=["URL required"])
    
    start_time = time.perf_counter()
    
    try:
        # Early validation with guard clause
        if not is_valid_url(params.url):
            return SEOResultModel(
                url=params.url,
                score=0.0,
                errors=[f"Invalid URL: {params.url}"],
                timestamp=time.time()
            )
        
        # Get dedicated async operations from dependency container
        async_db_ops = container.async_db_operations
        async_api_ops = container.async_api_operations
        async_persistence_ops = container.async_persistence_operations
        
        # Check database for existing result first
        existing_result = await async_db_ops.retrieve_seo_result(params.url)
        if existing_result:
            logger.info("Returning database result", url=params.url)
            return existing_result
        
        # Optimized cache check
        cache_key = generate_cache_key(params)
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            logger.info("Returning cached result", url=params.url)
            return SEOResultModel(**cached_result)
        
        # Use dedicated async API operations for content fetching
        content_data = await async_api_ops.fetch_page_content(params.url, config.timeout)
        if not content_data.get('success', False):
            return SEOResultModel(
                url=params.url,
                score=0.0,
                errors=[content_data.get('error', 'Failed to fetch content')],
                timestamp=time.time()
            )
        
        # Parallel execution of independent operations using dedicated async functions
        crawl_params = CrawlParamsModel(
            url=params.url,
            depth=params.depth,
            timeout=config.timeout,
            max_retries=3,
            user_agent='SEO-Bot/1.0'
        )
        
        # Execute crawl and performance analysis in parallel if both are needed
        tasks = [crawl_url(crawl_params)]
        
        if params.include_performance:
            perf_params = PerformanceParamsModel(
                url=params.url,
                timeout=config.timeout,
                follow_redirects=True
            )
            tasks.append(analyze_performance(perf_params))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        crawl_data = results[0]
        
        # Handle crawl errors early
        if isinstance(crawl_data, Exception) or crawl_data.error:
            error_msg = str(crawl_data) if isinstance(crawl_data, Exception) else crawl_data.error
            return SEOResultModel(
                url=params.url,
                score=0.0,
                errors=[error_msg],
                timestamp=time.time()
            )
        
        # Get performance data
        performance_data = PerformanceResultModel(error='Performance analysis disabled')
        if params.include_performance and len(results) > 1:
            perf_result = results[1]
            if not isinstance(perf_result, Exception):
                performance_data = perf_result
        
        # Optimized content analysis
        analysis_params = AnalysisParamsModel(
            crawl_data=crawl_data,
            keywords=params.keywords,
            include_meta=params.include_meta,
            include_links=params.include_links,
            include_images=params.include_images
        )
        content_analysis = await analyze_seo_content(analysis_params)
        
        # Optimized score calculation
        metrics = {
            'title': content_analysis.title_score,
            'description': content_analysis.description_score,
            'headings': content_analysis.headings_score,
            'links': content_analysis.links_score,
            'images': content_analysis.images_score,
            'performance': 100.0 if (performance_data.load_time and performance_data.load_time < 3.0) else 50.0
        }
        
        overall_score = calculate_seo_score(metrics)
        
        # Optimized response building
        result = SEOResultModel(
            url=params.url,
            score=overall_score,
            title=crawl_data.title,
            description=crawl_data.meta_tags.get('description'),
            keywords=crawl_data.meta_tags.get('keywords', '').split(',') if crawl_data.meta_tags.get('keywords') else [],
            headings=crawl_data.headings,
            links=crawl_data.links,
            images=crawl_data.images,
            meta_tags=crawl_data.meta_tags,
            performance=performance_data,
            warnings=content_analysis.warnings,
            suggestions=content_analysis.suggestions,
            errors=content_analysis.errors,
            timestamp=time.time()
        )
        
        # Use dedicated async persistence operations for data storage
        async def persist_data():
            
    """persist_data function."""
try:
                # Store in database
                await async_db_ops.store_seo_result(result)
                # Store in cache
                await set_cached_result(cache_key, result.dict())
                # Persist with full persistence operations
                await async_persistence_ops.persist_seo_analysis(result, params.cache_ttl)
            except Exception as e:
                logger.error("Failed to persist SEO result", error=str(e))
        
        # Async persistence operation (don't wait for it)
        asyncio.create_task(persist_data())
        
        # Optimized metrics logging
        duration = time.perf_counter() - start_time
        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(method='POST', endpoint='/analyze', status='200').inc()
        
        logger.info("SEO analysis completed", 
                   url=params.url, 
                   score=overall_score, 
                   duration=duration)
        
        return result
    
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/analyze', status='500').inc()
        logger.error("SEO analysis failed", url=params.url, error=str(e))
        return SEOResultModel(
            url=params.url,
            score=0.0,
            errors=[str(e)],
            timestamp=time.time()
        )

# FastAPI application
app = FastAPI(
    title="Ultra-Optimized SEO Service v15",
    description="High-performance SEO analysis service with advanced caching and monitoring",
    version="15.0.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    json_encoder=orjson.JSONEncoder,
    json_encoders={
        datetime: lambda v: v.isoformat(),
        BaseModel: lambda v: v.model_dump(mode='json')
    }
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# FastAPI Dependency Injection Functions
async def get_dependency_container() -> DependencyContainer:
    """Get dependency container instance."""
    return container

async def get_config() -> Config:
    """Get application configuration."""
    return container.config

async def get_logger() -> structlog.BoundLogger:
    """Get structured logger instance."""
    return container.logger

async def get_cache_manager() -> CacheManager:
    """Get cache manager instance."""
    return container.cache_manager

async def get_static_cache() -> StaticDataCache:
    """Get static cache instance."""
    return container.static_cache

async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    return container.rate_limiter

async def get_lazy_loader() -> LazyDataLoader:
    """Get lazy loader instance."""
    return container.lazy_loader

async def get_bulk_processor() -> BulkSEOProcessor:
    """Get bulk processor instance."""
    return container.bulk_processor

async def get_async_db_operations() -> AsyncDatabaseOperations:
    """Get async database operations instance."""
    return container.async_db_operations

async async def get_async_api_operations() -> AsyncExternalAPIOperations:
    """Get async external API operations instance."""
    return container.async_api_operations

async def get_async_persistence_operations() -> AsyncDataPersistenceOperations:
    """Get async data persistence operations instance."""
    return container.async_persistence_operations

async def get_redis() -> Optional[redis.Redis]:
    """Get Redis client with async connection pooling."""
    if container._redis_client is None:
        try:
            container._redis_client = redis.from_url(
                container.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            await container._redis_client.ping()
            container.logger.info("Redis connection established")
        except Exception as e:
            container.logger.warning("Redis connection failed", error=str(e))
            container._redis_client = None
    return container._redis_client

async def get_mongo() -> Optional[AsyncIOMotorClient]:
    """Get MongoDB client with async connection pooling."""
    if container._mongo_client is None:
        try:
            container._mongo_client = AsyncIOMotorClient(
                container.config.mongo_url,
                maxPoolSize=50,
                minPoolSize=10,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            # Test connection
            await container._mongo_client.admin.command('ping')
            container.logger.info("MongoDB connection established")
        except Exception as e:
            container.logger.warning("MongoDB connection failed", error=str(e))
            container._mongo_client = None
    return container._mongo_client

async async def get_http_client() -> httpx.AsyncClient:
    """Get HTTP client with async connection pooling."""
    if container._http_client is None:
        container._http_client = httpx.AsyncClient(
            timeout=container.config.timeout,
            limits=httpx.Limits(
                max_connections=container.config.max_connections,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
            http2=True,
            follow_redirects=True
        )
        container.logger.info("HTTP client initialized")
    return container._http_client

async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    config: Config = Depends(get_config),
    logger: structlog.BoundLogger = Depends(get_logger)
) -> None:
    """Check rate limit for request with dependency injection."""
    client_id = request.client.host if request.client else 'unknown'
    
    try:
        rate_limit_result = await rate_limiter.check_rate_limit(
            client_id, config.rate_limit, 60
        )
        
        if not rate_limit_result['allowed']:
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Try again in {int(rate_limit_result['reset_time'] - time.time())} seconds"
            )
    except Exception as e:
        logger.warning("Rate limiting check failed", error=str(e))
        # Allow request to proceed if rate limiting fails

async def get_seo_service(
    cache_manager: CacheManager = Depends(get_cache_manager),
    static_cache: StaticDataCache = Depends(get_static_cache),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    logger: structlog.BoundLogger = Depends(get_logger)
) -> 'SEOService':
    """Get SEO service instance with all dependencies."""
    return SEOService(
        cache_manager=cache_manager,
        static_cache=static_cache,
        http_client=http_client,
        logger=logger
    )

async def get_performance_manager() -> PerformanceMetricsManager:
    """Get performance metrics manager instance."""
    return performance_manager

# Performance Monitoring Middleware
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Middleware for comprehensive performance monitoring."""
    start_time = time.time()
    request_start_time = start_time
    
    # Get request size
    request_size_bytes = 0
    if request.body():
        try:
            body = await request.body()
            request_size_bytes = len(body)
        except Exception:
            request_size_bytes = 0
    
    # Get client information
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    
    # Track cache hit status
    is_cache_hit = False
    is_error = False
    is_timeout = False
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Get response size
        response_size_bytes = 0
        if hasattr(response, 'body'):
            try:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                response_size_bytes = len(response_body)
                # Recreate response with body
                response = Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
            except Exception:
                response_size_bytes = 0
        
        # Determine if it's an error
        is_error = response.status_code >= 400
        
        # Create API metrics
        api_metrics = APIMetricsModel(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            request_start_time=request_start_time,
            request_end_time=end_time,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            is_cache_hit=is_cache_hit,
            is_error=is_error,
            is_timeout=is_timeout,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Record metrics
        performance_manager.record_request(api_metrics)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
        response.headers["X-Request-ID"] = str(hash(f"{client_ip}:{start_time}"))
        
        return response
        
    except Exception as e:
        # Handle exceptions
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Create error metrics
        api_metrics = APIMetricsModel(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=500,
            request_start_time=request_start_time,
            request_end_time=end_time,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=0,
            is_cache_hit=is_cache_hit,
            is_error=True,
            is_timeout=is_timeout,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Record error metrics
        performance_manager.record_request(api_metrics)
        
        # Re-raise the exception
        raise

async def log_metrics(url: str, score: float, logger: structlog.BoundLogger) -> None:
    """Log metrics for SEO analysis."""
    logger.info(
        "SEO analysis metrics",
        url=url,
        score=score,
        timestamp=time.time()
    )

# Routes
@app.post("/analyze", response_model=SEOResponse)
async def analyze_seo_endpoint(
    request: SEORequest,
    background_tasks: BackgroundTasks,
    seo_service: SEOService = Depends(get_seo_service),
    rate_limit: None = Depends(check_rate_limit),
    container: DependencyContainer = Depends(get_dependency_container)
):
    """Analyze SEO for given URL with non-blocking optimizations."""
    try:
        # Increment request counter
        container.increment_request_count()
        
        # Use model_dump for optimized serialization
        params = SEOParamsModel(**request.model_dump(mode='json'))
        
        # Run SEO analysis with non-blocking optimizations
        result = await seo_service.analyze_seo(params)
        
        # Add background task for metrics using non-blocking manager
        await non_blocking_manager.add_background_task(
            log_metrics, 
            params.url, 
            result.score,
            container.logger
        )
        
        return SEOResponse(**result.model_dump(mode='json'))
        
    except Exception as e:
        container.increment_error_count()
        container.logger.error("Unexpected error in SEO analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze/async", response_model=SEOResponse)
async def analyze_seo_async_endpoint(
    request: SEORequest,
    rate_limit: None = Depends(check_rate_limit),
    container: DependencyContainer = Depends(get_dependency_container)
):
    """Analyze SEO asynchronously with immediate response and background processing."""
    try:
        container.increment_request_count()
        
        # Generate task ID for tracking
        task_id = optimized_functions['fast_hash'](f"{request.url}:{time.time()}")
        
        # Start analysis in background
        analysis_task = asyncio.create_task(
            _perform_async_seo_analysis(request, container)
        )
        
        # Return immediate response with task ID
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "SEO analysis started in background",
            "estimated_completion": time.time() + 30,  # 30 seconds estimate
            "check_status_url": f"/analyze/status/{task_id}"
        }
        
    except Exception as e:
        container.increment_error_count()
        container.logger.error("Error starting async SEO analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Error starting analysis")

@app.get("/analyze/status/{task_id}")
async def get_analysis_status(
    task_id: str,
    container: DependencyContainer = Depends(get_dependency_container)
):
    """Get status of async SEO analysis."""
    try:
        # Check cache for task status
        status = await get_cached_result(f"task_status:{task_id}")
        if status:
            return status
        
        return {
            "task_id": task_id,
            "status": "not_found",
            "message": "Task not found or expired"
        }
        
    except Exception as e:
        container.logger.error("Error getting task status", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving task status")

async def _perform_async_seo_analysis(request: SEORequest, container: DependencyContainer):
    """Perform SEO analysis in background."""
    task_id = optimized_functions['fast_hash'](f"{request.url}:{time.time()}")
    
    try:
        # Update status to processing
        await set_cached_result(f"task_status:{task_id}", {
            "task_id": task_id,
            "status": "processing",
            "progress": 0,
            "started_at": time.time()
        }, ttl=3600)
        
        # Get SEO service
        seo_service = await get_seo_service()
        params = SEOParamsModel(**request.model_dump(mode='json'))
        
        # Update progress
        await set_cached_result(f"task_status:{task_id}", {
            "task_id": task_id,
            "status": "processing",
            "progress": 25,
            "started_at": time.time()
        }, ttl=3600)
        
        # Perform analysis
        result = await seo_service.analyze_seo(params)
        
        # Update progress
        await set_cached_result(f"task_status:{task_id}", {
            "task_id": task_id,
            "status": "processing",
            "progress": 75,
            "started_at": time.time()
        }, ttl=3600)
        
        # Log metrics in background
        await non_blocking_manager.add_background_task(
            log_metrics, 
            params.url, 
            result.score,
            container.logger
        )
        
        # Store final result
        await set_cached_result(f"task_status:{task_id}", {
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "result": result.model_dump(mode='json'),
            "started_at": time.time(),
            "completed_at": time.time()
        }, ttl=3600)
        
    except Exception as e:
        container.increment_error_count()
        container.logger.error("Async SEO analysis failed", error=str(e))
        
        # Store error status
        await set_cached_result(f"task_status:{task_id}", {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "started_at": time.time(),
            "failed_at": time.time()
        }, ttl=3600)

@app.post("/cache/clear")
async def clear_cache_endpoint(
    pattern: str = None,
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Clear cache entries with dependency injection."""
    try:
        await cache_manager.clear(pattern)
        return {"message": f"Cache cleared successfully", "pattern": pattern}
    except Exception as e:
        logger.error("Cache clear failed", error=str(e))
        raise HTTPException(status_code=500, detail="Cache clear failed")

@app.get("/cache/stats")
async def cache_stats_endpoint(
    cache_manager: CacheManager = Depends(get_cache_manager),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Get cache statistics with dependency injection."""
    try:
        return cache_manager.get_stats()
    except Exception as e:
        logger.error("Cache stats failed", error=str(e))
        raise HTTPException(status_code=500, detail="Cache stats failed")

@app.get("/health")
async def health_check(
    container: DependencyContainer = Depends(get_dependency_container),
    redis_client: Optional[redis.Redis] = Depends(get_redis),
    mongo_client: Optional[AsyncIOMotorClient] = Depends(get_mongo)
):
    """Health check endpoint with dependency injection."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": container.uptime,
        "version": "15.0.0",
        "checks": {}
    }
    
    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            health_status["checks"]["redis"] = "healthy"
        except Exception as e:
            health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["checks"]["redis"] = "unavailable"
        health_status["status"] = "degraded"
    
    # Check MongoDB
    if mongo_client:
        try:
            await mongo_client.admin.command('ping')
            health_status["checks"]["mongodb"] = "healthy"
        except Exception as e:
            health_status["checks"]["mongodb"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["checks"]["mongodb"] = "unavailable"
        health_status["status"] = "degraded"
    
    # Check memory usage
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    health_status["checks"]["memory_usage_mb"] = memory_usage
    
    if memory_usage > 1000:  # More than 1GB
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return StreamingResponse(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )

@app.get("/performance/metrics")
async def get_performance_metrics(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """Get comprehensive performance metrics."""
    try:
        current_metrics = performance_manager.get_current_metrics()
        return current_metrics.model_dump(mode='json')
    except Exception as e:
        logger.error("Error getting performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving performance metrics")

@app.get("/performance/endpoints")
async def get_endpoint_metrics(
    endpoint: Optional[str] = None,
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """Get endpoint-specific performance metrics."""
    try:
        endpoint_metrics = performance_manager.get_endpoint_metrics(endpoint)
        return {
            "endpoint": endpoint,
            "metrics": endpoint_metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Error getting endpoint metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving endpoint metrics")

@app.get("/performance/alerts")
async def get_performance_alerts(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """Get current performance alerts."""
    try:
        alerts = performance_manager.get_performance_alerts()
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error("Error getting performance alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving performance alerts")

@app.get("/performance/summary")
async def get_performance_summary(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """Get performance summary with key metrics."""
    try:
        current_metrics = performance_manager.get_current_metrics()
        alerts = performance_manager.get_performance_alerts()
        
        # Calculate performance trends
        response_times = list(performance_manager.response_times)
        recent_response_times = response_times[-100:] if len(response_times) >= 100 else response_times
        
        trend = "stable"
        if len(recent_response_times) >= 10:
            first_half = statistics.mean(recent_response_times[:len(recent_response_times)//2])
            second_half = statistics.mean(recent_response_times[len(recent_response_times)//2:])
            if second_half > first_half * 1.2:
                trend = "degrading"
            elif second_half < first_half * 0.8:
                trend = "improving"
        
        return {
            "summary": {
                "response_time_ms": current_metrics.response_time_ms,
                "requests_per_second": current_metrics.requests_per_second,
                "error_rate": current_metrics.error_rate,
                "cache_hit_rate": current_metrics.cache_hit_rate,
                "cpu_usage_percent": current_metrics.cpu_usage_percent,
                "memory_usage_percent": current_metrics.memory_usage_percent,
                "performance_score": current_metrics.performance_score,
                "trend": trend
            },
            "alerts": {
                "count": len(alerts),
                "critical": len([a for a in alerts if a['level'] == 'critical']),
                "warning": len([a for a in alerts if a['level'] == 'warning'])
            },
            "system": {
                "total_requests": performance_manager.total_requests,
                "uptime_seconds": time.time() - container.startup_time,
                "timestamp": time.time()
            }
        }
    except Exception as e:
        logger.error("Error getting performance summary", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving performance summary")

@app.post("/performance/reset")
async def reset_performance_metrics(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Reset all performance metrics."""
    try:
        performance_manager.reset_metrics()
        logger.info("Performance metrics reset successfully")
        return {"message": "Performance metrics reset successfully", "timestamp": time.time()}
    except Exception as e:
        logger.error("Error resetting performance metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error resetting performance metrics")

@app.get("/performance/thresholds")
async def get_performance_thresholds(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """Get current performance thresholds."""
    try:
        return performance_manager.thresholds.model_dump(mode='json')
    except Exception as e:
        logger.error("Error getting performance thresholds", error=str(e))
        raise HTTPException(status_code=500, detail="Error retrieving performance thresholds")

@app.get("/performance/real-time")
async def get_real_time_performance(
    performance_manager: PerformanceMetricsManager = Depends(get_performance_manager)
):
    """Get real-time performance metrics with streaming updates."""
    try:
        async def generate_real_time_metrics():
            
    """generate_real_time_metrics function."""
while True:
                try:
                    current_metrics = performance_manager.get_current_metrics()
                    alerts = performance_manager.get_performance_alerts()
                    
                    data = {
                        "metrics": current_metrics.model_dump(mode='json'),
                        "alerts": alerts,
                        "timestamp": time.time()
                    }
                    
                    yield f"data: {orjson.dumps(data).decode()}\n\n"
                    
                    # Update every 5 seconds
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error("Error generating real-time metrics", error=str(e))
                    await asyncio.sleep(10)
        
        return StreamingResponse(
            generate_real_time_metrics(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    except Exception as e:
        logger.error("Error setting up real-time performance monitoring", error=str(e))
        raise HTTPException(status_code=500, detail="Error setting up real-time monitoring")

@app.get("/stats")
async def stats(
    container: DependencyContainer = Depends(get_dependency_container),
    cache_manager: CacheManager = Depends(get_cache_manager),
    static_cache: StaticDataCache = Depends(get_static_cache)
):
    """Application statistics with dependency injection."""
    return {
        "requests_total": container.request_count,
        "errors_total": container.error_count,
        "uptime_seconds": container.uptime,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.Process().cpu_percent(),
        "cache_stats": cache_manager.get_stats(),
        "static_cache_size": len(static_cache.seo_rules) + len(static_cache.keyword_scores)
    }

# Lazy Loading API Endpoints
@app.post("/analyze/bulk", response_model=BulkSEOResult)
async def bulk_analyze_seo_endpoint(
    request: BulkSEOParams,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Bulk SEO analysis with lazy loading and streaming."""
    try:
        # Validate URLs
        valid_urls = [url for url in request.urls if is_valid_url(url)]
        if len(valid_urls) != len(request.urls):
            logger.warning("Some URLs were invalid", total=len(request.urls), valid=len(valid_urls))
        
        if not valid_urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided")
        
        # Update request with valid URLs
        request.urls = valid_urls
        
        # Process bulk analysis with lazy loading
        results = []
        async for result in bulk_processor.process_bulk_analysis(request):
            results.append(result)
        
        # Return final result
        return results[-1] if results else BulkSEOResult()
        
    except Exception as e:
        logger.error("Bulk SEO analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Bulk analysis failed")

@app.post("/analyze/bulk/stream")
async def bulk_analyze_seo_stream_endpoint(
    request: BulkSEOParams,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Streaming bulk SEO analysis with lazy loading."""
    try:
        # Validate URLs
        valid_urls = [url for url in request.urls if is_valid_url(url)]
        if not valid_urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided")
        
        request.urls = valid_urls
        
        async def generate_stream():
            """Generate streaming response."""
            yield "{\n"
            yield '"results": [\n'
            
            first_result = True
            async for result in bulk_processor.process_bulk_analysis(request):
                if not first_result:
                    yield ",\n"
                else:
                    first_result = False
                
                json_result = orjson.dumps(result.model_dump(mode='json')).decode('utf-8')
                yield json_result
                
                # Yield control to event loop
                await asyncio.sleep(0.001)
            
            yield "\n],\n"
            yield '"status": "completed"\n'
            yield "}\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
        
    except Exception as e:
        logger.error("Streaming bulk analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Streaming analysis failed")

@app.get("/data/chunk/{chunk_index}")
async def get_data_chunk_endpoint(
    chunk_index: int,
    data_type: str = "seo_results",
    rate_limit: None = Depends(check_rate_limit)
):
    """Get a specific chunk of data with lazy loading."""
    try:
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail="Invalid chunk index")
        
        # This would typically load from database or cache
        # For demo purposes, we'll generate sample data
        sample_data = [
            {"id": i, "url": f"https://example{i}.com", "score": random.uniform(0, 100)}
            for i in range(chunk_index * 100, (chunk_index + 1) * 100)
        ]
        
        chunk_result = LazyLoadResult(
            data=sample_data,
            total_count=10000,  # Example total
            page=chunk_index + 1,
            page_size=100,
            has_next=chunk_index < 99,
            has_previous=chunk_index > 0,
            total_pages=100
        )
        
        return chunk_result
        
    except Exception as e:
        logger.error("Data chunk retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail="Data chunk retrieval failed")

@app.get("/data/paginated")
async def get_paginated_data_endpoint(
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    data_type: str = "seo_results",
    rate_limit: None = Depends(check_rate_limit)
):
    """Get paginated data with lazy loading."""
    try:
        pagination = PaginationParams(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Generate sample data for demonstration
        sample_data = [
            {"id": i, "url": f"https://example{i}.com", "score": random.uniform(0, 100)}
            for i in range(1000)
        ]
        
        # Apply sorting if specified
        if sort_by:
            sample_data = sorted(
                sample_data,
                key=lambda x: x.get(sort_by, 0),
                reverse=(sort_order == "desc")
            )
        
        result = await lazy_loader.get_paginated_data(sample_data, pagination)
        return result
        
    except Exception as e:
        logger.error("Paginated data retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail="Paginated data retrieval failed")

@app.get("/data/stream")
async def stream_data_endpoint(
    data_type: str = "seo_results",
    chunk_size: int = 100,
    rate_limit: None = Depends(check_rate_limit)
):
    """Stream data with lazy loading."""
    try:
        # Generate sample data for demonstration
        sample_data = [
            {"id": i, "url": f"https://example{i}.com", "score": random.uniform(0, 100)}
            for i in range(1000)
        ]
        
        config = LazyLoadingConfig(chunk_size=chunk_size)
        response_generator = LazyResponseGenerator(config)
        
        return StreamingResponse(
            response_generator.generate_streaming_response(sample_data),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
        
    except Exception as e:
        logger.error("Data streaming failed", error=str(e))
        raise HTTPException(status_code=500, detail="Data streaming failed")

@app.get("/data/compressed")
async def get_compressed_data_endpoint(
    data_type: str = "seo_results",
    compression_threshold: int = 1024,
    rate_limit: None = Depends(check_rate_limit)
):
    """Get compressed data for large datasets."""
    try:
        # Generate sample data for demonstration
        sample_data = [
            {"id": i, "url": f"https://example{i}.com", "score": random.uniform(0, 100)}
            for i in range(1000)
        ]
        
        config = LazyLoadingConfig(compression_threshold=compression_threshold)
        response_generator = LazyResponseGenerator(config)
        
        compressed_data = await response_generator.generate_compressed_response(sample_data)
        
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip" if len(compressed_data) > compression_threshold else "identity"
        }
        
        return Response(
            content=compressed_data,
            headers=headers
        )
        
    except Exception as e:
        logger.error("Compressed data retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail="Compressed data retrieval failed")

# ============================================================================
# NEW API ENDPOINTS USING DEDICATED ASYNC FUNCTIONS
# ============================================================================

@app.post("/database/store")
async def store_seo_result_endpoint(
    result: SEOResultModel,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Store SEO analysis result using dedicated async database operations."""
    try:
        success = await async_db_ops.store_seo_result(result)
        if success:
            return {"success": True, "message": "SEO result stored successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store SEO result")
    except Exception as e:
        logger.error("Failed to store SEO result", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@app.get("/database/retrieve/{url:path}")
async def retrieve_seo_result_endpoint(
    url: str,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Retrieve SEO analysis result using dedicated async database operations."""
    try:
        result = await async_db_ops.retrieve_seo_result(url)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="SEO result not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve SEO result", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@app.get("/database/history/{url:path}")
async def get_analysis_history_endpoint(
    url: str,
    limit: int = 10,
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Get analysis history using dedicated async database operations."""
    try:
        history = await async_db_ops.get_analysis_history(url, limit)
        return {"url": url, "history": history, "count": len(history)}
    except Exception as e:
        logger.error("Failed to get analysis history", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@app.get("/database/stats")
async def get_database_stats_endpoint(
    async_db_ops: AsyncDatabaseOperations = Depends(get_async_db_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Get database statistics using dedicated async database operations."""
    try:
        stats = await async_db_ops.get_database_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        raise HTTPException(status_code=500, detail="Database operation failed")

@app.post("/api/fetch-content")
async def fetch_page_content_endpoint(
    url: str,
    timeout: int = 30,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Fetch page content using dedicated async external API operations."""
    try:
        content_data = await async_api_ops.fetch_page_content(url, timeout)
        return content_data
    except Exception as e:
        logger.error("Failed to fetch page content", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@app.post("/api/check-accessibility")
async def check_url_accessibility_endpoint(
    url: str,
    timeout: int = 10,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Check URL accessibility using dedicated async external API operations."""
    try:
        accessibility_data = await async_api_ops.check_url_accessibility(url, timeout)
        return accessibility_data
    except Exception as e:
        logger.error("Failed to check URL accessibility", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@app.post("/api/batch-check-urls")
async def batch_check_urls_endpoint(
    urls: List[str],
    timeout: int = 10,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Batch check URLs using dedicated async external API operations."""
    try:
        results = await async_api_ops.batch_check_urls(urls, timeout)
        return {"results": results, "total_urls": len(urls)}
    except Exception as e:
        logger.error("Failed to batch check URLs", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@app.get("/api/robots-txt/{base_url:path}")
async def fetch_robots_txt_endpoint(
    base_url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Fetch robots.txt using dedicated async external API operations."""
    try:
        robots_data = await async_api_ops.fetch_robots_txt(base_url)
        return robots_data
    except Exception as e:
        logger.error("Failed to fetch robots.txt", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@app.get("/api/sitemap/{sitemap_url:path}")
async def fetch_sitemap_endpoint(
    sitemap_url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Fetch sitemap using dedicated async external API operations."""
    try:
        sitemap_data = await async_api_ops.fetch_sitemap(sitemap_url)
        return sitemap_data
    except Exception as e:
        logger.error("Failed to fetch sitemap", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@app.get("/api/metadata/{url:path}")
async def fetch_webpage_metadata_endpoint(
    url: str,
    async_api_ops: AsyncExternalAPIOperations = Depends(get_async_api_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Fetch webpage metadata using dedicated async external API operations."""
    try:
        metadata = await async_api_ops.fetch_webpage_metadata(url)
        return metadata
    except Exception as e:
        logger.error("Failed to fetch webpage metadata", error=str(e))
        raise HTTPException(status_code=500, detail="External API operation failed")

@app.post("/persistence/store")
async def persist_seo_analysis_endpoint(
    result: SEOResultModel,
    cache_ttl: int = 3600,
    async_persistence_ops: AsyncDataPersistenceOperations = Depends(get_async_persistence_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Persist SEO analysis using dedicated async persistence operations."""
    try:
        success = await async_persistence_ops.persist_seo_analysis(result, cache_ttl)
        if success:
            return {"success": True, "message": "SEO analysis persisted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to persist SEO analysis")
    except Exception as e:
        logger.error("Failed to persist SEO analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Persistence operation failed")

@app.post("/persistence/bulk-store")
async def persist_bulk_analyses_endpoint(
    results: List[SEOResultModel],
    cache_ttl: int = 3600,
    async_persistence_ops: AsyncDataPersistenceOperations = Depends(get_async_persistence_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Persist bulk SEO analyses using dedicated async persistence operations."""
    try:
        persisted_count = await async_persistence_ops.persist_bulk_analyses(results, cache_ttl)
        return {
            "success": True,
            "persisted_count": persisted_count,
            "total_count": len(results)
        }
    except Exception as e:
        logger.error("Failed to persist bulk analyses", error=str(e))
        raise HTTPException(status_code=500, detail="Persistence operation failed")

@app.post("/persistence/backup")
async def backup_analysis_data_endpoint(
    collection: str = "seo_results",
    async_persistence_ops: AsyncDataPersistenceOperations = Depends(get_async_persistence_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Create backup using dedicated async persistence operations."""
    try:
        backup_info = await async_persistence_ops.backup_analysis_data(collection)
        return backup_info
    except Exception as e:
        logger.error("Failed to backup analysis data", error=str(e))
        raise HTTPException(status_code=500, detail="Backup operation failed")

@app.post("/persistence/restore")
async def restore_analysis_data_endpoint(
    backup_file: str,
    collection: str = "seo_results",
    async_persistence_ops: AsyncDataPersistenceOperations = Depends(get_async_persistence_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Restore data using dedicated async persistence operations."""
    try:
        restore_info = await async_persistence_ops.restore_analysis_data(backup_file, collection)
        return restore_info
    except Exception as e:
        logger.error("Failed to restore analysis data", error=str(e))
        raise HTTPException(status_code=500, detail="Restore operation failed")

@app.post("/persistence/export")
async def export_analysis_data_endpoint(
    format: str = "json",
    collection: str = "seo_results",
    async_persistence_ops: AsyncDataPersistenceOperations = Depends(get_async_persistence_operations),
    logger: structlog.BoundLogger = Depends(get_logger)
):
    """Export data using dedicated async persistence operations."""
    try:
        export_info = await async_persistence_ops.export_analysis_data(format, collection)
        return export_info
    except Exception as e:
        logger.error("Failed to export analysis data", error=str(e))
        raise HTTPException(status_code=500, detail="Export operation failed")

# ============================================================================
# END OF NEW API ENDPOINTS
# ============================================================================

# Async Startup and shutdown events with dependency injection
@app.on_event("startup")
async def startup_event():
    """Application startup with non-blocking optimizations."""
    container.logger.info("Starting Ultra-Optimized SEO Service v15 with non-blocking optimizations")
    
    # Initialize Sentry
    if container.config.sentry_dsn:
        sentry_sdk.init(
            dsn=container.config.sentry_dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.1
        )
        container.logger.info("Sentry initialized")
    
    # Start non-blocking task scheduler
    await task_scheduler.start()
    container.logger.info("Non-blocking task scheduler started")
    
    # Initialize Redis with async connection pooling
    try:
        redis_client = await get_redis()
        if redis_client:
            container.logger.info("Redis connected successfully with async pooling")
        else:
            container.logger.warning("Redis connection failed, using memory cache only")
    except Exception as e:
        container.logger.warning("Redis connection failed", error=str(e))
    
    # Initialize MongoDB with async connection pooling
    try:
        mongo_client = await get_mongo()
        if mongo_client:
            container.logger.info("MongoDB connected successfully with async pooling")
        else:
            container.logger.warning("MongoDB connection failed")
    except Exception as e:
        container.logger.warning("MongoDB connection failed", error=str(e))
    
    # Initialize HTTP client with async connection pooling
    try:
        http_client = await get_http_client()
        container.logger.info("HTTP client initialized with async pooling")
    except Exception as e:
        container.logger.warning("HTTP client initialization failed", error=str(e))
    
    # Preload static data into cache in background
    await non_blocking_manager.add_background_task(
        container.static_cache.preload_static_data
    )
    container.logger.info("Static data preloading started in background")
    
    # Start performance monitoring
    performance_manager.start_monitoring()
    container.logger.info("Performance monitoring started")
    
    container.logger.info("Application startup completed with non-blocking optimizations")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown with non-blocking cleanup."""
    container.logger.info("Shutting down Ultra-Optimized SEO Service v15")
    
    # Stop non-blocking task scheduler
    await task_scheduler.stop()
    container.logger.info("Non-blocking task scheduler stopped")
    
    # Close all connection pools
    await connection_pool_manager.close_all()
    container.logger.info("Connection pools closed")
    
    # Close Redis connection asynchronously
    if container._redis_client:
        try:
            await container._redis_client.close()
            container.logger.info("Redis connection closed")
        except Exception as e:
            container.logger.warning("Error closing Redis connection", error=str(e))
    
    # Close MongoDB connection asynchronously
    if container._mongo_client:
        try:
            container._mongo_client.close()
            container.logger.info("MongoDB connection closed")
        except Exception as e:
            container.logger.warning("Error closing MongoDB connection", error=str(e))
    
    # Close HTTP client asynchronously
    if container._http_client:
        try:
            await container._http_client.aclose()
            container.logger.info("HTTP client closed")
        except Exception as e:
            container.logger.warning("Error closing HTTP client", error=str(e))
    
    # Stop performance monitoring
    performance_manager.stop_monitoring()
    container.logger.info("Performance monitoring stopped")
    
    # Shutdown thread and process pools
    _thread_pool.shutdown(wait=True)
    _process_pool.shutdown(wait=True)
    container.logger.info("Thread and process pools shutdown")
    
    container.logger.info("Application shutdown completed with non-blocking cleanup")

# Signal handlers
def signal_handler(signum, frame) -> Any:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main entry point
if __name__ == "__main__":
    # Use uvloop for better performance
    uvloop.install()
    
    # Start server
    uvicorn.run(
        "main_production_v15_ultra:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        loop="uvloop",
        http="httptools",
        access_log=True,
        log_level="info" if config.debug else "warning"
    ) 