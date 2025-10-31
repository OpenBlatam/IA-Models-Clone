"""
Ultimate BUL API Enhancements
============================

Advanced enhancements following modern Python and FastAPI best practices:
- Microservices architecture patterns
- Event-driven programming
- Advanced async patterns
- Machine learning integration
- Real-time analytics
- Advanced security patterns
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic, AsyncGenerator
from functools import wraps, lru_cache, partial
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path

# Advanced FastAPI imports
from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Advanced async and concurrency
import aiofiles
import aioredis
import asyncpg
from asyncio import Queue, Event, Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Advanced data processing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from textblob import TextBlob

# Advanced monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import psutil
import memory_profiler
from datadog import initialize, statsd

# Advanced caching and storage
import redis
from redis.asyncio import Redis
import pickle
import zlib
import lz4

# Advanced security
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import bcrypt
import argon2

# Advanced HTTP and networking
import httpx
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
import websockets
from websockets.server import serve

# Advanced database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, Float, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy import func, select, update, delete, insert

# Advanced validation and serialization
from pydantic import BaseModel, Field, validator, root_validator, create_model
from pydantic.types import PositiveInt, NonNegativeInt, EmailStr, HttpUrl, UUID4
import marshmallow
from marshmallow import Schema, fields, validate, post_load, pre_load
import orjson
import ujson

# Advanced logging
import structlog
from loguru import logger
import sys
from pythonjsonlogger import jsonlogger

# Advanced testing
import pytest
from pytest_asyncio import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

# Type variables
T = TypeVar('T')
R = TypeVar('R')
K = TypeVar('K')
V = TypeVar('V')

# Advanced Event System
class EventBus:
    """Advanced event bus for microservices communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue: Queue = Queue()
        self.processing = False
    
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, data: Any) -> None:
        """Publish event"""
        await self.event_queue.put({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(),
            "id": str(uuid.uuid4())
        })
    
    async def start_processing(self) -> None:
        """Start event processing"""
        self.processing = True
        while self.processing:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """Process event"""
        event_type = event["type"]
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event["data"])
                    else:
                        handler(event["data"])
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
    
    async def stop_processing(self) -> None:
        """Stop event processing"""
        self.processing = False

# Advanced Circuit Breaker
class CircuitBreaker:
    """Advanced circuit breaker pattern"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self) -> None:
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Advanced Rate Limiter
class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        algorithm: str = "token_bucket"
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.algorithm = algorithm
        
        # Token bucket algorithm
        self.tokens = burst_size
        self.last_update = time.time()
        self.token_rate = requests_per_minute / 60.0
        
        # Sliding window algorithm
        self.requests: List[float] = []
        self.window_size = 60.0
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        if self.algorithm == "token_bucket":
            return await self._token_bucket_check()
        elif self.algorithm == "sliding_window":
            return await self._sliding_window_check()
        else:
            return True
    
    async def _token_bucket_check(self) -> bool:
        """Token bucket algorithm"""
        now = time.time()
        time_passed = now - self.last_update
        self.last_update = now
        
        # Add tokens based on time passed
        self.tokens = min(
            self.burst_size,
            self.tokens + time_passed * self.token_rate
        )
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
    
    async def _sliding_window_check(self) -> bool:
        """Sliding window algorithm"""
        now = time.time()
        
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window_size]
        
        # Check if under limit
        if len(self.requests) < self.requests_per_minute:
            self.requests.append(now)
            return True
        return False

# Advanced Caching System
class AdvancedCache:
    """Advanced caching system with multiple strategies"""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        strategy: str = "lru",
        compression: bool = True
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.strategy = strategy
        self.compression = compression
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.creation_times: Dict[str, float] = {}
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        if time.time() - self.creation_times[key] > self.ttl:
            await self.delete(key)
            self.misses += 1
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        self.hits += 1
        
        # Get value
        value = self.cache[key]["value"]
        
        # Decompress if needed
        if self.compression and self.cache[key].get("compressed", False):
            value = zlib.decompress(value)
        
        # Deserialize
        if self.cache[key].get("serialized", False):
            value = pickle.loads(value)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        # Evict if needed
        if len(self.cache) >= self.max_size:
            await self._evict()
        
        # Serialize if needed
        if not isinstance(value, (str, int, float, bool)):
            value = pickle.dumps(value)
            serialized = True
        else:
            serialized = False
        
        # Compress if enabled
        if self.compression and isinstance(value, bytes):
            value = zlib.compress(value)
            compressed = True
        else:
            compressed = False
        
        # Store in cache
        self.cache[key] = {
            "value": value,
            "serialized": serialized,
            "compressed": compressed
        }
        self.creation_times[key] = time.time()
        self.access_times[key] = time.time()
    
    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.creation_times[key]
            del self.access_times[key]
    
    async def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    async def _evict(self) -> None:
        """Evict item based on strategy"""
        if self.strategy == "lru":
            # Remove least recently used
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            await self.delete(lru_key)
        elif self.strategy == "lfu":
            # Remove least frequently used (simplified)
            lfu_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            await self.delete(lfu_key)
        elif self.strategy == "ttl":
            # Remove oldest
            oldest_key = min(self.creation_times.keys(), key=lambda k: self.creation_times[k])
            await self.delete(oldest_key)
        
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "strategy": self.strategy
        }

# Advanced Machine Learning Integration
class MLProcessor:
    """Advanced machine learning processor"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.models: Dict[str, Any] = {}
        self.training_data: List[Dict[str, Any]] = []
    
    async def train_model(self, data: List[Dict[str, Any]], model_type: str = "classification") -> str:
        """Train ML model"""
        model_id = str(uuid.uuid4())
        
        # Prepare training data
        texts = [item.get("text", "") for item in data]
        labels = [item.get("label", "") for item in data]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train model based on type
        if model_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, labels)
        elif model_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, labels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Store model
        self.models[model_id] = {
            "model": model,
            "vectorizer": self.vectorizer,
            "type": model_type,
            "trained_at": datetime.now()
        }
        
        return model_id
    
    async def predict(self, model_id: str, text: str) -> Any:
        """Make prediction using trained model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_data = self.models[model_id]
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        
        # Vectorize input text
        X = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_id": model_id
        }
    
    async def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        # Vectorize texts
        texts = [text1, text2]
        X = self.vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
        
        return float(similarity)
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment"""
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        return {
            "polarity": sentiment.polarity,
            "subjectivity": sentiment.subjectivity,
            "sentiment": "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
        }

# Advanced Real-time Analytics
class RealTimeAnalytics:
    """Advanced real-time analytics system"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, float] = {}
        self.processors: List[Callable] = []
    
    async def record_metric(self, name: str, value: float) -> None:
        """Record metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only last 1000 values
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        # Check thresholds
        await self._check_thresholds(name, value)
        
        # Process with registered processors
        for processor in self.processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(name, value)
                else:
                    processor(name, value)
            except Exception as e:
                logger.error(f"Error in metric processor: {e}")
    
    async def _check_thresholds(self, name: str, value: float) -> None:
        """Check metric thresholds"""
        if name in self.thresholds:
            threshold = self.thresholds[name]
            if value > threshold:
                await self._create_alert(name, value, threshold, "high")
            elif value < -threshold:
                await self._create_alert(name, value, -threshold, "low")
    
    async def _create_alert(self, name: str, value: float, threshold: float, type: str) -> None:
        """Create alert"""
        alert = {
            "id": str(uuid.uuid4()),
            "metric": name,
            "value": value,
            "threshold": threshold,
            "type": type,
            "timestamp": datetime.now(),
            "resolved": False
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {name} {type} threshold exceeded", extra=alert)
    
    async def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Get metric statistics"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "trend": self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent = values[-10:] if len(values) >= 10 else values
        older = values[-20:-10] if len(values) >= 20 else values[:-10] if len(values) > 10 else []
        
        if not older:
            return "stable"
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    async def set_threshold(self, name: str, threshold: float) -> None:
        """Set metric threshold"""
        self.thresholds[name] = threshold
    
    async def add_processor(self, processor: Callable) -> None:
        """Add metric processor"""
        self.processors.append(processor)

# Advanced WebSocket Manager
class WebSocketManager:
    """Advanced WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_groups: Dict[str, List[str]] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        
        logger.info(f"WebSocket connected: {client_id}")
    
    async def disconnect(self, client_id: str) -> None:
        """Disconnect WebSocket"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            
            # Remove from groups
            for group_name, connections in self.connection_groups.items():
                if client_id in connections:
                    connections.remove(client_id)
            
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: str, client_id: str) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def send_group_message(self, message: str, group_name: str) -> None:
        """Send message to group"""
        if group_name in self.connection_groups:
            for client_id in self.connection_groups[group_name]:
                await self.send_personal_message(message, client_id)
    
    async def broadcast(self, message: str) -> None:
        """Broadcast message to all connections"""
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, client_id)
    
    async def add_to_group(self, client_id: str, group_name: str) -> None:
        """Add client to group"""
        if group_name not in self.connection_groups:
            self.connection_groups[group_name] = []
        
        if client_id not in self.connection_groups[group_name]:
            self.connection_groups[group_name].append(client_id)
    
    async def remove_from_group(self, client_id: str, group_name: str) -> None:
        """Remove client from group"""
        if group_name in self.connection_groups and client_id in self.connection_groups[group_name]:
            self.connection_groups[group_name].remove(client_id)
    
    def get_connection_count(self) -> int:
        """Get total connection count"""
        return len(self.active_connections)
    
    def get_group_count(self, group_name: str) -> int:
        """Get group connection count"""
        return len(self.connection_groups.get(group_name, []))

# Advanced Background Task Manager
class BackgroundTaskManager:
    """Advanced background task manager"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.task_queue: Queue = Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def start(self) -> None:
        """Start background task manager"""
        self.running = True
        
        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} background workers")
    
    async def stop(self) -> None:
        """Stop background task manager"""
        self.running = False
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Stopped background task manager")
    
    async def add_task(self, task_func: Callable, *args, **kwargs) -> str:
        """Add task to queue"""
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        await self.task_queue.put(task)
        return task_id
    
    async def _worker(self, worker_name: str) -> None:
        """Background worker"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self._execute_task(task, worker_name)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
    
    async def _execute_task(self, task: Dict[str, Any], worker_name: str) -> None:
        """Execute task"""
        task_id = task["id"]
        start_time = time.time()
        
        try:
            task["status"] = "running"
            task["worker"] = worker_name
            task["started_at"] = datetime.now()
            
            # Execute task
            if asyncio.iscoroutinefunction(task["func"]):
                result = await task["func"](*task["args"], **task["kwargs"])
            else:
                result = task["func"](*task["args"], **task["kwargs"])
            
            # Record success
            duration = time.time() - start_time
            task["status"] = "completed"
            task["completed_at"] = datetime.now()
            task["duration"] = duration
            task["result"] = result
            
            logger.info(f"Task {task_id} completed in {duration:.2f}s")
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            task["status"] = "failed"
            task["failed_at"] = datetime.now()
            task["duration"] = duration
            task["error"] = str(e)
            
            logger.error(f"Task {task_id} failed after {duration:.2f}s: {e}")
        
        # Update metrics
        self.task_metrics[task_id] = task
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return self.task_metrics.get(task_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get task manager metrics"""
        total_tasks = len(self.task_metrics)
        completed_tasks = sum(1 for task in self.task_metrics.values() if task["status"] == "completed")
        failed_tasks = sum(1 for task in self.task_metrics.values() if task["status"] == "failed")
        running_tasks = sum(1 for task in self.task_metrics.values() if task["status"] == "running")
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "queue_size": self.task_queue.qsize(),
            "active_workers": len(self.workers)
        }

# Advanced API Factory
class AdvancedAPIFactory:
    """Advanced API factory for creating enhanced FastAPI applications"""
    
    @staticmethod
    def create_app(
        title: str = "BUL API",
        version: str = "3.0.0",
        description: str = "Advanced Business Universal Language API",
        enable_cors: bool = True,
        enable_compression: bool = True,
        enable_trusted_host: bool = True,
        enable_metrics: bool = True,
        enable_websockets: bool = True,
        enable_background_tasks: bool = True
    ) -> FastAPI:
        """Create enhanced FastAPI application"""
        
        # Create FastAPI app
        app = FastAPI(
            title=title,
            version=version,
            description=description,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
        
        # Add middleware
        if enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
        
        if enable_compression:
            app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        if enable_trusted_host:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
        
        # Add advanced middleware
        app.add_middleware(AdvancedMiddleware)
        
        # Initialize components
        app.state.event_bus = EventBus()
        app.state.circuit_breaker = CircuitBreaker()
        app.state.rate_limiter = AdvancedRateLimiter()
        app.state.cache = AdvancedCache()
        app.state.ml_processor = MLProcessor()
        app.state.analytics = RealTimeAnalytics()
        app.state.websocket_manager = WebSocketManager()
        app.state.background_tasks = BackgroundTaskManager()
        
        # Add startup and shutdown events
        @app.on_event("startup")
        async def startup_event():
            """Startup event handler"""
            logger.info("Starting BUL API...")
            
            # Start background components
            await app.state.event_bus.start_processing()
            await app.state.background_tasks.start()
            
            logger.info("BUL API started successfully")
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Shutdown event handler"""
            logger.info("Shutting down BUL API...")
            
            # Stop background components
            await app.state.event_bus.stop_processing()
            await app.state.background_tasks.stop()
            
            logger.info("BUL API shut down successfully")
        
        # Add health check endpoint
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "version": version,
                "components": {
                    "event_bus": "running",
                    "cache": "running",
                    "analytics": "running",
                    "background_tasks": "running"
                }
            }
        
        # Add metrics endpoint
        if enable_metrics:
            @app.get("/metrics")
            async def metrics():
                """Metrics endpoint"""
                return {
                    "cache_stats": app.state.cache.get_stats(),
                    "analytics_stats": {
                        "metrics_count": len(app.state.analytics.metrics),
                        "alerts_count": len(app.state.analytics.alerts)
                    },
                    "background_tasks": app.state.background_tasks.get_metrics(),
                    "websocket_connections": app.state.websocket_manager.get_connection_count()
                }
        
        return app

# Advanced Middleware
class AdvancedMiddleware:
    """Advanced middleware for request/response processing"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Middleware call"""
        if scope["type"] == "http":
            # Process HTTP request
            request = Request(scope, receive)
            
            # Add request ID
            request_id = str(uuid.uuid4())
            scope["request_id"] = request_id
            
            # Add timing
            start_time = time.time()
            scope["start_time"] = start_time
            
            # Process request
            await self.app(scope, receive, send)
            
            # Log request
            duration = time.time() - start_time
            logger.info(
                f"Request processed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "duration": duration
                }
            )
        else:
            await self.app(scope, receive, send)

# Export all advanced components
__all__ = [
    # Event System
    "EventBus",
    
    # Circuit Breaker
    "CircuitBreaker",
    
    # Rate Limiter
    "AdvancedRateLimiter",
    
    # Caching
    "AdvancedCache",
    
    # Machine Learning
    "MLProcessor",
    
    # Analytics
    "RealTimeAnalytics",
    
    # WebSocket Manager
    "WebSocketManager",
    
    # Background Tasks
    "BackgroundTaskManager",
    
    # API Factory
    "AdvancedAPIFactory",
    
    # Middleware
    "AdvancedMiddleware"
]












