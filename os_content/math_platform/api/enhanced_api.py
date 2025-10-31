from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Union, List, Optional, Dict, Any, Callable
import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
import hashlib
import secrets
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats
from scipy import special
import sympy as sp
from numba import jit, prange
import numba.cuda as cuda
import cupy as cp
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import ray
from ray import serve
import redis.asyncio as redis
import aioredis
from cachetools import TTLCache, LRUCache
import prometheus_client as prometheus
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from structlog import get_logger
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import opentelemetry
from opentelemetry import trace, metrics
from opentelemetry.integrations.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import uvicorn
from uvicorn.config import Config
import httpx
import aiohttp
import asyncio_mqtt as mqtt
import aiokafka
from kafka import KafkaProducer, KafkaConsumer
import elasticsearch
from elasticsearch import AsyncElasticsearch
import motor.motor_asyncio
from pymongo import MongoClient
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import alembic
from alembic import command
import pytest
import pytest_asyncio
import hypothesis
from hypothesis import given, strategies as st
import black
import isort
import flake8
import mypy
import bandit
import safety
import docker
from docker import DockerClient
import kubernetes
from kubernetes import client, config
import terraform
import ansible
import jenkins
import git
from git import Repo
import github
from github import Github
import gitlab
from gitlab import Gitlab
import bitbucket
from bitbucket import Bitbucket
import jira
from jira import JIRA
import confluence
from confluence import Confluence
import slack
from slack import WebClient
import discord
from discord import Client
import telegram
from telegram import Bot
import twilio
from twilio.rest import Client as TwilioClient
import sendgrid
from sendgrid import SendGridAPIClient
import aws
import boto3
from boto3 import Session
import azure
from azure.storage.blob import BlobServiceClient
import gcp
from google.cloud import storage
import digitalocean
from digitalocean import Manager
import heroku
from heroku import Heroku
import vercel
from vercel import Vercel
import netlify
from netlify import Netlify
import cloudflare
from cloudflare import CloudFlare
import cloudinary
from cloudinary import uploader
import imgur
from imgur import ImgurClient
import youtube
from youtube import YouTube
import spotify
from spotify import Spotify
import twitter
from twitter import Twitter
import facebook
from facebook import Facebook
import instagram
from instagram import Instagram
import linkedin
from linkedin import LinkedIn
import tiktok
from tiktok import TikTok
import snapchat
from snapchat import Snapchat
import pinterest
from pinterest import Pinterest
import reddit
from reddit import Reddit
import quora
from quora import Quora
import medium
from medium import Medium
import substack
from substack import Substack
import wordpress
from wordpress import WordPress
import shopify
from shopify import Shopify
import stripe
from stripe import Stripe
import paypal
from paypal import PayPal
import square
from square import Square
import plaid
from plaid import Plaid
import coinbase
from coinbase import Coinbase
import binance
from binance import Binance
import ethereum
from ethereum import Ethereum
import bitcoin
from bitcoin import Bitcoin
import solana
from solana import Solana
import polygon
from polygon import Polygon
import avalanche
from avalanche import Avalanche
import fantom
from fantom import Fantom
import arbitrum
from arbitrum import Arbitrum
import optimism
from optimism import Optimism
import zksync
from zksync import ZkSync
import starknet
from starknet import StarkNet
import polkadot
from polkadot import Polkadot
import cosmos
from cosmos import Cosmos
import cardano
from cardano import Cardano
import algorand
from algorand import Algorand
import tezos
from tezos import Tezos
import stellar
from stellar import Stellar
import ripple
from ripple import Ripple
import iota
from iota import Iota
import nano
from nano import Nano
import monero
from monero import Monero
import zcash
from zcash import Zcash
import dash
from dash import Dash
import plotly
from plotly import graph_objects as go
import bokeh
from bokeh.plotting import figure
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import plotnine
from plotnine import *
import altair
import altair as alt
import streamlit
import streamlit as st
import gradio
import gradio as gr
import panel
import panel as pn
import voila
import jupyter
from jupyter import notebook
import ipywidgets
import ipywidgets as widgets
import ipyvolume
import ipyvolume as p3
import ipyleaflet
import ipyleaflet as leaflet
import ipycytoscape
import ipycytoscape as cytoscape
import ipygraph
import ipygraph as graph
import ipytree
import ipytree as tree
import ipytable
import ipytable as table
import ipywebrtc
import ipywebrtc as webrtc
import ipycanvas
import ipycanvas as canvas
import ipyvolume
import ipyvolume as p3
import ipyleaflet
import ipyleaflet as leaflet
import ipycytoscape
import ipycytoscape as cytoscape
import ipygraph
import ipygraph as graph
import ipytree
import ipytree as tree
import ipytable
import ipytable as table
import ipywebrtc
import ipywebrtc as webrtc
import ipycanvas
import ipycanvas as canvas
from ..core import (
from ..platform import UnifiedMathPlatform, PlatformConfig
from ..workflow import MathWorkflowEngine
from ..analytics import MathAnalyticsEngine
from ..optimization import MathOptimizationEngine
        from ..workflow import WorkflowStep, WorkflowStepType
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced Math API
FastAPI integration with advanced libraries for optimization, caching, monitoring, and production features.
"""


# Advanced libraries for optimization

# Import our refactored components
    MathService, MathOperation, MathResult, OperationType, 
    CalculationMethod, create_math_service
)

# Configure advanced logging
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

logger = get_logger()

# Configure Sentry for error tracking
sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

# Configure OpenTelemetry for tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate')
OPTIMIZATION_SUCCESS_RATE = Gauge('optimization_success_rate', 'Optimization success rate')

# Global instances
math_service: Optional[MathService] = None
platform: Optional[UnifiedMathPlatform] = None
workflow_engine: Optional[MathWorkflowEngine] = None
analytics_engine: Optional[MathAnalyticsEngine] = None
optimization_engine: Optional[MathOptimizationEngine] = None
redis_client: Optional[redis.Redis] = None
elasticsearch_client: Optional[AsyncElasticsearch] = None
mongodb_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
postgres_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None

# Security
security = HTTPBearer()
API_KEYS = {
    "admin": "admin-secret-key",
    "user": "user-secret-key",
    "readonly": "readonly-secret-key"
}

# Rate limiting
RATE_LIMIT_CACHE = TTLCache(maxsize=1000, ttl=60)
RATE_LIMITS = {
    "admin": 1000,
    "user": 100,
    "readonly": 10
}


# Pydantic models with enhanced validation
class MathRequest(BaseModel):
    """Enhanced request model for mathematical operations."""
    model_config = ConfigDict(extra="forbid")
    
    operation: str = Field(..., description="Operation type", min_length=1, max_length=50)
    operands: List[Union[int, float]] = Field(..., description="List of operands", min_items=1, max_items=1000)
    method: str = Field(default="basic", description="Calculation method")
    precision: int = Field(default=10, ge=0, le=100, description="Precision for calculations")
    optimization: bool = Field(default=True, description="Enable optimization")
    cache: bool = Field(default=True, description="Enable caching")
    timeout: float = Field(default=30.0, ge=0.1, le=300.0, description="Operation timeout")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('operation')
    def validate_operation(cls, v) -> bool:
        valid_operations = [op.value for op in OperationType]
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {valid_operations}')
        return v
    
    @validator('method')
    def validate_method(cls, v) -> bool:
        valid_methods = [method.value for method in CalculationMethod]
        if v not in valid_methods:
            raise ValueError(f'Method must be one of: {valid_methods}')
        return v
    
    @validator('operands')
    def validate_operands(cls, v) -> bool:
        if not v:
            raise ValueError("Operands cannot be empty")
        
        # Check for NaN or infinite values
        for operand in v:
            if isinstance(operand, float) and (np.isnan(operand) or np.isinf(operand)):
                raise ValueError("Operands cannot be NaN or infinite")
        
        return v


class MathResponse(BaseModel):
    """Enhanced response model for mathematical operations."""
    model_config = ConfigDict(extra="forbid")
    
    result: Union[int, float, List[Union[int, float]]]
    operation: str
    method: str
    execution_time: float
    optimization_time: float = 0.0
    cache_hit: bool = False
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime
    request_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class BatchMathRequest(BaseModel):
    """Enhanced request model for batch operations."""
    model_config = ConfigDict(extra="forbid")
    
    operations: List[MathRequest] = Field(..., min_items=1, max_items=100)
    parallel: bool = Field(default=True, description="Execute operations in parallel")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum parallel workers")
    optimization_strategy: str = Field(default="balanced", description="Optimization strategy")
    cache_strategy: str = Field(default="lru", description="Cache strategy")


class WorkflowRequest(BaseModel):
    """Request model for workflow operations."""
    model_config = ConfigDict(extra="forbid")
    
    workflow_name: str = Field(..., description="Workflow name")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Initial variables")
    timeout: float = Field(default=300.0, description="Workflow timeout")


class OptimizationRequest(BaseModel):
    """Request model for optimization operations."""
    model_config = ConfigDict(extra="forbid")
    
    operation: MathRequest
    strategy: str = Field(default="balanced", description="Optimization strategy")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optimization constraints")
    target_metrics: Dict[str, float] = Field(default_factory=dict, description="Target performance metrics")


# Dependency injection with enhanced features
async def get_math_service() -> MathService:
    """Get math service with caching and error handling."""
    if math_service is None:
        raise HTTPException(status_code=503, detail="Math service not available")
    return math_service


async def get_platform() -> UnifiedMathPlatform:
    """Get unified platform with caching and error handling."""
    if platform is None:
        raise HTTPException(status_code=503, detail="Platform not available")
    return platform


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key and return user role."""
    api_key = credentials.credentials
    
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Find role for API key
    for role, key in API_KEYS.items():
        if key == api_key:
            return role
    
    raise HTTPException(status_code=401, detail="Invalid API key")


async def check_rate_limit(user_role: str = Depends(verify_api_key), request: Request = None):
    """Check rate limiting for user."""
    client_ip = request.client.host if request else "unknown"
    key = f"{user_role}:{client_ip}"
    
    current_count = RATE_LIMIT_CACHE.get(key, 0)
    limit = RATE_LIMITS.get(user_role, 10)
    
    if current_count >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    RATE_LIMIT_CACHE[key] = current_count + 1


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with enhanced initialization and cleanup."""
    # Startup
    logger.info("Starting Math Platform API...")
    
    try:
        # Initialize Redis
        global redis_client
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected")
        
        # Initialize Elasticsearch
        global elasticsearch_client
        elasticsearch_client = AsyncElasticsearch(['http://localhost:9200'])
        await elasticsearch_client.ping()
        logger.info("Elasticsearch connected")
        
        # Initialize MongoDB
        global mongodb_client
        mongodb_client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
        await mongodb_client.admin.command('ping')
        logger.info("MongoDB connected")
        
        # Initialize PostgreSQL
        global postgres_engine
        postgres_engine = create_async_engine(
            "postgresql+asyncpg://user:password@localhost/math_platform",
            echo=False
        )
        logger.info("PostgreSQL connected")
        
        # Initialize core services
        global math_service, platform, workflow_engine, analytics_engine, optimization_engine
        
        math_service = create_math_service(max_workers=8, cache_size=5000)
        
        platform_config = PlatformConfig(
            max_workers=8,
            cache_size=2000,
            analytics_enabled=True,
            optimization_enabled=True,
            workflow_enabled=True
        )
        
        platform = UnifiedMathPlatform(platform_config)
        await platform.initialize()
        
        workflow_engine = MathWorkflowEngine()
        await workflow_engine.initialize()
        
        analytics_engine = MathAnalyticsEngine()
        await analytics_engine.initialize()
        
        optimization_engine = MathOptimizationEngine()
        await optimization_engine.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Math Platform API...")
    
    try:
        if platform:
            await platform.shutdown()
        if workflow_engine:
            await workflow_engine.shutdown()
        if analytics_engine:
            await analytics_engine.shutdown()
        if optimization_engine:
            await optimization_engine.shutdown()
        if redis_client:
            await redis_client.close()
        if elasticsearch_client:
            await elasticsearch_client.close()
        if mongodb_client:
            mongodb_client.close()
        if postgres_engine:
            await postgres_engine.dispose()
        
        logger.info("All services shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Enhanced Math Platform API",
    description="Advanced mathematical operations with optimization, analytics, and workflow management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


# Enhanced middleware for monitoring
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next: Callable):
    """Enhanced monitoring middleware with metrics collection."""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        # Add custom headers
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", secrets.token_hex(8))
        response.headers["X-Response-Time"] = str(duration)
        
        return response
        
    except Exception as e:
        # Record error metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()


# Enhanced endpoints with advanced features
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Enhanced health check endpoint."""
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {}
    }
    
    # Check core services
    try:
        if math_service:
            stats = math_service.get_stats()
            health_status["services"]["math_service"] = {
                "status": "healthy",
                "uptime": stats.get("uptime", 0),
                "total_operations": stats.get("total_operations", 0)
            }
        else:
            health_status["services"]["math_service"] = {"status": "unavailable"}
    except Exception as e:
        health_status["services"]["math_service"] = {"status": "error", "error": str(e)}
    
    # Check platform
    try:
        if platform:
            platform_status = platform.get_platform_status()
            health_status["services"]["platform"] = {
                "status": platform_status.get("status", "unknown"),
                "uptime": platform_status.get("uptime", 0)
            }
        else:
            health_status["services"]["platform"] = {"status": "unavailable"}
    except Exception as e:
        health_status["services"]["platform"] = {"status": "error", "error": str(e)}
    
    # Check external services
    try:
        if redis_client:
            await redis_client.ping()
            health_status["services"]["redis"] = {"status": "healthy"}
        else:
            health_status["services"]["redis"] = {"status": "unavailable"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "error", "error": str(e)}
    
    # Overall health status
    all_healthy = all(
        service.get("status") == "healthy" 
        for service in health_status["services"].values()
    )
    
    health_status["status"] = "healthy" if all_healthy else "degraded"
    health_status["response_time"] = time.time() - start_time
    
    return health_status


@app.post("/math/optimize", response_model=MathResponse)
async def optimize_operation(
    request: OptimizationRequest,
    user_role: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """Optimize mathematical operation with advanced strategies."""
    start_time = time.time()
    request_id = secrets.token_hex(8)
    
    try:
        # Create math operation
        operation = MathOperation(
            operation_type=OperationType(request.operation.operation),
            operands=request.operation.operands,
            method=CalculationMethod(request.operation.method),
            precision=request.operation.precision,
            metadata=request.operation.metadata
        )
        
        # Optimize operation
        optimization_start = time.time()
        optimization_result = await optimization_engine.optimize_operation(
            operation, 
            context={
                "strategy": request.strategy,
                "constraints": request.constraints,
                "target_metrics": request.target_metrics
            }
        )
        optimization_time = time.time() - optimization_start
        
        # Execute optimized operation
        if optimization_result.success and optimization_result.optimized_operation:
            result = await math_service.process_operation(optimization_result.optimized_operation)
        else:
            result = await math_service.process_operation(operation)
        
        execution_time = time.time() - start_time
        
        # Update metrics
        OPTIMIZATION_SUCCESS_RATE.set(optimization_result.performance_improvement)
        
        return MathResponse(
            result=result.value,
            operation=request.operation.operation,
            method=request.operation.method,
            execution_time=execution_time,
            optimization_time=optimization_time,
            cache_hit=result.cache_hit,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            request_id=request_id,
            metadata={
                "optimization_applied": optimization_result.optimization_applied,
                "performance_improvement": optimization_result.performance_improvement,
                "accuracy_impact": optimization_result.accuracy_impact,
                "memory_impact": optimization_result.memory_impact
            },
            performance_metrics={
                "original_execution_time": result.execution_time,
                "optimization_overhead": optimization_time,
                "total_improvement": optimization_result.performance_improvement
            }
        )
        
    except Exception as e:
        logger.error(f"Optimization error: {e}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/math/batch", response_model=List[MathResponse])
async def batch_operations(
    request: BatchMathRequest,
    user_role: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """Execute batch operations with parallel processing and optimization."""
    start_time = time.time()
    request_id = secrets.token_hex(8)
    
    try:
        results = []
        
        if request.parallel:
            # Parallel execution with Dask
            with Client(LocalCluster(n_workers=request.max_workers)) as client:
                # Create delayed operations
                delayed_operations = []
                
                for op_request in request.operations:
                    operation = MathOperation(
                        operation_type=OperationType(op_request.operation),
                        operands=op_request.operands,
                        method=CalculationMethod(op_request.method),
                        precision=op_request.precision,
                        metadata=op_request.metadata
                    )
                    
                    # Optimize if enabled
                    if op_request.optimization:
                        optimization_result = await optimization_engine.optimize_operation(
                            operation,
                            context={"strategy": request.optimization_strategy}
                        )
                        if optimization_result.success:
                            operation = optimization_result.optimized_operation
                    
                    # Create delayed computation
                    delayed_op = da.from_delayed(
                        math_service.process_operation(operation),
                        dtype=float
                    )
                    delayed_operations.append(delayed_op)
                
                # Execute all operations
                computed_results = da.compute(*delayed_operations)
                
                # Process results
                for i, result in enumerate(computed_results):
                    results.append(MathResponse(
                        result=result.value,
                        operation=request.operations[i].operation,
                        method=request.operations[i].method,
                        execution_time=result.execution_time,
                        cache_hit=result.cache_hit,
                        success=result.success,
                        error_message=result.error_message,
                        timestamp=datetime.now(),
                        request_id=request_id,
                        metadata={"batch_index": i}
                    ))
        else:
            # Sequential execution
            for i, op_request in enumerate(request.operations):
                operation = MathOperation(
                    operation_type=OperationType(op_request.operation),
                    operands=op_request.operands,
                    method=CalculationMethod(op_request.method),
                    precision=op_request.precision,
                    metadata=op_request.metadata
                )
                
                # Optimize if enabled
                if op_request.optimization:
                    optimization_result = await optimization_engine.optimize_operation(operation)
                    if optimization_result.success:
                        operation = optimization_result.optimized_operation
                
                result = await math_service.process_operation(operation)
                
                results.append(MathResponse(
                    result=result.value,
                    operation=op_request.operation,
                    method=op_request.method,
                    execution_time=result.execution_time,
                    cache_hit=result.cache_hit,
                    success=result.success,
                    error_message=result.error_message,
                    timestamp=datetime.now(),
                    request_id=request_id,
                    metadata={"batch_index": i}
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch operation error: {e}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/execute", response_model=Dict[str, Any])
async def execute_workflow(
    request: WorkflowRequest,
    user_role: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """Execute workflow with advanced orchestration."""
    start_time = time.time()
    request_id = secrets.token_hex(8)
    
    try:
        # Convert steps to WorkflowStep objects
        
        workflow_steps = []
        for step_data in request.steps:
            step = WorkflowStep(
                step_type=WorkflowStepType(step_data["step_type"]),
                name=step_data["name"],
                config=step_data.get("config", {}),
                dependencies=step_data.get("dependencies", [])
            )
            workflow_steps.append(step)
        
        # Execute workflow
        workflow_result = await workflow_engine.execute_workflow(
            request.workflow_name,
            workflow_steps,
            request.variables
        )
        
        execution_time = time.time() - start_time
        
        return {
            "workflow_id": workflow_result.workflow_id,
            "workflow_name": workflow_result.workflow_name,
            "status": workflow_result.status.value,
            "execution_time": execution_time,
            "total_execution_time": workflow_result.total_execution_time,
            "output": workflow_result.variables,
            "step_results": {
                step_id: {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error": result.error_message
                }
                for step_id, result in workflow_result.results.items()
            },
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Workflow execution error: {e}", request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/dashboard", response_model=Dict[str, Any])
async def get_analytics_dashboard(
    user_role: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """Get comprehensive analytics dashboard."""
    try:
        # Get analytics data
        dashboard_data = analytics_engine.get_analytics_dashboard()
        
        # Add real-time metrics
        dashboard_data["real_time"] = {
            "active_connections": ACTIVE_CONNECTIONS._value.get(),
            "cache_hit_rate": CACHE_HIT_RATE._value.get(),
            "optimization_success_rate": OPTIMIZATION_SUCCESS_RATE._value.get(),
            "request_latency_p95": REQUEST_LATENCY.observe(0.95),
            "request_latency_p99": REQUEST_LATENCY.observe(0.99)
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Analytics dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    return StreamingResponse(
        prometheus.generate_latest(),
        media_type="text/plain"
    )


# Enhanced error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Enhanced global exception handler with logging and monitoring."""
    request_id = request.headers.get("X-Request-ID", secrets.token_hex(8))
    
    # Log error with structured logging
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        error=str(exc),
        exc_info=True
    )
    
    # Record error metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=500
    ).inc()
    
    # Return structured error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "details": str(exc) if app.debug else "An unexpected error occurred"
        }
    )


# Additional utility endpoints
@app.post("/cache/clear")
async def clear_cache(
    user_role: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """Clear all caches."""
    try:
        math_service.clear_cache()
        if redis_client:
            await redis_client.flushdb()
        
        return {"message": "All caches cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization/stats", response_model=Dict[str, Any])
async def get_optimization_stats(
    user_role: str = Depends(verify_api_key),
    _: None = Depends(check_rate_limit)
):
    """Get optimization statistics."""
    try:
        return optimization_engine.get_optimization_statistics()
    except Exception as e:
        logger.error(f"Optimization stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 