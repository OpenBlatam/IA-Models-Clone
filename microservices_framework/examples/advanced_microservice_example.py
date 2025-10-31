"""
Advanced Microservice Example
Demonstrates: AI integration, performance optimization, database optimization, intelligent caching
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

# Import our advanced framework components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.core.service_registry import ServiceRegistry, ServiceInstance, ServiceType, ServiceStatus
from shared.core.circuit_breaker import CircuitBreakerConfig, HTTPCircuitBreaker
from shared.monitoring.observability import (
    ObservabilityManager, TracingConfig, MetricsConfig, 
    LoggingConfig, HealthCheckConfig, trace_function, measure_duration
)
from shared.serverless.serverless_adapter import (
    ServerlessConfig, ServerlessPlatform, optimize_for_serverless
)
from shared.ai.ai_integration import (
    AIModelManager, LoadForecastingModel, CacheOptimizationModel, 
    AnomalyDetectionModel, ai_cached
)
from shared.performance.performance_optimizer import (
    PerformanceOptimizer, IntelligentLoadBalancer, AutoScaler,
    monitor_performance
)
from shared.database.database_optimizer import (
    DatabaseOptimizer, DatabaseConfig, DatabaseType, optimized_query
)
from shared.caching.cache_manager import CacheManager, cached
from shared.security.security_manager import SecurityManager, SecurityConfig
from shared.messaging.message_broker import MessageBrokerFactory, BrokerConfig, MessageBrokerType

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

# Pydantic Models
class ProductCreate(BaseModel):
    """Product creation model"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=500)
    price: float = Field(..., gt=0)
    category: str = Field(..., max_length=50)
    stock_quantity: int = Field(..., ge=0)

class ProductUpdate(BaseModel):
    """Product update model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    price: Optional[float] = Field(None, gt=0)
    category: Optional[str] = Field(None, max_length=50)
    stock_quantity: Optional[int] = Field(None, ge=0)

class ProductResponse(BaseModel):
    """Product response model"""
    id: str
    name: str
    description: str
    price: float
    category: str
    stock_quantity: int
    created_at: str
    updated_at: str
    ai_recommendations: Optional[Dict[str, Any]] = None

class OrderCreate(BaseModel):
    """Order creation model"""
    product_id: str
    quantity: int = Field(..., gt=0)
    customer_id: str
    shipping_address: str

class OrderResponse(BaseModel):
    """Order response model"""
    id: str
    product_id: str
    quantity: int
    customer_id: str
    shipping_address: str
    total_price: float
    status: str
    created_at: str
    estimated_delivery: Optional[str] = None

# Global instances
service_registry: Optional[ServiceRegistry] = None
observability_manager: Optional[ObservabilityManager] = None
circuit_breaker: Optional[HTTPCircuitBreaker] = None
ai_model_manager: Optional[AIModelManager] = None
performance_optimizer: Optional[PerformanceOptimizer] = None
database_optimizer: Optional[DatabaseOptimizer] = None
cache_manager: Optional[CacheManager] = None
security_manager: Optional[SecurityManager] = None

# In-memory storage for demo
products_db: Dict[str, Dict[str, Any]] = {}
orders_db: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with all advanced features"""
    global service_registry, observability_manager, circuit_breaker
    global ai_model_manager, performance_optimizer, database_optimizer
    global cache_manager, security_manager
    
    logger.info("Starting Advanced Microservice with AI and Performance Optimization...")
    
    try:
        # Initialize observability
        observability_manager = ObservabilityManager(
            tracing_config=TracingConfig(
                service_name="advanced-product-service",
                service_version="2.0.0",
                enabled=True
            ),
            metrics_config=MetricsConfig(
                enabled=True,
                prometheus_port=8002
            ),
            logging_config=LoggingConfig(
                enabled=True,
                level=structlog.stdlib.INFO
            ),
            health_config=HealthCheckConfig(
                enabled=True,
                endpoint="/health"
            )
        )
        
        await observability_manager.initialize()
        observability_manager.instrument_fastapi(app)
        
        # Initialize service registry
        service_registry = ServiceRegistry("redis://localhost:6379")
        await service_registry.start()
        
        # Register this service
        service_instance = ServiceInstance(
            service_id="advanced-product-service-1",
            service_name="advanced-product-service",
            service_type=ServiceType.API,
            host="localhost",
            port=8002,
            version="2.0.0",
            status=ServiceStatus.HEALTHY,
            health_check_url="http://localhost:8002/health",
            metadata={
                "description": "Advanced product service with AI and performance optimization",
                "version": "2.0.0",
                "environment": "development",
                "features": ["ai", "performance_optimization", "database_optimization"]
            },
            last_heartbeat=time.time(),
            registered_at=time.time()
        )
        
        await service_registry.register_service(service_instance)
        
        # Initialize circuit breaker
        circuit_breaker = HTTPCircuitBreaker(
            "advanced-product-service",
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                timeout=30.0
            )
        )
        
        # Initialize AI models
        ai_model_manager = AIModelManager()
        
        # Initialize performance optimizer
        performance_optimizer = PerformanceOptimizer()
        await performance_optimizer.start_optimization()
        
        # Initialize database optimizer
        db_config = DatabaseConfig(
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="advanced_products",
            username="postgres",
            password="password"
        )
        database_optimizer = DatabaseOptimizer(db_config)
        await database_optimizer.initialize()
        
        # Initialize cache manager
        cache_manager = CacheManager()
        
        # Initialize security manager
        security_config = SecurityConfig(
            jwt_secret="advanced-service-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            rate_limit_enabled=True,
            max_requests_per_minute=100
        )
        security_manager = SecurityManager(security_config)
        
        logger.info("Advanced Microservice started successfully with all features")
        yield
        
    except Exception as e:
        logger.error("Failed to start Advanced Microservice", error=str(e))
        raise
    finally:
        logger.info("Shutting down Advanced Microservice...")
        
        if performance_optimizer:
            await performance_optimizer.stop_optimization()
        
        if database_optimizer:
            await database_optimizer.close()
        
        if service_registry:
            await service_registry.stop()

# Create FastAPI app
app = FastAPI(
    title="Advanced Product Service",
    description="AI-powered microservice with performance optimization and intelligent caching",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_observability() -> ObservabilityManager:
    if not observability_manager:
        raise HTTPException(status_code=503, detail="Observability not available")
    return observability_manager

async def get_ai_models() -> AIModelManager:
    if not ai_model_manager:
        raise HTTPException(status_code=503, detail="AI models not available")
    return ai_model_manager

async def get_performance_optimizer() -> PerformanceOptimizer:
    if not performance_optimizer:
        raise HTTPException(status_code=503, detail="Performance optimizer not available")
    return performance_optimizer

async def get_database_optimizer() -> DatabaseOptimizer:
    if not database_optimizer:
        raise HTTPException(status_code=503, detail="Database optimizer not available")
    return database_optimizer

async def get_cache_manager() -> CacheManager:
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager not available")
    return cache_manager

# Health check endpoints
@app.get("/health")
async def health_check(obs: ObservabilityManager = Depends(get_observability)):
    """Health check endpoint"""
    return await obs.get_health_status()

@app.get("/health/ready")
async def readiness_check(obs: ObservabilityManager = Depends(get_observability)):
    """Readiness check endpoint"""
    return await obs.get_readiness()

@app.get("/health/live")
async def liveness_check(obs: ObservabilityManager = Depends(get_observability)):
    """Liveness check endpoint"""
    return await obs.get_liveness()

@app.get("/metrics")
async def metrics_endpoint(obs: ObservabilityManager = Depends(get_observability)):
    """Prometheus metrics endpoint"""
    return obs.get_metrics()

# AI-powered product recommendations
@app.get("/products/{product_id}/recommendations")
@trace_function("get_ai_recommendations")
@ai_cached("cache_optimization")
async def get_ai_recommendations(
    product_id: str,
    ai_models: AIModelManager = Depends(get_ai_models),
    cache_mgr: CacheManager = Depends(get_cache_manager)
):
    """Get AI-powered product recommendations"""
    try:
        # Check cache first
        cache_key = f"recommendations:{product_id}"
        cached_recommendations = await cache_mgr.get(cache_key)
        
        if cached_recommendations:
            return cached_recommendations
        
        # Get product data
        product = products_db.get(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Use AI model for recommendations
        features = {
            "product_category": product["category"],
            "product_price": product["price"],
            "stock_quantity": product["stock_quantity"],
            "access_frequency": 1,  # Would be tracked in real implementation
            "data_size": len(json.dumps(product))
        }
        
        # Get cache optimization prediction
        cache_prediction = await ai_models.predict("cache_optimization", features)
        
        # Simulate AI recommendations
        recommendations = {
            "similar_products": [
                {"id": f"similar_{i}", "name": f"Similar Product {i}", "price": product["price"] * (0.8 + i * 0.1)}
                for i in range(1, 4)
            ],
            "frequently_bought_together": [
                {"id": f"together_{i}", "name": f"Frequently Bought Together {i}", "price": product["price"] * 0.5}
                for i in range(1, 3)
            ],
            "ai_confidence": cache_prediction.confidence if cache_prediction else 0.8,
            "cache_strategy": cache_prediction.prediction if cache_prediction else {"recommended_ttl": 300}
        }
        
        # Cache recommendations
        await cache_mgr.set(cache_key, recommendations, ttl=300)
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get AI recommendations", product_id=product_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Performance-optimized product creation
@app.post("/products", response_model=ProductResponse)
@trace_function("create_product")
@monitor_performance("create_product")
async def create_product(
    product_data: ProductCreate,
    background_tasks: BackgroundTasks,
    db_optimizer: DatabaseOptimizer = Depends(get_database_optimizer),
    perf_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    ai_models: AIModelManager = Depends(get_ai_models)
):
    """Create a new product with AI and performance optimization"""
    try:
        # Generate product ID
        product_id = f"prod_{int(time.time())}"
        
        # Create product
        product = {
            "id": product_id,
            "name": product_data.name,
            "description": product_data.description,
            "price": product_data.price,
            "category": product_data.category,
            "stock_quantity": product_data.stock_quantity,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Store in database (simulated)
        products_db[product_id] = product
        
        # Background task: Train AI models with new product data
        background_tasks.add_task(
            train_ai_models_with_product,
            product,
            ai_models
        )
        
        # Background task: Update performance metrics
        background_tasks.add_task(
            update_performance_metrics,
            "product_created",
            perf_optimizer
        )
        
        logger.info("Product created successfully", product_id=product_id)
        
        return ProductResponse(**product)
        
    except Exception as e:
        logger.error("Failed to create product", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Database-optimized product retrieval
@app.get("/products/{product_id}", response_model=ProductResponse)
@trace_function("get_product")
@optimized_query(use_cache=True)
async def get_product(
    product_id: str,
    db_optimizer: DatabaseOptimizer = Depends(get_database_optimizer),
    cache_mgr: CacheManager = Depends(get_cache_manager)
):
    """Get product with database optimization and caching"""
    try:
        # Check cache first
        cache_key = f"product:{product_id}"
        cached_product = await cache_mgr.get(cache_key)
        
        if cached_product:
            return ProductResponse(**cached_product)
        
        # Simulate database query
        query = "SELECT * FROM products WHERE id = $1"
        params = {"id": product_id}
        
        # This would use the database optimizer in a real implementation
        # result = await db_optimizer.execute_query(query, params, use_cache=True)
        
        # For demo, get from in-memory storage
        product = products_db.get(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Cache the result
        await cache_mgr.set(cache_key, product, ttl=600)
        
        return ProductResponse(**product)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get product", product_id=product_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# AI-powered order processing
@app.post("/orders", response_model=OrderResponse)
@trace_function("create_order")
async def create_order(
    order_data: OrderCreate,
    background_tasks: BackgroundTasks,
    ai_models: AIModelManager = Depends(get_ai_models),
    perf_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Create order with AI-powered processing"""
    try:
        # Check product availability
        product = products_db.get(order_data.product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        if product["stock_quantity"] < order_data.quantity:
            raise HTTPException(status_code=400, detail="Insufficient stock")
        
        # Generate order ID
        order_id = f"order_{int(time.time())}"
        
        # Calculate total price
        total_price = product["price"] * order_data.quantity
        
        # Use AI for delivery estimation
        delivery_features = {
            "product_category": product["category"],
            "quantity": order_data.quantity,
            "current_load": 50,  # Would be real-time data
            "historical_average": 3.5,
            "trend": 0.1
        }
        
        load_prediction = await ai_models.predict("load_forecasting", delivery_features)
        estimated_days = 3  # Default
        if load_prediction and load_prediction.prediction:
            estimated_days = max(1, min(7, int(load_prediction.prediction)))
        
        # Create order
        order = {
            "id": order_id,
            "product_id": order_data.product_id,
            "quantity": order_data.quantity,
            "customer_id": order_data.customer_id,
            "shipping_address": order_data.shipping_address,
            "total_price": total_price,
            "status": "processing",
            "created_at": time.time(),
            "estimated_delivery": f"{estimated_days} days"
        }
        
        # Store order
        orders_db[order_id] = order
        
        # Update product stock
        products_db[order_data.product_id]["stock_quantity"] -= order_data.quantity
        products_db[order_data.product_id]["updated_at"] = time.time()
        
        # Background task: Process order with AI
        background_tasks.add_task(
            process_order_with_ai,
            order,
            ai_models
        )
        
        # Background task: Update performance metrics
        background_tasks.add_task(
            update_performance_metrics,
            "order_created",
            perf_optimizer
        )
        
        logger.info("Order created successfully", order_id=order_id)
        
        return OrderResponse(**order)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create order", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Performance monitoring endpoint
@app.get("/performance/stats")
async def get_performance_stats(
    perf_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """Get performance optimization statistics"""
    return await perf_optimizer.get_optimization_stats()

# AI model management endpoint
@app.get("/ai/models/stats")
async def get_ai_model_stats(
    ai_models: AIModelManager = Depends(get_ai_models)
):
    """Get AI model statistics"""
    return await ai_models.get_model_metrics()

# Database optimization stats
@app.get("/database/stats")
async def get_database_stats(
    db_optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Get database optimization statistics"""
    return await db_optimizer.get_optimization_stats()

# Background task functions
async def train_ai_models_with_product(product: Dict[str, Any], ai_models: AIModelManager):
    """Train AI models with new product data"""
    try:
        # Simulate training data
        training_data = [{
            "product_category": product["category"],
            "product_price": product["price"],
            "stock_quantity": product["stock_quantity"],
            "target_load": 1.0  # New product gets some load
        }]
        
        # Train load forecasting model
        result = await ai_models.train_model("load_forecasting", training_data)
        logger.info("AI models trained with new product", result=result)
        
    except Exception as e:
        logger.error("Failed to train AI models", error=str(e))

async def update_performance_metrics(action: str, perf_optimizer: PerformanceOptimizer):
    """Update performance metrics"""
    try:
        # This would update real performance metrics
        logger.info("Performance metrics updated", action=action)
        
    except Exception as e:
        logger.error("Failed to update performance metrics", error=str(e))

async def process_order_with_ai(order: Dict[str, Any], ai_models: AIModelManager):
    """Process order with AI assistance"""
    try:
        # Simulate order processing
        await asyncio.sleep(1)
        
        # Use AI for anomaly detection
        order_features = {
            "response_time": 100,
            "error_rate": 0,
            "cpu_usage": 30,
            "memory_usage": 40,
            "request_count": 1,
            "service_name": "order_processing"
        }
        
        anomaly_prediction = await ai_models.predict("anomaly_detection", order_features)
        
        if anomaly_prediction and anomaly_prediction.prediction.get("is_anomaly"):
            logger.warning("Anomaly detected in order processing", 
                         order_id=order["id"], 
                         anomaly_score=anomaly_prediction.prediction.get("anomaly_score"))
        
        # Update order status
        orders_db[order["id"]]["status"] = "completed"
        
        logger.info("Order processed with AI", order_id=order["id"])
        
    except Exception as e:
        logger.error("Failed to process order with AI", error=str(e))

# Service discovery endpoint
@app.get("/service/info")
async def service_info():
    """Get service information"""
    return {
        "service_name": "advanced-product-service",
        "version": "2.0.0",
        "status": "healthy",
        "features": [
            "ai_integration",
            "performance_optimization", 
            "database_optimization",
            "intelligent_caching",
            "circuit_breaker",
            "observability",
            "security"
        ],
        "endpoints": [
            "/products",
            "/products/{product_id}",
            "/products/{product_id}/recommendations",
            "/orders",
            "/performance/stats",
            "/ai/models/stats",
            "/database/stats",
            "/health"
        ],
        "ai_models": [
            "load_forecasting",
            "cache_optimization", 
            "anomaly_detection"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create serverless adapter for deployment flexibility
    serverless_config = ServerlessConfig(
        platform=ServerlessPlatform.AWS_LAMBDA,
        cold_start_timeout=10.0,
        enable_compression=True,
        enable_caching=True
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info",
        access_log=True
    )






























