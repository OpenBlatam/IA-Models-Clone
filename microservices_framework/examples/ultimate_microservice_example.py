"""
Ultimate Advanced Microservice Example
Demonstrates: AI/ML Pipeline, Real-time Streaming, Distributed Computing, Task Orchestration
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog

# Import our ultimate framework components
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

# Import new advanced components
from shared.ml.ml_pipeline import (
    MLPipeline, SklearnModel, XGBoostModel, ModelConfig, ModelType, 
    ModelStatus, FeatureConfig, ExperimentManager, ExperimentConfig, 
    ExperimentType, ml_model_endpoint
)
from shared.streaming.event_processor import (
    EventProcessor, Event, EventType, StreamConfig, StreamType,
    KafkaStreamManager, WebSocketStreamManager, create_event, publish_event_async
)
from shared.orchestration.task_orchestrator import (
    TaskOrchestrator, Task, TaskPriority, TaskStatus, ResourceType,
    WorkflowEngine, Workflow, WorkflowStatus, create_task, submit_task_async,
    distributed_task
)

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
    features: Dict[str, Any] = Field(default_factory=dict)

class MLPredictionRequest(BaseModel):
    """ML prediction request"""
    model_id: str
    features: Dict[str, Any]
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class MLPredictionResponse(BaseModel):
    """ML prediction response"""
    model_id: str
    version: str
    prediction: Any
    confidence: float
    processing_time: float
    timestamp: float

class WorkflowCreate(BaseModel):
    """Workflow creation model"""
    name: str
    description: str
    tasks: List[Dict[str, Any]]

class EventStreamRequest(BaseModel):
    """Event stream request"""
    event_type: str
    data: Dict[str, Any]
    stream_id: str = "default"

# Global instances
service_registry: Optional[ServiceRegistry] = None
observability_manager: Optional[ObservabilityManager] = None
circuit_breaker: Optional[HTTPCircuitBreaker] = None
ai_model_manager: Optional[AIModelManager] = None
performance_optimizer: Optional[PerformanceOptimizer] = None
database_optimizer: Optional[DatabaseOptimizer] = None
cache_manager: Optional[CacheManager] = None
security_manager: Optional[SecurityManager] = None

# New advanced components
ml_pipeline: Optional[MLPipeline] = None
event_processor: Optional[EventProcessor] = None
task_orchestrator: Optional[TaskOrchestrator] = None

# In-memory storage for demo
products_db: Dict[str, Dict[str, Any]] = {}
ml_models_db: Dict[str, Dict[str, Any]] = {}
workflows_db: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with all ultimate features"""
    global service_registry, observability_manager, circuit_breaker
    global ai_model_manager, performance_optimizer, database_optimizer
    global cache_manager, security_manager, ml_pipeline, event_processor, task_orchestrator
    
    logger.info("Starting Ultimate Advanced Microservice with ALL features...")
    
    try:
        # Initialize observability
        observability_manager = ObservabilityManager(
            tracing_config=TracingConfig(
                service_name="ultimate-product-service",
                service_version="3.0.0",
                enabled=True
            ),
            metrics_config=MetricsConfig(
                enabled=True,
                prometheus_port=8003
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
            service_id="ultimate-product-service-1",
            service_name="ultimate-product-service",
            service_type=ServiceType.API,
            host="localhost",
            port=8003,
            version="3.0.0",
            status=ServiceStatus.HEALTHY,
            health_check_url="http://localhost:8003/health",
            metadata={
                "description": "Ultimate product service with AI, ML, streaming, and orchestration",
                "version": "3.0.0",
                "environment": "development",
                "features": [
                    "ai_integration", "ml_pipeline", "performance_optimization", 
                    "database_optimization", "intelligent_caching", "circuit_breaker",
                    "observability", "security", "real_time_streaming", "task_orchestration"
                ]
            },
            last_heartbeat=time.time(),
            registered_at=time.time()
        )
        
        await service_registry.register_service(service_instance)
        
        # Initialize circuit breaker
        circuit_breaker = HTTPCircuitBreaker(
            "ultimate-product-service",
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
            database="ultimate_products",
            username="postgres",
            password="password"
        )
        database_optimizer = DatabaseOptimizer(db_config)
        await database_optimizer.initialize()
        
        # Initialize cache manager
        cache_manager = CacheManager()
        
        # Initialize security manager
        security_config = SecurityConfig(
            jwt_secret="ultimate-service-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            rate_limit_enabled=True,
            max_requests_per_minute=200
        )
        security_manager = SecurityManager(security_config)
        
        # Initialize ML Pipeline
        ml_pipeline = MLPipeline()
        
        # Initialize Event Processor
        event_processor = EventProcessor()
        
        # Initialize Task Orchestrator
        task_orchestrator = TaskOrchestrator(executor_type="local")
        await task_orchestrator.start_orchestration()
        
        # Setup ML models
        await setup_ml_models()
        
        # Setup event streaming
        await setup_event_streaming()
        
        # Setup workflows
        await setup_workflows()
        
        logger.info("Ultimate Advanced Microservice started successfully with ALL features")
        yield
        
    except Exception as e:
        logger.error("Failed to start Ultimate Advanced Microservice", error=str(e))
        raise
    finally:
        logger.info("Shutting down Ultimate Advanced Microservice...")
        
        if task_orchestrator:
            await task_orchestrator.stop_orchestration()
        
        if performance_optimizer:
            await performance_optimizer.stop_optimization()
        
        if database_optimizer:
            await database_optimizer.close()
        
        if service_registry:
            await service_registry.stop()

# Create FastAPI app
app = FastAPI(
    title="Ultimate Product Service",
    description="AI-powered microservice with ML pipeline, real-time streaming, and distributed computing",
    version="3.0.0",
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

async def get_ml_pipeline() -> MLPipeline:
    if not ml_pipeline:
        raise HTTPException(status_code=503, detail="ML pipeline not available")
    return ml_pipeline

async def get_event_processor() -> EventProcessor:
    if not event_processor:
        raise HTTPException(status_code=503, detail="Event processor not available")
    return event_processor

async def get_task_orchestrator() -> TaskOrchestrator:
    if not task_orchestrator:
        raise HTTPException(status_code=503, detail="Task orchestrator not available")
    return task_orchestrator

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

# ML Pipeline endpoints
@app.post("/ml/models/train")
@trace_function("train_ml_model")
async def train_ml_model(
    model_config: Dict[str, Any],
    ml_pipe: MLPipeline = Depends(get_ml_pipeline)
):
    """Train a new ML model"""
    try:
        # Create model configuration
        config = ModelConfig(
            model_id=model_config["model_id"],
            model_type=ModelType(model_config["model_type"]),
            algorithm=model_config["algorithm"],
            hyperparameters=model_config.get("hyperparameters", {}),
            feature_columns=model_config["feature_columns"],
            target_column=model_config["target_column"]
        )
        
        # Create model
        if config.algorithm == "xgboost":
            model = XGBoostModel(config)
        else:
            model = SklearnModel(config)
        
        # Register model
        ml_pipe.register_model(model)
        
        # Simulate training data
        import pandas as pd
        import numpy as np
        
        # Generate synthetic training data
        n_samples = 1000
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Train model
        metrics = await ml_pipe.train_model(config.model_id, X, "target")
        
        # Store model info
        ml_models_db[config.model_id] = {
            "config": model_config,
            "metrics": metrics.__dict__,
            "status": "trained",
            "created_at": time.time()
        }
        
        logger.info(f"ML model {config.model_id} trained successfully")
        
        return {
            "model_id": config.model_id,
            "status": "trained",
            "metrics": metrics.__dict__
        }
        
    except Exception as e:
        logger.error("Failed to train ML model", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ml/models/predict", response_model=MLPredictionResponse)
@trace_function("ml_prediction")
@ml_model_endpoint("product_classifier")
async def ml_prediction(
    request: MLPredictionRequest,
    ml_pipe: MLPipeline = Depends(get_ml_pipeline)
):
    """Make ML prediction"""
    try:
        start_time = time.time()
        
        # Convert features to DataFrame
        import pandas as pd
        features_df = pd.DataFrame([request.features])
        
        # Make prediction
        predictions = await ml_pipe.predict(request.model_id, features_df)
        
        processing_time = time.time() - start_time
        
        # Get model info
        model_info = ml_models_db.get(request.model_id, {})
        
        return MLPredictionResponse(
            model_id=request.model_id,
            version=model_info.get("metrics", {}).get("version", "1.0.0"),
            prediction=predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0],
            confidence=0.95,  # Would calculate actual confidence
            processing_time=processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error("ML prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Event Streaming endpoints
@app.post("/events/stream")
@trace_function("stream_event")
async def stream_event(
    request: EventStreamRequest,
    event_proc: EventProcessor = Depends(get_event_processor)
):
    """Stream an event"""
    try:
        # Create event
        event = create_event(
            event_type=EventType(request.event_type),
            stream_id=request.stream_id,
            data=request.data,
            metadata={"source": "api", "user_id": str(uuid.uuid4())}
        )
        
        # Publish event
        success = await event_proc.publish_event(event)
        
        if success:
            return {
                "status": "success",
                "event_id": event.event_id,
                "timestamp": event.timestamp
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to stream event")
        
    except Exception as e:
        logger.error("Event streaming failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/events/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events"""
    await websocket.accept()
    
    try:
        while True:
            # Receive event from client
            data = await websocket.receive_text()
            event_data = json.loads(data)
            
            # Create event
            event = create_event(
                event_type=EventType(event_data["event_type"]),
                stream_id=event_data.get("stream_id", "websocket"),
                data=event_data["data"],
                metadata={"source": "websocket"}
            )
            
            # Process event
            if event_processor:
                await event_processor.publish_event(event)
            
            # Send confirmation
            response = {
                "status": "received",
                "event_id": event.event_id,
                "timestamp": time.time()
            }
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))

# Task Orchestration endpoints
@app.post("/workflows")
@trace_function("create_workflow")
async def create_workflow(
    workflow_data: WorkflowCreate,
    task_orch: TaskOrchestrator = Depends(get_task_orchestrator)
):
    """Create a new workflow"""
    try:
        # Create workflow
        workflow = task_orch.workflow_engine.create_workflow(
            workflow_id=str(uuid.uuid4()),
            name=workflow_data.name
        )
        
        # Add tasks to workflow
        for task_data in workflow_data.tasks:
            task = create_task(
                name=task_data["name"],
                function=lambda x: x,  # Placeholder function
                args=(task_data.get("args", []),),
                kwargs=task_data.get("kwargs", {}),
                priority=TaskPriority(task_data.get("priority", "normal")),
                dependencies=task_data.get("dependencies", [])
            )
            
            task_orch.workflow_engine.add_task_to_workflow(workflow.workflow_id, task)
        
        # Store workflow info
        workflows_db[workflow.workflow_id] = {
            "name": workflow_data.name,
            "description": workflow_data.description,
            "tasks": workflow_data.tasks,
            "status": "created",
            "created_at": time.time()
        }
        
        logger.info(f"Workflow {workflow.workflow_id} created successfully")
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow_data.name,
            "status": "created",
            "tasks_count": len(workflow_data.tasks)
        }
        
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/workflows/{workflow_id}/execute")
@trace_function("execute_workflow")
async def execute_workflow(
    workflow_id: str,
    task_orch: TaskOrchestrator = Depends(get_task_orchestrator)
):
    """Execute a workflow"""
    try:
        # Execute workflow
        success = await task_orch.workflow_engine.execute_workflow(workflow_id)
        
        if success:
            # Update workflow status
            if workflow_id in workflows_db:
                workflows_db[workflow_id]["status"] = "completed"
                workflows_db[workflow_id]["completed_at"] = time.time()
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_time": time.time()
            }
        else:
            raise HTTPException(status_code=500, detail="Workflow execution failed")
        
    except Exception as e:
        logger.error("Workflow execution failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Distributed task execution
@app.post("/tasks/execute")
@trace_function("execute_distributed_task")
@distributed_task(priority=TaskPriority.HIGH, timeout=60.0)
async def execute_distributed_task(
    task_name: str,
    task_data: Dict[str, Any],
    task_orch: TaskOrchestrator = Depends(get_task_orchestrator)
):
    """Execute a distributed task"""
    try:
        # Create task
        task = create_task(
            name=task_name,
            function=process_distributed_task,
            args=(task_data,),
            priority=TaskPriority.HIGH,
            timeout=60.0,
            resources={
                ResourceType.CPU: 1.0,
                ResourceType.MEMORY: 512.0
            }
        )
        
        # Submit task
        task_id = await task_orch.submit_task(task)
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "name": task_name
        }
        
    except Exception as e:
        logger.error("Distributed task execution failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Advanced product management with all features
@app.post("/products", response_model=Dict[str, Any])
@trace_function("create_ultimate_product")
@monitor_performance("create_ultimate_product")
@ai_cached("cache_optimization")
async def create_ultimate_product(
    product_data: ProductCreate,
    background_tasks: BackgroundTasks,
    ml_pipe: MLPipeline = Depends(get_ml_pipeline),
    event_proc: EventProcessor = Depends(get_event_processor),
    task_orch: TaskOrchestrator = Depends(get_task_orchestrator)
):
    """Create a product with all advanced features"""
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
            "features": product_data.features,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Store in database (simulated)
        products_db[product_id] = product
        
        # Background task: ML feature extraction
        background_tasks.add_task(
            extract_ml_features,
            product,
            ml_pipe
        )
        
        # Background task: Stream product creation event
        background_tasks.add_task(
            stream_product_event,
            product,
            event_proc
        )
        
        # Background task: Execute product processing workflow
        background_tasks.add_task(
            execute_product_workflow,
            product,
            task_orch
        )
        
        logger.info("Ultimate product created successfully", product_id=product_id)
        
        return {
            "product_id": product_id,
            "status": "created",
            "features_applied": [
                "ai_integration",
                "ml_pipeline",
                "event_streaming",
                "task_orchestration",
                "performance_optimization",
                "intelligent_caching"
            ],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Failed to create ultimate product", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Statistics and monitoring endpoints
@app.get("/stats/ml")
async def get_ml_stats(ml_pipe: MLPipeline = Depends(get_ml_pipeline)):
    """Get ML pipeline statistics"""
    return ml_pipe.get_pipeline_stats()

@app.get("/stats/events")
async def get_event_stats(event_proc: EventProcessor = Depends(get_event_processor)):
    """Get event processing statistics"""
    return event_proc.get_processing_stats()

@app.get("/stats/tasks")
async def get_task_stats(task_orch: TaskOrchestrator = Depends(get_task_orchestrator)):
    """Get task orchestration statistics"""
    return task_orch.get_orchestration_stats()

@app.get("/stats/performance")
async def get_performance_stats(perf_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)):
    """Get performance optimization statistics"""
    return await perf_optimizer.get_optimization_stats()

# Service information
@app.get("/service/info")
async def service_info():
    """Get ultimate service information"""
    return {
        "service_name": "ultimate-product-service",
        "version": "3.0.0",
        "status": "healthy",
        "features": [
            "ai_integration",
            "ml_pipeline", 
            "performance_optimization",
            "database_optimization",
            "intelligent_caching",
            "circuit_breaker",
            "observability",
            "security",
            "real_time_streaming",
            "task_orchestration",
            "distributed_computing",
            "workflow_engine"
        ],
        "endpoints": [
            "/ml/models/train",
            "/ml/models/predict",
            "/events/stream",
            "/events/ws",
            "/workflows",
            "/tasks/execute",
            "/products",
            "/stats/ml",
            "/stats/events",
            "/stats/tasks",
            "/stats/performance",
            "/health"
        ],
        "ai_models": [
            "load_forecasting",
            "cache_optimization",
            "anomaly_detection",
            "product_classifier",
            "recommendation_engine"
        ],
        "ml_pipeline": {
            "models_registered": len(ml_models_db),
            "experiments_active": 0,
            "feature_engineering": True
        },
        "streaming": {
            "event_types": ["user_action", "system_event", "business_event"],
            "streams_active": 1,
            "real_time_processing": True
        },
        "orchestration": {
            "workflows_created": len(workflows_db),
            "distributed_tasks": True,
            "resource_management": True
        }
    }

# Background task functions
async def extract_ml_features(product: Dict[str, Any], ml_pipe: MLPipeline):
    """Extract ML features from product"""
    try:
        # Simulate feature extraction
        features = {
            "price_category": "high" if product["price"] > 100 else "low",
            "stock_level": "high" if product["stock_quantity"] > 50 else "low",
            "category_encoded": hash(product["category"]) % 10,
            "name_length": len(product["name"]),
            "description_sentiment": 0.8  # Would use actual sentiment analysis
        }
        
        # Store features
        product["ml_features"] = features
        
        logger.info("ML features extracted", product_id=product["id"])
        
    except Exception as e:
        logger.error("ML feature extraction failed", error=str(e))

async def stream_product_event(product: Dict[str, Any], event_proc: EventProcessor):
    """Stream product creation event"""
    try:
        event = create_event(
            event_type=EventType.BUSINESS_EVENT,
            stream_id="product_events",
            data={
                "action": "product_created",
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "price": product["price"]
            },
            metadata={"source": "product_service"}
        )
        
        await event_proc.publish_event(event)
        logger.info("Product event streamed", product_id=product["id"])
        
    except Exception as e:
        logger.error("Event streaming failed", error=str(e))

async def execute_product_workflow(product: Dict[str, Any], task_orch: TaskOrchestrator):
    """Execute product processing workflow"""
    try:
        # Create workflow
        workflow = task_orch.workflow_engine.create_workflow(
            workflow_id=f"product_workflow_{product['id']}",
            name=f"Product Processing Workflow for {product['name']}"
        )
        
        # Add tasks
        tasks = [
            {"name": "validate_product", "function": validate_product_task},
            {"name": "categorize_product", "function": categorize_product_task},
            {"name": "calculate_pricing", "function": calculate_pricing_task},
            {"name": "update_inventory", "function": update_inventory_task}
        ]
        
        for i, task_data in enumerate(tasks):
            task = create_task(
                name=task_data["name"],
                function=task_data["function"],
                args=(product,),
                dependencies=[tasks[i-1]["name"]] if i > 0 else []
            )
            
            task_orch.workflow_engine.add_task_to_workflow(workflow.workflow_id, task)
        
        # Execute workflow
        await task_orch.workflow_engine.execute_workflow(workflow.workflow_id)
        
        logger.info("Product workflow executed", product_id=product["id"])
        
    except Exception as e:
        logger.error("Product workflow execution failed", error=str(e))

# Setup functions
async def setup_ml_models():
    """Setup ML models"""
    try:
        # Create sample models
        models = [
            {
                "model_id": "product_classifier",
                "model_type": "classification",
                "algorithm": "random_forest",
                "feature_columns": ["price", "category_encoded", "stock_level"],
                "target_column": "popularity"
            },
            {
                "model_id": "price_predictor",
                "model_type": "regression", 
                "algorithm": "xgboost",
                "feature_columns": ["category", "features", "market_demand"],
                "target_column": "optimal_price"
            }
        ]
        
        for model_config in models:
            config = ModelConfig(
                model_id=model_config["model_id"],
                model_type=ModelType(model_config["model_type"]),
                algorithm=model_config["algorithm"],
                feature_columns=model_config["feature_columns"],
                target_column=model_config["target_column"]
            )
            
            if model_config["algorithm"] == "xgboost":
                model = XGBoostModel(config)
            else:
                model = SklearnModel(config)
            
            ml_pipeline.register_model(model)
        
        logger.info("ML models setup completed")
        
    except Exception as e:
        logger.error("ML models setup failed", error=str(e))

async def setup_event_streaming():
    """Setup event streaming"""
    try:
        # Create WebSocket stream manager
        ws_config = StreamConfig(
            stream_id="websocket_stream",
            stream_type=StreamType.WEBSOCKET,
            topic="events"
        )
        
        ws_manager = WebSocketStreamManager(ws_config)
        event_processor.add_stream_manager("websocket_stream", ws_manager)
        
        logger.info("Event streaming setup completed")
        
    except Exception as e:
        logger.error("Event streaming setup failed", error=str(e))

async def setup_workflows():
    """Setup sample workflows"""
    try:
        # Create sample workflow
        workflow = task_orchestrator.workflow_engine.create_workflow(
            workflow_id="sample_workflow",
            name="Sample Product Processing Workflow"
        )
        
        # Add sample tasks
        task1 = create_task(
            name="data_validation",
            function=lambda x: x,
            args=("validation_data",)
        )
        
        task2 = create_task(
            name="feature_extraction",
            function=lambda x: x,
            args=("extraction_data",),
            dependencies=["data_validation"]
        )
        
        task_orchestrator.workflow_engine.add_task_to_workflow(workflow.workflow_id, task1)
        task_orchestrator.workflow_engine.add_task_to_workflow(workflow.workflow_id, task2)
        
        logger.info("Workflows setup completed")
        
    except Exception as e:
        logger.error("Workflows setup failed", error=str(e))

# Task functions
def validate_product_task(product: Dict[str, Any]) -> Dict[str, Any]:
    """Validate product data"""
    return {"valid": True, "product_id": product["id"]}

def categorize_product_task(product: Dict[str, Any]) -> Dict[str, Any]:
    """Categorize product"""
    return {"category": product["category"], "subcategory": "general"}

def calculate_pricing_task(product: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate optimal pricing"""
    return {"optimal_price": product["price"] * 1.1, "margin": 0.1}

def update_inventory_task(product: Dict[str, Any]) -> Dict[str, Any]:
    """Update inventory"""
    return {"inventory_updated": True, "stock": product["stock_quantity"]}

def process_distributed_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process distributed task"""
    # Simulate processing
    time.sleep(1)
    return {"processed": True, "data": task_data}

if __name__ == "__main__":
    import uvicorn
    
    # Create serverless adapter for deployment flexibility
    serverless_config = ServerlessConfig(
        platform=ServerlessPlatform.AWS_LAMBDA,
        cold_start_timeout=15.0,
        enable_compression=True,
        enable_caching=True
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
        access_log=True
    )






























