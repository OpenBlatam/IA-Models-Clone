"""
REST API Module - FastAPI service for the benchmark system.

Enhancements over the original implementation:
- Centralized FastAPI app factory with typed settings.
- Standardized API responses with metadata and error payloads.
- Dependency-based authentication/authorization hooks.
- WebSocket connection manager with structured events.
- Unified exception handling and logging utilities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

from api.metrics_endpoint import router as metrics_router
from api.webhooks import WebhookEvent, WebhookManager
from core.cost_tracking import CostTracker
from core.distributed import DistributedExecutor
from core.experiments import ExperimentConfig, ExperimentManager
from core.model_registry import ModelMetadata, ModelRegistry
from core.results import BenchmarkResult, ResultsManager
from .routers import ALL_ROUTERS
from .middleware import setup_middleware

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration & helpers
# -----------------------------------------------------------------------------


@dataclass
class APISettings:
    """Runtime configuration for the API service."""

    title: str = "Universal Model Benchmark AI API"
    description: str = "REST API for model benchmarking and orchestration"
    version: str = "1.1.0"
    allowed_origins: Optional[List[str]] = None
    enable_docs: bool = True
    strict_auth: bool = False


class ApiResponse(BaseModel):
    """Standard API response envelope."""

    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standardized error payload."""

    error: str
    detail: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


def create_app(settings: Optional[APISettings] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = settings or APISettings()
    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
    )

    cors_origins = settings.allowed_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup custom middleware
    setup_middleware(app)
    
    register_dependencies(app, settings)
    register_routes(app)
    register_exception_handlers(app)
    return app


# -----------------------------------------------------------------------------
# Dependencies & shared managers
# -----------------------------------------------------------------------------


security = HTTPBearer(auto_error=False)

results_manager = ResultsManager()
experiment_manager = ExperimentManager()
model_registry = ModelRegistry()
distributed_executor = DistributedExecutor()
cost_tracker = CostTracker()
webhook_manager = WebhookManager()


def register_dependencies(app: FastAPI, settings: APISettings) -> None:
    """Attach dependencies to the application instance."""

    async def verify_token(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> str:
        """Validate bearer token or enforce authentication when strict mode is enabled."""

        if not credentials:
            if settings.strict_auth:
                raise HTTPException(status_code=401, detail="Missing authentication token")
            return "anonymous"

        token = credentials.credentials
        if not token:
            raise HTTPException(status_code=401, detail="Invalid token")
        return token

    app.dependency_overrides[verify_token] = verify_token
    app.state.verify_token = verify_token


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent error responses."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
        payload = ErrorResponse(error=exc.detail or "HTTP error", detail={"status": exc.status_code})
        return Response(
            content=payload.json(),
            media_type="application/json",
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        logger.exception("Unhandled exception: %s", exc)
        payload = ErrorResponse(error="Internal server error")
        return Response(content=payload.json(), media_type="application/json", status_code=500)


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    model_name: str
    benchmark_name: str
    config: Optional[Dict[str, Any]] = None


class ExperimentRequest(BaseModel):
    name: str
    description: str = ""
    model_name: str
    benchmark_name: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ModelRegisterRequest(BaseModel):
    name: str
    version: str
    description: str = ""
    architecture: str = ""
    parameters: int = 0
    path: str
    tags: List[str] = Field(default_factory=list)


class WebhookRequest(BaseModel):
    url: str
    events: List[str]
    secret: Optional[str] = None

    @validator("events")
    def validate_events(cls, events: List[str]) -> List[str]:  # noqa: N805
        if not events:
            raise ValueError("At least one event must be provided")
        return events


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def success_response(data: Any, **metadata: Any) -> ApiResponse:
    """Wrap payloads in the standard response envelope."""

    return ApiResponse(data=data, metadata={"timestamp": datetime.now().isoformat(), **metadata})


def register_routes(app: FastAPI) -> None:
    """Register HTTP and WebSocket routes."""
    
    # Include modular routers
    for router in ALL_ROUTERS:
        app.include_router(router)
    
    # Include metrics router
    app.include_router(metrics_router)

    @app.get("/api/v1/results", response_model=ApiResponse)
    async def get_results(
        model_name: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        limit: int = 100,
        token: str = Depends(app.state.verify_token),
    ):
        results = results_manager.get_results(
            model_name=model_name,
            benchmark_name=benchmark_name,
            limit=limit,
        )
        return success_response([r.to_dict() for r in results], count=len(results), user=token)

    @app.get("/api/v1/results/{result_id}", response_model=ApiResponse)
    async def get_result(result_id: str, token: str = Depends(app.state.verify_token)):
        results = results_manager.get_results()
        result = next((r for r in results if r.benchmark_name == result_id), None)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return success_response(result.to_dict(), user=token)

    @app.post("/api/v1/results", response_model=ApiResponse)
    async def create_result(result: BenchmarkResult, token: str = Depends(app.state.verify_token)):
        results_manager.save_result(result)
        return success_response({"id": result.benchmark_name, "status": "created"}, user=token)

    @app.get("/api/v1/results/comparison/{benchmark_name}", response_model=ApiResponse)
    async def get_comparison(
        benchmark_name: str,
        model_names: Optional[List[str]] = None,
        token: str = Depends(app.state.verify_token),
    ):
        comparison = results_manager.get_comparison(
            benchmark_name=benchmark_name,
            model_names=model_names,
        )
        return success_response(comparison.__dict__, user=token)

    @app.get("/api/v1/experiments", response_model=ApiResponse)
    async def list_experiments(
        status: Optional[str] = None,
        token: str = Depends(app.state.verify_token),
    ):
        experiments = experiment_manager.list_experiments()
        if status:
            experiments = [e for e in experiments if e.status.value == status]
        return success_response([e.to_dict() for e in experiments], count=len(experiments), user=token)

    @app.post("/api/v1/experiments", response_model=ApiResponse)
    async def create_experiment(request: ExperimentRequest, token: str = Depends(app.state.verify_token)):
        config = ExperimentConfig(
            name=request.name,
            description=request.description,
            model_name=request.model_name,
            benchmark_name=request.benchmark_name,
            hyperparameters=request.hyperparameters,
            tags=request.tags,
        )
        experiment = experiment_manager.create_experiment(config)
        return success_response(experiment.to_dict(), user=token)

    @app.get("/api/v1/experiments/{experiment_id}", response_model=ApiResponse)
    async def get_experiment(experiment_id: str, token: str = Depends(app.state.verify_token)):
        experiment = experiment_manager.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return success_response(experiment.to_dict(), user=token)

    @app.post("/api/v1/experiments/{experiment_id}/start", response_model=ApiResponse)
    async def start_experiment(experiment_id: str, token: str = Depends(app.state.verify_token)):
        experiment = experiment_manager.start_experiment(experiment_id)
        await webhook_manager.trigger_event(
            WebhookEvent.EXPERIMENT_STARTED,
            experiment.to_dict(),
        )
        return success_response(experiment.to_dict(), user=token)

    @app.post("/api/v1/experiments/{experiment_id}/complete", response_model=ApiResponse)
    async def complete_experiment(
        experiment_id: str,
        results: Dict[str, Any],
        token: str = Depends(app.state.verify_token),
    ):
        experiment = experiment_manager.complete_experiment(experiment_id, results)
        await webhook_manager.trigger_event(
            WebhookEvent.EXPERIMENT_COMPLETED,
            experiment.to_dict(),
        )
        return success_response(experiment.to_dict(), user=token)

    @app.get("/api/v1/models", response_model=ApiResponse)
    async def list_models(status: Optional[str] = None, token: str = Depends(app.state.verify_token)):
        models = model_registry.list_models()
        if status:
            models = [m for m in models if m.status.value == status]
        return success_response([m.to_dict() for m in models], count=len(models), user=token)

    @app.post("/api/v1/models", response_model=ApiResponse)
    async def register_model(request: ModelRegisterRequest, token: str = Depends(app.state.verify_token)):
        metadata = ModelMetadata(
            name=request.name,
            version=request.version,
            description=request.description,
            architecture=request.architecture,
            parameters=request.parameters,
            tags=request.tags,
        )
        version = model_registry.register_model(metadata, request.path)
        await webhook_manager.trigger_event(WebhookEvent.MODEL_REGISTERED, version.to_dict())
        return success_response(version.to_dict(), user=token)

    @app.get("/api/v1/models/{model_name}", response_model=ApiResponse)
    async def get_model(model_name: str, version: Optional[str] = None, token: str = Depends(app.state.verify_token)):
        model = model_registry.get_model(model_name, version)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return success_response(model.to_dict(), user=token)

    @app.get("/api/v1/models/best/{benchmark_name}", response_model=ApiResponse)
    async def get_best_models(
        benchmark_name: str,
        top_k: int = 5,
        token: str = Depends(app.state.verify_token),
    ):
        models = model_registry.get_best_models(benchmark_name, top_k)
        return success_response([m.to_dict() for m in models], count=len(models), user=token)

    @app.get("/api/v1/nodes", response_model=ApiResponse)
    async def list_nodes(token: str = Depends(app.state.verify_token)):
        nodes = list(distributed_executor.nodes.values())
        return success_response([n.to_dict() for n in nodes], count=len(nodes), user=token)

    @app.post("/api/v1/nodes", response_model=ApiResponse)
    async def register_node(node_id: str, host: str, port: int, token: str = Depends(app.state.verify_token)):
        node = distributed_executor.register_node(node_id, host, port)
        return success_response(node.to_dict(), user=token)

    @app.get("/api/v1/tasks", response_model=ApiResponse)
    async def list_tasks(node_id: Optional[str] = None, token: str = Depends(app.state.verify_token)):
        if node_id:
            tasks = distributed_executor.get_node_tasks(node_id)
        else:
            tasks = list(distributed_executor.tasks.values())
        return success_response([t.to_dict() for t in tasks], count=len(tasks), user=token)

    @app.get("/api/v1/costs", response_model=ApiResponse)
    async def get_costs(
        model_name: Optional[str] = None,
        benchmark_name: Optional[str] = None,
        token: str = Depends(app.state.verify_token),
    ):
        total = cost_tracker.get_total_cost(model_name, benchmark_name)
        breakdown = cost_tracker.get_cost_breakdown(model_name)
        budget_status = cost_tracker.get_budget_status()
        payload = {"total_cost": total, "breakdown": breakdown, "budget_status": budget_status}
        return success_response(payload, user=token)

    @app.post("/api/v1/costs/budget", response_model=ApiResponse)
    async def set_budget(budget: float, token: str = Depends(app.state.verify_token)):
        cost_tracker.set_budget(budget)
        return success_response({"budget": budget, "status": "set"}, user=token)

    @app.get("/api/v1/webhooks", response_model=ApiResponse)
    async def list_webhooks(token: str = Depends(app.state.verify_token)):
        webhooks = list(webhook_manager.webhooks.values())
        return success_response([w.to_dict() for w in webhooks], count=len(webhooks), user=token)

    @app.post("/api/v1/webhooks", response_model=ApiResponse)
    async def register_webhook(request: WebhookRequest, token: str = Depends(app.state.verify_token)):
        events = [WebhookEvent(e) for e in request.events]
        webhook = webhook_manager.register_webhook(url=request.url, events=events, secret=request.secret)
        return success_response(webhook.to_dict(), user=token)

    @app.get("/api/v1/statistics", response_model=ApiResponse)
    async def get_statistics(token: str = Depends(app.state.verify_token)):
        results_stats = results_manager.get_statistics()
        experiments = experiment_manager.list_experiments()
        models = model_registry.list_models()
        payload = {
            "results": results_stats,
            "experiments": {
                "total": len(experiments),
                "completed": len([e for e in experiments if e.status.value == "completed"]),
                "running": len([e for e in experiments if e.status.value == "running"]),
            },
            "models": {
                "total": len(models),
                "production": len([m for m in models if m.status.value == "production"]),
            },
            "nodes": {
                "total": len(distributed_executor.nodes),
                "available": len(distributed_executor.get_available_nodes()),
            },
        }
        return success_response(payload, user=token)

    # ------------------------------------------------------------------
    # WebSocket channel for live updates
    # ------------------------------------------------------------------

    class ConnectionManager:
        """Manage WebSocket subscribers."""

        def __init__(self) -> None:
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket) -> None:
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket) -> None:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

        async def broadcast(self, message: Dict[str, Any]) -> None:
            for connection in self.active_connections:
                await connection.send_json(message)

    connection_manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await connection_manager.connect(websocket)
        try:
            while True:
                payload = await websocket.receive_json()
                response = success_response(payload, event="echo").dict()
                await connection_manager.broadcast(response)
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket)
            logger.info("WebSocket disconnected")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/health", response_model=ApiResponse, include_in_schema=False)
    async def health_check():
        payload = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": app.version,
        }
        return success_response(payload)


# Initialize default app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

