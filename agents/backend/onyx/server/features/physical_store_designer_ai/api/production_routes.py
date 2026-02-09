"""Production Routes"""
from fastapi import APIRouter
from typing import Dict, Any
from pydantic import BaseModel

from ..services.api_serving_service import APIServingService
from ..services.ab_testing_ml_service import ABTestingMLService
from ..services.model_rollback_service import ModelRollbackService
from ..services.benchmarking_service import BenchmarkingService
from ..services.auto_scaling_service import AutoScalingService
from ..services.health_check_ml_service import HealthCheckMLService
from ..core.route_helpers import handle_route_errors, track_route_metrics

router = APIRouter()

api_service = APIServingService()
ab_service = ABTestingMLService()
rollback_service = ModelRollbackService()
bench_service = BenchmarkingService()
scaling_service = AutoScalingService()
health_service = HealthCheckMLService()

@router.post("/api-serving/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("production.create_api")
async def create_model_api(model_id: str, endpoint: str, version: str = "v1"):
    return api_service.create_model_api(model_id, endpoint, version)

@router.post("/ab-testing/create", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("production.create_ab_test")
async def create_ab_test(model_a_id: str, model_b_id: str, traffic_split: float = 0.5):
    return ab_service.create_ab_test(model_a_id, model_b_id, traffic_split)

@router.post("/rollback/execute", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("production.rollback")
async def rollback_model(model_id: str, target_version: str):
    return rollback_service.rollback_model(model_id, target_version)

@router.post("/benchmark/run", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("production.benchmark")
async def benchmark_model(model_id: str, dataset_id: str):
    return bench_service.benchmark_model(model_id, dataset_id)

@router.post("/autoscaling/setup", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("production.autoscaling")
async def setup_autoscaling(model_id: str, min_replicas: int = 1, max_replicas: int = 10):
    return scaling_service.setup_autoscaling(model_id, min_replicas, max_replicas)

@router.get("/health/model/{model_id}", response_model=Dict[str, Any])
@handle_route_errors
@track_route_metrics("production.health_check")
async def check_model_health(model_id: str):
    return health_service.check_model_health(model_id)




