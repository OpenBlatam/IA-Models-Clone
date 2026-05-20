"""
Manufacturing API
=================

API RESTful para sistemas de manufactura inteligente.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from ..core.production_planner import (
    get_production_planner,
    Priority,
    OrderStatus
)
from ..core.quality_control import (
    get_quality_controller,
    QualityStatus
)
from ..core.process_optimizer import get_process_optimizer
from ..core.monitoring import (
    get_manufacturing_monitor,
    EquipmentStatus
)
from ..models.quality_predictor import get_quality_predictor_manager
from ..models.process_optimizer_model import get_process_optimizer_model_manager
from ..core.demand_forecasting import get_demand_forecasting_system
from ..core.predictive_maintenance import (
    get_predictive_maintenance_system,
    SensorData
)
from ..core.capacity_optimizer import get_capacity_optimizer
from ..core.inventory_optimizer import get_intelligent_inventory_system
from ..core.production_route_optimizer import get_production_route_optimizer
from ..core.production_analyzer import get_advanced_production_analyzer
from ..models.diffusion_config_generator import get_diffusion_config_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/manufacturing", tags=["Manufacturing"])


# Request/Response Models
class CreateOrderRequest(BaseModel):
    """Request para crear orden."""
    product_id: str = Field(..., description="ID del producto")
    quantity: int = Field(..., description="Cantidad")
    due_date: str = Field(..., description="Fecha límite (ISO format)")
    priority: str = Field(default="medium", description="Prioridad")
    estimated_duration: Optional[float] = Field(None, description="Duración estimada (horas)")


class RegisterResourceRequest(BaseModel):
    """Request para registrar recurso."""
    resource_id: str = Field(..., description="ID del recurso")
    name: str = Field(..., description="Nombre")
    resource_type: str = Field(..., description="Tipo de recurso")
    capacity: float = Field(default=1.0, description="Capacidad")


class QualityCheckRequest(BaseModel):
    """Request para check de calidad."""
    product_id: str = Field(..., description="ID del producto")
    check_type: str = Field(..., description="Tipo de check")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parámetros")


class DimensionalCheckRequest(BaseModel):
    """Request para check dimensional."""
    check_id: str = Field(..., description="ID del check")
    measurements: Dict[str, float] = Field(..., description="Mediciones")
    tolerances: Dict[str, List[float]] = Field(..., description="Tolerancias {dim: [min, max]}")


class RegisterProcessRequest(BaseModel):
    """Request para registrar proceso."""
    name: str = Field(..., description="Nombre del proceso")
    process_type: str = Field(..., description="Tipo de proceso")
    parameters: Dict[str, float] = Field(default_factory=dict, description="Parámetros")


class OptimizeProcessRequest(BaseModel):
    """Request para optimizar proceso."""
    process_id: str = Field(..., description="ID del proceso")
    objective: str = Field(default="efficiency", description="Objetivo")
    model_id: Optional[str] = Field(None, description="ID del modelo")


class RegisterEquipmentRequest(BaseModel):
    """Request para registrar equipo."""
    equipment_id: str = Field(..., description="ID del equipo")
    name: str = Field(..., description="Nombre")
    equipment_type: str = Field(..., description="Tipo de equipo")


class RegisterInventoryItemRequest(BaseModel):
    """Request para registrar item de inventario."""
    item_id: str = Field(..., description="ID del item")
    product_id: str = Field(..., description="ID del producto")
    current_quantity: float = Field(..., description="Cantidad actual")
    min_quantity: float = Field(..., description="Cantidad mínima")
    max_quantity: float = Field(..., description="Cantidad máxima")
    reorder_point: float = Field(..., description="Punto de reorden")
    lead_time: float = Field(..., description="Tiempo de entrega (días)")


class RegisterProductionStepRequest(BaseModel):
    """Request para registrar paso de producción."""
    step_id: str = Field(..., description="ID del paso")
    name: str = Field(..., description="Nombre")
    machine_id: str = Field(..., description="ID de máquina")
    duration: float = Field(..., description="Duración (horas)")
    dependencies: List[str] = Field(default_factory=list, description="Dependencias")
    position: List[float] = Field(default=[0.0, 0.0], description="Posición [x, y]")


class CreateRouteRequest(BaseModel):
    """Request para crear ruta."""
    step_ids: List[str] = Field(..., description="IDs de pasos")
    optimize: bool = Field(default=True, description="Optimizar orden")


class AnalyzeProductionRequest(BaseModel):
    """Request para analizar producción."""
    production_id: str = Field(..., description="ID de producción")
    process_description: str = Field(..., description="Descripción del proceso")
    process_features: List[float] = Field(..., description="Características numéricas")
    model_id: Optional[str] = Field(None, description="ID del modelo")


# Production Planning Endpoints
@router.post("/orders", response_model=Dict[str, Any])
async def create_order(request: CreateOrderRequest):
    """Crear orden de producción."""
    try:
        planner = get_production_planner()
        priority = Priority[request.priority.upper()]
        
        order_id = planner.create_order(
            request.product_id,
            request.quantity,
            request.due_date,
            priority,
            request.estimated_duration
        )
        
        return {
            "order_id": order_id,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders/{order_id}/schedule", response_model=Dict[str, Any])
async def schedule_order(order_id: str):
    """Programar orden."""
    try:
        planner = get_production_planner()
        success = planner.schedule_order(order_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to schedule order")
        
        return {
            "order_id": order_id,
            "status": "scheduled"
        }
    except Exception as e:
        logger.error(f"Error scheduling order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders/{order_id}/start", response_model=Dict[str, Any])
async def start_order(order_id: str):
    """Iniciar orden."""
    try:
        planner = get_production_planner()
        success = planner.start_order(order_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start order")
        
        return {
            "order_id": order_id,
            "status": "started"
        }
    except Exception as e:
        logger.error(f"Error starting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders/{order_id}/complete", response_model=Dict[str, Any])
async def complete_order(order_id: str, actual_duration: Optional[float] = None):
    """Completar orden."""
    try:
        planner = get_production_planner()
        success = planner.complete_order(order_id, actual_duration)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to complete order")
        
        return {
            "order_id": order_id,
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Error completing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-schedule", response_model=Dict[str, Any])
async def optimize_schedule():
    """Optimizar schedule."""
    try:
        planner = get_production_planner()
        result = planner.optimize_schedule()
        return result
    except Exception as e:
        logger.error(f"Error optimizing schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quality Control Endpoints
@router.post("/quality/checks", response_model=Dict[str, Any])
async def create_quality_check(request: QualityCheckRequest):
    """Crear check de calidad."""
    try:
        controller = get_quality_controller()
        check_id = controller.create_check(
            request.product_id,
            request.check_type,
            request.parameters
        )
        
        return {
            "check_id": check_id,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating quality check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/checks/{check_id}/dimensional", response_model=Dict[str, Any])
async def perform_dimensional_check(request: DimensionalCheckRequest):
    """Realizar check dimensional."""
    try:
        controller = get_quality_controller()
        
        # Convertir tolerancias
        tolerances = {}
        for key, value in request.tolerances.items():
            if len(value) == 2:
                tolerances[key] = (float(value[0]), float(value[1]))
        
        result = controller.perform_dimensional_check(
            request.check_id,
            request.measurements,
            tolerances
        )
        
        return {
            "result_id": result.result_id,
            "status": result.status.value,
            "score": result.score,
            "defects": result.defects
        }
    except Exception as e:
        logger.error(f"Error performing dimensional check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Process Optimization Endpoints
@router.post("/processes", response_model=Dict[str, Any])
async def register_process(request: RegisterProcessRequest):
    """Registrar proceso."""
    try:
        optimizer = get_process_optimizer()
        process_id = optimizer.register_process(
            request.name,
            request.process_type,
            request.parameters
        )
        
        return {
            "process_id": process_id,
            "status": "registered"
        }
    except Exception as e:
        logger.error(f"Error registering process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processes/optimize", response_model=Dict[str, Any])
async def optimize_process(request: OptimizeProcessRequest):
    """Optimizar proceso."""
    try:
        optimizer = get_process_optimizer()
        result = optimizer.optimize_process(
            request.process_id,
            request.objective,
            request.model_id
        )
        
        return {
            "result_id": result.result_id,
            "optimized_parameters": result.optimized_parameters,
            "predicted_improvement": result.predicted_improvement,
            "recommendations": result.recommendations
        }
    except Exception as e:
        logger.error(f"Error optimizing process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring Endpoints
@router.post("/equipment", response_model=Dict[str, Any])
async def register_equipment(request: RegisterEquipmentRequest):
    """Registrar equipo."""
    try:
        monitor = get_manufacturing_monitor()
        monitor.register_equipment(
            request.equipment_id,
            request.name,
            request.equipment_type
        )
        
        return {
            "equipment_id": request.equipment_id,
            "status": "registered"
        }
    except Exception as e:
        logger.error(f"Error registering equipment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quality/create", response_model=Dict[str, Any])
async def create_quality_model(
    model_id: str,
    image_input_size: int = 224,
    num_features: int = 10,
    use_advanced: bool = False
):
    """Crear modelo de predicción de calidad."""
    try:
        manager = get_quality_predictor_manager()
        manager.create_model(model_id, image_input_size, num_features, use_advanced)
        
        return {
            "model_id": model_id,
            "status": "created",
            "architecture": "advanced" if use_advanced else "standard"
        }
    except Exception as e:
        logger.error(f"Error creating quality model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/optimizer/create", response_model=Dict[str, Any])
async def create_optimizer_model(
    model_id: str,
    input_dim: int = 10,
    output_dim: int = 10,
    use_advanced: bool = False
):
    """Crear modelo de optimización."""
    try:
        manager = get_process_optimizer_model_manager()
        manager.create_model(model_id, input_dim, output_dim, use_advanced)
        
        return {
            "model_id": model_id,
            "status": "created",
            "architecture": "advanced" if use_advanced else "standard"
        }
    except Exception as e:
        logger.error(f"Error creating optimizer model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Demand Forecasting Endpoints
@router.post("/demand/forecast", response_model=Dict[str, Any])
async def forecast_demand(
    product_id: str,
    forecast_days: int = 30,
    sequence_length: int = 30
):
    """Predecir demanda."""
    try:
        system = get_demand_forecasting_system()
        forecast = system.forecast(product_id, forecast_days, sequence_length)
        
        return {
            "forecast_id": forecast.forecast_id,
            "product_id": forecast.product_id,
            "predicted_demand": forecast.predicted_demand,
            "confidence_interval": forecast.confidence_interval,
            "forecast_date": forecast.forecast_date
        }
    except Exception as e:
        logger.error(f"Error forecasting demand: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demand/add-data", response_model=Dict[str, Any])
async def add_demand_data(product_id: str, data: List[float]):
    """Agregar datos históricos de demanda."""
    try:
        system = get_demand_forecasting_system()
        system.add_historical_data(product_id, data)
        
        return {
            "product_id": product_id,
            "data_points_added": len(data),
            "status": "added"
        }
    except Exception as e:
        logger.error(f"Error adding demand data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Predictive Maintenance Endpoints
@router.post("/maintenance/predict", response_model=Dict[str, Any])
async def predict_maintenance(equipment_id: str, sequence_length: int = 100):
    """Predecir mantenimiento."""
    try:
        system = get_predictive_maintenance_system()
        prediction = system.predict_failure(equipment_id, sequence_length)
        
        return {
            "prediction_id": prediction.prediction_id,
            "equipment_id": prediction.equipment_id,
            "failure_probability": prediction.failure_probability,
            "remaining_life": prediction.remaining_life,
            "recommended_action": prediction.recommended_action,
            "predicted_failure_date": prediction.predicted_failure_date
        }
    except Exception as e:
        logger.error(f"Error predicting maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/sensor-data", response_model=Dict[str, Any])
async def add_sensor_data(
    equipment_id: str,
    temperature: float,
    vibration: float,
    pressure: float,
    current: float
):
    """Agregar datos de sensores."""
    try:
        system = get_predictive_maintenance_system()
        
        sensor_data = SensorData(
            equipment_id=equipment_id,
            timestamp=datetime.now().isoformat(),
            temperature=temperature,
            vibration=vibration,
            pressure=pressure,
            current=current
        )
        
        system.add_sensor_data(sensor_data)
        
        return {
            "equipment_id": equipment_id,
            "status": "added"
        }
    except Exception as e:
        logger.error(f"Error adding sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Capacity Optimization Endpoints
@router.post("/capacity/optimize", response_model=Dict[str, Any])
async def optimize_capacity(
    resource_id: str,
    current_load: float,
    demand_forecast: float,
    efficiency: float
):
    """Optimizar capacidad."""
    try:
        optimizer = get_capacity_optimizer()
        plan = optimizer.optimize_capacity(
            resource_id,
            current_load,
            demand_forecast,
            efficiency
        )
        
        return {
            "plan_id": plan.plan_id,
            "resource_id": plan.resource_id,
            "current_capacity": plan.current_capacity,
            "optimal_capacity": plan.optimal_capacity,
            "utilization_rate": plan.utilization_rate,
            "recommendations": plan.recommendations
        }
    except Exception as e:
        logger.error(f"Error optimizing capacity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capacity/allocate", response_model=Dict[str, Any])
async def allocate_resource(
    resource_id: str,
    task_id: str,
    required_capacity: float,
    duration: float
):
    """Asignar recurso."""
    try:
        optimizer = get_capacity_optimizer()
        allocation = optimizer.allocate_resource(
            resource_id,
            task_id,
            required_capacity,
            duration
        )
        
        return {
            "allocation_id": allocation.allocation_id,
            "resource_id": allocation.resource_id,
            "task_id": allocation.task_id,
            "allocated_capacity": allocation.allocated_capacity,
            "efficiency_score": allocation.efficiency_score
        }
    except Exception as e:
        logger.error(f"Error allocating resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Inventory Optimization Endpoints
@router.post("/inventory/register", response_model=Dict[str, Any])
async def register_inventory_item(request: RegisterInventoryItemRequest):
    """Registrar item de inventario."""
    try:
        system = get_intelligent_inventory_system()
        system.register_item(
            request.item_id,
            request.product_id,
            request.current_quantity,
            request.min_quantity,
            request.max_quantity,
            request.reorder_point,
            request.lead_time
        )
        
        return {
            "item_id": request.item_id,
            "status": "registered"
        }
    except Exception as e:
        logger.error(f"Error registering inventory item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/predict", response_model=Dict[str, Any])
async def predict_inventory(
    item_id: str,
    forecast_days: int = 30,
    demand_forecast: Optional[float] = None
):
    """Predecir inventario."""
    try:
        system = get_intelligent_inventory_system()
        prediction = system.predict_inventory(item_id, forecast_days, demand_forecast)
        
        return {
            "prediction_id": prediction.prediction_id,
            "item_id": prediction.item_id,
            "predicted_quantity": prediction.predicted_quantity,
            "reorder_recommendation": prediction.reorder_recommendation,
            "predicted_date": prediction.predicted_date
        }
    except Exception as e:
        logger.error(f"Error predicting inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/optimize", response_model=Dict[str, Any])
async def optimize_inventory():
    """Optimizar niveles de inventario."""
    try:
        system = get_intelligent_inventory_system()
        result = system.optimize_inventory_levels()
        return result
    except Exception as e:
        logger.error(f"Error optimizing inventory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Production Route Optimization Endpoints
@router.post("/routes/steps", response_model=Dict[str, Any])
async def register_production_step(request: RegisterProductionStepRequest):
    """Registrar paso de producción."""
    try:
        optimizer = get_production_route_optimizer()
        optimizer.register_step(
            request.step_id,
            request.name,
            request.machine_id,
            request.duration,
            request.dependencies,
            tuple(request.position) if request.position else None
        )
        
        return {
            "step_id": request.step_id,
            "status": "registered"
        }
    except Exception as e:
        logger.error(f"Error registering production step: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/routes/create", response_model=Dict[str, Any])
async def create_production_route(request: CreateRouteRequest):
    """Crear ruta de producción."""
    try:
        optimizer = get_production_route_optimizer()
        route = optimizer.create_route(request.step_ids, request.optimize)
        
        return {
            "route_id": route.route_id,
            "total_duration": route.total_duration,
            "total_distance": route.total_distance,
            "efficiency": route.efficiency,
            "steps": [s.step_id for s in route.steps]
        }
    except Exception as e:
        logger.error(f"Error creating route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Production Analysis Endpoints
@router.post("/analysis", response_model=Dict[str, Any])
async def analyze_production(request: AnalyzeProductionRequest):
    """Analizar producción."""
    try:
        analyzer = get_advanced_production_analyzer()
        analysis = analyzer.analyze_production(
            request.production_id,
            request.process_description,
            request.process_features,
            request.model_id
        )
        
        return {
            "analysis_id": analysis.analysis_id,
            "production_id": analysis.production_id,
            "predicted_efficiency": analysis.predicted_efficiency,
            "predicted_quality": analysis.predicted_quality,
            "predicted_cost": analysis.predicted_cost,
            "recommendations": analysis.recommendations,
            "risk_factors": analysis.risk_factors
        }
    except Exception as e:
        logger.error(f"Error analyzing production: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Diffusion Config Generator Endpoints
@router.post("/diffusion/create-model", response_model=Dict[str, Any])
async def create_diffusion_model(
    model_id: str,
    config_dim: int = 10
):
    """Crear modelo de difusión para configuraciones."""
    try:
        generator = get_diffusion_config_generator()
        generator.create_model(model_id, config_dim)
        
        return {
            "model_id": model_id,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating diffusion model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diffusion/generate-config", response_model=Dict[str, Any])
async def generate_config(
    model_id: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    """Generar configuración usando difusión."""
    try:
        generator = get_diffusion_config_generator()
        config = generator.generate_config(model_id, num_inference_steps, guidance_scale)
        
        return {
            "model_id": model_id,
            "generated_config": config
        }
    except Exception as e:
        logger.error(f"Error generating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_statistics():
    """Obtener estadísticas generales."""
    try:
        planner = get_production_planner()
        quality = get_quality_controller()
        optimizer = get_process_optimizer()
        monitor = get_manufacturing_monitor()
        demand = get_demand_forecasting_system()
        maintenance = get_predictive_maintenance_system()
        capacity = get_capacity_optimizer()
        inventory = get_intelligent_inventory_system()
        routes = get_production_route_optimizer()
        analyzer = get_advanced_production_analyzer()
        
        return {
            "production": planner.get_statistics(),
            "quality": quality.get_statistics(),
            "optimization": optimizer.get_statistics(),
            "monitoring": monitor.get_statistics(),
            "demand_forecasting": demand.get_statistics(),
            "predictive_maintenance": maintenance.get_statistics(),
            "capacity_optimization": capacity.get_statistics(),
            "inventory": inventory.get_statistics(),
            "route_optimization": routes.get_statistics(),
            "production_analysis": analyzer.get_statistics()
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Factory functions
_production_planner = None
_quality_controller = None
_process_optimizer = None
_manufacturing_monitor = None


def get_production_planner():
    """Obtener instancia global del planificador."""
    global _production_planner
    if _production_planner is None:
        from ..core.production_planner import ProductionPlanner
        _production_planner = ProductionPlanner()
    return _production_planner


def get_quality_controller():
    """Obtener instancia global del controlador de calidad."""
    global _quality_controller
    if _quality_controller is None:
        from ..core.quality_control import QualityController
        _quality_controller = QualityController()
    return _quality_controller


def get_process_optimizer():
    """Obtener instancia global del optimizador."""
    global _process_optimizer
    if _process_optimizer is None:
        from ..core.process_optimizer import ProcessOptimizer
        _process_optimizer = ProcessOptimizer()
    return _process_optimizer


def get_manufacturing_monitor():
    """Obtener instancia global del monitor."""
    global _manufacturing_monitor
    if _manufacturing_monitor is None:
        from ..core.monitoring import ManufacturingMonitor
        _manufacturing_monitor = ManufacturingMonitor()
    return _manufacturing_monitor
