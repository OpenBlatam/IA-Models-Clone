"""
Recommendations API for intelligent maintenance recommendations.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base_router import BaseRouter
from .dependencies import get_tutor
from ..core.ml_predictor import MLPredictor
from ..config.maintenance_config import MLConfig
from ..utils.file_helpers import get_iso_timestamp, get_timestamp_id, parse_iso_date
from ..utils.data_helpers import increment_dict_value

# Create base router instance
base = BaseRouter(
    prefix="/api/recommendations",
    tags=["Recommendations"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    robot_type: str = Field(..., description="Type of robot")
    sensor_data: Optional[Dict[str, Any]] = Field(None, description="Current sensor data")
    maintenance_history: Optional[List[Dict[str, Any]]] = Field(None, description="Maintenance history")
    context: Optional[str] = Field(None, description="Additional context")


@router.post("/maintenance")
@base.timed_endpoint("get_maintenance_recommendations")
async def get_maintenance_recommendations(
    request: RecommendationRequest,
    _: Dict = Depends(base.get_auth_dependency()),
    tutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """
    Get intelligent maintenance recommendations based on robot state and history.
    """
    base.log_request("get_maintenance_recommendations", robot_type=request.robot_type)
    
    recommendations = []
    
    # Get maintenance history if not provided
    if not request.maintenance_history:
        history = base.database.get_maintenance_history(robot_type=request.robot_type, limit=10)
        request.maintenance_history = history
    
    # Analyze sensor data if provided
    if request.sensor_data:
        ml_config = MLConfig()
        predictor = MLPredictor(ml_config)
        analysis = await predictor.analyze_sensor_data(request.sensor_data)
        
        # Generate recommendations based on analysis
        health_score = analysis.get("health_score", 0)
        
        if health_score < -0.5:
            recommendations.append({
                "type": "urgent",
                "priority": "high",
                "title": "Mantenimiento Urgente Requerido",
                "description": "El sistema detectó anomalías críticas. Se recomienda inspección inmediata.",
                "actions": [
                    "Detener operación si es posible",
                    "Inspección visual completa",
                    "Revisar componentes críticos",
                    "Contactar técnico especializado"
                ],
                "estimated_time": "2-4 horas",
                "cost_estimate": "Alto"
            })
        elif health_score < -0.2:
            recommendations.append({
                "type": "preventive",
                "priority": "medium",
                "title": "Mantenimiento Preventivo Recomendado",
                "description": "Se detectaron señales de desgaste. Se recomienda mantenimiento preventivo.",
                "actions": [
                    "Programar mantenimiento en las próximas 48 horas",
                    "Revisar componentes específicos",
                    "Verificar lubricación",
                    "Calibrar sensores"
                ],
                "estimated_time": "1-2 horas",
                "cost_estimate": "Medio"
            })
        
        # Add specific recommendations based on anomalies
        anomalies = analysis.get("anomalies", [])
        for anomaly in anomalies:
            recommendations.append({
                "type": "specific",
                "priority": "medium",
                "title": f"Revisar {anomaly.get('component', 'Componente')}",
                "description": anomaly.get("description", ""),
                "actions": anomaly.get("recommended_actions", []),
                "estimated_time": "30-60 minutos",
                "cost_estimate": "Bajo"
            })
    
    # Add recommendations based on maintenance history
    if request.maintenance_history:
        last_maintenance = request.maintenance_history[0] if request.maintenance_history else None
        if last_maintenance:
            last_date = parse_iso_date(last_maintenance.get("created_at", ""))
            if not last_date:
                last_date = datetime.now()
            days_since = (datetime.now() - last_date).days
            
            if days_since > 90:
                recommendations.append({
                    "type": "scheduled",
                    "priority": "medium",
                    "title": "Mantenimiento Programado Pendiente",
                    "description": f"Han pasado {days_since} días desde el último mantenimiento.",
                    "actions": [
                        "Programar mantenimiento preventivo",
                        "Revisar checklist de mantenimiento",
                        "Actualizar registros"
                    ],
                    "estimated_time": "2-3 horas",
                    "cost_estimate": "Medio"
                })
    
    # Add general recommendations
    recommendations.append({
        "type": "general",
        "priority": "low",
        "title": "Mantenimiento de Rutina",
        "description": "Mantenimiento regular recomendado para mantener el robot en óptimas condiciones.",
        "actions": [
            "Limpieza general",
            "Verificación de conexiones",
            "Actualización de software si aplica",
            "Documentación de estado"
        ],
        "estimated_time": "1 hora",
        "cost_estimate": "Bajo"
    })
    
    return base.success({
        "robot_type": request.robot_type,
        "recommendations": recommendations,
        "total": len(recommendations),
        "timestamp": get_iso_timestamp()
    })


@router.get("/optimization")
@base.timed_endpoint("get_optimization_recommendations")
async def get_optimization_recommendations(
    robot_type: str = Query(..., description="Type of robot"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get optimization recommendations for robot performance.
    """
    base.log_request("get_optimization_recommendations", robot_type=robot_type)
    
    history = base.database.get_maintenance_history(robot_type=robot_type, limit=50)
    
    recommendations = []
    
    # Analyze maintenance frequency
    if len(history) > 5:
        recommendations.append({
            "category": "frequency",
            "title": "Optimizar Frecuencia de Mantenimiento",
            "description": "Basado en el historial, se puede optimizar la frecuencia de mantenimiento.",
            "suggestion": "Considerar mantenimiento predictivo en lugar de preventivo fijo"
        })
    
    # Analyze common issues
    maintenance_types = {}
    for record in history:
        maint_type = record.get("maintenance_type", "unknown")
        increment_dict_value(maintenance_types, maint_type)
    
    if maintenance_types:
        most_common = max(maintenance_types.items(), key=lambda x: x[1])
        recommendations.append({
            "category": "pattern",
            "title": f"Mantenimiento Más Frecuente: {most_common[0]}",
            "description": f"Este tipo de mantenimiento representa el {most_common[1]/len(history)*100:.1f}% del total.",
            "suggestion": f"Considerar revisión preventiva más frecuente para {most_common[0]}"
        })
    
    return base.success({
        "robot_type": robot_type,
        "recommendations": recommendations,
        "analysis": {
            "total_records": len(history),
            "maintenance_types": maintenance_types
        }
    })


@router.post("/schedule")
@base.timed_endpoint("schedule_maintenance")
async def schedule_maintenance(
    robot_type: str = Field(..., description="Type of robot"),
    maintenance_type: str = Field(..., description="Type of maintenance"),
    scheduled_date: str = Field(..., description="Scheduled date (ISO format)"),
    notes: Optional[str] = Field(None, description="Additional notes"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Schedule a maintenance based on recommendations.
    """
    base.log_request("schedule_maintenance", robot_type=robot_type, maintenance_type=maintenance_type)
    
    scheduled = {
        "id": get_timestamp_id("schedule_"),
        "robot_type": robot_type,
        "maintenance_type": maintenance_type,
        "scheduled_date": scheduled_date,
        "notes": notes,
        "created_at": get_iso_timestamp(),
        "status": "scheduled"
    }
    
    return base.success(scheduled, message="Maintenance scheduled successfully")




