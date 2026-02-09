"""
Enterprise Routes - Endpoints para funcionalidades enterprise
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
import logging

from ..services.auth_service import AuthService
from ..services.optimization_service import OptimizationService
from ..services.predictive_analysis_service import PredictiveAnalysisService
from ..services.monitoring_service import MonitoringService, AlertLevel
from ..services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()

# Inicializar servicios
auth_service = AuthService()
optimization_service = OptimizationService()
predictive_service = PredictiveAnalysisService()
monitoring_service = MonitoringService()
storage_service = StorageService()


def verify_token(authorization: Optional[str] = Header(None)):
    """Verificar token de autenticación"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Token requerido")
    
    token = authorization.replace("Bearer ", "")
    payload = auth_service.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    
    return payload


@router.post("/auth/register")
async def register_user(
    email: str,
    password: str,
    name: str,
    role: str = "user"
):
    """Registrar nuevo usuario"""
    try:
        user = auth_service.register_user(email, password, name, role)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/auth/login")
async def login(email: str, password: str):
    """Iniciar sesión"""
    result = auth_service.authenticate(email, password)
    
    if not result:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    return result


@router.get("/auth/me")
async def get_current_user(token_data: dict = Depends(verify_token)):
    """Obtener usuario actual"""
    user = auth_service.get_user(token_data["user_id"])
    
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return user


@router.post("/optimize/budget/{store_id}")
async def optimize_budget(
    store_id: str,
    target_budget: Optional[float] = None,
    token_data: dict = Depends(verify_token)
):
    """Optimizar presupuesto de diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    optimization = optimization_service.optimize_budget(design, target_budget)
    return optimization


@router.get("/optimize/layout/{store_id}")
async def optimize_layout(
    store_id: str,
    token_data: dict = Depends(verify_token)
):
    """Optimizar layout"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    optimization = optimization_service.optimize_layout(design)
    return optimization


@router.post("/optimize/marketing/{store_id}")
async def optimize_marketing_budget(
    store_id: str,
    monthly_budget: float,
    token_data: dict = Depends(verify_token)
):
    """Optimizar presupuesto de marketing"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    optimization = optimization_service.optimize_marketing_budget(design, monthly_budget)
    return optimization


@router.get("/predict/success/{store_id}")
async def predict_success(
    store_id: str,
    token_data: dict = Depends(verify_token)
):
    """Predecir probabilidad de éxito"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    prediction = predictive_service.predict_success_probability(design)
    return prediction


@router.get("/predict/revenue/{store_id}")
async def predict_revenue(
    store_id: str,
    months: int = 12,
    token_data: dict = Depends(verify_token)
):
    """Predecir ingresos futuros"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    prediction = predictive_service.predict_revenue(design, months)
    return prediction


@router.get("/predict/traffic/{store_id}")
async def predict_traffic(
    store_id: str,
    location_score: Optional[float] = None,
    token_data: dict = Depends(verify_token)
):
    """Predecir tráfico de clientes"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    prediction = predictive_service.predict_customer_traffic(design, location_score)
    return prediction


@router.get("/monitoring/health/{store_id}")
async def check_design_health(
    store_id: str,
    token_data: dict = Depends(verify_token)
):
    """Verificar salud del diseño"""
    design = storage_service.load_design(store_id)
    if not design:
        raise HTTPException(status_code=404, detail="Diseño no encontrado")
    
    health = monitoring_service.check_design_health(design.dict())
    return health


@router.get("/monitoring/alerts/{store_id}")
async def get_alerts(
    store_id: str,
    level: Optional[str] = None,
    unresolved_only: bool = False,
    token_data: dict = Depends(verify_token)
):
    """Obtener alertas"""
    alert_level = None
    if level:
        try:
            alert_level = AlertLevel(level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Nivel de alerta inválido: {level}")
    
    alerts = monitoring_service.get_alerts(store_id, alert_level, unresolved_only)
    return {"store_id": store_id, "alerts": alerts}


@router.post("/monitoring/alerts/{store_id}/acknowledge/{alert_id}")
async def acknowledge_alert(
    store_id: str,
    alert_id: str,
    token_data: dict = Depends(verify_token)
):
    """Reconocer alerta"""
    acknowledged = monitoring_service.acknowledge_alert(store_id, alert_id)
    
    if not acknowledged:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")
    
    return {"message": "Alerta reconocida", "alert_id": alert_id}


@router.post("/monitoring/alerts/{store_id}/resolve/{alert_id}")
async def resolve_alert(
    store_id: str,
    alert_id: str,
    token_data: dict = Depends(verify_token)
):
    """Resolver alerta"""
    resolved = monitoring_service.resolve_alert(store_id, alert_id)
    
    if not resolved:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")
    
    return {"message": "Alerta resuelta", "alert_id": alert_id}


@router.get("/monitoring/metrics/{store_id}")
async def get_metrics(
    store_id: str,
    metric_name: Optional[str] = None,
    hours: int = 24,
    token_data: dict = Depends(verify_token)
):
    """Obtener métricas"""
    metrics = monitoring_service.get_metrics(store_id, metric_name, hours)
    return {"store_id": store_id, "metrics": metrics}




