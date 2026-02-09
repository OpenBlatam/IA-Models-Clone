"""
Realtime Competitor Service - Análisis de competencia en tiempo real
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..services.llm_service import LLMService
from ..services.external_apis_service import ExternalAPIsService

logger = logging.getLogger(__name__)


class RealtimeCompetitorService:
    """Servicio para análisis de competencia en tiempo real"""
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        external_apis: Optional[ExternalAPIsService] = None
    ):
        self.llm_service = llm_service or LLMService()
        self.external_apis = external_apis or ExternalAPIsService()
        self.competitor_data: Dict[str, Dict[str, Any]] = {}
        self.monitoring_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def start_monitoring(
        self,
        store_id: str,
        location: str,
        store_type: str,
        monitoring_frequency: str = "daily"  # "hourly", "daily", "weekly"
    ) -> Dict[str, Any]:
        """Iniciar monitoreo de competencia"""
        
        job_id = f"monitor_{store_id}_{datetime.now().strftime('%Y%m%d')}"
        
        job = {
            "job_id": job_id,
            "store_id": store_id,
            "location": location,
            "store_type": store_type,
            "frequency": monitoring_frequency,
            "status": "active",
            "started_at": datetime.now().isoformat(),
            "last_check": None,
            "checks_count": 0
        }
        
        self.monitoring_jobs[job_id] = job
        
        # Ejecutar primera verificación
        await self._perform_check(job_id)
        
        return job
    
    async def _perform_check(self, job_id: str):
        """Realizar verificación de competencia"""
        job = self.monitoring_jobs.get(job_id)
        
        if not job:
            return
        
        location = job["location"]
        store_type = job["store_type"]
        
        # Obtener lugares cercanos
        nearby = await self.external_apis.get_nearby_places(location, store_type)
        
        # Analizar competencia
        analysis = await self._analyze_competitors(nearby, store_type)
        
        # Guardar datos
        check_id = f"check_{job_id}_{job['checks_count'] + 1}"
        self.competitor_data[check_id] = {
            "check_id": check_id,
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "nearby_places": nearby,
            "analysis": analysis
        }
        
        job["last_check"] = datetime.now().isoformat()
        job["checks_count"] += 1
    
    async def _analyze_competitors(
        self,
        nearby_data: Dict[str, Any],
        store_type: str
    ) -> Dict[str, Any]:
        """Analizar competidores"""
        
        if "error" in nearby_data or not nearby_data.get("nearby_places"):
            return {
                "competitors_found": 0,
                "market_saturation": "unknown",
                "recommendations": ["Verificar datos de ubicación"]
            }
        
        competitors = nearby_data["nearby_places"]
        competitor_count = len(competitors)
        
        # Calcular saturación
        if competitor_count == 0:
            saturation = "low"
        elif competitor_count < 3:
            saturation = "medium"
        elif competitor_count < 6:
            saturation = "high"
        else:
            saturation = "very_high"
        
        # Análisis de ratings
        ratings = [c.get("rating", 0) for c in competitors if c.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Recomendaciones
        recommendations = []
        if saturation == "very_high":
            recommendations.append("Mercado muy saturado - considerar diferenciación fuerte")
        elif saturation == "high":
            recommendations.append("Alta competencia - enfocarse en nicho específico")
        else:
            recommendations.append("Oportunidad de mercado - competencia moderada")
        
        if avg_rating > 4.5:
            recommendations.append("Competencia de alta calidad - asegurar excelencia")
        
        return {
            "competitors_found": competitor_count,
            "market_saturation": saturation,
            "average_competitor_rating": round(avg_rating, 2),
            "top_competitors": competitors[:5],
            "recommendations": recommendations,
            "threat_level": "high" if saturation in ["high", "very_high"] else "medium" if saturation == "medium" else "low"
        }
    
    async def get_realtime_analysis(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Obtener análisis en tiempo real"""
        # Buscar job activo
        job = next(
            (j for j in self.monitoring_jobs.values() if j["store_id"] == store_id and j["status"] == "active"),
            None
        )
        
        if not job:
            return {"message": "No hay monitoreo activo para este diseño"}
        
        # Obtener última verificación
        latest_check = None
        for check_id, check_data in self.competitor_data.items():
            if check_data["job_id"] == job["job_id"]:
                if not latest_check or check_data["timestamp"] > latest_check["timestamp"]:
                    latest_check = check_data
        
        return {
            "store_id": store_id,
            "monitoring_active": True,
            "job": job,
            "latest_analysis": latest_check.get("analysis") if latest_check else None,
            "last_updated": latest_check.get("timestamp") if latest_check else None
        }
    
    def stop_monitoring(self, job_id: str) -> bool:
        """Detener monitoreo"""
        job = self.monitoring_jobs.get(job_id)
        
        if not job:
            return False
        
        job["status"] = "stopped"
        job["stopped_at"] = datetime.now().isoformat()
        
        return True
    
    def get_monitoring_history(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener historial de monitoreo"""
        history = []
        
        for check_id, check_data in self.competitor_data.items():
            job = self.monitoring_jobs.get(check_data["job_id"])
            if job and job["store_id"] == store_id:
                history.append(check_data)
        
        # Ordenar por timestamp
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return history
    
    async def compare_with_competitors(
        self,
        store_id: str,
        design_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comparar diseño con competidores"""
        
        # Obtener análisis en tiempo real
        realtime = await self.get_realtime_analysis(store_id)
        
        if not realtime.get("latest_analysis"):
            return {"message": "No hay datos de competencia disponibles"}
        
        analysis = realtime["latest_analysis"]
        competitors = analysis.get("top_competitors", [])
        
        comparison = {
            "your_design": {
                "store_name": design_data.get("store_name"),
                "style": design_data.get("style"),
                "estimated_rating": 4.0  # Estimado
            },
            "competitors": competitors,
            "differentiation_opportunities": self._find_differentiation(design_data, competitors),
            "competitive_advantages": self._identify_advantages(design_data, competitors),
            "recommendations": analysis.get("recommendations", [])
        }
        
        return comparison
    
    def _find_differentiation(
        self,
        design: Dict[str, Any],
        competitors: List[Dict[str, Any]]
    ) -> List[str]:
        """Encontrar oportunidades de diferenciación"""
        opportunities = []
        
        # Analizar estilos de competidores
        competitor_styles = set()  # En producción, extraer de datos reales
        
        design_style = design.get("style", "")
        if design_style not in competitor_styles:
            opportunities.append(f"Estilo único: {design_style} no es común en la competencia")
        
        # Otras oportunidades
        opportunities.extend([
            "Enfocarse en experiencia del cliente superior",
            "Marketing diferenciado",
            "Productos/servicios únicos"
        ])
        
        return opportunities
    
    def _identify_advantages(
        self,
        design: Dict[str, Any],
        competitors: List[Dict[str, Any]]
    ) -> List[str]:
        """Identificar ventajas competitivas"""
        advantages = []
        
        # Si tiene análisis financiero completo
        if design.get("financial_analysis"):
            advantages.append("Análisis financiero completo - mejor planificación")
        
        # Si tiene plan de marketing
        if design.get("marketing_plan"):
            advantages.append("Plan de marketing estructurado")
        
        advantages.extend([
            "Diseño profesional y planificado",
            "Análisis de competencia realizado"
        ])
        
        return advantages




