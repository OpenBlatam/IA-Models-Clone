"""
Job Alerts Service - Alertas de trabajos
=========================================

Sistema de alertas automáticas para nuevos trabajos que coincidan con el perfil.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class JobAlert:
    """Alerta de trabajo"""
    id: str
    user_id: str
    keywords: List[str]
    location: Optional[str] = None
    job_types: List[str] = field(default_factory=list)
    salary_range: Optional[Dict[str, float]] = None
    frequency: str = "daily"  # daily, weekly, real_time
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_check: Optional[datetime] = None
    matches_found: int = 0


class JobAlertsService:
    """Servicio de alertas de trabajos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.alerts: Dict[str, List[JobAlert]] = {}  # user_id -> [alerts]
        logger.info("JobAlertsService initialized")
    
    def create_alert(
        self,
        user_id: str,
        keywords: List[str],
        location: Optional[str] = None,
        job_types: Optional[List[str]] = None,
        salary_range: Optional[Dict[str, float]] = None,
        frequency: str = "daily"
    ) -> JobAlert:
        """Crear alerta de trabajo"""
        alert = JobAlert(
            id=f"alert_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            keywords=keywords,
            location=location,
            job_types=job_types or [],
            salary_range=salary_range,
            frequency=frequency,
        )
        
        if user_id not in self.alerts:
            self.alerts[user_id] = []
        
        self.alerts[user_id].append(alert)
        
        logger.info(f"Job alert created for user {user_id}")
        return alert
    
    def check_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """Verificar alertas y encontrar matches"""
        user_alerts = self.alerts.get(user_id, [])
        matches = []
        
        for alert in user_alerts:
            if not alert.active:
                continue
            
            # Verificar frecuencia
            if alert.frequency == "daily":
                if alert.last_check and (datetime.now() - alert.last_check).days < 1:
                    continue
            elif alert.frequency == "weekly":
                if alert.last_check and (datetime.now() - alert.last_check).days < 7:
                    continue
            
            # Buscar trabajos (simulado)
            # En producción, esto buscaría en las APIs reales
            found_jobs = self._search_matching_jobs(alert)
            
            if found_jobs:
                matches.append({
                    "alert_id": alert.id,
                    "jobs": found_jobs,
                    "count": len(found_jobs),
                })
                alert.matches_found += len(found_jobs)
            
            alert.last_check = datetime.now()
        
        return matches
    
    def _search_matching_jobs(self, alert: JobAlert) -> List[Dict[str, Any]]:
        """Buscar trabajos que coincidan con la alerta"""
        # En producción, esto buscaría en APIs reales
        # Por ahora, simulamos resultados
        return [
            {
                "id": f"job_match_{i}",
                "title": f"{alert.keywords[0]} Developer",
                "company": "Tech Company",
                "location": alert.location or "Remote",
            }
            for i in range(3)  # Simular 3 matches
        ]
    
    def get_user_alerts(self, user_id: str) -> List[JobAlert]:
        """Obtener alertas del usuario"""
        return self.alerts.get(user_id, [])
    
    def deactivate_alert(self, user_id: str, alert_id: str) -> bool:
        """Desactivar alerta"""
        alerts = self.alerts.get(user_id, [])
        for alert in alerts:
            if alert.id == alert_id:
                alert.active = False
                return True
        return False




