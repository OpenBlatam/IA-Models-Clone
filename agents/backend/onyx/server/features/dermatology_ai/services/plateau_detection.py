"""
Sistema de detección de mesetas en el progreso
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics
import numpy as np


@dataclass
class PlateauData:
    """Detección de meseta"""
    metric_name: str
    plateau_start_date: str
    plateau_duration_days: int
    average_value: float
    variation: float
    significance: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric_name": self.metric_name,
            "plateau_start_date": self.plateau_start_date,
            "plateau_duration_days": self.plateau_duration_days,
            "average_value": self.average_value,
            "variation": self.variation,
            "significance": self.significance
        }


@dataclass
class PlateauReport:
    """Reporte de mesetas"""
    id: str
    user_id: str
    plateaus: List[PlateauData]
    recommendations: List[str]
    action_items: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "plateaus": [p.to_dict() for p in self.plateaus],
            "recommendations": self.recommendations,
            "action_items": self.action_items,
            "created_at": self.created_at
        }


class PlateauDetectionSystem:
    """Sistema de detección de mesetas"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[PlateauReport]] = {}
        self.historical_data: Dict[str, List[Dict]] = {}
    
    def add_data_point(self, user_id: str, timestamp: str, metrics: Dict):
        """Agrega punto de datos"""
        data_point = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        if user_id not in self.historical_data:
            self.historical_data[user_id] = []
        
        self.historical_data[user_id].append(data_point)
        self.historical_data[user_id].sort(key=lambda x: x["timestamp"])
    
    def detect_plateaus(self, user_id: str, days: int = 60) -> PlateauReport:
        """Detecta mesetas en el progreso"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 10:
            return PlateauReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                plateaus=[],
                recommendations=["Necesitas más datos para detectar mesetas"],
                action_items=[]
            )
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            d for d in data_points
            if datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
        
        if len(recent_data) < 10:
            return PlateauReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                plateaus=[],
                recommendations=["Necesitas más datos recientes"],
                action_items=[]
            )
        
        plateaus = []
        
        # Analizar cada métrica
        for metric_name in ["overall_score", "hydration_score", "texture_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            timestamps = [d["timestamp"] for d in recent_data]
            
            if len(values) < 10:
                continue
            
            # Detectar meseta usando análisis de varianza
            plateau_info = self._detect_plateau_in_metric(values, timestamps, metric_name)
            
            if plateau_info:
                plateaus.append(plateau_info)
        
        # Recomendaciones
        recommendations = []
        action_items = []
        
        if plateaus:
            recommendations.append(f"Se detectaron {len(plateaus)} meseta(s) en tu progreso")
            
            for plateau in plateaus:
                if plateau.plateau_duration_days > 30:
                    recommendations.append(
                        f"Meseta en {plateau.metric_name} por {plateau.plateau_duration_days} días. "
                        "Considera ajustar tu rutina"
                    )
                    action_items.append(f"Revisa y ajusta tratamiento para {plateau.metric_name}")
                    action_items.append("Considera consultar con un dermatólogo")
                elif plateau.plateau_duration_days > 14:
                    recommendations.append(
                        f"Posible meseta en {plateau.metric_name}. Monitorea de cerca"
                    )
        else:
            recommendations.append("No se detectaron mesetas significativas")
            action_items.append("Continúa con tu rutina actual")
        
        report = PlateauReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            plateaus=plateaus,
            recommendations=recommendations,
            action_items=action_items
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _detect_plateau_in_metric(self, values: List[float], timestamps: List[str], 
                                  metric_name: str) -> Optional[PlateauData]:
        """Detecta meseta en una métrica específica"""
        if len(values) < 10:
            return None
        
        # Dividir en ventanas y analizar varianza
        window_size = min(7, len(values) // 3)
        
        for i in range(len(values) - window_size):
            window_values = values[i:i+window_size]
            window_timestamps = timestamps[i:i+window_size]
            
            # Calcular varianza
            variance = statistics.variance(window_values) if len(window_values) > 1 else 0
            
            # Si la varianza es muy baja, es una meseta
            if variance < 5.0:  # Umbral de varianza
                avg_value = statistics.mean(window_values)
                start_date = window_timestamps[0]
                
                # Calcular duración
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(window_timestamps[-1])
                duration_days = (end_dt - start_dt).days
                
                # Determinar significancia
                if duration_days > 30:
                    significance = "high"
                elif duration_days > 14:
                    significance = "medium"
                else:
                    significance = "low"
                
                return PlateauData(
                    metric_name=metric_name,
                    plateau_start_date=start_date,
                    plateau_duration_days=duration_days,
                    average_value=avg_value,
                    variation=variance,
                    significance=significance
                )
        
        return None
    
    def get_user_reports(self, user_id: str) -> List[PlateauReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])

