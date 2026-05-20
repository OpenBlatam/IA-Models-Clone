"""
Sistema de seguimiento de factores ambientales
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class EnvironmentalRecord:
    """Registro ambiental"""
    id: str
    user_id: str
    record_date: str
    location: Optional[str] = None
    air_quality_index: Optional[int] = None
    pollution_level: str = "unknown"  # "low", "medium", "high"
    humidity: Optional[float] = None
    temperature: Optional[float] = None
    uv_index: Optional[float] = None
    pollen_level: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "record_date": self.record_date,
            "location": self.location,
            "air_quality_index": self.air_quality_index,
            "pollution_level": self.pollution_level,
            "humidity": self.humidity,
            "temperature": self.temperature,
            "uv_index": self.uv_index,
            "pollen_level": self.pollen_level,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class EnvironmentalAnalysis:
    """Análisis ambiental"""
    user_id: str
    average_air_quality: Optional[float] = None
    average_pollution: str = "unknown"
    average_humidity: Optional[float] = None
    average_uv: Optional[float] = None
    skin_impact: Dict = None
    recommendations: List[str] = None
    days_analyzed: int = 0
    
    def __post_init__(self):
        if self.skin_impact is None:
            self.skin_impact = {}
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "average_air_quality": self.average_air_quality,
            "average_pollution": self.average_pollution,
            "average_humidity": self.average_humidity,
            "average_uv": self.average_uv,
            "skin_impact": self.skin_impact,
            "recommendations": self.recommendations,
            "days_analyzed": self.days_analyzed
        }


class EnvironmentalTracker:
    """Sistema de seguimiento de factores ambientales"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[EnvironmentalRecord]] = {}
    
    def add_environmental_record(self, user_id: str, record_date: str,
                                location: Optional[str] = None,
                                air_quality_index: Optional[int] = None,
                                pollution_level: str = "unknown",
                                humidity: Optional[float] = None,
                                temperature: Optional[float] = None,
                                uv_index: Optional[float] = None,
                                pollen_level: Optional[str] = None,
                                notes: Optional[str] = None) -> EnvironmentalRecord:
        """Agrega registro ambiental"""
        record = EnvironmentalRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            record_date=record_date,
            location=location,
            air_quality_index=air_quality_index,
            pollution_level=pollution_level,
            humidity=humidity,
            temperature=temperature,
            uv_index=uv_index,
            pollen_level=pollen_level,
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_environmental_impact(self, user_id: str, days: int = 30) -> EnvironmentalAnalysis:
        """Analiza impacto ambiental"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return EnvironmentalAnalysis(
                user_id=user_id,
                recommendations=["Agrega registros ambientales para análisis"]
            )
        
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r.record_date).date() >= cutoff
        ]
        
        if not recent_records:
            return EnvironmentalAnalysis(
                user_id=user_id,
                recommendations=["No hay registros recientes"]
            )
        
        # Calcular promedios
        aqi_values = [r.air_quality_index for r in recent_records if r.air_quality_index]
        humidity_values = [r.humidity for r in recent_records if r.humidity]
        uv_values = [r.uv_index for r in recent_records if r.uv_index]
        
        avg_aqi = statistics.mean(aqi_values) if aqi_values else None
        avg_humidity = statistics.mean(humidity_values) if humidity_values else None
        avg_uv = statistics.mean(uv_values) if uv_values else None
        
        # Determinar nivel de contaminación promedio
        pollution_levels = [r.pollution_level for r in recent_records if r.pollution_level != "unknown"]
        if pollution_levels:
            avg_pollution = max(set(pollution_levels), key=pollution_levels.count)
        else:
            avg_pollution = "unknown"
        
        # Impacto en la piel
        skin_impact = {
            "pollution_risk": "high" if avg_pollution == "high" else "medium" if avg_pollution == "medium" else "low",
            "hydration_impact": "negative" if avg_humidity and avg_humidity < 30 else "positive" if avg_humidity and avg_humidity > 70 else "neutral",
            "uv_risk": "high" if avg_uv and avg_uv > 6 else "medium" if avg_uv and avg_uv > 3 else "low"
        }
        
        # Recomendaciones
        recommendations = []
        
        if avg_pollution == "high":
            recommendations.append("Alta contaminación detectada. Usa productos con antioxidantes")
            recommendations.append("Limpieza doble recomendada para remover partículas")
        
        if avg_humidity and avg_humidity < 30:
            recommendations.append("Humedad baja. Hidratación intensa necesaria")
        
        if avg_uv and avg_uv > 6:
            recommendations.append("Índice UV alto. Protección solar extrema necesaria")
        
        if avg_aqi and avg_aqi > 100:
            recommendations.append("Calidad del aire pobre. Considera purificador de aire")
        
        return EnvironmentalAnalysis(
            user_id=user_id,
            average_air_quality=avg_aqi,
            average_pollution=avg_pollution,
            average_humidity=avg_humidity,
            average_uv=avg_uv,
            skin_impact=skin_impact,
            recommendations=recommendations,
            days_analyzed=len(recent_records)
        )






