"""
Sistema de seguimiento de cambios estacionales
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class SeasonalRecord:
    """Registro estacional"""
    id: str
    user_id: str
    record_date: str
    season: str  # "spring", "summer", "fall", "winter"
    skin_condition: Dict
    environmental_factors: Dict = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.environmental_factors is None:
            self.environmental_factors = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "record_date": self.record_date,
            "season": self.season,
            "skin_condition": self.skin_condition,
            "environmental_factors": self.environmental_factors,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class SeasonalAnalysis:
    """Análisis estacional"""
    user_id: str
    seasonal_patterns: Dict[str, Dict]
    most_problematic_season: Optional[str] = None
    best_season: Optional[str] = None
    recommendations: List[str] = None
    seasonal_routine_suggestions: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.seasonal_routine_suggestions is None:
            self.seasonal_routine_suggestions = {}
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "seasonal_patterns": self.seasonal_patterns,
            "most_problematic_season": self.most_problematic_season,
            "best_season": self.best_season,
            "recommendations": self.recommendations,
            "seasonal_routine_suggestions": self.seasonal_routine_suggestions
        }


class SeasonalChangesTracker:
    """Sistema de seguimiento de cambios estacionales"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[SeasonalRecord]] = {}
    
    def add_seasonal_record(self, user_id: str, record_date: str, season: str,
                           skin_condition: Dict, environmental_factors: Optional[Dict] = None,
                           notes: Optional[str] = None) -> SeasonalRecord:
        """Agrega registro estacional"""
        record = SeasonalRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            record_date=record_date,
            season=season,
            skin_condition=skin_condition,
            environmental_factors=environmental_factors or {},
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_seasonal_patterns(self, user_id: str) -> SeasonalAnalysis:
        """Analiza patrones estacionales"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return SeasonalAnalysis(
                user_id=user_id,
                seasonal_patterns={},
                recommendations=["Agrega registros estacionales para análisis"]
            )
        
        # Agrupar por estación
        by_season = {}
        for record in user_records:
            season = record.season
            if season not in by_season:
                by_season[season] = []
            by_season[season].append(record)
        
        # Analizar cada estación
        seasonal_patterns = {}
        season_scores = {}
        
        for season, records in by_season.items():
            # Calcular promedio de scores
            scores = []
            for record in records:
                overall = record.skin_condition.get("overall_score", 0)
                scores.append(overall)
            
            avg_score = statistics.mean(scores) if scores else 0.0
            
            seasonal_patterns[season] = {
                "average_score": avg_score,
                "record_count": len(records),
                "common_concerns": self._extract_common_concerns(records)
            }
            
            season_scores[season] = avg_score
        
        # Determinar estación más problemática y mejor
        most_problematic = None
        best_season = None
        
        if season_scores:
            most_problematic = min(season_scores.items(), key=lambda x: x[1])[0]
            best_season = max(season_scores.items(), key=lambda x: x[1])[0]
        
        # Recomendaciones
        recommendations = []
        
        if most_problematic:
            recommendations.append(f"La estación más problemática es {most_problematic}")
            recommendations.extend(self._get_seasonal_recommendations(most_problematic))
        
        # Sugerencias de rutina por estación
        routine_suggestions = {}
        for season in ["spring", "summer", "fall", "winter"]:
            routine_suggestions[season] = self._get_seasonal_routine(season)
        
        return SeasonalAnalysis(
            user_id=user_id,
            seasonal_patterns=seasonal_patterns,
            most_problematic_season=most_problematic,
            best_season=best_season,
            recommendations=recommendations,
            seasonal_routine_suggestions=routine_suggestions
        )
    
    def _extract_common_concerns(self, records: List[SeasonalRecord]) -> List[str]:
        """Extrae preocupaciones comunes"""
        concerns = []
        for record in records:
            condition = record.skin_condition
            if condition.get("acne_score", 0) > 60:
                concerns.append("acne")
            if condition.get("dryness_score", 0) > 70:
                concerns.append("dryness")
            if condition.get("sensitivity_score", 0) > 60:
                concerns.append("sensitivity")
        
        concern_counts = {}
        for concern in concerns:
            concern_counts[concern] = concern_counts.get(concern, 0) + 1
        
        return sorted(concern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _get_seasonal_recommendations(self, season: str) -> List[str]:
        """Obtiene recomendaciones para estación"""
        recommendations = {
            "winter": [
                "Hidratación intensa necesaria",
                "Protección contra viento y frío",
                "Considera humidificador"
            ],
            "summer": [
                "Protección solar extrema",
                "Hidratación ligera",
                "Productos refrescantes"
            ],
            "spring": [
                "Transición de productos",
                "Preparación para mayor exposición solar",
                "Renovación celular"
            ],
            "fall": [
                "Preparación para invierno",
                "Reparación de daño estival",
                "Hidratación moderada"
            ]
        }
        return recommendations.get(season, [])
    
    def _get_seasonal_routine(self, season: str) -> List[str]:
        """Obtiene rutina sugerida para estación"""
        routines = {
            "winter": ["Limpieza suave", "Hidratante rica", "Protector solar", "Aceite facial"],
            "summer": ["Limpieza refrescante", "Hidratante ligera", "SPF 50+", "Serum antioxidante"],
            "spring": ["Exfoliación suave", "Hidratante balanceada", "SPF 30+", "Serum renovador"],
            "fall": ["Limpieza profunda", "Hidratante reparadora", "SPF 30+", "Serum reparador"]
        }
        return routines.get(season, [])


