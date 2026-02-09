"""
Sistema de seguimiento de cambios hormonales
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class HormonalRecord:
    """Registro hormonal"""
    id: str
    user_id: str
    record_date: str
    cycle_day: Optional[int] = None  # Para mujeres: día del ciclo menstrual
    hormonal_state: str  # "menstrual", "follicular", "ovulation", "luteal", "menopause", "pregnancy", "normal"
    skin_condition: str  # "clear", "mild_breakout", "moderate_breakout", "severe_breakout", "dry", "oily"
    symptoms: List[str] = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.symptoms is None:
            self.symptoms = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "record_date": self.record_date,
            "cycle_day": self.cycle_day,
            "hormonal_state": self.hormonal_state,
            "skin_condition": self.skin_condition,
            "symptoms": self.symptoms,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class HormonalAnalysis:
    """Análisis hormonal"""
    user_id: str
    pattern_detected: bool
    hormonal_pattern: Dict
    skin_condition_by_state: Dict[str, str]
    recommendations: List[str]
    predicted_next_breakout: Optional[str] = None
    days_analyzed: int = 0
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "pattern_detected": self.pattern_detected,
            "hormonal_pattern": self.hormonal_pattern,
            "skin_condition_by_state": self.skin_condition_by_state,
            "recommendations": self.recommendations,
            "predicted_next_breakout": self.predicted_next_breakout,
            "days_analyzed": self.days_analyzed
        }


class HormonalTracker:
    """Sistema de seguimiento de cambios hormonales"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[HormonalRecord]] = {}  # user_id -> [records]
    
    def add_hormonal_record(self, user_id: str, record_date: str,
                           hormonal_state: str, skin_condition: str,
                           cycle_day: Optional[int] = None,
                           symptoms: Optional[List[str]] = None,
                           notes: Optional[str] = None) -> HormonalRecord:
        """Agrega registro hormonal"""
        record = HormonalRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            record_date=record_date,
            cycle_day=cycle_day,
            hormonal_state=hormonal_state,
            skin_condition=skin_condition,
            symptoms=symptoms or [],
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_hormonal_patterns(self, user_id: str, days: int = 90) -> HormonalAnalysis:
        """Analiza patrones hormonales"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return HormonalAnalysis(
                user_id=user_id,
                pattern_detected=False,
                hormonal_pattern={},
                skin_condition_by_state={},
                recommendations=["Agrega registros hormonales para análisis"],
                days_analyzed=0
            )
        
        # Filtrar por días
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r.record_date).date() >= cutoff
        ]
        
        if len(recent_records) < 5:
            return HormonalAnalysis(
                user_id=user_id,
                pattern_detected=False,
                hormonal_pattern={},
                skin_condition_by_state={},
                recommendations=["Necesitas más registros para detectar patrones"],
                days_analyzed=len(recent_records)
            )
        
        # Agrupar por estado hormonal
        by_state = {}
        for record in recent_records:
            state = record.hormonal_state
            if state not in by_state:
                by_state[state] = []
            by_state[state].append(record.skin_condition)
        
        # Determinar condición de piel por estado
        skin_condition_by_state = {}
        for state, conditions in by_state.items():
            # Contar ocurrencias
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            # Obtener condición más común
            most_common = max(condition_counts.items(), key=lambda x: x[1])[0]
            skin_condition_by_state[state] = most_common
        
        # Detectar patrones
        pattern_detected = len(by_state) >= 2
        
        # Patrón hormonal
        hormonal_pattern = {
            "states_tracked": list(by_state.keys()),
            "most_problematic_state": None,
            "cycle_length": None
        }
        
        # Encontrar estado más problemático
        problematic_states = [
            state for state, condition in skin_condition_by_state.items()
            if "breakout" in condition
        ]
        if problematic_states:
            hormonal_pattern["most_problematic_state"] = problematic_states[0]
        
        # Recomendaciones
        recommendations = []
        
        if "ovulation" in skin_condition_by_state and "breakout" in skin_condition_by_state["ovulation"]:
            recommendations.append("Brotes durante ovulación detectados. Prepara productos para acné antes de este período")
        
        if "menstrual" in skin_condition_by_state and "breakout" in skin_condition_by_state["menstrual"]:
            recommendations.append("Brotes menstruales detectados. Usa productos suaves durante este período")
        
        if pattern_detected:
            recommendations.append("Patrón hormonal detectado. Ajusta tu rutina según tu ciclo")
        
        # Predicción de próximo brote
        predicted_next_breakout = None
        if hormonal_pattern["most_problematic_state"]:
            recommendations.append(f"Prepara productos para acné antes del período de {hormonal_pattern['most_problematic_state']}")
            predicted_next_breakout = hormonal_pattern["most_problematic_state"]
        
        return HormonalAnalysis(
            user_id=user_id,
            pattern_detected=pattern_detected,
            hormonal_pattern=hormonal_pattern,
            skin_condition_by_state=skin_condition_by_state,
            recommendations=recommendations,
            predicted_next_breakout=predicted_next_breakout,
            days_analyzed=len(recent_records)
        )






