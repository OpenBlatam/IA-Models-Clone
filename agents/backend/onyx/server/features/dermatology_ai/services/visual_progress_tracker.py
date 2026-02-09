"""
Sistema de seguimiento de progreso visual
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class VisualProgressEntry:
    """Entrada de progreso visual"""
    id: str
    user_id: str
    image_url: str
    date: str
    analysis_data: Dict
    improvement_metrics: Dict
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
            "image_url": self.image_url,
            "date": self.date,
            "analysis_data": self.analysis_data,
            "improvement_metrics": self.improvement_metrics,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class ProgressTimeline:
    """Timeline de progreso"""
    user_id: str
    entries: List[VisualProgressEntry]
    overall_improvement: float
    key_milestones: List[Dict]
    trend: str  # "improving", "stable", "declining"
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "entries": [e.to_dict() for e in self.entries],
            "overall_improvement": self.overall_improvement,
            "key_milestones": self.key_milestones,
            "trend": self.trend
        }


class VisualProgressTracker:
    """Sistema de seguimiento de progreso visual"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.entries: Dict[str, List[VisualProgressEntry]] = {}  # user_id -> [entries]
    
    def add_progress_entry(self, user_id: str, image_url: str, date: str,
                          analysis_data: Dict, notes: Optional[str] = None) -> VisualProgressEntry:
        """Agrega entrada de progreso"""
        # Calcular métricas de mejora comparando con entrada anterior
        improvement_metrics = self._calculate_improvement(user_id, analysis_data)
        
        entry = VisualProgressEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_url=image_url,
            date=date,
            analysis_data=analysis_data,
            improvement_metrics=improvement_metrics,
            notes=notes
        )
        
        if user_id not in self.entries:
            self.entries[user_id] = []
        
        self.entries[user_id].append(entry)
        return entry
    
    def _calculate_improvement(self, user_id: str, current_analysis: Dict) -> Dict:
        """Calcula mejora comparando con entrada anterior"""
        user_entries = self.entries.get(user_id, [])
        
        if not user_entries:
            return {"improvement": 0.0, "message": "Primera entrada"}
        
        previous_entry = user_entries[-1]
        previous_scores = previous_entry.analysis_data.get("quality_scores", {})
        current_scores = current_analysis.get("quality_scores", {})
        
        improvements = {}
        
        for metric in ["overall_score", "hydration_score", "texture_score"]:
            prev_val = previous_scores.get(metric, 0)
            curr_val = current_scores.get(metric, 0)
            
            if prev_val > 0:
                change = ((curr_val - prev_val) / prev_val) * 100
                improvements[metric] = {
                    "previous": prev_val,
                    "current": curr_val,
                    "change_percentage": change
                }
        
        overall_change = improvements.get("overall_score", {}).get("change_percentage", 0)
        
        return {
            "improvement": overall_change,
            "metrics": improvements,
            "message": "Mejora significativa" if overall_change > 5 else "Progreso estable" if overall_change > -5 else "Requiere atención"
        }
    
    def get_progress_timeline(self, user_id: str) -> ProgressTimeline:
        """Obtiene timeline de progreso"""
        user_entries = self.entries.get(user_id, [])
        user_entries.sort(key=lambda x: x.date)
        
        if len(user_entries) < 2:
            return ProgressTimeline(
                user_id=user_id,
                entries=user_entries,
                overall_improvement=0.0,
                key_milestones=[],
                trend="stable"
            )
        
        # Calcular mejora general
        first_entry = user_entries[0]
        last_entry = user_entries[-1]
        
        first_score = first_entry.analysis_data.get("quality_scores", {}).get("overall_score", 0)
        last_score = last_entry.analysis_data.get("quality_scores", {}).get("overall_score", 0)
        
        if first_score > 0:
            overall_improvement = ((last_score - first_score) / first_score) * 100
        else:
            overall_improvement = 0.0
        
        # Identificar milestones
        milestones = []
        for i, entry in enumerate(user_entries):
            score = entry.analysis_data.get("quality_scores", {}).get("overall_score", 0)
            if score >= 80 and i > 0:
                milestones.append({
                    "date": entry.date,
                    "achievement": "Score excelente alcanzado",
                    "score": score
                })
        
        # Determinar tendencia
        if overall_improvement > 10:
            trend = "improving"
        elif overall_improvement < -10:
            trend = "declining"
        else:
            trend = "stable"
        
        return ProgressTimeline(
            user_id=user_id,
            entries=user_entries,
            overall_improvement=overall_improvement,
            key_milestones=milestones,
            trend=trend
        )
    
    def get_user_entries(self, user_id: str) -> List[VisualProgressEntry]:
        """Obtiene entradas del usuario"""
        return self.entries.get(user_id, [])






