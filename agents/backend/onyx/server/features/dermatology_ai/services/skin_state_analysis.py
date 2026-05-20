"""
Sistema de análisis de fotos con diferentes estados de la piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class SkinStateData:
    """Análisis con estado específico de la piel"""
    state_type: str  # "clean", "makeup", "post_workout", "post_shower", "stressed", "tired"
    image_url: str
    analysis_data: Dict
    state_impact: Dict
    recommendations: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "state_type": self.state_type,
            "image_url": self.image_url,
            "analysis_data": self.analysis_data,
            "state_impact": self.state_impact,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }


@dataclass
class SkinStateReport:
    """Reporte de estados de la piel"""
    id: str
    user_id: str
    analyses: List[SkinStateData]
    optimal_state: str
    state_comparison: Dict
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "analyses": [a.to_dict() for a in self.analyses],
            "optimal_state": self.optimal_state,
            "state_comparison": self.state_comparison,
            "created_at": self.created_at
        }


class SkinStateAnalysisSystem:
    """Sistema de análisis con diferentes estados de la piel"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[SkinStateReport]] = {}  # user_id -> [reports]
    
    def analyze_with_state(self, user_id: str, state_data: List[Dict]) -> SkinStateReport:
        """Analiza con diferentes estados de la piel"""
        analyses = []
        
        for state in state_data:
            state_type = state.get("state_type", "clean")
            image_url = state.get("image_url", "")
            
            # Análisis de impacto según estado
            state_impact = self._calculate_state_impact(state_type)
            
            analysis_data = {
                "quality_scores": {
                    "overall_score": 75.0 + state_impact.get("score_adjustment", 0),
                    "hydration_score": 65.0 + state_impact.get("hydration_adjustment", 0),
                    "texture_score": 70.0 + state_impact.get("texture_adjustment", 0)
                }
            }
            
            recommendations = self._generate_state_recommendations(state_type)
            
            analysis = SkinStateData(
                state_type=state_type,
                image_url=image_url,
                analysis_data=analysis_data,
                state_impact=state_impact,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Determinar estado óptimo
        optimal_state = "clean"  # Por defecto
        
        # Comparación de estados
        state_comparison = {
            "best_hydration": self._find_best_state(analyses, "hydration_score"),
            "best_texture": self._find_best_state(analyses, "texture_score"),
            "most_accurate": "clean"  # Estado limpio es más preciso
        }
        
        report = SkinStateReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_state=optimal_state,
            state_comparison=state_comparison
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _calculate_state_impact(self, state_type: str) -> Dict:
        """Calcula impacto según estado"""
        impacts = {
            "clean": {"score_adjustment": 0, "hydration_adjustment": 0, "texture_adjustment": 0},
            "makeup": {"score_adjustment": -5, "hydration_adjustment": -3, "texture_adjustment": -8},
            "post_workout": {"score_adjustment": -3, "hydration_adjustment": -5, "texture_adjustment": 2},
            "post_shower": {"score_adjustment": 5, "hydration_adjustment": 10, "texture_adjustment": 3},
            "stressed": {"score_adjustment": -8, "hydration_adjustment": -5, "texture_adjustment": -5},
            "tired": {"score_adjustment": -5, "hydration_adjustment": -3, "texture_adjustment": -3}
        }
        return impacts.get(state_type, impacts["clean"])
    
    def _generate_state_recommendations(self, state_type: str) -> List[str]:
        """Genera recomendaciones según estado"""
        recommendations = {
            "clean": ["Estado ideal para análisis preciso"],
            "makeup": ["Remueve el maquillaje completamente antes del análisis"],
            "post_workout": ["Limpia el sudor y espera 30 minutos antes de analizar"],
            "post_shower": ["Espera 15 minutos después de la ducha para análisis más preciso"],
            "stressed": ["El estrés afecta la piel. Practica técnicas de relajación"],
            "tired": ["El descanso es importante para la salud de la piel"]
        }
        return recommendations.get(state_type, [])
    
    def _find_best_state(self, analyses: List[SkinStateData], metric: str) -> str:
        """Encuentra el mejor estado para una métrica"""
        if not analyses:
            return "clean"
        
        best_analysis = max(analyses, key=lambda a: a.analysis_data.get("quality_scores", {}).get(metric, 0))
        return best_analysis.state_type
    
    def get_user_reports(self, user_id: str) -> List[SkinStateReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])

