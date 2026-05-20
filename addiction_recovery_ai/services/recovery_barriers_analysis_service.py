"""
Servicio de Análisis de Barreras de Recuperación - Sistema completo de análisis de barreras
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import statistics


class RecoveryBarriersAnalysisService:
    """Servicio de análisis de barreras de recuperación"""
    
    def __init__(self):
        """Inicializa el servicio de barreras"""
        pass
    
    def identify_barriers(
        self,
        user_id: str,
        user_data: Dict
    ) -> Dict:
        """
        Identifica barreras de recuperación
        
        Args:
            user_id: ID del usuario
            user_data: Datos del usuario
        
        Returns:
            Barreras identificadas
        """
        barriers = self._detect_barriers(user_data)
        
        return {
            "user_id": user_id,
            "analysis_id": f"barriers_{datetime.now().timestamp()}",
            "barriers": barriers,
            "total_barriers": len(barriers),
            "severity": self._calculate_severity(barriers),
            "barrier_categories": self._categorize_barriers(barriers),
            "recommendations": self._generate_barrier_recommendations(barriers),
            "identified_at": datetime.now().isoformat()
        }
    
    def analyze_barrier_impact(
        self,
        user_id: str,
        barriers: List[Dict],
        recovery_progress: List[Dict]
    ) -> Dict:
        """
        Analiza impacto de barreras
        
        Args:
            user_id: ID del usuario
            barriers: Lista de barreras
            recovery_progress: Progreso de recuperación
        
        Returns:
            Análisis de impacto
        """
        return {
            "user_id": user_id,
            "total_barriers": len(barriers),
            "impact_score": self._calculate_impact_score(barriers, recovery_progress),
            "most_significant_barriers": self._identify_significant_barriers(barriers),
            "barrier_progression": self._analyze_barrier_progression(barriers),
            "recommendations": self._generate_impact_recommendations(barriers, recovery_progress),
            "generated_at": datetime.now().isoformat()
        }
    
    def suggest_barrier_solutions(
        self,
        user_id: str,
        barrier: Dict
    ) -> Dict:
        """
        Sugiere soluciones para barrera
        
        Args:
            user_id: ID del usuario
            barrier: Barrera específica
        
        Returns:
            Soluciones sugeridas
        """
        solutions = self._generate_solutions(barrier)
        
        return {
            "user_id": user_id,
            "barrier": barrier,
            "solutions": solutions,
            "recommended_solution": solutions[0] if solutions else None,
            "implementation_steps": self._create_implementation_steps(barrier, solutions),
            "generated_at": datetime.now().isoformat()
        }
    
    def _detect_barriers(self, data: Dict) -> List[Dict]:
        """Detecta barreras"""
        barriers = []
        
        # Barreras financieras
        financial_stress = data.get("financial_stress", 5)
        if financial_stress >= 7:
            barriers.append({
                "type": "financial",
                "description": "Estrés financiero",
                "severity": "high" if financial_stress >= 8 else "medium"
            })
        
        # Barreras sociales
        support_level = data.get("support_level", 5)
        if support_level < 4:
            barriers.append({
                "type": "social",
                "description": "Falta de apoyo social",
                "severity": "high" if support_level < 3 else "medium"
            })
        
        # Barreras de acceso
        access_to_care = data.get("access_to_care", 5)
        if access_to_care < 4:
            barriers.append({
                "type": "access",
                "description": "Acceso limitado a servicios de salud",
                "severity": "medium"
            })
        
        # Barreras emocionales
        emotional_stability = data.get("emotional_stability", 5)
        if emotional_stability < 4:
            barriers.append({
                "type": "emotional",
                "description": "Inestabilidad emocional",
                "severity": "high" if emotional_stability < 3 else "medium"
            })
        
        return barriers
    
    def _calculate_severity(self, barriers: List[Dict]) -> str:
        """Calcula severidad general"""
        if not barriers:
            return "none"
        
        high_severity = sum(1 for b in barriers if b.get("severity") == "high")
        
        if high_severity >= 2:
            return "high"
        elif high_severity >= 1:
            return "medium"
        else:
            return "low"
    
    def _categorize_barriers(self, barriers: List[Dict]) -> Dict:
        """Categoriza barreras"""
        categories = defaultdict(int)
        
        for barrier in barriers:
            barrier_type = barrier.get("type", "unknown")
            categories[barrier_type] += 1
        
        return dict(categories)
    
    def _generate_barrier_recommendations(self, barriers: List[Dict]) -> List[str]:
        """Genera recomendaciones de barreras"""
        recommendations = []
        
        financial_barriers = [b for b in barriers if b.get("type") == "financial"]
        if financial_barriers:
            recommendations.append("Busca recursos financieros y programas de asistencia")
        
        social_barriers = [b for b in barriers if b.get("type") == "social"]
        if social_barriers:
            recommendations.append("Fortalecer tu red de apoyo social es crucial")
        
        return recommendations
    
    def _calculate_impact_score(self, barriers: List[Dict], progress: List[Dict]) -> float:
        """Calcula puntuación de impacto"""
        base_impact = 0.3
        
        high_severity = sum(1 for b in barriers if b.get("severity") == "high")
        base_impact += high_severity * 0.2
        
        return min(1.0, base_impact)
    
    def _identify_significant_barriers(self, barriers: List[Dict]) -> List[Dict]:
        """Identifica barreras más significativas"""
        significant = [b for b in barriers if b.get("severity") == "high"]
        return significant[:3]
    
    def _analyze_barrier_progression(self, barriers: List[Dict]) -> Dict:
        """Analiza progresión de barreras"""
        return {
            "trend": "stable",
            "new_barriers": 0,
            "resolved_barriers": 0
        }
    
    def _generate_impact_recommendations(self, barriers: List[Dict], progress: List[Dict]) -> List[str]:
        """Genera recomendaciones de impacto"""
        recommendations = []
        
        if len(barriers) >= 3:
            recommendations.append("Múltiples barreras detectadas. Considera apoyo profesional")
        
        return recommendations
    
    def _generate_solutions(self, barrier: Dict) -> List[Dict]:
        """Genera soluciones"""
        solutions = []
        
        barrier_type = barrier.get("type")
        
        if barrier_type == "financial":
            solutions.append({
                "solution": "Buscar programas de asistencia financiera",
                "priority": "high"
            })
        elif barrier_type == "social":
            solutions.append({
                "solution": "Unirse a grupos de apoyo",
                "priority": "high"
            })
        elif barrier_type == "emotional":
            solutions.append({
                "solution": "Terapia y técnicas de regulación emocional",
                "priority": "high"
            })
        
        return solutions
    
    def _create_implementation_steps(self, barrier: Dict, solutions: List[Dict]) -> List[str]:
        """Crea pasos de implementación"""
        steps = []
        
        if solutions:
            steps.append(f"1. Implementar solución: {solutions[0].get('solution', '')}")
            steps.append("2. Monitorear progreso")
            steps.append("3. Ajustar estrategia según sea necesario")
        
        return steps

