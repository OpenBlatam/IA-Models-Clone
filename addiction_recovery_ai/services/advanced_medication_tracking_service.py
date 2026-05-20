"""
Servicio de Seguimiento de Medicamentos Avanzado - Sistema completo de seguimiento de medicamentos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedMedicationTrackingService:
    """Servicio de seguimiento de medicamentos avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de medicamentos"""
        pass
    
    def register_medication(
        self,
        user_id: str,
        medication_data: Dict
    ) -> Dict:
        """
        Registra medicamento
        
        Args:
            user_id: ID del usuario
            medication_data: Datos del medicamento
        
        Returns:
            Medicamento registrado
        """
        medication = {
            "id": f"medication_{datetime.now().timestamp()}",
            "user_id": user_id,
            "medication_data": medication_data,
            "name": medication_data.get("name", ""),
            "dosage": medication_data.get("dosage", ""),
            "frequency": medication_data.get("frequency", "daily"),
            "prescribed_by": medication_data.get("prescribed_by", ""),
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        return medication
    
    def record_medication_dose(
        self,
        user_id: str,
        medication_id: str,
        dose_data: Dict
    ) -> Dict:
        """
        Registra dosis de medicamento
        
        Args:
            user_id: ID del usuario
            medication_id: ID del medicamento
            dose_data: Datos de la dosis
        
        Returns:
            Dosis registrada
        """
        return {
            "user_id": user_id,
            "medication_id": medication_id,
            "dose_id": f"dose_{datetime.now().timestamp()}",
            "dose_data": dose_data,
            "taken_at": dose_data.get("taken_at", datetime.now().isoformat()),
            "was_taken": dose_data.get("was_taken", True),
            "side_effects": dose_data.get("side_effects", []),
            "recorded_at": datetime.now().isoformat()
        }
    
    def analyze_medication_adherence(
        self,
        user_id: str,
        medication_id: str,
        doses: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza adherencia a medicamentos
        
        Args:
            user_id: ID del usuario
            medication_id: ID del medicamento
            doses: Lista de dosis
            days: Número de días
        
        Returns:
            Análisis de adherencia
        """
        if not doses:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        taken_doses = sum(1 for d in doses if d.get("was_taken", False))
        total_expected = days * 1  # Asumiendo 1 dosis por día
        
        adherence_rate = (taken_doses / total_expected * 100) if total_expected > 0 else 0
        
        return {
            "user_id": user_id,
            "medication_id": medication_id,
            "period_days": days,
            "total_expected_doses": total_expected,
            "taken_doses": taken_doses,
            "missed_doses": total_expected - taken_doses,
            "adherence_rate": round(adherence_rate, 2),
            "adherence_level": "excellent" if adherence_rate >= 90 else "good" if adherence_rate >= 75 else "needs_improvement",
            "side_effects_summary": self._analyze_side_effects(doses),
            "recommendations": self._generate_adherence_recommendations(adherence_rate),
            "generated_at": datetime.now().isoformat()
        }
    
    def check_medication_interactions(
        self,
        user_id: str,
        medications: List[Dict]
    ) -> Dict:
        """
        Verifica interacciones de medicamentos
        
        Args:
            user_id: ID del usuario
            medications: Lista de medicamentos
        
        Returns:
            Análisis de interacciones
        """
        return {
            "user_id": user_id,
            "total_medications": len(medications),
            "interactions_found": self._check_interactions(medications),
            "warnings": self._generate_warnings(medications),
            "recommendations": self._generate_interaction_recommendations(medications),
            "checked_at": datetime.now().isoformat()
        }
    
    def _analyze_side_effects(self, doses: List[Dict]) -> Dict:
        """Analiza efectos secundarios"""
        side_effects = []
        
        for dose in doses:
            effects = dose.get("side_effects", [])
            side_effects.extend(effects)
        
        from collections import Counter
        effect_counts = Counter(side_effects)
        
        return {
            "most_common": dict(effect_counts.most_common(3)),
            "total_reported": len(side_effects)
        }
    
    def _generate_adherence_recommendations(self, adherence_rate: float) -> List[str]:
        """Genera recomendaciones de adherencia"""
        recommendations = []
        
        if adherence_rate < 75:
            recommendations.append("Considera configurar recordatorios para tomar medicamentos")
            recommendations.append("Habla con tu médico sobre cualquier dificultad para seguir el régimen")
        
        return recommendations
    
    def _check_interactions(self, medications: List[Dict]) -> List[Dict]:
        """Verifica interacciones"""
        interactions = []
        
        # Lógica simplificada
        if len(medications) >= 2:
            interactions.append({
                "medication1": medications[0].get("name", ""),
                "medication2": medications[1].get("name", ""),
                "severity": "moderate",
                "description": "Posible interacción detectada"
            })
        
        return interactions
    
    def _generate_warnings(self, medications: List[Dict]) -> List[str]:
        """Genera advertencias"""
        warnings = []
        
        interactions = self._check_interactions(medications)
        if interactions:
            warnings.append("⚠️ Interacciones de medicamentos detectadas. Consulta con tu médico")
        
        return warnings
    
    def _generate_interaction_recommendations(self, medications: List[Dict]) -> List[str]:
        """Genera recomendaciones de interacciones"""
        recommendations = []
        
        interactions = self._check_interactions(medications)
        if interactions:
            recommendations.append("Consulta con tu médico sobre posibles interacciones")
            recommendations.append("Informa a todos tus proveedores de salud sobre todos los medicamentos que tomas")
        
        return recommendations

