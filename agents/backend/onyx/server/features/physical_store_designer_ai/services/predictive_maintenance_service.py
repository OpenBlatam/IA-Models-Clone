"""
Predictive Maintenance Service - Sistema de mantenimiento predictivo
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class PredictiveMaintenanceService:
    """Servicio para mantenimiento predictivo"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.equipment: Dict[str, Dict[str, Any]] = {}
        self.maintenance_records: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: Dict[str, Dict[str, Any]] = {}
    
    def register_equipment(
        self,
        store_id: str,
        equipment_name: str,
        equipment_type: str,
        manufacturer: str,
        installation_date: str,
        expected_lifetime_days: int,
        maintenance_interval_days: int = 90
    ) -> Dict[str, Any]:
        """Registrar equipo"""
        
        equipment_id = f"eq_{store_id}_{len(self.equipment.get(store_id, [])) + 1}"
        
        equipment = {
            "equipment_id": equipment_id,
            "store_id": store_id,
            "name": equipment_name,
            "type": equipment_type,
            "manufacturer": manufacturer,
            "installation_date": installation_date,
            "expected_lifetime_days": expected_lifetime_days,
            "maintenance_interval_days": maintenance_interval_days,
            "status": "operational",
            "registered_at": datetime.now().isoformat()
        }
        
        if store_id not in self.equipment:
            self.equipment[store_id] = {}
        
        self.equipment[store_id][equipment_id] = equipment
        
        return equipment
    
    def record_maintenance(
        self,
        equipment_id: str,
        maintenance_type: str,  # "preventive", "corrective", "inspection"
        description: str,
        cost: Optional[float] = None
    ) -> Dict[str, Any]:
        """Registrar mantenimiento"""
        
        record = {
            "record_id": f"maint_{equipment_id}_{len(self.maintenance_records.get(equipment_id, [])) + 1}",
            "equipment_id": equipment_id,
            "type": maintenance_type,
            "description": description,
            "cost": cost,
            "performed_at": datetime.now().isoformat()
        }
        
        if equipment_id not in self.maintenance_records:
            self.maintenance_records[equipment_id] = []
        
        self.maintenance_records[equipment_id].append(record)
        
        # Actualizar estado del equipo
        equipment = self._find_equipment(equipment_id)
        if equipment:
            equipment["last_maintenance"] = datetime.now().isoformat()
            equipment["status"] = "operational"
        
        return record
    
    def _find_equipment(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Encontrar equipo"""
        for store_equipment in self.equipment.values():
            if equipment_id in store_equipment:
                return store_equipment[equipment_id]
        return None
    
    async def predict_maintenance_needs(
        self,
        equipment_id: str
    ) -> Dict[str, Any]:
        """Predecir necesidades de mantenimiento"""
        
        equipment = self._find_equipment(equipment_id)
        
        if not equipment:
            raise ValueError(f"Equipo {equipment_id} no encontrado")
        
        # Obtener historial de mantenimiento
        maintenance_history = self.maintenance_records.get(equipment_id, [])
        
        # Calcular días desde última mantenimiento
        last_maintenance = equipment.get("last_maintenance")
        if last_maintenance:
            days_since = (datetime.now() - datetime.fromisoformat(last_maintenance)).days
        else:
            installation = datetime.fromisoformat(equipment["installation_date"])
            days_since = (datetime.now() - installation).days
        
        # Predecir próxima mantenimiento
        interval = equipment["maintenance_interval_days"]
        days_until_next = interval - (days_since % interval)
        
        # Calcular probabilidad de falla
        lifetime_used = days_since / equipment["expected_lifetime_days"]
        failure_probability = min(100, lifetime_used * 100)
        
        prediction = {
            "equipment_id": equipment_id,
            "equipment_name": equipment["name"],
            "days_since_last_maintenance": days_since,
            "days_until_next_maintenance": days_until_next,
            "maintenance_due": days_until_next <= 7,
            "failure_probability": round(failure_probability, 2),
            "recommended_action": self._get_recommended_action(days_until_next, failure_probability),
            "predicted_at": datetime.now().isoformat()
        }
        
        self.predictions[equipment_id] = prediction
        
        return prediction
    
    def _get_recommended_action(
        self,
        days_until: int,
        failure_prob: float
    ) -> str:
        """Obtener acción recomendada"""
        if days_until <= 0:
            return "Maintenance overdue - schedule immediately"
        elif days_until <= 7:
            return "Schedule maintenance soon"
        elif failure_prob > 80:
            return "High failure risk - consider replacement"
        elif failure_prob > 60:
            return "Monitor closely"
        else:
            return "Normal operation"
    
    def get_maintenance_schedule(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Obtener calendario de mantenimiento"""
        
        equipment_list = self.equipment.get(store_id, {})
        
        schedule = {
            "store_id": store_id,
            "equipment": [],
            "overdue": [],
            "due_soon": [],
            "upcoming": []
        }
        
        for equipment_id, equipment in equipment_list.items():
            try:
                prediction = await self.predict_maintenance_needs(equipment_id)
                
                equipment_schedule = {
                    "equipment_id": equipment_id,
                    "name": equipment["name"],
                    "type": equipment["type"],
                    "days_until_maintenance": prediction["days_until_next_maintenance"],
                    "maintenance_due": prediction["maintenance_due"],
                    "recommended_action": prediction["recommended_action"]
                }
                
                schedule["equipment"].append(equipment_schedule)
                
                if prediction["days_until_next_maintenance"] < 0:
                    schedule["overdue"].append(equipment_schedule)
                elif prediction["days_until_next_maintenance"] <= 7:
                    schedule["due_soon"].append(equipment_schedule)
                elif prediction["days_until_next_maintenance"] <= 30:
                    schedule["upcoming"].append(equipment_schedule)
            
            except Exception as e:
                logger.error(f"Error prediciendo mantenimiento para {equipment_id}: {e}")
        
        return schedule




