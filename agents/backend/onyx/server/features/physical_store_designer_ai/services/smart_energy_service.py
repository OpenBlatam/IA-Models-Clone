"""
Smart Energy Service - Gestión de energía inteligente
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SmartEnergyService:
    """Servicio para gestión de energía inteligente"""
    
    def __init__(self):
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.consumption: Dict[str, List[Dict[str, Any]]] = {}
        self.optimizations: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_energy_device(
        self,
        store_id: str,
        device_name: str,
        device_type: str,  # "lighting", "hvac", "appliance", "renewable"
        power_rating_watts: float,
        location: str
    ) -> Dict[str, Any]:
        """Registrar dispositivo de energía"""
        
        device_id = f"energy_{store_id}_{len(self.devices.get(store_id, [])) + 1}"
        
        device = {
            "device_id": device_id,
            "store_id": store_id,
            "name": device_name,
            "type": device_type,
            "power_rating_watts": power_rating_watts,
            "location": location,
            "is_active": True,
            "registered_at": datetime.now().isoformat()
        }
        
        if store_id not in self.devices:
            self.devices[store_id] = {}
        
        self.devices[store_id][device_id] = device
        
        return device
    
    def record_consumption(
        self,
        device_id: str,
        energy_kwh: float,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar consumo de energía"""
        
        consumption = {
            "consumption_id": f"cons_{device_id}_{len(self.consumption.get(device_id, [])) + 1}",
            "device_id": device_id,
            "energy_kwh": energy_kwh,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        if device_id not in self.consumption:
            self.consumption[device_id] = []
        
        self.consumption[device_id].append(consumption)
        
        return consumption
    
    def calculate_energy_usage(
        self,
        store_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calcular uso de energía"""
        
        devices = self.devices.get(store_id, {})
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        total_consumption = 0.0
        by_device = {}
        by_type = {}
        
        for device_id, device in devices.items():
            consumptions = self.consumption.get(device_id, [])
            recent = [
                c for c in consumptions
                if start_date <= datetime.fromisoformat(c["timestamp"]) <= end_date
            ]
            
            device_total = sum(c["energy_kwh"] for c in recent)
            total_consumption += device_total
            
            by_device[device["name"]] = device_total
            
            device_type = device["type"]
            if device_type not in by_type:
                by_type[device_type] = 0.0
            by_type[device_type] += device_total
        
        return {
            "store_id": store_id,
            "period_days": days,
            "total_energy_kwh": round(total_consumption, 2),
            "average_daily_kwh": round(total_consumption / days, 2),
            "by_device": by_device,
            "by_type": by_type,
            "estimated_cost": round(total_consumption * 0.12, 2),  # $0.12/kWh
            "calculated_at": datetime.now().isoformat()
        }
    
    def generate_energy_optimization(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Generar optimización de energía"""
        
        usage = self.calculate_energy_usage(store_id)
        devices = self.devices.get(store_id, {})
        
        recommendations = []
        potential_savings = 0.0
        
        # Analizar cada tipo de dispositivo
        for device_type, consumption in usage["by_type"].items():
            if device_type == "lighting":
                recommendations.append({
                    "type": "lighting",
                    "recommendation": "Switch to LED lighting",
                    "potential_savings_kwh": consumption * 0.5,
                    "priority": "high"
                })
                potential_savings += consumption * 0.5
            
            elif device_type == "hvac":
                recommendations.append({
                    "type": "hvac",
                    "recommendation": "Optimize HVAC schedule",
                    "potential_savings_kwh": consumption * 0.2,
                    "priority": "medium"
                })
                potential_savings += consumption * 0.2
        
        optimization = {
            "store_id": store_id,
            "current_usage_kwh": usage["total_energy_kwh"],
            "recommendations": recommendations,
            "potential_savings_kwh": round(potential_savings, 2),
            "potential_cost_savings": round(potential_savings * 0.12, 2),
            "generated_at": datetime.now().isoformat()
        }
        
        if store_id not in self.optimizations:
            self.optimizations[store_id] = []
        
        self.optimizations[store_id].append(optimization)
        
        return optimization




