"""
Smart Waste Service - Gestión de residuos inteligente
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class WasteType(str, Enum):
    """Tipos de residuos"""
    ORGANIC = "organic"
    RECYCLABLE = "recyclable"
    HAZARDOUS = "hazardous"
    GENERAL = "general"


class SmartWasteService:
    """Servicio para gestión de residuos inteligente"""
    
    def __init__(self):
        self.bins: Dict[str, Dict[str, Any]] = {}
        self.collections: Dict[str, List[Dict[str, Any]]] = {}
        self.analytics: Dict[str, Dict[str, Any]] = {}
    
    def register_waste_bin(
        self,
        store_id: str,
        bin_name: str,
        waste_type: WasteType,
        location: str,
        capacity_liters: float,
        sensor_enabled: bool = True
    ) -> Dict[str, Any]:
        """Registrar contenedor de residuos"""
        
        bin_id = f"bin_{store_id}_{len(self.bins.get(store_id, [])) + 1}"
        
        bin_data = {
            "bin_id": bin_id,
            "store_id": store_id,
            "name": bin_name,
            "waste_type": waste_type.value,
            "location": location,
            "capacity_liters": capacity_liters,
            "current_level_liters": 0.0,
            "fill_percentage": 0.0,
            "sensor_enabled": sensor_enabled,
            "registered_at": datetime.now().isoformat(),
            "last_collection": None
        }
        
        if store_id not in self.bins:
            self.bins[store_id] = {}
        
        self.bins[store_id][bin_id] = bin_data
        
        return bin_data
    
    def update_bin_level(
        self,
        bin_id: str,
        level_liters: float
    ) -> Dict[str, Any]:
        """Actualizar nivel del contenedor"""
        
        bin_data = self._find_bin(bin_id)
        
        if not bin_data:
            raise ValueError(f"Contenedor {bin_id} no encontrado")
        
        bin_data["current_level_liters"] = level_liters
        bin_data["fill_percentage"] = (level_liters / bin_data["capacity_liters"] * 100) if bin_data["capacity_liters"] > 0 else 0
        bin_data["last_update"] = datetime.now().isoformat()
        
        return bin_data
    
    def schedule_collection(
        self,
        bin_id: str,
        collection_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Programar recolección"""
        
        bin_data = self._find_bin(bin_id)
        
        if not bin_data:
            raise ValueError(f"Contenedor {bin_id} no encontrado")
        
        collection_id = f"coll_{bin_id}_{len(self.collections.get(bin_id, [])) + 1}"
        
        collection = {
            "collection_id": collection_id,
            "bin_id": bin_id,
            "store_id": bin_data["store_id"],
            "waste_type": bin_data["waste_type"],
            "scheduled_date": collection_date or (datetime.now() + timedelta(days=1)).isoformat(),
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        if bin_id not in self.collections:
            self.collections[bin_id] = []
        
        self.collections[bin_id].append(collection)
        
        return collection
    
    def get_waste_analytics(
        self,
        store_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Obtener analytics de residuos"""
        
        bins = self.bins.get(store_id, {})
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        total_waste = sum(b["current_level_liters"] for b in bins.values())
        
        by_type = {}
        for bin_data in bins.values():
            waste_type = bin_data["waste_type"]
            if waste_type not in by_type:
                by_type[waste_type] = 0.0
            by_type[waste_type] += bin_data["current_level_liters"]
        
        # Contenedores que necesitan recolección
        needs_collection = [
            {
                "bin_id": bin_id,
                "name": bin_data["name"],
                "fill_percentage": bin_data["fill_percentage"],
                "waste_type": bin_data["waste_type"]
            }
            for bin_id, bin_data in bins.items()
            if bin_data["fill_percentage"] >= 80
        ]
        
        return {
            "store_id": store_id,
            "period_days": days,
            "total_waste_liters": round(total_waste, 2),
            "waste_by_type": {k: round(v, 2) for k, v in by_type.items()},
            "bins_count": len(bins),
            "bins_needing_collection": len(needs_collection),
            "bins_details": needs_collection,
            "recycling_rate": self._calculate_recycling_rate(by_type),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_recycling_rate(self, by_type: Dict[str, float]) -> float:
        """Calcular tasa de reciclaje"""
        recyclable = by_type.get("recyclable", 0)
        total = sum(by_type.values())
        
        return (recyclable / total * 100) if total > 0 else 0.0
    
    def _find_bin(self, bin_id: str) -> Optional[Dict[str, Any]]:
        """Encontrar contenedor"""
        for store_bins in self.bins.values():
            if bin_id in store_bins:
                return store_bins[bin_id]
        return None




