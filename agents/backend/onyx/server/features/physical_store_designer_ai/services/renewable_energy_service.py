"""
Renewable Energy Service - Integración con energía renovable
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RenewableSource(str, Enum):
    """Fuentes de energía renovable"""
    SOLAR = "solar"
    WIND = "wind"
    GEOTHERMAL = "geothermal"
    HYDRO = "hydro"
    BIOMASS = "biomass"


class RenewableEnergyService:
    """Servicio para energía renovable"""
    
    def __init__(self):
        self.systems: Dict[str, Dict[str, Any]] = {}
        self.generation: Dict[str, List[Dict[str, Any]]] = {}
        self.credits: Dict[str, Dict[str, Any]] = {}
    
    def install_renewable_system(
        self,
        store_id: str,
        system_name: str,
        source: RenewableSource,
        capacity_kw: float,
        installation_date: str
    ) -> Dict[str, Any]:
        """Instalar sistema de energía renovable"""
        
        system_id = f"renew_{store_id}_{len(self.systems.get(store_id, [])) + 1}"
        
        system = {
            "system_id": system_id,
            "store_id": store_id,
            "name": system_name,
            "source": source.value,
            "capacity_kw": capacity_kw,
            "installation_date": installation_date,
            "is_active": True,
            "total_generated_kwh": 0.0,
            "registered_at": datetime.now().isoformat()
        }
        
        if store_id not in self.systems:
            self.systems[store_id] = {}
        
        self.systems[store_id][system_id] = system
        
        return system
    
    def record_generation(
        self,
        system_id: str,
        energy_kwh: float,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar generación de energía"""
        
        system = self._find_system(system_id)
        
        if not system:
            raise ValueError(f"Sistema {system_id} no encontrado")
        
        generation = {
            "generation_id": f"gen_{system_id}_{len(self.generation.get(system_id, [])) + 1}",
            "system_id": system_id,
            "energy_kwh": energy_kwh,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        if system_id not in self.generation:
            self.generation[system_id] = []
        
        self.generation[system_id].append(generation)
        system["total_generated_kwh"] += energy_kwh
        
        return generation
    
    def calculate_energy_savings(
        self,
        store_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calcular ahorros de energía"""
        
        systems = self.systems.get(store_id, {})
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        total_generated = 0.0
        by_source = {}
        
        for system_id, system in systems.items():
            generations = self.generation.get(system_id, [])
            recent = [
                g for g in generations
                if start_date <= datetime.fromisoformat(g["timestamp"]) <= end_date
            ]
            
            system_generated = sum(g["energy_kwh"] for g in recent)
            total_generated += system_generated
            
            source = system["source"]
            if source not in by_source:
                by_source[source] = 0.0
            by_source[source] += system_generated
        
        # Calcular ahorros (asumiendo $0.12/kWh)
        cost_per_kwh = 0.12
        savings = total_generated * cost_per_kwh
        
        # Calcular reducción de carbono (asumiendo 0.5 kg CO2/kWh ahorrado)
        co2_reduction_kg = total_generated * 0.5
        
        return {
            "store_id": store_id,
            "period_days": days,
            "total_generated_kwh": round(total_generated, 2),
            "generation_by_source": {k: round(v, 2) for k, v in by_source.items()},
            "cost_savings": round(savings, 2),
            "co2_reduction_kg": round(co2_reduction_kg, 2),
            "systems_count": len(systems),
            "calculated_at": datetime.now().isoformat()
        }
    
    def generate_energy_credits(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Generar créditos de energía renovable"""
        
        savings = self.calculate_energy_savings(store_id, days=365)
        
        # 1 MWh = 1 REC (Renewable Energy Credit)
        recs = savings["total_generated_kwh"] / 1000
        
        credit = {
            "credit_id": f"credit_{store_id}_{datetime.now().strftime('%Y%m%d')}",
            "store_id": store_id,
            "renewable_energy_certificates": round(recs, 2),
            "total_generated_mwh": round(savings["total_generated_kwh"] / 1000, 2),
            "co2_offset_kg": round(savings["co2_reduction_kg"], 2),
            "issued_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=365)).isoformat()
        }
        
        self.credits[store_id] = credit
        
        return credit
    
    def _find_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Encontrar sistema"""
        for store_systems in self.systems.values():
            if system_id in store_systems:
                return store_systems[system_id]
        return None




