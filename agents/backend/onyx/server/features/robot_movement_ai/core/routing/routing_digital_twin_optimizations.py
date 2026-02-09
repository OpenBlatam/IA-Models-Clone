"""
Optimizaciones de Digital Twins para Routing.

Este módulo implementa gemelos digitales para simulación,
predicción y optimización de rutas en tiempo real.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TwinType(Enum):
    """Tipos de gemelos digitales."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


@dataclass
class DigitalTwinState:
    """Estado del gemelo digital."""
    timestamp: float
    nodes: Dict[str, Dict[str, Any]]
    edges: Dict[str, Dict[str, Any]]
    routes: List[Dict[str, Any]]
    metrics: Dict[str, float]
    predictions: Dict[str, Any] = field(default_factory=dict)


class DigitalTwin:
    """Gemelo digital del sistema de routing."""
    
    def __init__(self, twin_id: str, twin_type: TwinType = TwinType.DYNAMIC):
        self.twin_id = twin_id
        self.twin_type = twin_type
        self.state_history: List[DigitalTwinState] = []
        self.current_state: Optional[DigitalTwinState] = None
        self.sync_frequency = 1.0  # Hz
        self.last_sync = 0.0
        self.predictions: Dict[str, Any] = {}
        self.simulation_results: List[Dict[str, Any]] = []
    
    def update_state(self, nodes: Dict[str, Any], edges: Dict[str, Any],
                    routes: List[Dict[str, Any]], metrics: Dict[str, float]):
        """Actualizar estado del gemelo."""
        state = DigitalTwinState(
            timestamp=time.time(),
            nodes=nodes,
            edges=edges,
            routes=routes,
            metrics=metrics
        )
        
        self.current_state = state
        self.state_history.append(state)
        
        # Mantener solo últimos 1000 estados
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
        
        self.last_sync = time.time()
    
    def predict(self, horizon: float = 60.0) -> Dict[str, Any]:
        """Predecir estado futuro."""
        if not self.current_state or len(self.state_history) < 2:
            return {}
        
        # Predicción simple basada en tendencias
        recent_states = self.state_history[-10:]
        
        predictions = {
            "predicted_latency": self._predict_latency(recent_states, horizon),
            "predicted_throughput": self._predict_throughput(recent_states, horizon),
            "predicted_error_rate": self._predict_error_rate(recent_states, horizon),
            "predicted_load": self._predict_load(recent_states, horizon)
        }
        
        self.predictions = predictions
        return predictions
    
    def _predict_latency(self, states: List[DigitalTwinState], horizon: float) -> float:
        """Predecir latencia."""
        if len(states) < 2:
            return 0.0
        
        latencies = [s.metrics.get("latency", 0.0) for s in states]
        trend = (latencies[-1] - latencies[0]) / len(latencies)
        predicted = latencies[-1] + trend * (horizon / self.sync_frequency)
        return max(0.0, predicted)
    
    def _predict_throughput(self, states: List[DigitalTwinState], horizon: float) -> float:
        """Predecir throughput."""
        if len(states) < 2:
            return 0.0
        
        throughputs = [s.metrics.get("throughput", 0.0) for s in states]
        trend = (throughputs[-1] - throughputs[0]) / len(throughputs)
        predicted = throughputs[-1] + trend * (horizon / self.sync_frequency)
        return max(0.0, predicted)
    
    def _predict_error_rate(self, states: List[DigitalTwinState], horizon: float) -> float:
        """Predecir tasa de error."""
        if len(states) < 2:
            return 0.0
        
        error_rates = [s.metrics.get("error_rate", 0.0) for s in states]
        trend = (error_rates[-1] - error_rates[0]) / len(error_rates)
        predicted = error_rates[-1] + trend * (horizon / self.sync_frequency)
        return max(0.0, min(1.0, predicted))
    
    def _predict_load(self, states: List[DigitalTwinState], horizon: float) -> float:
        """Predecir carga."""
        if len(states) < 2:
            return 0.0
        
        loads = [s.metrics.get("load", 0.0) for s in states]
        trend = (loads[-1] - loads[0]) / len(loads)
        predicted = loads[-1] + trend * (horizon / self.sync_frequency)
        return max(0.0, min(1.0, predicted))
    
    def simulate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simular escenario."""
        if not self.current_state:
            return {}
        
        # Simulación simplificada
        result = {
            "scenario": scenario,
            "predicted_impact": {},
            "recommendations": []
        }
        
        # Simular cambios
        if "add_nodes" in scenario:
            result["predicted_impact"]["latency"] = -0.1
            result["recommendations"].append("Add nodes to reduce latency")
        
        if "increase_load" in scenario:
            result["predicted_impact"]["latency"] = result["predicted_impact"].get("latency", 0) + 0.2
            result["recommendations"].append("Scale resources to handle load")
        
        self.simulation_results.append(result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "twin_id": self.twin_id,
            "twin_type": self.twin_type.value,
            "state_history_size": len(self.state_history),
            "sync_frequency": self.sync_frequency,
            "last_sync": self.last_sync,
            "predictions_count": len(self.predictions),
            "simulations_count": len(self.simulation_results)
        }


class DigitalTwinManager:
    """Gestor de gemelos digitales."""
    
    def __init__(self):
        self.twins: Dict[str, DigitalTwin] = {}
        self.active_twin: Optional[str] = None
        self.sync_interval = 1.0
        self.total_syncs = 0
    
    def create_twin(self, twin_id: str, twin_type: TwinType = TwinType.DYNAMIC) -> DigitalTwin:
        """Crear gemelo digital."""
        twin = DigitalTwin(twin_id=twin_id, twin_type=twin_type)
        self.twins[twin_id] = twin
        if not self.active_twin:
            self.active_twin = twin_id
        return twin
    
    def sync_twin(self, twin_id: str, nodes: Dict[str, Any], edges: Dict[str, Any],
                 routes: List[Dict[str, Any]], metrics: Dict[str, float]):
        """Sincronizar gemelo digital."""
        if twin_id not in self.twins:
            return
        
        twin = self.twins[twin_id]
        twin.update_state(nodes, edges, routes, metrics)
        self.total_syncs += 1
    
    def get_predictions(self, twin_id: str, horizon: float = 60.0) -> Dict[str, Any]:
        """Obtener predicciones."""
        if twin_id not in self.twins:
            return {}
        
        return self.twins[twin_id].predict(horizon)
    
    def simulate(self, twin_id: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simular escenario."""
        if twin_id not in self.twins:
            return {}
        
        return self.twins[twin_id].simulate_scenario(scenario)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_twins": len(self.twins),
            "active_twin": self.active_twin,
            "total_syncs": self.total_syncs,
            "sync_interval": self.sync_interval,
            "twins": {tid: twin.get_stats() for tid, twin in self.twins.items()}
        }


class DigitalTwinOptimizer:
    """Optimizador principal de gemelos digitales."""
    
    def __init__(self, enable_digital_twin: bool = True):
        self.enable_digital_twin = enable_digital_twin
        self.manager = DigitalTwinManager() if enable_digital_twin else None
        self.main_twin_id = "routing_system"
        
        if self.manager:
            self.manager.create_twin(self.main_twin_id, TwinType.DYNAMIC)
    
    def sync(self, nodes: Dict[str, Any], edges: Dict[str, Any],
            routes: List[Dict[str, Any]], metrics: Dict[str, float]):
        """Sincronizar gemelo digital."""
        if not self.enable_digital_twin or not self.manager:
            return
        
        self.manager.sync_twin(self.main_twin_id, nodes, edges, routes, metrics)
    
    def predict(self, horizon: float = 60.0) -> Dict[str, Any]:
        """Predecir estado futuro."""
        if not self.enable_digital_twin or not self.manager:
            return {}
        
        return self.manager.get_predictions(self.main_twin_id, horizon)
    
    def simulate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simular escenario."""
        if not self.enable_digital_twin or not self.manager:
            return {}
        
        return self.manager.simulate(self.main_twin_id, scenario)
    
    def optimize_based_on_predictions(self) -> Dict[str, Any]:
        """Optimizar basado en predicciones."""
        if not self.enable_digital_twin or not self.manager:
            return {}
        
        predictions = self.predict(horizon=300.0)  # 5 minutos
        
        recommendations = []
        
        if predictions.get("predicted_latency", 0) > 1000.0:
            recommendations.append({
                "action": "scale_up",
                "reason": "High predicted latency",
                "priority": "high"
            })
        
        if predictions.get("predicted_load", 0) > 0.8:
            recommendations.append({
                "action": "add_resources",
                "reason": "High predicted load",
                "priority": "medium"
            })
        
        return {
            "predictions": predictions,
            "recommendations": recommendations
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        if not self.enable_digital_twin:
            return {
                "digital_twin_enabled": False
            }
        
        stats = self.manager.get_stats()
        stats["digital_twin_enabled"] = True
        stats["main_twin_id"] = self.main_twin_id
        
        return stats


