"""
Prototype Comparator - Sistema de comparación de prototipos
===========================================================
"""

import logging
from typing import List, Dict, Any
from ..models.schemas import PrototypeResponse

logger = logging.getLogger(__name__)


class PrototypeComparator:
    """Comparador de múltiples prototipos"""
    
    def compare_prototypes(self, prototypes: List[PrototypeResponse]) -> Dict[str, Any]:
        """
        Compara múltiples prototipos
        
        Args:
            prototypes: Lista de prototipos a comparar
            
        Returns:
            Comparación detallada
        """
        if len(prototypes) < 2:
            return {"error": "Se necesitan al menos 2 prototipos para comparar"}
        
        # Comparación de costos
        cost_comparison = self._compare_costs(prototypes)
        
        # Comparación de complejidad
        complexity_comparison = self._compare_complexity(prototypes)
        
        # Comparación de materiales
        material_comparison = self._compare_materials(prototypes)
        
        # Comparación de tiempo
        time_comparison = self._compare_time(prototypes)
        
        # Mejor opción por categoría
        best_options = self._get_best_options(prototypes, cost_comparison, complexity_comparison)
        
        return {
            "num_prototypes": len(prototypes),
            "cost_comparison": cost_comparison,
            "complexity_comparison": complexity_comparison,
            "material_comparison": material_comparison,
            "time_comparison": time_comparison,
            "best_options": best_options,
            "summary": self._generate_summary(prototypes, best_options)
        }
    
    def _compare_costs(self, prototypes: List[PrototypeResponse]) -> Dict[str, Any]:
        """Compara los costos de los prototipos"""
        costs = [p.total_cost_estimate for p in prototypes]
        
        return {
            "costs": [
                {"prototype": p.product_name, "cost": p.total_cost_estimate}
                for p in prototypes
            ],
            "min_cost": min(costs),
            "max_cost": max(costs),
            "avg_cost": sum(costs) / len(costs),
            "cost_range": max(costs) - min(costs),
            "cheapest": prototypes[costs.index(min(costs))].product_name,
            "most_expensive": prototypes[costs.index(max(costs))].product_name
        }
    
    def _compare_complexity(self, prototypes: List[PrototypeResponse]) -> Dict[str, Any]:
        """Compara la complejidad de los prototipos"""
        complexities = []
        for p in prototypes:
            complexity_score = (
                len(p.cad_parts) * 10 +
                len(p.materials) * 5 +
                len(p.assembly_instructions) * 3
            )
            complexities.append({
                "prototype": p.product_name,
                "score": complexity_score,
                "difficulty": p.difficulty_level,
                "num_parts": len(p.cad_parts),
                "num_materials": len(p.materials),
                "num_steps": len(p.assembly_instructions)
            })
        
        complexities.sort(key=lambda x: x["score"])
        
        return {
            "complexities": complexities,
            "simplest": complexities[0]["prototype"],
            "most_complex": complexities[-1]["prototype"]
        }
    
    def _compare_materials(self, prototypes: List[PrototypeResponse]) -> Dict[str, Any]:
        """Compara los materiales de los prototipos"""
        material_counts = [len(p.materials) for p in prototypes]
        unique_materials = set()
        for p in prototypes:
            unique_materials.update(m.name for m in p.materials)
        
        return {
            "material_counts": [
                {"prototype": p.product_name, "count": len(p.materials)}
                for p in prototypes
            ],
            "min_materials": min(material_counts),
            "max_materials": max(material_counts),
            "total_unique_materials": len(unique_materials),
            "shared_materials": self._find_shared_materials(prototypes)
        }
    
    def _find_shared_materials(self, prototypes: List[PrototypeResponse]) -> List[str]:
        """Encuentra materiales compartidos entre prototipos"""
        if not prototypes:
            return []
        
        # Materiales del primer prototipo
        shared = set(m.name for m in prototypes[0].materials)
        
        # Intersectar con materiales de otros prototipos
        for p in prototypes[1:]:
            shared &= set(m.name for m in p.materials)
        
        return list(shared)
    
    def _compare_time(self, prototypes: List[PrototypeResponse]) -> Dict[str, Any]:
        """Compara los tiempos estimados"""
        times = []
        for p in prototypes:
            # Extraer horas del tiempo estimado
            hours = 0
            time_str = p.estimated_build_time
            if "hora" in time_str.lower():
                try:
                    hours = int(time_str.split("-")[0].split()[0])
                except:
                    hours = 3
            
            times.append({
                "prototype": p.product_name,
                "time": p.estimated_build_time,
                "hours": hours
            })
        
        times.sort(key=lambda x: x["hours"])
        
        return {
            "times": times,
            "fastest": times[0]["prototype"],
            "slowest": times[-1]["prototype"],
            "time_range": times[-1]["hours"] - times[0]["hours"]
        }
    
    def _get_best_options(self, prototypes: List[PrototypeResponse],
                          cost_comparison: Dict, complexity_comparison: Dict) -> Dict[str, Any]:
        """Obtiene las mejores opciones por categoría"""
        costs = [p.total_cost_estimate for p in prototypes]
        complexities = [
            len(p.cad_parts) * 10 + len(p.materials) * 5 + len(p.assembly_instructions) * 3
            for p in prototypes
        ]
        
        return {
            "best_cost": {
                "prototype": prototypes[costs.index(min(costs))].product_name,
                "cost": min(costs)
            },
            "best_complexity": {
                "prototype": prototypes[complexities.index(min(complexities))].product_name,
                "score": min(complexities)
            },
            "best_balance": self._find_best_balance(prototypes, costs, complexities)
        }
    
    def _find_best_balance(self, prototypes: List[PrototypeResponse],
                          costs: List[float], complexities: List[float]) -> Dict[str, Any]:
        """Encuentra el mejor balance costo-complejidad"""
        # Normalizar scores (0-1)
        max_cost = max(costs)
        min_cost = min(costs)
        max_complexity = max(complexities)
        min_complexity = min(complexities)
        
        normalized_costs = [
            (c - min_cost) / (max_cost - min_cost) if max_cost != min_cost else 0.5
            for c in costs
        ]
        normalized_complexities = [
            (c - min_complexity) / (max_complexity - min_complexity) if max_complexity != min_complexity else 0.5
            for c in complexities
        ]
        
        # Calcular score de balance (menor es mejor)
        balance_scores = [
            (cost + complexity) / 2
            for cost, complexity in zip(normalized_costs, normalized_complexities)
        ]
        
        best_index = balance_scores.index(min(balance_scores))
        
        return {
            "prototype": prototypes[best_index].product_name,
            "cost": costs[best_index],
            "complexity_score": complexities[best_index]
        }
    
    def _generate_summary(self, prototypes: List[PrototypeResponse],
                         best_options: Dict[str, Any]) -> str:
        """Genera un resumen de la comparación"""
        summary_parts = [
            f"Se compararon {len(prototypes)} prototipos.",
            f"Mejor opción por costo: {best_options['best_cost']['prototype']} (${best_options['best_cost']['cost']:.2f})",
            f"Mejor opción por simplicidad: {best_options['best_complexity']['prototype']}",
            f"Mejor balance general: {best_options['best_balance']['prototype']}"
        ]
        
        return " ".join(summary_parts)




