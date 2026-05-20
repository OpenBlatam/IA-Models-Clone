"""
Diagram Generator - Generador de diagramas y visualizaciones
===========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DiagramGenerator:
    """Generador de diagramas y visualizaciones para prototipos"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output/diagrams")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_assembly_diagram(self, cad_parts: List[Dict[str, Any]], 
                                 assembly_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera un diagrama de ensamblaje"""
        diagram = {
            "type": "assembly_diagram",
            "parts": [
                {
                    "id": f"part_{i+1}",
                    "name": part.get("part_name", ""),
                    "number": part.get("part_number", i+1),
                    "material": part.get("material", ""),
                    "dimensions": part.get("dimensions", {}),
                    "position": self._calculate_position(i, len(cad_parts))
                }
                for i, part in enumerate(cad_parts)
            ],
            "assembly_flow": [
                {
                    "step": step.get("step_number", i+1),
                    "description": step.get("description", ""),
                    "parts_involved": step.get("parts_involved", []),
                    "connections": self._get_connections(step, cad_parts)
                }
                for i, step in enumerate(assembly_steps)
            ],
            "metadata": {
                "total_parts": len(cad_parts),
                "total_steps": len(assembly_steps)
            }
        }
        
        return diagram
    
    def _calculate_position(self, index: int, total: int) -> Dict[str, float]:
        """Calcula posición visual de una parte"""
        # Distribución circular simple
        angle = (2 * 3.14159 * index) / total
        radius = 100
        return {
            "x": radius * (1 + 0.5 * (index % 2)) * (1 if index < total/2 else -1),
            "y": radius * (1 + 0.3 * ((index // 2) % 2)),
            "z": 0
        }
    
    def _get_connections(self, step: Dict[str, Any], parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Obtiene conexiones entre partes en un paso"""
        connections = []
        parts_involved = step.get("parts_involved", [])
        
        for i, part1_name in enumerate(parts_involved):
            for part2_name in parts_involved[i+1:]:
                connections.append({
                    "from": part1_name,
                    "to": part2_name,
                    "type": "assembly"
                })
        
        return connections
    
    def generate_cost_breakdown_chart(self, materials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera un gráfico de desglose de costos"""
        categories = {}
        
        for material in materials:
            category = material.get("category", "General")
            if category not in categories:
                categories[category] = {
                    "name": category,
                    "materials": [],
                    "total_cost": 0,
                    "percentage": 0
                }
            
            categories[category]["materials"].append({
                "name": material.get("name", ""),
                "cost": material.get("total_price", 0)
            })
            categories[category]["total_cost"] += material.get("total_price", 0)
        
        total_cost = sum(cat["total_cost"] for cat in categories.values())
        
        for category in categories.values():
            category["percentage"] = (category["total_cost"] / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "type": "cost_breakdown",
            "total_cost": total_cost,
            "categories": list(categories.values()),
            "chart_type": "pie",
            "data_points": [
                {
                    "label": cat["name"],
                    "value": cat["total_cost"],
                    "percentage": cat["percentage"]
                }
                for cat in categories.values()
            ]
        }
    
    def generate_timeline_diagram(self, assembly_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera un diagrama de línea de tiempo del ensamblaje"""
        timeline = []
        cumulative_time = 0
        
        for step in assembly_steps:
            time_str = step.get("time_estimate", "0 minutos")
            # Extraer minutos
            minutes = 0
            if "minuto" in time_str.lower():
                try:
                    minutes = int(time_str.split()[0])
                except:
                    minutes = 15
            
            timeline.append({
                "step": step.get("step_number", 0),
                "description": step.get("description", ""),
                "duration_minutes": minutes,
                "start_time": cumulative_time,
                "end_time": cumulative_time + minutes,
                "difficulty": step.get("difficulty", "media"),
                "parts": step.get("parts_involved", [])
            })
            
            cumulative_time += minutes
        
        return {
            "type": "timeline",
            "total_time_minutes": cumulative_time,
            "total_time_hours": cumulative_time / 60,
            "steps": timeline,
            "gantt_data": [
                {
                    "task": f"Paso {s['step']}",
                    "start": s["start_time"],
                    "end": s["end_time"],
                    "duration": s["duration_minutes"]
                }
                for s in timeline
            ]
        }
    
    def generate_material_flow_diagram(self, materials: List[Dict[str, Any]], 
                                      cad_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera un diagrama de flujo de materiales"""
        material_to_parts = {}
        
        for part in cad_parts:
            material = part.get("material", "")
            if material not in material_to_parts:
                material_to_parts[material] = []
            material_to_parts[material].append(part.get("part_name", ""))
        
        flow = {
            "type": "material_flow",
            "materials": [
                {
                    "material": mat.get("name", ""),
                    "quantity": mat.get("quantity", 0),
                    "unit": mat.get("unit", ""),
                    "cost": mat.get("total_price", 0),
                    "used_in": material_to_parts.get(mat.get("name", ""), [])
                }
                for mat in materials
            ],
            "flow_steps": [
                {
                    "step": "Compra",
                    "materials": [m.get("name", "") for m in materials]
                },
                {
                    "step": "Preparación",
                    "materials": [m.get("name", "") for m in materials]
                },
                {
                    "step": "Ensamblaje",
                    "parts": [p.get("part_name", "") for p in cad_parts]
                }
            ]
        }
        
        return flow
    
    def generate_all_diagrams(self, prototype_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera todos los diagramas para un prototipo"""
        cad_parts = prototype_data.get("cad_parts", [])
        assembly_steps = prototype_data.get("assembly_instructions", [])
        materials = prototype_data.get("materials", [])
        
        diagrams = {
            "assembly_diagram": self.generate_assembly_diagram(cad_parts, assembly_steps),
            "cost_breakdown": self.generate_cost_breakdown_chart(materials),
            "timeline": self.generate_timeline_diagram(assembly_steps),
            "material_flow": self.generate_material_flow_diagram(materials, cad_parts)
        }
        
        # Guardar diagramas
        product_name = prototype_data.get("product_name", "prototype")
        safe_name = product_name.lower().replace(" ", "_")
        diagrams_file = self.output_dir / f"{safe_name}_diagrams.json"
        
        with open(diagrams_file, "w", encoding="utf-8") as f:
            json.dump(diagrams, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Diagramas generados: {diagrams_file}")
        
        return {
            "diagrams": diagrams,
            "file_path": str(diagrams_file)
        }
    
    def export_diagram_svg(self, diagram: Dict[str, Any], diagram_type: str) -> str:
        """Exporta un diagrama a formato SVG (estructura básica)"""
        # Esto es una estructura básica - en producción se usaría una librería como matplotlib o graphviz
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
  <title>{diagram_type}</title>
  <text x="400" y="50" text-anchor="middle" font-size="24">{diagram_type}</text>
  <!-- Diagram content would be generated here -->
</svg>"""
        
        return svg_content




