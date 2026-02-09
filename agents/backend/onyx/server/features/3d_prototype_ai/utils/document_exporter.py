"""
Document Exporter - Exportación de documentos a múltiples formatos
===================================================================
"""

import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

from ..models.schemas import (
    PrototypeResponse,
    Material,
    CADPart,
    AssemblyStep,
    BudgetOption
)

logger = logging.getLogger(__name__)


class DocumentExporter:
    """Exportador de documentos a múltiples formatos"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def export_to_markdown(self, response: PrototypeResponse) -> str:
        """Exporta el prototipo a formato Markdown"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = response.product_name.lower().replace(" ", "_")
        file_path = self.output_dir / f"{safe_name}_{timestamp}.md"
        
        md_content = self._generate_markdown(response)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        logger.info(f"Documento Markdown exportado: {file_path}")
        return str(file_path)
    
    def _generate_markdown(self, response: PrototypeResponse) -> str:
        """Genera el contenido Markdown"""
        md = f"""# {response.product_name}

## 📋 Descripción

{response.product_description}

---

## 🔧 Especificaciones Técnicas

"""
        # Especificaciones
        for key, value in response.specifications.items():
            if isinstance(value, dict):
                md += f"### {key.replace('_', ' ').title()}\n\n"
                for k, v in value.items():
                    md += f"- **{k.replace('_', ' ').title()}**: {v}\n"
                md += "\n"
            else:
                md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        md += f"""
---

## 📦 Materiales Necesarios

**Costo Total Estimado: ${response.total_cost_estimate:.2f}**

"""
        # Materiales
        for i, material in enumerate(response.materials, 1):
            md += f"""### {i}. {material.name}

- **Cantidad**: {material.quantity} {material.unit}
- **Precio Unitario**: ${material.price_per_unit:.2f}
- **Precio Total**: ${material.total_price:.2f}
- **Categoría**: {material.category}

"""
            if material.sources:
                md += "**Fuentes de Suministro:**\n"
                for source in material.sources:
                    source_info = f"- {source.name}"
                    if source.location:
                        source_info += f" ({source.location})"
                    if source.url:
                        source_info += f" - [{source.url}]({source.url})"
                    md += source_info + "\n"
                md += "\n"
        
        md += f"""
---

## 🔧 Partes del Modelo CAD

"""
        # Partes CAD
        for part in response.cad_parts:
            md += f"""### {part.part_name} (Parte #{part.part_number})

- **Descripción**: {part.description}
- **Material**: {part.material}
- **Dimensiones**: {self._format_dimensions(part.dimensions)}
- **Formato**: {part.cad_format}
"""
            if part.quantity > 1:
                md += f"- **Cantidad**: {part.quantity}\n"
            md += "\n"
        
        md += f"""
---

## 📋 Instrucciones de Ensamblaje

**Tiempo Total Estimado**: {response.estimated_build_time}
**Nivel de Dificultad**: {response.difficulty_level}

"""
        # Instrucciones
        for step in response.assembly_instructions:
            md += f"""### Paso {step.step_number}: {step.description}

- **Partes Involucradas**: {', '.join(step.parts_involved)}
"""
            if step.tools_needed:
                md += f"- **Herramientas Necesarias**: {', '.join(step.tools_needed)}\n"
            if step.time_estimate:
                md += f"- **Tiempo Estimado**: {step.time_estimate}\n"
            md += f"- **Dificultad**: {step.difficulty}\n\n"
        
        md += f"""
---

## 💰 Opciones Según Presupuesto

"""
        # Opciones de presupuesto
        for option in response.budget_options:
            md += f"""### Opción {option.budget_level.upper()}

- **Costo Total**: ${option.total_cost:.2f}
- **Nivel de Calidad**: {option.quality_level}
- **Descripción**: {option.description}
"""
            if option.trade_offs:
                md += f"- **Compromisos**: {', '.join(option.trade_offs)}\n"
            md += "\n"
        
        md += f"""
---

## 📊 Resumen

- **Producto**: {response.product_name}
- **Costo Total**: ${response.total_cost_estimate:.2f}
- **Tiempo de Construcción**: {response.estimated_build_time}
- **Dificultad**: {response.difficulty_level}
- **Número de Partes**: {len(response.cad_parts)}
- **Número de Materiales**: {len(response.materials)}

---

*Documento generado el {response.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return md
    
    def _format_dimensions(self, dimensions: Dict[str, Any]) -> str:
        """Formatea las dimensiones para mostrar"""
        parts = []
        for key, value in dimensions.items():
            parts.append(f"{key}: {value}")
        return ", ".join(parts)
    
    async def export_to_json(self, response: PrototypeResponse) -> str:
        """Exporta a JSON (ya implementado, pero lo mantenemos aquí)"""
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = response.product_name.lower().replace(" ", "_")
        file_path = self.output_dir / f"{safe_name}_{timestamp}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        
        return str(file_path)




