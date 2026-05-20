"""
Advanced Exporter - Exportación avanzada a PDF, Excel, etc.
===========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AdvancedExporter:
    """Exportador avanzado a múltiples formatos"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_excel(self, prototype_data: Dict[str, Any]) -> str:
        """Exporta a formato Excel (CSV estructurado)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = prototype_data.get("product_name", "prototype").lower().replace(" ", "_")
        
        # Generar CSV estructurado (simulando Excel)
        csv_content = self._generate_excel_content(prototype_data)
        
        file_path = self.output_dir / f"{safe_name}_{timestamp}.csv"
        with open(file_path, "w", encoding="utf-8-sig") as f:  # UTF-8 con BOM para Excel
            f.write(csv_content)
        
        logger.info(f"Archivo Excel exportado: {file_path}")
        return str(file_path)
    
    def _generate_excel_content(self, prototype_data: Dict[str, Any]) -> str:
        """Genera contenido CSV estructurado"""
        lines = []
        
        # Encabezado
        lines.append("PROTOTIPO: " + prototype_data.get("product_name", ""))
        lines.append("")
        lines.append("ESPECIFICACIONES")
        lines.append("Campo,Valor")
        
        specs = prototype_data.get("specifications", {})
        for key, value in specs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"{key}.{k},{v}")
            else:
                lines.append(f"{key},{value}")
        
        lines.append("")
        lines.append("MATERIALES")
        lines.append("Nombre,Cantidad,Unidad,Precio Unitario,Precio Total,Categoría")
        
        materials = prototype_data.get("materials", [])
        for mat in materials:
            lines.append(f"{mat.get('name', '')},{mat.get('quantity', 0)},{mat.get('unit', '')},"
                        f"{mat.get('price_per_unit', 0)},{mat.get('total_price', 0)},{mat.get('category', '')}")
        
        lines.append("")
        lines.append(f"COSTO TOTAL,{prototype_data.get('total_cost_estimate', 0)}")
        
        lines.append("")
        lines.append("PARTES CAD")
        lines.append("Nombre,Número,Material,Descripción")
        
        cad_parts = prototype_data.get("cad_parts", [])
        for part in cad_parts:
            lines.append(f"{part.get('part_name', '')},{part.get('part_number', 0)},"
                        f"{part.get('material', '')},{part.get('description', '')}")
        
        lines.append("")
        lines.append("INSTRUCCIONES DE ENSAMBLAJE")
        lines.append("Paso,Descripción,Partes,Herramientas,Tiempo,Dificultad")
        
        instructions = prototype_data.get("assembly_instructions", [])
        for step in instructions:
            lines.append(f"{step.get('step_number', 0)},{step.get('description', '')},"
                        f"{','.join(step.get('parts_involved', []))},"
                        f"{','.join(step.get('tools_needed', []))},"
                        f"{step.get('time_estimate', '')},{step.get('difficulty', '')}")
        
        return "\n".join(lines)
    
    def export_to_pdf_structure(self, prototype_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera estructura para PDF (requiere librería externa para renderizado)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = prototype_data.get("product_name", "prototype").lower().replace(" ", "_")
        
        pdf_structure = {
            "metadata": {
                "title": prototype_data.get("product_name", "Prototipo"),
                "author": "3D Prototype AI",
                "created_at": datetime.now().isoformat()
            },
            "sections": [
                {
                    "title": "Descripción del Producto",
                    "content": prototype_data.get("product_description", "")
                },
                {
                    "title": "Especificaciones Técnicas",
                    "content": self._format_specifications_for_pdf(prototype_data.get("specifications", {}))
                },
                {
                    "title": "Lista de Materiales",
                    "content": self._format_materials_for_pdf(prototype_data.get("materials", []))
                },
                {
                    "title": "Partes del Modelo CAD",
                    "content": self._format_cad_parts_for_pdf(prototype_data.get("cad_parts", []))
                },
                {
                    "title": "Instrucciones de Ensamblaje",
                    "content": self._format_instructions_for_pdf(prototype_data.get("assembly_instructions", []))
                },
                {
                    "title": "Opciones de Presupuesto",
                    "content": self._format_budget_options_for_pdf(prototype_data.get("budget_options", []))
                }
            ]
        }
        
        # Guardar estructura JSON (para renderizado con librería PDF)
        file_path = self.output_dir / f"{safe_name}_{timestamp}_pdf_structure.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(pdf_structure, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Estructura PDF generada: {file_path}")
        return {
            "structure": pdf_structure,
            "file_path": str(file_path),
            "note": "Usar librería como reportlab o weasyprint para renderizar PDF"
        }
    
    def _format_specifications_for_pdf(self, specs: Dict[str, Any]) -> str:
        """Formatea especificaciones para PDF"""
        lines = []
        for key, value in specs.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def _format_materials_for_pdf(self, materials: List[Dict[str, Any]]) -> str:
        """Formatea materiales para PDF"""
        lines = []
        for i, mat in enumerate(materials, 1):
            lines.append(f"{i}. {mat.get('name', '')}")
            lines.append(f"   Cantidad: {mat.get('quantity', 0)} {mat.get('unit', '')}")
            lines.append(f"   Precio: ${mat.get('total_price', 0):.2f}")
        return "\n".join(lines)
    
    def _format_cad_parts_for_pdf(self, parts: List[Dict[str, Any]]) -> str:
        """Formatea partes CAD para PDF"""
        lines = []
        for part in parts:
            lines.append(f"Parte {part.get('part_number', 0)}: {part.get('part_name', '')}")
            lines.append(f"  Material: {part.get('material', '')}")
            lines.append(f"  Descripción: {part.get('description', '')}")
        return "\n".join(lines)
    
    def _format_instructions_for_pdf(self, instructions: List[Dict[str, Any]]) -> str:
        """Formatea instrucciones para PDF"""
        lines = []
        for step in instructions:
            lines.append(f"Paso {step.get('step_number', 0)}: {step.get('description', '')}")
            if step.get('tools_needed'):
                lines.append(f"  Herramientas: {', '.join(step.get('tools_needed', []))}")
            if step.get('time_estimate'):
                lines.append(f"  Tiempo: {step.get('time_estimate', '')}")
        return "\n".join(lines)
    
    def _format_budget_options_for_pdf(self, options: List[Dict[str, Any]]) -> str:
        """Formatea opciones de presupuesto para PDF"""
        lines = []
        for option in options:
            lines.append(f"{option.get('budget_level', '').upper()}: ${option.get('total_cost', 0):.2f}")
            lines.append(f"  {option.get('description', '')}")
        return "\n".join(lines)
    
    def export_all_formats(self, prototype_data: Dict[str, Any]) -> Dict[str, str]:
        """Exporta a todos los formatos disponibles"""
        exports = {}
        
        # Excel
        try:
            exports["excel"] = self.export_to_excel(prototype_data)
        except Exception as e:
            logger.error(f"Error exportando a Excel: {e}")
        
        # PDF Structure
        try:
            pdf_result = self.export_to_pdf_structure(prototype_data)
            exports["pdf_structure"] = pdf_result["file_path"]
        except Exception as e:
            logger.error(f"Error exportando estructura PDF: {e}")
        
        return exports




