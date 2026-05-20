"""
Material Validator - Validación de materiales y compatibilidad
==============================================================
"""

import logging
from typing import List, Dict, Any, Optional
from ..models.schemas import Material, CADPart

logger = logging.getLogger(__name__)


class MaterialValidator:
    """Validador de materiales y compatibilidad"""
    
    def __init__(self):
        self.compatibility_rules = self._load_compatibility_rules()
        self.material_properties = self._load_material_properties()
    
    def _load_compatibility_rules(self) -> Dict[str, List[str]]:
        """Carga reglas de compatibilidad entre materiales"""
        return {
            "acero_inoxidable": {
                "compatible_with": ["aluminio", "cobre", "plastico_abs", "vidrio"],
                "incompatible_with": ["magnesio"],
                "requires": ["proteccion_galvanica"]
            },
            "aluminio": {
                "compatible_with": ["acero_inoxidable", "plastico_abs"],
                "incompatible_with": ["cobre", "hierro"],
                "requires": ["proteccion_galvanica"]
            },
            "plastico_abs": {
                "compatible_with": ["acero_inoxidable", "aluminio", "vidrio"],
                "incompatible_with": [],
                "temperature_limit": 80  # Celsius
            },
            "vidrio": {
                "compatible_with": ["acero_inoxidable", "plastico_abs"],
                "incompatible_with": [],
                "fragility": "high"
            }
        }
    
    def _load_material_properties(self) -> Dict[str, Dict[str, Any]]:
        """Carga propiedades de materiales"""
        return {
            "acero_inoxidable": {
                "durability": "alta",
                "temperature_resistance": "alta",
                "corrosion_resistance": "muy_alta",
                "weight": "medio",
                "cost": "medio_alto"
            },
            "aluminio": {
                "durability": "media_alta",
                "temperature_resistance": "media",
                "corrosion_resistance": "alta",
                "weight": "bajo",
                "cost": "medio"
            },
            "plastico_abs": {
                "durability": "media",
                "temperature_resistance": "baja",
                "corrosion_resistance": "muy_alta",
                "weight": "muy_bajo",
                "cost": "bajo"
            },
            "vidrio": {
                "durability": "media",
                "temperature_resistance": "alta",
                "corrosion_resistance": "muy_alta",
                "weight": "medio",
                "cost": "medio"
            }
        }
    
    def validate_materials(self, materials: List[Material], 
                          cad_parts: List[CADPart]) -> Dict[str, Any]:
        """Valida materiales y su compatibilidad"""
        issues = []
        warnings = []
        
        # Validar disponibilidad
        availability_issues = self._check_availability(materials)
        issues.extend(availability_issues)
        
        # Validar compatibilidad entre materiales
        compatibility_issues = self._check_compatibility(materials, cad_parts)
        issues.extend(compatibility_issues)
        
        # Validar propiedades para el uso
        property_warnings = self._check_properties(materials, cad_parts)
        warnings.extend(property_warnings)
        
        # Validar costos razonables
        cost_warnings = self._check_costs(materials)
        warnings.extend(cost_warnings)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "summary": self._generate_summary(issues, warnings)
        }
    
    def _check_availability(self, materials: List[Material]) -> List[Dict[str, Any]]:
        """Verifica disponibilidad de materiales"""
        issues = []
        
        for material in materials:
            if len(material.sources) == 0:
                issues.append({
                    "type": "availability",
                    "severity": "high",
                    "material": material.name,
                    "message": f"Material '{material.name}' no tiene fuentes de suministro disponibles"
                })
            elif len(material.sources) == 1:
                issues.append({
                    "type": "availability",
                    "severity": "medium",
                    "material": material.name,
                    "message": f"Material '{material.name}' tiene solo una fuente. Considera tener alternativas."
                })
        
        return issues
    
    def _check_compatibility(self, materials: List[Material], 
                            cad_parts: List[CADPart]) -> List[Dict[str, Any]]:
        """Verifica compatibilidad entre materiales"""
        issues = []
        
        material_names = [m.name.lower() for m in materials]
        part_materials = [p.material.lower() for p in cad_parts]
        
        # Verificar que todos los materiales de las partes estén en la lista
        for part in cad_parts:
            part_material = part.material.lower()
            if not any(part_material in mat_name or mat_name in part_material 
                      for mat_name in material_names):
                issues.append({
                    "type": "compatibility",
                    "severity": "high",
                    "part": part.part_name,
                    "material": part.material,
                    "message": f"Material '{part.material}' usado en '{part.part_name}' no está en la lista de materiales"
                })
        
        # Verificar compatibilidad entre materiales usados juntos
        for i, mat1 in enumerate(materials):
            mat1_key = mat1.name.lower().replace(" ", "_")
            for mat2 in materials[i+1:]:
                mat2_key = mat2.name.lower().replace(" ", "_")
                
                # Verificar reglas de compatibilidad
                rules1 = self.compatibility_rules.get(mat1_key, {})
                incompatible = rules1.get("incompatible_with", [])
                
                if mat2_key in incompatible:
                    issues.append({
                        "type": "compatibility",
                        "severity": "high",
                        "materials": [mat1.name, mat2.name],
                        "message": f"Materiales '{mat1.name}' y '{mat2.name}' pueden ser incompatibles"
                    })
        
        return issues
    
    def _check_properties(self, materials: List[Material], 
                         cad_parts: List[CADPart]) -> List[Dict[str, Any]]:
        """Verifica propiedades de materiales para el uso"""
        warnings = []
        
        for part in cad_parts:
            part_material = part.material.lower().replace(" ", "_")
            properties = self.material_properties.get(part_material, {})
            
            # Verificar durabilidad para partes críticas
            if "motor" in part.part_name.lower() or "cuchilla" in part.part_name.lower():
                if properties.get("durability") in ["baja", "media"]:
                    warnings.append({
                        "type": "property",
                        "severity": "medium",
                        "part": part.part_name,
                        "material": part.material,
                        "message": f"Material '{part.material}' puede no ser suficientemente durable para '{part.part_name}'"
                    })
            
            # Verificar resistencia a temperatura
            if "estufa" in part.part_name.lower() or "quemador" in part.part_name.lower():
                if properties.get("temperature_resistance") in ["baja", "media"]:
                    warnings.append({
                        "type": "property",
                        "severity": "high",
                        "part": part.part_name,
                        "material": part.material,
                        "message": f"Material '{part.material}' puede no resistir altas temperaturas para '{part.part_name}'"
                    })
        
        return warnings
    
    def _check_costs(self, materials: List[Material]) -> List[Dict[str, Any]]:
        """Verifica costos razonables"""
        warnings = []
        
        total_cost = sum(m.total_price for m in materials)
        avg_cost = total_cost / len(materials) if materials else 0
        
        for material in materials:
            # Advertir si un material es muy caro comparado con el promedio
            if material.total_price > avg_cost * 3:
                warnings.append({
                    "type": "cost",
                    "severity": "low",
                    "material": material.name,
                    "message": f"Material '{material.name}' es significativamente más caro que el promedio. Considera alternativas."
                })
        
        return warnings
    
    def _generate_summary(self, issues: List[Dict], warnings: List[Dict]) -> str:
        """Genera un resumen de validación"""
        if not issues and not warnings:
            return "✅ Todos los materiales son válidos y compatibles."
        
        summary_parts = []
        
        if issues:
            high_severity = [i for i in issues if i["severity"] == "high"]
            if high_severity:
                summary_parts.append(f"⚠️ {len(high_severity)} problemas de alta severidad encontrados.")
            summary_parts.append(f"Total de problemas: {len(issues)}")
        
        if warnings:
            summary_parts.append(f"Total de advertencias: {len(warnings)}")
        
        return " ".join(summary_parts) if summary_parts else "Validación completada."




