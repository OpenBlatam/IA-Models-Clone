"""
Technical Plans Service - Generación de planos técnicos detallados
"""

import logging
from typing import Dict, Any, Optional, List
from ..core.models import StoreType, StoreLayout, DesignStyle

logger = logging.getLogger(__name__)


class TechnicalPlansService:
    """Servicio para generar planos técnicos detallados"""
    
    def generate_technical_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType,
        style: DesignStyle
    ) -> Dict[str, Any]:
        """Generar plano técnico detallado"""
        
        return {
            "floor_plan": self._generate_floor_plan(layout, store_type),
            "electrical_plan": self._generate_electrical_plan(layout, store_type),
            "plumbing_plan": self._generate_plumbing_plan(layout, store_type),
            "hvac_plan": self._generate_hvac_plan(layout, store_type),
            "lighting_plan": self._generate_lighting_plan(layout, store_type, style),
            "accessibility_plan": self._generate_accessibility_plan(layout),
            "fire_safety_plan": self._generate_fire_safety_plan(layout, store_type),
            "specifications": self._generate_specifications(layout, store_type, style)
        }
    
    def _generate_floor_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar plano de piso"""
        width = layout.dimensions.get("width", 10)
        length = layout.dimensions.get("length", 15)
        area = width * length
        
        return {
            "dimensions": {
                "width": width,
                "length": length,
                "height": layout.dimensions.get("height", 3),
                "total_area": area,
                "usable_area": area * 0.85  # 85% usable, 15% para muros, etc.
            },
            "zones": layout.zones,
            "scale": "1:50",
            "measurements": "En metros",
            "notes": [
                "Todas las medidas son aproximadas",
                "Verificar con arquitecto antes de construcción",
                "Considerar códigos de construcción locales"
            ]
        }
    
    def _generate_electrical_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar plano eléctrico"""
        
        # Estimación de puntos eléctricos según tipo de tienda
        electrical_points = {
            StoreType.RESTAURANT: {
                "outlets": 20,
                "lighting_circuits": 8,
                "equipment_circuits": 6,
                "total_power_kw": 50
            },
            StoreType.CAFE: {
                "outlets": 15,
                "lighting_circuits": 6,
                "equipment_circuits": 4,
                "total_power_kw": 30
            },
            StoreType.BOUTIQUE: {
                "outlets": 12,
                "lighting_circuits": 10,
                "equipment_circuits": 2,
                "total_power_kw": 20
            },
            StoreType.RETAIL: {
                "outlets": 18,
                "lighting_circuits": 8,
                "equipment_circuits": 3,
                "total_power_kw": 25
            }
        }
        
        points = electrical_points.get(store_type, {
            "outlets": 15,
            "lighting_circuits": 6,
            "equipment_circuits": 3,
            "total_power_kw": 25
        })
        
        return {
            "electrical_points": points,
            "outlet_placement": [
                "Distribuir uniformemente en todas las zonas",
                "Mínimo 1 cada 3 metros en perímetro",
                "Altura estándar: 30cm del piso",
                "Considerar puntos para equipos especiales"
            ],
            "circuit_requirements": [
                "Circuito dedicado para equipos de cocina (si aplica)",
                "Circuito para iluminación",
                "Circuito para equipos de punto de venta",
                "Circuito para sistemas de seguridad"
            ],
            "safety_requirements": [
                "Todos los circuitos con protección GFCI",
                "Cableado según código eléctrico local",
                "Panel eléctrico accesible",
                "Etiquetado claro de circuitos"
            ]
        }
    
    def _generate_plumbing_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar plano de plomería"""
        
        plumbing_needs = {
            StoreType.RESTAURANT: {
                "sinks": 4,
                "toilets": 3,
                "water_heaters": 1,
                "drainage_points": 6
            },
            StoreType.CAFE: {
                "sinks": 3,
                "toilets": 2,
                "water_heaters": 1,
                "drainage_points": 4
            },
            StoreType.BOUTIQUE: {
                "sinks": 2,
                "toilets": 2,
                "water_heaters": 1,
                "drainage_points": 3
            },
            StoreType.RETAIL: {
                "sinks": 2,
                "toilets": 2,
                "water_heaters": 1,
                "drainage_points": 3
            }
        }
        
        needs = plumbing_needs.get(store_type, {
            "sinks": 2,
            "toilets": 2,
            "water_heaters": 1,
            "drainage_points": 3
        })
        
        return {
            "plumbing_points": needs,
            "water_requirements": [
                "Conexión principal de agua",
                "Válvulas de cierre accesibles",
                "Sistema de filtración si es necesario",
                "Presión adecuada para todos los puntos"
            ],
            "drainage_requirements": [
                "Sistema de drenaje adecuado",
                "Trampas en todos los desagües",
                "Ventilación adecuada",
                "Cumplir con códigos de salud locales"
            ],
            "location_notes": [
                "Baños accesibles desde área pública",
                "Cocina/barra cerca de conexiones principales",
                "Considerar acceso para mantenimiento"
            ]
        }
    
    def _generate_hvac_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar plano de HVAC (calefacción, ventilación, aire acondicionado)"""
        
        area = layout.dimensions.get("width", 10) * layout.dimensions.get("length", 15)
        height = layout.dimensions.get("height", 3)
        volume = area * height
        
        # Estimación de capacidad HVAC (BTU/h por m³)
        hvac_capacity = {
            StoreType.RESTAURANT: 50,  # Mayor carga por cocina
            StoreType.CAFE: 40,
            StoreType.BOUTIQUE: 35,
            StoreType.RETAIL: 40
        }
        
        capacity_per_m3 = hvac_capacity.get(store_type, 40)
        total_capacity = volume * capacity_per_m3
        
        return {
            "hvac_requirements": {
                "total_capacity_btu": total_capacity,
                "zones": len(layout.zones),
                "ventilation_rate": "6 cambios de aire por hora mínimo"
            },
            "system_type": "Sistema split o central según espacio",
            "placement": [
                "Unidades interiores en cada zona principal",
                "Unidad exterior en lugar accesible",
                "Considerar ruido y estética"
            ],
            "controls": [
                "Termostatos programables por zona",
                "Control de humedad si es necesario",
                "Filtros de alta calidad",
                "Mantenimiento regular programado"
            ],
            "energy_efficiency": [
                "Sistema con certificación Energy Star",
                "Aislamiento adecuado",
                "Ventanas con doble vidrio",
                "Sellado de puertas y ventanas"
            ]
        }
    
    def _generate_lighting_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType,
        style: DesignStyle
    ) -> Dict[str, Any]:
        """Generar plano de iluminación"""
        
        area = layout.dimensions.get("width", 10) * layout.dimensions.get("length", 15)
        
        # Lúmenes por m² según tipo de tienda
        lumens_per_m2 = {
            StoreType.RESTAURANT: 300,
            StoreType.CAFE: 250,
            StoreType.BOUTIQUE: 500,  # Mayor iluminación para exhibición
            StoreType.RETAIL: 400
        }
        
        total_lumens = area * lumens_per_m2.get(store_type, 300)
        
        lighting_types = {
            DesignStyle.MODERN: ["LED downlights", "Track lighting", "Pendant lights"],
            DesignStyle.CLASSIC: ["Chandeliers", "Wall sconces", "Table lamps"],
            DesignStyle.MINIMALIST: ["Recessed LED", "Linear lighting", "Ambient strips"],
            DesignStyle.INDUSTRIAL: ["Exposed bulbs", "Track lighting", "Metal fixtures"],
            DesignStyle.RUSTIC: ["Pendant lights", "Wall sconces", "Natural materials"],
            DesignStyle.LUXURY: ["Crystal chandeliers", "Accent lighting", "Dimmable systems"],
            DesignStyle.ECO_FRIENDLY: ["LED efficient", "Natural light maximization", "Solar options"],
            DesignStyle.VINTAGE: ["Retro fixtures", "Edison bulbs", "Antique styles"]
        }
        
        return {
            "lighting_requirements": {
                "total_lumens": total_lumens,
                "lumens_per_m2": lumens_per_m2.get(store_type, 300),
                "fixture_count": int(total_lumens / 2000)  # Asumiendo 2000 lúmenes por fixture promedio
            },
            "lighting_types": lighting_types.get(style, ["LED general"]),
            "zones": {
                "ambient": "Iluminación general del espacio",
                "task": "Iluminación para áreas de trabajo",
                "accent": "Iluminación para destacar elementos",
                "decorative": "Iluminación decorativa según estilo"
            },
            "controls": [
                "Dimmers en todas las áreas",
                "Sistema de control inteligente opcional",
                "Programación de horarios",
                "Sensores de movimiento en áreas de almacén"
            ],
            "energy_efficiency": [
                "LED de alta eficiencia",
                "Sensores de ocupación",
                "Programación automática",
                "Mantenimiento regular"
            ]
        }
    
    def _generate_accessibility_plan(
        self,
        layout: StoreLayout
    ) -> Dict[str, Any]:
        """Generar plan de accesibilidad"""
        
        return {
            "requirements": [
                "Rampa de acceso si hay escalones (pendiente máxima 1:12)",
                "Puertas con ancho mínimo de 90cm",
                "Pasillos de mínimo 120cm de ancho",
                "Altura de mostradores accesible (máximo 90cm)",
                "Baños accesibles con barras de apoyo",
                "Espacios de giro de 150cm de diámetro"
            ],
            "signage": [
                "Señalización clara y visible",
                "Braille en información importante",
                "Contraste adecuado en textos",
                "Altura de señales accesible"
            ],
            "parking": [
                "Espacios de estacionamiento accesibles",
                "Ruta accesible desde estacionamiento",
                "Señalización clara"
            ],
            "compliance": [
                "Cumplir con ADA (Americans with Disabilities Act) o equivalente local",
                "Certificación de accesibilidad si es requerida",
                "Revisión por especialista en accesibilidad"
            ]
        }
    
    def _generate_fire_safety_plan(
        self,
        layout: StoreLayout,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar plan de seguridad contra incendios"""
        
        area = layout.dimensions.get("width", 10) * layout.dimensions.get("length", 15)
        
        # Extintores según área (1 por cada 75m²)
        extinguisher_count = max(2, int(area / 75))
        
        return {
            "fire_safety_equipment": {
                "extinguishers": extinguisher_count,
                "smoke_detectors": "Uno por zona",
                "fire_alarm_system": "Sistema completo con panel central",
                "emergency_lighting": "En todas las salidas",
                "sprinkler_system": "Requerido según código local"
            },
            "exit_requirements": [
                "Mínimo 2 salidas de emergencia",
                "Salidas claramente marcadas",
                "Rutas de evacuación visibles",
                "Puertas de salida que abren hacia afuera",
                "Ancho mínimo de salidas: 90cm"
            ],
            "evacuation_plan": [
                "Plan de evacuación visible",
                "Punto de reunión designado",
                "Entrenamiento de personal",
                "Simulacros regulares"
            ],
            "compliance": [
                "Cumplir con códigos de incendio locales",
                "Inspección regular por bomberos",
                "Mantenimiento de equipos",
                "Documentación actualizada"
            ]
        }
    
    def _generate_specifications(
        self,
        layout: StoreLayout,
        store_type: StoreType,
        style: DesignStyle
    ) -> Dict[str, Any]:
        """Generar especificaciones técnicas"""
        
        return {
            "structural": {
                "load_bearing": "Verificar con ingeniero estructural",
                "foundation": "Según tipo de suelo y construcción",
                "materials": "Según código de construcción local"
            },
            "finishes": {
                "floors": "Según estilo y uso",
                "walls": "Pintura o acabados según estilo",
                "ceiling": "Acabado según altura y estilo",
                "doors_windows": "Según estilo y eficiencia energética"
            },
            "mechanical": {
                "hvac": "Sistema completo según plan HVAC",
                "plumbing": "Según plan de plomería",
                "electrical": "Según plan eléctrico"
            },
            "special_requirements": {
                "sound_insulation": "Si es necesario según ubicación",
                "security": "Sistema de seguridad y alarmas",
                "signage": "Letrero exterior según permisos",
                "parking": "Si aplica según ubicación"
            },
            "permits_required": [
                "Permiso de construcción",
                "Permiso eléctrico",
                "Permiso de plomería",
                "Permiso de ocupación",
                "Permiso de letrero (si aplica)",
                "Permiso de salud (si aplica para restaurantes)"
            ]
        }




