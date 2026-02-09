"""
Schemas - Modelos de datos para el sistema de prototipos 3D
===========================================================
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ProductType(str, Enum):
    """Tipos de productos que se pueden prototipar"""
    LICUADORA = "licuadora"
    ESTUFA = "estufa"
    MAQUINA = "maquina"
    ELECTRODOMESTICO = "electrodomestico"
    HERRAMIENTA = "herramienta"
    MUEBLE = "mueble"
    DISPOSITIVO = "dispositivo"
    OTRO = "otro"


class MaterialSource(BaseModel):
    """Fuente donde se puede encontrar un material"""
    name: str = Field(..., description="Nombre del proveedor/tienda")
    url: Optional[str] = Field(None, description="URL del proveedor")
    location: Optional[str] = Field(None, description="Ubicación física")
    contact: Optional[str] = Field(None, description="Información de contacto")
    availability: str = Field(..., description="Disponibilidad del material")


class Material(BaseModel):
    """Material necesario para el prototipo"""
    name: str = Field(..., description="Nombre del material")
    quantity: float = Field(..., description="Cantidad necesaria")
    unit: str = Field(..., description="Unidad de medida (kg, m, piezas, etc.)")
    price_per_unit: float = Field(..., description="Precio por unidad")
    total_price: float = Field(..., description="Precio total")
    category: str = Field(..., description="Categoría del material")
    specifications: Optional[Dict[str, Any]] = Field(None, description="Especificaciones técnicas")
    sources: List[MaterialSource] = Field(default_factory=list, description="Fuentes donde obtener el material")
    alternatives: List[str] = Field(default_factory=list, description="Materiales alternativos")


class CADPart(BaseModel):
    """Parte individual del modelo CAD"""
    part_name: str = Field(..., description="Nombre de la parte")
    part_number: int = Field(..., description="Número de parte")
    description: str = Field(..., description="Descripción de la parte")
    material: str = Field(..., description="Material de la parte")
    dimensions: Dict[str, float] = Field(..., description="Dimensiones (largo, ancho, alto)")
    cad_file_path: Optional[str] = Field(None, description="Ruta al archivo CAD")
    cad_format: str = Field(default="STL", description="Formato CAD (STL, STEP, OBJ, etc.)")
    quantity: int = Field(default=1, description="Cantidad de esta parte necesaria")


class AssemblyStep(BaseModel):
    """Paso de ensamblaje"""
    step_number: int = Field(..., description="Número de paso")
    description: str = Field(..., description="Descripción del paso")
    parts_involved: List[str] = Field(..., description="Partes involucradas")
    tools_needed: List[str] = Field(default_factory=list, description="Herramientas necesarias")
    time_estimate: Optional[str] = Field(None, description="Tiempo estimado")
    difficulty: str = Field(default="media", description="Dificultad (fácil, media, difícil)")
    image_path: Optional[str] = Field(None, description="Ruta a imagen ilustrativa")


class BudgetOption(BaseModel):
    """Opción según presupuesto"""
    budget_level: str = Field(..., description="Nivel de presupuesto (bajo, medio, alto, premium)")
    total_cost: float = Field(..., description="Costo total estimado")
    materials: List[Material] = Field(..., description="Materiales incluidos")
    description: str = Field(..., description="Descripción de esta opción")
    trade_offs: List[str] = Field(default_factory=list, description="Compromisos de esta opción")
    quality_level: str = Field(..., description="Nivel de calidad esperado")


class PrototypeRequest(BaseModel):
    """Request para generar un prototipo"""
    product_description: str = Field(..., description="Descripción del producto a prototipar")
    product_type: Optional[ProductType] = Field(None, description="Tipo de producto")
    budget: Optional[float] = Field(None, description="Presupuesto disponible (opcional)")
    requirements: Optional[List[str]] = Field(default_factory=list, description="Requisitos adicionales")
    preferred_materials: Optional[List[str]] = Field(default_factory=list, description="Materiales preferidos")
    location: Optional[str] = Field(None, description="Ubicación para búsqueda de materiales")


class PrototypeResponse(BaseModel):
    """Response con el prototipo completo generado"""
    product_name: str = Field(..., description="Nombre del producto")
    product_description: str = Field(..., description="Descripción completa")
    specifications: Dict[str, Any] = Field(..., description="Especificaciones técnicas")
    materials: List[Material] = Field(..., description="Lista de materiales necesarios")
    cad_parts: List[CADPart] = Field(..., description="Partes del modelo CAD")
    assembly_instructions: List[AssemblyStep] = Field(..., description="Instrucciones de ensamblaje")
    budget_options: List[BudgetOption] = Field(..., description="Opciones según presupuesto")
    total_cost_estimate: float = Field(..., description="Costo total estimado")
    estimated_build_time: str = Field(..., description="Tiempo estimado de construcción")
    difficulty_level: str = Field(..., description="Nivel de dificultad general")
    generated_at: datetime = Field(default_factory=datetime.now, description="Fecha de generación")
    documents: Dict[str, str] = Field(default_factory=dict, description="Rutas a documentos generados")




