"""
Prototype Generator - Generador principal de prototipos 3D
===========================================================

Genera prototipos completos incluyendo:
- Documentación de materiales y precios
- Modelos CAD por partes
- Instrucciones de ensamblaje
- Opciones según presupuesto
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from ..models.schemas import (
    PrototypeRequest,
    PrototypeResponse,
    Material,
    MaterialSource,
    CADPart,
    AssemblyStep,
    BudgetOption,
    ProductType
)

logger = logging.getLogger(__name__)


# Cache simple para prototipos generados
_prototype_cache: Dict[str, Any] = {}


class MaterialDatabase:
    """Base de datos de materiales con precios y fuentes expandida"""
    
    def __init__(self):
        self.materials_db = {
            "acero_inoxidable": {
                "price_per_kg": 2.5,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"},
                    {"name": "Metales y Más", "url": None, "location": "Local"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"}
                ]
            },
            "aluminio": {
                "price_per_kg": 1.8,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "MercadoLibre", "url": "https://www.mercadolibre.com.mx", "location": "México"}
                ]
            },
            "plastico_abs": {
                "price_per_kg": 3.0,
                "sources": [
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "MatterHackers", "url": "https://www.matterhackers.com", "location": "Online"},
                    {"name": "MercadoLibre", "url": "https://www.mercadolibre.com.mx", "location": "México"}
                ]
            },
            "motor_electrico": {
                "price_per_unit": 25.0,
                "sources": [
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "SparkFun", "url": "https://www.sparkfun.com", "location": "Online"},
                    {"name": "Adafruit", "url": "https://www.adafruit.com", "location": "Online"},
                    {"name": "MercadoLibre", "url": "https://www.mercadolibre.com.mx", "location": "México"}
                ]
            },
            "cables": {
                "price_per_meter": 0.5,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"}
                ]
            },
            "tornillos": {
                "price_per_100": 2.0,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"}
                ]
            },
            "vidrio": {
                "price_per_kg": 1.5,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"}
                ]
            },
            "quemador": {
                "price_per_unit": 15.0,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "MercadoLibre", "url": "https://www.mercadolibre.com.mx", "location": "México"}
                ]
            },
            "valvula": {
                "price_per_unit": 8.0,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"}
                ]
            },
            "tubo_gas": {
                "price_per_meter": 3.0,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"}
                ]
            },
            "perilla": {
                "price_per_unit": 2.5,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"}
                ]
            },
            "cobre": {
                "price_per_kg": 6.0,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"}
                ]
            },
            "madera": {
                "price_per_kg": 0.8,
                "sources": [
                    {"name": "Home Depot", "url": "https://www.homedepot.com", "location": "Nacional"},
                    {"name": "Lowe's", "url": "https://www.lowes.com", "location": "Nacional"}
                ]
            },
            "resina_epoxi": {
                "price_per_kg": 12.0,
                "sources": [
                    {"name": "Amazon", "url": "https://www.amazon.com", "location": "Online"},
                    {"name": "MercadoLibre", "url": "https://www.mercadolibre.com.mx", "location": "México"}
                ]
            }
        }
    
    def get_material_info(self, material_name: str, quantity: float, unit: str = "kg") -> Dict[str, Any]:
        """Obtiene información de un material"""
        material_key = material_name.lower().replace(" ", "_")
        if material_key not in self.materials_db:
            # Material genérico si no está en la base de datos
            return {
                "price_per_unit": 1.0,
                "sources": [{"name": "Proveedor Local", "location": "Local"}]
            }
        
        info = self.materials_db[material_key]
        price_key = f"price_per_{unit}" if f"price_per_{unit}" in info else "price_per_kg"
        price_per_unit = info.get(price_key, info.get("price_per_unit", 1.0))
        
        return {
            "price_per_unit": price_per_unit,
            "sources": info.get("sources", [])
        }


class PrototypeGenerator:
    """Generador principal de prototipos 3D"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("output/prototypes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.material_db = MaterialDatabase()
    
    async def generate_prototype(self, request: PrototypeRequest) -> PrototypeResponse:
        """
        Genera un prototipo completo basado en la descripción del producto
        
        Args:
            request: Request con la descripción del producto
            
        Returns:
            PrototypeResponse con toda la información generada
        """
        logger.info(f"Generando prototipo para: {request.product_description}")
        
        # Verificar caché
        cache_key = f"{request.product_description}_{request.product_type}_{request.budget}"
        if cache_key in _prototype_cache:
            logger.info("Retornando prototipo desde caché")
            return _prototype_cache[cache_key]
        
        # Analizar la descripción y determinar tipo de producto
        product_type = request.product_type or self._detect_product_type(request.product_description)
        product_name = self._extract_product_name(request.product_description)
        
        # Generar especificaciones
        specifications = self._generate_specifications(request, product_type)
        
        # Generar lista de materiales
        materials = self._generate_materials(request, product_type, specifications)
        
        # Generar partes CAD
        cad_parts = self._generate_cad_parts(request, product_type, specifications)
        
        # Generar instrucciones de ensamblaje
        assembly_instructions = self._generate_assembly_instructions(cad_parts, product_type)
        
        # Generar opciones según presupuesto
        budget_options = self._generate_budget_options(materials, request.budget)
        
        # Calcular costo total
        total_cost = sum(m.total_price for m in materials)
        
        # Generar documentos
        documents = await self._generate_documents(
            product_name, specifications, materials, cad_parts, 
            assembly_instructions, budget_options
        )
        
        # Crear respuesta
        response = PrototypeResponse(
            product_name=product_name,
            product_description=request.product_description,
            specifications=specifications,
            materials=materials,
            cad_parts=cad_parts,
            assembly_instructions=assembly_instructions,
            budget_options=budget_options,
            total_cost_estimate=total_cost,
            estimated_build_time=self._estimate_build_time(product_type, len(cad_parts)),
            difficulty_level=self._estimate_difficulty(product_type, len(cad_parts)),
            documents=documents
        )
        
        # Exportar también a Markdown
        try:
            from ..utils.document_exporter import DocumentExporter
            exporter = DocumentExporter(self.output_dir)
            markdown_path = await exporter.export_to_markdown(response)
            documents["documento_markdown"] = markdown_path
        except Exception as e:
            logger.warning(f"No se pudo exportar a Markdown: {e}")
        
        # Guardar en caché (limitar tamaño del caché)
        if len(_prototype_cache) < 100:
            _prototype_cache[cache_key] = response
        
        return response
    
    def _detect_product_type(self, description: str) -> ProductType:
        """Detecta el tipo de producto basado en la descripción"""
        description_lower = description.lower()
        
        if "licuadora" in description_lower or "blender" in description_lower:
            return ProductType.LICUADORA
        elif "estufa" in description_lower or "stove" in description_lower:
            return ProductType.ESTUFA
        elif "maquina" in description_lower or "machine" in description_lower:
            return ProductType.MAQUINA
        elif any(word in description_lower for word in ["refrigerador", "lavadora", "secadora"]):
            return ProductType.ELECTRODOMESTICO
        elif any(word in description_lower for word in ["taladro", "sierra", "herramienta"]):
            return ProductType.HERRAMIENTA
        else:
            return ProductType.OTRO
    
    def _extract_product_name(self, description: str) -> str:
        """Extrae el nombre del producto de la descripción"""
        # Simplificación: tomar las primeras palabras
        words = description.split()[:3]
        return " ".join(words).title()
    
    def _generate_specifications(self, request: PrototypeRequest, product_type: ProductType) -> Dict[str, Any]:
        """Genera especificaciones técnicas del producto con más detalles"""
        base_specs = {
            "tipo": product_type.value,
            "dimensiones_estimadas": {
                "largo": "Variable según diseño",
                "ancho": "Variable según diseño",
                "alto": "Variable según diseño"
            },
            "peso_estimado": "Variable según materiales",
            "fuente_energia": "Eléctrica" if product_type in [ProductType.LICUADORA, ProductType.ESTUFA] else "Variable",
            "nivel_dificultad": "Media",
            "tiempo_ensamblaje": "Variable"
        }
        
        if product_type == ProductType.LICUADORA:
            base_specs.update({
                "potencia_motor": "500-1000W",
                "capacidad_vaso": "1-2 litros",
                "velocidades": "3-5 velocidades",
                "material_vaso": "Vidrio o plástico resistente",
                "tipo_cuchillas": "Acero inoxidable de 4-6 cuchillas",
                "voltaje": "110V/220V",
                "certificaciones": "CE, UL (recomendado)",
                "garantia_estimada": "1-2 años con uso normal"
            })
        elif product_type == ProductType.ESTUFA:
            base_specs.update({
                "quemadores": "4-6 quemadores",
                "tipo_gas": "Gas natural o LP",
                "material_superficie": "Acero inoxidable",
                "dimensiones_estandar": "60x60 cm",
                "potencia_quemadores": "5000-15000 BTU por quemador",
                "sistema_encendido": "Eléctrico o manual",
                "certificaciones": "ANSI, CSA (recomendado)",
                "garantia_estimada": "2-5 años"
            })
        elif product_type == ProductType.MAQUINA:
            base_specs.update({
                "tipo_motor": "Eléctrico o manual",
                "potencia": "Variable según aplicación",
                "precision": "Alta",
                "seguridad": "Sistemas de seguridad estándar",
                "certificaciones": "CE, ISO (recomendado)"
            })
        
        # Agregar requisitos del usuario si existen
        if request.requirements:
            base_specs["requisitos_especiales"] = request.requirements
        
        return base_specs
    
    def _generate_materials(self, request: PrototypeRequest, product_type: ProductType, 
                           specifications: Dict[str, Any]) -> List[Material]:
        """Genera la lista de materiales necesarios"""
        materials = []
        
        if product_type == ProductType.LICUADORA:
            materials.extend([
                self._create_material("Motor eléctrico", 1, "unidad", "motor_electrico"),
                self._create_material("Vaso de vidrio", 1, "unidad", "vidrio"),
                self._create_material("Base de plástico ABS", 0.5, "kg", "plastico_abs"),
                self._create_material("Cables eléctricos", 2, "metro", "cables"),
                self._create_material("Tornillos", 20, "unidad", "tornillos"),
                self._create_material("Cuchillas de acero", 1, "unidad", "acero_inoxidable")
            ])
        elif product_type == ProductType.ESTUFA:
            materials.extend([
                self._create_material("Superficie de acero inoxidable", 5, "kg", "acero_inoxidable"),
                self._create_material("Quemadores", 4, "unidad", "quemador"),
                self._create_material("Válvulas de gas", 4, "unidad", "valvula"),
                self._create_material("Tubos de gas", 2, "metro", "tubo_gas"),
                self._create_material("Tornillos", 30, "unidad", "tornillos"),
                self._create_material("Perillas de control", 4, "unidad", "perilla")
            ])
        else:
            # Materiales genéricos para otros tipos
            materials.extend([
                self._create_material("Acero inoxidable", 3, "kg", "acero_inoxidable"),
                self._create_material("Plástico ABS", 1, "kg", "plastico_abs"),
                self._create_material("Cables", 3, "metro", "cables"),
                self._create_material("Tornillos", 25, "unidad", "tornillos")
            ])
        
        return materials
    
    def _create_material(self, name: str, quantity: float, unit: str, material_key: str) -> Material:
        """Crea un objeto Material con información de la base de datos"""
        info = self.material_db.get_material_info(material_key, quantity, unit)
        price_per_unit = info["price_per_unit"]
        total_price = price_per_unit * quantity
        
        sources = [
            MaterialSource(**source) for source in info.get("sources", [])
        ]
        
        return Material(
            name=name,
            quantity=quantity,
            unit=unit,
            price_per_unit=price_per_unit,
            total_price=total_price,
            category=self._get_material_category(material_key),
            sources=sources
        )
    
    def _get_material_category(self, material_key: str) -> str:
        """Obtiene la categoría de un material"""
        categories = {
            "acero_inoxidable": "Metal",
            "aluminio": "Metal",
            "cobre": "Metal",
            "plastico_abs": "Plástico",
            "resina_epoxi": "Plástico",
            "motor_electrico": "Componente eléctrico",
            "cables": "Componente eléctrico",
            "tornillos": "Fijación",
            "vidrio": "Vidrio",
            "quemador": "Componente de cocina",
            "valvula": "Componente de gas",
            "tubo_gas": "Componente de gas",
            "perilla": "Control",
            "madera": "Material natural"
        }
        return categories.get(material_key, "General")
    
    def _generate_cad_parts(self, request: PrototypeRequest, product_type: ProductType,
                           specifications: Dict[str, Any]) -> List[CADPart]:
        """Genera las partes del modelo CAD"""
        parts = []
        
        if product_type == ProductType.LICUADORA:
            parts = [
                CADPart(
                    part_name="Base del motor",
                    part_number=1,
                    description="Base que contiene el motor eléctrico",
                    material="Plástico ABS",
                    dimensions={"largo": 15, "ancho": 15, "alto": 10},
                    cad_format="STL"
                ),
                CADPart(
                    part_name="Vaso",
                    part_number=2,
                    description="Vaso de vidrio para contener los ingredientes",
                    material="Vidrio",
                    dimensions={"diametro": 12, "alto": 20},
                    cad_format="STL"
                ),
                CADPart(
                    part_name="Cuchillas",
                    part_number=3,
                    description="Cuchillas de acero inoxidable",
                    material="Acero inoxidable",
                    dimensions={"diametro": 8, "alto": 2},
                    cad_format="STL"
                ),
                CADPart(
                    part_name="Tapa",
                    part_number=4,
                    description="Tapa con orificio para agregar ingredientes",
                    material="Plástico ABS",
                    dimensions={"diametro": 12, "alto": 3},
                    cad_format="STL"
                )
            ]
        elif product_type == ProductType.ESTUFA:
            parts = [
                CADPart(
                    part_name="Superficie principal",
                    part_number=1,
                    description="Superficie de cocción de acero inoxidable",
                    material="Acero inoxidable",
                    dimensions={"largo": 60, "ancho": 60, "alto": 2},
                    cad_format="STL"
                ),
                CADPart(
                    part_name="Quemador",
                    part_number=2,
                    description="Quemador individual de gas",
                    material="Acero inoxidable",
                    dimensions={"diametro": 15, "alto": 5},
                    cad_format="STL",
                    quantity=4
                ),
                CADPart(
                    part_name="Perilla de control",
                    part_number=3,
                    description="Perilla para controlar el flujo de gas",
                    material="Plástico",
                    dimensions={"diametro": 5, "alto": 3},
                    cad_format="STL",
                    quantity=4
                )
            ]
        else:
            # Partes genéricas
            parts = [
                CADPart(
                    part_name="Carcasa principal",
                    part_number=1,
                    description="Carcasa principal del dispositivo",
                    material="Acero inoxidable",
                    dimensions={"largo": 30, "ancho": 20, "alto": 15},
                    cad_format="STL"
                ),
                CADPart(
                    part_name="Componente interno",
                    part_number=2,
                    description="Componente interno principal",
                    material="Plástico ABS",
                    dimensions={"largo": 15, "ancho": 10, "alto": 8},
                    cad_format="STL"
                )
            ]
        
        return parts
    
    def _generate_assembly_instructions(self, cad_parts: List[CADPart], 
                                       product_type: ProductType) -> List[AssemblyStep]:
        """Genera las instrucciones de ensamblaje"""
        steps = []
        
        if product_type == ProductType.LICUADORA:
            steps = [
                AssemblyStep(
                    step_number=1,
                    description="Instalar el motor eléctrico en la base de plástico",
                    parts_involved=["Base del motor", "Motor eléctrico"],
                    tools_needed=["Destornillador", "Alicates"],
                    time_estimate="15 minutos",
                    difficulty="media"
                ),
                AssemblyStep(
                    step_number=2,
                    description="Conectar los cables eléctricos al motor",
                    parts_involved=["Cables", "Motor eléctrico"],
                    tools_needed=["Pelacables", "Soldador"],
                    time_estimate="10 minutos",
                    difficulty="fácil"
                ),
                AssemblyStep(
                    step_number=3,
                    description="Fijar las cuchillas al eje del motor",
                    parts_involved=["Cuchillas", "Eje del motor"],
                    tools_needed=["Llave ajustable"],
                    time_estimate="5 minutos",
                    difficulty="fácil"
                ),
                AssemblyStep(
                    step_number=4,
                    description="Colocar el vaso sobre la base y asegurar con tornillos",
                    parts_involved=["Vaso", "Base del motor"],
                    tools_needed=["Destornillador"],
                    time_estimate="10 minutos",
                    difficulty="media"
                ),
                AssemblyStep(
                    step_number=5,
                    description="Colocar la tapa sobre el vaso",
                    parts_involved=["Tapa", "Vaso"],
                    tools_needed=[],
                    time_estimate="1 minuto",
                    difficulty="fácil"
                )
            ]
        else:
            # Instrucciones genéricas
            steps = [
                AssemblyStep(
                    step_number=1,
                    description="Preparar todas las partes según el plano",
                    parts_involved=[part.part_name for part in cad_parts],
                    tools_needed=["Herramientas básicas"],
                    time_estimate="Variable",
                    difficulty="media"
                ),
                AssemblyStep(
                    step_number=2,
                    description="Ensamblar las partes principales",
                    parts_involved=[part.part_name for part in cad_parts[:2]],
                    tools_needed=["Destornillador", "Tornillos"],
                    time_estimate="30 minutos",
                    difficulty="media"
                ),
                AssemblyStep(
                    step_number=3,
                    description="Instalar componentes secundarios",
                    parts_involved=[part.part_name for part in cad_parts[2:]],
                    tools_needed=["Herramientas específicas"],
                    time_estimate="20 minutos",
                    difficulty="fácil"
                )
            ]
        
        return steps
    
    def _generate_budget_options(self, materials: List[Material], 
                                requested_budget: Optional[float]) -> List[BudgetOption]:
        """Genera opciones según diferentes niveles de presupuesto"""
        base_cost = sum(m.total_price for m in materials)
        
        options = [
            BudgetOption(
                budget_level="bajo",
                total_cost=base_cost * 0.7,
                materials=[self._get_cheaper_alternative(m) for m in materials],
                description="Opción económica con materiales básicos y funcionales",
                trade_offs=["Menor durabilidad", "Materiales más simples", "Menos acabados"],
                quality_level="Básica"
            ),
            BudgetOption(
                budget_level="medio",
                total_cost=base_cost,
                materials=materials,
                description="Opción balanceada con buena relación calidad-precio",
                trade_offs=[],
                quality_level="Estándar"
            ),
            BudgetOption(
                budget_level="alto",
                total_cost=base_cost * 1.5,
                materials=[self._get_premium_alternative(m) for m in materials],
                description="Opción premium con materiales de alta calidad",
                trade_offs=[],
                quality_level="Alta"
            ),
            BudgetOption(
                budget_level="premium",
                total_cost=base_cost * 2.0,
                materials=[self._get_premium_alternative(m) for m in materials],
                description="Opción premium con los mejores materiales y acabados",
                trade_offs=[],
                quality_level="Premium"
            )
        ]
        
        # Filtrar opciones según presupuesto solicitado
        if requested_budget:
            options = [opt for opt in options if opt.total_cost <= requested_budget * 1.1]
        
        return options
    
    def _get_cheaper_alternative(self, material: Material) -> Material:
        """Obtiene una versión más económica del material"""
        cheaper = material.model_copy()
        cheaper.price_per_unit = material.price_per_unit * 0.7
        cheaper.total_price = cheaper.price_per_unit * material.quantity
        cheaper.name = f"{material.name} (Económico)"
        return cheaper
    
    def _get_premium_alternative(self, material: Material) -> Material:
        """Obtiene una versión premium del material"""
        premium = material.model_copy()
        premium.price_per_unit = material.price_per_unit * 1.5
        premium.total_price = premium.price_per_unit * material.quantity
        premium.name = f"{material.name} (Premium)"
        return premium
    
    def _estimate_build_time(self, product_type: ProductType, num_parts: int) -> str:
        """Estima el tiempo de construcción"""
        base_time = num_parts * 30  # 30 minutos por parte
        if product_type == ProductType.LICUADORA:
            return "2-3 horas"
        elif product_type == ProductType.ESTUFA:
            return "4-6 horas"
        else:
            return f"{base_time // 60}-{base_time // 60 + 1} horas"
    
    def _estimate_difficulty(self, product_type: ProductType, num_parts: int) -> str:
        """Estima el nivel de dificultad"""
        if num_parts <= 3:
            return "Fácil"
        elif num_parts <= 6:
            return "Media"
        else:
            return "Difícil"
    
    async def _generate_documents(self, product_name: str, specifications: Dict[str, Any],
                                 materials: List[Material], cad_parts: List[CADPart],
                                 assembly_instructions: List[AssemblyStep],
                                 budget_options: List[BudgetOption]) -> Dict[str, str]:
        """Genera documentos en formato JSON/Markdown"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = product_name.lower().replace(" ", "_")
        
        # Documento principal con toda la información
        main_doc = {
            "producto": product_name,
            "especificaciones": specifications,
            "materiales": [m.model_dump() for m in materials],
            "partes_cad": [p.model_dump() for p in cad_parts],
            "instrucciones_ensamblaje": [a.model_dump() for a in assembly_instructions],
            "opciones_presupuesto": [b.model_dump() for b in budget_options]
        }
        
        main_doc_path = self.output_dir / f"{safe_name}_{timestamp}_completo.json"
        with open(main_doc_path, "w", encoding="utf-8") as f:
            json.dump(main_doc, f, indent=2, ensure_ascii=False)
        
        # Documento de materiales
        materials_doc = {
            "materiales": [m.model_dump() for m in materials],
            "costo_total": sum(m.total_price for m in materials)
        }
        
        materials_doc_path = self.output_dir / f"{safe_name}_{timestamp}_materiales.json"
        with open(materials_doc_path, "w", encoding="utf-8") as f:
            json.dump(materials_doc, f, indent=2, ensure_ascii=False)
        
        return {
            "documento_completo": str(main_doc_path),
            "documento_materiales": str(materials_doc_path)
        }

