"""
Chat Interface - Interfaz de chat para interactuar con el generador de prototipos
===================================================================================

Permite al usuario escribir en el chat qué producto quiere hacer
y el sistema genera toda la documentación necesaria.
"""

import asyncio
import json
from typing import Optional
from core.prototype_generator import PrototypeGenerator
from models.schemas import PrototypeRequest, ProductType


class ChatInterface:
    """Interfaz de chat para generar prototipos"""
    
    def __init__(self):
        self.generator = PrototypeGenerator()
        self.running = True
    
    def print_header(self):
        """Imprime el encabezado del chat"""
        print("\n" + "=" * 70)
        print("🤖 3D Prototype AI - Chat Interface")
        print("=" * 70)
        print("Escribe qué producto quieres hacer y te generaré:")
        print("  • Lista de materiales con precios")
        print("  • Modelos CAD por partes")
        print("  • Instrucciones de ensamblaje")
        print("  • Opciones según tu presupuesto")
        print("\nEjemplos:")
        print("  - 'Quiero hacer una nueva licuadora'")
        print("  - 'Necesito diseñar una estufa de gas'")
        print("  - 'Quiero crear una máquina para cortar madera'")
        print("\nEscribe 'salir' para terminar")
        print("=" * 70 + "\n")
    
    def parse_user_input(self, user_input: str) -> Optional[PrototypeRequest]:
        """Parsea la entrada del usuario y crea un PrototypeRequest"""
        user_input = user_input.strip()
        
        if not user_input or user_input.lower() in ['salir', 'exit', 'quit']:
            return None
        
        # Detectar tipo de producto
        product_type = None
        description_lower = user_input.lower()
        
        if "licuadora" in description_lower or "blender" in description_lower:
            product_type = ProductType.LICUADORA
        elif "estufa" in description_lower or "stove" in description_lower:
            product_type = ProductType.ESTUFA
        elif "maquina" in description_lower or "máquina" in description_lower:
            product_type = ProductType.MAQUINA
        elif any(word in description_lower for word in ["refrigerador", "lavadora", "secadora"]):
            product_type = ProductType.ELECTRODOMESTICO
        elif any(word in description_lower for word in ["taladro", "sierra", "herramienta"]):
            product_type = ProductType.HERRAMIENTA
        
        # Extraer presupuesto si se menciona
        budget = None
        words = user_input.split()
        for i, word in enumerate(words):
            if word.lower() in ['presupuesto', 'budget', 'costo', 'precio'] and i + 1 < len(words):
                try:
                    budget = float(words[i + 1].replace('$', '').replace(',', ''))
                except:
                    pass
        
        return PrototypeRequest(
            product_description=user_input,
            product_type=product_type,
            budget=budget
        )
    
    async def process_request(self, request: PrototypeRequest):
        """Procesa una solicitud y muestra los resultados"""
        print("\n🔄 Generando prototipo... Por favor espera...\n")
        
        try:
            response = await self.generator.generate_prototype(request)
            
            # Mostrar resultados
            print("=" * 70)
            print(f"✅ PROTOTIPO GENERADO: {response.product_name}")
            print("=" * 70)
            
            # Especificaciones
            print("\n📋 ESPECIFICACIONES:")
            for key, value in response.specifications.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                else:
                    print(f"  - {key}: {value}")
            
            # Materiales
            print(f"\n📦 MATERIALES NECESARIOS ({len(response.materials)}):")
            total_cost = 0
            for i, material in enumerate(response.materials, 1):
                print(f"\n  {i}. {material.name}")
                print(f"     Cantidad: {material.quantity} {material.unit}")
                print(f"     Precio unitario: ${material.price_per_unit:.2f}")
                print(f"     Precio total: ${material.total_price:.2f}")
                print(f"     Categoría: {material.category}")
                
                if material.sources:
                    print(f"     Fuentes:")
                    for source in material.sources[:3]:  # Mostrar máximo 3 fuentes
                        source_info = f"       - {source.name}"
                        if source.location:
                            source_info += f" ({source.location})"
                        if source.url:
                            source_info += f" - {source.url}"
                        print(source_info)
                
                total_cost += material.total_price
            
            print(f"\n  💰 COSTO TOTAL ESTIMADO: ${total_cost:.2f}")
            
            # Partes CAD
            print(f"\n🔧 PARTES DEL MODELO CAD ({len(response.cad_parts)}):")
            for part in response.cad_parts:
                print(f"\n  - {part.part_name} (Parte #{part.part_number})")
                print(f"    Material: {part.material}")
                print(f"    Dimensiones: {part.dimensions}")
                print(f"    Formato: {part.cad_format}")
                if part.quantity > 1:
                    print(f"    Cantidad: {part.quantity}")
            
            # Instrucciones de ensamblaje
            print(f"\n📋 INSTRUCCIONES DE ENSAMBLAJE ({len(response.assembly_instructions)} pasos):")
            for step in response.assembly_instructions:
                print(f"\n  Paso {step.step_number}: {step.description}")
                print(f"    Partes: {', '.join(step.parts_involved)}")
                if step.tools_needed:
                    print(f"    Herramientas: {', '.join(step.tools_needed)}")
                if step.time_estimate:
                    print(f"    Tiempo estimado: {step.time_estimate}")
                print(f"    Dificultad: {step.difficulty}")
            
            # Opciones de presupuesto
            print(f"\n💰 OPCIONES SEGÚN PRESUPUESTO ({len(response.budget_options)}):")
            for option in response.budget_options:
                print(f"\n  {option.budget_level.upper()}: ${option.total_cost:.2f}")
                print(f"    Calidad: {option.quality_level}")
                print(f"    Descripción: {option.description}")
                if option.trade_offs:
                    print(f"    Compromisos: {', '.join(option.trade_offs)}")
            
            # Información general
            print(f"\n📊 INFORMACIÓN GENERAL:")
            print(f"  ⏱️  Tiempo estimado de construcción: {response.estimated_build_time}")
            print(f"  📈 Nivel de dificultad: {response.difficulty_level}")
            
            # Documentos generados
            if response.documents:
                print(f"\n📄 DOCUMENTOS GENERADOS:")
                for doc_name, doc_path in response.documents.items():
                    print(f"  - {doc_name}: {doc_path}")
            
            print("\n" + "=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error al generar prototipo: {str(e)}\n")
    
    async def run(self):
        """Ejecuta el chat interactivo"""
        self.print_header()
        
        while self.running:
            try:
                user_input = input("💬 Tú: ").strip()
                
                if not user_input or user_input.lower() in ['salir', 'exit', 'quit']:
                    print("\n👋 ¡Hasta luego!\n")
                    break
                
                request = self.parse_user_input(user_input)
                if request:
                    await self.process_request(request)
                else:
                    print("Por favor, proporciona una descripción del producto.\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


async def main():
    """Función principal"""
    chat = ChatInterface()
    await chat.run()


if __name__ == "__main__":
    asyncio.run(main())




