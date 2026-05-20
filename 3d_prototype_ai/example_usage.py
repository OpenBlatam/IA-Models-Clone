"""
Example Usage - Ejemplos de uso del sistema de prototipos 3D
============================================================
"""

import asyncio
import json
from core.prototype_generator import PrototypeGenerator
from models.schemas import PrototypeRequest, ProductType


async def example_licuadora():
    """Ejemplo: Generar prototipo de licuadora"""
    print("=" * 60)
    print("Ejemplo 1: Generar prototipo de licuadora")
    print("=" * 60)
    
    generator = PrototypeGenerator()
    
    request = PrototypeRequest(
        product_description="Quiero hacer una nueva licuadora potente y fácil de limpiar",
        product_type=ProductType.LICUADORA,
        budget=150.0,
        requirements=["Potente", "Fácil de limpiar", "Durable"],
        location="México"
    )
    
    response = await generator.generate_prototype(request)
    
    print(f"\n✅ Prototipo generado: {response.product_name}")
    print(f"💰 Costo total estimado: ${response.total_cost_estimate:.2f}")
    print(f"⏱️  Tiempo estimado: {response.estimated_build_time}")
    print(f"📊 Dificultad: {response.difficulty_level}")
    print(f"\n📦 Materiales necesarios ({len(response.materials)}):")
    for material in response.materials:
        print(f"  - {material.name}: {material.quantity} {material.unit} (${material.total_price:.2f})")
    
    print(f"\n🔧 Partes CAD ({len(response.cad_parts)}):")
    for part in response.cad_parts:
        print(f"  - {part.part_name} ({part.material})")
    
    print(f"\n📋 Opciones de presupuesto ({len(response.budget_options)}):")
    for option in response.budget_options:
        print(f"  - {option.budget_level.upper()}: ${option.total_cost:.2f} ({option.quality_level})")
    
    print(f"\n📄 Documentos generados:")
    for doc_name, doc_path in response.documents.items():
        print(f"  - {doc_name}: {doc_path}")


async def example_estufa():
    """Ejemplo: Generar prototipo de estufa"""
    print("\n" + "=" * 60)
    print("Ejemplo 2: Generar prototipo de estufa")
    print("=" * 60)
    
    generator = PrototypeGenerator()
    
    request = PrototypeRequest(
        product_description="Necesito diseñar una estufa de gas de 4 quemadores para cocina",
        product_type=ProductType.ESTUFA,
        budget=300.0,
        location="México"
    )
    
    response = await generator.generate_prototype(request)
    
    print(f"\n✅ Prototipo generado: {response.product_name}")
    print(f"💰 Costo total estimado: ${response.total_cost_estimate:.2f}")
    print(f"⏱️  Tiempo estimado: {response.estimated_build_time}")
    print(f"\n📦 Materiales necesarios ({len(response.materials)}):")
    for material in response.materials:
        print(f"  - {material.name}: {material.quantity} {material.unit} (${material.total_price:.2f})")
        if material.sources:
            print(f"    Fuentes: {', '.join([s.name for s in material.sources[:2]])}")


async def example_maquina():
    """Ejemplo: Generar prototipo de máquina personalizada"""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Generar prototipo de máquina personalizada")
    print("=" * 60)
    
    generator = PrototypeGenerator()
    
    request = PrototypeRequest(
        product_description="Quiero crear una máquina para cortar madera de forma precisa y segura",
        product_type=ProductType.MAQUINA,
        requirements=["Segura", "Precisa", "Fácil de usar"]
    )
    
    response = await generator.generate_prototype(request)
    
    print(f"\n✅ Prototipo generado: {response.product_name}")
    print(f"💰 Costo total estimado: ${response.total_cost_estimate:.2f}")
    print(f"⏱️  Tiempo estimado: {response.estimated_build_time}")
    print(f"📊 Dificultad: {response.difficulty_level}")
    
    print(f"\n📋 Instrucciones de ensamblaje ({len(response.assembly_instructions)} pasos):")
    for step in response.assembly_instructions[:3]:  # Mostrar primeros 3 pasos
        print(f"  Paso {step.step_number}: {step.description}")
        print(f"    Dificultad: {step.difficulty}, Tiempo: {step.time_estimate}")


async def main():
    """Ejecutar todos los ejemplos"""
    await example_licuadora()
    await example_estufa()
    await example_maquina()
    
    print("\n" + "=" * 60)
    print("✅ Todos los ejemplos completados")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())




