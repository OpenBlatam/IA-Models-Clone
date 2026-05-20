"""
Ejemplo de uso de Manuales Hogar AI
=====================================

Ejemplos de cómo usar el sistema para generar manuales.
"""

import asyncio
import os
from manuales_hogar_ai import ManualGenerator
from manuales_hogar_ai.infrastructure import OpenRouterClient


async def example_text_manual():
    """Ejemplo: Generar manual desde texto."""
    print("=" * 60)
    print("EJEMPLO 1: Manual desde Texto")
    print("=" * 60)
    
    client = OpenRouterClient()
    generator = ManualGenerator(openrouter_client=client)
    
    result = await generator.generate_manual_from_text(
        problem_description="""
        Tengo una fuga de agua constante en el grifo de la cocina. 
        El agua gotea incluso cuando el grifo está cerrado. 
        He notado que el goteo es más fuerte por las mañanas.
        """,
        category="plomeria",
        include_safety=True,
        include_tools=True,
        include_materials=True
    )
    
    if result["success"]:
        print("\n✅ Manual generado exitosamente!")
        print(f"Modelo usado: {result['model_used']}")
        print(f"Tokens usados: {result['tokens_used']}")
        print(f"Categoría: {result['category']}")
        print("\n" + "=" * 60)
        print("MANUAL GENERADO:")
        print("=" * 60)
        print(result["manual"])
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    await client.close()


async def example_image_manual():
    """Ejemplo: Generar manual desde imagen."""
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Manual desde Imagen")
    print("=" * 60)
    
    # Nota: Necesitas tener una imagen de ejemplo
    image_path = "ejemplo_problema.jpg"  # Cambiar por ruta real
    
    if not os.path.exists(image_path):
        print(f"⚠️  Imagen no encontrada: {image_path}")
        print("   Crea una imagen de ejemplo o cambia la ruta.")
        return
    
    client = OpenRouterClient()
    generator = ManualGenerator(openrouter_client=client)
    
    result = await generator.generate_manual_from_image(
        image_path=image_path,
        problem_description="Fuga visible en la conexión del grifo",
        category="plomeria",
        include_safety=True,
        include_tools=True,
        include_materials=True
    )
    
    if result["success"]:
        print("\n✅ Manual generado exitosamente!")
        print(f"Modelo usado: {result['model_used']}")
        print(f"Categoría detectada: {result.get('detected_category', 'N/A')}")
        print("\n" + "=" * 60)
        print("ANÁLISIS DE IMAGEN:")
        print("=" * 60)
        print(result.get("image_analysis", "N/A"))
        print("\n" + "=" * 60)
        print("MANUAL GENERADO:")
        print("=" * 60)
        print(result["manual"])
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    await client.close()


async def example_combined_manual():
    """Ejemplo: Generar manual combinando texto e imagen."""
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Manual Combinado (Texto + Imagen)")
    print("=" * 60)
    
    client = OpenRouterClient()
    generator = ManualGenerator(openrouter_client=client)
    
    # Ejemplo solo con texto (sin imagen)
    result = await generator.generate_manual_combined(
        problem_description="""
        El techo de mi casa tiene goteras cuando llueve. 
        He notado manchas de humedad en el techo interior 
        y algunas gotas caen en días de lluvia intensa.
        """,
        category="techos",
        include_safety=True,
        include_tools=True,
        include_materials=True
    )
    
    if result["success"]:
        print("\n✅ Manual generado exitosamente!")
        print(f"Modelo usado: {result['model_used']}")
        print(f"Categoría: {result['category']}")
        print("\n" + "=" * 60)
        print("MANUAL GENERADO:")
        print("=" * 60)
        print(result["manual"])
    else:
        print(f"\n❌ Error: {result.get('error')}")
    
    await client.close()


async def example_different_categories():
    """Ejemplo: Generar manuales para diferentes categorías."""
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Diferentes Categorías")
    print("=" * 60)
    
    client = OpenRouterClient()
    generator = ManualGenerator(openrouter_client=client)
    
    categories_examples = {
        "carpinteria": "La puerta de madera no cierra bien, está desalineada",
        "electricidad": "El interruptor de la luz parpadea y a veces no funciona",
        "pintura": "La pintura de la pared se está descascarando",
        "jardineria": "Las plantas del jardín se están secando a pesar del riego"
    }
    
    for category, problem in categories_examples.items():
        print(f"\n--- Categoría: {category.upper()} ---")
        print(f"Problema: {problem}")
        
        result = await generator.generate_manual_from_text(
            problem_description=problem,
            category=category,
            include_safety=True,
            include_tools=True,
            include_materials=True
        )
        
        if result["success"]:
            print(f"✅ Manual generado para {category}")
            print(f"   Tokens usados: {result['tokens_used']}")
            # Mostrar solo los primeros 200 caracteres del manual
            manual_preview = result["manual"][:200] + "..." if len(result["manual"]) > 200 else result["manual"]
            print(f"   Preview: {manual_preview}")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    await client.close()


async def main():
    """Ejecutar todos los ejemplos."""
    # Verificar API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  ADVERTENCIA: OPENROUTER_API_KEY no está configurada")
        print("   Configura la variable de entorno antes de ejecutar los ejemplos.")
        return
    
    print("\n" + "=" * 60)
    print("MANUALES HOGAR AI - EJEMPLOS DE USO")
    print("=" * 60)
    
    # Ejecutar ejemplos
    await example_text_manual()
    # await example_image_manual()  # Descomentar si tienes una imagen de ejemplo
    await example_combined_manual()
    await example_different_categories()
    
    print("\n" + "=" * 60)
    print("EJEMPLOS COMPLETADOS")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())




