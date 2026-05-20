"""
PDF Variantes - Ejemplo de Uso
==============================

Ejemplos de cómo usar el módulo PDF Variantes.
"""

import asyncio
from pathlib import Path
from pdf_variantes import PDFVariantesService

async def ejemplo_subir_pdf():
    """Ejemplo de cómo subir un PDF."""
    
    print("=== Ejemplo: Subir PDF ===\n")
    
    # Inicializar servicio
    service = PDFVariantesService()
    
    # Leer archivo PDF
    pdf_path = Path("ejemplo.pdf")
    if not pdf_path.exists():
        print("⚠️  Archivo ejemplo.pdf no encontrado")
        return
    
    with open(pdf_path, "rb") as f:
        content = f.read()
    
    # Subir PDF
    print(f"📄 Subiendo {pdf_path.name} ({len(content)} bytes)...")
    
    result = await service.upload_and_process_pdf(
        content,
        pdf_path.name,
        auto_process=True,
        extract_text=True,
        detect_language=True
    )
    
    if result["success"]:
        file_id = result["document"]["id"]
        metadata = result["document"]["metadata"]
        
        print(f"✅ PDF cargado exitosamente!")
        print(f"   ID: {file_id}")
        print(f"   Páginas: {metadata['page_count']}")
        print(f"   Palabras: {metadata['word_count']}")
        print(f"   Idioma: {metadata['language']}")
        
        return file_id
    
    print("❌ Error al cargar PDF")
    return None


async def ejemplo_generar_variantes(file_id: str):
    """Ejemplo de cómo generar variantes."""
    
    print("\n=== Ejemplo: Generar Variantes ===\n")
    
    service = PDFVariantesService()
    
    print(f"🔄 Generando variantes para {file_id}...")
    
    result = await service.generate_variants(
        file_id,
        variant_type="rewrite",
        number_of_variants=5,
        continuous=False,
        options={
            "similarity_level": 0.7,
            "creativity_level": 0.6,
            "preserve_meaning": True
        }
    )
    
    if result["success"]:
        print(f"✅ Generadas {result['total_generated']} variantes")
        
        for i, variant in enumerate(result["variants"], 1):
            print(f"   Variante {i}: {variant['variant_id']}")
    
    else:
        print("❌ Error al generar variantes")


async def ejemplo_extraer_temas(file_id: str):
    """Ejemplo de cómo extraer temas."""
    
    print("\n=== Ejemplo: Extraer Temas ===\n")
    
    service = PDFVariantesService()
    
    print(f"🔍 Extrayendo temas de {file_id}...")
    
    result = await service.extract_topics(
        file_id,
        min_relevance=0.5,
        max_topics=10
    )
    
    if result["success"]:
        print(f"✅ Encontrados {result['total_topics']} temas")
        
        if result["main_topic"]:
            print(f"   📌 Tema principal: {result['main_topic']}")
        
        print("\n   Principales temas:")
        for topic in result["topics"][:5]:
            print(f"   - {topic['topic']}: {topic['relevance_score']:.2f}")


async def ejemplo_brainstorm(file_id: str):
    """Ejemplo de cómo generar ideas de brainstorming."""
    
    print("\n=== Ejemplo: Brainstorming ===\n")
    
    service = PDFVariantesService()
    
    print(f"💡 Generando ideas de brainstorming para {file_id}...")
    
    result = await service.generate_brainstorm_ideas(
        file_id,
        topics=None,
        number_of_ideas=10,
        diversity_level=0.7
    )
    
    if result["success"]:
        print(f"✅ Generadas {result['total_ideas']} ideas")
        
        print("\n   Categorías encontradas:")
        for category in result["categories"]:
            print(f"   - {category}")
        
        print("\n   Top ideas:")
        for idea in result["ideas"][:5]:
            print(f"   - [{idea['category']}] {idea['idea']}")
            print(f"     Impacto: {idea['potential_impact']}, Dificultad: {idea['implementation_difficulty']}")


async def ejemplo_completo():
    """Ejemplo completo del flujo."""
    
    print("\n" + "="*50)
    print("📚 EJEMPLO COMPLETO - PDF VARIANTES")
    print("="*50 + "\n")
    
    # 1. Subir PDF
    file_id = await ejemplo_subir_pdf()
    
    if not file_id:
        return
    
    # 2. Extraer temas
    await ejemplo_extraer_temas(file_id)
    
    # 3. Generar brainstorming
    await ejemplo_brainstorm(file_id)
    
    # 4. Generar variantes
    await ejemplo_generar_variantes(file_id)
    
    print("\n" + "="*50)
    print("✅ Ejemplo completado")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(ejemplo_completo())
