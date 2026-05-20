"""
Ejemplo de uso del detector multimodal de IA
"""

import asyncio
from api.router import detect_ai_content, batch_detect_ai_content
from schemas import AIDetectionInput, BatchDetectionInput, ContentType


async def ejemplo_deteccion_simple():
    """Ejemplo de detección simple de texto"""
    
    # Texto que parece generado por IA
    texto_ia = """
    As an AI language model, I'd be happy to help you understand this topic. 
    Let me provide a comprehensive explanation that covers the key aspects.
    
    First, it's important to note that this subject involves multiple 
    interconnected components. Second, we need to consider the various 
    implications and applications. Finally, I'll summarize the main points.
    """
    
    input_data = AIDetectionInput(
        content=texto_ia,
        content_type=ContentType.TEXT,
        metadata={"source": "example"}
    )
    
    resultado = await detect_ai_content(input_data)
    
    print(f"¿Es generado por IA? {resultado.is_ai_generated}")
    print(f"Porcentaje de IA: {resultado.ai_percentage:.2f}%")
    print(f"Confianza: {resultado.confidence_score:.2f}")
    
    if resultado.primary_model:
        print(f"Modelo detectado: {resultado.primary_model.model_name}")
        print(f"Proveedor: {resultado.primary_model.provider}")
    
    if resultado.forensic_analysis:
        print(f"Prompt estimado: {resultado.forensic_analysis.estimated_prompt}")
        print(f"Patrones: {resultado.forensic_analysis.prompt_patterns}")


async def ejemplo_deteccion_batch():
    """Ejemplo de detección en batch"""
    
    textos = [
        "Este es un texto normal escrito por un humano.",
        "As an AI, I can help you with that. Let me explain...",
        "Hola, ¿cómo estás? Este es otro texto humano.",
        "I'd be happy to assist you. Here's a comprehensive analysis..."
    ]
    
    items = [
        AIDetectionInput(
            content=texto,
            content_type=ContentType.TEXT
        )
        for texto in textos
    ]
    
    input_data = BatchDetectionInput(items=items, parallel=True)
    resultado = await batch_detect_ai_content(input_data)
    
    print(f"Total procesados: {resultado.total_processed}")
    print(f"Exitosos: {resultado.successful}")
    print(f"Fallidos: {resultado.failed}")
    
    for i, result in enumerate(resultado.results):
        print(f"\nTexto {i+1}:")
        print(f"  Es IA: {result.is_ai_generated}")
        print(f"  Porcentaje: {result.ai_percentage:.2f}%")


if __name__ == "__main__":
    print("=== Ejemplo de Detección Simple ===")
    asyncio.run(ejemplo_deteccion_simple())
    
    print("\n=== Ejemplo de Detección Batch ===")
    asyncio.run(ejemplo_deteccion_batch())






