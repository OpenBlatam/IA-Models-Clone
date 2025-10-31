"""
BUL Usage Examples
==================

Ejemplos de uso del sistema BUL para generación de documentos empresariales.
"""

import asyncio
import json
from datetime import datetime
from core import BULEngine, DocumentRequest, BusinessArea, DocumentType
from agents import get_global_agent_manager

async def example_basic_document_generation():
    """Ejemplo básico de generación de documentos"""
    print("=== Ejemplo Básico de Generación de Documentos ===")
    
    # Inicializar el motor BUL
    engine = BULEngine(
        openrouter_api_key="tu_clave_openrouter",
        openai_api_key="tu_clave_openai"  # Opcional
    )
    await engine.initialize()
    
    # Crear una solicitud de documento
    request = DocumentRequest(
        query="Necesito crear una estrategia de marketing digital para mi empresa de tecnología que se enfoca en software B2B",
        business_area=BusinessArea.MARKETING,
        document_type=DocumentType.MARKETING_STRATEGY,
        company_name="TechSolutions",
        industry="Tecnología",
        company_size="Mediana",
        target_audience="Empresas B2B",
        language="es",
        format="markdown"
    )
    
    # Generar el documento
    print("Generando documento...")
    response = await engine.generate_document(request)
    
    # Mostrar resultados
    print(f"Documento generado: {response.title}")
    print(f"Palabras: {response.word_count}")
    print(f"Tiempo de procesamiento: {response.processing_time:.2f}s")
    print(f"Confianza: {response.confidence_score:.2f}")
    print(f"Resumen: {response.summary}")
    print(f"Contenido (primeros 500 caracteres):\n{response.content[:500]}...")
    
    await engine.close()

async def example_agent_selection():
    """Ejemplo de selección automática de agentes"""
    print("\n=== Ejemplo de Selección de Agentes ===")
    
    # Obtener el gestor de agentes
    agent_manager = await get_global_agent_manager()
    
    # Crear diferentes tipos de solicitudes
    requests = [
        DocumentRequest(
            query="Crear un plan de ventas para productos SaaS",
            business_area=BusinessArea.SALES,
            document_type=DocumentType.SALES_PROPOSAL
        ),
        DocumentRequest(
            query="Desarrollar políticas de recursos humanos",
            business_area=BusinessArea.HR,
            document_type=DocumentType.HR_POLICY
        ),
        DocumentRequest(
            query="Análisis financiero y proyecciones",
            business_area=BusinessArea.FINANCE,
            document_type=DocumentType.FINANCIAL_REPORT
        )
    ]
    
    # Encontrar el mejor agente para cada solicitud
    for i, request in enumerate(requests, 1):
        best_agent = await agent_manager.get_best_agent(request)
        if best_agent:
            print(f"Solicitud {i}: {request.query[:50]}...")
            print(f"Mejor agente: {best_agent.name}")
            print(f"Tipo: {best_agent.agent_type.value}")
            print(f"Experiencia: {best_agent.experience_years} años")
            print(f"Tasa de éxito: {best_agent.success_rate:.2f}")
            print()

async def example_batch_generation():
    """Ejemplo de generación en lote"""
    print("\n=== Ejemplo de Generación en Lote ===")
    
    engine = BULEngine(
        openrouter_api_key="tu_clave_openrouter"
    )
    await engine.initialize()
    
    # Crear múltiples solicitudes
    requests = [
        DocumentRequest(
            query="Estrategia de marketing para restaurante local",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            company_name="Restaurante El Buen Sabor",
            industry="Gastronomía"
        ),
        DocumentRequest(
            query="Manual de operaciones para tienda online",
            business_area=BusinessArea.OPERATIONS,
            document_type=DocumentType.OPERATIONAL_MANUAL,
            company_name="TiendaOnline",
            industry="E-commerce"
        ),
        DocumentRequest(
            query="Plan de negocio para startup tecnológica",
            business_area=BusinessArea.STRATEGY,
            document_type=DocumentType.BUSINESS_PLAN,
            company_name="TechStartup",
            industry="Tecnología"
        )
    ]
    
    # Generar documentos en paralelo
    print("Generando documentos en lote...")
    tasks = [engine.generate_document(req) for req in requests]
    responses = await asyncio.gather(*tasks)
    
    # Mostrar resultados
    for i, response in enumerate(responses, 1):
        print(f"Documento {i}: {response.title}")
        print(f"  Área: {response.business_area.value}")
        print(f"  Tipo: {response.document_type.value}")
        print(f"  Palabras: {response.word_count}")
        print(f"  Tiempo: {response.processing_time:.2f}s")
        print()
    
    await engine.close()

async def example_different_languages():
    """Ejemplo de generación en diferentes idiomas"""
    print("\n=== Ejemplo de Generación Multiidioma ===")
    
    engine = BULEngine(
        openrouter_api_key="tu_clave_openrouter"
    )
    await engine.initialize()
    
    # Solicitudes en diferentes idiomas
    language_requests = [
        ("es", "Crear una estrategia de marketing digital"),
        ("en", "Create a digital marketing strategy"),
        ("pt", "Criar uma estratégia de marketing digital"),
        ("fr", "Créer une stratégie de marketing numérique")
    ]
    
    for language, query in language_requests:
        request = DocumentRequest(
            query=query,
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            company_name="GlobalCorp",
            language=language
        )
        
        print(f"Generando en {language.upper()}...")
        response = await engine.generate_document(request)
        print(f"Título: {response.title}")
        print(f"Idioma detectado: {response.metadata.get('language', 'N/A')}")
        print(f"Primeras palabras: {response.content.split()[:10]}")
        print()
    
    await engine.close()

async def example_system_statistics():
    """Ejemplo de estadísticas del sistema"""
    print("\n=== Estadísticas del Sistema ===")
    
    engine = BULEngine(
        openrouter_api_key="tu_clave_openrouter"
    )
    await engine.initialize()
    
    agent_manager = await get_global_agent_manager()
    
    # Generar algunos documentos para tener estadísticas
    sample_requests = [
        DocumentRequest(
            query="Estrategia de marketing",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY
        ),
        DocumentRequest(
            query="Plan de ventas",
            business_area=BusinessArea.SALES,
            document_type=DocumentType.SALES_PROPOSAL
        )
    ]
    
    for request in sample_requests:
        await engine.generate_document(request)
    
    # Obtener estadísticas
    engine_stats = await engine.get_stats()
    agent_stats = await agent_manager.get_agent_stats()
    
    print("Estadísticas del Motor:")
    print(f"  Documentos generados: {engine_stats['documents_generated']}")
    print(f"  Tiempo total de procesamiento: {engine_stats['total_processing_time']:.2f}s")
    print(f"  Tiempo promedio: {engine_stats['average_processing_time']:.2f}s")
    print(f"  Confianza promedio: {engine_stats['average_confidence']:.2f}")
    print(f"  Áreas de negocio usadas: {engine_stats['business_areas_used']}")
    print(f"  Tipos de documentos: {engine_stats['document_types_generated']}")
    
    print("\nEstadísticas de Agentes:")
    print(f"  Total de agentes: {agent_stats['total_agents']}")
    print(f"  Agentes activos: {agent_stats['active_agents']}")
    print(f"  Documentos generados: {agent_stats['total_documents_generated']}")
    print(f"  Tasa de éxito promedio: {agent_stats['average_success_rate']:.2f}")
    print(f"  Tipos de agentes: {agent_stats['agent_types']}")
    
    await engine.close()

async def main():
    """Función principal con todos los ejemplos"""
    print("BUL - Business Universal Language")
    print("Ejemplos de Uso del Sistema")
    print("=" * 50)
    
    try:
        # Ejecutar ejemplos
        await example_basic_document_generation()
        await example_agent_selection()
        await example_batch_generation()
        await example_different_languages()
        await example_system_statistics()
        
        print("\n¡Todos los ejemplos completados exitosamente!")
        
    except Exception as e:
        print(f"Error ejecutando ejemplos: {e}")
        print("Asegúrate de tener configuradas las claves API correctamente.")

if __name__ == "__main__":
    asyncio.run(main())
























