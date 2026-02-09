"""
Ejemplo de uso del AI Document Processor
=======================================

Script de ejemplo que demuestra cómo usar el procesador de documentos AI.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from services.document_processor import DocumentProcessor
from models.document_models import DocumentProcessingRequest, ProfessionalFormat

async def main():
    """Función principal de ejemplo"""
    print("🚀 AI Document Processor - Ejemplo de Uso")
    print("=" * 50)
    
    # Inicializar procesador
    processor = DocumentProcessor()
    await processor.initialize()
    
    # Crear archivo de ejemplo
    example_content = """
    # Análisis de Mercado Digital
    
    ## Resumen Ejecutivo
    
    El mercado digital ha experimentado un crecimiento exponencial en los últimos años.
    Las empresas necesitan adaptarse a las nuevas tecnologías para mantener su competitividad.
    
    ## Situación Actual
    
    - 85% de las empresas han adoptado herramientas digitales
    - El e-commerce creció 25% el año pasado
    - La transformación digital es prioritaria para el 90% de los CEO
    
    ## Recomendaciones
    
    1. Implementar estrategia de marketing digital
    2. Capacitar al personal en nuevas tecnologías
    3. Invertir en infraestructura tecnológica
    4. Desarrollar presencia en redes sociales
    
    ## Conclusión
    
    La transformación digital no es opcional, es necesaria para la supervivencia empresarial.
    """
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
        temp_file.write(example_content)
        temp_file_path = temp_file.name
    
    try:
        print(f"📄 Archivo de ejemplo creado: {temp_file_path}")
        
        # Crear solicitud de procesamiento
        request = DocumentProcessingRequest(
            filename="analisis_mercado_digital.md",
            target_format=ProfessionalFormat.CONSULTANCY,
            language="es",
            include_analysis=True
        )
        
        print("🔄 Procesando documento...")
        
        # Procesar documento
        result = await processor.process_document(temp_file_path, request)
        
        if result.success:
            print("✅ Documento procesado exitosamente!")
            print(f"⏱️  Tiempo de procesamiento: {result.processing_time:.2f} segundos")
            
            if result.analysis:
                print("\n📊 Análisis del Documento:")
                print(f"   Área: {result.analysis.area.value}")
                print(f"   Categoría: {result.analysis.category.value}")
                print(f"   Confianza: {result.analysis.confidence:.2f}")
                print(f"   Idioma: {result.analysis.language}")
                print(f"   Palabras: {result.analysis.word_count}")
                print(f"   Temas clave: {', '.join(result.analysis.key_topics[:5])}")
            
            if result.professional_document:
                print("\n📋 Documento Profesional Generado:")
                print(f"   Título: {result.professional_document.title}")
                print(f"   Formato: {result.professional_document.format.value}")
                print(f"   Idioma: {result.professional_document.language}")
                print(f"   Secciones: {len(result.professional_document.sections)}")
                
                print("\n📝 Contenido del documento:")
                print("-" * 40)
                print(result.professional_document.content[:500] + "..." if len(result.professional_document.content) > 500 else result.professional_document.content)
                print("-" * 40)
        else:
            print("❌ Error procesando documento:")
            for error in result.errors:
                print(f"   - {error}")
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    print("\n🎉 Ejemplo completado!")

async def test_different_formats():
    """Prueba diferentes formatos de salida"""
    print("\n🧪 Probando diferentes formatos de salida...")
    
    processor = DocumentProcessor()
    await processor.initialize()
    
    # Contenido técnico de ejemplo
    technical_content = """
    # Sistema de Gestión de Inventarios
    
    ## Introducción
    Este documento describe el sistema de gestión de inventarios desarrollado para la empresa.
    
    ## Especificaciones Técnicas
    - Base de datos: PostgreSQL 13
    - Backend: Python FastAPI
    - Frontend: React 18
    - Autenticación: JWT
    
    ## Arquitectura
    El sistema sigue una arquitectura de microservicios con separación clara de responsabilidades.
    
    ## Implementación
    La implementación se realizó en fases, comenzando con el módulo de inventario básico.
    """
    
    formats_to_test = [
        ProfessionalFormat.TECHNICAL,
        ProfessionalFormat.ACADEMIC,
        ProfessionalFormat.COMMERCIAL
    ]
    
    for format_type in formats_to_test:
        print(f"\n📋 Probando formato: {format_type.value}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file.write(technical_content)
            temp_file_path = temp_file.name
        
        try:
            request = DocumentProcessingRequest(
                filename="sistema_inventarios.md",
                target_format=format_type,
                language="es",
                include_analysis=True
            )
            
            result = await processor.process_document(temp_file_path, request)
            
            if result.success and result.professional_document:
                print(f"   ✅ {format_type.value}: {len(result.professional_document.content)} caracteres")
                print(f"   📊 Secciones: {len(result.professional_document.sections)}")
            else:
                print(f"   ❌ Error en {format_type.value}")
        
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

if __name__ == "__main__":
    # Ejecutar ejemplo principal
    asyncio.run(main())
    
    # Ejecutar pruebas de formatos
    asyncio.run(test_different_formats())


