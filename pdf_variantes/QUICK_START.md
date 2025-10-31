"""
PDF Variantes - Quick Start Guide
=================================

Guía rápida para comenzar con PDF Variantes.
"""

# 1. INSTALACIÓN
# ==============

# Instalar dependencias
pip install -r requirements.txt

# 2. USO BÁSICO
# =============

from pdf_variantes import (
    PDFUploadHandler, PDFEditor, PDFVariantGenerator,
    PDFTopicExtractor, PDFBrainstorming
)

# Inicializar componentes
upload_handler = PDFUploadHandler()
editor = PDFEditor()
variant_generator = PDFVariantGenerator()
topic_extractor = PDFTopicExtractor()
brainstorming = PDFBrainstorming()

# 3. SUBIR PDF
# ============

async def upload_pdf_example():
    with open("documento.pdf", "rb") as f:
        content = f.read()
    
    metadata, text = await upload_handler.upload_pdf(
        file_content=content,
        filename="documento.pdf",
        auto_process=True,
        extract_text=True
    )
    
    print(f"PDF subido: {metadata.file_id}")
    return metadata.file_id

# 4. GENERAR VARIANTES
# ====================

async def generate_variants_example(file_id):
    with open("documento.pdf", "rb") as f:
        # Generar resumen
        summary = await variant_generator.generate(
            file=f, variant_type="summary"
        )
        
        # Generar esquema
        outline = await variant_generator.generate(
            file=f, variant_type="outline"
        )
        
        # Generar highlights
        highlights = await variant_generator.generate(
            file=f, variant_type="highlights"
        )
    
    return summary, outline, highlights

# 5. EXTRAER TEMAS
# ================

async def extract_topics_example(file_id):
    topics = await topic_extractor.extract_topics(
        file_id=file_id,
        min_relevance=0.5,
        max_topics=20
    )
    
    main_topic = await topic_extractor.extract_main_topic(file_id)
    
    return topics, main_topic

# 6. BRAINSTORMING
# ================

async def brainstorming_example(file_id):
    # Primero extraer temas
    topics_data = await topic_extractor.extract_topics(file_id)
    topics = [t.topic for t in topics_data[:10]]
    
    # Generar ideas
    ideas = await brainstorming.generate_ideas(
        topics=topics,
        number_of_ideas=20,
        diversity_level=0.7
    )
    
    # Guardar ideas
    await brainstorming.save_ideas(file_id, ideas)
    
    return ideas

# 7. USO AVANZADO
# ===============

from pdf_variantes import (
    PDFVariantesAdvanced, AIPDFProcessor, WorkflowEngine,
    ConfigManager, MonitoringSystem
)

# Inicializar componentes avanzados
advanced = PDFVariantesAdvanced()
ai_processor = AIPDFProcessor()
workflow_engine = WorkflowEngine()
config_manager = ConfigManager()
monitoring = MonitoringSystem()

# Mejorar contenido con IA
async def enhance_content_example(file_id):
    result = await advanced.enhance_document_content(
        file_id, "clarity", ["introduction", "conclusion"]
    )
    return result

# Búsqueda semántica
async def semantic_search_example(file_id):
    results = await ai_processor.semantic_search(
        file_id, "artificial intelligence", max_results=10
    )
    return results

# Ejecutar workflow
async def workflow_example(file_id):
    execution_id = await workflow_engine.execute_workflow(
        "pdf_processing", file_id
    )
    return execution_id

# 8. CONFIGURACIÓN
# ================

# Obtener configuración
config = config_manager.get_config()
print(f"Entorno: {config.environment}")
print(f"Debug: {config.debug}")

# Actualizar feature toggle
config_manager.update_feature("ai_enhancement", True)

# 9. MONITOREO
# ============

# Registrar métrica
monitoring.record_metric("pdf_processed", 1.0)

# Obtener estado de salud
health = monitoring.get_health_status()
print(f"Estado: {health['status']}")

# 10. API REST
# ============

from fastapi import FastAPI
from pdf_variantes.api import router

app = FastAPI()
app.include_router(router)

# Ejecutar servidor
# uvicorn main:app --reload

# 11. EJEMPLO COMPLETO
# ===================

async def complete_example():
    # 1. Subir PDF
    file_id = await upload_pdf_example()
    
    # 2. Generar variantes
    summary, outline, highlights = await generate_variants_example(file_id)
    
    # 3. Extraer temas
    topics, main_topic = await extract_topics_example(file_id)
    
    # 4. Brainstorming
    ideas = await brainstorming_example(file_id)
    
    # 5. Mejorar contenido
    enhanced = await enhance_content_example(file_id)
    
    # 6. Búsqueda semántica
    search_results = await semantic_search_example(file_id)
    
    return {
        "file_id": file_id,
        "summary": summary,
        "outline": outline,
        "highlights": highlights,
        "topics": topics,
        "main_topic": main_topic,
        "ideas": ideas,
        "enhanced": enhanced,
        "search_results": search_results
    }

# 12. COMANDOS ÚTILES
# ===================

# Ejecutar tests
# python -m pytest tests/ -v

# Verificar linting
# flake8 pdf_variantes/

# Generar documentación
# sphinx-build -b html docs/ docs/_build/

# 13. TROUBLESHOOTING
# ==================

# Problema: Error de memoria
# Solución: Reducir max_topics o usar procesamiento por lotes

# Problema: PDF corrupto
# Solución: Verificar integridad del archivo

# Problema: Timeout
# Solución: Aumentar timeout en configuración

# 14. RECURSOS ADICIONALES
# ========================

# Documentación completa: README.md
# Lista de características: FEATURES.md
# Resumen de integración: INTEGRATION_SUMMARY.md
# Sistema completo: SYSTEM_COMPLETE.md
# Ejemplos de uso: example_usage.py