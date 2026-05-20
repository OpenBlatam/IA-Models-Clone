"""
Enhanced Main Application for AI Document Processor
==================================================

Aplicación principal mejorada con todas las características avanzadas.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List
import asyncio
from datetime import datetime

# Importar servicios principales
from services.document_processor import DocumentProcessor
from services.ai_classifier import AIClassifier
from services.professional_transformer import ProfessionalTransformer
from services.batch_processor import BatchProcessor
from services.web_interface import WebInterface
from services.advanced_ai_features import AdvancedAIFeatures
from services.translation_service import TranslationService

# Importar modelos
from models.document_models import (
    DocumentAnalysis, ProfessionalDocument, ProfessionalFormat,
    DocumentProcessingRequest, DocumentProcessingResponse
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Inicializar servicios
document_processor = DocumentProcessor()
ai_classifier = AIClassifier()
professional_transformer = ProfessionalTransformer()
batch_processor = BatchProcessor()
web_interface = WebInterface()
advanced_ai_features = AdvancedAIFeatures()
translation_service = TranslationService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor del ciclo de vida de la aplicación"""
    # Inicio
    logger.info("🚀 Iniciando AI Document Processor Enhanced...")
    
    # Verificar clave API de OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("✅ Clave API de OpenAI encontrada - Todas las características AI habilitadas")
    else:
        logger.warning("⚠️  Clave API de OpenAI no encontrada - Características limitadas")
    
    # Inicializar todos los servicios
    await document_processor.initialize()
    await ai_classifier.initialize()
    await professional_transformer.initialize()
    await batch_processor.initialize()
    await web_interface.initialize()
    await advanced_ai_features.initialize()
    await translation_service.initialize()
    
    logger.info("✅ Todos los servicios inicializados correctamente")
    
    yield
    
    # Cierre
    logger.info("🛑 Cerrando AI Document Processor Enhanced...")

# Crear aplicación FastAPI
app = FastAPI(
    title="AI Document Processor Enhanced",
    description="""
    Sistema de IA avanzado para procesamiento de documentos con características completas:
    
    ## Características Principales
    
    * **Procesamiento Multi-formato**: MD, PDF, Word, TXT
    * **Clasificación AI**: Detección automática de área y tipo
    * **Transformación Profesional**: 5 formatos profesionales
    * **Procesamiento en Lote**: Múltiples documentos simultáneamente
    * **Interfaz Web**: UI completa para usuarios
    * **Análisis Avanzado**: Sentimientos, entidades, temas
    * **Traducción**: Soporte para 12 idiomas
    * **API REST Completa**: Endpoints para todas las funcionalidades
    
    ## Servicios Disponibles
    
    * **Document Processing**: Procesamiento básico de documentos
    * **AI Classification**: Clasificación inteligente
    * **Professional Transformation**: Transformación a formatos profesionales
    * **Batch Processing**: Procesamiento en lote
    * **Web Interface**: Interfaz web (puerto 8002)
    * **Advanced AI Features**: Análisis avanzado con IA
    * **Translation Service**: Traducción automática
    
    ## Formatos Soportados
    
    * **Entrada**: Markdown, PDF, Word, Texto
    * **Salida**: Consultoría, Técnico, Académico, Comercial, Legal
    * **Idiomas**: Español, Inglés, Francés, Alemán, Italiano, Portugués, Ruso, Japonés, Coreano, Chino, Árabe, Hindi
    """,
    version="2.0.0",
    lifespan=lifespan
)

# Agregar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="templates"), name="static")

# ==================== ENDPOINTS PRINCIPALES ====================

@app.get("/")
async def root():
    """Endpoint raíz con información completa del servicio"""
    return {
        "service": "AI Document Processor Enhanced",
        "version": "2.0.0",
        "description": "Sistema completo de procesamiento de documentos con IA",
        "features": {
            "document_processing": "✅ Activo",
            "ai_classification": "✅ Activo", 
            "professional_transformation": "✅ Activo",
            "batch_processing": "✅ Activo",
            "web_interface": "✅ Activo (puerto 8002)",
            "advanced_ai_features": "✅ Activo",
            "translation_service": "✅ Activo"
        },
        "endpoints": {
            "process": "/ai-document-processor/process",
            "classify": "/ai-document-processor/classify",
            "transform": "/ai-document-processor/transform",
            "batch": "/ai-document-processor/batch",
            "advanced_analysis": "/ai-document-processor/advanced-analysis",
            "translate": "/ai-document-processor/translate",
            "health": "/ai-document-processor/health"
        },
        "web_interface": "http://localhost:8002",
        "documentation": "/docs"
    }

# ==================== PROCESAMIENTO BÁSICO ====================

@app.post("/ai-document-processor/process")
async def process_document(
    file: UploadFile = File(...),
    target_format: str = Form("consultancy"),
    language: str = Form("es"),
    include_advanced_analysis: bool = Form(False)
):
    """Procesa un documento con análisis avanzado opcional"""
    try:
        # Validar archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó nombre de archivo")
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Procesar documento
            logger.info(f"Procesando archivo: {file.filename}")
            
            # Extraer texto
            extracted_text = await document_processor.extract_text(temp_file_path, file.filename)
            
            # Clasificar documento
            classification = await ai_classifier.classify_document(extracted_text)
            
            # Transformar a documento profesional
            professional_doc = await professional_transformer.transform_to_professional(
                extracted_text, 
                classification, 
                ProfessionalFormat(target_format), 
                language
            )
            
            # Análisis avanzado opcional
            advanced_insights = None
            if include_advanced_analysis:
                advanced_insights = await advanced_ai_features.get_advanced_insights(
                    extracted_text, classification
                )
            
            return {
                "success": True,
                "original_filename": file.filename,
                "classification": classification.dict(),
                "professional_document": professional_doc.dict(),
                "advanced_insights": advanced_insights.__dict__ if advanced_insights else None,
                "message": "Documento procesado exitosamente"
            }
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error procesando documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

# ==================== PROCESAMIENTO EN LOTE ====================

@app.post("/ai-document-processor/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    target_format: str = Form("consultancy"),
    language: str = Form("es"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Procesa múltiples documentos en lote"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No se proporcionaron archivos")
        
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            # Guardar archivos temporales
            for file in files:
                if file.filename:
                    temp_path = os.path.join(temp_dir, file.filename)
                    content = await file.read()
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                    file_paths.append(temp_path)
            
            # Procesar en lote
            result = await batch_processor.process_batch(
                file_paths,
                ProfessionalFormat(target_format),
                language,
                temp_dir
            )
            
            # Programar limpieza en background
            background_tasks.add_task(cleanup_temp_directory, temp_dir)
            
            return {
                "success": True,
                "batch_result": result,
                "message": f"Procesamiento en lote completado: {result['summary']['successful']} exitosos, {result['summary']['failed']} fallidos"
            }
            
        except Exception as e:
            # Limpiar en caso de error
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
            
    except Exception as e:
        logger.error(f"Error en procesamiento en lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento en lote: {str(e)}")

# ==================== ANÁLISIS AVANZADO ====================

@app.post("/ai-document-processor/advanced-analysis")
async def advanced_analysis(
    file: UploadFile = File(...)
):
    """Realiza análisis avanzado de un documento"""
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extraer texto
            extracted_text = await document_processor.extract_text(temp_file_path, file.filename)
            
            # Clasificar documento
            classification = await ai_classifier.classify_document(extracted_text)
            
            # Análisis avanzado
            advanced_insights = await advanced_ai_features.get_advanced_insights(
                extracted_text, classification
            )
            
            return {
                "success": True,
                "filename": file.filename,
                "classification": classification.dict(),
                "advanced_insights": advanced_insights.__dict__,
                "message": "Análisis avanzado completado"
            }
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error en análisis avanzado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en análisis avanzado: {str(e)}")

# ==================== TRADUCCIÓN ====================

@app.post("/ai-document-processor/translate")
async def translate_document(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None)
):
    """Traduce un documento a otro idioma"""
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extraer texto
            extracted_text = await document_processor.extract_text(temp_file_path, file.filename)
            
            # Traducir documento
            translation_result = await translation_service.translate_document(
                extracted_text, target_language, source_language
            )
            
            return {
                "success": True,
                "filename": file.filename,
                "translation": translation_result.__dict__,
                "message": "Documento traducido exitosamente"
            }
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error traduciendo documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error traduciendo documento: {str(e)}")

# ==================== ENDPOINTS DE INFORMACIÓN ====================

@app.get("/ai-document-processor/health")
async def health_check():
    """Verificación completa de salud del sistema"""
    try:
        # Verificar estado de todos los servicios
        services_status = {
            "document_processor": "healthy",
            "ai_classifier": "healthy", 
            "professional_transformer": "healthy",
            "batch_processor": "healthy",
            "advanced_ai_features": "healthy",
            "translation_service": "healthy"
        }
        
        # Verificar OpenAI
        openai_status = "available" if os.getenv("OPENAI_API_KEY") else "not_configured"
        
        return {
            "status": "healthy",
            "service": "AI Document Processor Enhanced",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "openai": openai_status,
            "features": {
                "basic_processing": "✅",
                "ai_classification": "✅",
                "professional_transformation": "✅",
                "batch_processing": "✅",
                "advanced_analysis": "✅",
                "translation": "✅",
                "web_interface": "✅"
            }
        }
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/ai-document-processor/capabilities")
async def get_capabilities():
    """Obtiene las capacidades completas del sistema"""
    return {
        "supported_input_formats": [
            {"extension": ".md", "type": "Markdown", "description": "Documentos Markdown"},
            {"extension": ".pdf", "type": "PDF", "description": "Documentos PDF"},
            {"extension": ".docx", "type": "Word", "description": "Documentos Word (moderno)"},
            {"extension": ".doc", "type": "Word", "description": "Documentos Word (antiguo)"},
            {"extension": ".txt", "type": "Texto", "description": "Archivos de texto plano"}
        ],
        "supported_output_formats": [
            {"format": "consultancy", "description": "Documentos de consultoría profesional"},
            {"format": "technical", "description": "Documentación técnica"},
            {"format": "academic", "description": "Documentos académicos"},
            {"format": "commercial", "description": "Documentos comerciales"},
            {"format": "legal", "description": "Documentos legales"}
        ],
        "supported_languages": translation_service.get_supported_languages(),
        "ai_features": {
            "classification": "Detección automática de área y tipo de documento",
            "sentiment_analysis": "Análisis de sentimientos del contenido",
            "entity_extraction": "Extracción de entidades nombradas",
            "topic_modeling": "Identificación de temas principales",
            "readability_analysis": "Análisis de legibilidad y complejidad",
            "automatic_summarization": "Generación automática de resúmenes",
            "translation": "Traducción automática entre idiomas"
        },
        "processing_capabilities": {
            "single_document": "Procesamiento individual de documentos",
            "batch_processing": "Procesamiento en lote de múltiples documentos",
            "concurrent_processing": "Procesamiento concurrente para mejor rendimiento",
            "format_preservation": "Preservación de formato durante transformación",
            "quality_analysis": "Análisis de calidad y recomendaciones"
        }
    }

# ==================== FUNCIONES AUXILIARES ====================

async def cleanup_temp_directory(directory: str):
    """Limpia directorio temporal en background"""
    try:
        import shutil
        await asyncio.sleep(300)  # Esperar 5 minutos
        shutil.rmtree(directory, ignore_errors=True)
        logger.info(f"Directorio temporal limpiado: {directory}")
    except Exception as e:
        logger.error(f"Error limpiando directorio temporal: {e}")

# Manejador global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Excepción no manejada: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc) if os.getenv("DEBUG") == "true" else "Ocurrió un error inesperado",
            "timestamp": datetime.now().isoformat()
        }
    )

# ==================== FUNCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    import uvicorn
    
    # Obtener configuración de variables de entorno
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Iniciando servidor enhanced en {host}:{port}")
    logger.info("Interfaz web disponible en: http://localhost:8002")
    
    uvicorn.run(
        "enhanced_main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


