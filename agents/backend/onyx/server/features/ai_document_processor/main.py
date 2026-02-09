"""
Main FastAPI Application for AI Document Processor
=================================================

Este es el punto de entrada principal para el servicio de procesamiento de documentos AI.
Puede leer archivos MD, PDF, Word y transformarlos en documentos profesionales.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List
import asyncio

from services.document_processor import DocumentProcessor
from services.ai_classifier import AIClassifier
from services.professional_transformer import ProfessionalTransformer
from models.document_models import DocumentAnalysis, ProfessionalDocument
from utils.file_handlers import FileHandlerFactory

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor del ciclo de vida de la aplicación"""
    # Inicio
    logger.info("🚀 Iniciando servicio AI Document Processor...")
    
    # Verificar clave API de OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        logger.info("✅ Clave API de OpenAI encontrada - Clasificación AI habilitada")
    else:
        logger.warning("⚠️  Clave API de OpenAI no encontrada - usando clasificación basada en patrones")
    
    # Inicializar servicios
    await document_processor.initialize()
    await ai_classifier.initialize()
    await professional_transformer.initialize()
    
    yield
    
    # Cierre
    logger.info("🛑 Cerrando servicio AI Document Processor...")

# Crear aplicación FastAPI
app = FastAPI(
    title="AI Document Processor",
    description="""
    Un sistema de IA avanzado que puede leer cualquier tipo de archivo (MD, PDF, Word)
    y transformarlo en un documento profesional editable como documentos de consultoría.
    
    ## Características
    
    * **Lectura Multi-formato**: Soporta MD, PDF, Word y más formatos
    * **Clasificación AI**: Identifica automáticamente el área y tipo de documento
    * **Transformación Profesional**: Convierte documentos en formatos profesionales editables
    * **Documentos de Consultoría**: Genera documentos de consultoría estructurados
    * **API REST Completa**: Endpoints para procesamiento y transformación
    
    ## Formatos Soportados
    
    * **Markdown (.md)**: Documentos de texto con formato
    * **PDF (.pdf)**: Documentos PDF con extracción de texto
    * **Word (.docx, .doc)**: Documentos de Microsoft Word
    * **Texto plano (.txt)**: Archivos de texto simple
    
    ## Tipos de Documentos Profesionales
    
    * **Consultoría**: Documentos de consultoría empresarial
    * **Técnico**: Documentación técnica profesional
    * **Académico**: Documentos académicos y de investigación
    * **Comercial**: Documentos comerciales y de marketing
    * **Legal**: Documentos legales y contractuales
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Agregar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurar apropiadamente para producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos para templates
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    return {
        "service": "AI Document Processor",
        "version": "1.0.0",
        "description": "Procesador de documentos AI - Transforma cualquier documento en profesional",
        "endpoints": {
            "process": "/ai-document-processor/process",
            "classify": "/ai-document-processor/classify",
            "transform": "/ai-document-processor/transform",
            "health": "/ai-document-processor/health"
        },
        "documentation": "/docs"
    }

@app.post("/ai-document-processor/process")
async def process_document(
    file: UploadFile = File(...),
    target_format: str = Form("consultancy"),
    language: str = Form("es")
):
    """
    Procesa un documento y lo transforma en un documento profesional
    
    Args:
        file: Archivo a procesar (MD, PDF, Word)
        target_format: Formato objetivo (consultancy, technical, academic, commercial, legal)
        language: Idioma del documento (es, en, fr, etc.)
    
    Returns:
        Documento profesional transformado
    """
    try:
        # Validar tipo de archivo
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
            
            # Extraer texto del documento
            extracted_text = await document_processor.extract_text(temp_file_path, file.filename)
            
            # Clasificar documento
            classification = await ai_classifier.classify_document(extracted_text)
            
            # Transformar a documento profesional
            professional_doc = await professional_transformer.transform_to_professional(
                extracted_text, 
                classification, 
                target_format, 
                language
            )
            
            return {
                "success": True,
                "original_filename": file.filename,
                "classification": classification.dict(),
                "professional_document": professional_doc.dict(),
                "message": "Documento procesado exitosamente"
            }
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error procesando documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")

@app.post("/ai-document-processor/classify")
async def classify_document(
    file: UploadFile = File(...)
):
    """
    Clasifica un documento para identificar su área y tipo
    
    Args:
        file: Archivo a clasificar
    
    Returns:
        Clasificación del documento
    """
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extraer texto
            extracted_text = await document_processor.extract_text(temp_file_path, file.filename)
            
            # Clasificar
            classification = await ai_classifier.classify_document(extracted_text)
            
            return {
                "success": True,
                "filename": file.filename,
                "classification": classification.dict()
            }
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error clasificando documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clasificando documento: {str(e)}")

@app.post("/ai-document-processor/transform")
async def transform_document(
    text: str = Form(...),
    target_format: str = Form("consultancy"),
    language: str = Form("es"),
    document_type: Optional[str] = Form(None)
):
    """
    Transforma texto en un documento profesional
    
    Args:
        text: Texto a transformar
        target_format: Formato objetivo
        language: Idioma
        document_type: Tipo de documento (opcional)
    
    Returns:
        Documento profesional transformado
    """
    try:
        # Clasificar si no se proporciona tipo
        if not document_type:
            classification = await ai_classifier.classify_document(text)
            document_type = classification.document_type
        
        # Transformar
        professional_doc = await professional_transformer.transform_to_professional(
            text, 
            None,  # No necesitamos clasificación completa aquí
            target_format, 
            language
        )
        
        return {
            "success": True,
            "professional_document": professional_doc.dict(),
            "message": "Documento transformado exitosamente"
        }
        
    except Exception as e:
        logger.error(f"Error transformando documento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transformando documento: {str(e)}")

@app.get("/ai-document-processor/health")
async def health_check():
    """Verificación de salud del servicio"""
    return {
        "status": "healthy",
        "service": "AI Document Processor",
        "version": "1.0.0",
        "features": {
            "document_processing": "active",
            "ai_classification": "active",
            "professional_transformation": "active"
        }
    }

@app.get("/ai-document-processor/supported-formats")
async def get_supported_formats():
    """Obtiene los formatos de archivo soportados"""
    return {
        "input_formats": [
            {"extension": ".md", "type": "Markdown", "description": "Documentos Markdown"},
            {"extension": ".pdf", "type": "PDF", "description": "Documentos PDF"},
            {"extension": ".docx", "type": "Word", "description": "Documentos Word (nuevo formato)"},
            {"extension": ".doc", "type": "Word", "description": "Documentos Word (formato antiguo)"},
            {"extension": ".txt", "type": "Texto", "description": "Archivos de texto plano"}
        ],
        "output_formats": [
            {"format": "consultancy", "description": "Documentos de consultoría profesional"},
            {"format": "technical", "description": "Documentación técnica"},
            {"format": "academic", "description": "Documentos académicos"},
            {"format": "commercial", "description": "Documentos comerciales"},
            {"format": "legal", "description": "Documentos legales"}
        ]
    }

# Manejador global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Excepción no manejada: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc) if os.getenv("DEBUG") == "true" else "Ocurrió un error inesperado"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Obtener configuración de variables de entorno
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))  # Puerto diferente para evitar conflictos
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Iniciando servidor en {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


