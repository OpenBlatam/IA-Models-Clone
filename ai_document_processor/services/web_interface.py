"""
Interfaz Web para AI Document Processor
======================================

Interfaz web simple para interactuar con el procesador de documentos.
"""

import asyncio
import logging
from typing import Optional, List
from pathlib import Path
import tempfile
import os

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from services.document_processor import DocumentProcessor
from services.batch_processor import BatchProcessor
from models.document_models import ProfessionalFormat

logger = logging.getLogger(__name__)

class WebInterface:
    """Interfaz web para el procesador de documentos"""
    
    def __init__(self):
        self.app = FastAPI(title="AI Document Processor - Web Interface")
        self.document_processor = DocumentProcessor()
        self.batch_processor = BatchProcessor()
        self.templates = None
        self.setup_routes()
    
    async def initialize(self):
        """Inicializa la interfaz web"""
        await self.document_processor.initialize()
        await self.batch_processor.initialize()
        
        # Configurar templates
        templates_dir = Path(__file__).parent.parent / "templates"
        templates_dir.mkdir(exist_ok=True)
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        logger.info("Interfaz web inicializada")
    
    def setup_routes(self):
        """Configura las rutas de la interfaz web"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Página principal"""
            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "title": "AI Document Processor",
                "formats": [f.value for f in ProfessionalFormat]
            })
        
        @self.app.post("/upload")
        async def upload_document(
            request: Request,
            file: UploadFile = File(...),
            target_format: str = Form(...),
            language: str = Form(default="es")
        ):
            """Procesa un documento subido"""
            try:
                # Validar archivo
                if not file.filename:
                    raise HTTPException(status_code=400, detail="No se seleccionó archivo")
                
                # Crear archivo temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Procesar documento
                    from models.document_models import DocumentProcessingRequest
                    
                    request_obj = DocumentProcessingRequest(
                        filename=file.filename,
                        target_format=ProfessionalFormat(target_format),
                        language=language,
                        include_analysis=True
                    )
                    
                    result = await self.document_processor.process_document(
                        temp_file_path, request_obj
                    )
                    
                    if result.success:
                        return self.templates.TemplateResponse("result.html", {
                            "request": request,
                            "title": "Resultado del Procesamiento",
                            "result": result,
                            "filename": file.filename
                        })
                    else:
                        return self.templates.TemplateResponse("error.html", {
                            "request": request,
                            "title": "Error en el Procesamiento",
                            "error": result.message,
                            "errors": result.errors
                        })
                
                finally:
                    # Limpiar archivo temporal
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error procesando documento: {e}")
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error",
                    "error": str(e)
                })
        
        @self.app.get("/batch", response_class=HTMLResponse)
        async def batch_page(request: Request):
            """Página de procesamiento en lote"""
            return self.templates.TemplateResponse("batch.html", {
                "request": request,
                "title": "Procesamiento en Lote",
                "formats": [f.value for f in ProfessionalFormat]
            })
        
        @self.app.post("/batch/upload")
        async def batch_upload(
            request: Request,
            files: List[UploadFile] = File(...),
            target_format: str = Form(...),
            language: str = Form(default="es")
        ):
            """Procesa múltiples documentos"""
            try:
                if not files:
                    raise HTTPException(status_code=400, detail="No se seleccionaron archivos")
                
                # Crear directorio temporal para archivos
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                try:
                    # Guardar archivos temporales
                    for file in files:
                        if file.filename:
                            temp_path = Path(temp_dir) / file.filename
                            content = await file.read()
                            temp_path.write_bytes(content)
                            file_paths.append(str(temp_path))
                    
                    # Procesar en lote
                    result = await self.batch_processor.process_batch(
                        file_paths,
                        ProfessionalFormat(target_format),
                        language,
                        temp_dir
                    )
                    
                    return self.templates.TemplateResponse("batch_result.html", {
                        "request": request,
                        "title": "Resultado del Procesamiento en Lote",
                        "result": result,
                        "file_count": len(files)
                    })
                
                finally:
                    # Limpiar archivos temporales
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            except Exception as e:
                logger.error(f"Error en procesamiento en lote: {e}")
                return self.templates.TemplateResponse("error.html", {
                    "request": request,
                    "title": "Error en Lote",
                    "error": str(e)
                })
        
        @self.app.get("/api/health")
        async def health_check():
            """Verificación de salud de la API"""
            return {
                "status": "healthy",
                "service": "AI Document Processor Web Interface",
                "version": "1.0.0"
            }
        
        @self.app.get("/download/{filename}")
        async def download_file(filename: str):
            """Descarga un archivo procesado"""
            # Implementar descarga de archivos procesados
            return {"message": f"Descarga de {filename} - Funcionalidad en desarrollo"}
    
    def create_templates(self):
        """Crea las plantillas HTML necesarias"""
        templates_dir = Path(__file__).parent.parent / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Plantilla principal
        index_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: #f8f9fa; padding: 20px; border-radius: 4px; margin-top: 20px; }
        .error { background: #f8d7da; color: #721c24; padding: 20px; border-radius: 4px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Sube un documento y lo transformaremos en un documento profesional.</p>
    
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="file">Seleccionar archivo:</label>
            <input type="file" id="file" name="file" accept=".md,.pdf,.docx,.doc,.txt" required>
        </div>
        
        <div class="form-group">
            <label for="target_format">Formato objetivo:</label>
            <select id="target_format" name="target_format" required>
                {% for format in formats %}
                <option value="{{ format }}">{{ format|title }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="form-group">
            <label for="language">Idioma:</label>
            <select id="language" name="language">
                <option value="es">Español</option>
                <option value="en">English</option>
            </select>
        </div>
        
        <button type="submit">Procesar Documento</button>
    </form>
    
    <div style="margin-top: 40px;">
        <h2>Procesamiento en Lote</h2>
        <p>Procesa múltiples documentos a la vez.</p>
        <a href="/batch"><button>Ir a Procesamiento en Lote</button></a>
    </div>
    
    <div style="margin-top: 40px;">
        <h2>API</h2>
        <p>Para integración con otros sistemas, usa la API REST:</p>
        <ul>
            <li><code>POST /ai-document-processor/process</code> - Procesar documento</li>
            <li><code>POST /ai-document-processor/classify</code> - Clasificar documento</li>
            <li><code>GET /ai-document-processor/health</code> - Estado del servicio</li>
        </ul>
        <p>Documentación completa: <a href="/docs">/docs</a></p>
    </div>
</body>
</html>
        """
        
        # Guardar plantilla
        (templates_dir / "index.html").write_text(index_html, encoding='utf-8')
        
        # Plantilla de resultado
        result_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .result { background: #f8f9fa; padding: 20px; border-radius: 4px; margin-top: 20px; }
        .analysis { background: #e7f3ff; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .content { background: white; padding: 20px; border-radius: 4px; margin: 10px 0; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
        .back-link { margin-top: 20px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="result">
        <h2>Archivo Procesado: {{ filename }}</h2>
        
        {% if result.analysis %}
        <div class="analysis">
            <h3>Análisis del Documento</h3>
            <p><strong>Área:</strong> {{ result.analysis.area }}</p>
            <p><strong>Categoría:</strong> {{ result.analysis.category }}</p>
            <p><strong>Confianza:</strong> {{ "%.2f"|format(result.analysis.confidence) }}</p>
            <p><strong>Idioma:</strong> {{ result.analysis.language }}</p>
            <p><strong>Palabras:</strong> {{ result.analysis.word_count }}</p>
            <p><strong>Temas clave:</strong> {{ result.analysis.key_topics|join(", ") }}</p>
        </div>
        {% endif %}
        
        {% if result.professional_document %}
        <div class="content">
            <h3>Documento Profesional Generado</h3>
            <p><strong>Formato:</strong> {{ result.professional_document.format }}</p>
            <p><strong>Idioma:</strong> {{ result.professional_document.language }}</p>
            <p><strong>Secciones:</strong> {{ result.professional_document.sections|length }}</p>
            
            <h4>Contenido:</h4>
            <pre>{{ result.professional_document.content }}</pre>
        </div>
        {% endif %}
        
        <p><strong>Tiempo de procesamiento:</strong> {{ "%.2f"|format(result.processing_time) }} segundos</p>
    </div>
    
    <div class="back-link">
        <a href="/">← Volver al inicio</a>
    </div>
</body>
</html>
        """
        
        (templates_dir / "result.html").write_text(result_html, encoding='utf-8')
        
        # Plantilla de error
        error_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .error { background: #f8d7da; color: #721c24; padding: 20px; border-radius: 4px; margin-top: 20px; }
        .back-link { margin-top: 20px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="error">
        <h2>Error</h2>
        <p>{{ error }}</p>
        
        {% if errors %}
        <h3>Detalles:</h3>
        <ul>
            {% for err in errors %}
            <li>{{ err }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    
    <div class="back-link">
        <a href="/">← Volver al inicio</a>
    </div>
</body>
</html>
        """
        
        (templates_dir / "error.html").write_text(error_html, encoding='utf-8')
        
        # Plantilla de procesamiento en lote
        batch_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .back-link { margin-top: 20px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Selecciona múltiples documentos para procesar en lote.</p>
    
    <form action="/batch/upload" method="post" enctype="multipart/form-data">
        <div class="form-group">
            <label for="files">Seleccionar archivos:</label>
            <input type="file" id="files" name="files" accept=".md,.pdf,.docx,.doc,.txt" multiple required>
        </div>
        
        <div class="form-group">
            <label for="target_format">Formato objetivo:</label>
            <select id="target_format" name="target_format" required>
                {% for format in formats %}
                <option value="{{ format }}">{{ format|title }}</option>
                {% endfor %}
            </select>
        </div>
        
        <div class="form-group">
            <label for="language">Idioma:</label>
            <select id="language" name="language">
                <option value="es">Español</option>
                <option value="en">English</option>
            </select>
        </div>
        
        <button type="submit">Procesar Documentos</button>
    </form>
    
    <div class="back-link">
        <a href="/">← Volver al inicio</a>
    </div>
</body>
</html>
        """
        
        (templates_dir / "batch.html").write_text(batch_html, encoding='utf-8')
        
        # Plantilla de resultado en lote
        batch_result_html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 4px; margin: 20px 0; }
        .success { background: #d4edda; color: #155724; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .back-link { margin-top: 20px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="summary">
        <h2>Resumen del Procesamiento</h2>
        <p><strong>Total de archivos:</strong> {{ result.summary.total_files }}</p>
        <p><strong>Exitosos:</strong> {{ result.summary.successful }}</p>
        <p><strong>Fallidos:</strong> {{ result.summary.failed }}</p>
        <p><strong>Tiempo total:</strong> {{ "%.2f"|format(result.summary.processing_time) }} segundos</p>
        <p><strong>Tiempo promedio por archivo:</strong> {{ "%.2f"|format(result.summary.average_time_per_file) }} segundos</p>
    </div>
    
    {% if result.results %}
    <h2>Documentos Procesados Exitosamente</h2>
    {% for doc_result in result.results %}
    <div class="success">
        <h3>{{ doc_result.file }}</h3>
        <p><strong>Formato:</strong> {{ doc_result.professional_document.format if doc_result.professional_document else 'N/A' }}</p>
        <p><strong>Tiempo:</strong> {{ "%.2f"|format(doc_result.processing_time) }} segundos</p>
    </div>
    {% endfor %}
    {% endif %}
    
    {% if result.errors %}
    <h2>Errores</h2>
    {% for error in result.errors %}
    <div class="error">
        <h3>{{ error.file }}</h3>
        <p>{{ error.error }}</p>
    </div>
    {% endfor %}
    {% endif %}
    
    <div class="back-link">
        <a href="/">← Volver al inicio</a>
    </div>
</body>
</html>
        """
        
        (templates_dir / "batch_result.html").write_text(batch_result_html, encoding='utf-8')
        
        logger.info("Plantillas HTML creadas")
    
    def run(self, host: str = "0.0.0.0", port: int = 8002):
        """Ejecuta la interfaz web"""
        # Crear plantillas si no existen
        self.create_templates()
        
        logger.info(f"Iniciando interfaz web en {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Función para ejecutar la interfaz web
async def run_web_interface():
    """Función principal para ejecutar la interfaz web"""
    interface = WebInterface()
    await interface.initialize()
    interface.run()

if __name__ == "__main__":
    asyncio.run(run_web_interface())


