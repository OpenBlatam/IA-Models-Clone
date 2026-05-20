"""
API Routes para Analizador de Documentos
==========================================

Endpoints REST para el sistema de análisis de documentos.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.document_analyzer import (
    DocumentAnalyzer,
    DocumentType,
    AnalysisTask,
    DocumentAnalysisResult
)
from ..core.fine_tuning_model import FineTuningModel, FineTuningConfig
from ..core.document_processor import DocumentProcessor

# Importar utilidades mejoradas
import sys
from pathlib import Path
utils_path = Path(__file__).parent.parent / "utils"
if utils_path.exists():
    sys.path.insert(0, str(utils_path.parent))
    try:
        from utils.metrics import get_performance_monitor
        from utils.rate_limiter import get_rate_limiter, rate_limit
    except ImportError:
        get_performance_monitor = lambda: None
        get_rate_limiter = lambda *args, **kwargs: None
        rate_limit = lambda *args, **kwargs: lambda f: f

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analizador-documentos", tags=["Document Analyzer"])

# Instancia global del analizador (singleton)
_analyzer: Optional[DocumentAnalyzer] = None
_fine_tuning_model: Optional[FineTuningModel] = None


def get_analyzer() -> DocumentAnalyzer:
    """Dependency para obtener instancia del analizador"""
    global _analyzer
    if _analyzer is None:
        # Intentar cargar modelo fine-tuned si existe
        fine_tuned_path = os.path.join(
            Path(__file__).parent.parent,
            "models",
            "fine_tuned"
        )
        if os.path.exists(fine_tuned_path):
            _analyzer = DocumentAnalyzer(fine_tuned_model_path=fine_tuned_path)
        else:
            _analyzer = DocumentAnalyzer()
    return _analyzer


def get_monitor():
    """Dependency para obtener monitor de rendimiento"""
    return get_performance_monitor()


# ============================================================================
# Modelos Pydantic para requests/responses
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request para análisis de documento"""
    document_content: Optional[str] = Field(None, description="Contenido del documento")
    document_type: Optional[str] = Field(None, description="Tipo de documento")
    tasks: Optional[List[str]] = Field(
        None,
        description="Tareas de análisis a realizar"
    )
    document_id: Optional[str] = Field(None, description="ID del documento")


class ClassifyRequest(BaseModel):
    """Request para clasificación"""
    text: str = Field(..., description="Texto a clasificar")
    labels: Optional[List[str]] = Field(None, description="Etiquetas posibles")


class SummarizeRequest(BaseModel):
    """Request para resumen"""
    text: str = Field(..., description="Texto a resumir")
    max_length: int = Field(150, description="Longitud máxima del resumen")
    min_length: int = Field(30, description="Longitud mínima del resumen")


class QuestionAnswerRequest(BaseModel):
    """Request para pregunta-respuesta"""
    document_content: str = Field(..., description="Contenido del documento")
    question: str = Field(..., description="Pregunta a responder")


class FineTuningRequest(BaseModel):
    """Request para fine-tuning"""
    texts: List[str] = Field(..., description="Textos de entrenamiento")
    labels: List[int] = Field(..., description="Etiquetas de entrenamiento")
    model_name: Optional[str] = Field(None, description="Nombre del modelo base")
    num_labels: int = Field(2, description="Número de clases")
    num_epochs: int = Field(3, description="Número de épocas")
    learning_rate: float = Field(2e-5, description="Learning rate")
    batch_size: int = Field(16, description="Batch size")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "Analizador de Documentos Inteligente",
        "version": "1.0.0",
        "status": "active"
    }


@router.get("/health")
async def health_check():
    """Health check"""
    analyzer = get_analyzer()
    model_info = analyzer.get_model_info()
    return {
        "status": "healthy",
        "model": model_info
    }


@router.post("/analyze", response_model=Dict[str, Any])
@rate_limit(max_requests=50, time_window=60)
async def analyze_document(
    request: AnalyzeRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer),
    monitor = Depends(get_monitor)
):
    """
    Analizar un documento completo
    
    Realiza múltiples análisis sobre el documento incluyendo:
    - Clasificación
    - Resumen
    - Extracción de keywords
    - Análisis de sentimiento
    - Extracción de entidades
    - Modelado de temas
    """
    import time
    start_time = time.time()
    
    try:
        # Convertir tasks
        tasks = None
        if request.tasks:
            tasks = [AnalysisTask(task) for task in request.tasks]
        
        # Analizar
        result = await analyzer.analyze_document(
            document_content=request.document_content,
            document_type=DocumentType(request.document_type) if request.document_type else None,
            tasks=tasks,
            document_id=request.document_id
        )
        
        result_dict = convert_to_dict(result)
        
        # Registrar métricas
        duration = time.time() - start_time
        if monitor:
            monitor.record_request("/analyze", duration, True)
        
        return result_dict
    except Exception as e:
        logger.error(f"Error en análisis: {e}")
        duration = time.time() - start_time
        if monitor:
            monitor.record_request("/analyze", duration, False)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/upload")
async def analyze_uploaded_document(
    file: UploadFile = File(...),
    tasks: Optional[str] = Form(None),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """
    Analizar un documento subido
    
    El archivo se guarda temporalmente, se procesa y se elimina.
    """
    import tempfile
    
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Determinar tipo de documento
        doc_type = DocumentType(Path(file.filename).suffix[1:].lower())
        
        # Convertir tasks
        analysis_tasks = None
        if tasks:
            analysis_tasks = [AnalysisTask(task) for task in tasks.split(",")]
        
        # Analizar
        result = await analyzer.analyze_document(
            document_path=tmp_path,
            document_type=doc_type,
            tasks=analysis_tasks
        )
        
        return convert_to_dict(result)
    except Exception as e:
        logger.error(f"Error analizando archivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpiar archivo temporal
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/classify")
async def classify_text(
    request: ClassifyRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Clasificar un texto"""
    try:
        result = await analyzer.classify_document(
            request.text,
            labels=request.labels
        )
        return {"classification": result}
    except Exception as e:
        logger.error(f"Error en clasificación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize")
async def summarize_text(
    request: SummarizeRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Generar resumen de un texto"""
    try:
        summary = await analyzer.summarize_document(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error en resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keywords")
async def extract_keywords(
    text: str = Form(...),
    top_k: int = Form(10),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Extraer palabras clave"""
    try:
        keywords = await analyzer.extract_keywords(text, top_k=top_k)
        return {"keywords": keywords}
    except Exception as e:
        logger.error(f"Error extrayendo keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment")
async def analyze_sentiment(
    text: str = Form(...),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Analizar sentimiento"""
    try:
        sentiment = await analyzer.analyze_sentiment(text)
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"Error en análisis de sentimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities")
async def extract_entities(
    text: str = Form(...),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Extraer entidades nombradas"""
    try:
        entities = await analyzer.extract_entities(text)
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Error extrayendo entidades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics")
async def extract_topics(
    text: str = Form(...),
    num_topics: int = Form(3),
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Extraer temas del documento"""
    try:
        topics = await analyzer.extract_topics(text, num_topics=num_topics)
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error extrayendo temas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/question-answer")
async def answer_question(
    request: QuestionAnswerRequest,
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Responder pregunta sobre el documento"""
    try:
        result = await analyzer.answer_question(
            request.document_content,
            request.question
        )
        return result
    except Exception as e:
        logger.error(f"Error en Q&A: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tuning/train")
async def train_fine_tuning_model(
    request: FineTuningRequest,
    background_tasks: BackgroundTasks
):
    """
    Entrenar modelo con fine-tuning
    
    El entrenamiento se ejecuta en background.
    """
    try:
        # Configurar fine-tuning
        config = FineTuningConfig(
            model_name=request.model_name or "bert-base-multilingual-cased",
            num_labels=request.num_labels,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            output_dir=os.path.join(
                Path(__file__).parent.parent,
                "models",
                "fine_tuned",
                f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        )
        
        # Crear modelo
        model = FineTuningModel(config=config)
        
        # Preparar dataset
        train_dataset, eval_dataset = model.prepare_dataset(
            request.texts,
            request.labels
        )
        
        # Entrenar en background
        def train_model():
            model.train(train_dataset, eval_dataset)
        
        background_tasks.add_task(train_model)
        
        return {
            "status": "training_started",
            "model_config": {
                "num_labels": request.num_labels,
                "num_epochs": request.num_epochs,
                "output_dir": config.output_dir
            }
        }
    except Exception as e:
        logger.error(f"Error iniciando fine-tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fine-tuning/models")
async def list_fine_tuned_models():
    """Listar modelos fine-tuned disponibles"""
    models_dir = os.path.join(
        Path(__file__).parent.parent,
        "models",
        "fine_tuned"
    )
    
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, "fine_tuning_config.json")
            if os.path.exists(config_path):
                models.append({
                    "name": item,
                    "path": item_path,
                    "config_exists": True
                })
    
    return {"models": models}


@router.post("/fine-tuning/load")
async def load_fine_tuned_model(
    model_path: str = Form(...)
):
    """Cargar un modelo fine-tuned"""
    global _analyzer
    
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        _analyzer = DocumentAnalyzer(fine_tuned_model_path=model_path)
        
        return {
            "status": "loaded",
            "model_info": _analyzer.get_model_info()
        }
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info(
    analyzer: DocumentAnalyzer = Depends(get_analyzer)
):
    """Obtener información del modelo actual"""
    return analyzer.get_model_info()


# Helper para convertir dataclass a dict (fallback si no es dataclass)
def convert_to_dict(obj):
    """Convertir objeto a dict"""
    try:
        return asdict(obj)
    except TypeError:
        # Si no es dataclass, convertir manualmente
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return obj


if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    uvicorn.run(
        "routes:router",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

