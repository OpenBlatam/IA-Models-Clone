"""
AI Engine API Routes - Rutas API para motores de IA
"""

from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from ..ai.machine_learning_engine import MachineLearningEngine, ModelType, ModelStatus
from ..ai.deep_learning_engine import DeepLearningEngine, DeepLearningModelType, ModelArchitecture, TrainingConfig
from ..ai.computer_vision_engine import ComputerVisionEngine, VisionTaskType, ImageFormat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["AI Engine"])

# Instancias globales de los motores de IA
ml_engine = MachineLearningEngine()
dl_engine = DeepLearningEngine()
cv_engine = ComputerVisionEngine()


# Modelos Pydantic para Machine Learning
class CreateMLModelRequest(BaseModel):
    name: str
    model_type: str
    algorithm: str
    features: List[str]
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TrainMLModelRequest(BaseModel):
    model_id: str
    training_data: List[Dict[str, Any]]
    target_column: str
    test_size: float = 0.2
    validation_split: float = 0.1


class MLPredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]


# Modelos Pydantic para Deep Learning
class CreateDLModelRequest(BaseModel):
    name: str
    model_type: str
    input_shape: List[int]
    output_shape: List[int]
    layers_config: List[Dict[str, Any]]
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])


class TrainDLModelRequest(BaseModel):
    model_id: str
    training_data: List[List[float]]
    target_data: List[List[float]]
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001


class DLPredictionRequest(BaseModel):
    model_id: str
    data: List[List[float]]


# Modelos Pydantic para Computer Vision
class CVImageAnalysisRequest(BaseModel):
    image_data: str  # Base64 encoded image
    filename: str = "image.jpg"


class CVObjectDetectionRequest(BaseModel):
    image_id: str
    task_type: str = "object_detection"


class CVColorAnalysisRequest(BaseModel):
    image_id: str
    num_colors: int = 5


class CVImageEnhancementRequest(BaseModel):
    image_id: str
    enhancement_type: str = "auto"


# Rutas de Machine Learning
@router.post("/ml/models")
async def create_ml_model(request: CreateMLModelRequest):
    """Crear modelo de Machine Learning."""
    try:
        model_type = ModelType(request.model_type)
        
        model_id = await ml_engine.create_model(
            name=request.name,
            model_type=model_type,
            algorithm=request.algorithm,
            features=request.features,
            hyperparameters=request.hyperparameters
        )
        
        return {
            "model_id": model_id,
            "success": True,
            "message": "Modelo de ML creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de modelo inválido: {e}")
    except Exception as e:
        logger.error(f"Error al crear modelo ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/train")
async def train_ml_model(request: TrainMLModelRequest):
    """Entrenar modelo de Machine Learning."""
    try:
        # Convertir datos a DataFrame
        df = pd.DataFrame(request.training_data)
        
        job_id = await ml_engine.train_model(
            model_id=request.model_id,
            training_data=df,
            target_column=request.target_column,
            test_size=request.test_size,
            validation_split=request.validation_split
        )
        
        return {
            "job_id": job_id,
            "success": True,
            "message": "Entrenamiento de modelo ML iniciado",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al entrenar modelo ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/predict")
async def predict_ml_model(request: MLPredictionRequest):
    """Realizar predicción con modelo de ML."""
    try:
        # Convertir datos a DataFrame
        df = pd.DataFrame(request.data)
        
        prediction = await ml_engine.predict(
            model_id=request.model_id,
            data=df
        )
        
        return {
            "prediction": prediction,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al realizar predicción ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/models/{model_id}")
async def get_ml_model(model_id: str):
    """Obtener información de modelo ML."""
    try:
        performance = await ml_engine.get_model_performance(model_id)
        
        return {
            "model": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener modelo ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/jobs/{job_id}")
async def get_ml_training_job(job_id: str):
    """Obtener estado de trabajo de entrenamiento ML."""
    try:
        job_status = await ml_engine.get_training_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado")
        
        return {
            "job": job_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener trabajo ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Deep Learning
@router.post("/dl/models")
async def create_dl_model(request: CreateDLModelRequest):
    """Crear modelo de Deep Learning."""
    try:
        model_type = DeepLearningModelType(request.model_type)
        
        model_id = await dl_engine.create_model(
            name=request.name,
            model_type=model_type,
            input_shape=tuple(request.input_shape),
            output_shape=tuple(request.output_shape),
            layers_config=request.layers_config,
            optimizer=request.optimizer,
            loss_function=request.loss_function,
            metrics=request.metrics
        )
        
        return {
            "model_id": model_id,
            "success": True,
            "message": "Modelo de Deep Learning creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de modelo inválido: {e}")
    except Exception as e:
        logger.error(f"Error al crear modelo DL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dl/train")
async def train_dl_model(request: TrainDLModelRequest):
    """Entrenar modelo de Deep Learning."""
    try:
        # Convertir datos a arrays numpy
        X_train = np.array(request.training_data)
        y_train = np.array(request.target_data)
        
        # Crear configuración de entrenamiento
        config = TrainingConfig(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        
        job_id = await dl_engine.train_model(
            model_id=request.model_id,
            X_train=X_train,
            y_train=y_train,
            config=config
        )
        
        return {
            "job_id": job_id,
            "success": True,
            "message": "Entrenamiento de modelo DL iniciado",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al entrenar modelo DL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dl/predict")
async def predict_dl_model(request: DLPredictionRequest):
    """Realizar predicción con modelo de DL."""
    try:
        # Convertir datos a array numpy
        data = np.array(request.data)
        
        prediction = await dl_engine.predict(
            model_id=request.model_id,
            data=data
        )
        
        return {
            "prediction": prediction,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al realizar predicción DL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dl/models/{model_id}")
async def get_dl_model(model_id: str):
    """Obtener información de modelo DL."""
    try:
        model_summary = await dl_engine.get_model_summary(model_id)
        
        return {
            "model": model_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener modelo DL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Computer Vision
@router.post("/cv/process-image")
async def process_cv_image(request: CVImageAnalysisRequest):
    """Procesar imagen con Computer Vision."""
    try:
        image_id = await cv_engine.process_image(
            image_data=request.image_data,
            filename=request.filename
        )
        
        return {
            "image_id": image_id,
            "success": True,
            "message": "Imagen procesada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al procesar imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/upload-image")
async def upload_cv_image(file: UploadFile = File(...)):
    """Subir imagen para procesamiento."""
    try:
        # Leer contenido del archivo
        content = await file.read()
        
        # Convertir a base64
        import base64
        image_data = base64.b64encode(content).decode('utf-8')
        
        image_id = await cv_engine.process_image(
            image_data=image_data,
            filename=file.filename
        )
        
        return {
            "image_id": image_id,
            "filename": file.filename,
            "success": True,
            "message": "Imagen subida y procesada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al subir imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/detect-objects")
async def detect_objects(request: CVObjectDetectionRequest):
    """Detectar objetos en imagen."""
    try:
        task_type = VisionTaskType(request.task_type)
        
        detections = await cv_engine.detect_objects(
            image_id=request.image_id,
            task_type=task_type
        )
        
        return {
            "image_id": request.image_id,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "center": d.center,
                    "area": d.area
                }
                for d in detections
            ],
            "total_detections": len(detections),
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de tarea inválido: {e}")
    except Exception as e:
        logger.error(f"Error al detectar objetos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/analyze-colors")
async def analyze_colors(request: CVColorAnalysisRequest):
    """Analizar colores de imagen."""
    try:
        color_palette = await cv_engine.analyze_colors(
            image_id=request.image_id,
            num_colors=request.num_colors
        )
        
        return {
            "image_id": request.image_id,
            "color_palette": {
                "dominant_colors": color_palette.dominant_colors,
                "color_percentages": color_palette.color_percentages,
                "color_names": color_palette.color_names
            },
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al analizar colores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/enhance-image")
async def enhance_image(request: CVImageEnhancementRequest):
    """Mejorar imagen."""
    try:
        enhanced_id = await cv_engine.enhance_image(
            image_id=request.image_id,
            enhancement_type=request.enhancement_type
        )
        
        return {
            "original_image_id": request.image_id,
            "enhanced_image_id": enhanced_id,
            "enhancement_type": request.enhancement_type,
            "success": True,
            "message": "Imagen mejorada exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al mejorar imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cv/analyze/{image_id}")
async def analyze_image_complete(image_id: str):
    """Análisis completo de imagen."""
    try:
        analysis = await cv_engine.get_image_analysis(image_id)
        
        return {
            "analysis": analysis,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al analizar imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/extract-text")
async def extract_text_from_image(image_id: str, language: str = "eng"):
    """Extraer texto de imagen (OCR)."""
    try:
        text_result = await cv_engine.extract_text(
            image_id=image_id,
            language=language
        )
        
        return {
            "text_extraction": text_result,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al extraer texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cv/find-similar")
async def find_similar_images(
    image_id: str,
    threshold: float = 0.8,
    max_results: int = 10
):
    """Encontrar imágenes similares."""
    try:
        similar_images = await cv_engine.find_similar_images(
            image_id=image_id,
            threshold=threshold,
            max_results=max_results
        )
        
        return {
            "query_image_id": image_id,
            "similar_images": similar_images,
            "total_found": len(similar_images),
            "threshold": threshold,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al encontrar imágenes similares: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de estadísticas y salud
@router.get("/ml/stats")
async def get_ml_stats():
    """Obtener estadísticas de Machine Learning."""
    try:
        stats = await ml_engine.get_ml_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dl/stats")
async def get_dl_stats():
    """Obtener estadísticas de Deep Learning."""
    try:
        stats = await dl_engine.get_dl_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas DL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cv/stats")
async def get_cv_stats():
    """Obtener estadísticas de Computer Vision."""
    try:
        stats = await cv_engine.get_cv_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas CV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def ai_health_check():
    """Verificar salud de todos los motores de IA."""
    try:
        ml_health = await ml_engine.health_check()
        dl_health = await dl_engine.health_check()
        cv_health = await cv_engine.health_check()
        
        return {
            "overall_status": "healthy",
            "engines": {
                "machine_learning": ml_health,
                "deep_learning": dl_health,
                "computer_vision": cv_health
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de IA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de utilidad
@router.get("/ml/model-types")
async def get_ml_model_types():
    """Obtener tipos de modelos ML disponibles."""
    return {
        "model_types": [
            {
                "value": model_type.value,
                "name": model_type.name,
                "description": f"Modelo de tipo {model_type.value}"
            }
            for model_type in ModelType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/dl/model-types")
async def get_dl_model_types():
    """Obtener tipos de modelos DL disponibles."""
    return {
        "model_types": [
            {
                "value": model_type.value,
                "name": model_type.name,
                "description": f"Modelo de Deep Learning {model_type.value}"
            }
            for model_type in DeepLearningModelType
        ],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/cv/task-types")
async def get_cv_task_types():
    """Obtener tipos de tareas CV disponibles."""
    return {
        "task_types": [
            {
                "value": task_type.value,
                "name": task_type.name,
                "description": f"Tarea de Computer Vision {task_type.value}"
            }
            for task_type in VisionTaskType
        ],
        "timestamp": datetime.now().isoformat()
    }




