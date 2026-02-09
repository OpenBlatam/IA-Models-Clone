"""
FastAPI endpoints for Robot Maintenance Teaching AI.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import logging
import time
from datetime import datetime

from ..core.maintenance_tutor import RobotMaintenanceTutor
from ..core.nlp_processor import MaintenanceNLPProcessor
from ..core.ml_predictor import MaintenancePredictor
from ..config.maintenance_config import MaintenanceConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Robot Maintenance Teaching AI",
    description="API para enseñanza de mantenimiento de robots y máquinas",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SuccessResponse(BaseModel):
    """Success response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if logger.level <= logging.DEBUG else "An error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    return response

tutor_instance: Optional[RobotMaintenanceTutor] = None
nlp_processor: Optional[MaintenanceNLPProcessor] = None
ml_predictor: Optional[MaintenancePredictor] = None


class TeachingRequest(BaseModel):
    robot_type: str = Field(..., description="Tipo de robot")
    maintenance_type: str = Field(..., description="Tipo de mantenimiento")
    difficulty: str = Field(default="intermediate", description="Nivel de dificultad")
    context: Optional[str] = Field(None, description="Contexto adicional")


class DiagnosisRequest(BaseModel):
    symptoms: str = Field(..., description="Síntomas o problemas observados")
    robot_type: str = Field(..., description="Tipo de robot")
    context: Optional[str] = Field(None, description="Contexto adicional")


class ComponentRequest(BaseModel):
    component_name: str = Field(..., description="Nombre del componente")
    robot_type: str = Field(..., description="Tipo de robot")
    difficulty: str = Field(default="intermediate", description="Nivel de dificultad")


class ScheduleRequest(BaseModel):
    robot_type: str = Field(..., description="Tipo de robot")
    usage_hours: int = Field(..., description="Horas de operación")
    environment: Optional[str] = Field(None, description="Ambiente de operación")


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Pregunta sobre mantenimiento")
    robot_type: Optional[str] = Field(None, description="Tipo de robot")
    context: Optional[str] = Field(None, description="Contexto adicional")


class NLPRequest(BaseModel):
    text: str = Field(..., description="Texto a procesar")


class PredictionRequest(BaseModel):
    robot_type: str = Field(..., description="Tipo de robot")
    operating_hours: float = Field(..., ge=0, description="Horas de operación totales")
    error_count: int = Field(..., ge=0, description="Número de errores")
    temperature: float = Field(..., description="Temperatura de operación")
    vibration_level: float = Field(..., ge=0, le=1, description="Nivel de vibración (0-1)")
    last_maintenance_hours: float = Field(..., ge=0, description="Horas desde último mantenimiento")
    
    @validator('vibration_level')
    def validate_vibration(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Vibration level must be between 0 and 1')
        return v


class TrainingRequest(BaseModel):
    """Request model for ML training."""
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size ratio")
    save_model: bool = Field(default=True, description="Whether to save the trained model")


def get_tutor() -> RobotMaintenanceTutor:
    """Get or create tutor instance."""
    global tutor_instance
    if tutor_instance is None:
        config = MaintenanceConfig()
        tutor_instance = RobotMaintenanceTutor(config)
    return tutor_instance


def get_nlp_processor() -> MaintenanceNLPProcessor:
    """Get or create NLP processor instance."""
    global nlp_processor
    if nlp_processor is None:
        from ..config.maintenance_config import NLPConfig
        nlp_processor = MaintenanceNLPProcessor(NLPConfig())
    return nlp_processor


def get_ml_predictor() -> MaintenancePredictor:
    """Get or create ML predictor instance."""
    global ml_predictor
    if ml_predictor is None:
        from ..config.maintenance_config import MLConfig
        ml_predictor = MaintenancePredictor(MLConfig())
    return ml_predictor


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Robot Maintenance Teaching AI API",
        "version": "1.0.0",
        "endpoints": [
            "/api/teach",
            "/api/diagnose",
            "/api/explain-component",
            "/api/schedule",
            "/api/answer",
            "/api/nlp/analyze",
            "/api/ml/predict"
        ]
    }


@app.post("/api/teach")
async def teach_maintenance(request: TeachingRequest):
    """Teach a maintenance procedure."""
    try:
        tutor = get_tutor()
        result = await tutor.teach_maintenance_procedure(
            robot_type=request.robot_type,
            maintenance_type=request.maintenance_type,
            difficulty=request.difficulty,
            context=request.context
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error in teach_maintenance: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in teach_maintenance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/diagnose")
async def diagnose_problem(request: DiagnosisRequest):
    """Diagnose a robot problem."""
    try:
        tutor = get_tutor()
        result = await tutor.diagnose_problem(
            symptoms=request.symptoms,
            robot_type=request.robot_type,
            context=request.context
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error in diagnose_problem: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in diagnose_problem: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain-component")
async def explain_component(request: ComponentRequest):
    """Explain a robot component."""
    try:
        tutor = get_tutor()
        result = await tutor.explain_component(
            component_name=request.component_name,
            robot_type=request.robot_type,
            difficulty=request.difficulty
        )
        return result
    except Exception as e:
        logger.error(f"Error in explain_component: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedule")
async def generate_schedule(request: ScheduleRequest):
    """Generate a maintenance schedule."""
    try:
        tutor = get_tutor()
        result = await tutor.generate_maintenance_schedule(
            robot_type=request.robot_type,
            usage_hours=request.usage_hours,
            environment=request.environment
        )
        return result
    except Exception as e:
        logger.error(f"Error in generate_schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/answer")
async def answer_question(request: QuestionRequest):
    """Answer a maintenance question."""
    try:
        tutor = get_tutor()
        result = await tutor.answer_question(
            question=request.question,
            robot_type=request.robot_type,
            context=request.context
        )
        return result
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nlp/analyze")
async def analyze_text(request: NLPRequest):
    """Analyze text with NLP."""
    try:
        nlp = get_nlp_processor()
        result = nlp.process_maintenance_query(request.text)
        return result
    except Exception as e:
        logger.error(f"Error in analyze_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/predict")
async def predict_maintenance(request: PredictionRequest):
    """Predict maintenance needs using ML."""
    try:
        predictor = get_ml_predictor()
        result = predictor.predict_maintenance_need(
            robot_type=request.robot_type,
            operating_hours=request.operating_hours,
            error_count=request.error_count,
            temperature=request.temperature,
            vibration_level=request.vibration_level,
            last_maintenance_hours=request.last_maintenance_hours
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error in predict_maintenance: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in predict_maintenance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train", response_model=Dict[str, Any])
async def train_model(request: TrainingRequest = TrainingRequest()):
    """
    Train the ML model with sample data.
    
    Note: In production, this should use real training data.
    This endpoint generates synthetic data for demonstration.
    """
    try:
        import numpy as np
        predictor = get_ml_predictor()
        
        # Generate synthetic training data
        # In production, load from actual data source
        logger.info("Generating synthetic training data...")
        np.random.seed(42)
        n_samples = 1000
        
        features = np.random.rand(n_samples, 6)
        # Normalize features
        features[:, 0] = (features[:, 0] * 100).astype(int)  # robot_type_encoded
        features[:, 1] = features[:, 1] * 10000  # operating_hours
        features[:, 2] = (features[:, 2] * 10).astype(int)  # error_count
        features[:, 3] = features[:, 3] * 80 + 20  # temperature
        features[:, 4] = features[:, 4]  # vibration_level (0-1)
        features[:, 5] = features[:, 5] * 2000  # last_maintenance_hours
        
        # Generate labels based on heuristics
        labels = (
            (features[:, 1] > 8000).astype(int) |  # High operating hours
            (features[:, 2] > 5).astype(int) |  # High error count
            (features[:, 4] > 0.8).astype(int) |  # High vibration
            (features[:, 5] > 1500).astype(int)  # Long time since maintenance
        )
        
        # Train the model
        logger.info("Training model...")
        metrics = predictor.train(features, labels, test_size=request.test_size)
        
        # Save model if requested
        if request.save_model:
            predictor.save_model()
            logger.info("Model saved successfully")
        
        return {
            "success": True,
            "message": "Model trained successfully",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in train_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/history")
async def get_conversation_history(
    limit: Optional[int] = Query(default=50, ge=1, le=1000, description="Número máximo de conversaciones a retornar")
):
    """Get conversation history."""
    try:
        tutor = get_tutor()
        history = tutor.get_conversation_history(limit=limit)
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test tutor connection
        tutor = get_tutor()
        tutor_ready = tutor is not None
        
        # Test NLP processor
        nlp = get_nlp_processor()
        nlp_ready = nlp is not None and nlp.nlp is not None
        
        # Test ML predictor
        ml = get_ml_predictor()
        ml_ready = ml is not None
        
        status = "healthy" if (tutor_ready and nlp_ready and ml_ready) else "degraded"
        
        return {
            "status": status,
            "tutor_ready": tutor_ready,
            "nlp_ready": nlp_ready,
            "ml_ready": ml_ready,
            "ml_trained": ml.is_trained if ml else False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/metrics")
async def get_metrics():
    """Get API metrics."""
    try:
        # Get metrics from middleware
        # Note: In a real implementation, you'd access the middleware instance
        # For now, return basic metrics
        return {
            "message": "Metrics endpoint - implementation in progress",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nlp/intent")
async def analyze_intent(request: NLPRequest):
    """Analyze user intent from text."""
    try:
        nlp = get_nlp_processor()
        intent = nlp.extract_intent(request.text)
        urgency = nlp.extract_urgency(request.text)
        components = nlp.extract_component_mentions(request.text)
        
        return {
            "intent": intent,
            "urgency": urgency,
            "components": components,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in analyze_intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_maintenance_app() -> FastAPI:
    """Create and return the FastAPI app."""
    return app






