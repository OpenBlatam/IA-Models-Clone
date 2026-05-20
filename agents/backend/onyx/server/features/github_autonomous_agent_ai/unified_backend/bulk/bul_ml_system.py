"""
BUL - Business Universal Language (Advanced ML System)
=====================================================

Advanced Machine Learning system with AI models, predictions, and analytics.
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import redis
from prometheus_client import Counter, Histogram, Gauge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import openai
import anthropic
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bul_ml.db', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Prometheus Metrics
ML_REQUESTS = Counter('bul_ml_requests_total', 'Total ML requests', ['model_type', 'operation'])
ML_PREDICTIONS = Counter('bul_ml_predictions_total', 'ML predictions made', ['model_type', 'accuracy_range'])
ML_TRAINING_TIME = Histogram('bul_ml_training_duration_seconds', 'ML model training duration')
ML_PREDICTION_TIME = Histogram('bul_ml_prediction_duration_seconds', 'ML prediction duration')
ML_MODEL_ACCURACY = Gauge('bul_ml_model_accuracy', 'ML model accuracy', ['model_name'])

class MLModelType(str, Enum):
    """ML model type enumeration."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"

class MLTaskStatus(str, Enum):
    """ML task status enumeration."""
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"

class AIModelProvider(str, Enum):
    """AI model provider enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

# Database Models
class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    model_type = Column(String, nullable=False)
    provider = Column(String, default=AIModelProvider.LOCAL)
    description = Column(Text)
    model_path = Column(String)
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    r2_score = Column(Float, default=0.0)
    mse = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    is_deployed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    training_data_size = Column(Integer, default=0)
    features = Column(Text, default="[]")
    hyperparameters = Column(Text, default="{}")

class MLDataset(Base):
    __tablename__ = "ml_datasets"
    
    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, default=0)
    rows = Column(Integer, default=0)
    columns = Column(Integer, default=0)
    target_column = Column(String)
    feature_columns = Column(Text, default="[]")
    data_types = Column(Text, default="{}")
    is_preprocessed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    
    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("ml_models.id"))
    input_data = Column(Text, nullable=False)
    prediction = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    processing_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("MLModel")

class MLTrainingJob(Base):
    __tablename__ = "ml_training_jobs"
    
    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("ml_models.id"))
    dataset_id = Column(String, ForeignKey("ml_datasets.id"))
    status = Column(String, default=MLTaskStatus.PENDING)
    progress = Column(Float, default=0.0)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    error_message = Column(Text)
    hyperparameters = Column(Text, default="{}")
    metrics = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("MLModel")
    dataset = relationship("MLDataset")

# Create tables
Base.metadata.create_all(bind=engine)

# ML Configuration
ML_CONFIG = {
    "openai_api_key": "your-openai-api-key",
    "anthropic_api_key": "your-anthropic-api-key",
    "google_api_key": "your-google-api-key",
    "huggingface_token": "your-huggingface-token",
    "model_storage_path": "./ml_models",
    "dataset_storage_path": "./ml_datasets",
    "max_training_time": 3600,  # 1 hour
    "max_prediction_time": 30,  # 30 seconds
    "default_test_size": 0.2,
    "default_random_state": 42,
    "supported_file_formats": [".csv", ".json", ".parquet", ".xlsx"],
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "auto_scaling": True,
    "gpu_enabled": False
}

class AdvancedMLSystem:
    """Advanced Machine Learning system with comprehensive features."""
    
    def __init__(self):
        self.app = FastAPI(
            title="BUL Advanced ML System",
            description="Advanced Machine Learning system with AI models, predictions, and analytics",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Database session
        self.db = SessionLocal()
        
        # ML components
        self.loaded_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, Any] = {}
        
        # Setup components
        self.setup_middleware()
        self.setup_routes()
        self.setup_default_data()
        self.setup_ai_providers()
        
        logger.info("Advanced ML System initialized")
    
    def setup_middleware(self):
        """Setup ML middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup ML API routes."""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with ML system information."""
            return {
                "message": "BUL Advanced ML System",
                "version": "1.0.0",
                "status": "operational",
                "features": [
                    "Machine Learning Models",
                    "AI Model Integration",
                    "Data Processing",
                    "Model Training",
                    "Predictions",
                    "Model Deployment",
                    "Performance Monitoring",
                    "Auto-scaling"
                ],
                "model_types": [model_type.value for model_type in MLModelType],
                "ai_providers": [provider.value for provider in AIModelProvider],
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/datasets/upload", tags=["Datasets"])
        async def upload_dataset(file: UploadFile = File(...)):
            """Upload dataset for ML training."""
            try:
                # Validate file
                if not any(file.filename.endswith(ext) for ext in ML_CONFIG["supported_file_formats"]):
                    raise HTTPException(status_code=400, detail="Unsupported file format")
                
                # Check file size
                content = await file.read()
                if len(content) > ML_CONFIG["max_file_size"]:
                    raise HTTPException(status_code=400, detail="File too large")
                
                # Save file
                dataset_id = f"dataset_{int(time.time())}"
                file_path = Path(ML_CONFIG["dataset_storage_path"]) / f"{dataset_id}_{file.filename}"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(content)
                
                # Analyze dataset
                dataset_info = await self.analyze_dataset(file_path)
                
                # Create dataset record
                dataset = MLDataset(
                    id=dataset_id,
                    name=file.filename,
                    description=f"Uploaded dataset: {file.filename}",
                    file_path=str(file_path),
                    file_size=len(content),
                    rows=dataset_info["rows"],
                    columns=dataset_info["columns"],
                    feature_columns=json.dumps(dataset_info["feature_columns"]),
                    data_types=json.dumps(dataset_info["data_types"])
                )
                
                self.db.add(dataset)
                self.db.commit()
                
                return {
                    "message": "Dataset uploaded successfully",
                    "dataset_id": dataset_id,
                    "dataset_info": dataset_info
                }
                
            except Exception as e:
                logger.error(f"Error uploading dataset: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/datasets", tags=["Datasets"])
        async def get_datasets():
            """Get all datasets."""
            try:
                datasets = self.db.query(MLDataset).all()
                
                return {
                    "datasets": [
                        {
                            "id": dataset.id,
                            "name": dataset.name,
                            "description": dataset.description,
                            "file_size": dataset.file_size,
                            "rows": dataset.rows,
                            "columns": dataset.columns,
                            "target_column": dataset.target_column,
                            "feature_columns": json.loads(dataset.feature_columns),
                            "is_preprocessed": dataset.is_preprocessed,
                            "created_at": dataset.created_at.isoformat()
                        }
                        for dataset in datasets
                    ],
                    "total": len(datasets)
                }
                
            except Exception as e:
                logger.error(f"Error getting datasets: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/train", tags=["Models"])
        async def train_model(training_request: dict, background_tasks: BackgroundTasks):
            """Train ML model."""
            try:
                # Validate request
                required_fields = ["model_name", "model_type", "dataset_id", "target_column"]
                if not all(field in training_request for field in required_fields):
                    raise HTTPException(status_code=400, detail="Missing required fields")
                
                model_name = training_request["model_name"]
                model_type = training_request["model_type"]
                dataset_id = training_request["dataset_id"]
                target_column = training_request["target_column"]
                hyperparameters = training_request.get("hyperparameters", {})
                
                # Check if model already exists
                existing_model = self.db.query(MLModel).filter(MLModel.name == model_name).first()
                if existing_model:
                    raise HTTPException(status_code=400, detail="Model name already exists")
                
                # Get dataset
                dataset = self.db.query(MLDataset).filter(MLDataset.id == dataset_id).first()
                if not dataset:
                    raise HTTPException(status_code=404, detail="Dataset not found")
                
                # Create model record
                model = MLModel(
                    id=f"model_{int(time.time())}",
                    name=model_name,
                    model_type=model_type,
                    description=training_request.get("description", f"Trained {model_type} model"),
                    hyperparameters=json.dumps(hyperparameters)
                )
                
                self.db.add(model)
                self.db.commit()
                
                # Create training job
                training_job = MLTrainingJob(
                    id=f"job_{int(time.time())}",
                    model_id=model.id,
                    dataset_id=dataset_id,
                    status=MLTaskStatus.PENDING,
                    hyperparameters=json.dumps(hyperparameters)
                )
                
                self.db.add(training_job)
                self.db.commit()
                
                # Start training in background
                background_tasks.add_task(
                    self.execute_training,
                    model.id,
                    dataset_id,
                    target_column,
                    hyperparameters
                )
                
                ML_REQUESTS.labels(model_type=model_type, operation="train").inc()
                
                return {
                    "message": "Model training started",
                    "model_id": model.id,
                    "training_job_id": training_job.id,
                    "status": "training"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error starting training: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", tags=["Models"])
        async def get_models():
            """Get all ML models."""
            try:
                models = self.db.query(MLModel).all()
                
                return {
                    "models": [
                        {
                            "id": model.id,
                            "name": model.name,
                            "model_type": model.model_type,
                            "provider": model.provider,
                            "description": model.description,
                            "accuracy": model.accuracy,
                            "precision": model.precision,
                            "recall": model.recall,
                            "f1_score": model.f1_score,
                            "r2_score": model.r2_score,
                            "mse": model.mse,
                            "is_active": model.is_active,
                            "is_deployed": model.is_deployed,
                            "training_data_size": model.training_data_size,
                            "features": json.loads(model.features),
                            "hyperparameters": json.loads(model.hyperparameters),
                            "created_at": model.created_at.isoformat()
                        }
                        for model in models
                    ],
                    "total": len(models)
                }
                
            except Exception as e:
                logger.error(f"Error getting models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_id}/predict", tags=["Predictions"])
        async def make_prediction(model_id: str, prediction_request: dict):
            """Make prediction using trained model."""
            try:
                start_time = time.time()
                
                # Get model
                model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if not model.is_active:
                    raise HTTPException(status_code=400, detail="Model is not active")
                
                # Get input data
                input_data = prediction_request.get("input_data", [])
                if not input_data:
                    raise HTTPException(status_code=400, detail="Input data required")
                
                # Load model if not loaded
                if model_id not in self.loaded_models:
                    await self.load_model(model_id)
                
                # Make prediction
                prediction = await self.predict_with_model(model_id, input_data)
                
                processing_time = time.time() - start_time
                
                # Log prediction
                prediction_log = MLPrediction(
                    id=f"pred_{int(time.time())}",
                    model_id=model_id,
                    input_data=json.dumps(input_data),
                    prediction=json.dumps(prediction),
                    confidence=prediction.get("confidence", 0.0),
                    processing_time=processing_time
                )
                
                self.db.add(prediction_log)
                self.db.commit()
                
                ML_PREDICTIONS.labels(model_type=model.model_type, accuracy_range="high").inc()
                ML_PREDICTION_TIME.observe(processing_time)
                
                return {
                    "model_id": model_id,
                    "model_name": model.name,
                    "prediction": prediction,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_id}/deploy", tags=["Deployment"])
        async def deploy_model(model_id: str):
            """Deploy model for production use."""
            try:
                model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                if not model.is_active:
                    raise HTTPException(status_code=400, detail="Model is not active")
                
                # Load model
                await self.load_model(model_id)
                
                # Mark as deployed
                model.is_deployed = True
                self.db.commit()
                
                return {
                    "message": "Model deployed successfully",
                    "model_id": model_id,
                    "model_name": model.name,
                    "status": "deployed"
                }
                
            except Exception as e:
                logger.error(f"Error deploying model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}/status", tags=["Models"])
        async def get_model_status(model_id: str):
            """Get model training status."""
            try:
                model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                # Get latest training job
                training_job = self.db.query(MLTrainingJob).filter(
                    MLTrainingJob.model_id == model_id
                ).order_by(MLTrainingJob.created_at.desc()).first()
                
                return {
                    "model_id": model_id,
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "is_active": model.is_active,
                    "is_deployed": model.is_deployed,
                    "accuracy": model.accuracy,
                    "training_status": training_job.status if training_job else "unknown",
                    "training_progress": training_job.progress if training_job else 0.0,
                    "created_at": model.created_at.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting model status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/predictions", tags=["Predictions"])
        async def get_predictions(limit: int = 100):
            """Get prediction history."""
            try:
                predictions = self.db.query(MLPrediction).order_by(
                    MLPrediction.created_at.desc()
                ).limit(limit).all()
                
                return {
                    "predictions": [
                        {
                            "id": pred.id,
                            "model_id": pred.model_id,
                            "input_data": json.loads(pred.input_data),
                            "prediction": json.loads(pred.prediction),
                            "confidence": pred.confidence,
                            "processing_time": pred.processing_time,
                            "created_at": pred.created_at.isoformat()
                        }
                        for pred in predictions
                    ],
                    "total": len(predictions)
                }
                
            except Exception as e:
                logger.error(f"Error getting predictions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard", tags=["Dashboard"])
        async def get_ml_dashboard():
            """Get ML system dashboard."""
            try:
                # Get statistics
                total_models = self.db.query(MLModel).count()
                active_models = self.db.query(MLModel).filter(MLModel.is_active == True).count()
                deployed_models = self.db.query(MLModel).filter(MLModel.is_deployed == True).count()
                total_datasets = self.db.query(MLDataset).count()
                total_predictions = self.db.query(MLPrediction).count()
                
                # Get model type distribution
                model_types = {}
                for model_type in MLModelType:
                    count = self.db.query(MLModel).filter(MLModel.model_type == model_type.value).count()
                    model_types[model_type.value] = count
                
                # Get recent predictions
                recent_predictions = self.db.query(MLPrediction).order_by(
                    MLPrediction.created_at.desc()
                ).limit(10).all()
                
                # Get training jobs status
                training_jobs = self.db.query(MLTrainingJob).all()
                job_status_counts = {}
                for job in training_jobs:
                    status = job.status
                    job_status_counts[status] = job_status_counts.get(status, 0) + 1
                
                return {
                    "summary": {
                        "total_models": total_models,
                        "active_models": active_models,
                        "deployed_models": deployed_models,
                        "total_datasets": total_datasets,
                        "total_predictions": total_predictions
                    },
                    "model_type_distribution": model_types,
                    "training_job_status": job_status_counts,
                    "recent_predictions": [
                        {
                            "id": pred.id,
                            "model_id": pred.model_id,
                            "confidence": pred.confidence,
                            "processing_time": pred.processing_time,
                            "created_at": pred.created_at.isoformat()
                        }
                        for pred in recent_predictions
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def setup_default_data(self):
        """Setup default ML data."""
        try:
            # Create sample models
            sample_models = [
                {
                    "name": "customer_classification",
                    "model_type": MLModelType.CLASSIFICATION,
                    "description": "Customer classification model",
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.88,
                    "f1_score": 0.85
                },
                {
                    "name": "sales_prediction",
                    "model_type": MLModelType.REGRESSION,
                    "description": "Sales prediction model",
                    "accuracy": 0.0,
                    "r2_score": 0.78,
                    "mse": 0.15
                },
                {
                    "name": "customer_clustering",
                    "model_type": MLModelType.CLUSTERING,
                    "description": "Customer segmentation model",
                    "accuracy": 0.0
                }
            ]
            
            for model_data in sample_models:
                model = MLModel(
                    id=f"model_{model_data['name']}",
                    name=model_data["name"],
                    model_type=model_data["model_type"],
                    description=model_data["description"],
                    accuracy=model_data["accuracy"],
                    precision=model_data.get("precision", 0.0),
                    recall=model_data.get("recall", 0.0),
                    f1_score=model_data.get("f1_score", 0.0),
                    r2_score=model_data.get("r2_score", 0.0),
                    mse=model_data.get("mse", 0.0),
                    is_active=True,
                    features=json.dumps([]),
                    hyperparameters=json.dumps({})
                )
                
                self.db.add(model)
            
            self.db.commit()
            logger.info("Default ML data created")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default ML data: {e}")
    
    def setup_ai_providers(self):
        """Setup AI model providers."""
        try:
            # Configure OpenAI
            if ML_CONFIG["openai_api_key"]:
                openai.api_key = ML_CONFIG["openai_api_key"]
            
            # Configure Anthropic
            if ML_CONFIG["anthropic_api_key"]:
                anthropic.api_key = ML_CONFIG["anthropic_api_key"]
            
            # Configure Google
            if ML_CONFIG["google_api_key"]:
                genai.configure(api_key=ML_CONFIG["google_api_key"])
            
            logger.info("AI providers configured")
            
        except Exception as e:
            logger.error(f"Error configuring AI providers: {e}")
    
    async def analyze_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Analyze uploaded dataset."""
        try:
            # Read dataset based on file extension
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.json':
                df = pd.read_json(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Analyze dataset
            dataset_info = {
                "rows": len(df),
                "columns": len(df.columns),
                "feature_columns": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            raise
    
    async def execute_training(self, model_id: str, dataset_id: str, target_column: str, hyperparameters: dict):
        """Execute model training."""
        try:
            # Update training job status
            training_job = self.db.query(MLTrainingJob).filter(
                MLTrainingJob.model_id == model_id
            ).first()
            
            if not training_job:
                return
            
            training_job.status = MLTaskStatus.TRAINING
            training_job.start_time = datetime.utcnow()
            self.db.commit()
            
            # Get dataset
            dataset = self.db.query(MLDataset).filter(MLDataset.id == dataset_id).first()
            if not dataset:
                training_job.status = MLTaskStatus.FAILED
                training_job.error_message = "Dataset not found"
                self.db.commit()
                return
            
            # Load dataset
            df = pd.read_csv(dataset.file_path)
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ML_CONFIG["default_test_size"], 
                random_state=ML_CONFIG["default_random_state"]
            )
            
            # Get model
            model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
            
            # Train model based on type
            if model.model_type == MLModelType.CLASSIFICATION:
                trained_model = LogisticRegression(**hyperparameters)
                trained_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = trained_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Update model metrics
                model.accuracy = accuracy
                model.precision = precision
                model.recall = recall
                model.f1_score = f1
                
            elif model.model_type == MLModelType.REGRESSION:
                trained_model = LinearRegression(**hyperparameters)
                trained_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = trained_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                # Update model metrics
                model.r2_score = r2
                model.mse = mse
                
            elif model.model_type == MLModelType.CLUSTERING:
                n_clusters = hyperparameters.get('n_clusters', 3)
                trained_model = KMeans(n_clusters=n_clusters, **hyperparameters)
                trained_model.fit(X_train)
                
                # Update model metrics
                model.accuracy = 0.0  # Clustering doesn't have traditional accuracy
            
            # Save model
            model_path = Path(ML_CONFIG["model_storage_path"]) / f"{model_id}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(trained_model, model_path)
            
            # Update model record
            model.model_path = str(model_path)
            model.training_data_size = len(X_train)
            model.features = json.dumps(list(X.columns))
            model.is_active = True
            
            # Update training job
            training_job.status = MLTaskStatus.COMPLETED
            training_job.end_time = datetime.utcnow()
            training_job.progress = 100.0
            training_job.metrics = json.dumps({
                "accuracy": model.accuracy,
                "precision": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "r2_score": model.r2_score,
                "mse": model.mse
            })
            
            self.db.commit()
            
            # Update Prometheus metrics
            ML_MODEL_ACCURACY.labels(model_name=model.name).set(model.accuracy)
            
            logger.info(f"Model {model_id} training completed")
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            
            # Update training job status
            training_job = self.db.query(MLTrainingJob).filter(
                MLTrainingJob.model_id == model_id
            ).first()
            
            if training_job:
                training_job.status = MLTaskStatus.FAILED
                training_job.error_message = str(e)
                self.db.commit()
    
    async def load_model(self, model_id: str):
        """Load model into memory."""
        try:
            model = self.db.query(MLModel).filter(MLModel.id == model_id).first()
            if not model or not model.model_path:
                raise ValueError(f"Model {model_id} not found or not trained")
            
            # Load model
            loaded_model = joblib.load(model.model_path)
            self.loaded_models[model_id] = loaded_model
            
            logger.info(f"Model {model_id} loaded into memory")
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    async def predict_with_model(self, model_id: str, input_data: List[Any]) -> Dict[str, Any]:
        """Make prediction with loaded model."""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.loaded_models[model_id]
            
            # Convert input to numpy array
            input_array = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(input_array)
                prediction = model.predict(input_array)[0]
                confidence = float(np.max(prediction_proba))
            else:
                prediction = model.predict(input_array)[0]
                confidence = 1.0
            
            return {
                "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                "confidence": confidence,
                "model_id": model_id
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_id}: {e}")
            raise
    
    def run(self, host: str = "0.0.0.0", port: int = 8007, debug: bool = False):
        """Run the ML system."""
        logger.info(f"Starting Advanced ML System on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Advanced ML System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8007, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run ML system
    system = AdvancedMLSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
