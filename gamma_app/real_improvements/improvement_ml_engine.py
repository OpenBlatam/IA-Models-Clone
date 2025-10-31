"""
Gamma App - Real Improvement ML Engine
Machine Learning engine for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle

logger = logging.getLogger(__name__)

class MLModelType(Enum):
    """ML Model types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"

class MLModelStatus(Enum):
    """ML Model status"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    ERROR = "error"
    RETRAINING = "retraining"

@dataclass
class MLModel:
    """ML Model"""
    model_id: str
    name: str
    type: MLModelType
    description: str
    features: List[str]
    target: str
    model_data: Any = None
    status: MLModelStatus = MLModelStatus.TRAINING
    accuracy: float = 0.0
    created_at: datetime = None
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    training_data_size: int = 0
    prediction_count: int = 0
    last_prediction: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class MLPrediction:
    """ML Prediction"""
    prediction_id: str
    model_id: str
    input_data: Dict[str, Any]
    prediction: Any
    confidence: float = 0.0
    created_at: datetime = None
    processing_time: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementMLEngine:
    """
    Machine Learning engine for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize ML engine"""
        self.project_root = Path(project_root)
        self.models: Dict[str, MLModel] = {}
        self.predictions: Dict[str, MLPrediction] = {}
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.model_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with default models
        self._initialize_default_models()
        
        logger.info(f"Real Improvement ML Engine initialized for {self.project_root}")
    
    def _initialize_default_models(self):
        """Initialize default ML models"""
        # Improvement Priority Predictor
        priority_model = MLModel(
            model_id="improvement_priority_predictor",
            name="Improvement Priority Predictor",
            type=MLModelType.CLASSIFICATION,
            description="Predicts the priority level of improvements based on various factors",
            features=["effort_hours", "impact_score", "category", "complexity", "urgency"],
            target="priority"
        )
        self.models[priority_model.model_id] = priority_model
        
        # Improvement Success Predictor
        success_model = MLModel(
            model_id="improvement_success_predictor",
            name="Improvement Success Predictor",
            type=MLModelType.CLASSIFICATION,
            description="Predicts the success probability of improvements",
            features=["effort_hours", "impact_score", "team_size", "experience_level", "complexity"],
            target="success"
        )
        self.models[success_model.model_id] = success_model
        
        # Improvement Duration Predictor
        duration_model = MLModel(
            model_id="improvement_duration_predictor",
            name="Improvement Duration Predictor",
            type=MLModelType.REGRESSION,
            description="Predicts the duration of improvement implementation",
            features=["effort_hours", "complexity", "team_size", "experience_level", "dependencies"],
            target="duration"
        )
        self.models[duration_model.model_id] = duration_model
        
        # Code Quality Predictor
        quality_model = MLModel(
            model_id="code_quality_predictor",
            name="Code Quality Predictor",
            type=MLModelType.REGRESSION,
            description="Predicts code quality metrics based on various factors",
            features=["lines_of_code", "complexity", "test_coverage", "documentation", "refactoring_frequency"],
            target="quality_score"
        )
        self.models[quality_model.model_id] = quality_model
        
        # Bug Risk Predictor
        bug_risk_model = MLModel(
            model_id="bug_risk_predictor",
            name="Bug Risk Predictor",
            type=MLModelType.CLASSIFICATION,
            description="Predicts the risk of introducing bugs in code changes",
            features=["change_size", "complexity", "test_coverage", "review_quality", "experience_level"],
            target="bug_risk"
        )
        self.models[bug_risk_model.model_id] = bug_risk_model
    
    def create_ml_model(self, name: str, type: MLModelType, description: str,
                       features: List[str], target: str) -> str:
        """Create ML model"""
        try:
            model_id = f"model_{int(time.time() * 1000)}"
            
            model = MLModel(
                model_id=model_id,
                name=name,
                type=type,
                description=description,
                features=features,
                target=target
            )
            
            self.models[model_id] = model
            self.model_logs[model_id] = []
            
            logger.info(f"ML Model created: {name}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to create ML model: {e}")
            raise
    
    async def train_model(self, model_id: str, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train ML model"""
        try:
            if model_id not in self.models:
                return {"success": False, "error": "Model not found"}
            
            model = self.models[model_id]
            model.status = MLModelStatus.TRAINING
            
            self._log_model(model_id, "training_started", f"Training started for model {model.name}")
            
            # Prepare training data
            X = training_data[model.features]
            y = training_data[model.target]
            
            # Handle categorical variables
            X_encoded = self._encode_categorical_features(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model based on type
            if model.type == MLModelType.CLASSIFICATION:
                ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
                ml_model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = ml_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                model.accuracy = accuracy
                
            elif model.type == MLModelType.REGRESSION:
                ml_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                ml_model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = ml_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                model.accuracy = r2
                
            else:
                return {"success": False, "error": f"Unsupported model type: {model.type}"}
            
            # Save model
            model.model_data = {
                "model": ml_model,
                "scaler": scaler,
                "feature_names": model.features,
                "target_name": model.target
            }
            
            # Update model status
            model.status = MLModelStatus.TRAINED
            model.trained_at = datetime.utcnow()
            model.training_data_size = len(training_data)
            
            # Store training data
            self.training_data[model_id] = training_data
            
            self._log_model(model_id, "training_completed", f"Training completed with accuracy: {model.accuracy:.4f}")
            
            return {
                "success": True,
                "accuracy": model.accuracy,
                "training_data_size": model.training_data_size,
                "features": model.features,
                "target": model.target
            }
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            model.status = MLModelStatus.ERROR
            self._log_model(model_id, "training_failed", f"Training failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        try:
            X_encoded = X.copy()
            
            for column in X_encoded.columns:
                if X_encoded[column].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
            
            return X_encoded
            
        except Exception as e:
            logger.error(f"Failed to encode categorical features: {e}")
            return X
    
    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction"""
        try:
            if model_id not in self.models:
                return {"success": False, "error": "Model not found"}
            
            model = self.models[model_id]
            
            if model.status != MLModelStatus.TRAINED:
                return {"success": False, "error": "Model not trained"}
            
            if not model.model_data:
                return {"success": False, "error": "Model data not available"}
            
            prediction_id = f"pred_{int(time.time() * 1000)}"
            start_time = time.time()
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_encoded = self._encode_categorical_features(input_df)
            
            # Scale input
            scaler = model.model_data["scaler"]
            input_scaled = scaler.transform(input_encoded)
            
            # Make prediction
            ml_model = model.model_data["model"]
            prediction = ml_model.predict(input_scaled)[0]
            
            # Get confidence/probability if available
            confidence = 0.0
            if hasattr(ml_model, 'predict_proba'):
                probabilities = ml_model.predict_proba(input_scaled)[0]
                confidence = max(probabilities)
            
            processing_time = time.time() - start_time
            
            # Create prediction record
            prediction_record = MLPrediction(
                prediction_id=prediction_id,
                model_id=model_id,
                input_data=input_data,
                prediction=prediction,
                confidence=confidence,
                processing_time=processing_time
            )
            
            self.predictions[prediction_id] = prediction_record
            
            # Update model stats
            model.prediction_count += 1
            model.last_prediction = datetime.utcnow()
            
            self._log_model(model_id, "prediction_made", f"Prediction {prediction_id} made: {prediction}")
            
            return {
                "success": True,
                "prediction_id": prediction_id,
                "prediction": prediction,
                "confidence": confidence,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            return {"success": False, "error": str(e)}
    
    async def batch_predict(self, model_id: str, input_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            if model_id not in self.models:
                return {"success": False, "error": "Model not found"}
            
            model = self.models[model_id]
            
            if model.status != MLModelStatus.TRAINED:
                return {"success": False, "error": "Model not trained"}
            
            start_time = time.time()
            predictions = []
            prediction_ids = []
            
            # Prepare batch input data
            input_df = pd.DataFrame(input_data_list)
            input_encoded = self._encode_categorical_features(input_df)
            
            # Scale input
            scaler = model.model_data["scaler"]
            input_scaled = scaler.transform(input_encoded)
            
            # Make batch predictions
            ml_model = model.model_data["model"]
            batch_predictions = ml_model.predict(input_scaled)
            
            # Get confidence/probability if available
            confidences = []
            if hasattr(ml_model, 'predict_proba'):
                probabilities = ml_model.predict_proba(input_scaled)
                confidences = [max(probs) for probs in probabilities]
            else:
                confidences = [0.0] * len(batch_predictions)
            
            # Create prediction records
            for i, (input_data, prediction, confidence) in enumerate(zip(input_data_list, batch_predictions, confidences)):
                prediction_id = f"batch_pred_{int(time.time() * 1000)}_{i}"
                
                prediction_record = MLPrediction(
                    prediction_id=prediction_id,
                    model_id=model_id,
                    input_data=input_data,
                    prediction=prediction,
                    confidence=confidence,
                    processing_time=0.0
                )
                
                self.predictions[prediction_id] = prediction_record
                predictions.append(prediction)
                prediction_ids.append(prediction_id)
            
            processing_time = time.time() - start_time
            
            # Update model stats
            model.prediction_count += len(predictions)
            model.last_prediction = datetime.utcnow()
            
            self._log_model(model_id, "batch_prediction_made", f"Batch prediction made for {len(predictions)} items")
            
            return {
                "success": True,
                "predictions": predictions,
                "prediction_ids": prediction_ids,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to make batch prediction: {e}")
            return {"success": False, "error": str(e)}
    
    def save_model(self, model_id: str, file_path: str) -> bool:
        """Save model to file"""
        try:
            if model_id not in self.models:
                return False
            
            model = self.models[model_id]
            
            if not model.model_data:
                return False
            
            # Save model data
            with open(file_path, 'wb') as f:
                pickle.dump(model.model_data, f)
            
            self._log_model(model_id, "model_saved", f"Model saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_id: str, file_path: str) -> bool:
        """Load model from file"""
        try:
            if model_id not in self.models:
                return False
            
            model = self.models[model_id]
            
            # Load model data
            with open(file_path, 'rb') as f:
                model.model_data = pickle.load(f)
            
            model.status = MLModelStatus.TRAINED
            model.deployed_at = datetime.utcnow()
            
            self._log_model(model_id, "model_loaded", f"Model loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _log_model(self, model_id: str, event: str, message: str):
        """Log model event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if model_id not in self.model_logs:
            self.model_logs[model_id] = []
        
        self.model_logs[model_id].append(log_entry)
        
        logger.info(f"ML Model {model_id}: {event} - {message}")
    
    def get_model_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model status"""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        return {
            "model_id": model_id,
            "name": model.name,
            "type": model.type.value,
            "status": model.status.value,
            "accuracy": model.accuracy,
            "created_at": model.created_at.isoformat(),
            "trained_at": model.trained_at.isoformat() if model.trained_at else None,
            "deployed_at": model.deployed_at.isoformat() if model.deployed_at else None,
            "training_data_size": model.training_data_size,
            "prediction_count": model.prediction_count,
            "last_prediction": model.last_prediction.isoformat() if model.last_prediction else None,
            "features": model.features,
            "target": model.target
        }
    
    def get_prediction_status(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get prediction status"""
        if prediction_id not in self.predictions:
            return None
        
        prediction = self.predictions[prediction_id]
        
        return {
            "prediction_id": prediction_id,
            "model_id": prediction.model_id,
            "prediction": prediction.prediction,
            "confidence": prediction.confidence,
            "created_at": prediction.created_at.isoformat(),
            "processing_time": prediction.processing_time,
            "input_data": prediction.input_data
        }
    
    def get_ml_engine_summary(self) -> Dict[str, Any]:
        """Get ML engine summary"""
        total_models = len(self.models)
        trained_models = len([m for m in self.models.values() if m.status == MLModelStatus.TRAINED])
        deployed_models = len([m for m in self.models.values() if m.status == MLModelStatus.DEPLOYED])
        error_models = len([m for m in self.models.values() if m.status == MLModelStatus.ERROR])
        
        total_predictions = len(self.predictions)
        
        return {
            "total_models": total_models,
            "trained_models": trained_models,
            "deployed_models": deployed_models,
            "error_models": error_models,
            "total_predictions": total_predictions,
            "model_types": list(set(m.type.value for m in self.models.values())),
            "average_accuracy": np.mean([m.accuracy for m in self.models.values() if m.accuracy > 0])
        }
    
    def get_model_logs(self, model_id: str) -> List[Dict[str, Any]]:
        """Get model logs"""
        return self.model_logs.get(model_id, [])
    
    def retrain_model(self, model_id: str, new_training_data: pd.DataFrame) -> Dict[str, Any]:
        """Retrain model with new data"""
        try:
            if model_id not in self.models:
                return {"success": False, "error": "Model not found"}
            
            model = self.models[model_id]
            model.status = MLModelStatus.RETRAINING
            
            self._log_model(model_id, "retraining_started", f"Retraining started for model {model.name}")
            
            # Combine old and new training data
            if model_id in self.training_data:
                combined_data = pd.concat([self.training_data[model_id], new_training_data], ignore_index=True)
            else:
                combined_data = new_training_data
            
            # Train with combined data
            result = await self.train_model(model_id, combined_data)
            
            if result["success"]:
                self._log_model(model_id, "retraining_completed", f"Retraining completed for model {model.name}")
            else:
                self._log_model(model_id, "retraining_failed", f"Retraining failed for model {model.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            return {"success": False, "error": str(e)}

# Global ML engine instance
improvement_ml_engine = None

def get_improvement_ml_engine() -> RealImprovementMLEngine:
    """Get improvement ML engine instance"""
    global improvement_ml_engine
    if not improvement_ml_engine:
        improvement_ml_engine = RealImprovementMLEngine()
    return improvement_ml_engine













