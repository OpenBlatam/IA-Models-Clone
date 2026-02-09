"""
ML Predictor for maintenance failure prediction.
"""

import logging
import pickle
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from ..config.maintenance_config import MLConfig

logger = logging.getLogger(__name__)


class MaintenancePredictor:
    """
    Machine Learning predictor for maintenance failure prediction.
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ML model based on configuration."""
        if self.config.model_type == "ensemble":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = GradientBoostingClassifier(random_state=42)
        
        logger.info(f"Initialized {self.config.model_type} model")
    
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the maintenance prediction model.
        
        Args:
            features: Feature matrix
            labels: Target labels
            test_size: Test set size
        
        Returns:
            Training metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=self.config.cross_validation_folds
        )
        
        self.is_trained = True
        
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "classification_report": classification_report(y_test, test_pred)
        }
        
        logger.info(f"Model trained. Test accuracy: {test_accuracy:.4f}")
        return metrics
    
    def predict(
        self,
        features: np.ndarray,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Predict maintenance failures.
        
        Args:
            features: Feature matrix
            return_proba: Whether to return probabilities
        
        Returns:
            Predictions or probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform(features)
        
        if return_proba:
            return self.model.predict_proba(features_scaled)
        return self.model.predict(features_scaled)
    
    def predict_maintenance_need(
        self,
        robot_type: str,
        operating_hours: float,
        error_count: int,
        temperature: float,
        vibration_level: float,
        last_maintenance_hours: float
    ) -> Dict[str, Any]:
        """
        Predict if maintenance is needed based on robot parameters.
        
        Args:
            robot_type: Type of robot (encoded as feature)
            operating_hours: Total operating hours
            error_count: Number of errors
            temperature: Operating temperature
            vibration_level: Vibration level
            last_maintenance_hours: Hours since last maintenance
        
        Returns:
            Prediction with confidence
        
        Raises:
            ValueError: If model is not trained and use_pretrained is False
        """
        if not self.is_trained:
            if self.config.use_pretrained:
                # Try to load pretrained model
                try:
                    self.load_model()
                except FileNotFoundError:
                    logger.warning("No trained model available. Using fallback prediction.")
                    return self._fallback_prediction(
                        robot_type, operating_hours, error_count, 
                        temperature, vibration_level, last_maintenance_hours
                    )
            else:
                raise ValueError("Model must be trained before prediction. Call train() first or set use_pretrained=True.")
        
        robot_type_encoded = hash(robot_type) % 100
        
        features = np.array([[
            robot_type_encoded,
            operating_hours,
            error_count,
            temperature,
            vibration_level,
            last_maintenance_hours
        ]])
        
        prediction = self.predict(features)[0]
        probabilities = self.predict(features, return_proba=True)[0]
        
        return {
            "needs_maintenance": bool(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": {
                "no_maintenance": float(probabilities[0]),
                "maintenance_needed": float(probabilities[1])
            },
            "recommendation": self._generate_recommendation(
                prediction, probabilities, features[0]
            )
        }
    
    def _fallback_prediction(
        self,
        robot_type: str,
        operating_hours: float,
        error_count: int,
        temperature: float,
        vibration_level: float,
        last_maintenance_hours: float
    ) -> Dict[str, Any]:
        """Fallback prediction using simple heuristics when model is not available."""
        # Simple heuristic-based prediction
        needs_maintenance = False
        confidence = 0.5
        
        # Check various conditions
        if operating_hours > 10000:
            needs_maintenance = True
            confidence = 0.7
        if error_count > 5:
            needs_maintenance = True
            confidence = 0.8
        if last_maintenance_hours > 1000:
            needs_maintenance = True
            confidence = 0.75
        if vibration_level > 0.9:
            needs_maintenance = True
            confidence = 0.85
        
        return {
            "needs_maintenance": needs_maintenance,
            "confidence": confidence,
            "probabilities": {
                "no_maintenance": 1.0 - confidence,
                "maintenance_needed": confidence
            },
            "recommendation": self._generate_recommendation(
                1 if needs_maintenance else 0,
                np.array([1.0 - confidence, confidence]),
                np.array([robot_type, operating_hours, error_count, temperature, vibration_level, last_maintenance_hours])
            ),
            "note": "Fallback prediction - model not trained"
        }
    
    def _generate_recommendation(
        self,
        prediction: int,
        probabilities: np.ndarray,
        features: np.ndarray
    ) -> str:
        """Generate maintenance recommendation."""
        if prediction:
            return "Se recomienda realizar mantenimiento preventivo inmediato."
        elif probabilities[1] > 0.5:
            return "Se recomienda programar mantenimiento en las próximas semanas."
        else:
            return "El robot está en buen estado. Continuar con mantenimiento regular."
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        path = path or self.config.model_save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, f"{path}/model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Optional[str] = None):
        """Load a saved model."""
        path = path or self.config.model_save_path
        
        try:
            self.model = joblib.load(f"{path}/model.pkl")
            self.scaler = joblib.load(f"{path}/scaler.pkl")
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {path}. Using untrained model.")






