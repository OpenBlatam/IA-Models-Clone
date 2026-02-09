"""
ML Predictor for predictive maintenance and anomaly detection.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from ..utils.file_helpers import get_iso_timestamp

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ..config.maintenance_config import MLConfig

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    ML predictor for predictive maintenance using anomaly detection and pattern recognition.
    """
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.anomaly_detector = None
        self.scaler = None
        
        if self.config.enable_anomaly_detection and SKLEARN_AVAILABLE:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
    
    async def analyze_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sensor data for anomalies and maintenance predictions.
        
        Args:
            sensor_data: Dictionary with sensor readings
        
        Returns:
            Analysis results with anomalies and predictions
        """
        result = {
            "timestamp": get_iso_timestamp(),
            "anomalies": [],
            "predictions": {},
            "health_score": 0.0,
            "recommendations": []
        }
        
        if not sensor_data:
            return result
        
        try:
            features = self._extract_features(sensor_data)
            
            if self.anomaly_detector and features:
                features_array = np.array([features])
                features_scaled = self.scaler.fit_transform(features_array)
                
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                
                result["health_score"] = float(anomaly_score)
                
                if is_anomaly:
                    result["anomalies"] = self._identify_anomalies(sensor_data)
                    result["recommendations"].append("Se detectaron anomalías. Se recomienda inspección.")
            
            result["predictions"] = self._predict_maintenance_needs(sensor_data)
            
        except Exception as e:
            logger.error(f"Error in sensor data analysis: {e}")
            result["error"] = str(e)
        
        return result
    
    def _extract_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from sensor data."""
        features = []
        
        numeric_keys = [
            "temperature", "pressure", "vibration", "current", "voltage",
            "rpm", "torque", "humidity", "battery_level"
        ]
        
        for key in numeric_keys:
            if key in sensor_data:
                value = sensor_data[key]
                if isinstance(value, (int, float)):
                    features.append(float(value))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return features
    
    def _identify_anomalies(self, sensor_data: Dict[str, Any]) -> List[str]:
        """Identify specific anomalies in sensor data."""
        anomalies = []
        
        thresholds = {
            "temperature": (0, 100),
            "pressure": (0, 1000),
            "vibration": (0, 10),
            "current": (0, 50),
            "battery_level": (20, 100)
        }
        
        for key, (min_val, max_val) in thresholds.items():
            if key in sensor_data:
                value = sensor_data[key]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        anomalies.append(f"{key} fuera de rango: {value}")
        
        return anomalies
    
    def _predict_maintenance_needs(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict maintenance needs based on sensor data."""
        predictions = {
            "next_maintenance_days": 30,
            "priority": "normal",
            "components_at_risk": []
        }
        
        if "battery_level" in sensor_data:
            battery = sensor_data["battery_level"]
            if battery < 30:
                predictions["priority"] = "alta"
                predictions["components_at_risk"].append("batería")
                predictions["next_maintenance_days"] = 7
        
        if "vibration" in sensor_data:
            vibration = sensor_data["vibration"]
            if vibration > 7:
                predictions["priority"] = "alta"
                predictions["components_at_risk"].append("mecanismo")
                predictions["next_maintenance_days"] = 3
        
        if "temperature" in sensor_data:
            temp = sensor_data["temperature"]
            if temp > 80:
                predictions["priority"] = "media"
                predictions["components_at_risk"].append("sistema de enfriamiento")
        
        return predictions
    
    def detect_anomaly(self, sensor_data: Any) -> Dict[str, Any]:
        """
        Detect anomalies in sensor data.
        
        Args:
            sensor_data: Sensor data (can be dict or list)
        
        Returns:
            Anomaly detection results
        """
        if isinstance(sensor_data, list):
            sensor_data = {"values": sensor_data}
        
        if not isinstance(sensor_data, dict):
            return {"is_anomaly": False, "score": 0.0, "anomalies": []}
        
        try:
            features = self._extract_features(sensor_data)
            if not features:
                return {"is_anomaly": False, "score": 0.0, "anomalies": []}
            
            if self.anomaly_detector:
                features_array = np.array([features])
                features_scaled = self.scaler.fit_transform(features_array)
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                
                anomalies = self._identify_anomalies(sensor_data) if is_anomaly else []
                
                return {
                    "is_anomaly": bool(is_anomaly),
                    "score": float(anomaly_score),
                    "anomalies": anomalies
                }
            else:
                anomalies = self._identify_anomalies(sensor_data)
                return {
                    "is_anomaly": bool(anomalies),
                    "score": 0.0,
                    "anomalies": anomalies
                }
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"is_anomaly": False, "score": 0.0, "anomalies": [], "error": str(e)}
    
    def predict_maintenance_type(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict maintenance type based on equipment features.
        
        Args:
            features: Equipment features (hours, errors, etc.)
        
        Returns:
            Maintenance type prediction
        """
        hours = features.get("hours_operating", 0)
        days_since = features.get("days_since_maintenance", 0)
        error_count = features.get("error_count", 0)
        
        if error_count > 5 or days_since > 180:
            maintenance_type = "correctivo"
            confidence = 0.85
        elif days_since > 90 or hours > 10000:
            maintenance_type = "preventivo"
            confidence = 0.75
        elif days_since > 60:
            maintenance_type = "predictivo"
            confidence = 0.65
        else:
            maintenance_type = "inspección"
            confidence = 0.50
        
        return {
            "predicted_type": maintenance_type,
            "confidence": confidence,
            "reasoning": f"Basado en {hours}h operación, {days_since} días desde último mantenimiento, {error_count} errores"
        }
    
    def predict_failure_probability(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict failure probability based on equipment features.
        
        Args:
            features: Equipment features
        
        Returns:
            Failure probability prediction
        """
        hours = features.get("hours_operating", 0)
        days_since = features.get("days_since_maintenance", 0)
        error_count = features.get("error_count", 0)
        
        base_probability = 0.1
        
        if days_since > 180:
            base_probability += 0.3
        elif days_since > 120:
            base_probability += 0.2
        elif days_since > 90:
            base_probability += 0.1
        
        if hours > 15000:
            base_probability += 0.25
        elif hours > 10000:
            base_probability += 0.15
        elif hours > 5000:
            base_probability += 0.05
        
        if error_count > 10:
            base_probability += 0.3
        elif error_count > 5:
            base_probability += 0.2
        elif error_count > 0:
            base_probability += 0.1
        
        failure_probability = min(base_probability, 0.95)
        
        if failure_probability > 0.7:
            risk_level = "crítico"
        elif failure_probability > 0.5:
            risk_level = "alto"
        elif failure_probability > 0.3:
            risk_level = "medio"
        else:
            risk_level = "bajo"
        
        return {
            "failure_probability": failure_probability,
            "risk_level": risk_level,
            "recommended_action": "mantenimiento_urgente" if failure_probability > 0.7 else "mantenimiento_programado"
        }
    
    def predict_maintenance_need(
        self,
        sensor_data: Dict[str, float],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Predict maintenance need based on sensor data.
        
        Args:
            sensor_data: Current sensor readings
            historical_data: Optional historical data
        
        Returns:
            Maintenance need prediction
        """
        predictions = self._predict_maintenance_needs(sensor_data)
        
        maintenance_needed = predictions["priority"] in ["alta", "media"]
        
        result = {
            "maintenance_needed": maintenance_needed,
            "probability": 0.7 if maintenance_needed else 0.3,
            "next_maintenance_days": predictions.get("next_maintenance_days", 30),
            "priority": predictions.get("priority", "normal"),
            "components_at_risk": predictions.get("components_at_risk", []),
            "recommended_actions": []
        }
        
        if maintenance_needed:
            if "batería" in result["components_at_risk"]:
                result["recommended_actions"].append("Reemplazar batería")
            if "mecanismo" in result["components_at_risk"]:
                result["recommended_actions"].append("Inspeccionar y lubricar mecanismo")
            if "sistema de enfriamiento" in result["components_at_risk"]:
                result["recommended_actions"].append("Revisar sistema de enfriamiento")
        
        return result
    
    async def train_model(self, training_data: List[Dict[str, Any]]):
        """
        Train the ML model with historical data.
        
        Args:
            training_data: List of historical sensor readings
        """
        if not self.anomaly_detector:
            logger.warning("Anomaly detector not initialized")
            return
        
        try:
            features_list = [self._extract_features(data) for data in training_data]
            if features_list:
                X = np.array(features_list)
                X_scaled = self.scaler.fit_transform(X)
                self.anomaly_detector.fit(X_scaled)
                logger.info("Model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    async def close(self):
        """Cleanup resources."""
        pass
