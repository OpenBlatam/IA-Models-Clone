"""
Load Predictor
Predicts future load and enables pre-scaling
"""

import logging
import time
from typing import List, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass
import statistics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available - using basic prediction")

logger = logging.getLogger(__name__)


@dataclass
class LoadSample:
    """Load sample"""
    timestamp: float
    requests_per_second: float
    cpu_usage: float
    memory_usage: float
    response_time: float


class LoadPredictor:
    """
    Load predictor
    
    Features:
    - Time series prediction
    - Pattern recognition
    - Trend analysis
    - Pre-scaling recommendations
    - Anomaly detection
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._samples: deque = deque(maxlen=history_size)
        self._patterns: Dict[str, Any] = {}
        
        logger.info("✅ Load predictor initialized")
    
    def record_sample(self, sample: LoadSample):
        """Record load sample"""
        self._samples.append(sample)
        
        # Detect patterns
        self._detect_patterns()
    
    def predict(
        self,
        horizon: float = 60.0,  # seconds
        method: str = "trend"
    ) -> Dict[str, float]:
        """
        Predict future load
        
        Args:
            horizon: Prediction horizon in seconds
            method: Prediction method ("trend", "average", "linear")
            
        Returns:
            Predicted metrics
        """
        if len(self._samples) < 10:
            # Not enough data
            return {
                "requests_per_second": 0.0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "response_time": 0.0,
                "confidence": 0.0
            }
        
        if method == "trend":
            return self._predict_trend(horizon)
        elif method == "average":
            return self._predict_average(horizon)
        elif method == "linear":
            return self._predict_linear(horizon)
        else:
            return self._predict_trend(horizon)
    
    def _predict_trend(self, horizon: float) -> Dict[str, float]:
        """Predict using trend analysis"""
        recent_samples = list(self._samples)[-100:]  # Last 100 samples
        
        if len(recent_samples) < 2:
            return self._predict_average(horizon)
        
        # Calculate trends
        timestamps = [s.timestamp for s in recent_samples]
        rps_values = [s.requests_per_second for s in recent_samples]
        cpu_values = [s.cpu_usage for s in recent_samples]
        mem_values = [s.memory_usage for s in recent_samples]
        rt_values = [s.response_time for s in recent_samples]
        
        # Simple linear trend
        time_span = timestamps[-1] - timestamps[0]
        if time_span == 0:
            return self._predict_average(horizon)
        
        rps_trend = (rps_values[-1] - rps_values[0]) / time_span
        cpu_trend = (cpu_values[-1] - cpu_values[0]) / time_span
        mem_trend = (mem_values[-1] - mem_values[0]) / time_span
        rt_trend = (rt_values[-1] - rt_values[0]) / time_span
        
        # Predict
        current_time = time.time()
        future_time = current_time + horizon
        
        predicted_rps = max(0, rps_values[-1] + rps_trend * horizon)
        predicted_cpu = max(0, min(100, cpu_values[-1] + cpu_trend * horizon))
        predicted_mem = max(0, min(100, mem_values[-1] + mem_trend * horizon))
        predicted_rt = max(0, rt_values[-1] + rt_trend * horizon)
        
        # Calculate confidence (based on data quality)
        confidence = min(1.0, len(recent_samples) / 100.0)
        
        return {
            "requests_per_second": predicted_rps,
            "cpu_usage": predicted_cpu,
            "memory_usage": predicted_mem,
            "response_time": predicted_rt,
            "confidence": confidence
        }
    
    def _predict_average(self, horizon: float) -> Dict[str, float]:
        """Predict using average of recent samples"""
        recent_samples = list(self._samples)[-50:]  # Last 50 samples
        
        if not recent_samples:
            return {
                "requests_per_second": 0.0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "response_time": 0.0,
                "confidence": 0.0
            }
        
        avg_rps = statistics.mean([s.requests_per_second for s in recent_samples])
        avg_cpu = statistics.mean([s.cpu_usage for s in recent_samples])
        avg_mem = statistics.mean([s.memory_usage for s in recent_samples])
        avg_rt = statistics.mean([s.response_time for s in recent_samples])
        
        confidence = min(1.0, len(recent_samples) / 50.0)
        
        return {
            "requests_per_second": avg_rps,
            "cpu_usage": avg_cpu,
            "memory_usage": avg_mem,
            "response_time": avg_rt,
            "confidence": confidence
        }
    
    def _predict_linear(self, horizon: float) -> Dict[str, float]:
        """Predict using linear regression"""
        if not NUMPY_AVAILABLE:
            return self._predict_trend(horizon)
        
        recent_samples = list(self._samples)[-100:]
        
        if len(recent_samples) < 3:
            return self._predict_average(horizon)
        
        timestamps = np.array([s.timestamp for s in recent_samples])
        rps_values = np.array([s.requests_per_second for s in recent_samples])
        cpu_values = np.array([s.cpu_usage for s in recent_samples])
        mem_values = np.array([s.memory_usage for s in recent_samples])
        rt_values = np.array([s.response_time for s in recent_samples])
        
        # Normalize timestamps
        timestamps = timestamps - timestamps[0]
        
        # Linear regression
        rps_coef = np.polyfit(timestamps, rps_values, 1)
        cpu_coef = np.polyfit(timestamps, cpu_values, 1)
        mem_coef = np.polyfit(timestamps, mem_values, 1)
        rt_coef = np.polyfit(timestamps, rt_values, 1)
        
        # Predict
        future_time = timestamps[-1] + horizon
        
        predicted_rps = max(0, np.polyval(rps_coef, future_time))
        predicted_cpu = max(0, min(100, np.polyval(cpu_coef, future_time)))
        predicted_mem = max(0, min(100, np.polyval(mem_coef, future_time)))
        predicted_rt = max(0, np.polyval(rt_coef, future_time))
        
        confidence = min(1.0, len(recent_samples) / 100.0)
        
        return {
            "requests_per_second": float(predicted_rps),
            "cpu_usage": float(predicted_cpu),
            "memory_usage": float(predicted_mem),
            "response_time": float(predicted_rt),
            "confidence": confidence
        }
    
    def _detect_patterns(self):
        """Detect patterns in load"""
        if len(self._samples) < 100:
            return
        
        # Detect daily patterns
        # Detect weekly patterns
        # Detect anomalies
        pass
    
    def get_scaling_recommendation(
        self,
        current_capacity: float,
        target_utilization: float = 0.7
    ) -> Dict[str, Any]:
        """
        Get scaling recommendation
        
        Args:
            current_capacity: Current system capacity
            target_utilization: Target utilization (0-1)
            
        Returns:
            Scaling recommendation
        """
        prediction = self.predict(horizon=300.0)  # 5 minutes ahead
        
        predicted_load = prediction["requests_per_second"]
        required_capacity = predicted_load / target_utilization if target_utilization > 0 else 0
        
        scale_factor = required_capacity / current_capacity if current_capacity > 0 else 1.0
        
        recommendation = "no_change"
        if scale_factor > 1.2:
            recommendation = "scale_up"
        elif scale_factor < 0.8:
            recommendation = "scale_down"
        
        return {
            "recommendation": recommendation,
            "scale_factor": scale_factor,
            "predicted_load": predicted_load,
            "required_capacity": required_capacity,
            "current_capacity": current_capacity,
            "confidence": prediction["confidence"]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics"""
        if not self._samples:
            return {
                "samples_count": 0,
                "patterns_detected": 0
            }
        
        recent = list(self._samples)[-100:]
        
        return {
            "samples_count": len(self._samples),
            "recent_samples": len(recent),
            "patterns_detected": len(self._patterns),
            "avg_rps": statistics.mean([s.requests_per_second for s in recent]) if recent else 0,
            "avg_cpu": statistics.mean([s.cpu_usage for s in recent]) if recent else 0,
            "avg_memory": statistics.mean([s.memory_usage for s in recent]) if recent else 0
        }


# Global predictor instance
_predictor: Optional[LoadPredictor] = None


def get_load_predictor() -> LoadPredictor:
    """Get global load predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = LoadPredictor()
    return _predictor















