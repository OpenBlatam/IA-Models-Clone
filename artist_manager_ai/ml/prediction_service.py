"""
Prediction Service (PyTorch-based) - Optimized
===============================================

Servicio de predicciones usando PyTorch models con optimizaciones de velocidad.
"""

import torch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from .models import EventDurationPredictor, RoutineCompletionPredictor, OptimalTimePredictor
from .data.preprocessing import FeatureExtractor
from .evaluation import ModelEvaluator
from .optimization import SpeedOptimizer, FastInference, MemoryOptimizer

logger = logging.getLogger(__name__)


@dataclass
class EventPrediction:
    """Predicción de evento."""
    event_type: str
    predicted_duration_hours: float
    confidence: float
    factors: Dict[str, Any]
    recommendation: str


@dataclass
class RoutinePrediction:
    """Predicción de rutina."""
    routine_id: str
    predicted_completion_rate: float
    confidence: float
    optimal_time: Optional[str] = None
    recommendation: str = ""


class PredictionService:
    """
    Servicio de predicciones usando PyTorch models.
    
    Optimizations:
    - Model compilation
    - Fast inference
    - Batch processing
    - Memory optimization
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        optimize_for_speed: bool = True
    ):
        """
        Inicializar servicio de predicciones.
        
        Args:
            model_dir: Directorio con modelos entrenados
            device: Device ("cpu" or "cuda")
            optimize_for_speed: Enable speed optimizations
        """
        self._logger = logger
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._logger.info(f"Using device: {self.device}")
        
        # Model directory
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Initialize models
        self.event_model: Optional[EventDurationPredictor] = None
        self.routine_model: Optional[RoutineCompletionPredictor] = None
        self.time_model: Optional[OptimalTimePredictor] = None
        
        # Fast inference wrappers
        self.event_inference: Optional[FastInference] = None
        self.routine_inference: Optional[FastInference] = None
        self.time_inference: Optional[FastInference] = None
        
        # Load models if available
        self._load_models(optimize_for_speed)
    
    def _load_models(self, optimize: bool = True):
        """Load trained models if available."""
        # Event duration model
        event_model_path = self.model_dir / "event_duration_model.pt"
        if event_model_path.exists():
            try:
                self.event_model = EventDurationPredictor()
                checkpoint = torch.load(event_model_path, map_location=self.device)
                self.event_model.load_state_dict(checkpoint["model_state_dict"])
                self.event_model.to(self.device)
                self.event_model.eval()
                
                if optimize:
                    self.event_inference = FastInference(
                        self.event_model,
                        self.device,
                        batch_size=32,
                        use_compile=True
                    )
                
                self._logger.info("Loaded event duration model")
            except Exception as e:
                self._logger.warning(f"Could not load event model: {str(e)}")
        
        # Routine completion model
        routine_model_path = self.model_dir / "routine_completion_model.pt"
        if routine_model_path.exists():
            try:
                self.routine_model = RoutineCompletionPredictor()
                checkpoint = torch.load(routine_model_path, map_location=self.device)
                self.routine_model.load_state_dict(checkpoint["model_state_dict"])
                self.routine_model.to(self.device)
                self.routine_model.eval()
                
                if optimize:
                    self.routine_inference = FastInference(
                        self.routine_model,
                        self.device,
                        batch_size=32,
                        use_compile=True
                    )
                
                self._logger.info("Loaded routine completion model")
            except Exception as e:
                self._logger.warning(f"Could not load routine model: {str(e)}")
        
        # Optimal time model
        time_model_path = self.model_dir / "optimal_time_model.pt"
        if time_model_path.exists():
            try:
                self.time_model = OptimalTimePredictor()
                checkpoint = torch.load(time_model_path, map_location=self.device)
                self.time_model.load_state_dict(checkpoint["model_state_dict"])
                self.time_model.to(self.device)
                self.time_model.eval()
                
                if optimize:
                    self.time_inference = FastInference(
                        self.time_model,
                        self.device,
                        batch_size=32,
                        use_compile=True
                    )
                
                self._logger.info("Loaded optimal time model")
            except Exception as e:
                self._logger.warning(f"Could not load time model: {str(e)}")
    
    def predict_event_duration(
        self,
        event_type: str,
        historical_events: List[Dict[str, Any]],
        event_data: Optional[Dict[str, Any]] = None
    ) -> EventPrediction:
        """
        Predecir duración de evento usando PyTorch model (optimizado).
        
        Args:
            event_type: Tipo de evento
            historical_events: Eventos históricos (para fallback)
            event_data: Datos del evento actual
        
        Returns:
            Predicción
        """
        # Use optimized inference if available
        if self.event_inference is not None and event_data:
            try:
                # Extract features
                features = self.feature_extractor.extract_event_features(event_data)
                features = features.unsqueeze(0).to(self.device)
                
                # Fast batch prediction
                with torch.no_grad():
                    prediction = self.event_inference.predict(features)
                    duration = prediction.item()
                    confidence = 0.85
                
                return EventPrediction(
                    event_type=event_type,
                    predicted_duration_hours=round(max(0.1, duration), 2),
                    confidence=confidence,
                    factors={"method": "pytorch_model_optimized"},
                    recommendation="Predicción basada en modelo PyTorch optimizado."
                )
            except Exception as e:
                self._logger.warning(f"Optimized prediction failed: {str(e)}, using fallback")
        
        # Fallback to historical average
        return self._predict_event_duration_fallback(event_type, historical_events)
    
    def _predict_event_duration_fallback(
        self,
        event_type: str,
        historical_events: List[Dict[str, Any]]
    ) -> EventPrediction:
        """Fallback prediction using historical average."""
        if not historical_events:
            default_durations = {
                "concert": 3.0, "interview": 1.0, "photoshoot": 2.0,
                "rehearsal": 2.0, "meeting": 1.0
            }
            duration = default_durations.get(event_type, 2.0)
            return EventPrediction(
                event_type=event_type,
                predicted_duration_hours=duration,
                confidence=0.5,
                factors={"method": "default"},
                recommendation="No hay suficiente historial. Usando duración por defecto."
            )
        
        # Calculate average
        durations = []
        for event in historical_events:
            if "start_time" in event and "end_time" in event:
                try:
                    start = datetime.fromisoformat(event["start_time"])
                    end = datetime.fromisoformat(event["end_time"])
                    duration = (end - start).total_seconds() / 3600
                    durations.append(duration)
                except:
                    continue
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            confidence = min(0.9, 0.5 + (len(durations) / 20) * 0.4)
            
            return EventPrediction(
                event_type=event_type,
                predicted_duration_hours=round(avg_duration, 2),
                confidence=round(confidence, 2),
                factors={"method": "historical_average", "sample_size": len(durations)},
                recommendation=f"Basado en {len(durations)} eventos históricos similares."
            )
        
        return EventPrediction(
            event_type=event_type,
            predicted_duration_hours=2.0,
            confidence=0.5,
            factors={"method": "fallback"},
            recommendation="Datos históricos insuficientes."
        )
    
    def predict_routine_completion(
        self,
        routine_id: str,
        completion_history: List[Dict[str, Any]],
        routine_data: Optional[Dict[str, Any]] = None,
        days: int = 30
    ) -> RoutinePrediction:
        """
        Predecir tasa de completación usando PyTorch model (optimizado).
        
        Args:
            routine_id: ID de la rutina
            completion_history: Historial de completaciones
            routine_data: Datos de la rutina actual
            days: Período a analizar
        
        Returns:
            Predicción
        """
        # Use optimized inference if available
        if self.routine_inference is not None and routine_data:
            try:
                # Extract features
                features = self.feature_extractor.extract_routine_features(routine_data)
                features = features.unsqueeze(0).to(self.device)
                
                # Fast batch prediction
                with torch.no_grad():
                    prediction = self.routine_inference.predict(features)
                    prob = prediction.item()
                    confidence = abs(prob - 0.5) * 2
                
                recommendation = f"Probabilidad de completación: {prob:.1%}"
                if prob < 0.7:
                    recommendation += ". Considera ajustar la hora o prioridad."
                
                return RoutinePrediction(
                    routine_id=routine_id,
                    predicted_completion_rate=round(prob, 2),
                    confidence=round(confidence, 2),
                    recommendation=recommendation
                )
            except Exception as e:
                self._logger.warning(f"Optimized prediction failed: {str(e)}, using fallback")
        
        # Fallback to historical analysis
        return self._predict_routine_completion_fallback(routine_id, completion_history, days)
    
    def _predict_routine_completion_fallback(
        self,
        routine_id: str,
        completion_history: List[Dict[str, Any]],
        days: int
    ) -> RoutinePrediction:
        """Fallback prediction using historical analysis."""
        from datetime import timedelta
        
        if not completion_history:
            return RoutinePrediction(
                routine_id=routine_id,
                predicted_completion_rate=0.5,
                confidence=0.3,
                recommendation="No hay historial de completaciones."
            )
        
        # Filter by period
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_completions = [
            c for c in completion_history
            if datetime.fromisoformat(c.get("completed_at", "")) >= cutoff_date
        ]
        
        if not recent_completions:
            return RoutinePrediction(
                routine_id=routine_id,
                predicted_completion_rate=0.5,
                confidence=0.3,
                recommendation="No hay completaciones recientes."
            )
        
        # Calculate completion rate
        completed = sum(1 for c in recent_completions if c.get("status") == "completed")
        total = len(recent_completions)
        completion_rate = completed / total if total > 0 else 0.0
        
        # Optimal time
        optimal_time = None
        times = []
        for c in recent_completions:
            try:
                completed_at = datetime.fromisoformat(c.get("completed_at", ""))
                times.append(completed_at.hour)
            except:
                continue
        
        if times:
            from collections import Counter
            hour_counts = Counter(times)
            optimal_time = f"{hour_counts.most_common(1)[0][0]:02d}:00"
        
        confidence = min(0.9, 0.5 + (total / 30) * 0.4)
        recommendation = f"Tasa de completación: {completion_rate:.1%} basada en {total} registros."
        
        return RoutinePrediction(
            routine_id=routine_id,
            predicted_completion_rate=round(completion_rate, 2),
            confidence=round(confidence, 2),
            optimal_time=optimal_time,
            recommendation=recommendation
        )
    
    def predict_best_event_time(
        self,
        event_type: str,
        historical_events: List[Dict[str, Any]],
        event_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predecir mejor hora para evento usando PyTorch model (optimizado).
        
        Args:
            event_type: Tipo de evento
            historical_events: Eventos históricos
            event_data: Datos del evento actual
        
        Returns:
            Predicción de hora óptima
        """
        # Use optimized inference if available
        if self.time_inference is not None and event_data:
            try:
                # Extract features
                features = self.feature_extractor.extract_event_features(event_data)
                features = features.unsqueeze(0).to(self.device)
                
                # Fast batch prediction
                with torch.no_grad():
                    result = self.time_model.predict(features, return_probabilities=False)
                    optimal_hour = result["optimal_hour"]
                    confidence = result["confidence"]
                
                return {
                    "optimal_hour": optimal_hour,
                    "confidence": confidence,
                    "recommendation": f"Hora óptima predicha: {optimal_hour}:00 (confianza: {confidence:.1%})"
                }
            except Exception as e:
                self._logger.warning(f"Optimized prediction failed: {str(e)}, using fallback")
        
        # Fallback to historical analysis
        if not historical_events:
            return {
                "optimal_hour": 14,
                "confidence": 0.3,
                "recommendation": "No hay datos históricos suficientes."
            }
        
        # Analyze historical start times
        hours = []
        for event in historical_events:
            try:
                start = datetime.fromisoformat(event.get("start_time", ""))
                hours.append(start.hour)
            except:
                continue
        
        if not hours:
            return {
                "optimal_hour": 14,
                "confidence": 0.3,
                "recommendation": "Datos insuficientes."
            }
        
        from collections import Counter
        hour_counts = Counter(hours)
        optimal_hour = hour_counts.most_common(1)[0][0]
        confidence = min(0.9, len(hours) / 20)
        
        return {
            "optimal_hour": optimal_hour,
            "confidence": round(confidence, 2),
            "recommendation": f"Basado en {len(hours)} eventos históricos, la hora más común es {optimal_hour}:00."
        }
