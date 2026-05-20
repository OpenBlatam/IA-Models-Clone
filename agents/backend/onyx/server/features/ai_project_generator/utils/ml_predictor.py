"""
ML Predictor - Predictor de Machine Learning
============================================

Sistema de predicción basado en ML para optimizar generación.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class MLPredictor:
    """Predictor basado en Machine Learning"""

    def __init__(self):
        """Inicializa el predictor ML"""
        self.training_data: List[Dict[str, Any]] = []
        self.prediction_model: Optional[Dict[str, Any]] = None
        self.feature_weights: Dict[str, float] = defaultdict(float)

    def add_training_data(
        self,
        project_info: Dict[str, Any],
        generation_time: float,
        success: bool,
    ):
        """
        Agrega datos de entrenamiento.

        Args:
            project_info: Información del proyecto
            generation_time: Tiempo de generación
            success: Si fue exitoso
        """
        training_sample = {
            "features": {
                "ai_type": project_info.get("ai_type", "unknown"),
                "backend": project_info.get("backend_framework", "fastapi"),
                "frontend": project_info.get("frontend_framework", "react"),
                "features_count": len(project_info.get("features", [])),
                "description_length": len(project_info.get("description", "")),
            },
            "target": {
                "generation_time": generation_time,
                "success": 1 if success else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.training_data.append(training_sample)

        # Limitar tamaño
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]

        logger.debug(f"Dato de entrenamiento agregado. Total: {len(self.training_data)}")

    def train_model(self):
        """Entrena el modelo de predicción"""
        if len(self.training_data) < 10:
            logger.warning("No hay suficientes datos para entrenar")
            return

        # Modelo simple basado en promedios y patrones
        feature_stats = defaultdict(lambda: {"times": [], "successes": []})

        for sample in self.training_data:
            features = sample["features"]
            target = sample["target"]

            key = f"{features['ai_type']}_{features['backend']}_{features['frontend']}"
            feature_stats[key]["times"].append(target["generation_time"])
            feature_stats[key]["successes"].append(target["success"])

        # Calcular promedios
        self.prediction_model = {}
        for key, stats in feature_stats.items():
            if stats["times"]:
                self.prediction_model[key] = {
                    "avg_time": sum(stats["times"]) / len(stats["times"]),
                    "success_rate": sum(stats["successes"]) / len(stats["successes"]),
                    "sample_count": len(stats["times"]),
                }

        logger.info(f"Modelo entrenado con {len(self.training_data)} muestras")

    def predict_generation_time(
        self,
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predice el tiempo de generación.

        Args:
            project_info: Información del proyecto

        Returns:
            Predicción de tiempo
        """
        if not self.prediction_model:
            return {
                "predicted_time": 60.0,
                "confidence": 0.0,
                "message": "Modelo no entrenado",
            }

        features = {
            "ai_type": project_info.get("ai_type", "unknown"),
            "backend": project_info.get("backend_framework", "fastapi"),
            "frontend": project_info.get("frontend_framework", "react"),
        }

        key = f"{features['ai_type']}_{features['backend']}_{features['frontend']}"

        if key in self.prediction_model:
            model_data = self.prediction_model[key]
            return {
                "predicted_time": model_data["avg_time"],
                "confidence": min(model_data["sample_count"] / 10, 1.0),
                "success_rate": model_data["success_rate"],
                "sample_count": model_data["sample_count"],
            }
        else:
            # Predicción basada en promedios generales
            all_times = [
                stats["avg_time"]
                for stats in self.prediction_model.values()
            ]
            if all_times:
                avg_time = sum(all_times) / len(all_times)
                return {
                    "predicted_time": avg_time,
                    "confidence": 0.3,
                    "message": "Predicción basada en promedio general",
                }

        return {
            "predicted_time": 60.0,
            "confidence": 0.0,
            "message": "Sin datos suficientes",
        }

    def predict_success_probability(
        self,
        project_info: Dict[str, Any],
    ) -> float:
        """
        Predice la probabilidad de éxito.

        Args:
            project_info: Información del proyecto

        Returns:
            Probabilidad de éxito (0-1)
        """
        if not self.prediction_model:
            return 0.8  # Default

        features = {
            "ai_type": project_info.get("ai_type", "unknown"),
            "backend": project_info.get("backend_framework", "fastapi"),
            "frontend": project_info.get("frontend_framework", "react"),
        }

        key = f"{features['ai_type']}_{features['backend']}_{features['frontend']}"

        if key in self.prediction_model:
            return self.prediction_model[key]["success_rate"]

        # Promedio general
        all_rates = [
            stats["success_rate"]
            for stats in self.prediction_model.values()
        ]
        if all_rates:
            return sum(all_rates) / len(all_rates)

        return 0.8

    def get_model_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del modelo"""
        return {
            "training_samples": len(self.training_data),
            "model_trained": self.prediction_model is not None,
            "model_size": len(self.prediction_model) if self.prediction_model else 0,
        }


