"""
ML Learning - Sistema de aprendizaje automático mejorado
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MLLearningEngine:
    """Motor de aprendizaje automático"""

    def __init__(self):
        """Inicializar motor de ML"""
        self.training_data: List[Dict[str, Any]] = []
        self.models: Dict[str, Any] = {}
        self.feature_vectors: Dict[str, List[float]] = {}

    def record_training_example(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool
    ):
        """
        Registrar ejemplo de entrenamiento.

        Args:
            input_data: Datos de entrada
            output_data: Datos de salida
            success: Si fue exitoso
        """
        example = {
            "input": input_data,
            "output": output_data,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.training_data.append(example)
        
        # Limitar tamaño
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]
        
        logger.debug(f"Ejemplo de entrenamiento registrado: {success}")

    def extract_features(self, content: str) -> List[float]:
        """
        Extraer características del contenido.

        Args:
            content: Contenido

        Returns:
            Vector de características
        """
        features = []
        
        # Características básicas
        features.append(len(content))  # Longitud
        features.append(len(content.split()))  # Número de palabras
        features.append(len(content.split('\n')))  # Número de líneas
        features.append(len([p for p in content.split('\n\n') if p.strip()]))  # Párrafos
        
        # Características de complejidad
        import re
        features.append(len(re.findall(r'[.!?]', content)))  # Oraciones
        features.append(len(re.findall(r'[A-Z]', content)))  # Mayúsculas
        features.append(len(re.findall(r'\d', content)))  # Números
        
        # Características de formato
        features.append(1 if re.search(r'#+', content) else 0)  # Tiene headers
        features.append(1 if re.search(r'```', content) else 0)  # Tiene código
        features.append(1 if re.search(r'\[.*\]\(.*\)', content) else 0)  # Tiene links
        
        return features

    def learn_from_patterns(self) -> Dict[str, Any]:
        """
        Aprender de patrones en los datos.

        Returns:
            Modelo aprendido
        """
        if len(self.training_data) < 10:
            return {"status": "insufficient_data", "message": "Se necesitan más ejemplos"}
        
        # Análisis de patrones exitosos
        successful = [ex for ex in self.training_data if ex["success"]]
        failed = [ex for ex in self.training_data if not ex["success"]]
        
        # Extraer características comunes de exitosos
        success_features = []
        for ex in successful[:100]:  # Limitar para rendimiento
            if "content" in ex["input"]:
                features = self.extract_features(ex["input"]["content"])
                success_features.append(features)
        
        # Calcular promedios
        if success_features:
            avg_features = [
                sum(f[i] for f in success_features) / len(success_features)
                for i in range(len(success_features[0]))
            ]
        else:
            avg_features = []
        
        model = {
            "success_rate": len(successful) / len(self.training_data),
            "avg_success_features": avg_features,
            "total_examples": len(self.training_data),
            "successful_examples": len(successful),
            "failed_examples": len(failed)
        }
        
        self.models["pattern_model"] = model
        logger.info(f"Modelo aprendido de {len(self.training_data)} ejemplos")
        
        return model

    def predict_success(
        self,
        content: str,
        operation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predecir éxito de una operación.

        Args:
            content: Contenido
            operation: Tipo de operación
            **kwargs: Argumentos adicionales

        Returns:
            Predicción de éxito
        """
        if "pattern_model" not in self.models:
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "reason": "Modelo no entrenado"
            }
        
        model = self.models["pattern_model"]
        features = self.extract_features(content)
        
        # Comparar con características promedio de éxito
        if model.get("avg_success_features"):
            avg_features = model["avg_success_features"]
            similarity = self._calculate_feature_similarity(features, avg_features)
            
            # Basar predicción en similitud
            if similarity > 0.7:
                prediction = "high"
                confidence = similarity
            elif similarity > 0.5:
                prediction = "medium"
                confidence = similarity
            else:
                prediction = "low"
                confidence = 1.0 - similarity
        else:
            prediction = "unknown"
            confidence = 0.5
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "model_success_rate": model.get("success_rate", 0.0)
        }

    def _calculate_feature_similarity(self, features1: List[float], features2: List[float]) -> float:
        """
        Calcular similitud entre vectores de características.

        Args:
            features1: Vector 1
            features2: Vector 2

        Returns:
            Similitud (0-1)
        """
        if len(features1) != len(features2) or not features1:
            return 0.0
        
        # Normalizar características
        max_val = max(max(features1), max(features2), 1)
        norm1 = [f / max_val for f in features1]
        norm2 = [f / max_val for f in features2]
        
        # Calcular similitud coseno simplificada
        dot_product = sum(a * b for a, b in zip(norm1, norm2))
        magnitude1 = sum(a * a for a in norm1) ** 0.5
        magnitude2 = sum(b * b for b in norm2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, min(1.0, similarity))

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de aprendizaje.

        Returns:
            Estadísticas
        """
        return {
            "total_examples": len(self.training_data),
            "models_trained": len(self.models),
            "success_rate": (
                len([ex for ex in self.training_data if ex["success"]]) / len(self.training_data)
                if self.training_data else 0.0
            )
        }






