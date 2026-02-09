"""
Inspector de Calidad - Servicio principal de inspección
"""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
from datetime import datetime

from ..core.camera_controller import CameraController
from ..core.object_detector import ObjectDetector, DetectedObject
from ..core.anomaly_detector import AnomalyDetector, Anomaly
from ..core.defect_classifier import DefectClassifier, Defect
from ..config.camera_config import CameraConfig
from ..config.detection_config import DetectionConfig
from ..utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

QUALITY_SCORE_EXCELLENT = 90.0
QUALITY_SCORE_GOOD = 75.0
QUALITY_SCORE_ACCEPTABLE = 60.0
QUALITY_SCORE_POOR = 40.0

PENALTY_CRITICAL_DEFECT = 20.0
PENALTY_SEVERE_DEFECT = 15.0
PENALTY_MODERATE_DEFECT = 8.0
PENALTY_MINOR_DEFECT = 3.0
PENALTY_HIGH_ANOMALY = 10.0
PENALTY_MEDIUM_ANOMALY = 5.0
PENALTY_LOW_ANOMALY = 2.0


class QualityInspector:
    """
    Inspector de calidad que integra cámara, detección de objetos,
    detección de anomalías y clasificación de defectos
    """
    
    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        detection_config: Optional[DetectionConfig] = None
    ):
        """
        Inicializar inspector de calidad
        
        Args:
            camera_config: Configuración de la cámara
            detection_config: Configuración de detección
        """
        self.camera_config = camera_config or CameraConfig()
        self.detection_config = detection_config or DetectionConfig()
        
        # Inicializar componentes
        self.camera = CameraController(self.camera_config)
        self.object_detector = ObjectDetector(self.detection_config)
        self.anomaly_detector = AnomalyDetector(self.detection_config)
        self.defect_classifier = DefectClassifier(self.detection_config)
        self.image_utils = ImageUtils()
        
        logger.info("Quality Inspector initialized")
    
    def initialize_camera(self) -> bool:
        """
        Inicializar cámara
        
        Returns:
            True si se inicializó correctamente
        """
        return self.camera.initialize()
    
    def start_inspection(self) -> bool:
        """
        Iniciar inspección (inicializar cámara y streaming)
        
        Returns:
            True si se inició correctamente
        """
        if not self.camera.is_initialized:
            if not self.initialize_camera():
                return False
        
        return self.camera.start_streaming()
    
    def stop_inspection(self):
        """Detener inspección"""
        self.camera.stop_streaming()
        logger.info("Inspection stopped")
    
    def inspect_frame(self, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Inspeccionar un frame (desde cámara o imagen proporcionada)
        
        Args:
            image: Imagen opcional (si None, se captura de la cámara)
            
        Returns:
            Diccionario con resultados de inspección
        """
        if image is not None and not isinstance(image, (np.ndarray, str)):
            logger.warning(f"Invalid image type: {type(image)}")
            return {
                "success": False,
                "error": f"Invalid image type: expected np.ndarray or str, got {type(image).__name__}"
            }
        
        try:
            # Capturar o usar imagen proporcionada
            if image is None:
                image = self.camera.capture_frame()
                if image is None:
                    return {
                        "success": False,
                        "error": "Failed to capture frame from camera"
                    }
            else:
                image = self.image_utils.load_image(image)
                if image is None:
                    return {
                        "success": False,
                        "error": "Failed to load image"
                    }
            
            timestamp = datetime.now().isoformat()
            
            # Detectar objetos
            try:
                objects = self.object_detector.detect(image)
            except Exception as e:
                logger.error(f"Error in object detection: {e}", exc_info=True)
                objects = []
            
            # Detectar anomalías
            try:
                anomalies = self.anomaly_detector.detect_anomalies(image)
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}", exc_info=True)
                anomalies = []
            
            # Clasificar defectos
            try:
                defects = self.defect_classifier.classify_defects(image, anomalies)
            except Exception as e:
                logger.error(f"Error in defect classification: {e}", exc_info=True)
                defects = []
            
            # Calcular métricas de calidad
            quality_score = self._calculate_quality_score(objects, anomalies, defects)
            
            # Compilar resultados
            result = {
                "success": True,
                "timestamp": timestamp,
                "quality_score": quality_score,
                "objects_detected": len(objects),
                "anomalies_detected": len(anomalies),
                "defects_detected": len(defects),
                "objects": [
                    {
                        "class_name": obj.class_name,
                        "confidence": float(obj.confidence),
                        "bbox": obj.bbox,
                        "center": obj.center,
                        "area": obj.area
                    }
                    for obj in objects
                ],
                "anomalies": [
                    {
                        "type": anomaly.anomaly_type,
                        "confidence": float(anomaly.confidence),
                        "location": anomaly.location,
                        "severity": anomaly.severity,
                        "description": anomaly.description
                    }
                    for anomaly in anomalies
                ],
                "defects": [
                    {
                        "type": defect.defect_type.value,
                        "confidence": float(defect.confidence),
                        "location": defect.location,
                        "severity": defect.severity,
                        "area": defect.area,
                        "description": defect.description
                    }
                    for defect in defects
                ],
                "summary": self._generate_summary(objects, anomalies, defects, quality_score)
            }
            
            return result
        except Exception as e:
            logger.error(f"Unexpected error in inspect_frame: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error during inspection: {str(e)}"
            }
    
    def _calculate_quality_score(
        self,
        objects: List[DetectedObject],
        anomalies: List[Anomaly],
        defects: List[Defect]
    ) -> float:
        """
        Calcular score de calidad (0-100)
        
        Args:
            objects: Objetos detectados
            anomalies: Anomalías detectadas
            defects: Defectos detectados
            
        Returns:
            Score de calidad (0-100, mayor es mejor)
        """
        if not isinstance(objects, list):
            objects = []
        if not isinstance(anomalies, list):
            anomalies = []
        if not isinstance(defects, list):
            defects = []
        
        score = 100.0
        
        # Penalizar por anomalías
        for anomaly in anomalies:
            if anomaly.severity == "high":
                score -= PENALTY_HIGH_ANOMALY
            elif anomaly.severity == "medium":
                score -= PENALTY_MEDIUM_ANOMALY
            else:
                score -= PENALTY_LOW_ANOMALY
        
        # Penalizar por defectos
        for defect in defects:
            if defect.severity == "critical":
                score -= PENALTY_CRITICAL_DEFECT
            elif defect.severity == "severe":
                score -= PENALTY_SEVERE_DEFECT
            elif defect.severity == "moderate":
                score -= PENALTY_MODERATE_DEFECT
            else:
                score -= PENALTY_MINOR_DEFECT
        
        # Asegurar que esté en rango 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    def _generate_summary(
        self,
        objects: List[DetectedObject],
        anomalies: List[Anomaly],
        defects: List[Defect],
        quality_score: float
    ) -> Dict[str, Any]:
        """
        Generar resumen de inspección
        
        Args:
            objects: Objetos detectados
            anomalies: Anomalías detectadas
            defects: Defectos detectados
            quality_score: Score de calidad
            
        Returns:
            Diccionario con resumen
        """
        # Contar defectos por tipo
        defect_counts = {}
        for defect in defects:
            defect_type = defect.defect_type.value
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        # Contar por severidad
        severity_counts = {
            "critical": 0,
            "severe": 0,
            "moderate": 0,
            "minor": 0
        }
        for defect in defects:
            severity_counts[defect.severity] = severity_counts.get(defect.severity, 0) + 1
        
        # Determinar estado general
        if quality_score >= QUALITY_SCORE_EXCELLENT:
            status = "excellent"
        elif quality_score >= QUALITY_SCORE_GOOD:
            status = "good"
        elif quality_score >= QUALITY_SCORE_ACCEPTABLE:
            status = "acceptable"
        elif quality_score >= QUALITY_SCORE_POOR:
            status = "poor"
        else:
            status = "rejected"
        
        return {
            "status": status,
            "quality_score": quality_score,
            "total_objects": len(objects),
            "total_anomalies": len(anomalies),
            "total_defects": len(defects),
            "defect_counts": defect_counts,
            "severity_counts": severity_counts,
            "has_critical_defects": severity_counts["critical"] > 0,
            "recommendation": self._get_recommendation(quality_score, defects)
        }
    
    def _get_recommendation(self, quality_score: float, defects: List[Defect]) -> str:
        """
        Obtener recomendación basada en resultados
        
        Args:
            quality_score: Score de calidad
            defects: Defectos detectados
            
        Returns:
            Recomendación
        """
        if not isinstance(defects, list):
            defects = []
        
        if quality_score >= QUALITY_SCORE_EXCELLENT:
            return "Producto aprobado - calidad excelente"
        elif quality_score >= QUALITY_SCORE_GOOD:
            return "Producto aprobado - calidad buena"
        elif quality_score >= QUALITY_SCORE_ACCEPTABLE:
            return "Producto aprobado con observaciones menores"
        elif quality_score >= QUALITY_SCORE_POOR:
            return "Producto requiere revisión manual"
        else:
            critical_defects = [d for d in defects if d.severity == "critical"]
            if critical_defects:
                return "Producto rechazado - defectos críticos detectados"
            else:
                return "Producto rechazado - múltiples defectos detectados"
    
    def inspect_batch(self, images: List[Union[np.ndarray, str]]) -> List[Dict[str, Any]]:
        """
        Inspeccionar múltiples imágenes
        
        Args:
            images: Lista de imágenes
            
        Returns:
            Lista de resultados de inspección
        """
        if not images:
            logger.warning("Empty image list provided to inspect_batch")
            return []
        
        results = []
        
        for idx, image in enumerate(images):
            try:
                result = self.inspect_frame(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image {idx} in batch: {e}", exc_info=True)
                results.append({
                    "success": False,
                    "error": f"Error processing image {idx}: {str(e)}",
                    "image_index": idx
                })
        
        return results
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Obtener información de la cámara"""
        try:
            return self.camera.get_camera_info()
        except Exception as e:
            logger.error(f"Error getting camera info: {e}", exc_info=True)
            return {"error": str(e)}
    
    def update_camera_settings(self, **kwargs):
        """Actualizar configuración de la cámara"""
        self.camera.update_settings(**kwargs)
    
    def update_detection_settings(self, **kwargs):
        """Actualizar configuración de detección"""
        self.detection_config.update_settings(**kwargs)
        # Recrear detectores con nueva configuración
        self.object_detector = ObjectDetector(self.detection_config)
        self.anomaly_detector = AnomalyDetector(self.detection_config)
        self.defect_classifier = DefectClassifier(self.detection_config)
    
    def release(self):
        """Liberar recursos"""
        try:
            self.camera.release()
            logger.info("Quality Inspector resources released")
        except Exception as e:
            logger.error(f"Error releasing resources: {e}", exc_info=True)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically release resources"""
        self.release()
        return False






