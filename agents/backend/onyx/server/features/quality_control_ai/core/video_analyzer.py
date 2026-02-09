"""
Analizador de Video para Control de Calidad
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Iterator
from collections import deque
import time

from .camera_controller import CameraController
from .object_detector import ObjectDetector
from .anomaly_detector import AnomalyDetector
from .defect_classifier import DefectClassifier
from ..config.detection_config import DetectionConfig
from ..services.quality_inspector import QualityInspector

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """
    Analizador de video para inspección continua
    """
    
    def __init__(
        self,
        inspector: Optional[QualityInspector] = None,
        buffer_size: int = 30
    ):
        """
        Inicializar analizador de video
        
        Args:
            inspector: Inspector de calidad (opcional)
            buffer_size: Tamaño del buffer de frames
        """
        self.inspector = inspector or QualityInspector()
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.results_buffer = deque(maxlen=buffer_size)
        self.is_analyzing = False
        self.frame_count = 0
        self.start_time = None
        
        logger.info(f"Video Analyzer initialized with buffer size {buffer_size}")
    
    def analyze_video_file(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Analizar archivo de video
        
        Args:
            video_path: Ruta al archivo de video
            frame_skip: Analizar cada N frames
            max_frames: Máximo de frames a analizar
            
        Returns:
            Diccionario con resultados agregados
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return {"success": False, "error": "Failed to open video"}
        
        results = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Saltar frames según frame_skip
                if frame_idx % frame_skip == 0:
                    result = self.inspector.inspect_frame(frame)
                    if result.get("success"):
                        results.append(result)
                        self.frame_buffer.append(frame)
                        self.results_buffer.append(result)
                
                frame_idx += 1
            
            # Agregar resultados
            aggregated = self._aggregate_results(results)
            aggregated["total_frames"] = frame_idx
            aggregated["analyzed_frames"] = len(results)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            cap.release()
    
    def analyze_stream(
        self,
        camera: Optional[CameraController] = None,
        duration: Optional[float] = None,
        frame_skip: int = 1
    ) -> Iterator[Dict]:
        """
        Analizar stream de cámara en tiempo real
        
        Args:
            camera: Controlador de cámara (opcional)
            duration: Duración en segundos (None = infinito)
            frame_skip: Analizar cada N frames
            
        Yields:
            Resultados de inspección por frame
        """
        cam = camera or self.inspector.camera
        
        if not cam.is_initialized:
            if not cam.initialize():
                logger.error("Failed to initialize camera")
                return
        
        self.is_analyzing = True
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while self.is_analyzing:
                # Verificar duración
                if duration and (time.time() - self.start_time) > duration:
                    break
                
                # Capturar frame
                frame = cam.capture_frame()
                if frame is None:
                    continue
                
                # Analizar frame
                if self.frame_count % frame_skip == 0:
                    result = self.inspector.inspect_frame(frame)
                    result["frame_number"] = self.frame_count
                    result["timestamp"] = time.time()
                    
                    self.frame_buffer.append(frame)
                    self.results_buffer.append(result)
                    
                    yield result
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Stream analysis interrupted by user")
        finally:
            self.is_analyzing = False
    
    def stop_analysis(self):
        """Detener análisis de stream"""
        self.is_analyzing = False
        logger.info("Stream analysis stopped")
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Agregar resultados de múltiples frames
        
        Args:
            results: Lista de resultados
            
        Returns:
            Resultado agregado
        """
        if not results:
            return {"success": False, "error": "No results to aggregate"}
        
        # Calcular promedios
        quality_scores = [r.get("quality_score", 0) for r in results if r.get("success")]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Contar detecciones
        total_objects = sum(len(r.get("objects", [])) for r in results)
        total_anomalies = sum(len(r.get("anomalies", [])) for r in results)
        total_defects = sum(len(r.get("defects", [])) for r in results)
        
        # Agregar defectos por tipo
        defect_types = {}
        for result in results:
            for defect in result.get("defects", []):
                defect_type = defect.get("type", "unknown")
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
        
        # Agregar por severidad
        severity_counts = {"critical": 0, "severe": 0, "moderate": 0, "minor": 0}
        for result in results:
            for defect in result.get("defects", []):
                severity = defect.get("severity", "minor")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Determinar estado general
        if avg_quality >= 90:
            status = "excellent"
        elif avg_quality >= 75:
            status = "good"
        elif avg_quality >= 60:
            status = "acceptable"
        elif avg_quality >= 40:
            status = "poor"
        else:
            status = "rejected"
        
        return {
            "success": True,
            "aggregated": True,
            "total_frames_analyzed": len(results),
            "average_quality_score": round(avg_quality, 2),
            "min_quality_score": round(min(quality_scores) if quality_scores else 0, 2),
            "max_quality_score": round(max(quality_scores) if quality_scores else 0, 2),
            "total_objects": total_objects,
            "total_anomalies": total_anomalies,
            "total_defects": total_defects,
            "defect_types": defect_types,
            "severity_distribution": severity_counts,
            "status": status,
            "has_critical_defects": severity_counts["critical"] > 0,
            "recommendation": self._get_recommendation(avg_quality, severity_counts)
        }
    
    def _get_recommendation(self, avg_quality: float, severity_counts: Dict) -> str:
        """Obtener recomendación basada en resultados agregados"""
        if severity_counts.get("critical", 0) > 0:
            return "Rechazar lote - defectos críticos detectados en múltiples frames"
        elif avg_quality < 40:
            return "Rechazar lote - calidad promedio muy baja"
        elif avg_quality < 60:
            return "Revisar lote manualmente - calidad promedio baja"
        elif severity_counts.get("severe", 0) > 5:
            return "Revisar proceso de producción - múltiples defectos severos"
        else:
            return "Lote aprobado - calidad dentro de estándares"
    
    def get_recent_results(self, n: int = 10) -> List[Dict]:
        """
        Obtener resultados recientes del buffer
        
        Args:
            n: Número de resultados a retornar
            
        Returns:
            Lista de resultados recientes
        """
        return list(self.results_buffer)[-n:]
    
    def get_statistics(self) -> Dict:
        """Obtener estadísticas del análisis"""
        if not self.results_buffer:
            return {"status": "no_data"}
        
        results = list(self.results_buffer)
        quality_scores = [r.get("quality_score", 0) for r in results if r.get("success")]
        
        return {
            "frames_analyzed": len(results),
            "buffer_size": len(self.frame_buffer),
            "is_analyzing": self.is_analyzing,
            "average_quality": round(np.mean(quality_scores), 2) if quality_scores else 0,
            "analysis_duration": time.time() - self.start_time if self.start_time else 0
        }






