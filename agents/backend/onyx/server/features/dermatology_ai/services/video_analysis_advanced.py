"""
Análisis avanzado de video
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from collections import deque


class AdvancedVideoAnalysis:
    """Análisis avanzado de video"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.frame_buffer: deque = deque(maxlen=30)  # Buffer de frames
    
    def analyze_video_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analiza frames de video
        
        Args:
            frames: Lista de frames
            
        Returns:
            Diccionario con análisis
        """
        if not frames:
            return {"error": "No frames provided"}
        
        # Análisis por frame
        frame_analyses = []
        for i, frame in enumerate(frames):
            frame_analysis = self._analyze_single_frame(frame)
            frame_analysis["frame_number"] = i
            frame_analyses.append(frame_analysis)
        
        # Análisis temporal (cambios entre frames)
        temporal_analysis = self._analyze_temporal_changes(frames)
        
        # Análisis de estabilidad
        stability_analysis = self._analyze_stability(frames)
        
        # Análisis agregado
        aggregated = self._aggregate_frame_analyses(frame_analyses)
        
        return {
            "total_frames": len(frames),
            "frame_analyses": frame_analyses,
            "temporal_analysis": temporal_analysis,
            "stability_analysis": stability_analysis,
            "aggregated_analysis": aggregated
        }
    
    def _analyze_single_frame(self, frame: np.ndarray) -> Dict:
        """Analiza un frame individual"""
        # Métricas básicas
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # Contraste
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Nitidez (varianza de Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Estabilidad (movimiento)
        if len(self.frame_buffer) > 0:
            prev_frame = self.frame_buffer[-1]
            movement = self._calculate_movement(prev_frame, frame)
        else:
            movement = 0.0
        
        self.frame_buffer.append(frame)
        
        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "sharpness": float(sharpness),
            "movement": float(movement)
        }
    
    def _calculate_movement(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calcula movimiento entre frames"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        # Diferencia absoluta
        diff = cv2.absdiff(gray1, gray2)
        movement = np.mean(diff) / 255.0
        
        return float(movement)
    
    def _analyze_temporal_changes(self, frames: List[np.ndarray]) -> Dict:
        """Analiza cambios temporales"""
        if len(frames) < 2:
            return {"error": "Insufficient frames"}
        
        # Calcular cambios entre frames consecutivos
        changes = []
        for i in range(1, len(frames)):
            change = self._calculate_movement(frames[i-1], frames[i])
            changes.append(change)
        
        return {
            "average_change": float(np.mean(changes)),
            "max_change": float(np.max(changes)),
            "change_variance": float(np.var(changes)),
            "stability_score": float(1.0 - np.mean(changes))
        }
    
    def _analyze_stability(self, frames: List[np.ndarray]) -> Dict:
        """Analiza estabilidad del video"""
        if not frames:
            return {"error": "No frames"}
        
        # Calcular métricas de estabilidad
        brightness_values = []
        contrast_values = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            brightness_values.append(np.mean(gray))
            contrast_values.append(np.std(gray))
        
        brightness_std = np.std(brightness_values)
        contrast_std = np.std(contrast_values)
        
        # Score de estabilidad (menor variación = más estable)
        stability_score = 1.0 / (1.0 + brightness_std / 255.0 + contrast_std / 255.0)
        
        return {
            "brightness_stability": float(1.0 - (brightness_std / 255.0)),
            "contrast_stability": float(1.0 - (contrast_std / 255.0)),
            "overall_stability": float(stability_score),
            "is_stable": stability_score > 0.7
        }
    
    def _aggregate_frame_analyses(self, frame_analyses: List[Dict]) -> Dict:
        """Agrega análisis de frames"""
        if not frame_analyses:
            return {}
        
        brightness_values = [fa["brightness"] for fa in frame_analyses]
        contrast_values = [fa["contrast"] for fa in frame_analyses]
        sharpness_values = [fa["sharpness"] for fa in frame_analyses]
        
        return {
            "average_brightness": float(np.mean(brightness_values)),
            "average_contrast": float(np.mean(contrast_values)),
            "average_sharpness": float(np.mean(sharpness_values)),
            "brightness_range": [float(min(brightness_values)), float(max(brightness_values))],
            "contrast_range": [float(min(contrast_values)), float(max(contrast_values))],
            "quality_score": float(np.mean(sharpness_values) / 100.0)  # Normalizado
        }
    
    def extract_key_frames(self, frames: List[np.ndarray], num_frames: int = 5) -> List[int]:
        """
        Extrae frames clave del video
        
        Args:
            frames: Lista de frames
            num_frames: Número de frames a extraer
            
        Returns:
            Lista de índices de frames clave
        """
        if len(frames) <= num_frames:
            return list(range(len(frames)))
        
        # Calcular "interés" de cada frame
        frame_scores = []
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            sharpness = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            brightness = np.mean(gray)
            
            # Score basado en nitidez y exposición
            score = sharpness * (1.0 - abs(brightness - 128) / 128.0)
            frame_scores.append((i, score))
        
        # Seleccionar frames con mayor score
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        key_frame_indices = [idx for idx, _ in frame_scores[:num_frames]]
        key_frame_indices.sort()
        
        return key_frame_indices






