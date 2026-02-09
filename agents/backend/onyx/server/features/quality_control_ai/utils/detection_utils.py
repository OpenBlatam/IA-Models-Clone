"""
Utilidades para detección y análisis
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box"""
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convertir a tupla"""
        return (self.x, self.y, self.width, self.height)
    
    def area(self) -> int:
        """Calcular área"""
        return self.width * self.height
    
    def center(self) -> Tuple[int, int]:
        """Calcular centro"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calcular Intersection over Union (IoU)"""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


class DetectionUtils:
    """Utilidades para detección"""
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calcular Intersection over Union entre dos bounding boxes
        
        Args:
            box1: (x, y, width, height)
            box2: (x, y, width, height)
            
        Returns:
            IoU score (0-1)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calcular intersección
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def non_max_suppression(
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        iou_threshold: float = 0.5
    ) -> List[int]:
        """
        Aplicar Non-Maximum Suppression
        
        Args:
            boxes: Lista de bounding boxes
            scores: Lista de scores
            iou_threshold: Threshold de IoU
            
        Returns:
            Índices de boxes a mantener
        """
        if not boxes:
            return []
        
        # Ordenar por score (mayor primero)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            
            # Eliminar boxes con alto IoU
            indices = [
                i for i in indices
                if DetectionUtils.calculate_iou(boxes[current], boxes[i]) < iou_threshold
            ]
        
        return keep
    
    @staticmethod
    def merge_boxes(
        boxes: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fusionar boxes superpuestos
        
        Args:
            boxes: Lista de bounding boxes
            iou_threshold: Threshold de IoU para fusionar
            
        Returns:
            Lista de boxes fusionados
        """
        if not boxes:
            return []
        
        merged = []
        used = [False] * len(boxes)
        
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
            
            group = [box1]
            used[i] = True
            
            for j, box2 in enumerate(boxes[i+1:], start=i+1):
                if used[j]:
                    continue
                
                iou = DetectionUtils.calculate_iou(box1, box2)
                if iou > iou_threshold:
                    group.append(box2)
                    used[j] = True
            
            # Fusionar grupo
            if len(group) > 1:
                x_min = min(b[0] for b in group)
                y_min = min(b[1] for b in group)
                x_max = max(b[0] + b[2] for b in group)
                y_max = max(b[1] + b[3] for b in group)
                merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
            else:
                merged.append(box1)
        
        return merged
    
    @staticmethod
    def calculate_centroid(contour: np.ndarray) -> Tuple[float, float]:
        """
        Calcular centroide de un contorno
        
        Args:
            contour: Contorno
            
        Returns:
            (x, y) del centroide
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return (0, 0)
    
    @staticmethod
    def calculate_area_ratio(contour: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calcular ratio de área del contorno vs bounding box
        
        Args:
            contour: Contorno
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Ratio (0-1)
        """
        contour_area = cv2.contourArea(contour)
        bbox_area = bbox[2] * bbox[3]
        
        if bbox_area == 0:
            return 0.0
        
        return contour_area / bbox_area
    
    @staticmethod
    def calculate_aspect_ratio(bbox: Tuple[int, int, int, int]) -> float:
        """
        Calcular relación de aspecto
        
        Args:
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Aspect ratio (width/height)
        """
        width, height = bbox[2], bbox[3]
        if height == 0:
            return 0.0
        return width / height
    
    @staticmethod
    def calculate_circularity(contour: np.ndarray) -> float:
        """
        Calcular circularidad de un contorno
        
        Args:
            contour: Contorno
            
        Returns:
            Circularidad (0-1, 1 = perfecto círculo)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        return 4 * np.pi * area / (perimeter ** 2)
    
    @staticmethod
    def extract_features(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Extraer características de una región
        
        Args:
            image: Imagen completa
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Diccionario con características
        """
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return {}
        
        features = {}
        
        # Estadísticas básicas
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        features['mean_intensity'] = float(np.mean(gray))
        features['std_intensity'] = float(np.std(gray))
        features['min_intensity'] = int(np.min(gray))
        features['max_intensity'] = int(np.max(gray))
        
        # Histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features['histogram_peak'] = int(np.argmax(hist))
        
        # Textura (usando LBP simplificado)
        features['texture_variance'] = float(np.var(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
        
        return features






