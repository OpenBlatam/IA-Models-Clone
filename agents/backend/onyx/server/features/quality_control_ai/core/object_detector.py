"""
Detector de Objetos para Control de Calidad
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from ..config.detection_config import DetectionConfig, DetectionSettings
from ..utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Objeto detectado"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]  # (cx, cy)
    area: int
    mask: Optional[np.ndarray] = None  # Máscara de segmentación opcional


class ObjectDetector:
    """
    Detector de objetos usando modelos de deep learning
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Inicializar detector de objetos
        
        Args:
            config: Configuración de detección
        """
        self.config = config or DetectionConfig()
        self.model = None
        self.class_names = []
        self.image_utils = ImageUtils()
        
        if not self.config.validate():
            raise ValueError("Invalid detection configuration")
        
        self._load_model()
        logger.info(f"Object detector initialized with model {self.config.settings.object_detection_model}")
    
    def _load_model(self):
        """Cargar modelo de detección de objetos"""
        try:
            model_type = self.config.settings.object_detection_model
            
            if model_type == "yolov8":
                # Intentar cargar YOLOv8 (requiere ultralytics)
                try:
                    from ultralytics import YOLO
                    # Cargar modelo pre-entrenado o personalizado
                    self.model = YOLO('yolov8n.pt')  # nano version por defecto
                    logger.info("YOLOv8 model loaded successfully")
                except ImportError:
                    logger.warning("ultralytics not available, using OpenCV DNN")
                    self._load_opencv_model()
            
            elif model_type in ["faster_rcnn", "ssd"]:
                self._load_opencv_model()
            
            else:
                logger.warning(f"Unknown model type {model_type}, using basic detection")
                self.model = None
            
            # Clases COCO por defecto (80 clases)
            self.class_names = self._get_coco_classes()
            
        except Exception as e:
            logger.error(f"Error loading detection model: {e}", exc_info=True)
            self.model = None
    
    def _load_opencv_model(self):
        """Cargar modelo usando OpenCV DNN"""
        try:
            # Aquí se cargarían los archivos de modelo (weights, config)
            # Por ahora, placeholder
            logger.info("OpenCV DNN model placeholder")
            self.model = "opencv_dnn"
        except Exception as e:
            logger.error(f"Error loading OpenCV model: {e}")
            self.model = None
    
    def _get_coco_classes(self) -> List[str]:
        """Obtener nombres de clases COCO"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    
    def detect(self, image: Union[np.ndarray, str]) -> List[DetectedObject]:
        """
        Detectar objetos en una imagen
        
        Args:
            image: Imagen como numpy array o ruta de archivo
            
        Returns:
            Lista de objetos detectados
        """
        # Cargar imagen
        img = self.image_utils.load_image(image)
        if img is None:
            logger.error("Failed to load image")
            return []
        
        # Preprocesar si está habilitado
        if self.config.settings.preprocessing_enabled:
            img = self._preprocess_image(img)
        
        # Detectar objetos
        if self.model is None:
            # Detección básica usando contornos
            return self._detect_basic(img)
        
        # Detección con modelo
        if isinstance(self.model, str) and self.model == "opencv_dnn":
            return self._detect_opencv(img)
        else:
            # YOLOv8 u otro modelo
            return self._detect_yolo(img)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesar imagen para mejor detección"""
        img = image.copy()
        
        # Reducción de ruido
        if self.config.settings.noise_reduction:
            img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Mejora de contraste
        if self.config.settings.contrast_enhancement:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        return img
    
    def _detect_yolo(self, image: np.ndarray) -> List[DetectedObject]:
        """Detectar usando YOLOv8"""
        try:
            results = self.model(image, conf=self.config.settings.confidence_threshold)
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Obtener información del bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filtrar por confianza
                    if confidence < self.config.settings.confidence_threshold:
                        continue
                    
                    # Convertir a formato (x, y, width, height)
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    center = (x + width // 2, y + height // 2)
                    area = width * height
                    
                    obj = DetectedObject(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x, y, width, height),
                        center=center,
                        area=area
                    )
                    detected_objects.append(obj)
            
            # Aplicar NMS
            detected_objects = self._apply_nms(detected_objects)
            
            # Limitar número de objetos
            detected_objects = detected_objects[:self.config.settings.max_objects]
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}", exc_info=True)
            return []
    
    def _detect_opencv(self, image: np.ndarray) -> List[DetectedObject]:
        """Detectar usando OpenCV DNN"""
        # Placeholder para implementación con OpenCV DNN
        logger.warning("OpenCV DNN detection not fully implemented, using basic detection")
        return self._detect_basic(image)
    
    def _detect_basic(self, image: np.ndarray) -> List[DetectedObject]:
        """Detección básica usando contornos"""
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Aplicar threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100:  # Filtrar objetos muy pequeños
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                obj = DetectedObject(
                    class_id=0,
                    class_name="object",
                    confidence=0.5,
                    bbox=(x, y, w, h),
                    center=center,
                    area=area
                )
                detected_objects.append(obj)
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in basic detection: {e}", exc_info=True)
            return []
    
    def _apply_nms(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Aplicar Non-Maximum Suppression"""
        if not objects:
            return []
        
        # Convertir a formato OpenCV
        boxes = []
        scores = []
        for obj in objects:
            x, y, w, h = obj.bbox
            boxes.append([x, y, x + w, y + h])
            scores.append(obj.confidence)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Aplicar NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.config.settings.confidence_threshold,
            self.config.settings.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Filtrar objetos
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        
        return [objects[i] for i in indices]
    
    def draw_detections(self, image: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
        """
        Dibujar detecciones en la imagen
        
        Args:
            image: Imagen original
            objects: Objetos detectados
            
        Returns:
            Imagen con detecciones dibujadas
        """
        img = image.copy()
        
        for obj in objects:
            x, y, w, h = obj.bbox
            
            # Dibujar bounding box
            color = (0, 255, 0)  # Verde
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Dibujar etiqueta
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Dibujar centro
            cv2.circle(img, obj.center, 5, (255, 0, 0), -1)
        
        return img






