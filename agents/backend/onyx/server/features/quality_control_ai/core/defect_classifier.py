"""
Clasificador de Defectos para Control de Calidad
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..config.detection_config import DetectionConfig
from ..utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

MIN_LINE_LENGTH = 20
MIN_LINE_LENGTH_SCRATCH = 30
MIN_CONTOUR_AREA_DEFECT = 100
OVERLAP_THRESHOLD = 0.5
SEVERITY_CRITICAL_THRESHOLD = 2.0
SEVERITY_SEVERE_THRESHOLD = 1.0
SEVERITY_MODERATE_THRESHOLD = 0.5
MIN_CONFIDENCE_THRESHOLD = 0.5
AVG_LINE_LENGTH_THRESHOLD = 20


class DefectType(Enum):
    """Tipos de defectos"""
    SCRATCH = "scratch"
    CRACK = "crack"
    DENT = "dent"
    DISCOLORATION = "discoloration"
    DEFORMATION = "deformation"
    MISSING_PART = "missing_part"
    SURFACE_IMPERFECTION = "surface_imperfection"
    CONTAMINATION = "contamination"
    SIZE_VARIATION = "size_variation"
    OTHER = "other"


@dataclass
class Defect:
    """Defecto detectado y clasificado"""
    defect_type: DefectType
    confidence: float
    location: Tuple[int, int, int, int]  # (x, y, width, height)
    severity: str  # "minor", "moderate", "severe", "critical"
    area: int  # Área en píxeles
    description: str
    features: Optional[Dict] = None


class DefectClassifier:
    """
    Clasificador de defectos usando análisis de imagen y ML
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Inicializar clasificador de defectos
        
        Args:
            config: Configuración de detección
        """
        self.config = config or DetectionConfig()
        self.image_utils = ImageUtils()
        self.model = None
        
        # Cargar modelo si está disponible
        if self.config.settings.defect_classifier_model:
            self._load_model()
        
        logger.info("Defect classifier initialized")
    
    def _load_model(self):
        """Cargar modelo de clasificación de defectos"""
        try:
            # Placeholder para modelo de ML
            # En producción, se cargaría un modelo entrenado
            logger.info("Defect classifier model placeholder")
            self.model = "placeholder"
        except Exception as e:
            logger.warning(f"Could not load defect classifier model: {e}")
            self.model = None
    
    def classify_defects(
        self, 
        image: Union[np.ndarray, str],
        anomalies: Optional[List] = None
    ) -> List[Defect]:
        """
        Clasificar defectos en una imagen
        
        Args:
            image: Imagen como numpy array o ruta de archivo
            anomalies: Lista opcional de anomalías detectadas previamente
            
        Returns:
            Lista de defectos clasificados
        """
        # Cargar imagen
        img = self.image_utils.load_image(image)
        if img is None:
            logger.error("Failed to load image")
            return []
        
        defects = []
        
        # Si hay anomalías, clasificarlas como defectos
        if anomalies:
            for anomaly in anomalies:
                defect = self._classify_anomaly_as_defect(anomaly, img)
                if defect:
                    defects.append(defect)
        
        detection_methods = [
            self._detect_scratches,
            self._detect_cracks,
            self._detect_dents,
            self._detect_discolorations,
            self._detect_missing_parts,
            self._detect_contamination,
        ]
        
        for detection_method in detection_methods:
            try:
                detected = detection_method(img)
                defects.extend(detected)
            except Exception as e:
                logger.error(f"Error in {detection_method.__name__}: {e}", exc_info=True)
        
        # Filtrar por tamaño
        defects = [
            d for d in defects 
            if self.config.settings.min_defect_size <= d.area <= self.config.settings.max_defect_size
        ]
        
        # Eliminar duplicados (misma ubicación)
        defects = self._remove_duplicates(defects)
        
        return defects
    
    def _classify_anomaly_as_defect(self, anomaly, image: np.ndarray) -> Optional[Defect]:
        """Clasificar una anomalía como defecto específico"""
        x, y, w, h = anomaly.location
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # Analizar características para determinar tipo de defecto
        defect_type = self._determine_defect_type(roi, anomaly)
        
        if defect_type is None:
            return None
        
        area = w * h
        
        defect = self._create_defect_from_location(
            defect_type,
            anomaly.confidence,
            anomaly.location,
            area,
            f"{defect_type.value} detected: {anomaly.description}",
            anomaly.features
        )
        
        return defect
    
    def _determine_defect_type(self, roi: np.ndarray, anomaly) -> Optional[DefectType]:
        """Determinar tipo de defecto basado en características"""
        gray = self.image_utils.ensure_grayscale(roi)
        
        # Analizar características
        # Detectar líneas (scratch/crack)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=5)
        
        if lines is not None and len(lines) > 0:
            # Líneas largas y rectas = scratch o crack
            avg_length = np.mean([np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines])
            if avg_length > AVG_LINE_LENGTH_THRESHOLD:
                # Determinar si es crack (más ancho) o scratch (más delgado)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_width = max([cv2.boundingRect(c)[2] for c in contours])
                    if max_width > 5:
                        return DefectType.CRACK
                    else:
                        return DefectType.SCRATCH
        
        # Detectar cambios de color (discoloration)
        if anomaly.anomaly_type == "color":
            return DefectType.DISCOLORATION
        
        # Detectar deformaciones (basado en forma)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MIN_CONTOUR_AREA_DEFECT:
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        a, b = ellipse[1]
                        if a > 0 and b > 0:
                            eccentricity = np.sqrt(1 - (b/a)**2) if a >= b else np.sqrt(1 - (a/b)**2)
                            if eccentricity > 0.8:  # Muy elíptico = deformación
                                return DefectType.DEFORMATION
        
        # Por defecto, otro tipo
        return DefectType.OTHER
    
    def _detect_scratches(self, image: np.ndarray) -> List[Defect]:
        """Detectar rayones/scratch"""
        defects = []
        
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Detectar líneas (HoughLines)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=5)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if length > MIN_LINE_LENGTH_SCRATCH:
                        # Crear bounding box alrededor de la línea
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = abs(x2 - x1) + 5
                        h = abs(y2 - y1) + 5
                        
                        area = w * h
                        confidence = min(1.0, length / 100.0)
                        
                        if confidence >= MIN_CONFIDENCE_THRESHOLD:
                            defect = self._create_defect_from_location(
                                DefectType.SCRATCH,
                                confidence,
                                (x, y, w, h),
                                area,
                                f"Scratch detected (length: {length:.1f}px)",
                                {"length": float(length)}
                            )
                            if defect:
                                defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error detecting scratches: {e}", exc_info=True)
        
        return defects
    
    def _detect_cracks(self, image: np.ndarray) -> List[Defect]:
        """Detectar grietas/cracks"""
        defects = []
        
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Aplicar threshold adaptativo
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > 5000:
                    continue
                
                # Calcular relación aspecto
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                
                # Cracks suelen ser largos y delgados
                if aspect_ratio > 3.0:
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                    
                    # Cracks tienen baja circularidad
                    if circularity < 0.3:
                        confidence = min(1.0, (1.0 - circularity) * aspect_ratio / 10.0)
                        
                        if confidence >= MIN_CONFIDENCE_THRESHOLD:
                            defect = self._create_defect_from_location(
                                DefectType.CRACK,
                                confidence,
                                (x, y, w, h),
                                area,
                                f"Crack detected (aspect ratio: {aspect_ratio:.2f})",
                                {"aspect_ratio": float(aspect_ratio), "circularity": float(circularity)}
                            )
                            if defect:
                                defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error detecting cracks: {e}", exc_info=True)
        
        return defects
    
    def _detect_dents(self, image: np.ndarray) -> List[Defect]:
        """Detectar abolladuras/dents"""
        defects = []
        
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Aplicar filtro gaussiano
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # Detectar cambios de intensidad (dents causan sombras)
            diff = cv2.absdiff(gray, blurred)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > 10000:
                    continue
                
                # Dents suelen ser circulares/elípticas
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                
                if circularity > 0.5:  # Relativamente circular
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(1.0, area / 5000.0)
                    
                    if confidence >= MIN_CONFIDENCE_THRESHOLD:
                        defect = self._create_defect_from_location(
                            DefectType.DENT,
                            confidence,
                            (x, y, w, h),
                            area,
                            f"Dent detected (circularity: {circularity:.2f})",
                            {"circularity": float(circularity)}
                        )
                        if defect:
                            defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error detecting dents: {e}", exc_info=True)
        
        return defects
    
    def _detect_discolorations(self, image: np.ndarray) -> List[Defect]:
        """Detectar decoloraciones"""
        defects = []
        
        try:
            if not self.image_utils.is_color_image(image):
                return defects  # Necesita imagen a color
            
            # Convertir a LAB para mejor detección de color
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Calcular desviación estándar de L (luminosidad)
            mean_l = np.mean(l)
            std_l = np.std(l)
            
            # Detectar áreas con luminosidad anómala
            threshold = mean_l - 2 * std_l
            mask = (l < threshold).astype(np.uint8) * 255
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 200 or area > 50000:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(1.0, area / 10000.0)
                
                if confidence >= MIN_CONFIDENCE_THRESHOLD:
                    defect = self._create_defect_from_location(
                        DefectType.DISCOLORATION,
                        confidence,
                        (x, y, w, h),
                        area,
                        "Discoloration detected",
                        {"area": int(area)}
                    )
                    if defect:
                        defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error detecting discolorations: {e}", exc_info=True)
        
        return defects
    
    def _detect_missing_parts(self, image: np.ndarray) -> List[Defect]:
        """Detectar partes faltantes"""
        defects = []
        
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Aplicar threshold para detectar áreas vacías
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Encontrar contornos de áreas muy claras (posiblemente vacías)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:
                    continue
                
                # Verificar si es una forma regular (posible parte faltante)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                
                # Partes faltantes suelen tener formas regulares
                if circularity > 0.6:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(1.0, area / 20000.0)
                    
                    if confidence >= MIN_CONFIDENCE_THRESHOLD:
                        defect = self._create_defect_from_location(
                            DefectType.MISSING_PART,
                            confidence,
                            (x, y, w, h),
                            area,
                            "Missing part detected",
                            {"circularity": float(circularity), "area": int(area)}
                        )
                        if defect:
                            defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error detecting missing parts: {e}", exc_info=True)
        
        return defects
    
    def _detect_contamination(self, image: np.ndarray) -> List[Defect]:
        """Detectar contaminación"""
        defects = []
        
        try:
            # Convertir a HSV para mejor detección de colores anómalos
            if not self.image_utils.is_color_image(image):
                return defects
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detectar colores fuera del rango esperado
            # Asumir que el producto tiene colores específicos
            # Cualquier color muy diferente puede ser contaminación
            
            # Calcular histograma de colores
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # Detectar picos anómalos
            mean_h = np.mean(hist_h)
            std_h = np.std(hist_h)
            threshold_h = mean_h + 3 * std_h
            
            # Crear máscara para colores anómalos
            peaks = np.where(hist_h > threshold_h)[0]
            
            if len(peaks) > 0:
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for peak in peaks:
                    mask |= ((hsv[:, :, 0] >= peak - 5) & (hsv[:, :, 0] <= peak + 5)).astype(np.uint8) * 255
                
                # Encontrar contornos
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100 or area > 10000:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(1.0, area / 5000.0)
                    
                    if confidence >= MIN_CONFIDENCE_THRESHOLD:
                        defect = self._create_defect_from_location(
                            DefectType.CONTAMINATION,
                            confidence,
                            (x, y, w, h),
                            area,
                            "Contamination detected",
                            {"area": int(area)}
                        )
                        if defect:
                            defects.append(defect)
            
        except Exception as e:
            logger.error(f"Error detecting contamination: {e}", exc_info=True)
        
        return defects
    
    def _determine_severity(
        self, 
        confidence: float, 
        area: int, 
        defect_type: DefectType
    ) -> str:
        """Determinar severidad del defecto"""
        # Factores según tipo de defecto
        severity_factors = {
            DefectType.CRACK: 1.5,  # Cracks son más críticos
            DefectType.MISSING_PART: 2.0,
            DefectType.DEFORMATION: 1.3,
            DefectType.SCRATCH: 1.0,
            DefectType.DENT: 1.2,
            DefectType.DISCOLORATION: 0.8,
            DefectType.SURFACE_IMPERFECTION: 0.9,
            DefectType.CONTAMINATION: 1.1,
            DefectType.SIZE_VARIATION: 1.0,
            DefectType.OTHER: 1.0
        }
        
        factor = severity_factors.get(defect_type, 1.0)
        severity_score = confidence * factor * (area / 1000.0)
        
        if severity_score > 2.0:
            return "critical"
        elif severity_score > 1.0:
            return "severe"
        elif severity_score > 0.5:
            return "moderate"
        else:
            return "minor"
    
    def _remove_duplicates(self, defects: List[Defect]) -> List[Defect]:
        """Eliminar defectos duplicados (misma ubicación)"""
        if not defects:
            return []
        
        # Ordenar por confianza (mayor primero)
        sorted_defects = sorted(defects, key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        for defect in sorted_defects:
            # Verificar si se superpone con defectos ya agregados
            is_duplicate = False
            x1, y1, w1, h1 = defect.location
            
            for existing in filtered:
                x2, y2, w2, h2 = existing.location
                
                # Calcular intersección
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                if overlap_area > OVERLAP_THRESHOLD * min(w1 * h1, w2 * h2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(defect)
        
        return filtered
    
    def draw_defects(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """
        Dibujar defectos en la imagen
        
        Args:
            image: Imagen original
            defects: Defectos detectados
            
        Returns:
            Imagen con defectos marcados
        """
        img = image.copy()
        
        # Colores según severidad
        color_map = {
            "minor": (0, 255, 255),  # Amarillo
            "moderate": (0, 165, 255),  # Naranja
            "severe": (0, 0, 255),  # Rojo
            "critical": (0, 0, 139)  # Rojo oscuro
        }
        
        for defect in defects:
            x, y, w, h = defect.location
            color = color_map.get(defect.severity, (0, 255, 0))
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Dibujar etiqueta
            label = f"{defect.defect_type.value}: {defect.severity}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img

