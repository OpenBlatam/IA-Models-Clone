"""
Detector de Anomalías para Control de Calidad
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config.detection_config import DetectionConfig
from ..utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

BLOCK_SIZE = 64
PATCH_SIZE = 32
MIN_CONTOUR_AREA = 50
MIN_COLOR_ANOMALY_AREA = 100
EPSILON = 1e-6
CIRCULARITY_THRESHOLD = 0.3
CONFIDENCE_HIGH_THRESHOLD = 0.8
CONFIDENCE_MEDIUM_THRESHOLD = 0.6


@dataclass
class Anomaly:
    """Anomalía detectada"""
    anomaly_type: str  # "statistical", "autoencoder", "edge", "color"
    confidence: float
    location: Tuple[int, int, int, int]  # (x, y, width, height)
    severity: str  # "low", "medium", "high"
    description: str
    features: Optional[Dict] = None


class AnomalyDetector:
    """
    Detector de anomalías usando múltiples métodos
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Inicializar detector de anomalías
        
        Args:
            config: Configuración de detección
        """
        self.config = config or DetectionConfig()
        self.image_utils = ImageUtils()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.pca = None
        self.reference_features = None
        self.autoencoder_model = None
        
        if self.config.settings.use_autoencoder:
            self._initialize_autoencoder()
        
        logger.info("Anomaly detector initialized")
    
    def _initialize_autoencoder(self):
        """Inicializar modelo de autoencoder para detección de anomalías"""
        try:
            # Placeholder para autoencoder
            # En producción, se cargaría un modelo pre-entrenado
            logger.info("Autoencoder model placeholder initialized")
            self.autoencoder_model = "placeholder"
        except Exception as e:
            logger.warning(f"Could not initialize autoencoder: {e}")
            self.autoencoder_model = None
    
    def detect_anomalies(self, image: Union[np.ndarray, str]) -> List[Anomaly]:
        """
        Detectar anomalías en una imagen
        
        Args:
            image: Imagen como numpy array o ruta de archivo
            
        Returns:
            Lista de anomalías detectadas
        """
        # Cargar imagen
        img = self.image_utils.load_image(image)
        if img is None:
            logger.error("Failed to load image")
            return []
        
        anomalies = []
        
        # Detección estadística
        if self.config.settings.use_statistical:
            statistical_anomalies = self._detect_statistical_anomalies(img)
            anomalies.extend(statistical_anomalies)
        
        # Detección con autoencoder
        if self.config.settings.use_autoencoder and self.autoencoder_model:
            autoencoder_anomalies = self._detect_autoencoder_anomalies(img)
            anomalies.extend(autoencoder_anomalies)
        
        # Detección por bordes
        edge_anomalies = self._detect_edge_anomalies(img)
        anomalies.extend(edge_anomalies)
        
        # Detección por color
        color_anomalies = self._detect_color_anomalies(img)
        anomalies.extend(color_anomalies)
        
        # Filtrar por threshold
        filtered_anomalies = [
            a for a in anomalies 
            if a.confidence >= self.config.settings.anomaly_threshold
        ]
        
        return filtered_anomalies
    
    def _detect_statistical_anomalies(self, image: np.ndarray) -> List[Anomaly]:
        """Detección de anomalías usando métodos estadísticos"""
        anomalies = []
        
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Dividir imagen en bloques
            h, w = gray.shape
            blocks = []
            block_locations = []
            
            for y in range(0, h, BLOCK_SIZE):
                for x in range(0, w, BLOCK_SIZE):
                    block = gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
                    if block.size > 0:
                        blocks.append(block.flatten())
                        block_locations.append((x, y, min(BLOCK_SIZE, w-x), min(BLOCK_SIZE, h-y)))
            
            if not blocks:
                return anomalies
            
            blocks = np.array(blocks)
            
            # Calcular estadísticas
            means = np.mean(blocks, axis=1)
            stds = np.std(blocks, axis=1)
            
            # Detectar outliers usando Z-score
            mean_mean = np.mean(means)
            std_mean = np.std(means)
            
            mean_std = np.mean(stds)
            std_std = np.std(stds)
            
            threshold = self.config.settings.statistical_threshold
            
            for i, (mean_val, std_val) in enumerate(zip(means, stds)):
                z_score_mean = abs((mean_val - mean_mean) / (std_mean + EPSILON))
                z_score_std = abs((std_val - mean_std) / (std_std + EPSILON))
                
                if z_score_mean > threshold or z_score_std > threshold:
                    confidence = min(1.0, (z_score_mean + z_score_std) / (2 * threshold))
                    severity = "high" if confidence > CONFIDENCE_HIGH_THRESHOLD else "medium" if confidence > CONFIDENCE_MEDIUM_THRESHOLD else "low"
                    
                    x, y, w, h = block_locations[i]
                    anomaly = Anomaly(
                        anomaly_type="statistical",
                        confidence=float(confidence),
                        location=(x, y, w, h),
                        severity=severity,
                        description=f"Statistical anomaly detected (Z-score: {z_score_mean:.2f})",
                        features={"z_score_mean": float(z_score_mean), "z_score_std": float(z_score_std)}
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}", exc_info=True)
        
        return anomalies
    
    def _detect_autoencoder_anomalies(self, image: np.ndarray) -> List[Anomaly]:
        """Detección de anomalías usando autoencoder"""
        anomalies = []
        
        try:
            # Placeholder para autoencoder
            # En producción, se usaría un modelo entrenado
            if self.autoencoder_model == "placeholder":
                # Simulación básica
                h, w = image.shape[:2]
                
                # Dividir en patches
                for y in range(0, h, PATCH_SIZE):
                    for x in range(0, w, PATCH_SIZE):
                        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        if patch.size == 0:
                            continue
                        
                        # Simular reconstrucción error
                        reconstruction_error = np.random.random()  # Placeholder
                        
                        if reconstruction_error > self.config.settings.anomaly_threshold:
                            confidence = float(reconstruction_error)
                            severity = "high" if confidence > CONFIDENCE_HIGH_THRESHOLD else "medium" if confidence > CONFIDENCE_MEDIUM_THRESHOLD else "low"
                            
                            anomaly = Anomaly(
                                anomaly_type="autoencoder",
                                confidence=confidence,
                                location=(x, y, min(PATCH_SIZE, w-x), min(PATCH_SIZE, h-y)),
                                severity=severity,
                                description=f"Autoencoder reconstruction error: {reconstruction_error:.2f}",
                                features={"reconstruction_error": float(reconstruction_error)}
                            )
                            anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in autoencoder anomaly detection: {e}", exc_info=True)
        
        return anomalies
    
    def _detect_edge_anomalies(self, image: np.ndarray) -> List[Anomaly]:
        """Detección de anomalías basada en bordes"""
        anomalies = []
        
        try:
            gray = self.image_utils.ensure_grayscale(image)
            
            # Detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analizar contornos irregulares
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA:
                    continue
                
                # Calcular circularidad
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # Contornos muy irregulares pueden ser anomalías
                if circularity < CIRCULARITY_THRESHOLD:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    confidence = min(1.0, (1.0 - circularity) * 1.5)
                    if confidence >= self.config.settings.anomaly_threshold:
                        severity = "high" if confidence > CONFIDENCE_HIGH_THRESHOLD else "medium" if confidence > CONFIDENCE_MEDIUM_THRESHOLD else "low"
                        
                        anomaly = Anomaly(
                            anomaly_type="edge",
                            confidence=float(confidence),
                            location=(x, y, w, h),
                            severity=severity,
                            description=f"Irregular edge detected (circularity: {circularity:.2f})",
                            features={"circularity": float(circularity), "area": int(area)}
                        )
                        anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in edge anomaly detection: {e}", exc_info=True)
        
        return anomalies
    
    def _detect_color_anomalies(self, image: np.ndarray) -> List[Anomaly]:
        """Detección de anomalías basada en color"""
        anomalies = []
        
        try:
            if not self.image_utils.is_color_image(image):
                return anomalies
            
            # Convertir a HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calcular histograma de colores
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalizar
            hist_h = hist_h / (hist_h.sum() + 1e-6)
            hist_s = hist_s / (hist_s.sum() + 1e-6)
            hist_v = hist_v / (hist_v.sum() + 1e-6)
            
            # Detectar picos anómalos en el histograma
            # Picos muy altos pueden indicar áreas con color anómalo
            h_peaks = np.where(hist_h > np.mean(hist_h) + 2 * np.std(hist_h))[0]
            s_peaks = np.where(hist_s > np.mean(hist_s) + 2 * np.std(hist_s))[0]
            v_peaks = np.where(hist_v > np.mean(hist_v) + 2 * np.std(hist_v))[0]
            
            if len(h_peaks) > 0 or len(s_peaks) > 0 or len(v_peaks) > 0:
                # Crear máscara para colores anómalos
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                for h_val in h_peaks:
                    mask |= (hsv[:, :, 0] == h_val).astype(np.uint8)
                
                # Encontrar regiones
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < MIN_COLOR_ANOMALY_AREA:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calcular confianza basada en el tamaño y la desviación
                    confidence = min(1.0, area / 10000.0)
                    if confidence >= self.config.settings.anomaly_threshold:
                        severity = "high" if confidence > CONFIDENCE_HIGH_THRESHOLD else "medium" if confidence > CONFIDENCE_MEDIUM_THRESHOLD else "low"
                        
                        anomaly = Anomaly(
                            anomaly_type="color",
                            confidence=float(confidence),
                            location=(x, y, w, h),
                            severity=severity,
                            description="Color anomaly detected",
                            features={"area": int(area)}
                        )
                        anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Error in color anomaly detection: {e}", exc_info=True)
        
        return anomalies
    
    def set_reference(self, reference_image: Union[np.ndarray, str]):
        """
        Establecer imagen de referencia para comparación
        
        Args:
            reference_image: Imagen de referencia
        """
        img = self.image_utils.load_image(reference_image)
        if img is not None:
            # Extraer características de referencia
            self.reference_features = self._extract_features(img)
            logger.info("Reference image set for anomaly detection")
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extraer características de la imagen"""
        gray = self.image_utils.ensure_grayscale(image)
        
        # Redimensionar para características
        gray_resized = cv2.resize(gray, (64, 64))
        
        # Aplanar
        features = gray_resized.flatten()
        
        return features
    
    def draw_anomalies(self, image: np.ndarray, anomalies: List[Anomaly]) -> np.ndarray:
        """
        Dibujar anomalías en la imagen
        
        Args:
            image: Imagen original
            anomalies: Anomalías detectadas
            
        Returns:
            Imagen con anomalías marcadas
        """
        img = image.copy()
        
        # Colores según severidad
        color_map = {
            "low": (0, 255, 255),  # Amarillo
            "medium": (0, 165, 255),  # Naranja
            "high": (0, 0, 255)  # Rojo
        }
        
        for anomaly in anomalies:
            x, y, w, h = anomaly.location
            color = color_map.get(anomaly.severity, (0, 255, 0))
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Dibujar etiqueta
            label = f"{anomaly.anomaly_type}: {anomaly.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img

