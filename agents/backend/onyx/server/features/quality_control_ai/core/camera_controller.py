"""
Controlador de Cámara para Control de Calidad
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union
from PIL import Image

from ..config.camera_config import CameraConfig, CameraSettings

logger = logging.getLogger(__name__)


class CameraController:
    """
    Controlador para gestionar la cámara y capturar imágenes
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Inicializar controlador de cámara
        
        Args:
            config: Configuración de la cámara
        """
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        self.is_streaming = False
        
        if not self.config.validate():
            raise ValueError("Invalid camera configuration")
        
        logger.info(f"Camera controller initialized for camera {self.config.settings.camera_index}")
    
    def initialize(self) -> bool:
        """
        Inicializar y abrir la cámara
        
        Returns:
            True si se inicializó correctamente
        """
        try:
            self.cap = cv2.VideoCapture(self.config.settings.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.config.settings.camera_index}")
                return False
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.settings.resolution_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.settings.resolution_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.settings.fps)
            
            # Configurar propiedades adicionales si están disponibles
            if self.config.settings.brightness is not None:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.settings.brightness)
            if self.config.settings.contrast is not None:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.settings.contrast)
            if self.config.settings.saturation is not None:
                self.cap.set(cv2.CAP_PROP_SATURATION, self.config.settings.saturation)
            if self.config.settings.exposure is not None:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.settings.exposure)
            if self.config.settings.auto_focus is not None:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.config.settings.auto_focus else 0)
            
            self.is_initialized = True
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}", exc_info=True)
            return False
    
    def start_streaming(self) -> bool:
        """
        Iniciar streaming de la cámara
        
        Returns:
            True si se inició correctamente
        """
        if not self.is_initialized:
            if not self.initialize():
                return False
        
        self.is_streaming = True
        logger.info("Camera streaming started")
        return True
    
    def stop_streaming(self):
        """Detener streaming de la cámara"""
        self.is_streaming = False
        logger.info("Camera streaming stopped")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capturar un frame de la cámara
        
        Returns:
            Frame como numpy array (BGR) o None si falla
        """
        if not self.is_initialized:
            logger.warning("Camera not initialized, attempting to initialize...")
            if not self.initialize():
                return None
        
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not opened")
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to capture frame")
            return None
        
        # Aplicar transformaciones según configuración
        frame = self._apply_transformations(frame)
        
        return frame
    
    def _apply_transformations(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplicar transformaciones según configuración
        
        Args:
            frame: Frame original
            
        Returns:
            Frame transformado
        """
        # Flip horizontal
        if self.config.settings.flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        # Flip vertical
        if self.config.settings.flip_vertical:
            frame = cv2.flip(frame, 0)
        
        # Convertir formato
        if self.config.settings.format == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.config.settings.format == "GRAY":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return frame
    
    def capture_image(self, save_path: Optional[str] = None) -> Optional[Union[np.ndarray, Image.Image]]:
        """
        Capturar una imagen completa
        
        Args:
            save_path: Ruta opcional para guardar la imagen
            
        Returns:
            Imagen como numpy array o PIL Image
        """
        frame = self.capture_frame()
        
        if frame is None:
            return None
        
        # Guardar si se especifica ruta
        if save_path:
            cv2.imwrite(save_path, frame)
            logger.info(f"Image saved to {save_path}")
        
        # Convertir a PIL Image si el formato es RGB
        if self.config.settings.format == "RGB":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
            return Image.fromarray(frame_rgb)
        
        return frame
    
    def get_camera_info(self) -> dict:
        """
        Obtener información de la cámara
        
        Returns:
            Diccionario con información de la cámara
        """
        if not self.is_initialized or self.cap is None:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized" if self.is_initialized else "not_initialized",
            "streaming": self.is_streaming,
            "camera_index": self.config.settings.camera_index,
            "resolution": {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            },
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
        }
        
        return info
    
    def update_settings(self, **kwargs):
        """
        Actualizar configuración de la cámara en tiempo de ejecución
        
        Args:
            **kwargs: Parámetros a actualizar
        """
        self.config.update_settings(**kwargs)
        
        if self.is_initialized and self.cap is not None:
            # Aplicar cambios a la cámara
            if "brightness" in kwargs:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.settings.brightness)
            if "contrast" in kwargs:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.settings.contrast)
            if "saturation" in kwargs:
                self.cap.set(cv2.CAP_PROP_SATURATION, self.config.settings.saturation)
            if "exposure" in kwargs:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.settings.exposure)
        
        logger.info(f"Camera settings updated: {kwargs}")
    
    def release(self):
        """Liberar recursos de la cámara"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_initialized = False
        self.is_streaming = False
        logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()






