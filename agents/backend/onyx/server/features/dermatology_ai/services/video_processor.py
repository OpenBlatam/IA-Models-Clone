"""
Procesador de videos para análisis de piel
"""

from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import io
import cv2


class VideoProcessor:
    """Procesa videos para análisis de piel"""
    
    def __init__(self, target_fps: int = 1, max_frames: int = 30):
        """
        Inicializa el procesador de video
        
        Args:
            target_fps: FPS objetivo para extraer frames (1 = 1 frame por segundo)
            max_frames: Máximo número de frames a procesar
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
    
    def extract_frames(self, video_bytes: bytes) -> List[np.ndarray]:
        """
        Extrae frames de un video
        
        Args:
            video_bytes: Video como bytes
            
        Returns:
            Lista de frames como numpy arrays
        """
        frames = []
        
        # Guardar bytes temporalmente
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Abrir video con OpenCV
            cap = cv2.VideoCapture(tmp_path)
            
            if not cap.isOpened():
                raise ValueError("No se pudo abrir el video")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.target_fps) if fps > 0 else 1
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < self.max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convertir BGR a RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return frames
    
    def extract_frames_from_path(self, video_path: str) -> List[np.ndarray]:
        """
        Extrae frames de un video desde un path
        
        Args:
            video_path: Path al archivo de video
            
        Returns:
            Lista de frames como numpy arrays
        """
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / self.target_fps) if fps > 0 else 1
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convertir BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        return frames
    
    def validate_video(self, video_bytes: bytes) -> Tuple[bool, str]:
        """
        Valida la calidad del video
        
        Args:
            video_bytes: Video como bytes
            
        Returns:
            Tupla (es_válido, mensaje)
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            
            if not cap.isOpened():
                return False, "No se pudo abrir el video"
            
            # Verificar que tenga frames
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False, "Video sin frames válidos"
            
            # Verificar resolución mínima
            height, width = frame.shape[:2]
            if height < 100 or width < 100:
                cap.release()
                return False, "Resolución muy baja. Mínimo 100x100 píxeles"
            
            # Verificar duración
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            if duration < 1:
                cap.release()
                return False, "Video muy corto. Mínimo 1 segundo"
            
            cap.release()
            return True, f"Video válido: {duration:.1f}s, {width}x{height}, {fps:.1f}fps"
        
        except Exception as e:
            return False, f"Error validando video: {str(e)}"
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

