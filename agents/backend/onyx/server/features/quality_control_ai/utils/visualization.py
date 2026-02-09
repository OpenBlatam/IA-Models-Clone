"""
Utilidades de Visualización para Control de Calidad
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from PIL import Image as PILImage

from ..core.object_detector import DetectedObject
from ..core.anomaly_detector import Anomaly
from ..core.defect_classifier import Defect

logger = logging.getLogger(__name__)


class QualityVisualizer:
    """
    Visualizador de resultados de inspección de calidad
    """
    
    def __init__(self):
        """Inicializar visualizador"""
        # Colores para diferentes tipos
        self.colors = {
            "object": (0, 255, 0),      # Verde
            "anomaly_low": (0, 255, 255),    # Amarillo
            "anomaly_medium": (0, 165, 255),  # Naranja
            "anomaly_high": (0, 0, 255),      # Rojo
            "defect_minor": (0, 255, 255),    # Amarillo
            "defect_moderate": (0, 165, 255), # Naranja
            "defect_severe": (0, 0, 255),     # Rojo
            "defect_critical": (0, 0, 139),   # Rojo oscuro
        }
        logger.info("Quality Visualizer initialized")
    
    def visualize_inspection(
        self,
        image: np.ndarray,
        objects: Optional[List[DetectedObject]] = None,
        anomalies: Optional[List[Anomaly]] = None,
        defects: Optional[List[Defect]] = None,
        quality_score: Optional[float] = None,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualizar resultados completos de inspección
        
        Args:
            image: Imagen original
            objects: Objetos detectados
            anomalies: Anomalías detectadas
            defects: Defectos detectados
            quality_score: Score de calidad
            show_labels: Mostrar etiquetas
            
        Returns:
            Imagen con visualizaciones
        """
        vis_image = image.copy()
        
        # Dibujar objetos
        if objects:
            vis_image = self._draw_objects(vis_image, objects, show_labels)
        
        # Dibujar anomalías
        if anomalies:
            vis_image = self._draw_anomalies(vis_image, anomalies, show_labels)
        
        # Dibujar defectos
        if defects:
            vis_image = self._draw_defects(vis_image, defects, show_labels)
        
        # Agregar información de calidad
        if quality_score is not None:
            vis_image = self._draw_quality_info(vis_image, quality_score, defects)
        
        return vis_image
    
    def _draw_objects(
        self,
        image: np.ndarray,
        objects: List[DetectedObject],
        show_labels: bool
    ) -> np.ndarray:
        """Dibujar objetos detectados"""
        img = image.copy()
        
        for obj in objects:
            x, y, w, h = obj.bbox
            color = self.colors["object"]
            
            # Dibujar bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            if show_labels:
                label = f"{obj.class_name}: {obj.confidence:.2f}"
                self._draw_label(img, label, (x, y), color)
            
            # Dibujar centro
            cv2.circle(img, obj.center, 5, color, -1)
        
        return img
    
    def _draw_anomalies(
        self,
        image: np.ndarray,
        anomalies: List[Anomaly],
        show_labels: bool
    ) -> np.ndarray:
        """Dibujar anomalías detectadas"""
        img = image.copy()
        
        for anomaly in anomalies:
            x, y, w, h = anomaly.location
            
            # Color según severidad
            if anomaly.severity == "high":
                color = self.colors["anomaly_high"]
            elif anomaly.severity == "medium":
                color = self.colors["anomaly_medium"]
            else:
                color = self.colors["anomaly_low"]
            
            # Dibujar rectángulo con línea punteada
            self._draw_dashed_rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            if show_labels:
                label = f"{anomaly.anomaly_type}: {anomaly.confidence:.2f}"
                self._draw_label(img, label, (x, y), color)
        
        return img
    
    def _draw_defects(
        self,
        image: np.ndarray,
        defects: List[Defect],
        show_labels: bool
    ) -> np.ndarray:
        """Dibujar defectos detectados"""
        img = image.copy()
        
        for defect in defects:
            x, y, w, h = defect.location
            
            # Color según severidad
            color_key = f"defect_{defect.severity}"
            color = self.colors.get(color_key, (0, 255, 0))
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            
            # Rellenar con transparencia
            overlay = img.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            
            if show_labels:
                label = f"{defect.defect_type.value}: {defect.severity}"
                self._draw_label(img, label, (x, y), color, bg_opacity=0.8)
        
        return img
    
    def _draw_quality_info(
        self,
        image: np.ndarray,
        quality_score: float,
        defects: Optional[List[Defect]] = None
    ) -> np.ndarray:
        """Dibujar información de calidad"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Determinar color según score
        if quality_score >= 90:
            color = (0, 255, 0)  # Verde
            status = "EXCELENTE"
        elif quality_score >= 75:
            color = (0, 255, 255)  # Amarillo
            status = "BUENO"
        elif quality_score >= 60:
            color = (0, 165, 255)  # Naranja
            status = "ACEPTABLE"
        elif quality_score >= 40:
            color = (0, 0, 255)  # Rojo
            status = "POBRE"
        else:
            color = (0, 0, 139)  # Rojo oscuro
            status = "RECHAZADO"
        
        # Panel de información
        panel_height = 120
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"QUALITY SCORE: {quality_score:.1f}/100", 
                   (20, 40), font, 0.7, color, 2)
        cv2.putText(img, f"STATUS: {status}", 
                   (20, 70), font, 0.7, color, 2)
        
        if defects:
            cv2.putText(img, f"DEFECTS: {len(defects)}", 
                       (20, 100), font, 0.6, (255, 255, 255), 2)
        
        return img
    
    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        bg_opacity: float = 0.7
    ):
        """Dibujar etiqueta con fondo"""
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Obtener tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Dibujar fondo
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x, y - text_height - 10),
            (x + text_width + 10, y + baseline),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, bg_opacity, image, 1 - bg_opacity, 0, image)
        
        # Dibujar texto
        cv2.putText(
            image,
            text,
            (x + 5, y - 5),
            font,
            font_scale,
            color,
            thickness
        )
    
    def _draw_dashed_rectangle(
        self,
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_length: int = 10
    ):
        """Dibujar rectángulo con línea punteada"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Líneas horizontales
        for x in range(x1, x2, dash_length * 2):
            cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Líneas verticales
        for y in range(y1, y2, dash_length * 2):
            cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def create_summary_image(
        self,
        image: np.ndarray,
        inspection_result: Dict,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Crear imagen resumen con todos los resultados
        
        Args:
            image: Imagen original
            inspection_result: Resultados de inspección
            save_path: Ruta opcional para guardar
            
        Returns:
            Imagen resumen
        """
        # Crear imagen con visualizaciones
        vis_image = self.visualize_inspection(
            image,
            objects=inspection_result.get("objects"),
            anomalies=inspection_result.get("anomalies"),
            defects=inspection_result.get("defects"),
            quality_score=inspection_result.get("quality_score"),
            show_labels=True
        )
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            logger.info(f"Summary image saved to {save_path}")
        
        return vis_image
    
    def create_statistics_plot(
        self,
        inspection_result: Dict,
        save_path: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Crear gráfico de estadísticas
        
        Args:
            inspection_result: Resultados de inspección
            save_path: Ruta opcional para guardar
            
        Returns:
            Bytes de la imagen del gráfico
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Quality Inspection Statistics', fontsize=16, fontweight='bold')
            
            # Gráfico 1: Score de calidad
            ax1 = axes[0, 0]
            quality_score = inspection_result.get("quality_score", 0)
            colors_bar = ['green' if quality_score >= 75 else 'orange' if quality_score >= 60 else 'red']
            ax1.barh(['Quality Score'], [quality_score], color=colors_bar)
            ax1.set_xlim(0, 100)
            ax1.set_xlabel('Score')
            ax1.set_title('Quality Score')
            ax1.axvline(x=75, color='orange', linestyle='--', alpha=0.5, label='Good Threshold')
            ax1.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='Acceptable Threshold')
            ax1.legend()
            
            # Gráfico 2: Distribución de defectos por tipo
            ax2 = axes[0, 1]
            defects = inspection_result.get("defects", [])
            if defects:
                defect_types = {}
                for defect in defects:
                    defect_type = defect.get("type", "unknown")
                    defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
                
                if defect_types:
                    types = list(defect_types.keys())
                    counts = list(defect_types.values())
                    ax2.bar(types, counts, color='coral')
                    ax2.set_xlabel('Defect Type')
                    ax2.set_ylabel('Count')
                    ax2.set_title('Defects by Type')
                    ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'No Defects', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Defects by Type')
            
            # Gráfico 3: Distribución por severidad
            ax3 = axes[1, 0]
            if defects:
                severities = {}
                for defect in defects:
                    severity = defect.get("severity", "unknown")
                    severities[severity] = severities.get(severity, 0) + 1
                
                if severities:
                    sev_names = list(severities.keys())
                    sev_counts = list(severities.values())
                    colors_sev = {
                        "minor": "yellow",
                        "moderate": "orange",
                        "severe": "red",
                        "critical": "darkred"
                    }
                    bar_colors = [colors_sev.get(s, "gray") for s in sev_names]
                    ax3.bar(sev_names, sev_counts, color=bar_colors)
                    ax3.set_xlabel('Severity')
                    ax3.set_ylabel('Count')
                    ax3.set_title('Defects by Severity')
            else:
                ax3.text(0.5, 0.5, 'No Defects', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Defects by Severity')
            
            # Gráfico 4: Resumen de conteos
            ax4 = axes[1, 1]
            categories = ['Objects', 'Anomalies', 'Defects']
            counts = [
                len(inspection_result.get("objects", [])),
                len(inspection_result.get("anomalies", [])),
                len(inspection_result.get("defects", []))
            ]
            ax4.bar(categories, counts, color=['green', 'orange', 'red'])
            ax4.set_ylabel('Count')
            ax4.set_title('Detection Summary')
            
            plt.tight_layout()
            
            # Guardar o retornar
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Statistics plot saved to {save_path}")
            
            # Convertir a bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf.read()
            
        except Exception as e:
            logger.error(f"Error creating statistics plot: {e}", exc_info=True)
            plt.close()
            return None






