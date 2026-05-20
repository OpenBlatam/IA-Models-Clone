"""
Sistema de visualización de datos de análisis
"""

import json
from typing import Dict, List, Optional
import base64
import io

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class VisualizationGenerator:
    """Genera visualizaciones de análisis de piel"""
    
    def __init__(self):
        """Inicializa el generador de visualizaciones"""
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
    
    def generate_radar_chart(self, quality_scores: Dict, 
                           output_format: str = "base64") -> str:
        """
        Genera gráfico radar de métricas de calidad
        
        Args:
            quality_scores: Diccionario con scores
            output_format: "base64" o "bytes"
            
        Returns:
            Imagen en formato especificado
        """
        if not self.matplotlib_available:
            raise ImportError(
                "matplotlib no está instalado. Instale con: pip install matplotlib"
            )
        
        # Preparar datos
        metrics = []
        values = []
        
        for key, value in quality_scores.items():
            if key != "overall_score":
                metric_name = key.replace("_", " ").title()
                metrics.append(metric_name)
                values.append(value)
        
        # Configurar gráfico
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Cerrar el círculo
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Dibujar
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
        ax.grid(True)
        
        plt.title('Métricas de Calidad de Piel', size=16, fontweight='bold', pad=20)
        
        # Guardar en buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        
        if output_format == "base64":
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return img_base64
        else:
            return buffer.read()
    
    def generate_timeline_chart(self, timeline_data: List[Dict],
                               metric: str = "overall_score",
                               output_format: str = "base64") -> str:
        """
        Genera gráfico de línea de tiempo
        
        Args:
            timeline_data: Lista de puntos en el tiempo
            metric: Métrica a visualizar
            output_format: "base64" o "bytes"
            
        Returns:
            Imagen en formato especificado
        """
        if not self.matplotlib_available:
            raise ImportError("matplotlib no está instalado")
        
        from datetime import datetime
        
        # Preparar datos
        dates = [datetime.fromisoformat(d["timestamp"]) for d in timeline_data]
        values = [d["value"] for d in timeline_data]
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(dates, values, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax.fill_between(dates, values, alpha=0.3, color='#3498db')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Progreso de {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Rotar etiquetas de fecha
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Guardar
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        
        if output_format == "base64":
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return img_base64
        else:
            return buffer.read()
    
    def generate_comparison_chart(self, scores_before: Dict, scores_after: Dict,
                                 output_format: str = "base64") -> str:
        """
        Genera gráfico de comparación antes/después
        
        Args:
            scores_before: Scores antes
            scores_after: Scores después
            output_format: "base64" o "bytes"
            
        Returns:
            Imagen en formato especificado
        """
        if not self.matplotlib_available:
            raise ImportError("matplotlib no está instalado")
        
        # Preparar datos
        metrics = []
        before_values = []
        after_values = []
        
        for key in scores_before:
            if key != "overall_score" and key in scores_after:
                metric_name = key.replace("_", " ").title()
                metrics.append(metric_name)
                before_values.append(scores_before[key])
                after_values.append(scores_after[key])
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, before_values, width, label='Antes', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, after_values, width, label='Después', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Métricas', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparación Antes/Después', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Guardar
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        
        if output_format == "base64":
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return img_base64
        else:
            return buffer.read()
    
    def generate_score_distribution(self, quality_scores: Dict,
                                   output_format: str = "base64") -> str:
        """
        Genera gráfico de distribución de scores
        
        Args:
            quality_scores: Diccionario con scores
            output_format: "base64" o "bytes"
            
        Returns:
            Imagen en formato especificado
        """
        if not self.matplotlib_available:
            raise ImportError("matplotlib no está instalado")
        
        # Preparar datos
        metrics = []
        values = []
        colors_list = []
        
        for key, value in quality_scores.items():
            if key != "overall_score":
                metric_name = key.replace("_", " ").title()
                metrics.append(metric_name)
                values.append(value)
                # Color según score
                if value >= 80:
                    colors_list.append('#2ecc71')  # Verde
                elif value >= 60:
                    colors_list.append('#3498db')  # Azul
                elif value >= 40:
                    colors_list.append('#f39c12')  # Naranja
                else:
                    colors_list.append('#e74c3c')  # Rojo
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(metrics, values, color=colors_list, alpha=0.8)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Distribución de Métricas', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Agregar valores en las barras
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 1, i, f'{value:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        
        if output_format == "base64":
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return img_base64
        else:
            return buffer.read()






