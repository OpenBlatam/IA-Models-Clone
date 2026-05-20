"""
Sistema de exportación avanzada (Excel, CSV)
"""

import csv
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import io

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ExportManager:
    """Gestor de exportación de datos"""
    
    def __init__(self):
        """Inicializa el gestor de exportación"""
        self.pandas_available = PANDAS_AVAILABLE
    
    def export_to_csv(self, data: List[Dict], filename: Optional[str] = None) -> bytes:
        """
        Exporta datos a CSV
        
        Args:
            data: Lista de diccionarios con datos
            filename: Nombre del archivo (opcional)
            
        Returns:
            Bytes del archivo CSV
        """
        if not data:
            raise ValueError("No hay datos para exportar")
        
        buffer = io.StringIO()
        
        # Obtener todas las claves
        fieldnames = set()
        for item in data:
            fieldnames.update(self._flatten_dict(item).keys())
        
        fieldnames = sorted(fieldnames)
        
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in data:
            flattened = self._flatten_dict(item)
            writer.writerow(flattened)
        
        buffer.seek(0)
        return buffer.getvalue().encode('utf-8')
    
    def export_to_excel(self, data: Dict[str, List[Dict]], 
                       filename: Optional[str] = None) -> bytes:
        """
        Exporta datos a Excel (múltiples hojas)
        
        Args:
            data: Diccionario con nombre de hoja -> lista de datos
            filename: Nombre del archivo (opcional)
            
        Returns:
            Bytes del archivo Excel
        """
        if not self.pandas_available:
            raise ImportError(
                "pandas no está instalado. Instale con: pip install pandas openpyxl"
            )
        
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, sheet_data in data.items():
                if sheet_data:
                    df = pd.DataFrame([self._flatten_dict(item) for item in sheet_data])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        buffer.seek(0)
        return buffer.read()
    
    def export_history_to_excel(self, history: List[Dict]) -> bytes:
        """
        Exporta historial completo a Excel
        
        Args:
            history: Lista de registros de historial
            
        Returns:
            Bytes del archivo Excel
        """
        if not self.pandas_available:
            raise ImportError("pandas no está instalado")
        
        # Preparar datos
        analyses_data = []
        conditions_data = []
        scores_data = []
        
        for record in history:
            # Datos de análisis
            analyses_data.append({
                "id": record.get("id"),
                "user_id": record.get("user_id"),
                "timestamp": record.get("timestamp"),
                "skin_type": record.get("skin_type"),
                "overall_score": record.get("quality_scores", {}).get("overall_score", 0)
            })
            
            # Datos de condiciones
            for condition in record.get("conditions", []):
                conditions_data.append({
                    "analysis_id": record.get("id"),
                    "condition_name": condition.get("name"),
                    "severity": condition.get("severity"),
                    "confidence": condition.get("confidence"),
                    "affected_area": condition.get("affected_area_percentage", 0)
                })
            
            # Datos de scores
            scores = record.get("quality_scores", {})
            scores_data.append({
                "analysis_id": record.get("id"),
                "timestamp": record.get("timestamp"),
                **scores
            })
        
        # Crear Excel con múltiples hojas
        data = {
            "Analyses": analyses_data,
            "Conditions": conditions_data,
            "Scores": scores_data
        }
        
        return self.export_to_excel(data)
    
    def export_analytics_to_excel(self, analytics_data: Dict) -> bytes:
        """
        Exporta analytics a Excel
        
        Args:
            analytics_data: Datos de analytics
            
        Returns:
            Bytes del archivo Excel
        """
        if not self.pandas_available:
            raise ImportError("pandas no está instalado")
        
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Hoja de estadísticas
            if "statistics" in analytics_data:
                stats_df = pd.DataFrame([analytics_data["statistics"]])
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)
            
            # Hoja de tendencias
            if "trend" in analytics_data:
                trend_df = pd.DataFrame([analytics_data["trend"]])
                trend_df.to_excel(writer, sheet_name="Trend", index=False)
            
            # Hoja de condiciones
            if "most_common_conditions" in analytics_data:
                conditions_df = pd.DataFrame(analytics_data["most_common_conditions"])
                conditions_df.to_excel(writer, sheet_name="Conditions", index=False)
        
        buffer.seek(0)
        return buffer.read()
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: '_') -> Dict:
        """
        Aplana un diccionario anidado
        
        Args:
            d: Diccionario a aplanar
            parent_key: Clave padre
            sep: Separador
            
        Returns:
            Diccionario aplanado
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convertir listas a strings
                items.append((new_key, json.dumps(v) if v else ""))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_comparison_to_csv(self, comparison: Dict) -> bytes:
        """
        Exporta comparación a CSV
        
        Args:
            comparison: Datos de comparación
            
        Returns:
            Bytes del archivo CSV
        """
        data = []
        
        score_diffs = comparison.get("score_differences", {})
        for metric, diff in score_diffs.items():
            data.append({
                "metric": metric,
                "before": diff.get("before", 0),
                "after": diff.get("after", 0),
                "difference": diff.get("difference", 0),
                "improvement": diff.get("improvement", False)
            })
        
        return self.export_to_csv(data)






