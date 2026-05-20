"""
Servicio de Visualizaciones - Generación de gráficos y visualizaciones avanzadas
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json


class VisualizationService:
    """Servicio de visualizaciones y gráficos"""
    
    def __init__(self):
        """Inicializa el servicio de visualizaciones"""
        pass
    
    def generate_progress_chart(
        self,
        user_id: str,
        data: List[Dict],
        chart_type: str = "line"
    ) -> Dict:
        """
        Genera gráfico de progreso
        
        Args:
            user_id: ID del usuario
            data: Datos de progreso
            chart_type: Tipo de gráfico (line, bar, area)
        
        Returns:
            Datos del gráfico
        """
        chart_data = {
            "user_id": user_id,
            "chart_type": chart_type,
            "labels": [],
            "datasets": [
                {
                    "label": "Días Sobrios",
                    "data": [],
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "borderColor": "rgba(75, 192, 192, 1)"
                },
                {
                    "label": "Cravings",
                    "data": [],
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "borderColor": "rgba(255, 99, 132, 1)"
                }
            ],
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        }
        
        # Procesar datos
        for entry in data:
            chart_data["labels"].append(entry.get("date", ""))
            chart_data["datasets"][0]["data"].append(1 if not entry.get("consumed", False) else 0)
            chart_data["datasets"][1]["data"].append(entry.get("cravings_level", 0))
        
        return chart_data
    
    def generate_heatmap(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "cravings_level"
    ) -> Dict:
        """
        Genera mapa de calor de actividad
        
        Args:
            user_id: ID del usuario
            data: Datos de entrada
            metric: Métrica a visualizar
        
        Returns:
            Datos del mapa de calor
        """
        heatmap_data = {
            "user_id": user_id,
            "type": "heatmap",
            "data": [],
            "color_scale": {
                "min": 0,
                "max": 10,
                "colors": ["#ebedf0", "#c6e48b", "#7bc96f", "#239a3b", "#196127"]
            }
        }
        
        # Agrupar por día de la semana y hora
        for entry in data:
            date_str = entry.get("date", "")
            try:
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                day_of_week = date.weekday()
                hour = date.hour
                value = entry.get(metric, 0)
                
                heatmap_data["data"].append({
                    "day": day_of_week,
                    "hour": hour,
                    "value": value
                })
            except:
                continue
        
        return heatmap_data
    
    def generate_radar_chart(
        self,
        user_id: str,
        metrics: Dict
    ) -> Dict:
        """
        Genera gráfico de radar con múltiples métricas
        
        Args:
            user_id: ID del usuario
            metrics: Diccionario con métricas
        
        Returns:
            Datos del gráfico de radar
        """
        radar_data = {
            "user_id": user_id,
            "type": "radar",
            "labels": [
                "Sobriedad",
                "Salud Física",
                "Salud Mental",
                "Apoyo Social",
                "Motivación",
                "Estrategias"
            ],
            "datasets": [
                {
                    "label": "Estado Actual",
                    "data": [
                        metrics.get("sobriety_score", 0),
                        metrics.get("physical_health", 0),
                        metrics.get("mental_health", 0),
                        metrics.get("social_support", 0),
                        metrics.get("motivation", 0),
                        metrics.get("strategies", 0)
                    ],
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)"
                }
            ],
            "options": {
                "scale": {
                    "min": 0,
                    "max": 100
                }
            }
        }
        
        return radar_data
    
    def generate_timeline_visualization(
        self,
        user_id: str,
        events: List[Dict]
    ) -> Dict:
        """
        Genera visualización de línea de tiempo
        
        Args:
            user_id: ID del usuario
            events: Lista de eventos
        
        Returns:
            Datos de línea de tiempo
        """
        timeline = {
            "user_id": user_id,
            "type": "timeline",
            "events": []
        }
        
        for event in events:
            timeline["events"].append({
                "date": event.get("date"),
                "title": event.get("title"),
                "description": event.get("description"),
                "type": event.get("type", "event"),
                "icon": event.get("icon", "●")
            })
        
        return timeline
    
    def generate_comparison_chart(
        self,
        user_id: str,
        period1_data: List[Dict],
        period2_data: List[Dict],
        metric: str = "success_rate"
    ) -> Dict:
        """
        Genera gráfico de comparación entre períodos
        
        Args:
            user_id: ID del usuario
            period1_data: Datos del primer período
            period2_data: Datos del segundo período
            metric: Métrica a comparar
        
        Returns:
            Datos del gráfico de comparación
        """
        comparison = {
            "user_id": user_id,
            "type": "comparison",
            "periods": [
                {
                    "label": "Período 1",
                    "data": [entry.get(metric, 0) for entry in period1_data]
                },
                {
                    "label": "Período 2",
                    "data": [entry.get(metric, 0) for entry in period2_data]
                }
            ],
            "summary": {
                "period1_avg": sum(entry.get(metric, 0) for entry in period1_data) / len(period1_data) if period1_data else 0,
                "period2_avg": sum(entry.get(metric, 0) for entry in period2_data) / len(period2_data) if period2_data else 0,
                "improvement": 0
            }
        }
        
        if comparison["summary"]["period1_avg"] > 0:
            improvement = ((comparison["summary"]["period2_avg"] - comparison["summary"]["period1_avg"]) / comparison["summary"]["period1_avg"]) * 100
            comparison["summary"]["improvement"] = round(improvement, 2)
        
        return comparison

