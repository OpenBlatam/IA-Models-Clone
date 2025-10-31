"""
Advanced Trend Analysis and Prediction System for AI History Comparison
Sistema avanzado de análisis de tendencias y predicciones para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendType(Enum):
    """Tipos de tendencias"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"
    VOLATILE = "volatile"
    BREAKTHROUGH = "breakthrough"
    DECLINE = "decline"

class PredictionType(Enum):
    """Tipos de predicciones"""
    SHORT_TERM = "short_term"  # 1-7 días
    MEDIUM_TERM = "medium_term"  # 1-4 semanas
    LONG_TERM = "long_term"  # 1-12 meses
    SEASONAL = "seasonal"
    CUSTOM = "custom"

class ConfidenceLevel(Enum):
    """Niveles de confianza"""
    VERY_HIGH = "very_high"  # > 90%
    HIGH = "high"  # 80-90%
    MEDIUM = "medium"  # 60-80%
    LOW = "low"  # 40-60%
    VERY_LOW = "very_low"  # < 40%

@dataclass
class TrendData:
    """Datos de tendencia"""
    metric_name: str
    values: List[float]
    timestamps: List[datetime]
    trend_type: TrendType
    slope: float
    r_squared: float
    p_value: float
    confidence: float
    volatility: float
    seasonality: Optional[Dict[str, float]] = None

@dataclass
class Prediction:
    """Predicción"""
    metric_name: str
    prediction_type: PredictionType
    predicted_values: List[float]
    prediction_dates: List[datetime]
    confidence_level: ConfidenceLevel
    confidence_score: float
    upper_bound: List[float]
    lower_bound: List[float]
    model_used: str
    accuracy_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TrendInsight:
    """Insight de tendencia"""
    id: str
    title: str
    description: str
    trend_type: TrendType
    impact_level: str
    confidence: float
    actionable_recommendations: List[str]
    related_metrics: List[str]
    timeframe: str
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedTrendAnalyzer:
    """
    Analizador avanzado de tendencias y predicciones
    """
    
    def __init__(
        self,
        enable_ml_predictions: bool = True,
        enable_seasonality_analysis: bool = True,
        enable_anomaly_detection: bool = True,
        prediction_horizon_days: int = 30
    ):
        self.enable_ml_predictions = enable_ml_predictions
        self.enable_seasonality_analysis = enable_seasonality_analysis
        self.enable_anomaly_detection = enable_anomaly_detection
        self.prediction_horizon_days = prediction_horizon_days
        
        # Almacenamiento de datos
        self.trend_data: Dict[str, TrendData] = {}
        self.predictions: Dict[str, Prediction] = {}
        self.insights: Dict[str, TrendInsight] = {}
        
        # Modelos de predicción
        self.prediction_models: Dict[str, Any] = {}
        
        # Configuración
        self.config = {
            "min_data_points": 10,
            "trend_threshold": 0.1,
            "confidence_threshold": 0.05,
            "seasonality_periods": [7, 30, 365],  # días
            "anomaly_threshold": 2.0,  # desviaciones estándar
            "prediction_intervals": [1, 7, 30, 90, 365]  # días
        }
        
        # Métricas a analizar
        self.metrics_to_analyze = [
            "quality_score",
            "readability_score",
            "engagement_score",
            "completion_rate",
            "user_satisfaction",
            "content_volume",
            "processing_time",
            "error_rate"
        ]
    
    async def analyze_trends(
        self,
        historical_data: Dict[str, List[Dict[str, Any]]],
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, TrendData]:
        """
        Analizar tendencias en datos históricos
        
        Args:
            historical_data: Datos históricos por métrica
            time_range: Rango de tiempo para análisis
            
        Returns:
            Análisis de tendencias por métrica
        """
        try:
            logger.info("Starting trend analysis")
            
            trends = {}
            
            for metric_name, data_points in historical_data.items():
                if len(data_points) < self.config["min_data_points"]:
                    logger.warning(f"Insufficient data for metric {metric_name}")
                    continue
                
                # Filtrar por rango de tiempo si se especifica
                if time_range:
                    filtered_data = [
                        dp for dp in data_points
                        if time_range[0] <= dp["timestamp"] <= time_range[1]
                    ]
                else:
                    filtered_data = data_points
                
                if len(filtered_data) < self.config["min_data_points"]:
                    continue
                
                # Extraer valores y timestamps
                values = [dp["value"] for dp in filtered_data]
                timestamps = [dp["timestamp"] for dp in filtered_data]
                
                # Analizar tendencia
                trend = await self._analyze_single_trend(metric_name, values, timestamps)
                trends[metric_name] = trend
                
                # Almacenar datos de tendencia
                self.trend_data[metric_name] = trend
            
            logger.info(f"Trend analysis completed for {len(trends)} metrics")
            return trends
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            raise
    
    async def _analyze_single_trend(
        self,
        metric_name: str,
        values: List[float],
        timestamps: List[datetime]
    ) -> TrendData:
        """Analizar tendencia para una métrica específica"""
        
        # Convertir timestamps a números para análisis
        time_numeric = [(ts - timestamps[0]).total_seconds() / 86400 for ts in timestamps]  # días
        
        # Análisis de regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        r_squared = r_value ** 2
        
        # Determinar tipo de tendencia
        trend_type = self._determine_trend_type(slope, r_squared, p_value, values)
        
        # Calcular volatilidad
        volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        # Análisis de estacionalidad
        seasonality = None
        if self.enable_seasonality_analysis:
            seasonality = await self._analyze_seasonality(values, timestamps)
        
        # Calcular confianza
        confidence = self._calculate_confidence(r_squared, p_value, len(values))
        
        return TrendData(
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            trend_type=trend_type,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            confidence=confidence,
            volatility=volatility,
            seasonality=seasonality
        )
    
    def _determine_trend_type(
        self,
        slope: float,
        r_squared: float,
        p_value: float,
        values: List[float]
    ) -> TrendType:
        """Determinar tipo de tendencia"""
        
        # Verificar significancia estadística
        if p_value > self.config["confidence_threshold"]:
            return TrendType.STABLE
        
        # Verificar fuerza de la tendencia
        if r_squared < 0.3:
            return TrendType.VOLATILE
        
        # Determinar dirección
        if abs(slope) < self.config["trend_threshold"]:
            return TrendType.STABLE
        elif slope > 0:
            # Verificar si es un breakthrough
            if self._is_breakthrough(values):
                return TrendType.BREAKTHROUGH
            else:
                return TrendType.INCREASING
        else:
            # Verificar si es un decline
            if self._is_decline(values):
                return TrendType.DECLINE
            else:
                return TrendType.DECREASING
    
    def _is_breakthrough(self, values: List[float]) -> bool:
        """Detectar si hay un breakthrough"""
        if len(values) < 5:
            return False
        
        # Verificar si los últimos valores son significativamente mayores
        recent_avg = np.mean(values[-3:])
        historical_avg = np.mean(values[:-3])
        
        return recent_avg > historical_avg * 1.5
    
    def _is_decline(self, values: List[float]) -> bool:
        """Detectar si hay un decline"""
        if len(values) < 5:
            return False
        
        # Verificar si los últimos valores son significativamente menores
        recent_avg = np.mean(values[-3:])
        historical_avg = np.mean(values[:-3])
        
        return recent_avg < historical_avg * 0.5
    
    async def _analyze_seasonality(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, float]:
        """Analizar estacionalidad en los datos"""
        seasonality = {}
        
        for period in self.config["seasonality_periods"]:
            # Calcular autocorrelación para el período
            if len(values) >= period * 2:
                autocorr = self._calculate_autocorrelation(values, period)
                seasonality[f"period_{period}_days"] = autocorr
        
        return seasonality
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calcular autocorrelación para un lag específico"""
        if len(values) <= lag:
            return 0.0
        
        # Calcular correlación entre valores y valores desplazados
        x = values[:-lag]
        y = values[lag:]
        
        if len(x) == 0 or len(y) == 0:
            return 0.0
        
        correlation, _ = stats.pearsonr(x, y)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_confidence(self, r_squared: float, p_value: float, sample_size: int) -> float:
        """Calcular nivel de confianza"""
        # Factor de tamaño de muestra
        sample_factor = min(1.0, sample_size / 50)
        
        # Factor de significancia estadística
        significance_factor = 1.0 - p_value
        
        # Factor de bondad de ajuste
        fit_factor = r_squared
        
        # Confianza combinada
        confidence = (significance_factor * 0.4 + fit_factor * 0.4 + sample_factor * 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    async def generate_predictions(
        self,
        metric_name: str,
        prediction_type: PredictionType,
        custom_days: Optional[int] = None
    ) -> Prediction:
        """
        Generar predicciones para una métrica
        
        Args:
            metric_name: Nombre de la métrica
            prediction_type: Tipo de predicción
            custom_days: Días personalizados para predicción
            
        Returns:
            Predicción generada
        """
        try:
            if metric_name not in self.trend_data:
                raise ValueError(f"No trend data available for metric {metric_name}")
            
            trend_data = self.trend_data[metric_name]
            
            # Determinar horizonte de predicción
            prediction_days = self._get_prediction_days(prediction_type, custom_days)
            
            # Generar predicciones
            predicted_values, prediction_dates, model_used = await self._generate_prediction_values(
                trend_data, prediction_days
            )
            
            # Calcular intervalos de confianza
            upper_bound, lower_bound = self._calculate_confidence_intervals(
                predicted_values, trend_data.confidence
            )
            
            # Determinar nivel de confianza
            confidence_level = self._determine_confidence_level(trend_data.confidence)
            
            # Calcular métricas de precisión (usando datos históricos)
            accuracy_metrics = self._calculate_accuracy_metrics(trend_data)
            
            # Crear predicción
            prediction = Prediction(
                metric_name=metric_name,
                prediction_type=prediction_type,
                predicted_values=predicted_values,
                prediction_dates=prediction_dates,
                confidence_level=confidence_level,
                confidence_score=trend_data.confidence,
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                model_used=model_used,
                accuracy_metrics=accuracy_metrics
            )
            
            # Almacenar predicción
            prediction_id = f"{metric_name}_{prediction_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.predictions[prediction_id] = prediction
            
            logger.info(f"Prediction generated for {metric_name}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            raise
    
    def _get_prediction_days(self, prediction_type: PredictionType, custom_days: Optional[int]) -> int:
        """Obtener número de días para predicción"""
        if prediction_type == PredictionType.SHORT_TERM:
            return 7
        elif prediction_type == PredictionType.MEDIUM_TERM:
            return 30
        elif prediction_type == PredictionType.LONG_TERM:
            return 365
        elif prediction_type == PredictionType.SEASONAL:
            return 365
        elif prediction_type == PredictionType.CUSTOM:
            return custom_days or 30
        else:
            return 30
    
    async def _generate_prediction_values(
        self,
        trend_data: TrendData,
        prediction_days: int
    ) -> Tuple[List[float], List[datetime], str]:
        """Generar valores de predicción"""
        
        # Preparar datos para modelado
        time_numeric = [(ts - trend_data.timestamps[0]).total_seconds() / 86400 for ts in trend_data.timestamps]
        X = np.array(time_numeric).reshape(-1, 1)
        y = np.array(trend_data.values)
        
        # Seleccionar modelo basado en el tipo de tendencia
        if trend_data.trend_type in [TrendType.INCREASING, TrendType.DECREASING, TrendType.STABLE]:
            # Modelo lineal
            model = LinearRegression()
            model.fit(X, y)
            model_name = "linear_regression"
        else:
            # Modelo polinomial para tendencias complejas
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            model_name = "polynomial_regression"
        
        # Generar fechas futuras
        last_date = trend_data.timestamps[-1]
        prediction_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
        
        # Generar predicciones
        future_time_numeric = [(date - trend_data.timestamps[0]).total_seconds() / 86400 for date in prediction_dates]
        X_future = np.array(future_time_numeric).reshape(-1, 1)
        
        if model_name == "polynomial_regression":
            X_future_poly = poly_features.transform(X_future)
            predicted_values = model.predict(X_future_poly).tolist()
        else:
            predicted_values = model.predict(X_future).tolist()
        
        return predicted_values, prediction_dates, model_name
    
    def _calculate_confidence_intervals(
        self,
        predicted_values: List[float],
        confidence: float
    ) -> Tuple[List[float], List[float]]:
        """Calcular intervalos de confianza"""
        # Calcular desviación estándar basada en la confianza
        std_dev = (1 - confidence) * np.mean(predicted_values) * 0.1
        
        upper_bound = [val + 1.96 * std_dev for val in predicted_values]
        lower_bound = [val - 1.96 * std_dev for val in predicted_values]
        
        return upper_bound, lower_bound
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determinar nivel de confianza"""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_accuracy_metrics(self, trend_data: TrendData) -> Dict[str, float]:
        """Calcular métricas de precisión del modelo"""
        # Usar validación cruzada simple
        if len(trend_data.values) < 10:
            return {"r_squared": trend_data.r_squared, "p_value": trend_data.p_value}
        
        # Dividir datos en entrenamiento y prueba
        split_point = len(trend_data.values) // 2
        train_values = trend_data.values[:split_point]
        test_values = trend_data.values[split_point:]
        
        train_times = [(ts - trend_data.timestamps[0]).total_seconds() / 86400 for ts in trend_data.timestamps[:split_point]]
        test_times = [(ts - trend_data.timestamps[0]).total_seconds() / 86400 for ts in trend_data.timestamps[split_point:]]
        
        # Entrenar modelo
        X_train = np.array(train_times).reshape(-1, 1)
        y_train = np.array(train_values)
        X_test = np.array(test_times).reshape(-1, 1)
        y_test = np.array(test_values)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predecir y calcular métricas
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "r_squared": r2,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "p_value": trend_data.p_value
        }
    
    async def generate_insights(self, trends: Dict[str, TrendData]) -> List[TrendInsight]:
        """Generar insights basados en las tendencias"""
        insights = []
        
        for metric_name, trend in trends.items():
            # Generar insight para cada tendencia significativa
            if trend.confidence > 0.6:
                insight = await self._create_trend_insight(metric_name, trend)
                insights.append(insight)
                self.insights[insight.id] = insight
        
        # Generar insights comparativos
        comparative_insights = await self._generate_comparative_insights(trends)
        insights.extend(comparative_insights)
        
        return insights
    
    async def _create_trend_insight(self, metric_name: str, trend: TrendData) -> TrendInsight:
        """Crear insight para una tendencia específica"""
        
        # Generar título y descripción
        title, description = self._generate_insight_content(metric_name, trend)
        
        # Determinar nivel de impacto
        impact_level = self._determine_impact_level(trend)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(metric_name, trend)
        
        # Crear insight
        insight_id = f"insight_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TrendInsight(
            id=insight_id,
            title=title,
            description=description,
            trend_type=trend.trend_type,
            impact_level=impact_level,
            confidence=trend.confidence,
            actionable_recommendations=recommendations,
            related_metrics=[metric_name],
            timeframe=f"Últimos {len(trend.values)} puntos de datos"
        )
    
    def _generate_insight_content(self, metric_name: str, trend: TrendData) -> Tuple[str, str]:
        """Generar contenido del insight"""
        
        trend_descriptions = {
            TrendType.INCREASING: "tendencia creciente",
            TrendType.DECREASING: "tendencia decreciente",
            TrendType.STABLE: "tendencia estable",
            TrendType.CYCLICAL: "patrón cíclico",
            TrendType.SEASONAL: "patrón estacional",
            TrendType.VOLATILE: "alta volatilidad",
            TrendType.BREAKTHROUGH: "breakthrough significativo",
            TrendType.DECLINE: "decline preocupante"
        }
        
        title = f"{metric_name.replace('_', ' ').title()}: {trend_descriptions[trend.trend_type]}"
        
        description = f"La métrica {metric_name} muestra una {trend_descriptions[trend.trend_type]} "
        description += f"con una confianza del {trend.confidence:.1%}. "
        
        if trend.slope != 0:
            direction = "aumentando" if trend.slope > 0 else "disminuyendo"
            description += f"El valor está {direction} a una tasa de {abs(trend.slope):.3f} por día. "
        
        if trend.volatility > 0.2:
            description += f"Se observa alta volatilidad ({trend.volatility:.1%}). "
        
        return title, description
    
    def _determine_impact_level(self, trend: TrendData) -> str:
        """Determinar nivel de impacto de la tendencia"""
        if trend.trend_type in [TrendType.BREAKTHROUGH, TrendType.DECLINE]:
            return "high"
        elif trend.confidence > 0.8 and abs(trend.slope) > 0.1:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, metric_name: str, trend: TrendData) -> List[str]:
        """Generar recomendaciones basadas en la tendencia"""
        recommendations = []
        
        if trend.trend_type == TrendType.INCREASING:
            recommendations.extend([
                "Monitorea el crecimiento para evitar sobrecarga",
                "Considera escalar recursos si es necesario",
                "Documenta las mejores prácticas que están funcionando"
            ])
        elif trend.trend_type == TrendType.DECREASING:
            recommendations.extend([
                "Investiga las causas del declive",
                "Implementa medidas correctivas inmediatas",
                "Revisa los procesos y procedimientos actuales"
            ])
        elif trend.trend_type == TrendType.VOLATILE:
            recommendations.extend([
                "Analiza los factores que causan la volatilidad",
                "Implementa controles de estabilidad",
                "Considera promedios móviles para suavizar las fluctuaciones"
            ])
        elif trend.trend_type == TrendType.BREAKTHROUGH:
            recommendations.extend([
                "Capitaliza el momentum positivo",
                "Documenta los factores de éxito",
                "Considera expandir las iniciativas exitosas"
            ])
        elif trend.trend_type == TrendType.DECLINE:
            recommendations.extend([
                "Actúa inmediatamente para revertir la tendencia",
                "Identifica y elimina las causas raíz",
                "Implementa un plan de recuperación urgente"
            ])
        
        return recommendations
    
    async def _generate_comparative_insights(self, trends: Dict[str, TrendData]) -> List[TrendInsight]:
        """Generar insights comparativos entre métricas"""
        insights = []
        
        # Comparar métricas relacionadas
        metric_pairs = [
            ("quality_score", "user_satisfaction"),
            ("readability_score", "completion_rate"),
            ("content_volume", "processing_time"),
            ("engagement_score", "error_rate")
        ]
        
        for metric1, metric2 in metric_pairs:
            if metric1 in trends and metric2 in trends:
                insight = await self._create_comparative_insight(metric1, metric2, trends[metric1], trends[metric2])
                if insight:
                    insights.append(insight)
                    self.insights[insight.id] = insight
        
        return insights
    
    async def _create_comparative_insight(
        self,
        metric1: str,
        metric2: str,
        trend1: TrendData,
        trend2: TrendData
    ) -> Optional[TrendInsight]:
        """Crear insight comparativo entre dos métricas"""
        
        # Verificar si hay correlación significativa
        correlation = np.corrcoef(trend1.values, trend2.values)[0, 1]
        
        if abs(correlation) < 0.5:
            return None
        
        # Determinar tipo de correlación
        if correlation > 0.7:
            relationship = "correlación positiva fuerte"
            impact = "high"
        elif correlation > 0.5:
            relationship = "correlación positiva moderada"
            impact = "medium"
        elif correlation < -0.7:
            relationship = "correlación negativa fuerte"
            impact = "high"
        elif correlation < -0.5:
            relationship = "correlación negativa moderada"
            impact = "medium"
        else:
            return None
        
        title = f"Correlación entre {metric1.replace('_', ' ')} y {metric2.replace('_', ' ')}"
        description = f"Se observa una {relationship} (r={correlation:.2f}) entre {metric1} y {metric2}. "
        description += f"Esto sugiere que los cambios en una métrica están relacionados con cambios en la otra."
        
        recommendations = [
            f"Monitorea ambas métricas en conjunto",
            f"Considera el impacto de cambios en {metric1} sobre {metric2}",
            f"Optimiza los procesos que afectan ambas métricas"
        ]
        
        insight_id = f"comparative_{metric1}_{metric2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return TrendInsight(
            id=insight_id,
            title=title,
            description=description,
            trend_type=TrendType.STABLE,  # Placeholder
            impact_level=impact,
            confidence=abs(correlation),
            actionable_recommendations=recommendations,
            related_metrics=[metric1, metric2],
            timeframe="Análisis comparativo"
        )
    
    async def get_trend_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis de tendencias"""
        if not self.trend_data:
            return {"message": "No trend data available"}
        
        # Estadísticas generales
        total_metrics = len(self.trend_data)
        trend_types = Counter([trend.trend_type.value for trend in self.trend_data.values()])
        confidence_levels = Counter([
            self._determine_confidence_level(trend.confidence).value 
            for trend in self.trend_data.values()
        ])
        
        # Métricas promedio
        avg_confidence = np.mean([trend.confidence for trend in self.trend_data.values()])
        avg_volatility = np.mean([trend.volatility for trend in self.trend_data.values()])
        
        return {
            "total_metrics_analyzed": total_metrics,
            "trend_type_distribution": dict(trend_types),
            "confidence_level_distribution": dict(confidence_levels),
            "average_confidence": avg_confidence,
            "average_volatility": avg_volatility,
            "total_predictions": len(self.predictions),
            "total_insights": len(self.insights),
            "last_analysis": max([trend.timestamps[-1] for trend in self.trend_data.values()]).isoformat()
        }
    
    async def export_trend_analysis(self, filepath: str = None) -> str:
        """Exportar análisis de tendencias"""
        if filepath is None:
            filepath = f"exports/trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Preparar datos para exportación
        export_data = {
            "trend_data": {
                name: {
                    "metric_name": trend.metric_name,
                    "trend_type": trend.trend_type.value,
                    "slope": trend.slope,
                    "r_squared": trend.r_squared,
                    "p_value": trend.p_value,
                    "confidence": trend.confidence,
                    "volatility": trend.volatility,
                    "values": trend.values,
                    "timestamps": [ts.isoformat() for ts in trend.timestamps]
                }
                for name, trend in self.trend_data.items()
            },
            "predictions": {
                name: {
                    "metric_name": pred.metric_name,
                    "prediction_type": pred.prediction_type.value,
                    "confidence_level": pred.confidence_level.value,
                    "confidence_score": pred.confidence_score,
                    "model_used": pred.model_used,
                    "predicted_values": pred.predicted_values,
                    "prediction_dates": [date.isoformat() for date in pred.prediction_dates],
                    "accuracy_metrics": pred.accuracy_metrics
                }
                for name, pred in self.predictions.items()
            },
            "insights": {
                name: {
                    "title": insight.title,
                    "description": insight.description,
                    "trend_type": insight.trend_type.value,
                    "impact_level": insight.impact_level,
                    "confidence": insight.confidence,
                    "actionable_recommendations": insight.actionable_recommendations,
                    "related_metrics": insight.related_metrics,
                    "timeframe": insight.timeframe
                }
                for name, insight in self.insights.items()
            },
            "summary": await self.get_trend_summary(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Trend analysis exported to {filepath}")
        return filepath

























