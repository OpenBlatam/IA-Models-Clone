"""
Advanced Temporal Analysis System for AI History Comparison
Sistema avanzado de análisis temporal para análisis de historial de IA
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
import math
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Time series analysis
from scipy import stats
from scipy.signal import find_peaks
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing import ExponentialSmoothing

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
    BREAKING = "breaking"
    CONVERGING = "converging"
    DIVERGING = "diverging"

class PatternType(Enum):
    """Tipos de patrones"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POLYNOMIAL = "polynomial"
    SINUSOIDAL = "sinusoidal"
    STEP = "step"
    SPIKE = "spike"
    PLATEAU = "plateau"
    OSCILLATION = "oscillation"

class AnomalyType(Enum):
    """Tipos de anomalías"""
    OUTLIER = "outlier"
    TREND_BREAK = "trend_break"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    VOLATILITY_SPIKE = "volatility_spike"
    LEVEL_SHIFT = "level_shift"
    STRUCTURAL_BREAK = "structural_break"

@dataclass
class TemporalPoint:
    """Punto temporal"""
    timestamp: datetime
    value: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Análisis de tendencia"""
    id: str
    metric_name: str
    trend_type: TrendType
    pattern_type: PatternType
    slope: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    start_date: datetime
    end_date: datetime
    data_points: List[TemporalPoint]
    anomalies: List[Dict[str, Any]]
    seasonality: Optional[Dict[str, Any]]
    forecast: Optional[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TemporalInsight:
    """Insight temporal"""
    id: str
    insight_type: str
    description: str
    significance: float
    confidence: float
    timeframe: Tuple[datetime, datetime]
    related_metrics: List[str]
    implications: List[str]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedTemporalAnalyzer:
    """
    Analizador temporal avanzado para análisis de historial de IA
    """
    
    def __init__(
        self,
        enable_forecasting: bool = True,
        enable_anomaly_detection: bool = True,
        enable_seasonality_analysis: bool = True,
        enable_correlation_analysis: bool = True
    ):
        self.enable_forecasting = enable_forecasting
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_seasonality_analysis = enable_seasonality_analysis
        self.enable_correlation_analysis = enable_correlation_analysis
        
        # Almacenamiento de análisis
        self.trend_analyses: Dict[str, TrendAnalysis] = {}
        self.temporal_insights: Dict[str, TemporalInsight] = {}
        self.temporal_data: Dict[str, List[TemporalPoint]] = {}
        
        # Configuración
        self.config = {
            "min_data_points": 10,
            "max_forecast_periods": 30,
            "anomaly_threshold": 2.0,  # Z-score threshold
            "seasonality_periods": [7, 30, 365],  # Daily, monthly, yearly
            "correlation_threshold": 0.7,
            "trend_significance_threshold": 0.05
        }
    
    async def add_temporal_data(
        self,
        metric_name: str,
        data_points: List[TemporalPoint],
        replace_existing: bool = False
    ) -> bool:
        """
        Agregar datos temporales para análisis
        
        Args:
            metric_name: Nombre de la métrica
            data_points: Puntos de datos temporales
            replace_existing: Si reemplazar datos existentes
            
        Returns:
            True si se agregaron exitosamente
        """
        try:
            if len(data_points) < self.config["min_data_points"]:
                logger.warning(f"Insufficient data points for {metric_name}: {len(data_points)}")
                return False
            
            # Validar y ordenar datos
            validated_points = []
            for point in data_points:
                if isinstance(point.timestamp, str):
                    point.timestamp = datetime.fromisoformat(point.timestamp)
                validated_points.append(point)
            
            # Ordenar por timestamp
            validated_points.sort(key=lambda x: x.timestamp)
            
            if replace_existing or metric_name not in self.temporal_data:
                self.temporal_data[metric_name] = validated_points
            else:
                self.temporal_data[metric_name].extend(validated_points)
                self.temporal_data[metric_name].sort(key=lambda x: x.timestamp)
            
            logger.info(f"Added {len(validated_points)} data points for metric {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding temporal data for {metric_name}: {e}")
            return False
    
    async def analyze_trends(
        self,
        metric_name: str,
        include_forecast: bool = True,
        include_anomalies: bool = True
    ) -> TrendAnalysis:
        """
        Analizar tendencias temporales
        
        Args:
            metric_name: Nombre de la métrica
            include_forecast: Si incluir pronóstico
            include_anomalies: Si incluir detección de anomalías
            
        Returns:
            Análisis de tendencia
        """
        try:
            if metric_name not in self.temporal_data:
                raise ValueError(f"No data available for metric {metric_name}")
            
            data_points = self.temporal_data[metric_name]
            logger.info(f"Analyzing trends for {metric_name} with {len(data_points)} data points")
            
            # Extraer timestamps y valores
            timestamps = [point.timestamp for point in data_points]
            values = [point.value for point in data_points]
            
            # Convertir timestamps a números para análisis
            time_numeric = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]  # Horas
            
            # Análisis de tendencia básico
            trend_type, pattern_type, slope, r_squared, p_value = await self._analyze_basic_trend(
                time_numeric, values
            )
            
            # Calcular intervalo de confianza
            confidence_interval = await self._calculate_confidence_interval(
                time_numeric, values, slope
            )
            
            # Detectar anomalías
            anomalies = []
            if include_anomalies and self.enable_anomaly_detection:
                anomalies = await self._detect_anomalies(values, timestamps)
            
            # Análisis de estacionalidad
            seasonality = None
            if self.enable_seasonality_analysis:
                seasonality = await self._analyze_seasonality(values, timestamps)
            
            # Pronóstico
            forecast = None
            if include_forecast and self.enable_forecasting:
                forecast = await self._generate_forecast(values, timestamps)
            
            # Crear análisis de tendencia
            analysis = TrendAnalysis(
                id=f"trend_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metric_name=metric_name,
                trend_type=trend_type,
                pattern_type=pattern_type,
                slope=slope,
                r_squared=r_squared,
                p_value=p_value,
                confidence_interval=confidence_interval,
                start_date=timestamps[0],
                end_date=timestamps[-1],
                data_points=data_points,
                anomalies=anomalies,
                seasonality=seasonality,
                forecast=forecast
            )
            
            # Almacenar análisis
            self.trend_analyses[analysis.id] = analysis
            
            logger.info(f"Trend analysis completed for {metric_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends for {metric_name}: {e}")
            raise
    
    async def _analyze_basic_trend(
        self,
        time_numeric: List[float],
        values: List[float]
    ) -> Tuple[TrendType, PatternType, float, float, float]:
        """Analizar tendencia básica"""
        try:
            # Regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            r_squared = r_value ** 2
            
            # Determinar tipo de tendencia
            if p_value < self.config["trend_significance_threshold"]:
                if slope > 0:
                    trend_type = TrendType.INCREASING
                else:
                    trend_type = TrendType.DECREASING
            else:
                trend_type = TrendType.STABLE
            
            # Determinar tipo de patrón
            pattern_type = await self._determine_pattern_type(time_numeric, values)
            
            return trend_type, pattern_type, slope, r_squared, p_value
            
        except Exception as e:
            logger.error(f"Error in basic trend analysis: {e}")
            return TrendType.STABLE, PatternType.LINEAR, 0.0, 0.0, 1.0
    
    async def _determine_pattern_type(
        self,
        time_numeric: List[float],
        values: List[float]
    ) -> PatternType:
        """Determinar tipo de patrón"""
        try:
            # Probar diferentes tipos de patrones
            patterns = {}
            
            # Patrón lineal
            slope, _, r_value, _, _ = stats.linregress(time_numeric, values)
            patterns[PatternType.LINEAR] = r_value ** 2
            
            # Patrón exponencial
            try:
                log_values = [math.log(max(v, 0.001)) for v in values]
                _, _, r_value, _, _ = stats.linregress(time_numeric, log_values)
                patterns[PatternType.EXPONENTIAL] = r_value ** 2
            except:
                patterns[PatternType.EXPONENTIAL] = 0.0
            
            # Patrón logarítmico
            try:
                log_time = [math.log(max(t, 0.001)) for t in time_numeric]
                _, _, r_value, _, _ = stats.linregress(log_time, values)
                patterns[PatternType.LOGARITHMIC] = r_value ** 2
            except:
                patterns[PatternType.LOGARITHMIC] = 0.0
            
            # Patrón polinomial (cuadrático)
            try:
                poly_coeffs = np.polyfit(time_numeric, values, 2)
                poly_values = np.polyval(poly_coeffs, time_numeric)
                r_squared = 1 - (np.sum((values - poly_values) ** 2) / np.sum((values - np.mean(values)) ** 2))
                patterns[PatternType.POLYNOMIAL] = max(0, r_squared)
            except:
                patterns[PatternType.POLYNOMIAL] = 0.0
            
            # Patrón sinusoidal
            try:
                # Ajustar sinusoide
                def sin_func(t, a, b, c, d):
                    return a * np.sin(b * t + c) + d
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(sin_func, time_numeric, values, maxfev=1000)
                sin_values = sin_func(time_numeric, *popt)
                r_squared = 1 - (np.sum((values - sin_values) ** 2) / np.sum((values - np.mean(values)) ** 2))
                patterns[PatternType.SINUSOIDAL] = max(0, r_squared)
            except:
                patterns[PatternType.SINUSOIDAL] = 0.0
            
            # Retornar el patrón con mejor ajuste
            best_pattern = max(patterns.items(), key=lambda x: x[1])
            
            if best_pattern[1] > 0.7:
                return best_pattern[0]
            else:
                return PatternType.LINEAR
                
        except Exception as e:
            logger.error(f"Error determining pattern type: {e}")
            return PatternType.LINEAR
    
    async def _calculate_confidence_interval(
        self,
        time_numeric: List[float],
        values: List[float],
        slope: float
    ) -> Tuple[float, float]:
        """Calcular intervalo de confianza para la pendiente"""
        try:
            n = len(values)
            if n < 3:
                return (slope, slope)
            
            # Calcular error estándar
            residuals = [values[i] - (slope * time_numeric[i]) for i in range(n)]
            mse = sum(r**2 for r in residuals) / (n - 2)
            
            x_mean = sum(time_numeric) / n
            sxx = sum((x - x_mean)**2 for x in time_numeric)
            
            se_slope = math.sqrt(mse / sxx) if sxx > 0 else 0
            
            # Intervalo de confianza del 95%
            t_value = stats.t.ppf(0.975, n - 2)
            margin_error = t_value * se_slope
            
            return (slope - margin_error, slope + margin_error)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (slope, slope)
    
    async def _detect_anomalies(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> List[Dict[str, Any]]:
        """Detectar anomalías en los datos"""
        try:
            anomalies = []
            
            if len(values) < 5:
                return anomalies
            
            # Método 1: Z-score
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 0:
                z_scores = [(v - mean_val) / std_val for v in values]
                
                for i, z_score in enumerate(z_scores):
                    if abs(z_score) > self.config["anomaly_threshold"]:
                        anomalies.append({
                            "type": AnomalyType.OUTLIER.value,
                            "timestamp": timestamps[i].isoformat(),
                            "value": values[i],
                            "z_score": z_score,
                            "severity": "high" if abs(z_score) > 3 else "medium",
                            "method": "z_score"
                        })
            
            # Método 2: Detección de picos
            try:
                peaks, _ = find_peaks(values, height=np.mean(values) + np.std(values))
                valleys, _ = find_peaks([-v for v in values], height=-(np.mean(values) - np.std(values)))
                
                for peak_idx in peaks:
                    anomalies.append({
                        "type": AnomalyType.SPIKE.value,
                        "timestamp": timestamps[peak_idx].isoformat(),
                        "value": values[peak_idx],
                        "severity": "high",
                        "method": "peak_detection"
                    })
                
                for valley_idx in valleys:
                    anomalies.append({
                        "type": AnomalyType.SPIKE.value,
                        "timestamp": timestamps[valley_idx].isoformat(),
                        "value": values[valley_idx],
                        "severity": "high",
                        "method": "valley_detection"
                    })
            except:
                pass
            
            # Método 3: Cambios de nivel
            if len(values) > 10:
                window_size = max(3, len(values) // 10)
                for i in range(window_size, len(values) - window_size):
                    before_mean = np.mean(values[i-window_size:i])
                    after_mean = np.mean(values[i:i+window_size])
                    
                    if abs(after_mean - before_mean) > 2 * np.std(values):
                        anomalies.append({
                            "type": AnomalyType.LEVEL_SHIFT.value,
                            "timestamp": timestamps[i].isoformat(),
                            "value": values[i],
                            "before_mean": before_mean,
                            "after_mean": after_mean,
                            "severity": "medium",
                            "method": "level_shift"
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def _analyze_seasonality(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Analizar estacionalidad"""
        try:
            if len(values) < 20:
                return None
            
            # Crear serie temporal
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': values
            })
            df.set_index('timestamp', inplace=True)
            
            # Análisis de descomposición estacional
            try:
                decomposition = seasonal_decompose(
                    df['value'],
                    model='additive',
                    period=min(7, len(values) // 4)  # Período adaptativo
                )
                
                # Calcular fuerza de estacionalidad
                seasonal_strength = np.var(decomposition.seasonal) / np.var(df['value'])
                
                # Detectar períodos estacionales
                seasonal_periods = []
                for period in self.config["seasonality_periods"]:
                    if len(values) >= period * 2:
                        # Calcular autocorrelación
                        autocorr = df['value'].autocorr(lag=period)
                        if abs(autocorr) > 0.3:
                            seasonal_periods.append({
                                "period": period,
                                "autocorrelation": autocorr,
                                "strength": abs(autocorr)
                            })
                
                return {
                    "seasonal_strength": seasonal_strength,
                    "seasonal_periods": seasonal_periods,
                    "trend": decomposition.trend.tolist() if hasattr(decomposition.trend, 'tolist') else [],
                    "seasonal": decomposition.seasonal.tolist() if hasattr(decomposition.seasonal, 'tolist') else [],
                    "residual": decomposition.resid.tolist() if hasattr(decomposition.resid, 'tolist') else []
                }
                
            except Exception as e:
                logger.warning(f"Seasonal decomposition failed: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing seasonality: {e}")
            return None
    
    async def _generate_forecast(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Generar pronóstico"""
        try:
            if len(values) < 10:
                return None
            
            # Crear serie temporal
            df = pd.DataFrame({
                'timestamp': timestamps,
                'value': values
            })
            df.set_index('timestamp', inplace=True)
            
            forecasts = {}
            
            # Método 1: Suavizado exponencial
            try:
                model = ExponentialSmoothing(
                    df['value'],
                    trend='add',
                    seasonal='add',
                    seasonal_periods=min(7, len(values) // 4)
                )
                fitted_model = model.fit()
                forecast_periods = min(self.config["max_forecast_periods"], len(values) // 2)
                forecast = fitted_model.forecast(forecast_periods)
                
                forecasts["exponential_smoothing"] = {
                    "values": forecast.tolist(),
                    "confidence_intervals": None  # Simplificado
                }
            except:
                pass
            
            # Método 2: ARIMA
            try:
                # Determinar orden ARIMA automáticamente
                model = ARIMA(df['value'], order=(1, 1, 1))
                fitted_model = model.fit()
                forecast_periods = min(self.config["max_forecast_periods"], len(values) // 2)
                forecast = fitted_model.forecast(forecast_periods)
                
                forecasts["arima"] = {
                    "values": forecast.tolist(),
                    "model_order": (1, 1, 1)
                }
            except:
                pass
            
            # Método 3: Regresión lineal simple
            try:
                time_numeric = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
                slope, intercept, _, _, _ = stats.linregress(time_numeric, values)
                
                forecast_periods = min(self.config["max_forecast_periods"], len(values) // 2)
                last_time = time_numeric[-1]
                forecast_times = [last_time + i for i in range(1, forecast_periods + 1)]
                forecast_values = [slope * t + intercept for t in forecast_times]
                
                forecasts["linear_regression"] = {
                    "values": forecast_values,
                    "slope": slope,
                    "intercept": intercept
                }
            except:
                pass
            
            if forecasts:
                return {
                    "methods": forecasts,
                    "forecast_periods": min(self.config["max_forecast_periods"], len(values) // 2),
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None
    
    async def compare_temporal_metrics(
        self,
        metric_names: List[str]
    ) -> Dict[str, Any]:
        """Comparar métricas temporales"""
        try:
            if len(metric_names) < 2:
                raise ValueError("Se necesitan al menos 2 métricas para comparar")
            
            # Obtener análisis de tendencias
            analyses = []
            for metric_name in metric_names:
                if metric_name in self.temporal_data:
                    analysis = await self.analyze_trends(metric_name)
                    analyses.append(analysis)
                else:
                    logger.warning(f"No data available for metric {metric_name}")
            
            if len(analyses) < 2:
                raise ValueError("No hay suficientes análisis para comparar")
            
            # Calcular correlaciones
            correlations = await self._calculate_temporal_correlations(analyses)
            
            # Encontrar diferencias significativas
            significant_differences = await self._find_temporal_differences(analyses)
            
            # Generar insights comparativos
            comparative_insights = await self._generate_temporal_insights(analyses)
            
            return {
                "metric_names": metric_names,
                "analyses_count": len(analyses),
                "correlations": correlations,
                "significant_differences": significant_differences,
                "comparative_insights": comparative_insights,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing temporal metrics: {e}")
            raise
    
    async def _calculate_temporal_correlations(
        self,
        analyses: List[TrendAnalysis]
    ) -> Dict[str, float]:
        """Calcular correlaciones temporales"""
        correlations = {}
        
        try:
            for i, analysis1 in enumerate(analyses):
                for j, analysis2 in enumerate(analyses[i+1:], i+1):
                    # Alinear datos temporales
                    aligned_data = self._align_temporal_data(
                        analysis1.data_points,
                        analysis2.data_points
                    )
                    
                    if len(aligned_data) > 5:
                        values1 = [point[0].value for point in aligned_data]
                        values2 = [point[1].value for point in aligned_data]
                        
                        correlation, p_value = stats.pearsonr(values1, values2)
                        
                        correlations[f"{analysis1.metric_name}_vs_{analysis2.metric_name}"] = {
                            "correlation": correlation,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "data_points": len(aligned_data)
                        }
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating temporal correlations: {e}")
            return {}
    
    def _align_temporal_data(
        self,
        data1: List[TemporalPoint],
        data2: List[TemporalPoint]
    ) -> List[Tuple[TemporalPoint, TemporalPoint]]:
        """Alinear datos temporales"""
        aligned = []
        
        # Crear diccionarios para búsqueda rápida
        dict1 = {point.timestamp: point for point in data1}
        dict2 = {point.timestamp: point for point in data2}
        
        # Encontrar timestamps comunes
        common_timestamps = set(dict1.keys()) & set(dict2.keys())
        
        for timestamp in sorted(common_timestamps):
            aligned.append((dict1[timestamp], dict2[timestamp]))
        
        return aligned
    
    async def _find_temporal_differences(
        self,
        analyses: List[TrendAnalysis]
    ) -> List[Dict[str, Any]]:
        """Encontrar diferencias temporales significativas"""
        differences = []
        
        try:
            # Comparar tipos de tendencia
            trend_types = [analysis.trend_type for analysis in analyses]
            if len(set(trend_types)) > 1:
                differences.append({
                    "type": "trend_type",
                    "description": "Diferentes tipos de tendencia detectados",
                    "trends": [trend.value for trend in trend_types],
                    "impact": "high"
                })
            
            # Comparar pendientes
            slopes = [analysis.slope for analysis in analyses]
            if max(slopes) - min(slopes) > 0.1:
                differences.append({
                    "type": "slope",
                    "description": "Diferencias significativas en pendientes de tendencia",
                    "slope_range": f"{min(slopes):.4f} to {max(slopes):.4f}",
                    "impact": "medium"
                })
            
            # Comparar R²
            r_squared_values = [analysis.r_squared for analysis in analyses]
            if max(r_squared_values) - min(r_squared_values) > 0.3:
                differences.append({
                    "type": "fit_quality",
                    "description": "Diferencias significativas en calidad de ajuste",
                    "r_squared_range": f"{min(r_squared_values):.3f} to {max(r_squared_values):.3f}",
                    "impact": "medium"
                })
            
            return differences
            
        except Exception as e:
            logger.error(f"Error finding temporal differences: {e}")
            return []
    
    async def _generate_temporal_insights(
        self,
        analyses: List[TrendAnalysis]
    ) -> List[str]:
        """Generar insights temporales"""
        insights = []
        
        try:
            # Insight sobre consistencia de tendencias
            trend_types = [analysis.trend_type for analysis in analyses]
            if len(set(trend_types)) == 1:
                insights.append(f"Todas las métricas muestran tendencia {trend_types[0].value}")
            else:
                insights.append("Variedad de tendencias detectada entre métricas")
            
            # Insight sobre calidad de ajuste
            r_squared_values = [analysis.r_squared for analysis in analyses]
            avg_r_squared = np.mean(r_squared_values)
            
            if avg_r_squared > 0.8:
                insights.append("Alta calidad de ajuste en las tendencias")
            elif avg_r_squared < 0.3:
                insights.append("Baja calidad de ajuste - considerar más datos o diferentes modelos")
            
            # Insight sobre anomalías
            total_anomalies = sum(len(analysis.anomalies) for analysis in analyses)
            if total_anomalies > 0:
                insights.append(f"Se detectaron {total_anomalies} anomalías en total")
            
            # Insight sobre estacionalidad
            seasonal_analyses = [a for a in analyses if a.seasonality and a.seasonality.get("seasonal_strength", 0) > 0.1]
            if seasonal_analyses:
                insights.append(f"{len(seasonal_analyses)} métricas muestran patrones estacionales")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating temporal insights: {e}")
            return []
    
    async def get_temporal_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis temporales"""
        if not self.trend_analyses:
            return {"message": "No temporal analyses available"}
        
        # Estadísticas generales
        total_analyses = len(self.trend_analyses)
        total_data_points = sum(len(analysis.data_points) for analysis in self.trend_analyses.values())
        
        # Distribución de tipos de tendencia
        trend_types = [analysis.trend_type for analysis in self.trend_analyses.values()]
        trend_distribution = Counter(trend_types)
        
        # Distribución de tipos de patrón
        pattern_types = [analysis.pattern_type for analysis in self.trend_analyses.values()]
        pattern_distribution = Counter(pattern_types)
        
        # Estadísticas de calidad
        r_squared_values = [analysis.r_squared for analysis in self.trend_analyses.values()]
        slopes = [analysis.slope for analysis in self.trend_analyses.values()]
        
        # Anomalías
        total_anomalies = sum(len(analysis.anomalies) for analysis in self.trend_analyses.values())
        
        return {
            "total_analyses": total_analyses,
            "total_data_points": total_data_points,
            "trend_distribution": {trend.value: count for trend, count in trend_distribution.items()},
            "pattern_distribution": {pattern.value: count for pattern, count in pattern_distribution.items()},
            "average_r_squared": np.mean(r_squared_values),
            "average_slope": np.mean(slopes),
            "total_anomalies": total_anomalies,
            "total_insights": len(self.temporal_insights),
            "last_analysis": max([analysis.created_at for analysis in self.trend_analyses.values()]).isoformat()
        }
    
    async def export_temporal_analysis(self, filepath: str = None) -> str:
        """Exportar análisis temporal"""
        try:
            if filepath is None:
                filepath = f"exports/temporal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                "trend_analyses": {
                    analysis_id: {
                        "metric_name": analysis.metric_name,
                        "trend_type": analysis.trend_type.value,
                        "pattern_type": analysis.pattern_type.value,
                        "slope": analysis.slope,
                        "r_squared": analysis.r_squared,
                        "p_value": analysis.p_value,
                        "confidence_interval": analysis.confidence_interval,
                        "start_date": analysis.start_date.isoformat(),
                        "end_date": analysis.end_date.isoformat(),
                        "data_points": [
                            {
                                "timestamp": point.timestamp.isoformat(),
                                "value": point.value,
                                "confidence": point.confidence,
                                "metadata": point.metadata
                            }
                            for point in analysis.data_points
                        ],
                        "anomalies": analysis.anomalies,
                        "seasonality": analysis.seasonality,
                        "forecast": analysis.forecast,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.trend_analyses.items()
                },
                "temporal_insights": {
                    insight_id: {
                        "insight_type": insight.insight_type,
                        "description": insight.description,
                        "significance": insight.significance,
                        "confidence": insight.confidence,
                        "timeframe": [insight.timeframe[0].isoformat(), insight.timeframe[1].isoformat()],
                        "related_metrics": insight.related_metrics,
                        "implications": insight.implications,
                        "recommendations": insight.recommendations,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.temporal_insights.items()
                },
                "summary": await self.get_temporal_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Temporal analysis exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting temporal analysis: {e}")
            raise

























