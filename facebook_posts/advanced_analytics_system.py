#!/usr/bin/env python3
"""
📊 Advanced Analytics System for Enhanced Facebook Content Optimization
=====================================================================

This module provides comprehensive analytics capabilities:
- Real-time Analytics (Prometheus, Grafana)
- Business Intelligence (Streamlit, Dash, Plotly)
- Time Series Analysis (Prophet, Statsmodels)
- Geospatial Analytics (GeoPandas, Folium)
- Financial Analytics (yfinance, pandas-ta)
- Advanced Visualization (Bokeh, Altair, HoloViews)
- Performance Monitoring (psutil, GPUtil)
- Data Processing (Pandas, Polars, Dask)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import json
import time
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

# ===== DATA PROCESSING LIBRARIES =====

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logging.warning("Polars not available. Install with: pip install polars")

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logging.warning("Dask not available. Install with: pip install dask")

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    logging.warning("Vaex not available. Install with: pip install vaex")

# ===== VISUALIZATION LIBRARIES =====

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

try:
    import bokeh
    from bokeh.plotting import figure, show
    from bokeh.layouts import column, row
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    logging.warning("Bokeh not available. Install with: pip install bokeh")

try:
    import altair as alt
    ALTAR_AVAILABLE = True
except ImportError:
    ALTAR_AVAILABLE = False
    logging.warning("Altair not available. Install with: pip install altair")

try:
    import holoviews as hv
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    HOLOVIEWS_AVAILABLE = False
    logging.warning("HoloViews not available. Install with: pip install holoviews")

# ===== TIME SERIES ANALYSIS =====

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")

# ===== GEOSPATIAL ANALYTICS =====

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logging.warning("GeoPandas not available. Install with: pip install geopandas")

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logging.warning("Folium not available. Install with: pip install folium")

# ===== FINANCIAL ANALYTICS =====

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available. Install with: pip install yfinance")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not available. Install with: pip install pandas-ta")

# ===== PERFORMANCE MONITORING =====

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Install with: pip install psutil")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available. Install with: pip install GPUtil")

# ===== MONITORING & OBSERVABILITY =====

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

try:
    from structlog import get_logger
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    logging.warning("Structlog not available. Install with: pip install structlog")

# ===== BUSINESS INTELLIGENCE =====

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logging.warning("Streamlit not available. Install with: pip install streamlit")

try:
    from dash import Dash, html, dcc
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logging.warning("Dash not available. Install with: pip install dash")


@dataclass
class AdvancedAnalyticsConfig:
    """Configuration for advanced analytics system"""
    
    # Data Processing
    enable_polars: bool = True
    enable_dask: bool = True
    enable_vaex: bool = True
    
    # Visualization
    enable_plotly: bool = True
    enable_bokeh: bool = True
    enable_altair: bool = True
    enable_holoviews: bool = True
    
    # Time Series
    enable_prophet: bool = True
    enable_statsmodels: bool = True
    
    # Geospatial
    enable_geopandas: bool = True
    enable_folium: bool = True
    
    # Financial
    enable_yfinance: bool = True
    enable_pandas_ta: bool = True
    
    # Monitoring
    enable_psutil: bool = True
    enable_gputil: bool = True
    enable_prometheus: bool = True
    enable_structlog: bool = True
    
    # Business Intelligence
    enable_streamlit: bool = True
    enable_dash: bool = True
    
    # Performance
    cache_size: int = 10000
    max_workers: int = 8
    update_interval: int = 60  # seconds


class DataProcessor:
    """Advanced data processing using multiple engines"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
        self.cache = {}
        self.processing_stats = defaultdict(int)
    
    def process_with_pandas(self, data: Union[List, Dict, pd.DataFrame]) -> pd.DataFrame:
        """Process data using Pandas"""
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            self.processing_stats['pandas_operations'] += 1
            return df
            
        except Exception as e:
            logging.error(f"Error in Pandas processing: {e}")
            return pd.DataFrame()
    
    def process_with_polars(self, data: Union[List, Dict, pd.DataFrame]) -> Optional[Any]:
        """Process data using Polars"""
        if not POLARS_AVAILABLE or not self.config.enable_polars:
            return None
        
        try:
            if isinstance(data, pd.DataFrame):
                df = pl.from_pandas(data)
            elif isinstance(data, list):
                df = pl.DataFrame(data)
            elif isinstance(data, dict):
                df = pl.DataFrame([data])
            else:
                df = pl.DataFrame(data)
            
            self.processing_stats['polars_operations'] += 1
            return df
            
        except Exception as e:
            logging.error(f"Error in Polars processing: {e}")
            return None
    
    def process_with_dask(self, data: Union[List, Dict, pd.DataFrame]) -> Optional[Any]:
        """Process data using Dask"""
        if not DASK_AVAILABLE or not self.config.enable_dask:
            return None
        
        try:
            if isinstance(data, pd.DataFrame):
                df = dd.from_pandas(data, npartitions=4)
            elif isinstance(data, list):
                df = dd.from_pandas(pd.DataFrame(data), npartitions=4)
            elif isinstance(data, dict):
                df = dd.from_pandas(pd.DataFrame([data]), npartitions=4)
            else:
                df = dd.from_pandas(pd.DataFrame(data), npartitions=4)
            
            self.processing_stats['dask_operations'] += 1
            return df
            
        except Exception as e:
            logging.error(f"Error in Dask processing: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "pandas_operations": self.processing_stats['pandas_operations'],
            "polars_operations": self.processing_stats['polars_operations'],
            "dask_operations": self.processing_stats['dask_operations'],
            "cache_size": len(self.cache)
        }


class AdvancedVisualizer:
    """Advanced visualization using multiple libraries"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
        self.plot_cache = {}
    
    def create_plotly_dashboard(self, data: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Create interactive Plotly dashboard"""
        if not PLOTLY_AVAILABLE or not self.config.enable_plotly:
            return {"error": "Plotly not available"}
        
        try:
            figures = {}
            
            # Time series plot
            if 'timestamp' in data.columns and len(metrics) > 0:
                fig = px.line(data, x='timestamp', y=metrics[0], title=f'{metrics[0]} Over Time')
                figures['time_series'] = fig.to_dict()
            
            # Scatter plot
            if len(metrics) >= 2:
                fig = px.scatter(data, x=metrics[0], y=metrics[1], title=f'{metrics[0]} vs {metrics[1]}')
                figures['scatter'] = fig.to_dict()
            
            # Histogram
            if len(metrics) > 0:
                fig = px.histogram(data, x=metrics[0], title=f'Distribution of {metrics[0]}')
                figures['histogram'] = fig.to_dict()
            
            # Correlation heatmap
            if len(metrics) > 1:
                corr_matrix = data[metrics].corr()
                fig = px.imshow(corr_matrix, title='Correlation Matrix')
                figures['correlation'] = fig.to_dict()
            
            return {
                "dashboard_type": "plotly",
                "figures": figures,
                "metrics": metrics
            }
            
        except Exception as e:
            logging.error(f"Error creating Plotly dashboard: {e}")
            return {"error": str(e)}
    
    def create_bokeh_dashboard(self, data: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Create interactive Bokeh dashboard"""
        if not BOKEH_AVAILABLE or not self.config.enable_bokeh:
            return {"error": "Bokeh not available"}
        
        try:
            figures = {}
            
            # Time series plot
            if 'timestamp' in data.columns and len(metrics) > 0:
                p = figure(width=800, height=400, title=f'{metrics[0]} Over Time')
                p.line(data['timestamp'], data[metrics[0]], line_width=2)
                figures['time_series'] = p
            
            # Scatter plot
            if len(metrics) >= 2:
                p = figure(width=800, height=400, title=f'{metrics[0]} vs {metrics[1]}')
                p.circle(data[metrics[0]], data[metrics[1]], size=8, alpha=0.6)
                figures['scatter'] = p
            
            return {
                "dashboard_type": "bokeh",
                "figures": figures,
                "metrics": metrics
            }
            
        except Exception as e:
            logging.error(f"Error creating Bokeh dashboard: {e}")
            return {"error": str(e)}
    
    def create_altair_dashboard(self, data: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Create interactive Altair dashboard"""
        if not ALTAR_AVAILABLE or not self.config.enable_altair:
            return {"error": "Altair not available"}
        
        try:
            charts = {}
            
            # Time series chart
            if 'timestamp' in data.columns and len(metrics) > 0:
                chart = alt.Chart(data).mark_line().encode(
                    x='timestamp:T',
                    y=f'{metrics[0]}:Q'
                ).properties(title=f'{metrics[0]} Over Time')
                charts['time_series'] = chart
            
            # Scatter chart
            if len(metrics) >= 2:
                chart = alt.Chart(data).mark_circle().encode(
                    x=f'{metrics[0]}:Q',
                    y=f'{metrics[1]}:Q'
                ).properties(title=f'{metrics[0]} vs {metrics[1]}')
                charts['scatter'] = chart
            
            return {
                "dashboard_type": "altair",
                "charts": charts,
                "metrics": metrics
            }
            
        except Exception as e:
            logging.error(f"Error creating Altair dashboard: {e}")
            return {"error": str(e)}


class TimeSeriesAnalyzer:
    """Advanced time series analysis using Prophet and Statsmodels"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
        self.models = {}
    
    def analyze_with_prophet(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze time series using Prophet"""
        if not PROPHET_AVAILABLE or not self.config.enable_prophet:
            return {"error": "Prophet not available"}
        
        try:
            # Prepare data for Prophet
            df_prophet = data[['timestamp', target_column]].copy()
            df_prophet.columns = ['ds', 'y']
            
            # Create and fit model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(df_prophet)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Extract components
            components = model.plot_components(forecast)
            
            self.models[f'prophet_{target_column}'] = model
            
            return {
                "model_type": "prophet",
                "target_column": target_column,
                "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict('records'),
                "components": components,
                "model_params": model.params
            }
            
        except Exception as e:
            logging.error(f"Error in Prophet analysis: {e}")
            return {"error": str(e)}
    
    def analyze_with_statsmodels(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze time series using Statsmodels"""
        if not STATSMODELS_AVAILABLE or not self.config.enable_statsmodels:
            return {"error": "Statsmodels not available"}
        
        try:
            # Prepare data
            ts_data = data.set_index('timestamp')[target_column]
            
            # Decompose time series
            decomposition = seasonal_decompose(ts_data, period=7, extrapolate_trend='freq')
            
            # Stationarity test
            adf_result = adfuller(ts_data.dropna())
            
            # ARIMA model
            model = sm.tsa.ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=30)
            
            return {
                "model_type": "statsmodels",
                "target_column": target_column,
                "decomposition": {
                    "trend": decomposition.trend.tolist(),
                    "seasonal": decomposition.seasonal.tolist(),
                    "residual": decomposition.resid.tolist()
                },
                "stationarity_test": {
                    "adf_statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "critical_values": adf_result[4]
                },
                "forecast": forecast.tolist(),
                "model_summary": fitted_model.summary().as_text()
            }
            
        except Exception as e:
            logging.error(f"Error in Statsmodels analysis: {e}")
            return {"error": str(e)}


class GeospatialAnalyzer:
    """Geospatial analysis using GeoPandas and Folium"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
    
    def create_geospatial_visualization(self, data: pd.DataFrame, lat_col: str, lon_col: str, 
                                      value_col: str = None) -> Dict[str, Any]:
        """Create geospatial visualization"""
        if not FOLIUM_AVAILABLE or not self.config.enable_folium:
            return {"error": "Folium not available"}
        
        try:
            # Create map
            center_lat = data[lat_col].mean()
            center_lon = data[lon_col].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add markers
            for idx, row in data.iterrows():
                popup_text = f"Value: {row[value_col]}" if value_col else f"Point {idx}"
                folium.Marker(
                    [row[lat_col], row[lon_col]],
                    popup=popup_text,
                    tooltip=f"Point {idx}"
                ).add_to(m)
            
            # Add heatmap if value column exists
            if value_col:
                heat_data = data[[lat_col, lon_col, value_col]].values.tolist()
                folium.HeatMap(heat_data).add_to(m)
            
            return {
                "map_type": "folium",
                "center": [center_lat, center_lon],
                "data_points": len(data),
                "map_html": m._repr_html_()
            }
            
        except Exception as e:
            logging.error(f"Error in geospatial visualization: {e}")
            return {"error": str(e)}
    
    def analyze_spatial_patterns(self, data: pd.DataFrame, lat_col: str, lon_col: str, 
                               value_col: str) -> Dict[str, Any]:
        """Analyze spatial patterns"""
        if not GEOPANDAS_AVAILABLE or not self.config.enable_geopandas:
            return {"error": "GeoPandas not available"}
        
        try:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data[lon_col], data[lat_col])
            )
            
            # Spatial statistics
            spatial_stats = {
                "total_points": len(gdf),
                "bounding_box": gdf.total_bounds.tolist(),
                "centroid": gdf.geometry.centroid.mean().coords[0],
                "spatial_extent": gdf.geometry.bounds
            }
            
            # Value statistics
            if value_col:
                spatial_stats["value_stats"] = {
                    "mean": gdf[value_col].mean(),
                    "std": gdf[value_col].std(),
                    "min": gdf[value_col].min(),
                    "max": gdf[value_col].max(),
                    "median": gdf[value_col].median()
                }
            
            return {
                "analysis_type": "spatial_patterns",
                "spatial_stats": spatial_stats,
                "geometry_type": gdf.geometry.geom_type.iloc[0]
            }
            
        except Exception as e:
            logging.error(f"Error in spatial pattern analysis: {e}")
            return {"error": str(e)}


class FinancialAnalyzer:
    """Financial analysis using yfinance and pandas-ta"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
    
    def analyze_stock_data(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Analyze stock data using yfinance"""
        if not YFINANCE_AVAILABLE or not self.config.enable_yfinance:
            return {"error": "yfinance not available"}
        
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Basic statistics
            stats = {
                "symbol": symbol,
                "period": period,
                "data_points": len(data),
                "price_range": {
                    "high": data['High'].max(),
                    "low": data['Low'].min(),
                    "current": data['Close'].iloc[-1]
                },
                "volume_stats": {
                    "total_volume": data['Volume'].sum(),
                    "avg_volume": data['Volume'].mean()
                }
            }
            
            # Technical indicators
            if PANDAS_TA_AVAILABLE and self.config.enable_pandas_ta:
                # Add technical indicators
                data_ta = data.copy()
                data_ta['SMA_20'] = ta.sma(data_ta['Close'], length=20)
                data_ta['EMA_20'] = ta.ema(data_ta['Close'], length=20)
                data_ta['RSI'] = ta.rsi(data_ta['Close'], length=14)
                data_ta['MACD'] = ta.macd(data_ta['Close'])['MACD_12_26_9']
                
                stats["technical_indicators"] = {
                    "sma_20": data_ta['SMA_20'].iloc[-1],
                    "ema_20": data_ta['EMA_20'].iloc[-1],
                    "rsi": data_ta['RSI'].iloc[-1],
                    "macd": data_ta['MACD'].iloc[-1]
                }
            
            return {
                "analysis_type": "stock_analysis",
                "data": data.tail(10).to_dict('records'),
                "statistics": stats
            }
            
        except Exception as e:
            logging.error(f"Error in stock analysis: {e}")
            return {"error": str(e)}
    
    def calculate_technical_indicators(self, data: pd.DataFrame, price_col: str = 'Close') -> Dict[str, Any]:
        """Calculate technical indicators using pandas-ta"""
        if not PANDAS_TA_AVAILABLE or not self.config.enable_pandas_ta:
            return {"error": "pandas-ta not available"}
        
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = ta.sma(data[price_col], length=20).iloc[-1]
            indicators['ema_20'] = ta.ema(data[price_col], length=20).iloc[-1]
            
            # Oscillators
            indicators['rsi'] = ta.rsi(data[price_col], length=14).iloc[-1]
            indicators['stoch'] = ta.stoch(data['High'], data['Low'], data[price_col]).iloc[-1]
            
            # Trend indicators
            indicators['macd'] = ta.macd(data[price_col])['MACD_12_26_9'].iloc[-1]
            indicators['bbands'] = ta.bbands(data[price_col]).iloc[-1].to_dict()
            
            # Volume indicators
            if 'Volume' in data.columns:
                indicators['obv'] = ta.obv(data[price_col], data['Volume']).iloc[-1]
            
            return {
                "analysis_type": "technical_indicators",
                "indicators": indicators
            }
            
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return {"error": str(e)}


class PerformanceMonitor:
    """System performance monitoring using psutil and GPUtil"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
        self.monitoring_data = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logging.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.get_system_metrics()
                self.monitoring_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                time.sleep(self.config.update_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {}
        
        # CPU metrics
        if PSUTIL_AVAILABLE and self.config.enable_psutil:
            metrics['cpu'] = {
                'usage_percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'usage_percent': memory.percent
            }
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk'] = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': (disk.used / disk.total) * 100
            }
        
        # GPU metrics
        if GPUTIL_AVAILABLE and self.config.enable_gputil:
            try:
                gpus = GPUtil.getGPUs()
                metrics['gpu'] = []
                for gpu in gpus:
                    metrics['gpu'].append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load_percent': gpu.load * 100,
                        'memory_used_gb': gpu.memoryUsed,
                        'memory_total_gb': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
            except Exception as e:
                logging.error(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def get_monitoring_history(self) -> List[Dict[str, Any]]:
        """Get monitoring history"""
        return list(self.monitoring_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.monitoring_data:
            return {"error": "No monitoring data available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': item['timestamp'],
                **item['metrics'].get('cpu', {}),
                **item['metrics'].get('memory', {}),
                **item['metrics'].get('disk', {})
            }
            for item in self.monitoring_data
        ])
        
        if df.empty:
            return {"error": "No valid monitoring data"}
        
        # Calculate summary statistics
        summary = {
            "monitoring_duration_hours": (datetime.fromisoformat(df['timestamp'].iloc[-1]) - 
                                        datetime.fromisoformat(df['timestamp'].iloc[0])).total_seconds() / 3600,
            "data_points": len(df),
            "cpu": {
                "avg_usage": df.get('usage_percent', pd.Series()).mean(),
                "max_usage": df.get('usage_percent', pd.Series()).max(),
                "min_usage": df.get('usage_percent', pd.Series()).min()
            },
            "memory": {
                "avg_usage_gb": df.get('used_gb', pd.Series()).mean(),
                "max_usage_gb": df.get('used_gb', pd.Series()).max(),
                "avg_usage_percent": df.get('usage_percent', pd.Series()).mean()
            },
            "disk": {
                "avg_usage_gb": df.get('used_gb', pd.Series()).mean(),
                "avg_usage_percent": df.get('usage_percent', pd.Series()).mean()
            }
        }
        
        return summary


class PrometheusExporter:
    """Prometheus metrics exporter"""
    
    def __init__(self, config: AdvancedAnalyticsConfig):
        self.config = config
        self.metrics = {}
        
        if PROMETHEUS_AVAILABLE and self.config.enable_prometheus:
            self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            # Counters
            self.metrics['requests_total'] = Counter('facebook_content_requests_total', 'Total content optimization requests')
            self.metrics['errors_total'] = Counter('facebook_content_errors_total', 'Total errors')
            
            # Gauges
            self.metrics['active_requests'] = Gauge('facebook_content_active_requests', 'Currently active requests')
            self.metrics['system_memory_usage'] = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
            self.metrics['system_cpu_usage'] = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
            
            # Histograms
            self.metrics['request_duration'] = Histogram('facebook_content_request_duration_seconds', 'Request duration in seconds')
            
            # Summaries
            self.metrics['content_optimization_score'] = Summary('facebook_content_optimization_score', 'Content optimization scores')
            
            logging.info("Prometheus metrics initialized")
            
        except Exception as e:
            logging.error(f"Error initializing Prometheus metrics: {e}")
    
    def record_request(self, duration: float, success: bool = True):
        """Record a request"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.metrics['requests_total'].inc()
            self.metrics['request_duration'].observe(duration)
            
            if not success:
                self.metrics['errors_total'].inc()
                
        except Exception as e:
            logging.error(f"Error recording request metrics: {e}")
    
    def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            if 'memory' in metrics:
                memory_used = metrics['memory'].get('used_gb', 0) * (1024**3)  # Convert to bytes
                self.metrics['system_memory_usage'].set(memory_used)
            
            if 'cpu' in metrics:
                cpu_usage = metrics['cpu'].get('usage_percent', 0)
                self.metrics['system_cpu_usage'].set(cpu_usage)
                
        except Exception as e:
            logging.error(f"Error updating system metrics: {e}")
    
    def record_optimization_score(self, score: float):
        """Record optimization score"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.metrics['content_optimization_score'].observe(score)
        except Exception as e:
            logging.error(f"Error recording optimization score: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available"
        
        try:
            return generate_latest()
        except Exception as e:
            logging.error(f"Error generating metrics: {e}")
            return "# Error generating metrics"


class AdvancedAnalyticsSystem:
    """Main system that integrates all advanced analytics capabilities"""
    
    def __init__(self, config: AdvancedAnalyticsConfig = None):
        self.config = config or AdvancedAnalyticsConfig()
        
        # Initialize all analytics modules
        self.data_processor = DataProcessor(self.config)
        self.visualizer = AdvancedVisualizer(self.config)
        self.time_series_analyzer = TimeSeriesAnalyzer(self.config)
        self.geospatial_analyzer = GeospatialAnalyzer(self.config)
        self.financial_analyzer = FinancialAnalyzer(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.prometheus_exporter = PrometheusExporter(self.config)
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        logging.info("Advanced Analytics System initialized successfully")
    
    def analyze_content_performance(self, content_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content performance using multiple analytics methods"""
        try:
            # Convert to DataFrame
            df = self.data_processor.process_with_pandas(content_data)
            
            if df.empty:
                return {"error": "No data to analyze"}
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "data_points": len(df),
                "analytics_modules": {}
            }
            
            # Basic statistics
            analysis["analytics_modules"]["basic_stats"] = {
                "engagement_mean": df.get('engagement_score', pd.Series()).mean(),
                "engagement_std": df.get('engagement_score', pd.Series()).std(),
                "viral_potential_mean": df.get('viral_potential', pd.Series()).mean(),
                "optimization_score_mean": df.get('optimization_score', pd.Series()).mean()
            }
            
            # Time series analysis if timestamp available
            if 'timestamp' in df.columns and 'engagement_score' in df.columns:
                ts_analysis = self.time_series_analyzer.analyze_with_statsmodels(df, 'engagement_score')
                analysis["analytics_modules"]["time_series"] = ts_analysis
            
            # Visualization
            metrics = [col for col in df.columns if 'score' in col.lower() or 'engagement' in col.lower()]
            if metrics:
                plotly_dashboard = self.visualizer.create_plotly_dashboard(df, metrics[:3])
                analysis["analytics_modules"]["visualization"] = plotly_dashboard
            
            # Performance monitoring
            system_metrics = self.performance_monitor.get_system_metrics()
            analysis["analytics_modules"]["system_performance"] = system_metrics
            
            # Update Prometheus metrics
            self.prometheus_exporter.update_system_metrics(system_metrics)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in content performance analysis: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all analytics modules"""
        return {
            "data_processing": {
                "pandas": True,
                "polars": POLARS_AVAILABLE and self.config.enable_polars,
                "dask": DASK_AVAILABLE and self.config.enable_dask
            },
            "visualization": {
                "plotly": PLOTLY_AVAILABLE and self.config.enable_plotly,
                "bokeh": BOKEH_AVAILABLE and self.config.enable_bokeh,
                "altair": ALTAR_AVAILABLE and self.config.enable_altair
            },
            "time_series": {
                "prophet": PROPHET_AVAILABLE and self.config.enable_prophet,
                "statsmodels": STATSMODELS_AVAILABLE and self.config.enable_statsmodels
            },
            "geospatial": {
                "geopandas": GEOPANDAS_AVAILABLE and self.config.enable_geopandas,
                "folium": FOLIUM_AVAILABLE and self.config.enable_folium
            },
            "financial": {
                "yfinance": YFINANCE_AVAILABLE and self.config.enable_yfinance,
                "pandas_ta": PANDAS_TA_AVAILABLE and self.config.enable_pandas_ta
            },
            "monitoring": {
                "psutil": PSUTIL_AVAILABLE and self.config.enable_psutil,
                "gputil": GPUTIL_AVAILABLE and self.config.enable_gputil,
                "prometheus": PROMETHEUS_AVAILABLE and self.config.enable_prometheus
            },
            "performance_monitor": {
                "active": self.performance_monitor.monitoring_active,
                "data_points": len(self.performance_monitor.monitoring_data)
            }
        }
    
    def shutdown(self):
        """Shutdown the analytics system"""
        self.performance_monitor.stop_monitoring()
        logging.info("Advanced Analytics System shutdown complete")


# ===== UTILITY FUNCTIONS =====

def create_advanced_analytics_system(config: AdvancedAnalyticsConfig = None) -> AdvancedAnalyticsSystem:
    """Factory function to create Advanced Analytics System"""
    return AdvancedAnalyticsSystem(config)


def demo_advanced_analytics():
    """Demo function to showcase advanced analytics capabilities"""
    print("📊 Advanced Analytics System Demo")
    print("=" * 50)
    
    # Create system
    config = AdvancedAnalyticsConfig(
        enable_polars=True,
        enable_plotly=True,
        enable_statsmodels=True,
        enable_psutil=True,
        enable_prometheus=True
    )
    
    analytics_system = create_advanced_analytics_system(config)
    
    # Get system status
    status = analytics_system.get_system_status()
    print(f"System Status: {status}")
    
    # Demo with sample data
    sample_data = [
        {
            "timestamp": datetime.now() - timedelta(days=i),
            "engagement_score": np.random.uniform(0.3, 0.9),
            "viral_potential": np.random.uniform(0.2, 0.8),
            "optimization_score": np.random.uniform(0.4, 0.95)
        }
        for i in range(30)
    ]
    
    print(f"\n📈 Analyzing {len(sample_data)} data points...")
    analysis = analytics_system.analyze_content_performance(sample_data)
    
    print(f"Analysis completed with {len(analysis.get('analytics_modules', {}))} modules")
    
    return analysis


if __name__ == "__main__":
    # Run demo
    demo_advanced_analytics()
