"""
Advanced Geospatial and Spatiotemporal Analysis System
Sistema avanzado de análisis geoespacial y espacio-temporal
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
import warnings
warnings.filterwarnings('ignore')

# Geospatial imports
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import unary_union
    import folium
    import folium.plugins
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False

# Visualization imports
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialAnalysisType(Enum):
    """Tipos de análisis espacial"""
    POINT_PATTERN = "point_pattern"
    CLUSTERING = "clustering"
    HOTSPOT = "hotspot"
    INTERPOLATION = "interpolation"
    BUFFER_ANALYSIS = "buffer_analysis"
    OVERLAY_ANALYSIS = "overlay_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    ACCESSIBILITY = "accessibility"

class TemporalAnalysisType(Enum):
    """Tipos de análisis temporal"""
    TIME_SERIES = "time_series"
    SEASONALITY = "seasonality"
    TREND_ANALYSIS = "trend_analysis"
    CHANGE_DETECTION = "change_detection"
    EVENT_ANALYSIS = "event_analysis"
    PATTERN_MINING = "pattern_mining"

class CoordinateSystem(Enum):
    """Sistemas de coordenadas"""
    WGS84 = "WGS84"  # EPSG:4326
    UTM = "UTM"      # Universal Transverse Mercator
    WEB_MERCATOR = "WEB_MERCATOR"  # EPSG:3857
    CUSTOM = "CUSTOM"

@dataclass
class SpatialPoint:
    """Punto espacial"""
    id: str
    longitude: float
    latitude: float
    elevation: Optional[float] = None
    timestamp: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpatialPolygon:
    """Polígono espacial"""
    id: str
    coordinates: List[Tuple[float, float]]
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpatialAnalysis:
    """Análisis espacial"""
    id: str
    analysis_type: SpatialAnalysisType
    spatial_extent: Dict[str, float]  # bbox
    point_count: int
    polygon_count: int
    results: Dict[str, Any]
    statistics: Dict[str, float]
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SpatiotemporalInsight:
    """Insight espacio-temporal"""
    id: str
    insight_type: str
    description: str
    spatial_extent: Dict[str, float]
    temporal_extent: Tuple[datetime, datetime]
    significance: float
    confidence: float
    related_features: List[str]
    implications: List[str]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedGeospatialAnalyzer:
    """
    Analizador avanzado geoespacial y espacio-temporal
    """
    
    def __init__(
        self,
        enable_geospatial: bool = True,
        enable_visualization: bool = True,
        enable_clustering: bool = True,
        coordinate_system: CoordinateSystem = CoordinateSystem.WGS84
    ):
        self.enable_geospatial = enable_geospatial and GEOSPATIAL_AVAILABLE
        self.enable_visualization = enable_visualization
        self.enable_clustering = enable_clustering
        self.coordinate_system = coordinate_system
        
        # Almacenamiento
        self.spatial_points: Dict[str, List[SpatialPoint]] = {}
        self.spatial_polygons: Dict[str, List[SpatialPolygon]] = {}
        self.spatial_analyses: Dict[str, SpatialAnalysis] = {}
        self.spatiotemporal_insights: Dict[str, SpatiotemporalInsight] = {}
        
        # Configuración
        self.config = {
            "default_buffer_distance": 1000,  # metros
            "clustering_algorithm": "kmeans",
            "max_clusters": 10,
            "interpolation_method": "idw",  # Inverse Distance Weighting
            "hotspot_threshold": 0.05,
            "visualization_zoom": 10,
            "map_center": [0, 0]  # lat, lon
        }
        
        if not self.enable_geospatial:
            logger.warning("Geospatial analysis disabled - required packages not available")
    
    async def add_spatial_points(
        self,
        dataset_id: str,
        points: List[SpatialPoint],
        replace_existing: bool = False
    ) -> bool:
        """
        Agregar puntos espaciales
        
        Args:
            dataset_id: ID del dataset
            points: Lista de puntos espaciales
            replace_existing: Si reemplazar puntos existentes
            
        Returns:
            True si se agregaron exitosamente
        """
        try:
            if not self.enable_geospatial:
                logger.warning("Geospatial analysis not available")
                return False
            
            # Validar puntos
            validated_points = []
            for point in points:
                if -180 <= point.longitude <= 180 and -90 <= point.latitude <= 90:
                    validated_points.append(point)
                else:
                    logger.warning(f"Invalid coordinates for point {point.id}: {point.longitude}, {point.latitude}")
            
            if replace_existing or dataset_id not in self.spatial_points:
                self.spatial_points[dataset_id] = validated_points
            else:
                self.spatial_points[dataset_id].extend(validated_points)
            
            logger.info(f"Added {len(validated_points)} spatial points to dataset {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding spatial points: {e}")
            return False
    
    async def add_spatial_polygons(
        self,
        dataset_id: str,
        polygons: List[SpatialPolygon],
        replace_existing: bool = False
    ) -> bool:
        """
        Agregar polígonos espaciales
        
        Args:
            dataset_id: ID del dataset
            polygons: Lista de polígonos espaciales
            replace_existing: Si reemplazar polígonos existentes
            
        Returns:
            True si se agregaron exitosamente
        """
        try:
            if not self.enable_geospatial:
                logger.warning("Geospatial analysis not available")
                return False
            
            # Validar polígonos
            validated_polygons = []
            for polygon in polygons:
                if len(polygon.coordinates) >= 3:  # Mínimo 3 puntos para un polígono
                    validated_polygons.append(polygon)
                else:
                    logger.warning(f"Invalid polygon {polygon.id}: insufficient coordinates")
            
            if replace_existing or dataset_id not in self.spatial_polygons:
                self.spatial_polygons[dataset_id] = validated_polygons
            else:
                self.spatial_polygons[dataset_id].extend(validated_polygons)
            
            logger.info(f"Added {len(validated_polygons)} spatial polygons to dataset {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding spatial polygons: {e}")
            return False
    
    async def analyze_spatial_patterns(
        self,
        dataset_id: str,
        analysis_type: SpatialAnalysisType,
        parameters: Optional[Dict[str, Any]] = None
    ) -> SpatialAnalysis:
        """
        Analizar patrones espaciales
        
        Args:
            dataset_id: ID del dataset
            analysis_type: Tipo de análisis
            parameters: Parámetros adicionales
            
        Returns:
            Análisis espacial
        """
        try:
            if not self.enable_geospatial:
                raise ValueError("Geospatial analysis not available")
            
            if dataset_id not in self.spatial_points:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            points = self.spatial_points[dataset_id]
            if not points:
                raise ValueError(f"No points found in dataset {dataset_id}")
            
            logger.info(f"Analyzing spatial patterns for dataset {dataset_id} with {len(points)} points")
            
            # Calcular extensión espacial
            spatial_extent = self._calculate_spatial_extent(points)
            
            # Realizar análisis según el tipo
            if analysis_type == SpatialAnalysisType.POINT_PATTERN:
                results = await self._analyze_point_patterns(points, parameters)
            elif analysis_type == SpatialAnalysisType.CLUSTERING:
                results = await self._analyze_spatial_clustering(points, parameters)
            elif analysis_type == SpatialAnalysisType.HOTSPOT:
                results = await self._analyze_hotspots(points, parameters)
            elif analysis_type == SpatialAnalysisType.INTERPOLATION:
                results = await self._analyze_interpolation(points, parameters)
            elif analysis_type == SpatialAnalysisType.BUFFER_ANALYSIS:
                results = await self._analyze_buffer_analysis(points, parameters)
            else:
                results = await self._analyze_general_spatial(points, parameters)
            
            # Calcular estadísticas
            statistics = self._calculate_spatial_statistics(points, results)
            
            # Generar insights
            insights = self._generate_spatial_insights(points, results, analysis_type)
            
            # Crear análisis
            analysis = SpatialAnalysis(
                id=f"spatial_{dataset_id}_{analysis_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                analysis_type=analysis_type,
                spatial_extent=spatial_extent,
                point_count=len(points),
                polygon_count=len(self.spatial_polygons.get(dataset_id, [])),
                results=results,
                statistics=statistics,
                insights=insights
            )
            
            # Almacenar análisis
            self.spatial_analyses[analysis.id] = analysis
            
            logger.info(f"Spatial analysis completed: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing spatial patterns: {e}")
            raise
    
    def _calculate_spatial_extent(self, points: List[SpatialPoint]) -> Dict[str, float]:
        """Calcular extensión espacial"""
        try:
            longitudes = [point.longitude for point in points]
            latitudes = [point.latitude for point in points]
            
            return {
                "min_longitude": min(longitudes),
                "max_longitude": max(longitudes),
                "min_latitude": min(latitudes),
                "max_latitude": max(latitudes),
                "center_longitude": np.mean(longitudes),
                "center_latitude": np.mean(latitudes)
            }
        except Exception as e:
            logger.error(f"Error calculating spatial extent: {e}")
            return {}
    
    async def _analyze_point_patterns(
        self,
        points: List[SpatialPoint],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar patrones de puntos"""
        try:
            results = {}
            
            # Análisis de densidad
            if len(points) > 10:
                # Crear grid de densidad
                longitudes = [point.longitude for point in points]
                latitudes = [point.latitude for point in points]
                
                # Calcular densidad por grid
                grid_size = parameters.get("grid_size", 0.01) if parameters else 0.01
                lon_min, lon_max = min(longitudes), max(longitudes)
                lat_min, lat_max = min(latitudes), max(latitudes)
                
                density_grid = {}
                for point in points:
                    grid_lon = int((point.longitude - lon_min) / grid_size)
                    grid_lat = int((point.latitude - lat_min) / grid_size)
                    grid_key = (grid_lon, grid_lat)
                    density_grid[grid_key] = density_grid.get(grid_key, 0) + 1
                
                results["density_analysis"] = {
                    "grid_size": grid_size,
                    "density_grid": density_grid,
                    "max_density": max(density_grid.values()) if density_grid else 0,
                    "avg_density": np.mean(list(density_grid.values())) if density_grid else 0
                }
            
            # Análisis de distancia al vecino más cercano
            if len(points) > 2:
                distances = []
                for i, point1 in enumerate(points):
                    min_distance = float('inf')
                    for j, point2 in enumerate(points):
                        if i != j:
                            distance = self._calculate_distance(point1, point2)
                            min_distance = min(min_distance, distance)
                    distances.append(min_distance)
                
                results["nearest_neighbor"] = {
                    "distances": distances,
                    "mean_distance": np.mean(distances),
                    "std_distance": np.std(distances),
                    "min_distance": np.min(distances),
                    "max_distance": np.max(distances)
                }
            
            # Análisis de distribución espacial
            results["spatial_distribution"] = {
                "total_points": len(points),
                "spatial_extent": self._calculate_spatial_extent(points),
                "area_coverage": self._calculate_area_coverage(points)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing point patterns: {e}")
            return {}
    
    async def _analyze_spatial_clustering(
        self,
        points: List[SpatialPoint],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar clustering espacial"""
        try:
            results = {}
            
            if len(points) < 3:
                return {"error": "Insufficient points for clustering analysis"}
            
            # Preparar datos para clustering
            coordinates = np.array([[point.longitude, point.latitude] for point in points])
            
            # K-means clustering
            from sklearn.cluster import KMeans
            
            max_clusters = min(parameters.get("max_clusters", self.config["max_clusters"]), len(points))
            
            # Determinar número óptimo de clusters
            inertias = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(coordinates)
                inertias.append(kmeans.inertia_)
            
            # Usar método del codo para determinar k óptimo
            optimal_k = self._find_optimal_clusters(inertias, k_range)
            
            # Clustering final
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            
            # Analizar clusters
            clusters = {}
            for i, point in enumerate(points):
                cluster_id = cluster_labels[i]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(point)
            
            # Calcular estadísticas de clusters
            cluster_stats = {}
            for cluster_id, cluster_points in clusters.items():
                cluster_stats[cluster_id] = {
                    "point_count": len(cluster_points),
                    "center": kmeans.cluster_centers_[cluster_id].tolist(),
                    "spatial_extent": self._calculate_spatial_extent(cluster_points),
                    "density": len(cluster_points) / self._calculate_area_coverage(cluster_points) if self._calculate_area_coverage(cluster_points) > 0 else 0
                }
            
            results["clustering_analysis"] = {
                "algorithm": "kmeans",
                "optimal_clusters": optimal_k,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "inertias": inertias,
                "clusters": cluster_stats,
                "silhouette_score": self._calculate_silhouette_score(coordinates, cluster_labels)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing spatial clustering: {e}")
            return {}
    
    async def _analyze_hotspots(
        self,
        points: List[SpatialPoint],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar hotspots espaciales"""
        try:
            results = {}
            
            if len(points) < 10:
                return {"error": "Insufficient points for hotspot analysis"}
            
            # Análisis de Getis-Ord Gi*
            # Implementación simplificada
            
            # Crear grid de densidad
            longitudes = [point.longitude for point in points]
            latitudes = [point.latitude for point in points]
            
            grid_size = parameters.get("grid_size", 0.01) if parameters else 0.01
            lon_min, lon_max = min(longitudes), max(longitudes)
            lat_min, lat_max = min(latitudes), max(latitudes)
            
            # Crear grid
            grid_cells = {}
            for point in points:
                grid_lon = int((point.longitude - lon_min) / grid_size)
                grid_lat = int((point.latitude - lat_min) / grid_size)
                grid_key = (grid_lon, grid_lat)
                
                if grid_key not in grid_cells:
                    grid_cells[grid_key] = {
                        "count": 0,
                        "points": [],
                        "center_lon": lon_min + (grid_lon + 0.5) * grid_size,
                        "center_lat": lat_min + (grid_lat + 0.5) * grid_size
                    }
                
                grid_cells[grid_key]["count"] += 1
                grid_cells[grid_key]["points"].append(point)
            
            # Calcular estadísticas globales
            counts = [cell["count"] for cell in grid_cells.values()]
            global_mean = np.mean(counts)
            global_std = np.std(counts)
            
            # Identificar hotspots
            hotspots = []
            coldspots = []
            
            threshold = parameters.get("hotspot_threshold", self.config["hotspot_threshold"])
            
            for grid_key, cell in grid_cells.items():
                # Calcular z-score
                z_score = (cell["count"] - global_mean) / global_std if global_std > 0 else 0
                
                if z_score > 1.96:  # 95% confidence
                    hotspots.append({
                        "grid_key": grid_key,
                        "center": [cell["center_lon"], cell["center_lat"]],
                        "count": cell["count"],
                        "z_score": z_score,
                        "points": [point.id for point in cell["points"]]
                    })
                elif z_score < -1.96:
                    coldspots.append({
                        "grid_key": grid_key,
                        "center": [cell["center_lon"], cell["center_lat"]],
                        "count": cell["count"],
                        "z_score": z_score,
                        "points": [point.id for point in cell["points"]]
                    })
            
            results["hotspot_analysis"] = {
                "grid_size": grid_size,
                "total_grid_cells": len(grid_cells),
                "global_mean": global_mean,
                "global_std": global_std,
                "hotspots": hotspots,
                "coldspots": coldspots,
                "hotspot_count": len(hotspots),
                "coldspot_count": len(coldspots)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing hotspots: {e}")
            return {}
    
    async def _analyze_interpolation(
        self,
        points: List[SpatialPoint],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar interpolación espacial"""
        try:
            results = {}
            
            if len(points) < 3:
                return {"error": "Insufficient points for interpolation analysis"}
            
            # Verificar si los puntos tienen valores para interpolar
            has_values = any(hasattr(point, 'value') or 'value' in point.attributes for point in points)
            
            if not has_values:
                return {"error": "No values found for interpolation"}
            
            # Extraer valores
            coordinates = []
            values = []
            
            for point in points:
                coordinates.append([point.longitude, point.latitude])
                if hasattr(point, 'value'):
                    values.append(point.value)
                elif 'value' in point.attributes:
                    values.append(point.attributes['value'])
                else:
                    values.append(0)
            
            coordinates = np.array(coordinates)
            values = np.array(values)
            
            # Interpolación por IDW (Inverse Distance Weighting)
            grid_size = parameters.get("grid_size", 0.01) if parameters else 0.01
            power = parameters.get("power", 2) if parameters else 2
            
            # Crear grid de interpolación
            lon_min, lon_max = min(coordinates[:, 0]), max(coordinates[:, 0])
            lat_min, lat_max = min(coordinates[:, 1]), max(coordinates[:, 1])
            
            # Extender ligeramente el grid
            lon_range = lon_max - lon_min
            lat_range = lat_max - lat_min
            lon_min -= lon_range * 0.1
            lon_max += lon_range * 0.1
            lat_min -= lat_range * 0.1
            lat_max += lat_range * 0.1
            
            # Crear puntos de grid
            lon_grid = np.arange(lon_min, lon_max, grid_size)
            lat_grid = np.arange(lat_min, lat_max, grid_size)
            
            interpolated_values = np.zeros((len(lat_grid), len(lon_grid)))
            
            for i, lat in enumerate(lat_grid):
                for j, lon in enumerate(lon_grid):
                    # Calcular IDW
                    distances = np.sqrt((coordinates[:, 0] - lon)**2 + (coordinates[:, 1] - lat)**2)
                    
                    # Evitar división por cero
                    distances = np.where(distances == 0, 1e-10, distances)
                    
                    weights = 1 / (distances ** power)
                    interpolated_values[i, j] = np.sum(weights * values) / np.sum(weights)
            
            results["interpolation_analysis"] = {
                "method": "idw",
                "power": power,
                "grid_size": grid_size,
                "grid_shape": interpolated_values.shape,
                "lon_range": [lon_min, lon_max],
                "lat_range": [lat_min, lat_max],
                "interpolated_values": interpolated_values.tolist(),
                "min_value": np.min(interpolated_values),
                "max_value": np.max(interpolated_values),
                "mean_value": np.mean(interpolated_values)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing interpolation: {e}")
            return {}
    
    async def _analyze_buffer_analysis(
        self,
        points: List[SpatialPoint],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analizar análisis de buffer"""
        try:
            results = {}
            
            buffer_distance = parameters.get("buffer_distance", self.config["default_buffer_distance"]) if parameters else self.config["default_buffer_distance"]
            
            # Crear buffers alrededor de cada punto
            buffers = []
            
            for point in points:
                # Crear buffer circular (simplificado)
                buffer_info = {
                    "point_id": point.id,
                    "center": [point.longitude, point.latitude],
                    "radius": buffer_distance,
                    "area": math.pi * (buffer_distance ** 2)  # Área aproximada
                }
                buffers.append(buffer_info)
            
            # Análisis de solapamiento
            overlaps = []
            for i, buffer1 in enumerate(buffers):
                for j, buffer2 in enumerate(buffers[i+1:], i+1):
                    # Calcular distancia entre centros
                    distance = self._calculate_distance(
                        SpatialPoint("", buffer1["center"][0], buffer1["center"][1]),
                        SpatialPoint("", buffer2["center"][0], buffer2["center"][1])
                    )
                    
                    # Verificar solapamiento
                    if distance < (buffer1["radius"] + buffer2["radius"]):
                        overlap_area = self._calculate_overlap_area(buffer1, buffer2)
                        overlaps.append({
                            "buffer1_id": buffer1["point_id"],
                            "buffer2_id": buffer2["point_id"],
                            "distance": distance,
                            "overlap_area": overlap_area
                        })
            
            results["buffer_analysis"] = {
                "buffer_distance": buffer_distance,
                "total_buffers": len(buffers),
                "total_overlaps": len(overlaps),
                "overlaps": overlaps,
                "coverage_area": sum(buffer["area"] for buffer in buffers),
                "overlap_percentage": (len(overlaps) / (len(buffers) * (len(buffers) - 1) / 2)) * 100 if len(buffers) > 1 else 0
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing buffer analysis: {e}")
            return {}
    
    async def _analyze_general_spatial(
        self,
        points: List[SpatialPoint],
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Análisis espacial general"""
        try:
            results = {}
            
            # Estadísticas básicas
            longitudes = [point.longitude for point in points]
            latitudes = [point.latitude for point in points]
            
            results["basic_statistics"] = {
                "point_count": len(points),
                "longitude_stats": {
                    "min": min(longitudes),
                    "max": max(longitudes),
                    "mean": np.mean(longitudes),
                    "std": np.std(longitudes)
                },
                "latitude_stats": {
                    "min": min(latitudes),
                    "max": max(latitudes),
                    "mean": np.mean(latitudes),
                    "std": np.std(latitudes)
                },
                "spatial_extent": self._calculate_spatial_extent(points),
                "area_coverage": self._calculate_area_coverage(points)
            }
            
            # Análisis temporal si hay timestamps
            temporal_points = [point for point in points if point.timestamp]
            if temporal_points:
                timestamps = [point.timestamp for point in temporal_points]
                results["temporal_analysis"] = {
                    "temporal_points": len(temporal_points),
                    "time_span": (max(timestamps) - min(timestamps)).total_seconds(),
                    "earliest": min(timestamps).isoformat(),
                    "latest": max(timestamps).isoformat()
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in general spatial analysis: {e}")
            return {}
    
    def _calculate_distance(self, point1: SpatialPoint, point2: SpatialPoint) -> float:
        """Calcular distancia entre dos puntos (Haversine)"""
        try:
            # Radio de la Tierra en metros
            R = 6371000
            
            # Convertir a radianes
            lat1_rad = math.radians(point1.latitude)
            lat2_rad = math.radians(point2.latitude)
            delta_lat = math.radians(point2.latitude - point1.latitude)
            delta_lon = math.radians(point2.longitude - point1.longitude)
            
            # Fórmula de Haversine
            a = (math.sin(delta_lat / 2) ** 2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * 
                 math.sin(delta_lon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            return R * c
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def _calculate_area_coverage(self, points: List[SpatialPoint]) -> float:
        """Calcular área de cobertura"""
        try:
            if len(points) < 3:
                return 0.0
            
            # Usar convex hull para calcular área
            from scipy.spatial import ConvexHull
            
            coordinates = np.array([[point.longitude, point.latitude] for point in points])
            
            if len(coordinates) >= 3:
                hull = ConvexHull(coordinates)
                # Convertir área a metros cuadrados (aproximado)
                area_deg2 = hull.volume
                # Conversión aproximada: 1 grado ≈ 111,000 metros
                area_m2 = area_deg2 * (111000 ** 2)
                return area_m2
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating area coverage: {e}")
            return 0.0
    
    def _find_optimal_clusters(self, inertias: List[float], k_range: range) -> int:
        """Encontrar número óptimo de clusters usando método del codo"""
        try:
            if len(inertias) < 2:
                return 2
            
            # Calcular segunda derivada
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)
            
            # Encontrar el punto de máxima curvatura
            if second_derivatives:
                optimal_idx = np.argmax(second_derivatives) + 1
                return k_range[optimal_idx]
            else:
                return k_range[0]
                
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {e}")
            return 2
    
    def _calculate_silhouette_score(self, coordinates: np.ndarray, labels: np.ndarray) -> float:
        """Calcular silhouette score"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(coordinates, labels)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            return 0.0
    
    def _calculate_overlap_area(self, buffer1: Dict, buffer2: Dict) -> float:
        """Calcular área de solapamiento entre dos buffers"""
        try:
            # Implementación simplificada
            distance = self._calculate_distance(
                SpatialPoint("", buffer1["center"][0], buffer1["center"][1]),
                SpatialPoint("", buffer2["center"][0], buffer2["center"][1])
            )
            
            r1, r2 = buffer1["radius"], buffer2["radius"]
            
            if distance >= r1 + r2:
                return 0.0  # No overlap
            elif distance <= abs(r1 - r2):
                return math.pi * min(r1, r2) ** 2  # One circle inside the other
            else:
                # Partial overlap
                a = (r1**2 - r2**2 + distance**2) / (2 * distance)
                h = math.sqrt(r1**2 - a**2)
                return r1**2 * math.acos(a/r1) + r2**2 * math.acos((distance-a)/r2) - distance * h
                
        except Exception as e:
            logger.error(f"Error calculating overlap area: {e}")
            return 0.0
    
    def _calculate_spatial_statistics(
        self,
        points: List[SpatialPoint],
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcular estadísticas espaciales"""
        try:
            statistics = {}
            
            # Estadísticas básicas
            longitudes = [point.longitude for point in points]
            latitudes = [point.latitude for point in points]
            
            statistics["spatial_variance"] = np.var(longitudes) + np.var(latitudes)
            statistics["spatial_std"] = math.sqrt(statistics["spatial_variance"])
            statistics["spatial_range"] = max(longitudes) - min(longitudes) + max(latitudes) - min(latitudes)
            
            # Estadísticas de clustering si están disponibles
            if "clustering_analysis" in results:
                clustering = results["clustering_analysis"]
                statistics["clustering_quality"] = clustering.get("silhouette_score", 0)
                statistics["optimal_clusters"] = clustering.get("optimal_clusters", 0)
            
            # Estadísticas de hotspots si están disponibles
            if "hotspot_analysis" in results:
                hotspot = results["hotspot_analysis"]
                statistics["hotspot_ratio"] = hotspot.get("hotspot_count", 0) / max(hotspot.get("total_grid_cells", 1), 1)
                statistics["spatial_autocorrelation"] = hotspot.get("global_std", 0) / max(hotspot.get("global_mean", 1), 1)
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating spatial statistics: {e}")
            return {}
    
    def _generate_spatial_insights(
        self,
        points: List[SpatialPoint],
        results: Dict[str, Any],
        analysis_type: SpatialAnalysisType
    ) -> List[str]:
        """Generar insights espaciales"""
        try:
            insights = []
            
            # Insight sobre densidad
            if "density_analysis" in results:
                density = results["density_analysis"]
                max_density = density.get("max_density", 0)
                avg_density = density.get("avg_density", 0)
                
                if max_density > avg_density * 3:
                    insights.append(f"Alta concentración espacial detectada: densidad máxima {max_density} vs promedio {avg_density:.1f}")
            
            # Insight sobre clustering
            if "clustering_analysis" in results:
                clustering = results["clustering_analysis"]
                optimal_clusters = clustering.get("optimal_clusters", 0)
                silhouette_score = clustering.get("silhouette_score", 0)
                
                if silhouette_score > 0.5:
                    insights.append(f"Patrones de clustering bien definidos: {optimal_clusters} clusters con calidad {silhouette_score:.2f}")
                elif optimal_clusters > 5:
                    insights.append(f"Distribución espacial fragmentada: {optimal_clusters} clusters identificados")
            
            # Insight sobre hotspots
            if "hotspot_analysis" in results:
                hotspot = results["hotspot_analysis"]
                hotspot_count = hotspot.get("hotspot_count", 0)
                total_cells = hotspot.get("total_grid_cells", 1)
                
                if hotspot_count > 0:
                    hotspot_ratio = hotspot_count / total_cells
                    insights.append(f"Hotspots espaciales identificados: {hotspot_count} de {total_cells} celdas ({hotspot_ratio:.1%})")
            
            # Insight sobre cobertura espacial
            if "basic_statistics" in results:
                basic = results["basic_statistics"]
                area_coverage = basic.get("area_coverage", 0)
                
                if area_coverage > 0:
                    density = len(points) / area_coverage
                    insights.append(f"Densidad espacial: {density:.2e} puntos por m² en área de {area_coverage:.0f} m²")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating spatial insights: {e}")
            return []
    
    async def create_visualization(
        self,
        dataset_id: str,
        visualization_type: str = "interactive_map",
        output_format: str = "html"
    ) -> str:
        """
        Crear visualización geoespacial
        
        Args:
            dataset_id: ID del dataset
            visualization_type: Tipo de visualización
            output_format: Formato de salida
            
        Returns:
            Ruta del archivo de visualización
        """
        try:
            if not self.enable_geospatial:
                raise ValueError("Geospatial analysis not available")
            
            if dataset_id not in self.spatial_points:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            points = self.spatial_points[dataset_id]
            if not points:
                raise ValueError(f"No points found in dataset {dataset_id}")
            
            logger.info(f"Creating {visualization_type} visualization for dataset {dataset_id}")
            
            if visualization_type == "interactive_map":
                return await self._create_interactive_map(points, dataset_id, output_format)
            elif visualization_type == "heatmap":
                return await self._create_heatmap(points, dataset_id, output_format)
            elif visualization_type == "clustering_map":
                return await self._create_clustering_map(points, dataset_id, output_format)
            else:
                return await self._create_basic_map(points, dataset_id, output_format)
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return ""
    
    async def _create_interactive_map(
        self,
        points: List[SpatialPoint],
        dataset_id: str,
        output_format: str
    ) -> str:
        """Crear mapa interactivo"""
        try:
            if not self.enable_geospatial:
                return ""
            
            # Crear mapa con Folium
            center_lat = np.mean([point.latitude for point in points])
            center_lon = np.mean([point.longitude for point in points])
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=self.config["visualization_zoom"]
            )
            
            # Agregar puntos
            for point in points:
                popup_text = f"ID: {point.id}<br>Lat: {point.latitude:.6f}<br>Lon: {point.longitude:.6f}"
                if point.timestamp:
                    popup_text += f"<br>Time: {point.timestamp}"
                
                folium.Marker(
                    [point.latitude, point.longitude],
                    popup=popup_text,
                    tooltip=point.id
                ).add_to(m)
            
            # Agregar plugins
            folium.plugins.HeatMap(
                [[point.latitude, point.longitude] for point in points]
            ).add_to(m)
            
            # Guardar mapa
            output_path = f"exports/geospatial_map_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            m.save(output_path)
            logger.info(f"Interactive map saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {e}")
            return ""
    
    async def _create_heatmap(
        self,
        points: List[SpatialPoint],
        dataset_id: str,
        output_format: str
    ) -> str:
        """Crear mapa de calor"""
        try:
            # Crear visualización con Plotly
            fig = go.Figure()
            
            # Agregar puntos
            fig.add_trace(go.Scattermapbox(
                lat=[point.latitude for point in points],
                lon=[point.longitude for point in points],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.7
                ),
                text=[point.id for point in points],
                hovertemplate='<b>%{text}</b><br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=f'Heatmap: {dataset_id}',
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=np.mean([point.latitude for point in points]),
                        lon=np.mean([point.longitude for point in points])
                    ),
                    zoom=10
                ),
                height=600
            )
            
            # Guardar archivo
            output_path = f"exports/geospatial_heatmap_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fig.write_html(output_path)
            logger.info(f"Heatmap saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return ""
    
    async def _create_clustering_map(
        self,
        points: List[SpatialPoint],
        dataset_id: str,
        output_format: str
    ) -> str:
        """Crear mapa de clustering"""
        try:
            # Realizar clustering
            coordinates = np.array([[point.longitude, point.latitude] for point in points])
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(5, len(points)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            
            # Crear visualización
            fig = go.Figure()
            
            # Agregar puntos por cluster
            colors = px.colors.qualitative.Set1
            for cluster_id in range(kmeans.n_clusters):
                cluster_points = [points[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                fig.add_trace(go.Scattermapbox(
                    lat=[point.latitude for point in cluster_points],
                    lon=[point.longitude for point in cluster_points],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[cluster_id % len(colors)],
                        opacity=0.7
                    ),
                    name=f'Cluster {cluster_id}',
                    text=[point.id for point in cluster_points],
                    hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_id) + '<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>'
                ))
            
            # Agregar centroides
            fig.add_trace(go.Scattermapbox(
                lat=kmeans.cluster_centers_[:, 1],
                lon=kmeans.cluster_centers_[:, 0],
                mode='markers',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='x'
                ),
                name='Centroids',
                hovertemplate='Centroid<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>'
            ))
            
            # Configurar layout
            fig.update_layout(
                title=f'Clustering Map: {dataset_id}',
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=np.mean([point.latitude for point in points]),
                        lon=np.mean([point.longitude for point in points])
                    ),
                    zoom=10
                ),
                height=600
            )
            
            # Guardar archivo
            output_path = f"exports/geospatial_clustering_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fig.write_html(output_path)
            logger.info(f"Clustering map saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating clustering map: {e}")
            return ""
    
    async def _create_basic_map(
        self,
        points: List[SpatialPoint],
        dataset_id: str,
        output_format: str
    ) -> str:
        """Crear mapa básico"""
        try:
            # Crear visualización simple con Matplotlib
            plt.figure(figsize=(12, 8))
            
            longitudes = [point.longitude for point in points]
            latitudes = [point.latitude for point in points]
            
            plt.scatter(longitudes, latitudes, alpha=0.6, s=50)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Spatial Distribution: {dataset_id}')
            plt.grid(True, alpha=0.3)
            
            # Agregar etiquetas para algunos puntos
            for i, point in enumerate(points[:10]):  # Solo primeros 10 puntos
                plt.annotate(point.id, (point.longitude, point.latitude), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Guardar archivo
            output_path = f"exports/geospatial_basic_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Basic map saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating basic map: {e}")
            return ""
    
    async def get_geospatial_summary(self) -> Dict[str, Any]:
        """Obtener resumen de análisis geoespacial"""
        try:
            return {
                "total_datasets": len(self.spatial_points),
                "total_points": sum(len(points) for points in self.spatial_points.values()),
                "total_polygons": sum(len(polygons) for polygons in self.spatial_polygons.values()),
                "total_analyses": len(self.spatial_analyses),
                "total_insights": len(self.spatiotemporal_insights),
                "geospatial_available": self.enable_geospatial,
                "coordinate_system": self.coordinate_system.value,
                "analysis_types": {
                    analysis_type.value: len([a for a in self.spatial_analyses.values() if a.analysis_type == analysis_type])
                    for analysis_type in SpatialAnalysisType
                },
                "last_analysis": max([analysis.created_at for analysis in self.spatial_analyses.values()]).isoformat() if self.spatial_analyses else None
            }
        except Exception as e:
            logger.error(f"Error getting geospatial summary: {e}")
            return {}
    
    async def export_geospatial_data(self, filepath: str = None) -> str:
        """Exportar datos geoespaciales"""
        try:
            if filepath is None:
                filepath = f"exports/geospatial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "spatial_points": {
                    dataset_id: [
                        {
                            "id": point.id,
                            "longitude": point.longitude,
                            "latitude": point.latitude,
                            "elevation": point.elevation,
                            "timestamp": point.timestamp.isoformat() if point.timestamp else None,
                            "attributes": point.attributes
                        }
                        for point in points
                    ]
                    for dataset_id, points in self.spatial_points.items()
                },
                "spatial_polygons": {
                    dataset_id: [
                        {
                            "id": polygon.id,
                            "coordinates": polygon.coordinates,
                            "attributes": polygon.attributes
                        }
                        for polygon in polygons
                    ]
                    for dataset_id, polygons in self.spatial_polygons.items()
                },
                "spatial_analyses": {
                    analysis_id: {
                        "analysis_type": analysis.analysis_type.value,
                        "spatial_extent": analysis.spatial_extent,
                        "point_count": analysis.point_count,
                        "polygon_count": analysis.polygon_count,
                        "results": analysis.results,
                        "statistics": analysis.statistics,
                        "insights": analysis.insights,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.spatial_analyses.items()
                },
                "spatiotemporal_insights": {
                    insight_id: {
                        "insight_type": insight.insight_type,
                        "description": insight.description,
                        "spatial_extent": insight.spatial_extent,
                        "temporal_extent": [insight.temporal_extent[0].isoformat(), insight.temporal_extent[1].isoformat()],
                        "significance": insight.significance,
                        "confidence": insight.confidence,
                        "related_features": insight.related_features,
                        "implications": insight.implications,
                        "recommendations": insight.recommendations,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.spatiotemporal_insights.items()
                },
                "summary": await self.get_geospatial_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Geospatial data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting geospatial data: {e}")
            raise
























