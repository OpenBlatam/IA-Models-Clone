"""
Point Cloud Processor for Humanoid Robot (Optimizado)
=====================================================

Procesamiento profesional de nubes de puntos usando PCL y Open3D.
Incluye validaciones robustas, manejo de errores mejorado, y optimizaciones.
"""

import logging
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logging.warning("Open3D not available. Install with: pip install open3d")

try:
    import pcl
    PCL_AVAILABLE = True
except ImportError:
    PCL_AVAILABLE = False
    logging.warning("PCL not available.")

logger = logging.getLogger(__name__)


class PointCloudError(Exception):
    """Excepción personalizada para errores de procesamiento de nubes de puntos."""
    pass


class PointCloudProcessor:
    """
    Procesador de nubes de puntos para robot humanoide.
    
    Incluye filtrado, segmentación, y detección de obstáculos.
    """
    
    def __init__(self):
        """Inicializar procesador de nubes de puntos."""
        self.available = OPEN3D_AVAILABLE or PCL_AVAILABLE
        self.use_open3d = OPEN3D_AVAILABLE
        
        if not self.available:
            logger.warning("Point cloud processing not available")
        else:
            logger.info(f"Point cloud processor initialized (Open3D: {OPEN3D_AVAILABLE}, PCL: {PCL_AVAILABLE})")
    
    def filter_point_cloud(
        self,
        points: Union[np.ndarray, List[List[float]]],
        method: str = "statistical",
        **kwargs
    ) -> np.ndarray:
        """
        Filtrar nube de puntos (optimizado).
        
        Args:
            points: Array de puntos (N, 3) o lista de listas
            method: Método de filtrado ("statistical", "radius")
            **kwargs: Parámetros adicionales
            
        Returns:
            Puntos filtrados como numpy array
            
        Raises:
            ValueError: Si los parámetros son inválidos
            PointCloudError: Si hay error en el filtrado
        """
        if not self.available:
            logger.warning("Point cloud processing not available, returning original points")
            return np.array(points, dtype=np.float64) if not isinstance(points, np.ndarray) else points
        
        # Validar y convertir puntos
        try:
            points_array = np.array(points, dtype=np.float64)
            if len(points_array.shape) != 2 or points_array.shape[1] != 3:
                raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
            
            if points_array.shape[0] == 0:
                raise ValueError("points array cannot be empty")
            
            if not np.all(np.isfinite(points_array)):
                raise ValueError("All point coordinates must be finite numbers")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid points array: {e}") from e
        
        # Validar método
        valid_methods = ["statistical", "radius"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")
        
        try:
            if self.use_open3d:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_array)
                
                if method == "statistical":
                    nb_neighbors = kwargs.get("nb_neighbors", 20)
                    std_ratio = kwargs.get("std_ratio", 2.0)
                    
                    if not isinstance(nb_neighbors, int) or nb_neighbors < 1:
                        raise ValueError(f"nb_neighbors must be a positive integer, got {nb_neighbors}")
                    if not isinstance(std_ratio, (int, float)) or std_ratio <= 0:
                        raise ValueError(f"std_ratio must be a positive number, got {std_ratio}")
                    
                    pcd, _ = pcd.remove_statistical_outlier(
                        nb_neighbors=nb_neighbors,
                        std_ratio=float(std_ratio)
                    )
                elif method == "radius":
                    nb_points = kwargs.get("nb_points", 16)
                    radius = kwargs.get("radius", 0.05)
                    
                    if not isinstance(nb_points, int) or nb_points < 1:
                        raise ValueError(f"nb_points must be a positive integer, got {nb_points}")
                    if not isinstance(radius, (int, float)) or radius <= 0:
                        raise ValueError(f"radius must be a positive number, got {radius}")
                    
                    pcd, _ = pcd.remove_radius_outlier(
                        nb_points=nb_points,
                        radius=float(radius)
                    )
                
                filtered_points = np.asarray(pcd.points)
                logger.debug(f"Filtered point cloud: {points_array.shape[0]} -> {filtered_points.shape[0]} points")
                return filtered_points
            else:
                # Fallback: filtrado simple
                logger.warning("Open3D not available, returning original points")
                return points_array
        except Exception as e:
            logger.error(f"Error filtering point cloud: {e}", exc_info=True)
            raise PointCloudError(f"Failed to filter point cloud: {str(e)}") from e
    
    def detect_obstacles(
        self, 
        points: Union[np.ndarray, List[List[float]]],
        min_points: int = 10,
        max_distance: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detectar obstáculos en nube de puntos (optimizado).
        
        Args:
            points: Array de puntos (N, 3) o lista de listas
            min_points: Número mínimo de puntos para considerar un obstáculo
            max_distance: Distancia máxima para clustering (metros)
            
        Returns:
            Lista de obstáculos detectados con posición y tamaño
            
        Raises:
            ValueError: Si los parámetros son inválidos
            PointCloudError: Si hay error en la detección
        """
        if not self.available:
            logger.warning("Point cloud processing not available")
            return []
        
        # Validar y convertir puntos
        try:
            points_array = np.array(points, dtype=np.float64)
            if len(points_array.shape) != 2 or points_array.shape[1] != 3:
                raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
            
            if points_array.shape[0] == 0:
                return []
            
            if not np.all(np.isfinite(points_array)):
                raise ValueError("All point coordinates must be finite numbers")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid points array: {e}") from e
        
        # Validar parámetros
        if not isinstance(min_points, int) or min_points < 1:
            raise ValueError(f"min_points must be a positive integer, got {min_points}")
        
        if not isinstance(max_distance, (int, float)) or max_distance <= 0:
            raise ValueError(f"max_distance must be a positive number, got {max_distance}")
        
        try:
            if self.use_open3d:
                # Usar DBSCAN para clustering
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_array)
                
                # Clustering simple basado en distancia
                labels = np.array(pcd.cluster_dbscan(eps=max_distance, min_points=min_points))
                
                obstacles = []
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[unique_labels >= 0]  # Remover ruido (-1)
                
                for label in unique_labels:
                    cluster_points = points_array[labels == label]
                    if len(cluster_points) >= min_points:
                        # Calcular centro y tamaño del obstáculo
                        center = np.mean(cluster_points, axis=0)
                        size = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                        
                        obstacles.append({
                            "id": int(label),
                            "center": center.tolist(),
                            "size": size.tolist(),
                            "point_count": len(cluster_points),
                            "bounds": {
                                "min": np.min(cluster_points, axis=0).tolist(),
                                "max": np.max(cluster_points, axis=0).tolist()
                            }
                        })
                
                logger.debug(f"Detected {len(obstacles)} obstacles from {points_array.shape[0]} points")
                return obstacles
            else:
                # Fallback simple: detectar puntos cercanos al origen
                obstacles = []
                for i, point in enumerate(points_array):
                    if np.linalg.norm(point) < max_distance:
                        obstacles.append({
                            "id": i,
                            "center": point.tolist(),
                            "size": [0.05, 0.05, 0.05],
                            "point_count": 1
                        })
                return obstacles[:10]  # Limitar a 10 obstáculos
                
        except Exception as e:
            logger.error(f"Error detecting obstacles: {e}", exc_info=True)
            raise PointCloudError(f"Failed to detect obstacles: {str(e)}") from e
