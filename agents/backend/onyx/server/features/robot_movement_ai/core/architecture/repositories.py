"""
Repository Layer
================

Implementación del patrón Repository para persistencia.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .interfaces import IRouteRepository, RouteResponse
from .domain import Route, RouteStatus, RouteMetrics

logger = logging.getLogger(__name__)


class RouteRepository:
    """
    Repositorio en memoria para rutas.
    
    En producción, implementar con base de datos real.
    """
    
    def __init__(self):
        """Inicializar repositorio."""
        self._routes: Dict[str, Route] = {}
        self._index: Dict[str, List[str]] = {}  # start -> [route_ids]
    
    def save_route(self, response: RouteResponse) -> str:
        """
        Guardar ruta.
        
        Args:
            response: Response de ruta
            
        Returns:
            ID de la ruta guardada
        """
        route_id = response.metadata.get("route_id") if response.metadata else None
        if not route_id:
            route_id = str(uuid.uuid4())
        
        # Crear entidad Route
        route = Route(
            id=route_id,
            start_node=response.metadata.get("start_node", ""),
            end_node=response.metadata.get("end_node", ""),
            path=response.route,
            metrics=RouteMetrics(
                distance=response.metrics.get("distance", 0.0),
                time=response.metrics.get("time", 0.0),
                cost=response.metrics.get("cost", 0.0),
                efficiency=response.metrics.get("efficiency", 0.0),
                reliability=response.metrics.get("reliability", 0.0),
                confidence=response.confidence,
                metadata=response.metrics
            ),
            strategy=response.metadata.get("strategy", ""),
            status=RouteStatus.COMPLETED,
            metadata=response.metadata or {}
        )
        
        # Guardar
        self._routes[route_id] = route
        
        # Indexar
        start = route.start_node
        if start not in self._index:
            self._index[start] = []
        if route_id not in self._index[start]:
            self._index[start].append(route_id)
        
        logger.debug(f"Ruta guardada: {route_id}")
        return route_id
    
    def get_route(self, route_id: str) -> Optional[RouteResponse]:
        """
        Obtener ruta por ID.
        
        Args:
            route_id: ID de la ruta
            
        Returns:
            Response de ruta o None
        """
        route = self._routes.get(route_id)
        if not route:
            return None
        
        # Convertir a Response
        return RouteResponse(
            route=route.path,
            metrics=route.metrics.to_dict(),
            confidence=route.metrics.confidence,
            metadata={
                "route_id": route.id,
                "start_node": route.start_node,
                "end_node": route.end_node,
                "strategy": route.strategy,
                **route.metadata
            }
        )
    
    def find_routes(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> List[RouteResponse]:
        """
        Buscar rutas.
        
        Args:
            start: Nodo de inicio (opcional)
            end: Nodo de fin (opcional)
            limit: Límite de resultados
            
        Returns:
            Lista de responses
        """
        results = []
        
        if start:
            # Buscar por nodo de inicio
            route_ids = self._index.get(start, [])
            for route_id in route_ids[:limit]:
                route = self._routes.get(route_id)
                if route:
                    if end is None or route.end_node == end:
                        results.append(self.get_route(route_id))
        else:
            # Buscar todas
            for route in list(self._routes.values())[:limit]:
                if end is None or route.end_node == end:
                    results.append(self.get_route(route.id))
        
        return results
    
    def delete_route(self, route_id: str) -> bool:
        """
        Eliminar ruta.
        
        Args:
            route_id: ID de la ruta
            
        Returns:
            True si se eliminó, False si no existe
        """
        route = self._routes.pop(route_id, None)
        if route:
            # Remover del índice
            start = route.start_node
            if start in self._index:
                self._index[start] = [
                    rid for rid in self._index[start] if rid != route_id
                ]
            return True
        return False
    
    def count(self) -> int:
        """Contar rutas."""
        return len(self._routes)
    
    def clear(self):
        """Limpiar todas las rutas."""
        self._routes.clear()
        self._index.clear()


class ModelRepository:
    """
    Repositorio para modelos.
    """
    
    def __init__(self):
        """Inicializar repositorio."""
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def save_model(
        self,
        name: str,
        model: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Guardar modelo.
        
        Args:
            name: Nombre del modelo
            model: Modelo
            metadata: Metadatos (opcional)
        """
        self._models[name] = model
        self._metadata[name] = metadata or {}
        logger.debug(f"Modelo guardado: {name}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """
        Obtener modelo.
        
        Args:
            name: Nombre del modelo
            
        Returns:
            Modelo o None
        """
        return self._models.get(name)
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Obtener metadatos del modelo."""
        return self._metadata.get(name, {})
    
    def list_models(self) -> List[str]:
        """Listar nombres de modelos."""
        return list(self._models.keys())
    
    def delete_model(self, name: str) -> bool:
        """Eliminar modelo."""
        if name in self._models:
            del self._models[name]
            del self._metadata[name]
            return True
        return False


class DataRepository:
    """
    Repositorio para datos.
    """
    
    def __init__(self):
        """Inicializar repositorio."""
        self._datasets: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def save_dataset(
        self,
        name: str,
        dataset: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Guardar dataset."""
        self._datasets[name] = dataset
        self._metadata[name] = metadata or {}
    
    def get_dataset(self, name: str) -> Optional[Any]:
        """Obtener dataset."""
        return self._datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """Listar datasets."""
        return list(self._datasets.keys())

