"""
Serialization Utilities
=======================

Utilidades para serialización y deserialización de objetos.
Optimizado con orjson y msgpack para máximo rendimiento.
"""

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

import json

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

import pickle
from typing import Any, Dict, Optional, Type, TypeVar
from pathlib import Path
import numpy as np

try:
    from ..optimization.trajectory_optimizer import TrajectoryPoint, TrajectoryOptimizer
except ImportError:
    # Fallback if trajectory_optimizer not available
    TrajectoryPoint = None
    TrajectoryOptimizer = None
# Serializable type stub
try:
    from .types import Serializable
except ImportError:
    from typing import Protocol
    class Serializable(Protocol):
        """Protocol for serializable objects."""
        pass

T = TypeVar('T')


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON para arrays de NumPy."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def serialize_trajectory(
    trajectory: list,
    filepath: str,
    format: str = "json"
) -> bool:
    """
    Serializar trayectoria a archivo.
    
    Args:
        trajectory: Lista de TrajectoryPoint
        filepath: Ruta al archivo
        format: Formato ("json" o "pickle")
        
    Returns:
        True si fue exitoso
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            data = {
                "trajectory": [
                    {
                        "position": point.position.tolist(),
                        "orientation": point.orientation.tolist(),
                        "velocity": point.velocity.tolist() if point.velocity is not None else None,
                        "acceleration": point.acceleration.tolist() if point.acceleration is not None else None,
                        "timestamp": point.timestamp
                    }
                    for point in trajectory
                ]
            }
            if ORJSON_AVAILABLE:
                with open(path, 'wb') as f:
                    f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2))
            else:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        elif format.lower() == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(trajectory, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return True
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to serialize trajectory: {e}")
        return False


def deserialize_trajectory(
    filepath: str,
    format: Optional[str] = None
) -> Optional[list]:
    """
    Deserializar trayectoria desde archivo.
    
    Args:
        filepath: Ruta al archivo
        format: Formato ("json" o "pickle"), None para auto-detectar
        
    Returns:
        Lista de TrajectoryPoint o None si falla
    """
    try:
        path = Path(filepath)
        
        # Auto-detectar formato
        if format is None:
            if path.suffix == ".json":
                format = "json"
            elif path.suffix in [".pkl", ".pickle"]:
                format = "pickle"
            else:
                raise ValueError(f"Cannot auto-detect format for {path.suffix}")
        
        if format.lower() == "json":
            if ORJSON_AVAILABLE:
                with open(path, 'rb') as f:
                    data = orjson.loads(f.read())
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
            
            trajectory = []
            for point_data in data.get("trajectory", []):
                point = TrajectoryPoint(
                    position=np.array(point_data["position"]),
                    orientation=np.array(point_data["orientation"]),
                    velocity=np.array(point_data["velocity"]) if point_data.get("velocity") else None,
                    acceleration=np.array(point_data["acceleration"]) if point_data.get("acceleration") else None,
                    timestamp=point_data.get("timestamp", 0.0)
                )
                trajectory.append(point)
            
            return trajectory
        
        elif format.lower() == "pickle":
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to deserialize trajectory: {e}")
        return None


def serialize_config(config: Any, filepath: str) -> bool:
    """
    Serializar configuración a archivo JSON.
    
    Args:
        config: Objeto de configuración con método to_dict()
        filepath: Ruta al archivo
        
    Returns:
        True si fue exitoso
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(config, 'to_dict'):
            data = config.to_dict()
        else:
            data = dict(config) if isinstance(config, dict) else config.__dict__
        
        if ORJSON_AVAILABLE:
            with open(path, 'wb') as f:
                f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2))
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        return True
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to serialize config: {e}")
        return False


def deserialize_config(
    filepath: str,
    config_class: Type[T]
) -> Optional[T]:
    """
    Deserializar configuración desde archivo JSON.
    
    Args:
        filepath: Ruta al archivo
        config_class: Clase de configuración
        
    Returns:
        Instancia de configuración o None si falla
    """
    try:
        path = Path(filepath)
        
        if ORJSON_AVAILABLE:
            with open(path, 'rb') as f:
                data = orjson.loads(f.read())
        else:
            with open(path, 'r') as f:
                data = json.load(f)
        
        if hasattr(config_class, 'from_dict'):
            return config_class.from_dict(data)
        else:
            return config_class(**data)
    
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to deserialize config: {e}")
        return None


def to_json(obj: Any, indent: int = 2) -> str:
    """
    Convertir objeto a JSON string.
    Usa orjson si está disponible para máximo rendimiento.
    
    Args:
        obj: Objeto a serializar
        indent: Indentación
        
    Returns:
        String JSON
    """
    if ORJSON_AVAILABLE:
        options = orjson.OPT_SERIALIZE_NUMPY
        if indent > 0:
            options |= orjson.OPT_INDENT_2
        return orjson.dumps(obj, option=options).decode('utf-8')
    else:
        return json.dumps(obj, indent=indent, cls=NumpyEncoder)


def from_json(json_str: str) -> Any:
    """
    Convertir JSON string a objeto.
    Usa orjson si está disponible para máximo rendimiento.
    
    Args:
        json_str: String JSON
        
    Returns:
        Objeto deserializado
    """
    if ORJSON_AVAILABLE:
        if isinstance(json_str, str):
            return orjson.loads(json_str.encode('utf-8'))
        return orjson.loads(json_str)
    else:
        return json.loads(json_str)






