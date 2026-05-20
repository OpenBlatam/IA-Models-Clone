"""
Deployment Generator - Generador de utilidades de deployment
============================================================

Genera utilidades para deployment y serving de modelos:
- Model serving
- API endpoints optimizados
- Health checks
- Model versioning
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DeploymentGenerator:
    """Generador de utilidades de deployment"""
    
    def __init__(self):
        """Inicializa el generador de deployment"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de deployment.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        deploy_dir = utils_dir / "deployment"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_model_serving(deploy_dir, keywords, project_info)
        self._generate_health_checks(deploy_dir, keywords, project_info)
        self._generate_model_versioning(deploy_dir, keywords, project_info)
        self._generate_deployment_init(deploy_dir, keywords)
    
    def _generate_deployment_init(
        self,
        deploy_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de deployment"""
        
        init_content = '''"""
Deployment Utilities Module
============================

Utilidades para deployment y serving de modelos.
"""

from .model_serving import ModelServer, create_model_server
from .health_checks import HealthChecker, check_model_health
from .model_versioning import ModelVersionManager, load_model_version

__all__ = [
    "ModelServer",
    "create_model_server",
    "HealthChecker",
    "check_model_health",
    "ModelVersionManager",
    "load_model_version",
]
'''
        
        (deploy_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_model_serving(
        self,
        deploy_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de model serving"""
        
        serving_content = '''"""
Model Serving - Utilidades para serving de modelos
==================================================

Sistema de serving optimizado para producción.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Callable
import logging
from pathlib import Path
import time
from threading import Lock

logger = logging.getLogger(__name__)


class ModelServer:
    """
    Servidor de modelos para producción.
    
    Incluye:
    - Carga lazy de modelos
    - Caché de resultados
    - Rate limiting
    - Health checks
    - Métricas de performance
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_path: Optional[Path] = None,
        device: str = "cuda",
        max_batch_size: int = 32,
        enable_cache: bool = True,
        cache_size: int = 1000,
    ):
        """
        Inicializa el servidor de modelos.
        
        Args:
            model: Modelo pre-cargado (opcional)
            model_path: Ruta al modelo (carga lazy)
            device: Dispositivo a usar
            max_batch_size: Tamaño máximo de batch
            enable_cache: Si habilitar caché
            cache_size: Tamaño del caché
        """
        self.device = device
        self.max_batch_size = max_batch_size
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        self.model = model
        self.model_path = Path(model_path) if model_path else None
        self.model_loaded = model is not None
        
        self.cache: Dict[str, Any] = {}
        self.lock = Lock()
        
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_inference_time": 0.0,
            "errors": 0,
        }
        
        if model is not None:
            self._prepare_model()
    
    def _prepare_model(self) -> None:
        """Prepara el modelo para inferencia"""
        if self.model is None:
            return
        
        self.model.eval()
        self.model.to(self.device)
        
        # Optimizaciones
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Compilar si es posible
        try:
            if hasattr(torch, "compile"):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Modelo compilado para serving")
        except Exception as e:
            logger.warning(f"No se pudo compilar modelo: {e}")
    
    def load_model(self) -> None:
        """Carga el modelo (lazy loading)"""
        if self.model_loaded:
            return
        
        if self.model_path is None or not self.model_path.exists():
            raise ValueError(f"Modelo no encontrado: {self.model_path}")
        
        logger.info(f"Cargando modelo desde {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Asumir que el checkpoint tiene 'model_state_dict'
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Necesitarías tener acceso a la clase del modelo
            # Por ahora, asumimos que el modelo ya está inicializado
            if self.model is None:
                raise ValueError("Modelo debe ser proporcionado o inicializado antes de cargar checkpoint")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Checkpoint es el modelo completo
            self.model = checkpoint
        
        self._prepare_model()
        self.model_loaded = True
        logger.info("Modelo cargado exitosamente")
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Genera clave de caché"""
        import hashlib
        import pickle
        
        try:
            key_data = (args, tuple(sorted(kwargs.items())))
            key_bytes = pickle.dumps(key_data)
            return hashlib.md5(key_bytes).hexdigest()
        except Exception:
            return str(id(args)) + str(id(kwargs))
    
    def predict(
        self,
        *args,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        """
        Realiza predicción.
        
        Args:
            *args: Argumentos posicionales
            use_cache: Si usar caché (override)
            **kwargs: Argumentos de palabra clave
        
        Returns:
            Predicción
        """
        if not self.model_loaded:
            self.load_model()
        
        use_cache = use_cache if use_cache is not None else self.enable_cache
        
        # Verificar caché
        if use_cache:
            cache_key = self._get_cache_key(*args, **kwargs)
            with self.lock:
                if cache_key in self.cache:
                    self.stats["cache_hits"] += 1
                    self.stats["total_requests"] += 1
                    return self.cache[cache_key]
                self.stats["cache_misses"] += 1
        
        # Inferencia
        start_time = time.time()
        try:
            with torch.no_grad():
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                output = self.model(*args, **kwargs)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            self.stats["total_inference_time"] += inference_time
            self.stats["total_requests"] += 1
            
            # Guardar en caché
            if use_cache:
                cache_key = self._get_cache_key(*args, **kwargs)
                with self.lock:
                    if len(self.cache) >= self.cache_size:
                        # Remover entrada más antigua
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    self.cache[cache_key] = output
            
            return output
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error en predicción: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del servidor.
        
        Returns:
            Diccionario con estadísticas
        """
        total = self.stats["total_requests"]
        avg_time = (
            self.stats["total_inference_time"] / total
            if total > 0 else 0
        )
        hit_rate = (
            self.stats["cache_hits"] / total * 100
            if total > 0 else 0
        )
        
        return {
            **self.stats,
            "avg_inference_time_ms": avg_time * 1000,
            "cache_hit_rate": hit_rate,
            "model_loaded": self.model_loaded,
        }
    
    def clear_cache(self) -> None:
        """Limpia el caché"""
        with self.lock:
            self.cache.clear()
            logger.info("Caché limpiado")


def create_model_server(
    model_path: Path,
    device: str = "cuda",
    **kwargs,
) -> ModelServer:
    """
    Factory function para crear servidor de modelos.
    
    Args:
        model_path: Ruta al modelo
        device: Dispositivo a usar
        **kwargs: Argumentos adicionales
    
    Returns:
        ModelServer configurado
    """
    return ModelServer(
        model_path=model_path,
        device=device,
        **kwargs,
    )
'''
        
        (deploy_dir / "model_serving.py").write_text(serving_content, encoding="utf-8")
    
    def _generate_health_checks(
        self,
        deploy_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de health checks"""
        
        health_content = '''"""
Health Checks - Utilidades de health checks
===========================================

Sistema de health checks para modelos y servicios.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Verificador de salud para modelos y servicios.
    """
    
    def __init__(self, model: Optional[nn.Module] = None):
        """
        Inicializa el verificador.
        
        Args:
            model: Modelo a verificar (opcional)
        """
        self.model = model
    
    def check_model_health(
        self,
        model: Optional[nn.Module] = None,
        test_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Verifica salud del modelo.
        
        Args:
            model: Modelo a verificar (usa self.model si no se proporciona)
            test_input: Input de prueba (opcional)
        
        Returns:
            Diccionario con estado de salud
        """
        model = model or self.model
        if model is None:
            return {
                "healthy": False,
                "error": "No se proporcionó modelo",
            }
        
        health_status = {
            "healthy": True,
            "checks": {},
        }
        
        # Verificar parámetros
        try:
            has_nan = False
            has_inf = False
            
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    has_nan = True
                    logger.warning(f"NaN encontrado en {name}")
                if torch.isinf(param).any():
                    has_inf = True
                    logger.warning(f"Inf encontrado en {name}")
            
            health_status["checks"]["parameters"] = {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "valid": not (has_nan or has_inf),
            }
            
            if has_nan or has_inf:
                health_status["healthy"] = False
        
        except Exception as e:
            health_status["checks"]["parameters"] = {
                "valid": False,
                "error": str(e),
            }
            health_status["healthy"] = False
        
        # Test forward pass si se proporciona input
        if test_input is not None:
            try:
                model.eval()
                start_time = time.time()
                
                with torch.no_grad():
                    output = model(test_input)
                
                inference_time = time.time() - start_time
                
                # Verificar output
                output_valid = True
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any():
                        output_valid = False
                    if torch.isinf(output).any():
                        output_valid = False
                
                health_status["checks"]["forward_pass"] = {
                    "valid": output_valid,
                    "inference_time_ms": inference_time * 1000,
                }
                
                if not output_valid:
                    health_status["healthy"] = False
            
            except Exception as e:
                health_status["checks"]["forward_pass"] = {
                    "valid": False,
                    "error": str(e),
                }
                health_status["healthy"] = False
        
        # Verificar memoria si es CUDA
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                
                health_status["checks"]["memory"] = {
                    "allocated_mb": memory_allocated,
                    "reserved_mb": memory_reserved,
                    "valid": memory_allocated < 10000,  # Menos de 10GB
                }
            except Exception as e:
                health_status["checks"]["memory"] = {
                    "valid": False,
                    "error": str(e),
                }
        
        return health_status


def check_model_health(
    model: nn.Module,
    test_input: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Función helper para verificar salud del modelo.
    
    Args:
        model: Modelo a verificar
        test_input: Input de prueba (opcional)
    
    Returns:
        Diccionario con estado de salud
    """
    checker = HealthChecker(model)
    return checker.check_model_health(test_input=test_input)
'''
        
        (deploy_dir / "health_checks.py").write_text(health_content, encoding="utf-8")
    
    def _generate_model_versioning(
        self,
        deploy_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de versionado de modelos"""
        
        versioning_content = '''"""
Model Versioning - Utilidades de versionado de modelos
=======================================================

Sistema de versionado para gestión de modelos en producción.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """
    Gestor de versiones de modelos.
    
    Permite:
    - Guardar modelos con versiones
    - Cargar versiones específicas
    - Listar versiones disponibles
    - Metadata de versiones
    """
    
    def __init__(self, models_dir: Path):
        """
        Inicializa el gestor de versiones.
        
        Args:
            models_dir: Directorio donde guardar modelos
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.models_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Carga información de versiones"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando versiones: {e}")
                return {}
        return {}
    
    def _save_versions(self) -> None:
        """Guarda información de versiones"""
        try:
            with open(self.versions_file, "w") as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando versiones: {e}")
    
    def save_version(
        self,
        model: torch.nn.Module,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Guarda una versión del modelo.
        
        Args:
            model: Modelo a guardar
            version: Versión (ej: "1.0.0", "v2.1")
            metadata: Metadata adicional (opcional)
        
        Returns:
            Ruta donde se guardó el modelo
        """
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = version_dir / "model.pt"
        
        # Guardar modelo
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, model_path)
        
        # Guardar metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "version": version,
                "timestamp": checkpoint["timestamp"],
                "metadata": metadata or {},
            }, f, indent=2)
        
        # Actualizar registro de versiones
        self.versions[version] = {
            "path": str(model_path),
            "timestamp": checkpoint["timestamp"],
            "metadata": metadata or {},
        }
        self._save_versions()
        
        logger.info(f"Modelo versión {version} guardado en {model_path}")
        return model_path
    
    def list_versions(self) -> List[str]:
        """
        Lista versiones disponibles.
        
        Returns:
            Lista de versiones
        """
        return sorted(self.versions.keys(), reverse=True)
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de una versión.
        
        Args:
            version: Versión a consultar
        
        Returns:
            Información de la versión o None
        """
        return self.versions.get(version)
    
    def load_version(
        self,
        model_class: type,
        version: str,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """
        Carga una versión específica del modelo.
        
        Args:
            model_class: Clase del modelo
            version: Versión a cargar
            device: Dispositivo a usar
        
        Returns:
            Modelo cargado
        """
        if version not in self.versions:
            raise ValueError(f"Versión {version} no encontrada")
        
        model_path = Path(self.versions[version]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Inicializar modelo
        model = model_class()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        logger.info(f"Modelo versión {version} cargado desde {model_path}")
        return model


def load_model_version(
    model_class: type,
    version: str,
    models_dir: Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Función helper para cargar versión de modelo.
    
    Args:
        model_class: Clase del modelo
        version: Versión a cargar
        models_dir: Directorio de modelos
        device: Dispositivo a usar
    
    Returns:
        Modelo cargado
    """
    manager = ModelVersionManager(models_dir)
    return manager.load_version(model_class, version, device)
'''
        
        (deploy_dir / "model_versioning.py").write_text(versioning_content, encoding="utf-8")

