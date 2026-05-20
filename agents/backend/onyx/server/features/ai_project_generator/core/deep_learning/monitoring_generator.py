"""
Monitoring Generator
===================

Generador de sistemas de monitoreo y logging para modelos.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo."""
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_wandb: bool = False
    enable_tensorboard: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"


class MonitoringGenerator:
    """
    Generador de sistemas de monitoreo.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_prometheus_config(
        self,
        project_dir: Path,
        config: Optional[MonitoringConfig] = None
    ) -> str:
        """
        Generar configuración de Prometheus.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido de prometheus.yml
        """
        if config is None:
            config = MonitoringConfig()
        
        prometheus_content = f"""global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'model-api'
    static_configs:
      - targets: ['localhost:{config.metrics_port}']
        labels:
          service: 'model-api'
          environment: 'production'
"""
        
        return prometheus_content
    
    def generate_metrics_exporter(
        self,
        project_dir: Path,
        config: Optional[MonitoringConfig] = None
    ) -> str:
        """
        Generar exporter de métricas.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del exporter
        """
        if config is None:
            config = MonitoringConfig()
        
        exporter_content = f""""""
Metrics Exporter para modelo de Deep Learning
==============================================

Generado automáticamente por DeepLearningGenerator
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Métricas
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_name', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)

model_loaded = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)',
    ['model_name']
)

active_requests = Gauge(
    'model_active_requests',
    'Number of active prediction requests',
    ['model_name']
)

def track_metrics(model_name: str = "default"):
    \"\"\"Decorador para trackear métricas.\"\"\"
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active_requests.labels(model_name=model_name).inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                prediction_counter.labels(
                    model_name=model_name,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                prediction_counter.labels(
                    model_name=model_name,
                    status="error"
                ).inc()
                raise
            finally:
                latency = time.time() - start_time
                prediction_latency.labels(model_name=model_name).observe(latency)
                active_requests.labels(model_name=model_name).dec()
        
        return wrapper
    return decorator

def start_metrics_server(port: int = {config.metrics_port}):
    \"\"\"Iniciar servidor de métricas.\"\"\"
    start_http_server(port)
    print(f"Metrics server started on port {{port}}")

if __name__ == "__main__":
    start_metrics_server()
"""
        
        return exporter_content
    
    def generate_logging_config(
        self,
        project_dir: Path,
        config: Optional[MonitoringConfig] = None
    ) -> str:
        """
        Generar configuración de logging.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido de logging config
        """
        if config is None:
            config = MonitoringConfig()
        
        logging_config = f""""""
Logging Configuration
======================

Generado automáticamente por DeepLearningGenerator
"""

import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime

def setup_logging(
    log_dir: Path = Path("logs"),
    log_level: str = "{config.log_level}",
    enable_file_logging: bool = True,
    enable_json_logging: bool = True
):
    \"\"\"Configurar logging.\"\"\"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level))
    
    # Handler de archivo
    if enable_file_logging:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level))
    
    # Handler JSON
    if enable_json_logging:
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.json.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(getattr(logging, log_level))
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        root_logger.addHandler(file_handler)
    
    if enable_json_logging:
        root_logger.addHandler(json_handler)
    
    return root_logger

class JSONFormatter(logging.Formatter):
    \"\"\"Formatter JSON para logs.\"\"\"
    def format(self, record):
        log_data = {{
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }}
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging configured successfully")
"""
        
        return logging_config
    
    def generate_all(
        self,
        project_dir: Path,
        config: Optional[MonitoringConfig] = None
    ) -> Dict[str, str]:
        """
        Generar todos los archivos de monitoreo.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if config is None:
            config = MonitoringConfig()
        
        files = {}
        monitoring_dir = project_dir / "monitoring"
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Prometheus config
        if config.enable_prometheus:
            prometheus_content = self.generate_prometheus_config(project_dir, config)
            prometheus_path = monitoring_dir / "prometheus.yml"
            prometheus_path.write_text(prometheus_content, encoding='utf-8')
            files['monitoring/prometheus.yml'] = prometheus_content
        
        # Metrics exporter
        if config.enable_prometheus:
            exporter_content = self.generate_metrics_exporter(project_dir, config)
            exporter_path = project_dir / "app" / "monitoring" / "metrics.py"
            exporter_path.parent.mkdir(parents=True, exist_ok=True)
            exporter_path.write_text(exporter_content, encoding='utf-8')
            files['app/monitoring/metrics.py'] = exporter_content
        
        # Logging config
        logging_content = self.generate_logging_config(project_dir, config)
        logging_path = project_dir / "app" / "monitoring" / "logging_config.py"
        logging_path.parent.mkdir(parents=True, exist_ok=True)
        logging_path.write_text(logging_content, encoding='utf-8')
        files['app/monitoring/logging_config.py'] = logging_content
        
        logger.info(f"Archivos de monitoreo generados")
        
        return files


# Instancia global
_global_monitoring_generator: Optional[MonitoringGenerator] = None


def get_monitoring_generator() -> MonitoringGenerator:
    """
    Obtener instancia global del generador de monitoreo.
    
    Returns:
        Instancia del generador
    """
    global _global_monitoring_generator
    
    if _global_monitoring_generator is None:
        _global_monitoring_generator = MonitoringGenerator()
    
    return _global_monitoring_generator
