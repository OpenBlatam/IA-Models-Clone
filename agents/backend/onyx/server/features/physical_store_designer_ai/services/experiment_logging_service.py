"""
Experiment Logging Service - Integración con Tensorboard/WandB
"""

from typing import Dict, Any, Optional, List
from ..core.service_base import TimestampedService

# Placeholders para Tensorboard y WandB
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class LoggingBackend(str):
    """Backends de logging"""
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    BOTH = "both"


class ExperimentLoggingService(TimestampedService):
    """Servicio para logging de experimentos"""
    
    def __init__(self):
        super().__init__("ExperimentLoggingService")
        self.writers: Dict[str, Any] = {}
        self.wandb_runs: Dict[str, Any] = {}
        
        if not TENSORBOARD_AVAILABLE:
            self.log_warning("Tensorboard no disponible")
        if not WANDB_AVAILABLE:
            self.log_warning("WandB no disponible")
    
    def initialize_logging(
        self,
        experiment_id: str,
        backend: str = LoggingBackend.TENSORBOARD,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Inicializar logging"""
        
        init_info = {
            "experiment_id": experiment_id,
            "backend": backend,
            "initialized_at": datetime.now().isoformat()
        }
        
        if backend in [LoggingBackend.TENSORBOARD, LoggingBackend.BOTH] and TENSORBOARD_AVAILABLE:
            log_dir = f"runs/{experiment_id}"
            writer = SummaryWriter(log_dir=log_dir)
            self.writers[experiment_id] = writer
            init_info["tensorboard_log_dir"] = log_dir
            init_info["tensorboard_available"] = True
        else:
            init_info["tensorboard_available"] = False
        
        if backend in [LoggingBackend.WANDB, LoggingBackend.BOTH] and WANDB_AVAILABLE:
            try:
                run = wandb.init(
                    project=project_name or "store_designer",
                    name=run_name or experiment_id,
                    config=config or {}
                )
                self.wandb_runs[experiment_id] = run
                init_info["wandb_run_id"] = run.id
                init_info["wandb_available"] = True
            except Exception as e:
                self.log_error(f"Error inicializando WandB: {e}", exc_info=True)
                init_info["wandb_available"] = False
        else:
            init_info["wandb_available"] = False
        
        return init_info
    
    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar métrica"""
        
        log_info = {
            "experiment_id": experiment_id,
            "metric_name": metric_name,
            "value": value,
            "step": step,
            "logged_at": datetime.now().isoformat()
        }
        
        if backend is None:
            backend = LoggingBackend.BOTH
        
        if backend in [LoggingBackend.TENSORBOARD, LoggingBackend.BOTH]:
            writer = self.writers.get(experiment_id)
            if writer:
                writer.add_scalar(metric_name, value, step or 0)
                log_info["tensorboard_logged"] = True
        
        if backend in [LoggingBackend.WANDB, LoggingBackend.BOTH]:
            run = self.wandb_runs.get(experiment_id)
            if run:
                run.log({metric_name: value}, step=step)
                log_info["wandb_logged"] = True
        
        return log_info
    
    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Registrar múltiples métricas"""
        
        log_info = {
            "experiment_id": experiment_id,
            "metrics": metrics,
            "step": step,
            "logged_at": datetime.now().isoformat()
        }
        
        writer = self.writers.get(experiment_id)
        if writer:
            for name, value in metrics.items():
                writer.add_scalar(name, value, step or 0)
            log_info["tensorboard_logged"] = True
        
        run = self.wandb_runs.get(experiment_id)
        if run:
            run.log(metrics, step=step)
            log_info["wandb_logged"] = True
        
        return log_info
    
    def log_image(
        self,
        experiment_id: str,
        tag: str,
        image_tensor: Any,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Registrar imagen"""
        
        log_info = {
            "experiment_id": experiment_id,
            "tag": tag,
            "step": step,
            "logged_at": datetime.now().isoformat()
        }
        
        writer = self.writers.get(experiment_id)
        if writer:
            writer.add_image(tag, image_tensor, step or 0)
            log_info["tensorboard_logged"] = True
        
        return log_info
    
    def log_histogram(
        self,
        experiment_id: str,
        tag: str,
        values: Any,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Registrar histograma"""
        
        log_info = {
            "experiment_id": experiment_id,
            "tag": tag,
            "step": step,
            "logged_at": datetime.now().isoformat()
        }
        
        writer = self.writers.get(experiment_id)
        if writer:
            writer.add_histogram(tag, values, step or 0)
            log_info["tensorboard_logged"] = True
        
        return log_info
    
    def finish_logging(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """Finalizar logging"""
        
        finish_info = {
            "experiment_id": experiment_id,
            "finished_at": datetime.now().isoformat()
        }
        
        writer = self.writers.get(experiment_id)
        if writer:
            writer.close()
            finish_info["tensorboard_closed"] = True
        
        run = self.wandb_runs.get(experiment_id)
        if run:
            run.finish()
            finish_info["wandb_finished"] = True
        
        return finish_info




