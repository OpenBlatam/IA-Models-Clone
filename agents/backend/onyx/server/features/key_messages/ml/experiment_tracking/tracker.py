from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import time
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import structlog
from pathlib import Path
import json
from datetime import datetime
            from torch.utils.tensorboard import SummaryWriter
                import torch
            import wandb
            import wandb
            import mlflow
            import mlflow
            import mlflow
            import mlflow
                import mlflow
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Experiment Tracking Classes
Provides unified interfaces for TensorBoard, Weights & Biases, and MLflow
"""


logger = structlog.get_logger(__name__)

class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""
    
    def __init__(self) -> Any:
        self.experiment_name = None
        self.run_id = None
        self.config = None
        self.is_initialized = False
        
    @abstractmethod
    def init_experiment(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Initialize experiment and return run ID."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to the tracker."""
        pass
    
    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters."""
        pass
    
    @abstractmethod
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifacts."""
        pass
    
    @abstractmethod
    def finalize_experiment(self) -> Any:
        """Finalize the experiment."""
        pass
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a single scalar value."""
        self.log_metrics({name: value}, step=step)
    
    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values."""
        self.log_metrics(scalars, step=step)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None):
        """Log text data."""
        self.log_metrics({f"{name}_text": text}, step=step)
    
    def log_histogram(self, name: str, values: List[float], step: Optional[int] = None):
        """Log histogram data."""
        self.log_metrics({f"{name}_histogram": values}, step=step)
    
    def log_image(self, name: str, image_data: Any, step: Optional[int] = None):
        """Log image data."""
        self.log_metrics({f"{name}_image": image_data}, step=step)

class NoOpTracker(ExperimentTracker):
    """No-operation tracker for when no tracking is enabled."""
    
    def init_experiment(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Initialize experiment (no-op)."""
        self.experiment_name = experiment_name
        self.config = config or {}
        self.run_id = f"noop_{int(time.time())}"
        self.is_initialized = True
        logger.info("NoOpTracker initialized", experiment_name=experiment_name)
        return self.run_id
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics (no-op)."""
        if self.is_initialized:
            logger.debug("NoOpTracker logging metrics", metrics=metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration (no-op)."""
        if self.is_initialized:
            logger.debug("NoOpTracker logging config", config=config)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model (no-op)."""
        if self.is_initialized:
            logger.debug("NoOpTracker logging model", model_path=model_path, model_name=model_name)
    
    def finalize_experiment(self) -> Any:
        """Finalize experiment (no-op)."""
        if self.is_initialized:
            logger.info("NoOpTracker experiment finalized", experiment_name=self.experiment_name)
            self.is_initialized = False

class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker."""
    
    def __init__(self, log_dir: str = "./logs", update_freq: int = 100, flush_secs: int = 120):
        
    """__init__ function."""
super().__init__()
        self.log_dir = Path(log_dir)
        self.update_freq = update_freq
        self.flush_secs = flush_secs
        self.writer = None
        self.last_flush = time.time()
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("TensorBoardTracker initialized", 
                   log_dir=str(self.log_dir),
                   update_freq=update_freq,
                   flush_secs=flush_secs)
    
    def init_experiment(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Initialize TensorBoard experiment."""
        try:
            
            self.experiment_name = experiment_name
            self.config = config or {}
            self.run_id = f"tb_{int(time.time())}"
            
            # Create experiment directory
            experiment_dir = self.log_dir / experiment_name / self.run_id
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize TensorBoard writer
            self.writer = SummaryWriter(
                log_dir=str(experiment_dir),
                flush_secs=self.flush_secs
            )
            
            # Log configuration
            if config:
                self.log_config(config)
            
            self.is_initialized = True
            
            logger.info("TensorBoard experiment initialized", 
                       experiment_name=experiment_name,
                       run_id=self.run_id,
                       experiment_dir=str(experiment_dir))
            
            return self.run_id
            
        except ImportError:
            logger.error("TensorBoard not available. Install with: pip install tensorboard")
            return self._fallback_init(experiment_name, config)
        except Exception as e:
            logger.error("Failed to initialize TensorBoard", error=str(e))
            return self._fallback_init(experiment_name, config)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to TensorBoard."""
        if not self.is_initialized or not self.writer:
            return
        
        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, step or 0)
                elif isinstance(value, str):
                    self.writer.add_text(name, value, step or 0)
                elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                    self.writer.add_histogram(name, value, step or 0)
            
            # Flush periodically
            current_time = time.time()
            if current_time - self.last_flush > self.flush_secs:
                self.writer.flush()
                self.last_flush = current_time
                
        except Exception as e:
            logger.error("Failed to log metrics to TensorBoard", error=str(e))
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to TensorBoard."""
        if not self.is_initialized or not self.writer:
            return
        
        try:
            # Log as text
            config_text = json.dumps(config, indent=2)
            self.writer.add_text("config", config_text, 0)
            
            # Log as scalar parameters
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"config/{key}", value, 0)
                    
        except Exception as e:
            logger.error("Failed to log config to TensorBoard", error=str(e))
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to TensorBoard."""
        if not self.is_initialized or not self.writer:
            return
        
        try:
            # Log model graph (if it's a PyTorch model)
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                model = torch.load(model_path, map_location='cpu')
                if hasattr(model, 'forward'):
                    dummy_input = torch.randn(1, 512)  # Adjust based on your model
                    self.writer.add_graph(model, dummy_input)
                    
        except Exception as e:
            logger.error("Failed to log model to TensorBoard", error=str(e))
    
    def finalize_experiment(self) -> Any:
        """Finalize TensorBoard experiment."""
        if self.writer:
            try:
                self.writer.flush()
                self.writer.close()
                logger.info("TensorBoard experiment finalized", 
                           experiment_name=self.experiment_name,
                           run_id=self.run_id)
            except Exception as e:
                logger.error("Failed to finalize TensorBoard experiment", error=str(e))
        
        self.is_initialized = False
    
    def _fallback_init(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Fallback initialization when TensorBoard is not available."""
        logger.warning("Falling back to NoOpTracker for TensorBoard")
        noop_tracker = NoOpTracker()
        return noop_tracker.init_experiment(experiment_name, config)

class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, project: str = "key_messages", entity: Optional[str] = None,
                 tags: Optional[List[str]] = None, notes: str = "",
                 config_exclude_keys: Optional[List[str]] = None):
        
    """__init__ function."""
super().__init__()
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.notes = notes
        self.config_exclude_keys = config_exclude_keys or []
        self.run = None
        
        logger.info("WandbTracker initialized", 
                   project=project,
                   entity=entity,
                   tags=tags)
    
    def init_experiment(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Initialize Weights & Biases experiment."""
        try:
            
            self.experiment_name = experiment_name
            self.config = config or {}
            
            # Filter config to exclude sensitive keys
            filtered_config = self._filter_config(self.config)
            
            # Initialize W&B run
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=experiment_name,
                tags=self.tags,
                notes=self.notes,
                config=filtered_config,
                reinit=True
            )
            
            self.run_id = self.run.id
            self.is_initialized = True
            
            logger.info("Weights & Biases experiment initialized", 
                       experiment_name=experiment_name,
                       run_id=self.run_id,
                       project=self.project)
            
            return self.run_id
            
        except ImportError:
            logger.error("Weights & Biases not available. Install with: pip install wandb")
            return self._fallback_init(experiment_name, config)
        except Exception as e:
            logger.error("Failed to initialize Weights & Biases", error=str(e))
            return self._fallback_init(experiment_name, config)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to Weights & Biases."""
        if not self.is_initialized or not self.run:
            return
        
        try:
            # Add step to metrics if provided
            if step is not None:
                metrics["step"] = step
            
            self.run.log(metrics)
            
        except Exception as e:
            logger.error("Failed to log metrics to Weights & Biases", error=str(e))
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to Weights & Biases."""
        if not self.is_initialized or not self.run:
            return
        
        try:
            filtered_config = self._filter_config(config)
            self.run.config.update(filtered_config)
            
        except Exception as e:
            logger.error("Failed to log config to Weights & Biases", error=str(e))
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to Weights & Biases."""
        if not self.is_initialized or not self.run:
            return
        
        try:
            
            # Log model artifact
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=f"Model checkpoint for {self.experiment_name}"
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            
        except Exception as e:
            logger.error("Failed to log model to Weights & Biases", error=str(e))
    
    def finalize_experiment(self) -> Any:
        """Finalize Weights & Biases experiment."""
        if self.run:
            try:
                self.run.finish()
                logger.info("Weights & Biases experiment finalized", 
                           experiment_name=self.experiment_name,
                           run_id=self.run_id)
            except Exception as e:
                logger.error("Failed to finalize Weights & Biases experiment", error=str(e))
        
        self.is_initialized = False
    
    def _filter_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter configuration to exclude sensitive keys."""
        filtered = {}
        for key, value in config.items():
            if key not in self.config_exclude_keys:
                filtered[key] = value
        return filtered
    
    def _fallback_init(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Fallback initialization when W&B is not available."""
        logger.warning("Falling back to NoOpTracker for Weights & Biases")
        noop_tracker = NoOpTracker()
        return noop_tracker.init_experiment(experiment_name, config)

class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", 
                 experiment_name: str = "key_messages", log_models: bool = True):
        
    """__init__ function."""
super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_models = log_models
        self.run = None
        
        logger.info("MLflowTracker initialized", 
                   tracking_uri=tracking_uri,
                   experiment_name=experiment_name,
                   log_models=log_models)
    
    def init_experiment(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Initialize MLflow experiment."""
        try:
            
            self.experiment_name = experiment_name
            self.config = config or {}
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Start run
            self.run = mlflow.start_run()
            self.run_id = self.run.info.run_id
            self.is_initialized = True
            
            # Log configuration
            if config:
                self.log_config(config)
            
            logger.info("MLflow experiment initialized", 
                       experiment_name=experiment_name,
                       run_id=self.run_id,
                       tracking_uri=self.tracking_uri)
            
            return self.run_id
            
        except ImportError:
            logger.error("MLflow not available. Install with: pip install mlflow")
            return self._fallback_init(experiment_name, config)
        except Exception as e:
            logger.error("Failed to initialize MLflow", error=str(e))
            return self._fallback_init(experiment_name, config)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not self.is_initialized or not self.run:
            return
        
        try:
            
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(name, value, step=step or 0)
                elif isinstance(value, str):
                    mlflow.log_text(value, f"{name}.txt")
                    
        except Exception as e:
            logger.error("Failed to log metrics to MLflow", error=str(e))
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to MLflow."""
        if not self.is_initialized or not self.run:
            return
        
        try:
            
            # Log parameters
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
            
            # Log full config as artifact
            config_path = "config.json"
            with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(config, f, indent=2)
            mlflow.log_artifact(config_path)
            os.remove(config_path)
            
        except Exception as e:
            logger.error("Failed to log config to MLflow", error=str(e))
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to MLflow."""
        if not self.is_initialized or not self.run or not self.log_models:
            return
        
        try:
            
            # Log model artifact
            mlflow.log_artifact(model_path, model_name)
            
        except Exception as e:
            logger.error("Failed to log model to MLflow", error=str(e))
    
    def finalize_experiment(self) -> Any:
        """Finalize MLflow experiment."""
        if self.run:
            try:
                mlflow.end_run()
                logger.info("MLflow experiment finalized", 
                           experiment_name=self.experiment_name,
                           run_id=self.run_id)
            except Exception as e:
                logger.error("Failed to finalize MLflow experiment", error=str(e))
        
        self.is_initialized = False
    
    def _fallback_init(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Fallback initialization when MLflow is not available."""
        logger.warning("Falling back to NoOpTracker for MLflow")
        noop_tracker = NoOpTracker()
        return noop_tracker.init_experiment(experiment_name, config)

class CompositeTracker(ExperimentTracker):
    """Composite tracker that combines multiple trackers."""
    
    def __init__(self, trackers: List[ExperimentTracker]):
        
    """__init__ function."""
super().__init__()
        self.trackers = trackers
        
        logger.info("CompositeTracker initialized", num_trackers=len(trackers))
    
    def init_experiment(self, experiment_name: str, config: Dict[str, Any] = None) -> str:
        """Initialize all trackers."""
        self.experiment_name = experiment_name
        self.config = config or {}
        
        run_ids = []
        for tracker in self.trackers:
            try:
                run_id = tracker.init_experiment(experiment_name, config)
                run_ids.append(run_id)
            except Exception as e:
                logger.error(f"Failed to initialize tracker {type(tracker).__name__}", error=str(e))
        
        self.run_id = run_ids[0] if run_ids else f"composite_{int(time.time())}"
        self.is_initialized = True
        
        logger.info("CompositeTracker experiment initialized", 
                   experiment_name=experiment_name,
                   run_ids=run_ids)
        
        return self.run_id
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all trackers."""
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Failed to log metrics to {type(tracker).__name__}", error=str(e))
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to all trackers."""
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            try:
                tracker.log_config(config)
            except Exception as e:
                logger.error(f"Failed to log config to {type(tracker).__name__}", error=str(e))
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to all trackers."""
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            try:
                tracker.log_model(model_path, model_name)
            except Exception as e:
                logger.error(f"Failed to log model to {type(tracker).__name__}", error=str(e))
    
    def finalize_experiment(self) -> Any:
        """Finalize all trackers."""
        if not self.is_initialized:
            return
        
        for tracker in self.trackers:
            try:
                tracker.finalize_experiment()
            except Exception as e:
                logger.error(f"Failed to finalize {type(tracker).__name__}", error=str(e))
        
        self.is_initialized = False 