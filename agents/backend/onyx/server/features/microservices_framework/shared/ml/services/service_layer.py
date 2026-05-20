"""
Service Layer
Service layer pattern for business logic separation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Base service class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
    
    def initialize(self):
        """Initialize the service."""
        if not self._initialized:
            self._initialize()
            self._initialized = True
    
    @abstractmethod
    def _initialize(self):
        """Internal initialization."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the service."""
        pass


class ModelService(BaseService):
    """Service for model operations."""
    
    def __init__(self, model_manager, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_manager = model_manager
        self._model = None
    
    def _initialize(self):
        """Initialize model service."""
        model_name = self.config.get("model_name", "gpt2")
        self._model = self.model_manager.get_model(model_name)
        logger.info(f"Model service initialized with {model_name}")
    
    def execute(self, operation: str, **kwargs) -> Any:
        """Execute model operation."""
        if not self._initialized:
            self.initialize()
        
        if operation == "load":
            return self._load_model(**kwargs)
        elif operation == "unload":
            return self._unload_model(**kwargs)
        elif operation == "info":
            return self._get_model_info(**kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _load_model(self, model_name: str, **kwargs):
        """Load a model."""
        self._model = self.model_manager.get_model(model_name, **kwargs)
        return {"status": "loaded", "model_name": model_name}
    
    def _unload_model(self, model_name: str):
        """Unload a model."""
        self.model_manager.unload(model_name)
        return {"status": "unloaded", "model_name": model_name}
    
    def _get_model_info(self, model_name: Optional[str] = None):
        """Get model information."""
        model = self._model if model_name is None else self.model_manager.get_model(model_name)
        from ..model_utils import get_model_summary
        return get_model_summary(model)


class InferenceService(BaseService):
    """Service for inference operations."""
    
    def __init__(self, inference_engine, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.inference_engine = inference_engine
    
    def _initialize(self):
        """Initialize inference service."""
        logger.info("Inference service initialized")
    
    def execute(self, operation: str, **kwargs) -> Any:
        """Execute inference operation."""
        if not self._initialized:
            self.initialize()
        
        if operation == "generate":
            return self._generate(**kwargs)
        elif operation == "embeddings":
            return self._get_embeddings(**kwargs)
        elif operation == "batch_generate":
            return self._batch_generate(**kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _generate(self, prompt: str, **kwargs):
        """Generate text."""
        return self.inference_engine.generate(prompt, **kwargs)
    
    def _get_embeddings(self, texts: List[str], **kwargs):
        """Get embeddings."""
        return self.inference_engine.get_embeddings(texts, **kwargs)
    
    def _batch_generate(self, prompts: List[str], **kwargs):
        """Batch generate."""
        return self.inference_engine.batch_generate(prompts, **kwargs)


class TrainingService(BaseService):
    """Service for training operations."""
    
    def __init__(self, trainer, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.trainer = trainer
    
    def _initialize(self):
        """Initialize training service."""
        logger.info("Training service initialized")
    
    def execute(self, operation: str, **kwargs) -> Any:
        """Execute training operation."""
        if not self._initialized:
            self.initialize()
        
        if operation == "train":
            return self._train(**kwargs)
        elif operation == "validate":
            return self._validate(**kwargs)
        elif operation == "resume":
            return self._resume(**kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _train(self, num_epochs: int, **kwargs):
        """Train the model."""
        self.trainer.train(num_epochs=num_epochs, **kwargs)
        return {"status": "completed", "epochs": num_epochs}
    
    def _validate(self, **kwargs):
        """Validate the model."""
        metrics = self.trainer.validate()
        return metrics
    
    def _resume(self, checkpoint_path: str, **kwargs):
        """Resume training from checkpoint."""
        self.trainer.load_checkpoint(checkpoint_path)
        return {"status": "resumed", "checkpoint": checkpoint_path}


class ServiceRegistry:
    """Registry for services."""
    
    def __init__(self):
        self._services: Dict[str, BaseService] = {}
    
    def register(self, name: str, service: BaseService):
        """Register a service."""
        self._services[name] = service
        logger.info(f"Service '{name}' registered")
    
    def get(self, name: str) -> Optional[BaseService]:
        """Get a service by name."""
        return self._services.get(name)
    
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self._services.keys())
    
    def execute(self, service_name: str, operation: str, **kwargs) -> Any:
        """Execute an operation on a service."""
        service = self.get(service_name)
        if service is None:
            raise ValueError(f"Service '{service_name}' not found")
        return service.execute(operation, **kwargs)



