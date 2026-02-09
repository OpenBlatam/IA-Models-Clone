"""
Model Serving System - Sistema de serving de modelos
====================================================
"""

import logging
import torch
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
import threading

from .base_classes import BaseManager, BaseConfig
from .common_utils import get_device, move_to_device, get_model_output
from .constants import DEFAULT_DEVICE, DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


@dataclass
class ModelServerConfig(BaseConfig):
    """Configuración del servidor de modelos"""
    model_path: str
    device: Optional[str] = None  # Se resuelve con get_device()
    max_batch_size: int = DEFAULT_BATCH_SIZE
    max_queue_size: int = 100
    num_workers: int = 1
    timeout: float = 30.0


class ModelServer(BaseManager):
    """Servidor de modelos para inferencia"""
    
    def __init__(self, config: ModelServerConfig):
        super().__init__(config)
        self.config = config
        self.model = None
        self.device = get_device(config.device)
        self.request_queue = Queue(maxsize=config.max_queue_size)
        self.workers = []
        self.running = False
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo"""
        try:
            self.model = torch.load(self.config.model_path, map_location=self.device)
            if isinstance(self.model, dict):
                # Si es un checkpoint, extraer el modelo
                if "model_state_dict" in self.model:
                    # Necesitaríamos la arquitectura del modelo
                    logger.warning("Checkpoint detectado, se requiere arquitectura del modelo")
                else:
                    self.model = self.model.get("model", None)
            
            if self.model:
                self.model.eval()
                self.model.to(self.device)
                self.log_event("model_loaded", {"model_path": self.config.model_path})
                logger.info(f"Modelo cargado desde {self.config.model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.log_event("model_load_error", {"error": str(e)})
            raise
    
    def predict(self, inputs: Any) -> Any:
        """Predicción síncrona usando utilidades compartidas"""
        if self.model is None:
            raise ValueError("Modelo no cargado")
        
        # Usar utilidades compartidas para obtener output
        return get_model_output(self.model, inputs, str(self.device))
    
    async def predict_async(self, inputs: Any) -> Any:
        """Predicción asíncrona"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, inputs)
    
    def start_workers(self):
        """Inicia workers para procesamiento batch"""
        self.running = True
        for i in range(self.config.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"{self.config.num_workers} workers iniciados")
    
    def _worker_loop(self):
        """Loop del worker"""
        while self.running:
            try:
                request = self.request_queue.get(timeout=1.0)
                # Procesar request
                # Implementación específica según necesidad
                self.request_queue.task_done()
            except:
                continue
    
    def stop(self):
        """Detiene el servidor"""
        self.running = False
        for worker in self.workers:
            worker.join()
        logger.info("Servidor de modelos detenido")

