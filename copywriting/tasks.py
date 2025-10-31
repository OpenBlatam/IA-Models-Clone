from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
import os
from celery import Celery, states
from celery.utils.log import get_task_logger
from .service import CopywritingService
from .models import CopywritingInput, get_settings
from typing import List, Dict, Any

    from dask.distributed import Client
        import asyncio
        import asyncio
        from celery import group
from typing import Any, List, Dict, Optional
celery_app = Celery(
    "copywriting",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)
logger = get_task_logger(__name__)

# --- Dask Integration for Distributed Batch Processing ---
# pip install dask distributed
try:
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Use settings.dask_scheduler_address for Dask
settings = get_settings()
DASK_SCHEDULER_ADDRESS = settings.dask_scheduler_address

@celery_app.task(bind=True)
def generate_copywriting_task(self, input_data_dict: Dict[str, Any], model_name: str = "gpt2"):
    """
    Tarea Celery para generar copywriting usando el servicio LLM. Maneja errores y reporta progreso.
    """
    logger.info(f"[Celery] Generando copywriting con modelo={model_name}")
    try:
        service = CopywritingService(model_name=model_name)
        input_data = CopywritingInput(**input_data_dict)
        result = asyncio.run(service.generate(input_data))
        logger.info(f"[Celery] Generación exitosa para tracking_id={getattr(input_data, 'tracking_id', None)}")
        return result.dict()
    except Exception as e:
        logger.error(f"[Celery] Error en generación: {e}")
        self.update_state(state=states.FAILURE, meta={"exc": str(e)})
        raise

# --- Dask-enabled batch processing ---
def batch_generate_copywriting(inputs: List[Dict[str, Any]], model_name: str = "gpt2"):
    """
    Batch generate copywriting using Dask if available, else fallback to Celery group.
    """
    if DASK_AVAILABLE:
        logger.info("Using Dask for distributed batch copywriting generation.")
        client = Client(DASK_SCHEDULER_ADDRESS)
        service = CopywritingService(model_name=model_name)
        def run_one(input_data_dict) -> Any:
            input_data = CopywritingInput(**input_data_dict)
            return asyncio.run(service.generate(input_data)).dict()
        futures = client.map(run_one, inputs)
        results = client.gather(futures)
        logger.info(f"[Dask] Batch procesado con {len(results)} resultados")
        return results
    else:
        logger.info("Dask not available, using Celery group for batch generation.")
        jobs = [generate_copywriting_task.s(input_data, model_name) for input_data in inputs]
        result = group(jobs).apply_async()
        return result.get() 