from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, Tuple
import os
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .logging_utils import logger
    from optimum.bettertransformer import BetterTransformer
    from accelerate import infer_auto_device_map, dispatch_model
import orjson
from typing import Any, List, Dict, Optional
import logging
import asyncio
try:
    ACCEL_AVAILABLE = True
except ImportError:
    ACCEL_AVAILABLE = False

MODEL_PATH: str = './fine_tuned_model'
CHECK_INTERVAL: int = 10  # segundos

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
last_loaded: float = 0

def _get_model_mtime() -> float:
    """Devuelve el mtime más reciente de los archivos del modelo."""
    try:
        return max(
            os.path.getmtime(os.path.join(MODEL_PATH, f))
            for f in os.listdir(MODEL_PATH)
            if f.endswith('.bin') or f.endswith('.json') or f.endswith('.txt')
        )
    except Exception as e:
        logger.warning({"event": "model_mtime_failed", "error": str(e)})
        return 0

def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Carga el modelo y tokenizer desde disco, soporta optimum/accelerate y los pone en device."""
    global model, tokenizer, last_loaded
    logger.info({"event": "loading_model"})
    try:
        model_ = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer_ = AutoTokenizer.from_pretrained(MODEL_PATH)
        if ACCEL_AVAILABLE:
            try:
                device_map = infer_auto_device_map(model_, max_memory={0: "13GiB", 1: "13GiB"})
                model_ = dispatch_model(model_, device_map=device_map)
                model_ = BetterTransformer.transform(model_)
                logger.info({"event": "accelerate_enabled"})
            except Exception as e:
                logger.warning({"event": "accelerate_failed", "error": str(e)})
        model_.to(device)
        model_.eval()
        last_loaded = _get_model_mtime()
        logger.info({"event": "model_loaded"})
        return model_, tokenizer_
    except Exception as e:
        logger.error({"event": "model_load_failed", "error": str(e)})
        raise

def maybe_reload_model() -> None:
    """Recarga el modelo si hay cambios en disco o si no está cargado."""
    global model, tokenizer, last_loaded
    try:
        mtime = _get_model_mtime()
        if mtime > last_loaded or model is None or tokenizer is None:
            m, t = load_model()
            model, tokenizer = m, t
            last_loaded = mtime
            logger.info({"event": "model_reloaded"})
    except Exception as e:
        logger.error({"event": "reload_error", "error": str(e)})

def background_reloader() -> None:
    """Hilo en background que recarga el modelo periódicamente."""
    while True:
        maybe_reload_model()
        time.sleep(CHECK_INTERVAL)

def startup_event() -> None:
    """Evento FastAPI para cargar el modelo y lanzar el reloader."""
    global model, tokenizer
    model, tokenizer = load_model()
    threading.Thread(target=background_reloader, daemon=True).start() 
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")