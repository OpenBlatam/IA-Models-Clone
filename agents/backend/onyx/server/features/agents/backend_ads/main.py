# app/main.py
import os
import httpx
import logging
from contextlib import asynccontextmanager
import datetime
import traceback
import numpy as np
import cv2
import pyvips
import psutil
from loguru import logger

from fastapi import FastAPI, HTTPException, Body, Request as FastAPIRequest, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List, Optional

from models import AdsIaRequest, AdsResponse, BrandKitResponse, ErrorResponse, RemoveBackgroundRequest
from scraper import get_website_text
from llm_interface import (
    generate_ads_lcel,
    generate_brand_kit_lcel,
    generate_custom_content_lcel,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL_NAME,
    generate_ads_lcel_streaming,
    generate_ads_lcel_streaming_parallel
)
from config import settings
import base64
import io
from PIL import Image
try:
    from rembg import remove, new_session
except ImportError:
    remove = None
from remove_bg_api import router as remove_bg_router
from ads_api import router as ads_router
from key_messages.api import router as key_messages_router
from key_messages.llm_service import LLMKeyMessageService
from key_messages.models import MessageType, MessageTone

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a global rembg session for reuse (u2netp is fast and good for photos)
rembg_session = new_session("u2netp")
rembg_session_fast = new_session("u2netp")
rembg_session_std = new_session("u2net")

ERROR_LOG_PATH = "remove_bg_errors.log"

def log_error(input_info, exc):
    with open(ERROR_LOG_PATH, "a") as f:
        f.write(f"[{datetime.datetime.now()}] INPUT: {input_info}\nEXCEPTION: {exc}\nTRACE:\n{traceback.format_exc()}\n{'-'*60}\n")

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Iniciando aplicación y cliente HTTPX...")
    app_instance.state.httpx_client = httpx.AsyncClient(timeout=120.0)
    
    if not DEEPSEEK_API_KEY:
        logger.critical("DEEPSEEK_API_KEY no está configurado. El LLM no funcionará.")
    else:
        logger.info(f"Configurado para usar DeepSeek con modelo: {DEEPSEEK_MODEL_NAME}")
        try:
            # Verificar la conexión con DeepSeek
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            response = await app_instance.state.httpx_client.get(
                f"{settings.DEEPSEEK_API_URL}/models",
                headers=headers,
                timeout=5.0
            )
            if response.status_code == 200:
                logger.info("Conexión exitosa con DeepSeek API.")
            else:
                logger.warning(f"No se pudo confirmar la conexión con DeepSeek (status: {response.status_code}).")
        except Exception as e:
            logger.error(f"CRÍTICO: No se pudo conectar a DeepSeek API. Error: {e}")
    
    yield

    logger.info("Cerrando cliente HTTPX y finalizando aplicación...")
    await app_instance.state.httpx_client.aclose()

app = FastAPI(title="Ads IA API - DeepSeek Integration", version="1.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(GZipMiddleware, minimum_size=100)

app.include_router(remove_bg_router)
app.include_router(ads_router)
app.include_router(key_messages_router)

@app.post("/api/remove-background")
async def remove_background_endpoint(
    payload: RemoveBackgroundRequest = Body(None),
    file: UploadFile = File(None)
):
    if remove is None:
        return {"error": "rembg library not installed. Please install with 'pip install rembg pillow pillow-simd pyvips opencv-python-headless'"}
    image_bytes = None
    # 1. File upload (multipart)
    if file is not None:
        image_bytes = await file.read()
    # 2. Image URL
    elif payload and payload.image_url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(payload.image_url)
            if resp.status_code != 200:
                logger.error(f"Failed to download image: {resp.status_code}")
                return {"error": f"Failed to download image: {resp.status_code}"}
            image_bytes = resp.content
    # 3. Base64
    elif payload and payload.image_base64:
        try:
            image_bytes = base64.b64decode(payload.image_base64)
        except Exception as e:
            logger.error(f"Invalid base64: {e}")
            return {"error": f"Invalid base64: {e}"}
    else:
        logger.error("No image provided (file, image_url, or image_base64)")
        return {"error": "No image provided (file, image_url, or image_base64)"}
    try:
        # Log system usage before processing
        logger.info(f"RAM: {psutil.virtual_memory().percent}%, CPU: {psutil.cpu_percent()}% before processing")
        # Try pyvips for fast resize and conversion
        try:
            img_vips = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")
            max_size = 256
            scale = min(max_size / img_vips.width, max_size / img_vips.height, 1.0)
            if scale < 1.0:
                img_vips = img_vips.resize(scale)
            # Convert to PNG bytes for PIL
            png_bytes = img_vips.write_to_buffer(".png")
            input_image = Image.open(io.BytesIO(png_bytes))
        except Exception as e:
            logger.warning(f"pyvips failed: {e}, falling back to OpenCV/numpy")
            arr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                logger.error("cv2.imdecode failed")
                return {"error": "Failed to decode image with OpenCV"}
            max_size = 256
            h, w = img_cv.shape[:2]
            scale = min(max_size / h, max_size / w, 1.0)
            if scale < 1.0:
                img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if img_cv.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
            elif img_cv.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            else:
                logger.error(f"Unsupported channel count: {img_cv.shape[2]}")
                return {"error": "Unsupported image format"}
            input_image = Image.fromarray(img_rgb)
        # Remove background
        output_image = remove(input_image, session=rembg_session_fast)
        # Decide output format
        has_alpha = output_image.mode == "RGBA" and np.array(output_image)[..., 3].max() > 0
        buffered = io.BytesIO()
        if has_alpha:
            output_image.save(buffered, format="PNG")
            mime = "image/png"
        else:
            output_image = output_image.convert("RGB")
            output_image.save(buffered, format="JPEG", quality=90)
            mime = "image/jpeg"
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Log system usage after processing
        logger.info(f"RAM: {psutil.virtual_memory().percent}%, CPU: {psutil.cpu_percent()}% after processing")
        return {"image_base64": img_str, "mime": mime}
    except Exception as e:
        logger.exception(f"Failed to process image: {e}")
        return {"error": f"Failed to process image: {e}"}

class KeyMessageRequest(BaseModel):
    message: str
    message_type: MessageType
    tone: MessageTone
    target_audience: Optional[str] = None
    context: Optional[str] = None
    keywords: Optional[List[str]] = None
    max_length: Optional[int] = 10000

@app.post("/api/key-messages/generate")
async def generate_key_message(request: KeyMessageRequest):
    try:
        llm_service = LLMKeyMessageService()
        response = await llm_service.generate_message(
            message=request.message,
            message_type=request.message_type,
            tone=request.tone,
            target_audience=request.target_audience,
            context=request.context,
            keywords=request.keywords,
            max_length=request.max_length
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Para ejecutar localmente (desde la carpeta `ads-ia-backend`):
# pip install -r requirements.txt
# cp .env.example .env  (y edita .env)
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000