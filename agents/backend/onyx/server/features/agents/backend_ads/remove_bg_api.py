from fastapi import APIRouter, Body, UploadFile, File
import base64
import io
import numpy as np
import cv2
import pyvips
import psutil
from loguru import logger
import httpx
from rembg import remove, new_session
from models import RemoveBackgroundRequest
import blosc
import imageio.v3 as iio
import asyncio
import aiofiles
import time
import gc
try:
    from PIL import Image
except ImportError:
    import pillow_simd as Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
rembg_session_fast = new_session("u2netp")

def get_resize_and_format(payload_img, default_size=256, default_format=None):
    resize = payload_img.get("resize") if isinstance(payload_img, dict) else None
    fmt = payload_img.get("output_format") if isinstance(payload_img, dict) else None
    return resize or default_size, fmt or default_format

async def process_single_image(image_bytes, resize=256, output_format=None):
    t0 = time.perf_counter()
    logger.info(f"RAM: {psutil.virtual_memory().percent}%, CPU: {psutil.cpu_percent()}% before processing")
    input_image = None
    # Try pyvips
    try:
        img_vips = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")
        scale = min(resize / img_vips.width, resize / img_vips.height, 1.0)
        if scale < 1.0:
            img_vips = img_vips.resize(scale)
        png_bytes = img_vips.write_to_buffer(".png")
        input_image = Image.open(io.BytesIO(png_bytes))
    except Exception as e:
        logger.warning(f"pyvips failed: {e}, falling back to OpenCV/numpy")
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise Exception("cv2.imdecode failed")
            scale = min(resize / img_cv.shape[1], resize / img_cv.shape[0], 1.0)
            if scale < 1.0:
                img_cv = cv2.resize(img_cv, (int(img_cv.shape[1] * scale), int(img_cv.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            if img_cv.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
            elif img_cv.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            else:
                raise Exception(f"Unsupported channel count: {img_cv.shape[2]}")
            input_image = Image.fromarray(img_rgb)
        except Exception as e2:
            logger.warning(f"OpenCV failed: {e2}, falling back to imageio")
            try:
                img_arr = iio.imread(image_bytes)
                if img_arr.shape[2] == 4:
                    img_rgb = img_arr[..., [0,1,2,3]]
                elif img_arr.shape[2] == 3:
                    img_rgb = img_arr
                else:
                    raise Exception(f"Unsupported channel count: {img_arr.shape[2]}")
                input_image = Image.fromarray(img_rgb)
            except Exception as e3:
                logger.error(f"All decoders failed: {e3}")
                return {"error": f"Failed to decode/process image: {e3}"}
    # Remove background
    try:
        output_image = remove(input_image, session=rembg_session_fast)
        has_alpha = output_image.mode == "RGBA" and np.array(output_image)[..., 3].max() > 0
        buffered = io.BytesIO()
        # Decide output format
        fmt = output_format or ("png" if has_alpha else "jpeg")
        if fmt == "png":
            output_image.save(buffered, format="PNG")
            mime = "image/png"
        else:
            output_image = output_image.convert("RGB")
            output_image.save(buffered, format="JPEG", quality=90)
            mime = "image/jpeg"
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode("utf-8")
        # Compress if large
        compressed = False
        if len(img_bytes) > 500_000:
            img_bytes = blosc.compress(img_bytes)
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            compressed = True
            logger.info(f"Compressed image with blosc, original: {len(buffered.getvalue())}, compressed: {len(img_bytes)} bytes")
        t1 = time.perf_counter()
        meta = {
            "mime": mime,
            "compressed": compressed,
            "original_size": len(image_bytes),
            "output_size": len(img_bytes),
            "processing_time": round(t1-t0, 4)
        }
        # Free memory
        del input_image, output_image, img_bytes, buffered
        gc.collect()
        logger.info(f"RAM: {psutil.virtual_memory().percent}%, CPU: {psutil.cpu_percent()}% after processing")
        return {"image_base64": img_str, "meta": meta}
    except Exception as e:
        logger.exception(f"Failed to process image: {e}")
        return {"error": f"Failed to process image: {e}"}

@router.post("/api/remove-background")
async def remove_background_endpoint(
    payload: RemoveBackgroundRequest = Body(None),
    file: UploadFile = File(None),
    files: list[UploadFile] = File(None)
):
    if remove is None:
        return {"error": "rembg library not installed. Please install with 'pip install rembg pillow pillow-simd pyvips opencv-python-headless imageio blosc'"}
    # Batch mode: files (multipart)
    if files:
        tasks = [process_single_image(await f.read()) for f in files]
        results = await asyncio.gather(*tasks)
        return {"results": results}
    # Single file upload
    if file is not None:
        image_bytes = await file.read()
        resize, fmt = 256, None
        if payload and hasattr(payload, "resize"): resize = payload.resize
        if payload and hasattr(payload, "output_format"): fmt = payload.output_format
        return await process_single_image(image_bytes, resize=resize, output_format=fmt)
    # Batch mode: list of base64/url in payload
    if payload and hasattr(payload, "images") and isinstance(payload.images, list):
        tasks = []
        for img in payload.images:
            resize, fmt = get_resize_and_format(img)
            if "image_url" in img:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(img["image_url"])
                    if resp.status_code != 200:
                        tasks.append(asyncio.create_task(asyncio.sleep(0, result={"error": f"Failed to download image: {resp.status_code}"})))
                        continue
                    img_bytes = resp.content
            elif "image_base64" in img:
                try:
                    img_bytes = base64.b64decode(img["image_base64"])
                except Exception as e:
                    tasks.append(asyncio.create_task(asyncio.sleep(0, result={"error": f"Invalid base64: {e}"})))
                    continue
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result={"error": "No image_url or image_base64 provided in batch item"})))
                continue
            tasks.append(process_single_image(img_bytes, resize=resize, output_format=fmt))
        results = await asyncio.gather(*tasks)
        return {"results": results}
    # Single image: url or base64
    image_bytes = None
    resize, fmt = 256, None
    if payload and hasattr(payload, "resize"): resize = payload.resize
    if payload and hasattr(payload, "output_format"): fmt = payload.output_format
    if payload and payload.image_url:
        async with httpx.AsyncClient() as client:
            resp = await client.get(payload.image_url)
            if resp.status_code != 200:
                logger.error(f"Failed to download image: {resp.status_code}")
                return {"error": f"Failed to download image: {resp.status_code}"}
            image_bytes = resp.content
    elif payload and payload.image_base64:
        try:
            image_bytes = base64.b64decode(payload.image_base64)
        except Exception as e:
            logger.error(f"Invalid base64: {e}")
            return {"error": f"Invalid base64: {e}"}
    else:
        logger.error("No image provided (file, image_url, or image_base64)")
        return {"error": "No image provided (file, image_url, or image_base64)"}
    return await process_single_image(image_bytes, resize=resize, output_format=fmt) 