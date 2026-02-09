"""
Refactored Diffusion Engine for the Blaze AI module.

High-performance diffusion engine with advanced features including model caching,
quantization, adaptive batching, and dynamic resource management.
"""

from __future__ import annotations

import asyncio
import gc
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import (
    DiffusionPipeline, StableDiffusionPipeline, DDIMPipeline,
    DPMSolverMultistepScheduler, EulerDiscreteScheduler
)
from PIL import Image
import numpy as np

from . import Engine, EngineStatus
from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger

@dataclass
class DiffusionConfig:
    model_name: str = "runwayml/stable-diffusion-v1-5"
    model_path: Optional[str] = None
    device: str = "auto"
    precision: str = "float16"
    enable_amp: bool = True
    cache_capacity: int = 100
    image_size: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    enable_safety_checker: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_sequential_cpu_offload: bool = False
    enable_model_cpu_offload: bool = False
    enable_memory_efficient_attention: bool = True
    enable_quantization: bool = False
    quantization_bits: int = 8
    enable_dynamic_batching: bool = True
    max_batch_size: int = 4
    enable_memory_optimization: bool = True
    enable_gradient_checkpointing: bool = False

@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    image_size: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    batch_id: Optional[str] = None

@dataclass
class GenerationResponse:
    images: List[Image.Image]
    metadata: Dict[str, Any]
    processing_time: float = 0.0
    batch_id: Optional[str] = None

class ModelCache:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.memory_usage: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, memory_estimate: int = 0):
        async with self._lock:
            if key in self.cache:
                self.access_order.remove(key)
            else:
                await self._evict_if_needed(memory_estimate)
            
            self.cache[key] = value
            self.access_order.append(key)
            self.memory_usage[key] = memory_estimate
    
    async def _evict_if_needed(self, required_memory: int):
        while len(self.cache) >= self.capacity or self._get_total_memory() + required_memory > self.capacity * 10000:
            if not self.access_order:
                break
            
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            del self.memory_usage[oldest_key]
            gc.collect()
    
    def _get_total_memory(self) -> int:
        return sum(self.memory_usage.values())

class DynamicBatcher:
    def __init__(self, max_batch_size: int = 4, timeout: float = 0.1):
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.batches: Dict[str, List[Tuple[asyncio.Future, GenerationRequest]]] = {}
        self._lock = asyncio.Lock()
    
    async def add_request(self, batch_id: str, request: GenerationRequest) -> asyncio.Future:
        future = asyncio.Future()
        
        async with self._lock:
            if batch_id not in self.batches:
                self.batches[batch_id] = []
            
            self.batches[batch_id].append((future, request))
            
            if len(self.batches[batch_id]) >= self.max_batch_size:
                await self._process_batch(batch_id)
        
        return future
    
    async def _process_batch(self, batch_id: str):
        if batch_id not in self.batches:
            return
        
        batch = self.batches.pop(batch_id)
        requests = [req for _, req in batch]
        futures = [future for future, _ in batch]
        
        try:
            results = await self._process_requests(requests)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    async def _process_requests(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        raise NotImplementedError

class DiffusionEngine(Engine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.diffusion_config = DiffusionConfig(**config)
        self.model_cache = ModelCache(self.diffusion_config.cache_capacity)
        self.dynamic_batcher = DynamicBatcher(
            self.diffusion_config.max_batch_size,
            timeout=0.1
        )
        self.pipeline: Optional[Any] = None
        self.device: Optional[torch.device] = None
        self.scheduler: Optional[Any] = None
    
    async def _initialize_engine(self) -> None:
        self.device = self._determine_device()
        self.pipeline = await self._load_pipeline()
        self.scheduler = self._create_scheduler()
        
        if self.diffusion_config.enable_memory_optimization:
            self._optimize_memory()
    
    def _determine_device(self) -> torch.device:
        if self.diffusion_config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.diffusion_config.device)
    
    async def _load_pipeline(self):
        cache_key = f"pipeline_{self.diffusion_config.model_name}_{self.diffusion_config.precision}"
        cached_pipeline = await self.model_cache.get(cache_key)
        
        if cached_pipeline:
            return cached_pipeline
        
        if "stable-diffusion" in self.diffusion_config.model_name.lower():
            pipeline_class = StableDiffusionPipeline
        elif "ddim" in self.diffusion_config.model_name.lower():
            pipeline_class = DDIMPipeline
        else:
            pipeline_class = DiffusionPipeline
        
        pipeline = pipeline_class.from_pretrained(
            self.diffusion_config.model_path or self.diffusion_config.model_name,
            torch_dtype=self._get_torch_dtype(),
            safety_checker=None if not self.diffusion_config.enable_safety_checker else None,
            requires_safety_checker=self.diffusion_config.enable_safety_checker
        )
        
        if self.diffusion_config.enable_quantization:
            pipeline = self._quantize_pipeline(pipeline)
        
        pipeline.to(self.device)
        await self.model_cache.set(cache_key, pipeline, memory_estimate=10000)
        return pipeline
    
    def _get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.diffusion_config.precision, torch.float16)
    
    def _quantize_pipeline(self, pipeline):
        if self.diffusion_config.quantization_bits == 8:
            for component in [pipeline.unet, pipeline.vae, pipeline.text_encoder]:
                if hasattr(component, 'quantize'):
                    component.quantize()
        return pipeline
    
    def _create_scheduler(self):
        if "stable-diffusion" in self.diffusion_config.model_name.lower():
            return DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        elif "ddim" in self.diffusion_config.model_name.lower():
            return EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        return self.pipeline.scheduler
    
    def _optimize_memory(self):
        if self.diffusion_config.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.diffusion_config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.diffusion_config.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        
        if self.diffusion_config.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        
        if self.diffusion_config.enable_memory_efficient_attention:
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                self.pipeline.enable_xformers_memory_efficient_attention()
        
        if self.diffusion_config.enable_gradient_checkpointing:
            self.pipeline.unet.enable_gradient_checkpointing()
    
    async def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "generate":
            return await self._generate_image(params)
        elif operation == "generate_batch":
            return await self._generate_image_batch(params)
        elif operation == "img2img":
            return await self._img2img_generation(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _generate_image(self, params: Dict[str, Any]) -> GenerationResponse:
        request = GenerationRequest(**params)
        start_time = time.time()
        
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(request.seed)
        
        generation_kwargs = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "height": request.image_size or self.diffusion_config.image_size,
            "width": request.image_size or self.diffusion_config.image_size,
            "num_inference_steps": request.num_inference_steps or self.diffusion_config.num_inference_steps,
            "guidance_scale": request.guidance_scale or self.diffusion_config.guidance_scale,
            "scheduler": self.scheduler
        }
        
        with torch.no_grad():
            if self.diffusion_config.enable_amp:
                with torch.autocast(device_type=self.device.type):
                    result = self.pipeline(**generation_kwargs)
            else:
                result = self.pipeline(**generation_kwargs)
        
        processing_time = time.time() - start_time
        
        return GenerationResponse(
            images=result.images,
            metadata={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed,
                "guidance_scale": generation_kwargs["guidance_scale"],
                "num_inference_steps": generation_kwargs["num_inference_steps"]
            },
            processing_time=processing_time,
            batch_id=request.batch_id
        )
    
    async def _generate_image_batch(self, params: Dict[str, Any]) -> List[GenerationResponse]:
        requests = [GenerationRequest(**req) for req in params.get("requests", [])]
        
        if not self.diffusion_config.enable_dynamic_batching:
            return await self._simple_batch_generate(requests)
        
        batch_id = f"batch_{int(time.time() * 1000)}"
        futures = []
        
        for request in requests:
            request.batch_id = batch_id
            future = await self.dynamic_batcher.add_request(batch_id, request)
            futures.append(future)
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    async def _simple_batch_generate(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        results = []
        for request in requests:
            try:
                result = await self._generate_image({
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "image_size": request.image_size,
                    "num_inference_steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": request.seed,
                    "batch_id": request.batch_id
                })
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch generation failed for request: {e}")
                results.append(GenerationResponse(
                    images=[],
                    metadata={},
                    processing_time=0.0,
                    batch_id=request.batch_id
                ))
        
        return results
    
    async def _img2img_generation(self, params: Dict[str, Any]) -> GenerationResponse:
        start_time = time.time()
        
        image = params.get("image")
        if image is None:
            raise ValueError("Image is required for img2img generation")
        
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        strength = params.get("strength", 0.8)
        guidance_scale = params.get("guidance_scale", self.diffusion_config.guidance_scale)
        num_inference_steps = params.get("num_inference_steps", self.diffusion_config.num_inference_steps)
        
        if not hasattr(self.pipeline, 'img2img'):
            raise ValueError("This pipeline does not support img2img generation")
        
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "scheduler": self.scheduler
        }
        
        with torch.no_grad():
            if self.diffusion_config.enable_amp:
                with torch.autocast(device_type=self.device.type):
                    result = self.pipeline.img2img(**generation_kwargs)
            else:
                result = self.pipeline.img2img(**generation_kwargs)
        
        processing_time = time.time() - start_time
        
        return GenerationResponse(
            images=result.images,
            metadata={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            },
            processing_time=processing_time
        )
    
    async def shutdown(self):
        await super().shutdown()
        
        if self.pipeline:
            del self.pipeline
        if self.scheduler:
            del self.scheduler
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()


