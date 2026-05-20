"""
Refactored LLM Engine for the Blaze AI module.

High-performance LLM engine with advanced features including model caching,
quantization, adaptive batching, and dynamic resource management.
"""

from __future__ import annotations

import asyncio
import gc
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, TextIteratorStreamer, GenerationConfig
)
from threading import Thread
import numpy as np

from . import Engine, EngineStatus
from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger

@dataclass
class LLMConfig:
    model_name: str = "gpt2"
    model_path: Optional[str] = None
    device: str = "auto"
    precision: str = "float16"
    enable_amp: bool = True
    cache_capacity: int = 1000
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    enable_streaming: bool = True
    batch_size: int = 1
    enable_quantization: bool = False
    quantization_bits: int = 8
    enable_dynamic_batching: bool = True
    max_batch_size: int = 8
    enable_memory_optimization: bool = True
    enable_gradient_checkpointing: bool = False

@dataclass
class GenerationRequest:
    prompt: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    stream: bool = False
    batch_id: Optional[str] = None

@dataclass
class GenerationResponse:
    text: str
    tokens: List[str]
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    batch_id: Optional[str] = None
    processing_time: float = 0.0

class ModelCache:
    def __init__(self, capacity: int = 1000):
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
        while len(self.cache) >= self.capacity or self._get_total_memory() + required_memory > self.capacity * 1000:
            if not self.access_order:
                break
            
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            del self.memory_usage[oldest_key]
            gc.collect()
    
    def _get_total_memory(self) -> int:
        return sum(self.memory_usage.values())

class DynamicBatcher:
    def __init__(self, max_batch_size: int = 8, timeout: float = 0.1):
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

class LLMEngine(Engine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.llm_config = LLMConfig(**config)
        self.model_cache = ModelCache(self.llm_config.cache_capacity)
        self.dynamic_batcher = DynamicBatcher(
            self.llm_config.max_batch_size,
            timeout=0.1
        )
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device: Optional[torch.device] = None
        self.generation_config: Optional[GenerationConfig] = None
    
    async def _initialize_engine(self) -> None:
        self.device = self._determine_device()
        self.tokenizer = await self._load_tokenizer()
        self.model = await self._load_model()
        self.generation_config = self._create_generation_config()
        
        if self.llm_config.enable_memory_optimization:
            self._optimize_memory()
    
    def _determine_device(self) -> torch.device:
        if self.llm_config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.llm_config.device)
    
    async def _load_tokenizer(self):
        cache_key = f"tokenizer_{self.llm_config.model_name}"
        cached_tokenizer = await self.model_cache.get(cache_key)
        
        if cached_tokenizer:
            return cached_tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.model_path or self.llm_config.model_name
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        await self.model_cache.set(cache_key, tokenizer, memory_estimate=1000)
        return tokenizer
    
    async def _load_model(self):
        cache_key = f"model_{self.llm_config.model_name}_{self.llm_config.precision}"
        cached_model = await self.model_cache.get(cache_key)
        
        if cached_model:
            return cached_model
        
        model_class = AutoModelForCausalLM
        if "t5" in self.llm_config.model_name.lower():
            model_class = AutoModelForSeq2SeqLM
        
        model = model_class.from_pretrained(
            self.llm_config.model_path or self.llm_config.model_name,
            torch_dtype=self._get_torch_dtype(),
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        if self.llm_config.enable_quantization:
            model = self._quantize_model(model)
        
        model.to(self.device)
        await self.model_cache.set(cache_key, model, memory_estimate=5000)
        return model
    
    def _get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(self.llm_config.precision, torch.float16)
    
    def _quantize_model(self, model):
        if self.llm_config.quantization_bits == 8:
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        return model
    
    def _create_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_length=self.llm_config.max_length,
            temperature=self.llm_config.temperature,
            top_p=self.llm_config.top_p,
            top_k=self.llm_config.top_k,
            repetition_penalty=self.llm_config.repetition_penalty,
            do_sample=self.llm_config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def _optimize_memory(self):
        if self.llm_config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
            self.model.enable_xformers_memory_efficient_attention()
    
    async def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Any:
        if operation == "generate":
            return await self._generate_text(params)
        elif operation == "generate_stream":
            return await self._generate_text_stream(params)
        elif operation == "batch_generate":
            return await self._batch_generate(params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _generate_text(self, params: Dict[str, Any]) -> GenerationResponse:
        request = GenerationRequest(**params)
        start_time = time.time()
        
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.llm_config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            if self.llm_config.enable_amp:
                with torch.autocast(device_type=self.device.type):
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=self.generation_config
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(outputs[0])
        
        processing_time = time.time() - start_time
        
        return GenerationResponse(
            text=generated_text,
            tokens=tokens,
            processing_time=processing_time,
            batch_id=request.batch_id
        )
    
    async def _generate_text_stream(self, params: Dict[str, Any]):
        request = GenerationRequest(**params)
        
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            generation_config=self.generation_config
        )
        
        thread = Thread(target=self._generate_in_thread, kwargs=generation_kwargs)
        thread.start()
        
        async def stream_generator():
            for text in streamer:
                yield text
            thread.join()
        
        return stream_generator()
    
    def _generate_in_thread(self, **kwargs):
        with torch.no_grad():
            if self.llm_config.enable_amp:
                with torch.autocast(device_type=self.device.type):
                    self.model.generate(**kwargs)
            else:
                self.model.generate(**kwargs)
    
    async def _batch_generate(self, params: Dict[str, Any]) -> List[GenerationResponse]:
        requests = [GenerationRequest(**req) for req in params.get("requests", [])]
        
        if not self.llm_config.enable_dynamic_batching:
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
                result = await self._generate_text({
                    "prompt": request.prompt,
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty,
                    "do_sample": request.do_sample,
                    "batch_id": request.batch_id
                })
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch generation failed for request: {e}")
                results.append(GenerationResponse(
                    text="",
                    tokens=[],
                    processing_time=0.0,
                    batch_id=request.batch_id
                ))
        
        return results
    
    async def shutdown(self):
        await super().shutdown()
        
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()


