"""
LLM Service - Servicio avanzado de Large Language Models
========================================================

Sistema profesional para trabajar con LLMs usando Transformers.
Sigue mejores prácticas de PyTorch y Transformers.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import nullcontext

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        pipeline,
    )
    import torch
    from torch.cuda.amp import autocast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Using simulation mode.")


@dataclass
class LLMConfig:
    """Configuración de LLM"""
    model_name: str = "gpt2"
    device_map: Optional[str] = "auto"  # auto, cuda, cpu
    torch_dtype: str = "float16"  # float16, float32, bfloat16
    trust_remote_code: bool = False
    use_gpu: bool = True
    max_memory: Optional[Dict[int, str]] = None
    low_cpu_mem_usage: bool = True
    # Generation parameters
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1


@dataclass
class GenerationResult:
    """Resultado de generación"""
    text: str
    prompt: str
    model_name: str
    generation_time: float
    num_tokens: int
    config: LLMConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMService:
    """Servicio avanzado de LLMs"""
    
    def __init__(self, default_config: Optional[LLMConfig] = None):
        """
        Inicializar servicio de LLM.
        
        Args:
            default_config: Configuración por defecto
        """
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.configs: Dict[str, LLMConfig] = {}
        self.default_config = default_config or LLMConfig()
        
        # Setup device
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.default_config.use_gpu
                else "cpu"
            )
        else:
            self.device = None
        
        logger.info(
            f"LLMService initialized on device: {self.device} "
            f"(Transformers: {TRANSFORMERS_AVAILABLE})"
        )
    
    def load_model(
        self,
        model_name: str,
        config: Optional[LLMConfig] = None
    ) -> bool:
        """
        Cargar modelo LLM con optimizaciones.
        
        Args:
            model_name: Nombre del modelo (HuggingFace ID)
            config: Configuración personalizada
        
        Returns:
            True si se cargó exitosamente
        """
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        try:
            if TRANSFORMERS_AVAILABLE:
                model_config = config or self.default_config
                
                # Determine device
                device_map = model_config.device_map
                if not model_config.use_gpu:
                    device_map = None
                
                # Determine dtype
                dtype_map = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }
                torch_dtype = dtype_map.get(
                    model_config.torch_dtype,
                    torch.float16
                )
                
                # Load tokenizer
                logger.info(f"Loading tokenizer for {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=model_config.trust_remote_code,
                )
                
                # Set pad token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model
                logger.info(f"Loading model {model_name}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=model_config.trust_remote_code,
                    low_cpu_mem_usage=model_config.low_cpu_mem_usage,
                    max_memory=model_config.max_memory,
                )
                
                # Move to device if device_map is None
                if device_map is None:
                    model = model.to(self.device)
                
                model.eval()  # Set to evaluation mode
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.configs[model_name] = model_config
                
                logger.info(f"Model {model_name} loaded successfully")
                return True
            else:
                # Simulation mode
                logger.warning(f"Simulation mode: Model {model_name} not actually loaded")
                self.models[model_name] = None
                self.tokenizers[model_name] = None
                self.configs[model_name] = config or self.default_config
                return True
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            return False
    
    def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generar texto usando el modelo LLM.
        
        Args:
            prompt: Texto de entrada
            model_name: Nombre del modelo (usa default si None)
            max_tokens: Máximo número de tokens a generar
            temperature: Temperatura para sampling
            top_p: Nucleus sampling
            top_k: Top-k sampling
            **kwargs: Parámetros adicionales de generación
        
        Returns:
            GenerationResult con el texto generado
        """
        import time
        start_time = time.time()
        
        # Determine model to use
        model_key = model_name or list(self.models.keys())[0] if self.models else None
        if not model_key:
            raise ValueError("No model loaded. Call load_model() first.")
        
        config = self.configs.get(model_key, self.default_config)
        
        # Override config with provided parameters
        max_new_tokens = max_tokens or config.max_new_tokens
        temp = temperature if temperature is not None else config.temperature
        top_p_val = top_p if top_p is not None else config.top_p
        top_k_val = top_k if top_k is not None else config.top_k
        
        try:
            if TRANSFORMERS_AVAILABLE and self.models.get(model_key):
                model = self.models[model_key]
                tokenizer = self.tokenizers[model_key]
                
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Move to device
                if hasattr(model, "device"):
                    device = next(model.parameters()).device
                else:
                    device = self.device
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Create generation config
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_p=top_p_val,
                    top_k=top_k_val,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    num_return_sequences=config.num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **kwargs
                )
                
                # Generate with mixed precision if using GPU
                torch_dtype = getattr(torch, config.torch_dtype, torch.float16)
                autocast_context = (
                    autocast(device_type=device.type, dtype=torch_dtype)
                    if device.type == "cuda" and config.torch_dtype == "float16"
                    else nullcontext()
                )
                
                with torch.no_grad():
                    with autocast_context:
                        outputs = model.generate(
                            **inputs,
                            generation_config=generation_config
                        )
                
                # Decode output
                generated_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                num_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
                
            else:
                # Simulation mode
                generated_text = f"[Simulated generation for: {prompt[:50]}...]"
                num_tokens = max_new_tokens
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                text=generated_text,
                prompt=prompt,
                model_name=model_key,
                generation_time=generation_time,
                num_tokens=num_tokens,
                config=config,
                metadata={
                    "temperature": temp,
                    "top_p": top_p_val,
                    "top_k": top_k_val,
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            raise
    
    def get_embeddings(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> List[float]:
        """
        Obtener embeddings del texto.
        
        Args:
            text: Texto a procesar
            model_name: Nombre del modelo
        
        Returns:
            Lista de embeddings
        """
        model_key = model_name or list(self.models.keys())[0] if self.models else None
        if not model_key:
            raise ValueError("No model loaded")
        
        try:
            if TRANSFORMERS_AVAILABLE and self.models.get(model_key):
                model = self.models[model_key]
                tokenizer = self.tokenizers[model_key]
                
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                
                if hasattr(model, "device"):
                    device = next(model.parameters()).device
                else:
                    device = self.device
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    # Use last hidden state mean pooling
                    embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()
                    return embeddings.cpu().tolist()
            else:
                # Simulation
                return [0.0] * 768
                
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}", exc_info=True)
            return []
    
    def unload_model(self, model_name: str) -> bool:
        """Descargar modelo de memoria"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                del self.tokenizers[model_name]
                del self.configs[model_name]
                
                # Clear CUDA cache if using GPU
                if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Model {model_name} unloaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    def list_loaded_models(self) -> List[str]:
        """Listar modelos cargados"""
        return list(self.models.keys())
