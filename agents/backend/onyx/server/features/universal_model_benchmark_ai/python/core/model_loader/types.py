"""
Model Loader Types - Enums and configuration classes.

This module defines all types, enums, and configuration classes
used by the model loader system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ════════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════════

class ModelType(str, Enum):
    """Supported model types."""
    CAUSAL_LM = "causal_lm"
    VISION_LM = "vision_lm"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    SEQ2SEQ = "seq2seq"


class QuantizationType(str, Enum):
    """Supported quantization types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"


class BackendType(str, Enum):
    """Available backends."""
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"
    TENSORRT_LLM = "tensorrt_llm"
    AUTO = "auto"


# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASSES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Model loading configuration."""
    model_name: str
    model_path: Optional[str] = None
    model_type: ModelType = ModelType.CAUSAL_LM
    quantization: QuantizationType = QuantizationType.FP16
    device: str = "cuda"
    backend: BackendType = BackendType.AUTO
    use_flash_attention: bool = True
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    enable_prefix_caching: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ModelType",
    "QuantizationType",
    "BackendType",
    "ModelConfig",
    "GenerationConfig",
]












