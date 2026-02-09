# ⚡ Especificación de Motores de Inferencia - Optimization Core

## 📋 Resumen

Este documento especifica los motores de inferencia de alto rendimiento para LLMs, incluyendo vLLM, TensorRT-LLM, y motores genéricos.

## 🎯 Objetivos

1. **Alto Rendimiento**: 5-10x más rápido que PyTorch estándar
2. **Eficiencia de Memoria**: Reducción de 3-5x en uso de memoria
3. **Batching Optimizado**: Continuous batching y PagedAttention
4. **Multi-GPU**: Soporte para tensor parallelism
5. **Cuantización**: Soporte para AWQ, GPTQ, etc.

## 🏗️ Arquitectura

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────┐
│              IInferenceEngine (Interface)               │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐
│ BaseEngine   │ │ VLLMEngine │ │TensorRT   │
│ (Abstract)   │ │            │ │Engine      │
└──────┬───────┘ └────────────┘ └───────────┘
       │
┌──────▼──────┐
│ GenericEngine│
│ (PyTorch)   │
└─────────────┘
```

## 📦 Componentes

### BaseInferenceEngine

**Propósito**: Clase base abstracta con funcionalidad común.

**Especificación**:

```python
class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    
    Provides common interface and shared functionality.
    """
    
    def __init__(
        self,
        model: Union[str, Path],
        **kwargs
    ):
        """
        Initialize base inference engine.
        
        Args:
            model: Model name (HuggingFace) or path
            **kwargs: Engine-specific parameters
        """
        self.model_path = Path(model) if isinstance(model, (str, Path)) else model
        self._initialized = False
        self._model_loaded = False
        self._config = self._default_config()
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def _initialize_engine(self, **kwargs) -> Any:
        """
        Initialize the underlying engine.
        
        Must be implemented by subclasses.
        
        Returns:
            Initialized engine object
        """
        pass
    
    @abstractmethod
    def _load_model_impl(
        self,
        model: Union[str, Path],
        **kwargs
    ) -> bool:
        """
        Load model implementation.
        
        Must be implemented by subclasses.
        
        Returns:
            True if successful
        """
        pass
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters
        
        Returns:
            Generated text(s)
        """
        # Normalize prompts
        prompts_list, was_single = self._normalize_prompts(prompts)
        
        # Validate
        self._validate_generation_params(max_tokens, temperature, top_p)
        
        # Generate
        results = self._generate_impl(
            prompts_list,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Return in same format as input
        return results[0] if was_single else results
    
    @abstractmethod
    def _generate_impl(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Implementation of generation.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _normalize_prompts(
        self,
        prompts: Union[str, List[str]]
    ) -> Tuple[List[str], bool]:
        """Normalize prompts to list format."""
        if isinstance(prompts, str):
            return [prompts], True
        return list(prompts), False
    
    def _validate_generation_params(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> None:
        """Validate generation parameters."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be in [0.0, 1.0]")
```

### VLLMEngine

**Propósito**: Motor de inferencia usando vLLM (5-10x más rápido).

**Especificación**:

```python
class VLLMEngine(BaseInferenceEngine):
    """
    vLLM inference engine.
    
    Features:
    - PagedAttention (3-5x memory reduction)
    - Continuous batching
    - Multi-GPU support
    - Quantization (AWQ, GPTQ)
    """
    
    def __init__(
        self,
        model: Union[str, Path],
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "float16",
        quantization: Optional[str] = None,
        max_model_len: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize vLLM engine.
        
        Args:
            model: Model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            dtype: Model dtype (float16, bfloat16, float32)
            quantization: Quantization method (awq, gptq, None)
            max_model_len: Maximum sequence length
        """
        super().__init__(model, **kwargs)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.quantization = quantization
        self.max_model_len = max_model_len
        self._llm = None
    
    def _initialize_engine(self, **kwargs) -> Any:
        """Initialize vLLM LLM object."""
        from vllm import LLM
        
        llm_params = {
            "model": str(self.model_path),
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
        }
        
        if self.quantization:
            llm_params["quantization"] = self.quantization
        
        if self.max_model_len:
            llm_params["max_model_len"] = self.max_model_len
        
        llm_params.update(kwargs)
        
        self._llm = LLM(**llm_params)
        return self._llm
    
    def _load_model_impl(
        self,
        model: Union[str, Path],
        **kwargs
    ) -> bool:
        """Load model (vLLM loads on initialization)."""
        if self._llm is None:
            self._initialize_engine(**kwargs)
        self._model_loaded = True
        return True
    
    def _generate_impl(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Generate using vLLM."""
        if self._llm is None:
            raise NotInitializedError("Engine not initialized")
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        outputs = self._llm.generate(prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information."""
        if self._llm is None:
            raise NotInitializedError("Engine not initialized")
        
        return {
            "model_name": str(self.model_path),
            "engine": "vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
```

**Características**:
- PagedAttention para eficiencia de memoria
- Continuous batching automático
- Soporte multi-GPU
- Cuantización AWQ/GPTQ

### TensorRTLLMEngine

**Propósito**: Motor de inferencia usando TensorRT-LLM (2-10x más rápido en GPUs NVIDIA).

**Especificación**:

```python
class TensorRTLLMEngine(BaseInferenceEngine):
    """
    TensorRT-LLM inference engine.
    
    Features:
    - Optimized CUDA kernels
    - INT8/FP16/BF16 support
    - Multi-GPU support
    - Custom CUDA kernels
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        engine_dir: Optional[Union[str, Path]] = None,
        max_batch_size: int = 8,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        dtype: str = "float16",
        **kwargs
    ):
        """
        Initialize TensorRT-LLM engine.
        
        Args:
            model_path: Path to model or engine
            engine_dir: Directory with compiled engine
            max_batch_size: Maximum batch size
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
            dtype: Precision (float16, bfloat16, int8)
        """
        super().__init__(model_path, **kwargs)
        self.engine_dir = Path(engine_dir) if engine_dir else None
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.dtype = dtype
        self._trt_llm = None
    
    def _initialize_engine(self, **kwargs) -> Any:
        """Initialize TensorRT-LLM runtime."""
        from tensorrt_llm.runtime import PYRuntime
        
        if self.engine_dir is None:
            raise ValueError("engine_dir required for TensorRT-LLM")
        
        runtime = PYRuntime(
            engine_dir=str(self.engine_dir),
            tokenizer_dir=str(self.model_path),
            **kwargs
        )
        
        self._trt_llm = runtime
        return runtime
    
    def _generate_impl(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Generate using TensorRT-LLM."""
        if self._trt_llm is None:
            raise NotInitializedError("Engine not initialized")
        
        # Tokenize
        input_ids = self._trt_llm.tokenizer.encode(prompts)
        
        # Generate
        outputs = self._trt_llm.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        # Decode
        results = self._trt_llm.tokenizer.decode(outputs)
        return results
```

**Características**:
- Kernels CUDA optimizados
- Soporte INT8/FP16/BF16
- Compilación estática para máximo rendimiento

### GenericEngine

**Propósito**: Motor genérico usando PyTorch (fallback).

**Especificación**:

```python
class GenericEngine(BaseInferenceEngine):
    """
    Generic PyTorch inference engine.
    
    Fallback when vLLM/TensorRT not available.
    """
    
    def __init__(
        self,
        model: Union[str, Path],
        device: str = "cuda",
        torch_dtype: str = "float16",
        **kwargs
    ):
        """
        Initialize generic engine.
        
        Args:
            model: Model name or path
            device: Device (cuda, cpu)
            torch_dtype: Dtype (float16, bfloat16, float32)
        """
        super().__init__(model, **kwargs)
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
    
    def _load_model_impl(
        self,
        model: Union[str, Path],
        **kwargs
    ) -> bool:
        """Load model using transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model),
            **kwargs
        )
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model),
            torch_dtype=dtype_map[self.torch_dtype],
            device_map=self.device,
            **kwargs
        )
        
        self._model.eval()
        return True
    
    def _generate_impl(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Generate using PyTorch."""
        if self._model is None or self._tokenizer is None:
            raise NotInitializedError("Model not loaded")
        
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                **kwargs
            )
        
        # Decode
        results = self._tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return results
```

## 🏭 EngineFactory

**Propósito**: Factory para crear motores de inferencia.

**Especificación**:

```python
class EngineType(Enum):
    """Engine type enumeration."""
    AUTO = "auto"  # Auto-select best available
    VLLM = "vllm"
    TENSORRT = "tensorrt"
    GENERIC = "generic"

class EngineFactory:
    """Factory for creating inference engines."""
    
    @staticmethod
    def create_engine(
        model: Union[str, Path],
        engine_type: EngineType = EngineType.AUTO,
        **kwargs
    ) -> BaseInferenceEngine:
        """
        Create inference engine.
        
        Args:
            model: Model name or path
            engine_type: Engine type (AUTO selects best available)
            **kwargs: Engine-specific parameters
        
        Returns:
            Inference engine instance
        """
        if engine_type == EngineType.AUTO:
            engine_type = EngineFactory._select_best_engine()
        
        if engine_type == EngineType.VLLM:
            return VLLMEngine(model, **kwargs)
        elif engine_type == EngineType.TENSORRT:
            return TensorRTLLMEngine(model, **kwargs)
        elif engine_type == EngineType.GENERIC:
            return GenericEngine(model, **kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
    
    @staticmethod
    def _select_best_engine() -> EngineType:
        """Select best available engine."""
        # Try vLLM first
        try:
            import vllm
            return EngineType.VLLM
        except ImportError:
            pass
        
        # Try TensorRT-LLM
        try:
            import tensorrt_llm
            return EngineType.TENSORRT
        except ImportError:
            pass
        
        # Fallback to generic
        return EngineType.GENERIC
```

## 📊 Métricas y Rendimiento

### Métricas Esperadas

| Engine | Latency (ms/token) | Throughput (tokens/s) | Memory (GB for 7B) |
|--------|-------------------|---------------------|-------------------|
| vLLM | 20-30 | 2000-5000 | 8-12 |
| TensorRT-LLM | 15-25 | 3000-6000 | 6-10 |
| Generic (PyTorch) | 50-100 | 500-1000 | 14-20 |

### Optimizaciones

1. **PagedAttention**: Reduce memoria 3-5x
2. **Continuous Batching**: Mejora utilización GPU
3. **Tensor Parallelism**: Escala a múltiples GPUs
4. **Quantization**: Reduce memoria y mejora velocidad

## 🧪 Testing

### Tests Requeridos

1. **Unit Tests**: Cada método individual
2. **Integration Tests**: Flujo completo de generación
3. **Performance Tests**: Benchmarks de rendimiento
4. **Memory Tests**: Uso de memoria

### Ejemplo de Test

```python
def test_vllm_engine():
    engine = VLLMEngine("mistralai/Mistral-7B-Instruct-v0.2")
    engine.initialize()
    engine.load_model()
    
    result = engine.generate("Hello, world!", max_tokens=10)
    assert isinstance(result, str)
    assert len(result) > 0
```

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025




