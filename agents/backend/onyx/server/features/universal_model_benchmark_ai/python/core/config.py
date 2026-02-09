"""
Configuration Module - Comprehensive Configuration Management.

Features:
- YAML-based configuration
- Environment variable support
- Auto-detection of devices
- Validation and defaults
- Type-safe configuration classes
"""

import os
import yaml
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════════

class DeviceType(str, Enum):
    """Device types for model execution."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class QuantizationType(str, Enum):
    """Quantization types for model optimization."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"


class ModelType(str, Enum):
    """Model types."""
    CAUSAL_LM = "causal_lm"
    VISION_LM = "vision_lm"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    SEQ2SEQ = "seq2seq"


# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASSES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    Model configuration.
    
    Attributes:
        name: Model name or HuggingFace ID
        path: Local path to model (optional)
        model_type: Type of model
        quantization: Quantization type
        device: Device to use
        max_seq_length: Maximum sequence length
        trust_remote_code: Trust remote code from HuggingFace
        extra_kwargs: Additional model-specific arguments
    """
    name: str
    path: Optional[str] = None
    model_type: ModelType = ModelType.CAUSAL_LM
    quantization: QuantizationType = QuantizationType.FP16
    device: DeviceType = DeviceType.AUTO
    max_seq_length: Optional[int] = None
    trust_remote_code: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Model name is required")
        
        # Convert string enums if needed
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type.lower())
        if isinstance(self.quantization, str):
            self.quantization = QuantizationType(self.quantization.lower())
        if isinstance(self.device, str):
            self.device = DeviceType(self.device.lower())


@dataclass
class BenchmarkConfig:
    """
    Benchmark configuration.
    
    Attributes:
        name: Benchmark name
        dataset: Dataset name (HuggingFace)
        shots: Number of few-shot examples
        max_samples: Maximum samples to evaluate
        batch_size: Batch size for evaluation
        extra_kwargs: Additional benchmark-specific arguments
    """
    name: str
    dataset: Optional[str] = None
    shots: int = 0
    max_samples: Optional[int] = None
    batch_size: int = 1
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Benchmark name is required")
        
        if self.shots < 0:
            raise ValueError("Shots must be non-negative")
        
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive")


@dataclass
class ExecutionConfig:
    """
    Execution configuration.
    
    Attributes:
        workers: Number of parallel workers
        batch_size: Default batch size
        device: Default device
        timeout_seconds: Timeout for operations
        max_memory_gb: Maximum memory usage in GB
        retry_attempts: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds
        extra_kwargs: Additional execution arguments
    """
    workers: int = 4
    batch_size: int = 32
    device: DeviceType = DeviceType.AUTO
    timeout_seconds: Optional[int] = None
    max_memory_gb: Optional[float] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if isinstance(self.device, str):
            self.device = DeviceType(self.device.lower())


@dataclass
class SystemConfig:
    """
    Complete system configuration.
    
    Combines model, benchmark, and execution configurations.
    """
    models: List[ModelConfig] = field(default_factory=list)
    benchmarks: List[BenchmarkConfig] = field(default_factory=list)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "SystemConfig":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
        
        Returns:
            SystemConfig instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Parse models
        models = []
        for model_data in data.get("models", []):
            # Convert string enums
            if "model_type" in model_data and isinstance(model_data["model_type"], str):
                model_data["model_type"] = ModelType(model_data["model_type"].lower())
            if "quantization" in model_data and isinstance(model_data["quantization"], str):
                model_data["quantization"] = QuantizationType(model_data["quantization"].lower())
            if "device" in model_data and isinstance(model_data["device"], str):
                model_data["device"] = DeviceType(model_data["device"].lower())
            
            models.append(ModelConfig(**model_data))
        
        # Parse benchmarks
        benchmarks = []
        for benchmark_data in data.get("benchmarks", []):
            benchmarks.append(BenchmarkConfig(**benchmark_data))
        
        # Parse execution config
        execution_data = data.get("execution", {})
        if "device" in execution_data and isinstance(execution_data["device"], str):
            execution_data["device"] = DeviceType(execution_data["device"].lower())
        execution = ExecutionConfig(**execution_data)
        
        return cls(
            models=models,
            benchmarks=benchmarks,
            execution=execution,
        )
    
    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        data = {
            "models": [asdict(model) for model in self.models],
            "benchmarks": [asdict(benchmark) for benchmark in self.benchmarks],
            "execution": asdict(self.execution),
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by name.
        
        Args:
            name: Model name
        
        Returns:
            ModelConfig or None if not found
        """
        for model in self.models:
            if model.name == name:
                return model
        return None
    
    def get_benchmark_config(self, name: str) -> Optional[BenchmarkConfig]:
        """
        Get benchmark configuration by name.
        
        Args:
            name: Benchmark name
        
        Returns:
            BenchmarkConfig or None if not found
        """
        for benchmark in self.benchmarks:
            if benchmark.name == name:
                return benchmark
        return None
    
    def add_model(self, model: ModelConfig) -> None:
        """Add a model configuration."""
        self.models.append(model)
    
    def add_benchmark(self, benchmark: BenchmarkConfig) -> None:
        """Add a benchmark configuration."""
        self.benchmarks.append(benchmark)
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.models:
            errors.append("No models configured")
        
        if not self.benchmarks:
            errors.append("No benchmarks configured")
        
        # Validate models
        model_names = set()
        for model in self.models:
            if model.name in model_names:
                errors.append(f"Duplicate model name: {model.name}")
            model_names.add(model.name)
        
        # Validate benchmarks
        benchmark_names = set()
        for benchmark in self.benchmarks:
            if benchmark.name in benchmark_names:
                errors.append(f"Duplicate benchmark name: {benchmark.name}")
            benchmark_names.add(benchmark.name)
        
        return errors


# ════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def load_config(path: Optional[Path] = None) -> SystemConfig:
    """
    Load configuration from file or environment.
    
    Tries multiple locations:
    1. Provided path
    2. Environment variable BENCHMARK_CONFIG
    3. config/config.yaml
    4. config/example.yaml
    5. config.yaml in current directory
    
    Args:
        path: Optional path to config file
    
    Returns:
        SystemConfig instance (with defaults if no config found)
    """
    if path is None:
        # Try environment variable
        env_path = os.getenv("BENCHMARK_CONFIG")
        if env_path:
            path = Path(env_path)
        
        # Try default locations
        if path is None or not path.exists():
            default_paths = [
                Path("config/config.yaml"),
                Path("config/example.yaml"),
                Path.cwd() / "config.yaml",
            ]
            
            for default_path in default_paths:
                if default_path.exists():
                    path = default_path
                    break
    
    if path is None or not path.exists():
        logger.warning("No config file found, using defaults")
        return SystemConfig()
    
    try:
        config = SystemConfig.from_yaml(path)
        validation_errors = config.validate()
        if validation_errors:
            logger.warning(f"Config validation warnings: {validation_errors}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}, using defaults")
        return SystemConfig()


def get_device() -> DeviceType:
    """
    Get device type from environment or auto-detect.
    
    Auto-detection priority:
    1. CUDA (if CUDA_VISIBLE_DEVICES is set or torch.cuda.is_available())
    2. MPS (if torch.backends.mps.is_available())
    3. CPU (fallback)
    
    Returns:
        DeviceType
    """
    device_str = os.getenv("DEVICE", "auto").lower()
    
    if device_str != "auto":
        try:
            return DeviceType(device_str)
        except ValueError:
            logger.warning(f"Invalid device type: {device_str}, using auto")
    
    # Auto-detect
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        return DeviceType.CUDA
    
    try:
        import torch
        if torch.cuda.is_available():
            return DeviceType.CUDA
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceType.MPS
    except ImportError:
        logger.debug("PyTorch not available for device detection")
    
    return DeviceType.CPU


def create_default_config(path: Path) -> None:
    """
    Create a default configuration file.
    
    Args:
        path: Path to save default config
    """
    default_config = SystemConfig(
        models=[
            ModelConfig(
                name="meta-llama/Llama-2-7b-hf",
                quantization=QuantizationType.FP16,
                device=DeviceType.AUTO,
            )
        ],
        benchmarks=[
            BenchmarkConfig(
                name="mmlu",
                shots=5,
                max_samples=100,
            )
        ],
        execution=ExecutionConfig(
            workers=4,
            batch_size=32,
        )
    )
    
    default_config.to_yaml(path)
    logger.info(f"Created default config at {path}")


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    "DeviceType",
    "QuantizationType",
    "ModelType",
    "ModelConfig",
    "BenchmarkConfig",
    "ExecutionConfig",
    "SystemConfig",
    "load_config",
    "get_device",
    "create_default_config",
]
