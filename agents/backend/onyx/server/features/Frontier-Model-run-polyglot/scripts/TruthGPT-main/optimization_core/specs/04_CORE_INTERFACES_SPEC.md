# 🔌 Especificación de Interfaces Core - Optimization Core

## 📋 Resumen

Este documento especifica todas las interfaces y contratos base del sistema `optimization_core`. Estas interfaces definen los contratos que deben cumplir todos los componentes del sistema.

## 🎯 Principios de Diseño

1. **Interfaces Primero**: Todas las funcionalidades se definen primero como interfaces
2. **Contratos Claros**: Cada interfaz especifica claramente qué debe hacer, no cómo
3. **Extensibilidad**: Fácil agregar nuevas implementaciones
4. **Testabilidad**: Interfaces permiten fácil mocking y testing

## 📦 Interfaces Base

### IComponent

**Propósito**: Interfaz base para todos los componentes del sistema.

**Especificación**:

```python
class IComponent(ABC):
    """Base interface for all components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get component name.
        
        Returns:
            Component name (e.g., "VLLMEngine", "PolarsProcessor")
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Get component version.
        
        Returns:
            Version string (e.g., "1.0.0", "2.3.1")
        """
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the component.
        
        Args:
            **kwargs: Initialization parameters (component-specific)
        
        Returns:
            True if initialization successful, False otherwise
        
        Raises:
            InitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup component resources.
        
        This method should:
        - Release all allocated resources
        - Close connections
        - Free memory
        - Be idempotent (safe to call multiple times)
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Dictionary with status information:
            {
                "name": str,
                "version": str,
                "initialized": bool,
                "ready": bool,
                "health": str,  # "healthy", "degraded", "unhealthy"
                "metrics": Dict[str, Any],  # Component-specific metrics
                "last_error": Optional[str],
                "uptime_seconds": float
            }
        """
        pass
```

**Contrato**:
- `name` y `version` deben ser constantes (no cambian durante la vida del componente)
- `initialize()` debe ser llamado antes de usar el componente
- `cleanup()` debe ser llamado cuando el componente ya no se necesita
- `get_status()` debe ser thread-safe y no bloquear

### IInferenceEngine

**Propósito**: Interfaz para motores de inferencia de LLMs.

**Especificación**:

```python
class IInferenceEngine(IComponent):
    """Interface for inference engines."""
    
    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts.
        
        Args:
            prompts: Single prompt string or list of prompt strings
            config: Generation configuration (max_tokens, temperature, etc.)
            **kwargs: Additional generation parameters:
                - max_tokens: int (overrides config)
                - temperature: float (overrides config)
                - top_p: float (overrides config)
                - top_k: int (overrides config)
                - repetition_penalty: float (overrides config)
                - stop_sequences: List[str] (overrides config)
                - stream: bool (streaming generation)
                - seed: Optional[int] (random seed)
        
        Returns:
            If single prompt: single string
            If list of prompts: list of strings (same order)
        
        Raises:
            NotInitializedError: If engine not initialized
            InvalidInputError: If prompts invalid
            GenerationError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information:
            {
                "model_name": str,
                "model_type": str,
                "vocab_size": int,
                "max_seq_len": int,
                "num_layers": int,
                "hidden_size": int,
                "num_heads": int,
                "dtype": str,  # "float32", "float16", "bfloat16", "int8"
                "quantization": Optional[str],  # "awq", "gptq", None
                "memory_usage_gb": float,
                "device": str,  # "cuda:0", "cpu", etc.
            }
        """
        pass
    
    @abstractmethod
    def load_model(
        self,
        model: Union[str, Path],
        **kwargs
    ) -> bool:
        """
        Load a model.
        
        Args:
            model: Model name (HuggingFace) or path to model
            **kwargs: Model loading parameters:
                - dtype: str (precision)
                - quantization: str (quantization method)
                - tensor_parallel_size: int (multi-GPU)
                - gpu_memory_utilization: float (0.0-1.0)
        
        Returns:
            True if model loaded successfully
        
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload the current model.
        
        Frees memory and resources.
        """
        pass
    
    @property
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        pass
```

**Contrato**:
- `generate()` debe manejar tanto strings individuales como listas
- `generate()` debe preservar el orden de los prompts
- `generate()` debe ser thread-safe si el engine lo soporta
- `get_model_info()` debe retornar información actualizada

### IDataProcessor

**Propósito**: Interfaz para procesadores de datos.

**Especificación**:

```python
class IDataProcessor(IComponent):
    """Interface for data processors."""
    
    @abstractmethod
    def process(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Process data.
        
        Args:
            data: Data to process (format depends on processor type)
            **kwargs: Processing parameters:
                - lazy: bool (lazy evaluation)
                - parallel: bool (parallel processing)
                - batch_size: int (batch size)
                - filters: Dict[str, Any] (filtering criteria)
                - transformations: List[Callable] (transformations to apply)
        
        Returns:
            Processed data (format depends on processor type)
        
        Raises:
            InvalidInputError: If data format invalid
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate data.
        
        Args:
            data: Data to validate
        
        Returns:
            True if data is valid, False otherwise
        
        Raises:
            ValidationError: If validation fails with details
        """
        pass
    
    @abstractmethod
    def read(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> Any:
        """
        Read data from file.
        
        Args:
            path: Path to data file
            **kwargs: Reading parameters:
                - format: str (auto-detect if None)
                - lazy: bool (lazy loading)
                - columns: List[str] (columns to read)
                - filters: Dict[str, Any] (pushdown filters)
        
        Returns:
            Data object (lazy or eager depending on parameters)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ReadError: If reading fails
        """
        pass
    
    @abstractmethod
    def write(
        self,
        data: Any,
        path: Union[str, Path],
        **kwargs
    ) -> bool:
        """
        Write data to file.
        
        Args:
            data: Data to write
            path: Output path
            **kwargs: Writing parameters:
                - format: str (output format)
                - compression: str (compression method)
                - partition_by: List[str] (partition columns)
        
        Returns:
            True if write successful
        
        Raises:
            WriteError: If writing fails
        """
        pass
```

**Contrato**:
- `process()` debe soportar lazy evaluation cuando sea posible
- `validate()` debe proporcionar mensajes de error descriptivos
- `read()` y `write()` deben soportar múltiples formatos

## 📊 Tipos de Datos

### GenerationConfig

```python
@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_tokens: int = 100
    """Maximum tokens to generate."""
    
    temperature: float = 0.7
    """Sampling temperature (0.0 = deterministic, >1.0 = more random)."""
    
    top_p: float = 1.0
    """Nucleus sampling parameter (0.0-1.0)."""
    
    top_k: int = -1
    """Top-k sampling (-1 = disabled)."""
    
    repetition_penalty: float = 1.0
    """Repetition penalty (>1.0 = penalize repetition)."""
    
    stop_sequences: Optional[List[str]] = None
    """Stop sequences (generation stops when encountered)."""
    
    seed: Optional[int] = None
    """Random seed for reproducibility."""
    
    stream: bool = False
    """Whether to stream results."""
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValidationError: If configuration invalid
        """
        if self.max_tokens < 1:
            raise ValidationError("max_tokens must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValidationError("temperature must be in [0.0, 2.0]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValidationError("top_p must be in [0.0, 1.0]")
        if self.top_k < -1:
            raise ValidationError("top_k must be >= -1")
        if self.repetition_penalty < 0.0:
            raise ValidationError("repetition_penalty must be >= 0.0")
```

## 🏭 Factories

### ComponentFactory

**Propósito**: Factory genérico para crear componentes.

**Especificación**:

```python
class ComponentFactory(ABC):
    """Generic factory for creating components."""
    
    @abstractmethod
    def register(
        self,
        name: str,
        component_class: Type[IComponent],
        **kwargs
    ) -> None:
        """
        Register a component class.
        
        Args:
            name: Component name
            component_class: Component class (must implement IComponent)
            **kwargs: Default initialization parameters
        """
        pass
    
    @abstractmethod
    def create(
        self,
        name: str,
        **kwargs
    ) -> IComponent:
        """
        Create a component instance.
        
        Args:
            name: Component name (must be registered)
            **kwargs: Initialization parameters (override defaults)
        
        Returns:
            Component instance
        
        Raises:
            ComponentNotFoundError: If component not registered
            InitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    def list_components(self) -> List[str]:
        """
        List all registered component names.
        
        Returns:
            List of component names
        """
        pass
    
    @abstractmethod
    def is_registered(self, name: str) -> bool:
        """
        Check if component is registered.
        
        Args:
            name: Component name
        
        Returns:
            True if registered
        """
        pass
```

## 🔄 Flujos de Uso

### Flujo de Inicialización

```
1. Usuario crea factory
2. Factory registra componentes
3. Usuario llama factory.create(name, **kwargs)
4. Factory crea instancia
5. Factory llama initialize(**kwargs)
6. Factory retorna componente inicializado
```

### Flujo de Uso

```
1. Usuario obtiene componente (via factory o directo)
2. Usuario verifica is_initialized
3. Usuario usa componente (generate, process, etc.)
4. Usuario llama get_status() para monitoreo
5. Usuario llama cleanup() cuando termina
```

## ✅ Validación y Errores

### Jerarquía de Excepciones

```python
class OptimizationCoreError(Exception):
    """Base exception for optimization_core."""
    pass

class InitializationError(OptimizationCoreError):
    """Component initialization failed."""
    pass

class NotInitializedError(OptimizationCoreError):
    """Component not initialized."""
    pass

class InvalidInputError(OptimizationCoreError):
    """Invalid input provided."""
    pass

class ValidationError(OptimizationCoreError):
    """Validation failed."""
    pass

class ModelLoadError(OptimizationCoreError):
    """Model loading failed."""
    pass

class GenerationError(OptimizationCoreError):
    """Text generation failed."""
    pass

class ProcessingError(OptimizationCoreError):
    """Data processing failed."""
    pass

class ComponentNotFoundError(OptimizationCoreError):
    """Component not found in factory."""
    pass
```

## 🧪 Testing

### Mocking

Todas las interfaces deben ser fácilmente mockeables:

```python
from unittest.mock import Mock

# Mock IComponent
mock_component = Mock(spec=IComponent)
mock_component.name = "MockComponent"
mock_component.version = "1.0.0"
mock_component.initialize.return_value = True
mock_component.get_status.return_value = {"ready": True}

# Mock IInferenceEngine
mock_engine = Mock(spec=IInferenceEngine)
mock_engine.generate.return_value = "Generated text"
mock_engine.get_model_info.return_value = {"model_name": "test"}
```

## 📝 Convenciones

### Nombres
- Interfaces: Prefijo `I` (IComponent, IInferenceEngine)
- Implementaciones: Sin prefijo (VLLMEngine, PolarsProcessor)
- Factories: Sufijo `Factory` (EngineFactory, ProcessorFactory)

### Documentación
- Todas las interfaces deben tener docstrings completos
- Todos los métodos deben documentar Args, Returns, Raises
- Ejemplos de uso en docstrings cuando sea apropiado

### Versionado
- Versiones siguen Semantic Versioning (MAJOR.MINOR.PATCH)
- Cambios breaking requieren incremento de MAJOR
- Nuevas features requieren incremento de MINOR
- Bug fixes requieren incremento de PATCH

---

**Versión**: 1.0.0  
**Última actualización**: Enero 2025




