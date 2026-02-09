# Refactoring Code Comparison - Before and After

This document provides detailed before/after code comparisons showing the improvements made during refactoring.

## 1. BaseSeparator Refactoring

### ❌ Before: Duplicated Lifecycle Code

```python
class BaseSeparator(IAudioSeparator):
    def __init__(self, config: Optional[SeparationConfig] = None, **kwargs):
        self._config = config or SeparationConfig()
        self._config.validate()
        
        # ❌ Duplicated lifecycle state
        self._initialized = False
        self._ready = False
        self._start_time = None
        self._last_error = None
        self._model = None
    
    def initialize(self, **kwargs) -> bool:
        # ❌ Duplicated initialization logic
        try:
            if self._initialized:
                return True
            
            self._start_time = time.time()
            self._model = self._load_model(**kwargs)
            self._initialized = True
            self._ready = True
            self._last_error = None
            return True
        except Exception as e:
            self._last_error = str(e)
            self._ready = False
            raise AudioSeparationError(...) from e
    
    def cleanup(self) -> None:
        # ❌ Duplicated cleanup logic
        if self._model is not None:
            try:
                self._cleanup_model()
            except Exception:
                pass
            finally:
                self._model = None
        
        self._initialized = False
        self._ready = False
    
    def get_status(self) -> Dict:
        # ❌ Duplicated status logic
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time
        
        health = "healthy"
        if not self._ready:
            health = "unhealthy"
        elif self._last_error:
            health = "degraded"
        
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "ready": self._ready,
            "health": health,
            "metrics": self._get_metrics(),
            "last_error": self._last_error,
            "uptime_seconds": uptime,
        }
```

### ✅ After: Using BaseComponent

```python
class BaseSeparator(BaseComponent, IAudioSeparator):
    def __init__(self, config: Optional[SeparationConfig] = None, **kwargs):
        # ✅ Initialize BaseComponent first
        super().__init__()
        
        self._config = config or SeparationConfig()
        self._config.validate()
        self._model = None
    
    def _do_initialize(self) -> None:
        # ✅ Only component-specific initialization
        self._model = self._load_model()
    
    def _do_cleanup(self) -> None:
        # ✅ Only component-specific cleanup
        if self._model is not None:
            try:
                self._cleanup_model()
            except Exception:
                pass
            finally:
                self._model = None
    
    def get_status(self) -> Dict:
        # ✅ Reuse BaseComponent status and add metrics
        status = super().get_status()
        status["metrics"] = self._get_metrics()
        return status
    
    # ✅ Lifecycle methods (initialize, cleanup) inherited from BaseComponent
    # ✅ No need to duplicate status calculation
```

**Benefits**:
- **~50 lines removed**: Eliminated duplicate lifecycle code
- **Single source of truth**: Lifecycle logic in `BaseComponent`
- **Easier maintenance**: Changes to lifecycle only in one place

---

## 2. BaseMixer Refactoring

### ❌ Before: Duplicated Lifecycle Code

```python
class BaseMixer(IAudioMixer):
    def __init__(self, config: Optional[MixingConfig] = None, **kwargs):
        self._config = config or MixingConfig()
        self._config.validate()
        
        # ❌ Duplicated lifecycle state
        self._initialized = False
        self._ready = False
        self._start_time = None
        self._last_error = None
    
    def initialize(self, **kwargs) -> bool:
        # ❌ Duplicated initialization logic
        try:
            if self._initialized:
                return True
            
            self._start_time = time.time()
            self._initialized = True
            self._ready = True
            self._last_error = None
            return True
        except Exception as e:
            self._last_error = str(e)
            self._ready = False
            raise AudioProcessingError(...) from e
    
    def cleanup(self) -> None:
        # ❌ Duplicated cleanup logic
        self._initialized = False
        self._ready = False
    
    def get_status(self) -> Dict:
        # ❌ Duplicated status logic
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time
        
        health = "healthy"
        if not self._ready:
            health = "unhealthy"
        elif self._last_error:
            health = "degraded"
        
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "ready": self._ready,
            "health": health,
            "metrics": {},
            "last_error": self._last_error,
            "uptime_seconds": uptime,
        }
```

### ✅ After: Using BaseComponent

```python
class BaseMixer(BaseComponent, IAudioMixer):
    def __init__(self, config: Optional[MixingConfig] = None, **kwargs):
        # ✅ Initialize BaseComponent first
        super().__init__()
        
        self._config = config or MixingConfig()
        self._config.validate()
    
    # ✅ No _do_initialize needed (no model to load)
    # ✅ No _do_cleanup needed (no resources to clean)
    # ✅ All lifecycle methods inherited from BaseComponent
```

**Benefits**:
- **~50 lines removed**: Eliminated duplicate lifecycle code
- **Consistency**: Same lifecycle pattern as `BaseSeparator`
- **Simpler**: Less code to maintain

---

## 3. Factory Pattern Refactoring

### ❌ Before: Duplicated Factory Code

```python
class AudioSeparatorFactory:
    _separators: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, separator_class: type) -> None:
        if not issubclass(separator_class, IAudioSeparator):
            raise TypeError(...)
        cls._separators[name.lower()] = separator_class
    
    @classmethod
    def create(cls, separator_type: str = "auto", config: Optional[SeparationConfig] = None, **kwargs):
        separator_type = separator_type.lower()
        
        if separator_type == "auto":
            separator_type = cls._detect_best_separator()
        
        if separator_type not in cls._separators:
            # ❌ Duplicated dynamic import logic
            try:
                if separator_type == "spleeter":
                    from ..separators.spleeter_separator import SpleeterSeparator
                    cls.register("spleeter", SpleeterSeparator)
                # ... more imports
            except ImportError as e:
                raise AudioConfigurationError(...) from e
        
        separator_class = cls._separators[separator_type]
        
        if config is None:
            config = SeparationConfig(model_type=separator_type)
        
        try:
            return separator_class(config=config, **kwargs)
        except Exception as e:
            raise AudioSeparationError(...) from e

# ❌ Same pattern repeated in AudioMixerFactory and AudioProcessorFactory
```

### ✅ After: Generic BaseFactory

```python
class BaseFactory(ABC):
    """Generic factory with common factory pattern logic."""
    
    _registry: Dict[str, type] = {}
    _interface_type: type = None  # Must be set by subclasses
    
    @classmethod
    def register(cls, name: str, component_class: type) -> None:
        """Register a component class."""
        if cls._interface_type and not issubclass(component_class, cls._interface_type):
            raise TypeError(f"{component_class} must implement {cls._interface_type}")
        cls._registry[name.lower()] = component_class
    
    @classmethod
    def create(cls, component_type: str, config=None, **kwargs):
        """Create a component instance."""
        component_type = component_type.lower()
        
        if component_type not in cls._registry:
            cls._try_import(component_type)
        
        component_class = cls._registry[component_type]
        
        if config is None:
            config = cls._create_default_config(component_type)
        
        try:
            return component_class(config=config, **kwargs)
        except Exception as e:
            raise cls._create_error(component_type, e) from e
    
    @classmethod
    @abstractmethod
    def _try_import(cls, component_type: str) -> None:
        """Try to import component dynamically. Override in subclasses."""
        pass
    
    @classmethod
    @abstractmethod
    def _create_default_config(cls, component_type: str):
        """Create default config. Override in subclasses."""
        pass
    
    @classmethod
    @abstractmethod
    def _create_error(cls, component_type: str, error: Exception):
        """Create appropriate error. Override in subclasses."""
        pass


class AudioSeparatorFactory(BaseFactory):
    """Factory for audio separators."""
    
    _interface_type = IAudioSeparator
    
    @classmethod
    def _try_import(cls, separator_type: str) -> None:
        # ✅ Only separator-specific imports
        if separator_type == "spleeter":
            from ..separators.spleeter_separator import SpleeterSeparator
            cls.register("spleeter", SpleeterSeparator)
        elif separator_type == "demucs":
            from ..separators.demucs_separator import DemucsSeparator
            cls.register("demucs", DemucsSeparator)
        # ...
    
    @classmethod
    def _create_default_config(cls, separator_type: str):
        return SeparationConfig(model_type=separator_type)
    
    @classmethod
    def _create_error(cls, separator_type: str, error: Exception):
        return AudioSeparationError(
            f"Failed to create separator '{separator_type}': {error}",
            component="AudioSeparatorFactory"
        )


class AudioMixerFactory(BaseFactory):
    """Factory for audio mixers."""
    
    _interface_type = IAudioMixer
    
    @classmethod
    def _try_import(cls, mixer_type: str) -> None:
        # ✅ Only mixer-specific imports
        if mixer_type == "simple":
            from ..mixers.simple_mixer import SimpleMixer
            cls.register("simple", SimpleMixer)
        # ...
    
    @classmethod
    def _create_default_config(cls, mixer_type: str):
        return MixingConfig(mixer_type=mixer_type)
    
    @classmethod
    def _create_error(cls, mixer_type: str, error: Exception):
        return AudioSeparationError(
            f"Failed to create mixer '{mixer_type}': {error}",
            component="AudioMixerFactory"
        )
```

**Benefits**:
- **~200 lines removed**: Eliminated duplicate factory code
- **Consistency**: All factories follow the same pattern
- **Easier to extend**: New factories only need to implement 3 abstract methods

---

## 4. Naming Improvements

### Before: Inconsistent Naming

```python
class BaseSeparator:
    def _get_default_components(self) -> List[str]:
        # ❌ "default" is ambiguous
        pass

class BaseMixer:
    def _apply_effect_impl(self, ...):
        # ❌ "_impl" suffix is inconsistent
        pass
```

### After: Consistent Naming

```python
class BaseSeparator:
    def _get_supported_components(self) -> List[str]:
        # ✅ More descriptive name
        pass

class BaseMixer:
    def _apply_effect(self, ...):
        # ✅ Consistent with other _perform_* methods
        pass
```

---

## Summary of Changes

| Component | Lines Removed | Improvement |
|-----------|---------------|-------------|
| `BaseSeparator` | ~50 | Uses `BaseComponent` |
| `BaseMixer` | ~50 | Uses `BaseComponent` |
| `AudioSeparatorFactory` | ~100 | Uses `BaseFactory` |
| `AudioMixerFactory` | ~50 | Uses `BaseFactory` |
| `AudioProcessorFactory` | ~50 | Uses `BaseFactory` |
| **Total** | **~300** | **Significant reduction in duplication** |

### Key Principles Applied

1. ✅ **DRY**: Eliminated ~300 lines of duplicate code
2. ✅ **SRP**: Each class has a single, clear responsibility
3. ✅ **Inheritance**: Proper use of inheritance to share common functionality
4. ✅ **Abstraction**: Abstract base classes for extensibility
5. ✅ **Consistency**: Uniform naming and patterns across codebase

