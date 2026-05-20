# Refactored Class Structure - Complete Documentation

## 📋 Overview

This document provides a comprehensive summary of the refactored class structure for Audio Separation Core, including detailed before/after code comparisons, class responsibilities, and explanations for each change.

---

## ✅ Current Refactoring Status

### Already Implemented ✅
1. **BaseSeparator** - Now inherits from `BaseComponent` (eliminates lifecycle duplication)
2. **BaseMixer** - Now inherits from `BaseComponent` (eliminates lifecycle duplication)

### Still Needs Refactoring ⚠️
1. **Factories** - Still have duplicated code (~200 lines)
2. **Config Validation** - Could be separated into validators

---

## 📦 Complete Refactored Class Structure

### 1. Core Layer - Base Components

#### BaseComponent (`core/base_component.py`)

**Purpose**: Provides shared lifecycle management for all audio components.

**Responsibilities**:
- Component initialization and cleanup
- State tracking (initialized, ready, health)
- Status reporting
- Error handling

**Class Structure**:
```python
class BaseComponent(ABC):
    """
    Componente base simplificado.
    
    Single Responsibility: Gestión del ciclo de vida de componentes.
    """
    
    # State Properties
    @property
    def name(self) -> str
    @property
    def version(self) -> str
    @property
    def is_initialized(self) -> bool
    @property
    def is_ready(self) -> bool
    
    # Lifecycle Methods
    def initialize() -> bool
    def cleanup() -> None
    def get_status() -> Dict[str, Any]
    def _ensure_ready() -> None
    
    # Abstract Methods (for subclasses)
    @abstractmethod
    def _do_initialize() -> None
    def _do_cleanup() -> None  # Optional override
```

**Key Design Decisions**:
- ✅ Uses Template Method pattern (`initialize()` calls `_do_initialize()`)
- ✅ Idempotent operations (safe to call multiple times)
- ✅ Clear separation between base lifecycle and component-specific logic

---

### 2. Implementation Layer - Base Classes

#### BaseSeparator (`separators/base_separator.py`)

**Purpose**: Base class for audio separators.

**Responsibilities**:
- Format validation
- Component validation
- Separation orchestration
- Model lifecycle management

**Refactoring: BEFORE → AFTER**

##### ❌ BEFORE: Duplicated Lifecycle Code

```python
class BaseSeparator(IAudioSeparator):
    """❌ Duplicated lifecycle management"""
    
    def __init__(self, config: Optional[SeparationConfig] = None, **kwargs):
        self._config = config or SeparationConfig()
        self._config.validate()
        
        # ❌ Duplicated state management
        self._initialized = False
        self._ready = False
        self._start_time = None
        self._last_error = None
        self._model = None
    
    def initialize(self, **kwargs) -> bool:
        """❌ ~20 lines of duplicated initialization logic"""
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
        """❌ ~10 lines of duplicated cleanup logic"""
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
        """❌ ~15 lines of duplicated status logic"""
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

**Problems**:
- ❌ ~50 lines of lifecycle code duplicated from BaseMixer
- ❌ State management mixed with business logic
- ❌ Hard to maintain (changes need to be made in multiple places)

##### ✅ AFTER: Using BaseComponent

```python
class BaseSeparator(BaseComponent, IAudioSeparator):
    """
    Clase base abstracta para separadores de audio.
    
    Single Responsibility: Proporcionar funcionalidad común para separadores.
    Lifecycle management delegated to BaseComponent.
    """
    
    def __init__(self, config: Optional[SeparationConfig] = None, **kwargs):
        # ✅ Initialize BaseComponent first
        super().__init__()
        
        self._config = config or SeparationConfig()
        self._config.validate()
        self._model = None  # ✅ Only separator-specific state
    
    @property
    def config(self) -> SeparationConfig:
        """Configuración del separador."""
        return self._config
    
    def initialize(self, **kwargs) -> bool:
        """
        Inicializa el separador.
        
        ✅ Wraps BaseComponent.initialize() with separator-specific error handling.
        """
        try:
            return super().initialize(**kwargs)
        except Exception as e:
            raise AudioSeparationError(
                f"Failed to initialize {self.name}: {e}",
                component=self.name
            ) from e
    
    def _do_initialize(self, **kwargs) -> None:
        """
        ✅ Separator-specific initialization.
        Called by BaseComponent.initialize()
        """
        self._model = self._load_model(**kwargs)
    
    def _do_cleanup(self) -> None:
        """
        ✅ Separator-specific cleanup.
        Called by BaseComponent.cleanup()
        """
        if self._model is not None:
            try:
                self._cleanup_model()
            except Exception:
                pass
            finally:
                self._model = None
    
    def separate(self, input_path, output_dir=None, components=None, **kwargs):
        """
        Separa un archivo de audio en componentes.
        
        ✅ Uses _ensure_ready() from BaseComponent.
        """
        self._ensure_ready()  # ✅ Reusable method from BaseComponent
        
        # ... separation-specific logic ...
    
    # Abstract methods remain the same
    @abstractmethod
    def _load_model(self, **kwargs):
        pass
    
    @abstractmethod
    def _cleanup_model(self) -> None:
        pass
    
    @abstractmethod
    def _perform_separation(self, input_path, output_dir, components, **kwargs):
        pass
```

**Benefits**:
- ✅ Eliminated ~50 lines of duplicated code
- ✅ Single source of truth for lifecycle management
- ✅ Clear separation: lifecycle (BaseComponent) vs business logic (BaseSeparator)
- ✅ Easier to maintain and test

**Methods Summary**:
```python
class BaseSeparator(BaseComponent, IAudioSeparator):
    # Inherited from BaseComponent:
    # - initialize(), cleanup(), get_status(), _ensure_ready()
    
    # Properties
    @property
    def config(self) -> SeparationConfig
    
    # Public Methods
    def separate(...) -> Dict[str, str]
    def get_supported_components() -> List[str]
    def get_supported_formats() -> List[str]
    def estimate_separation_time(...) -> float
    
    # Protected Methods (override from BaseComponent)
    def _do_initialize(**kwargs) -> None
    def _do_cleanup() -> None
    
    # Abstract Methods (must implement in subclasses)
    @abstractmethod
    def _load_model(**kwargs)
    @abstractmethod
    def _cleanup_model() -> None
    @abstractmethod
    def _perform_separation(...) -> Dict[str, str]
    @abstractmethod
    def _get_supported_components() -> List[str]
```

---

#### BaseMixer (`mixers/base_mixer.py`)

**Purpose**: Base class for audio mixers.

**Responsibilities**:
- File validation
- Volume validation
- Mixing orchestration
- Effect application

**Refactoring: BEFORE → AFTER**

##### ❌ BEFORE: Duplicated Lifecycle Code

```python
class BaseMixer(IAudioMixer):
    """❌ Duplicated lifecycle management"""
    
    def __init__(self, config: Optional[MixingConfig] = None, **kwargs):
        self._config = config or MixingConfig()
        self._config.validate()
        
        # ❌ Same duplicated state as BaseSeparator
        self._initialized = False
        self._ready = False
        self._start_time = None
        self._last_error = None
    
    def initialize(self, **kwargs) -> bool:
        # ❌ Same duplicated logic as BaseSeparator
        ...
    
    def cleanup(self) -> None:
        # ❌ Same duplicated logic as BaseSeparator
        ...
    
    def get_status(self) -> Dict:
        # ❌ Same duplicated logic as BaseSeparator
        ...
```

##### ✅ AFTER: Using BaseComponent

```python
class BaseMixer(BaseComponent, IAudioMixer):
    """
    Clase base abstracta para mezcladores de audio.
    
    Single Responsibility: Proporcionar funcionalidad común para mezcladores.
    Lifecycle management delegated to BaseComponent.
    """
    
    def __init__(self, config: Optional[MixingConfig] = None, **kwargs):
        # ✅ Initialize BaseComponent first
        super().__init__()
        
        self._config = config or MixingConfig()
        if config:
            self._config.validate()
        # ✅ No additional state needed for simple mixers
    
    @property
    def config(self) -> MixingConfig:
        """Configuración del mezclador."""
        return self._config
    
    # ✅ No _do_initialize needed (no model to load)
    # ✅ No _do_cleanup needed (no resources to clean)
    # ✅ All lifecycle methods inherited from BaseComponent
    
    def mix(self, audio_files, output_path, volumes=None, effects=None, **kwargs):
        """
        Mezcla múltiples archivos de audio.
        
        ✅ Uses _ensure_ready() from BaseComponent.
        """
        self._ensure_ready()  # ✅ Reusable method
        
        # ... mixing-specific logic ...
```

**Benefits**:
- ✅ Eliminated ~50 lines of duplicated code
- ✅ Consistent lifecycle pattern with BaseSeparator
- ✅ Simpler: less code to maintain

**Methods Summary**:
```python
class BaseMixer(BaseComponent, IAudioMixer):
    # Inherited from BaseComponent:
    # - initialize(), cleanup(), get_status(), _ensure_ready()
    
    # Properties
    @property
    def config(self) -> MixingConfig
    
    # Public Methods
    def mix(...) -> str
    def get_supported_formats() -> List[str]
    def apply_effect(...) -> str
    
    # Abstract Methods (must implement in subclasses)
    @abstractmethod
    def _perform_mixing(...) -> str
    @abstractmethod
    def _apply_effect(...) -> str
```

---

### 3. Factory Pattern - Needs Refactoring ⚠️

#### Current State: Duplicated Factory Code

**Problem**: All three factories have nearly identical structure (~200 lines of duplicate code).

##### ❌ BEFORE: AudioSeparatorFactory (Current Implementation)

```python
class AudioSeparatorFactory:
    """
    ❌ Multiple responsibilities:
    1. Registration
    2. Dynamic loading
    3. Auto-detection
    4. Instance creation
    """
    
    _separators: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, separator_class: type) -> None:
        """❌ Registration logic"""
        if not issubclass(separator_class, IAudioSeparator):
            raise TypeError(...)
        cls._separators[name.lower()] = separator_class
    
    @classmethod
    def create(cls, separator_type: str = "auto", config=None, **kwargs):
        """❌ Multiple responsibilities mixed together"""
        separator_type = separator_type.lower()
        
        # ❌ Auto-detection logic
        if separator_type == "auto":
            separator_type = cls._detect_best_separator()
        
        # ❌ Dynamic import logic
        if separator_type not in cls._separators:
            try:
                if separator_type == "spleeter":
                    from ..separators.spleeter_separator import SpleeterSeparator
                    cls.register("spleeter", SpleeterSeparator)
                elif separator_type == "demucs":
                    from ..separators.demucs_separator import DemucsSeparator
                    cls.register("demucs", DemucsSeparator)
                # ... more imports
            except ImportError as e:
                raise AudioConfigurationError(...) from e
        
        # ❌ Creation logic
        separator_class = cls._separators[separator_type]
        if config is None:
            config = SeparationConfig(model_type=separator_type)
        
        try:
            return separator_class(config=config, **kwargs)
        except Exception as e:
            raise AudioSeparationError(...) from e
    
    @classmethod
    def _detect_best_separator(cls) -> str:
        """❌ Detection logic mixed in"""
        for preferred in ["demucs", "spleeter", "lalal"]:
            try:
                if preferred == "demucs":
                    import demucs
                    return "demucs"
                # ... more detection logic
            except ImportError:
                continue
        return "spleeter"
```

**Problems**:
- ❌ ~140 lines with multiple responsibilities
- ❌ Same pattern duplicated in AudioMixerFactory and AudioProcessorFactory
- ❌ Hard to test (too many concerns)
- ❌ Hard to extend (need to duplicate pattern)

##### ✅ AFTER: Refactored Factory (Proposed)

```python
# core/registry.py (NEW)
class ComponentRegistry:
    """
    Generic registry for component classes.
    
    Single Responsibility: Register and retrieve component classes.
    """
    
    def __init__(self):
        self._components: Dict[str, Type[T]] = {}
    
    def register(self, name: str, component_class: Type[T]) -> None:
        """Register a component class."""
        self._components[name.lower()] = component_class
    
    def get(self, name: str) -> Type[T]:
        """Get a registered component class."""
        if not self.is_registered(name):
            raise KeyError(f"Component '{name}' not registered")
        return self._components[name.lower()]
    
    def is_registered(self, name: str) -> bool:
        """Check if a component is registered."""
        return name.lower() in self._components
    
    def list_registered(self) -> List[str]:
        """List all registered components."""
        return list(self._components.keys())


# core/loader.py (NEW)
class ComponentLoader:
    """
    Load components dynamically from modules.
    
    Single Responsibility: Import and load component classes.
    """
    
    SEPARATOR_MAP = {
        "spleeter": ("..separators.spleeter_separator", "SpleeterSeparator"),
        "demucs": ("..separators.demucs_separator", "DemucsSeparator"),
        "lalal": ("..separators.lalal_separator", "LALALSeparator"),
    }
    
    MIXER_MAP = {
        "simple": ("..mixers.simple_mixer", "SimpleMixer"),
        "advanced": ("..mixers.advanced_mixer", "AdvancedMixer"),
    }
    
    @classmethod
    def load_separator(cls, separator_type: str) -> Type:
        """Load a separator class dynamically."""
        return cls._load_component(separator_type, cls.SEPARATOR_MAP)
    
    @classmethod
    def load_mixer(cls, mixer_type: str) -> Type:
        """Load a mixer class dynamically."""
        return cls._load_component(mixer_type, cls.MIXER_MAP)
    
    @classmethod
    def _load_component(cls, component_type: str, component_map: Dict) -> Type:
        """Generic component loading logic."""
        component_type = component_type.lower()
        
        if component_type not in component_map:
            raise AudioConfigurationError(
                f"Unknown component type: {component_type}"
            )
        
        module_path, class_name = component_map[component_type]
        
        try:
            module = __import__(module_path, fromlist=[class_name], level=1)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise AudioConfigurationError(
                f"Failed to load {component_type}: {e}"
            ) from e


# core/detector.py (NEW)
class SeparatorDetector:
    """
    Detect available separators in the system.
    
    Single Responsibility: Determine which separators can be used.
    """
    
    PRIORITY = ["demucs", "spleeter", "lalal"]
    
    @classmethod
    def detect_best(cls) -> str:
        """Detect the best available separator."""
        for separator_type in cls.PRIORITY:
            if cls.is_available(separator_type):
                return separator_type
        return "spleeter"  # Fallback
    
    @classmethod
    def is_available(cls, separator_type: str) -> bool:
        """Check if a separator is available."""
        try:
            if separator_type == "demucs":
                import demucs
                return True
            elif separator_type == "spleeter":
                import spleeter
                return True
            elif separator_type == "lalal":
                return True  # API-based
        except ImportError:
            return False
        return False
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available separators."""
        return [s for s in cls.PRIORITY if cls.is_available(s)]


# core/factories.py (REFACTORED)
class AudioSeparatorFactory:
    """
    Factory for creating audio separators.
    
    Single Responsibility: Create separator instances using helpers.
    """
    
    _registry = ComponentRegistry()
    _loader = ComponentLoader()
    _detector = SeparatorDetector()
    
    @classmethod
    def register(cls, name: str, separator_class: type) -> None:
        """Register a separator class."""
        if not issubclass(separator_class, IAudioSeparator):
            raise TypeError(f"{separator_class} must implement IAudioSeparator")
        cls._registry.register(name, separator_class)
    
    @classmethod
    def create(
        cls,
        separator_type: str = "auto",
        config: Optional[SeparationConfig] = None,
        **kwargs
    ) -> IAudioSeparator:
        """
        Create a separator instance.
        
        ✅ Single responsibility: orchestrate creation using helpers.
        """
        # ✅ Delegate detection
        if separator_type == "auto":
            separator_type = cls._detector.detect_best()
        
        # ✅ Delegate loading
        if not cls._registry.is_registered(separator_type):
            separator_class = cls._loader.load_separator(separator_type)
            cls.register(separator_type, separator_class)
        else:
            separator_class = cls._registry.get(separator_type)
        
        # ✅ Create instance
        if config is None:
            config = SeparationConfig(model_type=separator_type)
        
        try:
            return separator_class(config=config, **kwargs)
        except Exception as e:
            raise AudioSeparationError(
                f"Failed to create separator '{separator_type}': {e}",
                component="AudioSeparatorFactory"
            ) from e
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available separators."""
        return cls._detector.list_available()
```

**Benefits**:
- ✅ Eliminated ~200 lines of duplicated code
- ✅ Each class has a single responsibility
- ✅ Easy to test each component independently
- ✅ Easy to extend (add new component types)
- ✅ Reusable components (Registry, Loader, Detector)

---

## 📊 Summary of Refactored Structure

### Class Hierarchy

```
IAudioComponent (interface)
    └── BaseComponent (implementation)
        ├── BaseSeparator (inherits BaseComponent, implements IAudioSeparator)
        │   ├── SpleeterSeparator
        │   ├── DemucsSeparator
        │   └── LALALSeparator
        └── BaseMixer (inherits BaseComponent, implements IAudioMixer)
            ├── SimpleMixer
            └── AdvancedMixer

ComponentRegistry (reusable)
    └── Used by all factories

ComponentLoader (reusable)
    └── Used by all factories

SeparatorDetector (specialized)
    └── Used by AudioSeparatorFactory

AudioSeparatorFactory (simplified)
    └── Uses Registry, Loader, Detector
```

### Responsibilities Summary

| Class | Responsibility | Status |
|-------|---------------|--------|
| `BaseComponent` | Lifecycle management | ✅ Implemented |
| `BaseSeparator` | Separation-specific logic | ✅ Refactored |
| `BaseMixer` | Mixing-specific logic | ✅ Refactored |
| `ComponentRegistry` | Component registration | ⚠️ Proposed |
| `ComponentLoader` | Dynamic loading | ⚠️ Proposed |
| `SeparatorDetector` | Auto-detection | ⚠️ Proposed |
| `AudioSeparatorFactory` | Create separators | ⚠️ Needs refactoring |
| `AudioMixerFactory` | Create mixers | ⚠️ Needs refactoring |
| `AudioProcessorFactory` | Create processors | ⚠️ Needs refactoring |

### Code Metrics

| Metric | Before | After (Current) | After (Proposed) |
|--------|--------|-----------------|------------------|
| **Duplicated Lifecycle Code** | ~100 lines | 0 lines | 0 lines |
| **Duplicated Factory Code** | ~200 lines | ~200 lines | 0 lines |
| **Total Duplication** | ~300 lines | ~200 lines | 0 lines |
| **Classes** | 8 | 8 | 12 (but simpler) |
| **Avg Lines/Class** | ~100 | ~75 | ~42 |

---

## 🎯 Key Refactoring Principles Applied

### 1. Single Responsibility Principle (SRP)

**Before**: Classes had 2-4 responsibilities  
**After**: Each class has exactly 1 responsibility

**Examples**:
- `BaseSeparator`: Was managing lifecycle AND separation → Now only separation
- `AudioSeparatorFactory`: Was registering AND loading AND detecting AND creating → Should only create

### 2. DRY (Don't Repeat Yourself)

**Before**: ~300 lines of duplicated code  
**After (Current)**: ~200 lines (lifecycle fixed, factories still duplicated)  
**After (Proposed)**: 0 lines

**Examples**:
- Lifecycle code duplicated in `BaseSeparator` and `BaseMixer` → ✅ Extracted to `BaseComponent`
- Factory pattern duplicated 3 times → ⚠️ Should be extracted to reusable components

### 3. Separation of Concerns

**Before**: Concerns mixed together  
**After**: Each concern in its own class

**Examples**:
- Lifecycle management → ✅ `BaseComponent`
- Registration → ⚠️ Should be `ComponentRegistry`
- Dynamic loading → ⚠️ Should be `ComponentLoader`
- Detection → ⚠️ Should be `SeparatorDetector`
- Validation → Could be validators

### 4. Improved Naming

**Before**: Inconsistent naming  
**After**: Consistent, descriptive names

**Examples**:
- `_get_metrics()` → `_get_separator_metrics()` (more specific)
- `initialize()` → `_do_initialize()` (clear pattern for overrides)

---

## ✅ Benefits Summary

### Maintainability
- ✅ Single source of truth for lifecycle management
- ✅ Changes in one place affect all components
- ✅ Easier to understand and modify

### Testability
- ✅ Each component can be tested independently
- ✅ Mock dependencies easily
- ✅ Clear boundaries between components

### Extensibility
- ✅ Easy to add new components (inherit from `BaseComponent`)
- ⚠️ Easy to add new factories (use `ComponentRegistry` and `ComponentLoader`) - Proposed
- ⚠️ Easy to add new validators (extend base validator) - Proposed

### Readability
- ✅ Less code to read
- ✅ Clearer intent
- ✅ Better organization

---

## 📝 Migration Notes

### Already Completed ✅
- `BaseSeparator` now uses `BaseComponent`
- `BaseMixer` now uses `BaseComponent`
- Lifecycle duplication eliminated

### Still To Do ⚠️
- Refactor factories to use `ComponentRegistry`, `ComponentLoader`, `SeparatorDetector`
- Separate config validation into validators (optional improvement)

### Backward Compatibility

✅ **All public APIs remain unchanged**

- `BaseSeparator` and `BaseMixer` APIs unchanged
- Factory APIs unchanged
- Config APIs unchanged

### Internal Changes Only

- Lifecycle management now uses `BaseComponent`
- Factories would use shared components internally (proposed)
- Validation would use external validators (proposed)

---

## ✅ Conclusion

### Completed Refactoring ✅
1. **BaseSeparator** - Now inherits from `BaseComponent` (eliminates ~50 lines)
2. **BaseMixer** - Now inherits from `BaseComponent` (eliminates ~50 lines)

### Proposed Refactoring ⚠️
1. **Factories** - Extract to `ComponentRegistry`, `ComponentLoader`, `SeparatorDetector` (would eliminate ~200 lines)
2. **Config Validation** - Extract to validators (optional improvement)

### Overall Impact
- **Current**: ~100 lines of duplication eliminated
- **Proposed**: Additional ~200 lines could be eliminated
- **Total Potential**: ~300 lines of duplication eliminated

The refactored structure follows SOLID principles, eliminates code duplication, and improves maintainability without adding unnecessary complexity.

