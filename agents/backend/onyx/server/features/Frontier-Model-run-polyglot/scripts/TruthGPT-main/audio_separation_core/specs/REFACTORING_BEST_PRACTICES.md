# Refactoring Best Practices Guide

## 📋 Overview

This document provides best practices, patterns, and guidelines for working with the refactored Audio Separation Core architecture.

---

## 🎯 Core Principles

### 1. Single Responsibility Principle (SRP)

**Rule**: Each class should have one reason to change.

**✅ Good Example**:
```python
class ComponentRegistry:
    """✅ Single responsibility: Registration only"""
    def register(self, name: str, component_class: Type) -> None:
        ...
    def get(self, name: str) -> Type:
        ...
```

**❌ Bad Example**:
```python
class ComponentRegistry:
    """❌ Multiple responsibilities: Registration + Loading + Validation"""
    def register(self, name: str, component_class: Type) -> None:
        ...
    def load_from_module(self, module_path: str) -> Type:
        ...
    def validate_component(self, component: Type) -> bool:
        ...
```

---

### 2. DRY (Don't Repeat Yourself)

**Rule**: Every piece of knowledge should have a single, unambiguous representation.

**✅ Good Example**:
```python
# ✅ Single implementation in BaseComponent
class BaseComponent(ABC):
    def initialize(self) -> bool:
        # Single source of truth
        ...

# ✅ All components inherit it
class BaseSeparator(BaseComponent):
    # No duplicate initialization code
    def _do_initialize(self):
        # Only component-specific logic
        ...
```

**❌ Bad Example**:
```python
# ❌ Duplicated in multiple classes
class BaseSeparator:
    def initialize(self) -> bool:
        # Duplicate code
        ...

class BaseMixer:
    def initialize(self) -> bool:
        # Same duplicate code
        ...
```

---

### 3. Separation of Concerns

**Rule**: Each concern should be handled by a separate, focused component.

**✅ Good Example**:
```python
# ✅ Separate concerns
class ComponentRegistry:
    """Only handles registration"""
    ...

class ComponentLoader:
    """Only handles dynamic loading"""
    ...

class AudioSeparatorFactory:
    """Only orchestrates creation"""
    def create(...):
        # Uses helpers, doesn't implement their logic
        registry = ComponentRegistry()
        loader = ComponentLoader()
        ...
```

**❌ Bad Example**:
```python
# ❌ Mixed concerns
class AudioSeparatorFactory:
    """Handles registration, loading, detection, and creation"""
    def register(...):  # Registration concern
        ...
    def _load_dynamically(...):  # Loading concern
        ...
    def _detect_best(...):  # Detection concern
        ...
    def create(...):  # Creation concern
        ...
```

---

## 🏗️ Architecture Patterns

### Pattern 1: Template Method Pattern

**Used in**: `BaseComponent`

**Purpose**: Define skeleton of algorithm, let subclasses fill in details.

**Example**:
```python
class BaseComponent(ABC):
    def initialize(self) -> bool:
        """Template method - defines algorithm structure"""
        if self._initialized:
            return True
        try:
            self._start_time = time.time()
            self._do_initialize()  # ✅ Subclass implements this
            self._initialized = True
            self._ready = True
            return True
        except Exception as e:
            self._last_error = str(e)
            raise
    
    @abstractmethod
    def _do_initialize(self) -> None:
        """Subclass implements specific initialization"""
        pass

class BaseSeparator(BaseComponent):
    def _do_initialize(self) -> None:
        """✅ Only separator-specific initialization"""
        self._model = self._load_model()
```

**Benefits**:
- ✅ Common algorithm structure in base class
- ✅ Specific steps implemented by subclasses
- ✅ No code duplication

---

### Pattern 2: Composition Over Inheritance

**Used in**: Factories using helpers

**Purpose**: Favor composition over inheritance for flexibility.

**Example**:
```python
class AudioSeparatorFactory:
    """✅ Uses composition - has helpers"""
    _registry = ComponentRegistry()  # ✅ Composed
    _loader = ComponentLoader()      # ✅ Composed
    _detector = SeparatorDetector()  # ✅ Composed
    
    @classmethod
    def create(cls, separator_type: str = "auto", ...):
        # ✅ Delegates to composed helpers
        if separator_type == "auto":
            separator_type = cls._detector.detect_best()
        
        if not cls._registry.is_registered(separator_type):
            separator_class = cls._loader.load_separator(separator_type)
            cls._registry.register(separator_type, separator_class)
        ...
```

**Benefits**:
- ✅ More flexible than inheritance
- ✅ Easy to swap implementations
- ✅ Easy to test (mock helpers)

---

### Pattern 3: Factory Pattern

**Used in**: All factory classes

**Purpose**: Create objects without specifying exact class.

**Example**:
```python
class AudioSeparatorFactory:
    @classmethod
    def create(cls, separator_type: str = "auto", ...) -> IAudioSeparator:
        """✅ Factory method - creates without knowing exact class"""
        separator_type = cls._detector.detect_best() if separator_type == "auto" else separator_type
        separator_class = cls._loader.load_separator(separator_type)
        return separator_class(config=config, **kwargs)  # ✅ Returns interface
```

**Benefits**:
- ✅ Decouples creation from usage
- ✅ Easy to add new types
- ✅ Centralized creation logic

---

## 📝 Coding Guidelines

### Naming Conventions

#### Classes

**✅ Good**:
```python
class ComponentRegistry:  # ✅ Clear, descriptive
class SeparatorDetector:  # ✅ Noun, specific
class BaseSeparator:      # ✅ Base prefix for base classes
```

**❌ Bad**:
```python
class Registry:           # ❌ Too generic
class Detector:           # ❌ Too generic
class Separator:          # ❌ Not clear it's a base class
```

#### Methods

**✅ Good**:
```python
def register(name: str, component_class: Type) -> None:  # ✅ Verb, clear action
def is_registered(name: str) -> bool:                    # ✅ Boolean question
def get_status() -> Dict[str, Any]:                      # ✅ Clear return
def _do_initialize() -> None:                           # ✅ Private, template method
```

**❌ Bad**:
```python
def reg(name: str, cls: Type) -> None:  # ❌ Abbreviated
def registered(name: str) -> bool:      # ❌ Not a question
def status() -> Dict:                    # ❌ Ambiguous
def init() -> None:                      # ❌ Abbreviated, not clear it's template
```

#### Private Methods

**✅ Good**:
```python
def _do_initialize() -> None:      # ✅ Template method pattern
def _do_cleanup() -> None:         # ✅ Template method pattern
def _load_model() -> Model:         # ✅ Component-specific helper
def _validate_input() -> Path:      # ✅ Validation helper
```

**❌ Bad**:
```python
def _init() -> None:                # ❌ Abbreviated
def _clean() -> None:               # ❌ Abbreviated
def _load() -> Model:               # ❌ Too generic
def _validate() -> Path:            # ❌ Too generic
```

---

### Method Organization

**✅ Good Order**:
```python
class BaseSeparator(BaseComponent, IAudioSeparator):
    # 1. Constructor
    def __init__(self, config=None, **kwargs):
        ...
    
    # 2. Properties
    @property
    def config(self) -> SeparationConfig:
        ...
    
    # 3. Public methods (interface)
    def separate(self, ...) -> Dict[str, str]:
        ...
    
    def get_supported_components(self) -> List[str]:
        ...
    
    # 4. Protected methods (template methods)
    def _do_initialize(self, **kwargs) -> None:
        ...
    
    def _do_cleanup(self) -> None:
        ...
    
    # 5. Abstract methods (must implement)
    @abstractmethod
    def _load_model(self, **kwargs):
        ...
    
    @abstractmethod
    def _perform_separation(self, ...):
        ...
    
    # 6. Private helpers
    def _validate_input(self, path: Path) -> Path:
        ...
```

---

### Error Handling

**✅ Good**:
```python
def create(cls, separator_type: str, ...) -> IAudioSeparator:
    try:
        separator_class = cls._loader.load_separator(separator_type)
        return separator_class(config=config, **kwargs)
    except AudioConfigurationError:
        # ✅ Re-raise specific exceptions
        raise
    except Exception as e:
        # ✅ Wrap generic exceptions with context
        raise AudioSeparationError(
            f"Failed to create separator '{separator_type}': {e}",
            component="AudioSeparatorFactory"
        ) from e
```

**❌ Bad**:
```python
def create(cls, separator_type: str, ...) -> IAudioSeparator:
    try:
        separator_class = cls._loader.load_separator(separator_type)
        return separator_class(config=config, **kwargs)
    except Exception as e:
        # ❌ Loses exception type information
        raise Exception(f"Error: {e}")
```

---

### Type Hints

**✅ Good**:
```python
from typing import Dict, List, Optional, Type, TypeVar

T = TypeVar('T')

def register(self, name: str, component_class: Type[T]) -> None:
    """✅ Clear types"""
    ...

def get(self, name: str) -> Type[T]:
    """✅ Returns type"""
    ...

def list_registered(self) -> List[str]:
    """✅ Returns list of strings"""
    ...
```

**❌ Bad**:
```python
def register(self, name, component_class):
    """❌ No type hints"""
    ...

def get(self, name):
    """❌ No type hints"""
    ...

def list_registered(self):
    """❌ No type hints"""
    ...
```

---

## 🔧 Extension Guidelines

### Adding a New Separator

**Step 1**: Create the separator class
```python
# separators/my_separator.py
from ..separators.base_separator import BaseSeparator
from ..core.config import SeparationConfig

class MySeparator(BaseSeparator):
    """✅ Inherit from BaseSeparator"""
    
    def _load_model(self, **kwargs):
        """✅ Implement abstract method"""
        # Load your model
        return MyModel()
    
    def _cleanup_model(self) -> None:
        """✅ Implement abstract method"""
        # Cleanup your model
        pass
    
    def _perform_separation(self, input_path, output_dir, components, **kwargs):
        """✅ Implement abstract method"""
        # Perform separation
        results = {}
        for component in components:
            # ... separation logic ...
            results[component] = str(output_file)
        return results
    
    def _get_supported_components(self) -> List[str]:
        """✅ Implement abstract method"""
        return ["vocals", "drums", "bass", "other"]
```

**Step 2**: Register in ComponentLoader
```python
# core/loader.py
class ComponentLoader:
    SEPARATOR_MAP = {
        "spleeter": ("..separators.spleeter_separator", "SpleeterSeparator"),
        "demucs": ("..separators.demucs_separator", "DemucsSeparator"),
        "lalal": ("..separators.lalal_separator", "LALALSeparator"),
        "my_separator": ("..separators.my_separator", "MySeparator"),  # ✅ Add here
    }
```

**Step 3**: Use it
```python
# ✅ Works automatically
separator = AudioSeparatorFactory.create("my_separator")
```

---

### Adding a New Mixer

**Step 1**: Create the mixer class
```python
# mixers/my_mixer.py
from ..mixers.base_mixer import BaseMixer
from ..core.config import MixingConfig

class MyMixer(BaseMixer):
    """✅ Inherit from BaseMixer"""
    
    def _perform_mixing(self, audio_files, output_path, volumes, effects, **kwargs):
        """✅ Implement abstract method"""
        # Perform mixing
        return str(output_path)
    
    def _apply_effect_impl(self, audio_path, effect_type, effect_params, output_path):
        """✅ Optional: implement if you support effects"""
        # Apply effect
        return str(output_path)
```

**Step 2**: Register in ComponentLoader
```python
# core/loader.py
class ComponentLoader:
    MIXER_MAP = {
        "simple": ("..mixers.simple_mixer", "SimpleMixer"),
        "advanced": ("..mixers.advanced_mixer", "AdvancedMixer"),
        "my_mixer": ("..mixers.my_mixer", "MyMixer"),  # ✅ Add here
    }
```

**Step 3**: Use it
```python
# ✅ Works automatically
mixer = AudioMixerFactory.create("my_mixer")
```

---

### Extending BaseComponent

**✅ Good**: Extend status
```python
class BaseSeparator(BaseComponent):
    def get_status(self) -> Dict[str, Any]:
        """✅ Extend base status"""
        status = super().get_status()  # ✅ Get base status
        status["metrics"].update(self._get_separator_metrics())  # ✅ Add custom
        return status
```

**❌ Bad**: Replace status
```python
class BaseSeparator(BaseComponent):
    def get_status(self) -> Dict[str, Any]:
        """❌ Replaces base status"""
        return {"custom": "status"}  # ❌ Loses base status
```

---

## 🧪 Testing Best Practices

### Testing BaseComponent

```python
def test_base_component_lifecycle():
    """✅ Test lifecycle once in BaseComponent"""
    component = ConcreteComponent()
    
    # Test initialization
    assert not component.is_initialized
    component.initialize()
    assert component.is_initialized
    assert component.is_ready
    
    # Test status
    status = component.get_status()
    assert status["initialized"] is True
    assert status["health"] == "healthy"
    
    # Test cleanup
    component.cleanup()
    assert not component.is_initialized
```

---

### Testing Factories

```python
def test_factory_creation():
    """✅ Test factory with mocked helpers"""
    # Mock helpers
    with patch.object(AudioSeparatorFactory, '_loader') as mock_loader:
        mock_loader.load_separator.return_value = MockSeparator
        
        # Test creation
        separator = AudioSeparatorFactory.create("spleeter")
        
        # Verify
        assert isinstance(separator, MockSeparator)
        mock_loader.load_separator.assert_called_once_with("spleeter")
```

---

### Testing Components

```python
def test_separator():
    """✅ Only test component-specific logic"""
    separator = MockSeparator()
    
    # ✅ Lifecycle already tested in test_base_component_lifecycle()
    # ✅ Only test separation logic
    results = separator.separate("input.wav")
    assert "vocals" in results
    assert "accompaniment" in results
```

---

## 🚫 Anti-Patterns to Avoid

### Anti-Pattern 1: God Class

**❌ Bad**:
```python
class AudioProcessor:
    """❌ Does everything"""
    def process_audio(self): ...
    def validate_input(self): ...
    def load_model(self): ...
    def save_output(self): ...
    def format_audio(self): ...
    def apply_effects(self): ...
    # ... 20+ methods ...
```

**✅ Good**:
```python
# ✅ Separate classes
class AudioValidator:
    def validate_input(self): ...

class ModelLoader:
    def load_model(self): ...

class AudioFormatter:
    def format_audio(self): ...

class AudioProcessor:
    """✅ Only processes audio"""
    def __init__(self):
        self._validator = AudioValidator()
        self._loader = ModelLoader()
        self._formatter = AudioFormatter()
    
    def process_audio(self):
        # Uses composed helpers
        ...
```

---

### Anti-Pattern 2: Copy-Paste Programming

**❌ Bad**:
```python
class Factory1:
    def register(self, name, cls):
        if not issubclass(cls, Interface1):
            raise TypeError(...)
        self._registry[name] = cls

class Factory2:
    def register(self, name, cls):
        if not issubclass(cls, Interface2):
            raise TypeError(...)
        self._registry[name] = cls  # ❌ Duplicated
```

**✅ Good**:
```python
# ✅ Shared registry
class ComponentRegistry:
    def register(self, name, component_class):
        self._components[name] = component_class

class Factory1:
    _registry = ComponentRegistry()
    
    def register(self, name, cls):
        if not issubclass(cls, Interface1):
            raise TypeError(...)
        self._registry.register(name, cls)  # ✅ Uses shared

class Factory2:
    _registry = ComponentRegistry()
    
    def register(self, name, cls):
        if not issubclass(cls, Interface2):
            raise TypeError(...)
        self._registry.register(name, cls)  # ✅ Uses shared
```

---

### Anti-Pattern 3: Premature Optimization

**❌ Bad**:
```python
class ComponentRegistry:
    """❌ Over-engineered for simple use case"""
    def __init__(self):
        self._components: Dict[str, Type] = {}
        self._cache: LRUCache = LRUCache(maxsize=1000)
        self._metrics: MetricsCollector = MetricsCollector()
        self._lock: threading.Lock = threading.Lock()
        # ... unnecessary complexity ...
```

**✅ Good**:
```python
class ComponentRegistry:
    """✅ Simple and clear"""
    def __init__(self):
        self._components: Dict[str, Type] = {}
    
    def register(self, name: str, component_class: Type) -> None:
        self._components[name.lower()] = component_class
```

---

## 📚 Documentation Standards

### Class Docstrings

**✅ Good**:
```python
class ComponentRegistry:
    """
    Generic registry for component classes.
    
    Single Responsibility: Maintain a registry of component classes.
    Used by all factories to eliminate duplicate registration code.
    
    Example:
        registry = ComponentRegistry()
        registry.register("spleeter", SpleeterSeparator)
        separator_class = registry.get("spleeter")
    """
```

**❌ Bad**:
```python
class ComponentRegistry:
    """Registry for components."""  # ❌ Too brief, no context
```

---

### Method Docstrings

**✅ Good**:
```python
def register(self, name: str, component_class: Type[T]) -> None:
    """
    Register a component class.
    
    Args:
        name: Name of the component (case-insensitive)
        component_class: Class to register
        
    Raises:
        TypeError: If component_class is not a class
        
    Example:
        registry.register("spleeter", SpleeterSeparator)
    """
```

**❌ Bad**:
```python
def register(self, name, component_class):
    """Registers a component."""  # ❌ No details
```

---

## ✅ Checklist for New Code

When adding new code, ensure:

- [ ] **Single Responsibility**: Class has one clear purpose
- [ ] **No Duplication**: Code is not duplicated elsewhere
- [ ] **Clear Naming**: Names are descriptive and consistent
- [ ] **Type Hints**: All methods have type hints
- [ ] **Error Handling**: Exceptions are handled appropriately
- [ ] **Documentation**: Classes and methods are documented
- [ ] **Tests**: New code has tests
- [ ] **Backward Compatible**: Changes don't break existing code

---

## 🎓 Key Takeaways

1. **Keep it Simple**: Don't over-engineer
2. **One Responsibility**: Each class does one thing
3. **No Duplication**: Extract shared code
4. **Clear Names**: Code should be self-documenting
5. **Test Everything**: Write tests for new code
6. **Document Well**: Good docs save time later

---

## 📖 Further Reading

- **SOLID Principles**: https://en.wikipedia.org/wiki/SOLID
- **Design Patterns**: "Design Patterns: Elements of Reusable Object-Oriented Software"
- **Clean Code**: "Clean Code: A Handbook of Agile Software Craftsmanship"
- **Refactoring**: "Refactoring: Improving the Design of Existing Code"

---

This guide should be followed when working with or extending the refactored architecture.

