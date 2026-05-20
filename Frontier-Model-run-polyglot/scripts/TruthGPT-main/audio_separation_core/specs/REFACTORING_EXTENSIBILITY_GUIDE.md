# Extensibility Guide - Extending the Refactored Architecture

## 📋 Overview

This guide provides detailed instructions for extending the refactored Audio Separation Core architecture. It covers adding new components, extending existing functionality, and creating custom implementations.

---

## 🎯 Extension Points

### 1. Adding New Separators

#### Step-by-Step Guide

**Step 1: Create Separator Class**

```python
# separators/my_custom_separator.py
from pathlib import Path
from typing import Dict, List
from ..separators.base_separator import BaseSeparator
from ..core.config import SeparationConfig
from ..core.exceptions import AudioSeparationError

class MyCustomSeparator(BaseSeparator):
    """
    Custom separator implementation.
    
    Inherits from BaseSeparator to get:
    - Lifecycle management (initialize, cleanup, get_status)
    - Common validation logic
    - Error handling
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize custom separator."""
        super().__init__(config, **kwargs)
        # Add any custom initialization here
        self._custom_field = None
    
    def _load_model(self, **kwargs):
        """
        Load your custom model.
        
        This is called automatically by BaseComponent.initialize()
        """
        # Load your model here
        # Return the model object
        return MyCustomModel()
    
    def _cleanup_model(self) -> None:
        """
        Cleanup your custom model.
        
        This is called automatically by BaseComponent.cleanup()
        """
        # Cleanup your model here
        if self._model is not None:
            self._model.cleanup()
    
    def _perform_separation(
        self,
        input_path: Path,
        output_dir: Path,
        components: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """
        Perform the actual separation.
        
        This is called by BaseSeparator.separate() after validation.
        """
        results = {}
        
        for component in components:
            # Perform separation for this component
            output_file = output_dir / f"{component}.wav"
            
            # Your separation logic here
            self._model.separate_component(
                input_path,
                output_file,
                component
            )
            
            results[component] = str(output_file)
        
        return results
    
    def _get_supported_components(self) -> List[str]:
        """
        Return list of components this separator supports.
        
        Examples: ["vocals", "drums", "bass", "other"]
        """
        return ["vocals", "accompaniment"]
    
    def get_supported_formats(self) -> List[str]:
        """
        Override if you support different formats.
        
        Default: [".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"]
        """
        return [".wav", ".mp3"]  # Your custom formats
    
    def estimate_separation_time(
        self,
        input_path: Path,
        components: List[str]
    ) -> float:
        """
        Override if you have custom estimation logic.
        
        Default: Uses file size and component count.
        """
        # Your custom estimation logic
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        return file_size_mb * 0.5  # Example: 0.5 seconds per MB
```

**Step 2: Register in ComponentLoader**

```python
# core/loader.py
class ComponentLoader:
    SEPARATOR_MAP = {
        "spleeter": ("..separators.spleeter_separator", "SpleeterSeparator"),
        "demucs": ("..separators.demucs_separator", "DemucsSeparator"),
        "lalal": ("..separators.lalal_separator", "LALALSeparator"),
        "my_custom": ("..separators.my_custom_separator", "MyCustomSeparator"),  # ✅ Add here
    }
```

**Step 3: Use Your Separator**

```python
# ✅ Works automatically
from audio_separation_core.core import AudioSeparatorFactory

# Create your custom separator
separator = AudioSeparatorFactory.create("my_custom")

# Use it
results = separator.separate("input.wav", components=["vocals", "accompaniment"])
```

---

### 2. Adding New Mixers

#### Step-by-Step Guide

**Step 1: Create Mixer Class**

```python
# mixers/my_custom_mixer.py
from pathlib import Path
from typing import Dict, Optional, Any
from ..mixers.base_mixer import BaseMixer
from ..core.config import MixingConfig

class MyCustomMixer(BaseMixer):
    """
    Custom mixer implementation.
    
    Inherits from BaseMixer to get:
    - Lifecycle management
    - Common validation
    - Volume normalization
    - Effect application framework
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize custom mixer."""
        super().__init__(config, **kwargs)
        # Add any custom initialization here
    
    def _perform_mixing(
        self,
        audio_files: Dict[str, Path],
        output_path: Path,
        volumes: Dict[str, float],
        effects: Optional[Dict[str, Dict[str, Any]]],
        **kwargs
    ) -> str:
        """
        Perform the actual mixing.
        
        This is called by BaseMixer.mix() after validation.
        """
        # Your mixing logic here
        import librosa
        import soundfile as sf
        import numpy as np
        
        mixed_audio = None
        
        for name, file_path in audio_files.items():
            audio, sr = librosa.load(str(file_path), sr=None)
            volume = volumes.get(name, 1.0)
            audio = audio * volume
            
            if mixed_audio is None:
                mixed_audio = audio
            else:
                # Mix audio (ensure same length)
                min_len = min(len(mixed_audio), len(audio))
                mixed_audio[:min_len] += audio[:min_len]
        
        # Normalize if needed
        if self._config.normalize_output:
            max_val = np.abs(mixed_audio).max()
            if max_val > 0:
                mixed_audio = mixed_audio / max_val
        
        # Save output
        sf.write(str(output_path), mixed_audio, sr)
        
        return str(output_path)
    
    def _apply_effect_impl(
        self,
        audio_path: Path,
        effect_type: str,
        effect_params: Dict[str, Any],
        output_path: Path
    ) -> str:
        """
        Apply effect to audio file.
        
        Optional: Override if you support effects.
        """
        if effect_type == "reverb":
            # Apply reverb
            ...
        elif effect_type == "eq":
            # Apply EQ
            ...
        else:
            raise NotImplementedError(f"Effect '{effect_type}' not supported")
        
        return str(output_path)
```

**Step 2: Register in ComponentLoader**

```python
# core/loader.py
class ComponentLoader:
    MIXER_MAP = {
        "simple": ("..mixers.simple_mixer", "SimpleMixer"),
        "advanced": ("..mixers.advanced_mixer", "AdvancedMixer"),
        "my_custom": ("..mixers.my_custom_mixer", "MyCustomMixer"),  # ✅ Add here
    }
```

**Step 3: Use Your Mixer**

```python
# ✅ Works automatically
from audio_separation_core.core import AudioMixerFactory

mixer = AudioMixerFactory.create("my_custom")
result = mixer.mix(
    {"vocals": "vocals.wav", "accompaniment": "accompaniment.wav"},
    "output.wav"
)
```

---

### 3. Adding New Processors

#### Step-by-Step Guide

**Step 1: Create Processor Class**

```python
# processors/my_custom_processor.py
from pathlib import Path
from typing import Dict, Any
from ..core.base_component import BaseComponent
from ..core.interfaces import IAudioProcessor
from ..core.config import ProcessorConfig

class MyCustomProcessor(BaseComponent, IAudioProcessor):
    """
    Custom processor implementation.
    
    Inherits from BaseComponent for lifecycle management.
    Implements IAudioProcessor interface.
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize custom processor."""
        super().__init__()
        self._config = config or ProcessorConfig()
        if config:
            self._config.validate()
    
    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> str:
        """
        Process audio file.
        
        Required by IAudioProcessor interface.
        """
        self._ensure_ready()
        
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
        output_path = Path(output_path)
        
        # Your processing logic here
        # ...
        
        return str(output_path)
    
    def get_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """Get metadata from audio file."""
        # Your metadata extraction logic
        return {
            "duration": 120.5,
            "sample_rate": 44100,
            "channels": 2,
        }
    
    def validate(self, audio_path: Path) -> bool:
        """Validate audio file."""
        # Your validation logic
        return audio_path.exists() and audio_path.suffix in [".wav", ".mp3"]
```

**Step 2: Register in ComponentLoader**

```python
# core/loader.py
class ComponentLoader:
    PROCESSOR_MAP = {
        "extractor": ("..processors.video_extractor", "VideoAudioExtractor"),
        "converter": ("..processors.format_converter", "AudioFormatConverter"),
        "enhancer": ("..processors.audio_enhancer", "AudioEnhancer"),
        "my_custom": ("..processors.my_custom_processor", "MyCustomProcessor"),  # ✅ Add here
    }
```

**Step 3: Use Your Processor**

```python
# ✅ Works automatically
from audio_separation_core.core import AudioProcessorFactory

processor = AudioProcessorFactory.create("my_custom")
result = processor.process("input.wav", "output.wav")
```

---

## 🔧 Advanced Extensions

### Extending ComponentRegistry

```python
# Create custom registry with caching
class CachedComponentRegistry(ComponentRegistry):
    """Registry with instance caching."""
    
    def __init__(self):
        super().__init__()
        self._instances: Dict[str, Any] = {}
    
    def get_cached(self, name: str, factory_func: Callable) -> Any:
        """Get or create cached instance."""
        if name not in self._instances:
            self._instances[name] = factory_func()
        return self._instances[name]
    
    def clear_cache(self) -> None:
        """Clear cached instances."""
        for instance in self._instances.values():
            if hasattr(instance, 'cleanup'):
                instance.cleanup()
        self._instances.clear()

# Use in factory
class AudioSeparatorFactory:
    _registry = CachedComponentRegistry()  # ✅ Use custom registry
    ...
```

---

### Extending ComponentLoader

```python
# Add custom loading logic
class CustomComponentLoader(ComponentLoader):
    """Loader with custom import strategies."""
    
    @classmethod
    def load_separator(cls, separator_type: str) -> Type:
        """Override with custom loading logic."""
        # Try local import first
        try:
            return super().load_separator(separator_type)
        except AudioConfigurationError:
            # Fallback to remote import
            return cls._load_from_remote(separator_type)
    
    @classmethod
    def _load_from_remote(cls, separator_type: str) -> Type:
        """Load component from remote source."""
        # Your remote loading logic
        ...
```

---

### Extending SeparatorDetector

```python
# Add custom detection logic
class CustomSeparatorDetector(SeparatorDetector):
    """Detector with custom priority and checks."""
    
    PRIORITY = ["my_custom", "demucs", "spleeter", "lalal"]  # ✅ Custom priority
    
    @classmethod
    def is_available(cls, separator_type: str) -> bool:
        """Override with custom availability checks."""
        if separator_type == "my_custom":
            # Custom check for your separator
            return cls._check_my_custom_available()
        return super().is_available(separator_type)
    
    @classmethod
    def _check_my_custom_available(cls) -> bool:
        """Check if custom separator is available."""
        # Your custom check logic
        return True
```

---

## 🎨 Custom Factory Patterns

### Factory with Caching

```python
class CachedAudioSeparatorFactory(AudioSeparatorFactory):
    """Factory with instance caching."""
    
    _instance_cache: Dict[str, IAudioSeparator] = {}
    
    @classmethod
    def create(
        cls,
        separator_type: str = "auto",
        config: Optional[SeparationConfig] = None,
        use_cache: bool = True,
        **kwargs
    ) -> IAudioSeparator:
        """Create with optional caching."""
        if not use_cache:
            return super().create(separator_type, config, **kwargs)
        
        # Generate cache key
        cache_key = cls._generate_cache_key(separator_type, config, kwargs)
        
        # Check cache
        if cache_key in cls._instance_cache:
            cached = cls._instance_cache[cache_key]
            if cached.is_ready:  # ✅ Use BaseComponent.is_ready
                return cached
        
        # Create new instance
        instance = super().create(separator_type, config, **kwargs)
        cls._instance_cache[cache_key] = instance
        return instance
    
    @classmethod
    def _generate_cache_key(cls, separator_type, config, kwargs):
        """Generate cache key from parameters."""
        import hashlib
        import json
        key_data = {
            "type": separator_type,
            "config": config.__dict__ if config else None,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear instance cache."""
        for instance in cls._instance_cache.values():
            instance.cleanup()  # ✅ Use BaseComponent.cleanup()
        cls._instance_cache.clear()
```

---

### Factory with Pooling

```python
class PooledAudioSeparatorFactory(AudioSeparatorFactory):
    """Factory with instance pooling."""
    
    _pools: Dict[str, List[IAudioSeparator]] = {}
    _pool_size: int = 5
    
    @classmethod
    def create(
        cls,
        separator_type: str = "auto",
        config: Optional[SeparationConfig] = None,
        **kwargs
    ) -> IAudioSeparator:
        """Create or get from pool."""
        separator_type = separator_type.lower()
        
        if separator_type == "auto":
            separator_type = cls._detector.detect_best()
        
        # Get from pool if available
        if separator_type in cls._pools and cls._pools[separator_type]:
            separator = cls._pools[separator_type].pop()
            if separator.is_ready:  # ✅ Use BaseComponent.is_ready
                return separator
            separator.cleanup()
        
        # Create new instance
        return super().create(separator_type, config, **kwargs)
    
    @classmethod
    def return_to_pool(cls, separator: IAudioSeparator, separator_type: str) -> None:
        """Return separator to pool."""
        separator_type = separator_type.lower()
        
        if separator_type not in cls._pools:
            cls._pools[separator_type] = []
        
        if len(cls._pools[separator_type]) < cls._pool_size:
            status = separator.get_status()  # ✅ Use BaseComponent.get_status()
            if status["health"] == "healthy":
                cls._pools[separator_type].append(separator)
            else:
                separator.cleanup()
        else:
            separator.cleanup()
```

---

## 🔌 Plugin System

### Creating a Plugin Interface

```python
# core/plugin.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class IAudioPlugin(ABC):
    """Interface for audio plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize plugin."""
        pass
    
    @abstractmethod
    def process(self, audio_data: Any, **kwargs) -> Any:
        """Process audio data."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin."""
        pass

# Plugin registry
class PluginRegistry:
    """Registry for plugins."""
    
    _plugins: Dict[str, IAudioPlugin] = {}
    
    @classmethod
    def register(cls, plugin: IAudioPlugin) -> None:
        """Register a plugin."""
        cls._plugins[plugin.name] = plugin
    
    @classmethod
    def get(cls, name: str) -> IAudioPlugin:
        """Get a plugin."""
        if name not in cls._plugins:
            raise KeyError(f"Plugin '{name}' not found")
        return cls._plugins[name]
    
    @classmethod
    def list_plugins(cls) -> List[str]:
        """List all registered plugins."""
        return list(cls._plugins.keys())
```

---

### Using Plugins

```python
# Create a plugin
class ReverbPlugin(IAudioPlugin):
    def __init__(self):
        self._name = "reverb"
        self._version = "1.0.0"
        self._initialized = False
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    def initialize(self, **kwargs) -> bool:
        # Initialize plugin
        self._initialized = True
        return True
    
    def process(self, audio_data: Any, **kwargs) -> Any:
        # Apply reverb
        return audio_data  # Processed audio
    
    def cleanup(self) -> None:
        self._initialized = False

# Register and use
PluginRegistry.register(ReverbPlugin())
plugin = PluginRegistry.get("reverb")
plugin.initialize()
processed = plugin.process(audio_data)
```

---

## 📊 Monitoring and Observability

### Adding Metrics Collection

```python
class MetricsCollector:
    """Collect metrics from components."""
    
    @staticmethod
    def collect_component_metrics(component: BaseComponent) -> Dict[str, Any]:
        """Collect metrics from a component."""
        status = component.get_status()  # ✅ Use BaseComponent.get_status()
        
        return {
            "name": status["name"],
            "uptime": status["uptime_seconds"],
            "health": status["health"],
            "initialized": status["initialized"],
            "ready": status["ready"],
            "last_error": status.get("last_error"),
        }
    
    @staticmethod
    def collect_all_metrics(components: List[BaseComponent]) -> Dict[str, Dict[str, Any]]:
        """Collect metrics from multiple components."""
        return {
            component.name: MetricsCollector.collect_component_metrics(component)
            for component in components
        }

# Usage
separator = AudioSeparatorFactory.create("spleeter")
mixer = AudioMixerFactory.create("simple")

metrics = MetricsCollector.collect_all_metrics([separator, mixer])
```

---

## ✅ Best Practices for Extensions

1. **Always Inherit from Base Classes**: Use `BaseSeparator`, `BaseMixer`, or `BaseComponent`
2. **Implement Abstract Methods**: All abstract methods must be implemented
3. **Use Template Methods**: Override `_do_initialize()` and `_do_cleanup()`, not `initialize()` and `cleanup()`
4. **Extend, Don't Replace**: When overriding methods, call `super()` to preserve base functionality
5. **Register in ComponentLoader**: Add your component to the appropriate map
6. **Test Your Extensions**: Write tests for your custom components
7. **Document Well**: Document your extensions clearly

---

## 🎓 Summary

The refactored architecture is highly extensible:

- ✅ **Easy to Add Components**: Just inherit from base classes and register
- ✅ **Clear Extension Points**: Well-defined interfaces and abstract methods
- ✅ **Composable**: Can combine components in different ways
- ✅ **Testable**: Easy to test extensions independently
- ✅ **Maintainable**: Changes don't affect other components

Follow this guide to extend the architecture while maintaining its quality and consistency.

