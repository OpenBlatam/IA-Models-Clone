# Deep Learning Generator - Usage Examples

## Basic Usage

### Simple Generator Creation

```python
from core.deep_learning_generator import create_generator

# Create a basic generator
generator = create_generator(
    framework="pytorch",
    model_type="transformer"
)
```

### Using Presets

```python
from core.deep_learning_generator import create_generator, get_preset, list_presets

# List available presets
print(list_presets())
# ['transformer_pytorch', 'cnn_pytorch', 'llm_pytorch', ...]

# Get a preset configuration
config = get_preset("llm_pytorch")

# Create generator with preset
generator = create_generator(**config)
```

### Using Config Builder (Fluent API)

```python
from core.deep_learning_generator import create_generator, create_config_builder

# Build configuration fluently
config = (create_config_builder()
          .with_framework("pytorch")
          .with_model_type("transformer")
          .with_gpu(True)
          .with_mixed_precision(True)
          .with_batch_size(32)
          .with_learning_rate(1e-4)
          .with_epochs(10)
          .with_early_stopping(True, patience=5)
          .with_gradient_clipping(True, max_norm=1.0)
          .with_checkpointing(True, save_best=True)
          .with_experiment_tracking(True, backend="wandb")
          .build())

# Create generator
generator = create_generator(**config)
```

## Advanced Usage

### With Validation

```python
from core.deep_learning_generator import (
    create_generator,
    validate_generator_config,
    ValidationError
)

# Validate before creating
config = {"framework": "pytorch", "model_type": "transformer"}
is_valid, error = validate_generator_config(config)

if is_valid:
    generator = create_generator(**config)
else:
    print(f"Validation error: {error}")
```

### With Monitoring

```python
from core.deep_learning_generator import (
    create_generator,
    get_metrics,
    record_generator_creation
)

try:
    generator = create_generator(
        framework="pytorch",
        model_type="transformer"
    )
    record_generator_creation("pytorch", "transformer", success=True)
except Exception as e:
    record_generator_creation("pytorch", "transformer", success=False, error=str(e))

# Get metrics
metrics = get_metrics()
print(metrics.get_stats())
```

### Using Factory Directly

```python
from core.deep_learning_generator.factory import get_factory

factory = get_factory()
if factory.is_available:
    generator = factory.create(
        framework="pytorch",
        model_type="transformer",
        enable_advanced_features=True
    )
```

### Custom Configuration

```python
from core.deep_learning_generator import create_generator, create_config_builder

# Start with preset and customize
config = (create_config_builder()
          .with_dict(get_preset("production_ready"))
          .with_custom("custom_parameter", "custom_value")
          .with_batch_size(64)  # Override preset
          .build())

generator = create_generator(**config)
```

## Common Patterns

### Production-Ready Configuration

```python
config = get_preset("production_ready")
generator = create_generator(**config)
```

### Fast Prototyping

```python
config = get_preset("fast_prototyping")
generator = create_generator(**config)
```

### LLM Fine-tuning

```python
config = get_preset("llm_pytorch")
# Customize for fine-tuning
config["learning_rate"] = 2e-5
config["num_epochs"] = 5
generator = create_generator(**config)
```

### Diffusion Model Training

```python
config = get_preset("diffusion_pytorch")
generator = create_generator(**config)
```

## Error Handling

```python
from core.deep_learning_generator import (
    create_generator,
    ValidationError,
    get_metrics
)

try:
    generator = create_generator(
        framework="pytorch",
        model_type="transformer"
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
except ImportError as e:
    print(f"Generator not available: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    # Check metrics for details
    metrics = get_metrics()
    errors = metrics.get_errors(limit=5)
    for error in errors:
        print(f"Error: {error}")
```

## Advanced Examples

### Configuration Optimization

```python
from core.deep_learning_generator import (
    create_generator,
    optimize_config,
    get_preset
)

# Get base config
config = get_preset("llm_pytorch")

# Optimize for hardware
hardware_info = {
    "gpu_memory_gb": 24.0,
    "cpu_cores": 16,
    "available_ram_gb": 64.0
}

optimized = optimize_config(config, hardware_info=hardware_info)
generator = create_generator(**optimized)
```

### Save/Load Configurations

```python
from core.deep_learning_generator import (
    get_preset,
    save_config,
    load_config,
    create_generator
)

# Save configuration
config = get_preset("production_ready")
save_config(config, "my_config.json")

# Load and use
loaded_config = load_config("my_config.json")
generator = create_generator(**loaded_config)
```

### Testing

```python
from core.deep_learning_generator import create_tester

tester = create_tester()

# Test specific configuration
result = tester.test_config_validation({
    "framework": "pytorch",
    "model_type": "transformer"
})

# Test generator creation
result = tester.test_generator_creation("pytorch", "transformer")

# Run all tests
summary = tester.run_all_tests()
print(f"Tests passed: {summary['passed']}/{summary['total_tests']}")
```

### Custom Plugins

```python
from core.deep_learning_generator import (
    GeneratorPlugin,
    register_plugin,
    create_generator
)

class CustomOptimizationPlugin(GeneratorPlugin):
    def name(self) -> str:
        return "custom_optimizer"
    
    def version(self) -> str:
        return "1.0.0"
    
    def before_create(self, config):
        # Modify config before creation
        config["custom_param"] = "custom_value"
        return config
    
    def after_create(self, generator, config):
        # Modify generator after creation
        return generator

# Register plugin
register_plugin(CustomOptimizationPlugin())

# Plugin will be applied automatically
generator = create_generator(framework="pytorch")
```

### Benchmarking

```python
from core.deep_learning_generator import (
    create_benchmark_runner,
    create_generator
)

runner = create_benchmark_runner()

# Benchmark generator creation
result = runner.benchmark(
    "generator_creation",
    create_generator,
    iterations=10,
    framework="pytorch",
    model_type="transformer"
)

print(f"Average time: {result.avg_time:.4f}s")
print(f"Min: {result.min_time:.4f}s, Max: {result.max_time:.4f}s")

# Compare multiple configs
configs = [
    {"framework": "pytorch", "model_type": "transformer"},
    {"framework": "pytorch", "model_type": "cnn"}
]

comparison = runner.compare_configs(configs, create_generator)
print(f"Fastest: {comparison['fastest']}")
```

### Intelligent Recommendations

```python
from core.deep_learning_generator import recommend_config

# Recommend by use case
config = recommend_config(use_case="text_classification")
generator = create_generator(**config)

# Recommend based on data
config = recommend_config(
    data_size="large",
    data_type="text",
    task_type="classification"
)

# Recommend for budget
config = recommend_config(
    budget_type="time",
    time_budget="fast"
)
```

### Configuration Comparison

```python
from core.deep_learning_generator import (
    compare_configs,
    merge_configs,
    get_preset
)

# Compare two configs
config1 = get_preset("transformer_pytorch")
config2 = get_preset("llm_pytorch")

comparison = compare_configs(config1, config2)
print(f"Similarity: {comparison['similarity_score']:.2%}")
print(f"Differences: {list(comparison['differences'].keys())}")

# Merge multiple configs
configs = [config1, config2]
merged = merge_configs(configs, strategy="override")
```

### Configuration Versioning

```python
from core.deep_learning_generator import (
    save_config_version,
    get_config_versions,
    get_latest_config_version,
    get_preset
)

# Save version
config = get_preset("production_ready")
version = save_config_version("my_project", config, {
    "description": "Production configuration",
    "author": "Team A"
})

# Get all versions
versions = get_config_versions("my_project")
print(f"Total versions: {len(versions)}")

# Get latest
latest = get_latest_config_version("my_project")
print(f"Latest version: {latest.version}")
print(f"Created: {latest.metadata['created_at']}")
```

## Best Practices

1. **Use Presets**: Start with presets and customize as needed
2. **Validate Configs**: Always validate configurations before use
3. **Monitor Usage**: Track generator creation for debugging
4. **Use Builder**: Use ConfigBuilder for complex configurations
5. **Handle Errors**: Always handle ValidationError and ImportError
6. **Check Availability**: Verify generator is available before use
7. **Optimize Configs**: Use optimizer for hardware-specific tuning
8. **Save Configs**: Save optimized configs for reuse
9. **Test Configs**: Use tester to validate configurations
10. **Extend with Plugins**: Create plugins for custom behavior
11. **Benchmark Performance**: Use benchmark to compare configs
12. **Get Recommendations**: Use recommender for optimal configs
13. **Version Configs**: Track configuration changes over time
14. **Compare Configs**: Compare before/after optimizations

