# Quick Reference - Advanced Upscaling

## 🚀 Inicio Rápido

```python
from .models import AdvancedUpscaling

# Crear instancia
upscaler = AdvancedUpscaling(
    enable_cache=True,
    auto_select_method=True
)

# Upscaling básico
result = upscaler.upscale("image.jpg", scale_factor=2.0)
```

## 📚 Métodos por Categoría

### Core Upscaling
```python
upscaler.upscale(image, scale_factor, method="lanczos")
upscaler.upscale_with_retry(image, scale_factor, method="lanczos")
upscaler.upscale_lanczos(image, scale_factor)
upscaler.upscale_bicubic_enhanced(image, scale_factor)
upscaler.upscale_opencv_edsr(image, scale_factor)
upscaler.multi_scale_upscale(image, scale_factor)
upscaler.upscale_adaptive(image, scale_factor)
```

### Enhancement
```python
upscaler.enhance_edges(image, strength=1.2)
upscaler.apply_anti_aliasing(image, strength=0.3)
upscaler.reduce_artifacts(image, method="bilateral")
upscaler.texture_enhancement(image, strength=0.3)
upscaler.color_enhancement(image, saturation=1.1)
upscaler.adaptive_contrast_enhancement(image)
```

### Advanced Methods
```python
upscaler.upscale_with_smart_enhancement(image, scale_factor, mode="auto")
upscaler.upscale_with_quality_boosting(image, scale_factor, level="ultra")
upscaler.upscale_with_hybrid_method(image, scale_factor, primary="real_esrgan_like")
upscaler.upscale_with_adaptive_quality_control(image, scale_factor, target=0.85)
```

### Specialized
```python
upscaler.upscale_face(image, scale_factor)
upscaler.upscale_text(image, scale_factor)
upscaler.upscale_artwork(image, scale_factor)
upscaler.upscale_photo(image, scale_factor)
upscaler.upscale_anime(image, scale_factor)
upscaler.auto_detect_and_upscale(image, scale_factor)
```

### Batch Processing
```python
upscaler.upscale_async(image, scale_factor, method="lanczos")
upscaler.batch_upscale(images, scale_factor, method="lanczos")
upscaler.batch_upscale_async(images, scale_factor)
upscaler.batch_upscale_with_analysis(images, scale_factor)
```

### Analysis
```python
upscaler.analyze_image_characteristics(image)
upscaler.get_processing_recommendations(image, scale_factor)
upscaler.compare_methods(image, scale_factor, methods=None)
upscaler.get_statistics()
```

### Optimization
```python
upscaler.optimize_upscaling_method(image, scale_factor)
upscaler.optimize_for_speed(image, scale_factor, min_quality=0.7)
upscaler.optimize_for_quality(image, scale_factor, max_time=None)
upscaler.get_optimization_recommendations(image, scale_factor, priority="balanced")
```

### Quality Assurance
```python
upscaler.validate_upscale_quality(original, upscaled, min_quality=0.7)
upscaler.upscale_with_quality_assurance(image, scale_factor, min_quality=0.85)
upscaler.compare_quality(images_dict, reference=None)
upscaler.get_quality_report(image, scale_factor, method="auto")
```

### Cache Management
```python
upscaler.clear_cache()
upscaler.get_cache_stats()
upscaler.optimize_memory()
upscaler.reset_statistics()
upscaler.get_memory_usage()
```

### Configuration
```python
upscaler.get_config()
upscaler.update_config(**kwargs)
upscaler.create_preset(name, config, description="")
upscaler.load_preset(name)
upscaler.list_presets()
upscaler.save_config(file_path)
upscaler.load_config(file_path)
```

### Benchmarking
```python
upscaler.benchmark_methods(image, scale_factor, methods=None, iterations=1)
upscaler.benchmark_quality(image, scale_factor, method="auto")
upscaler.benchmark_speed(image, scale_factor, methods=None, iterations=5)
upscaler.comprehensive_benchmark(image, scale_factor, methods=None)
```

### Export
```python
upscaler.export_image(image, output_path, format="PNG", quality=95)
upscaler.export_batch(images, output_dir, base_name="upscaled")
upscaler.export_report(report, output_path, format="json")
upscaler.export_statistics(output_path, format="json")
upscaler.export_comparison(comparison, output_path, format="json")
```

### Utilities
```python
upscaler.get_optimal_resolution(original_size, scale_factor, max_dimension=None)
upscaler.resize_to_fit(image, max_size, maintain_aspect=True)
upscaler.convert_format(image, target_format="RGB")
upscaler.get_image_info(image)
upscaler.validate_image_file(file_path)
upscaler.batch_get_image_info(images)
upscaler.create_thumbnail(image, size=(128, 128))
```

## 🎯 Casos de Uso Comunes

### Upscaling Simple
```python
result = upscaler.upscale("image.jpg", 2.0)
```

### Upscaling con Mejora Inteligente
```python
result = upscaler.upscale_with_smart_enhancement("image.jpg", 2.0, "auto")
```

### Upscaling Especializado
```python
# Para rostros
result = upscaler.upscale_face("portrait.jpg", 2.0)

# Para texto
result = upscaler.upscale_text("document.jpg", 2.0)

# Auto-detección
result = upscaler.auto_detect_and_upscale("image.jpg", 2.0)
```

### Procesamiento por Lotes
```python
results = upscaler.batch_upscale(["img1.jpg", "img2.jpg"], 2.0)
```

### Optimización
```python
# Encontrar mejor método
optimization = upscaler.optimize_upscaling_method("image.jpg", 2.0)
result = upscaler.upscale("image.jpg", 2.0, method=optimization["best_method"])
```

### Benchmarking
```python
benchmark = upscaler.benchmark_methods("image.jpg", 2.0)
print(f"Best: {benchmark['_summary']['best_quality']}")
```

### Configuración
```python
# Usar preset
upscaler.load_preset("quality")

# Crear preset personalizado
upscaler.create_preset("my_preset", {"cache_size": 256})
```

## 📊 Métodos Disponibles: 70+

- Core: 7 métodos
- Enhancement: 7 métodos
- Advanced: 4 métodos
- Specialized: 6 métodos
- Batch: 4 métodos
- Analysis: 4 métodos
- Cache: 5 métodos
- Optimization: 4 métodos
- Quality: 4 métodos
- Configuration: 10 métodos
- Benchmarking: 4 métodos
- Export: 5 métodos
- Utilities: 7 métodos

## 🔧 Configuración Rápida

```python
# Preset rápido
upscaler = AdvancedUpscaling()
upscaler.load_preset("fast")

# Preset calidad
upscaler.load_preset("quality")

# Preset balanceado
upscaler.load_preset("balanced")

# Personalizado
upscaler.update_config(
    enable_cache=True,
    cache_size=128,
    auto_select_method=True
)
```

## ✅ Sistema Completo

- 15 mixins modulares
- 70+ métodos disponibles
- Compatibilidad completa
- Documentación completa
- Listo para producción


