# 🏗️ Flux2 Official Architecture Integration

## 📋 Resumen

Se ha integrado la arquitectura oficial de Flux2 del repositorio de Black Forest Labs:
**https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py**

## ✨ Implementación

### 1. **Flux2 Core (`flux2_core.py`)**

Implementación completa de la arquitectura Flux2 oficial:

- ✅ **Flux2Core**: Modelo principal con doble stream y single stream blocks
- ✅ **DoubleStreamBlock**: Bloques de doble stream para interacción imagen-texto
- ✅ **SingleStreamBlock**: Bloques de single stream
- ✅ **SelfAttention**: Atención con QKNorm
- ✅ **RoPE (Rotary Position Embedding)**: Embeddings posicionales rotatorios
- ✅ **Modulation**: Normalización adaptativa con modulación
- ✅ **SiLUActivation**: Activación SiLU con mecanismo de gate

### 2. **Flux2 Clothing Changer V2 (`flux2_clothing_model_v2.py`)**

Wrapper que integra la arquitectura oficial con capacidades de cambio de ropa:

- ✅ Usa `FluxInpaintPipeline` de diffusers (que internamente usa Flux2 oficial)
- ✅ Mantiene compatibilidad con el sistema de embeddings
- ✅ Integra detección automática de máscaras
- ✅ Soporte para inpainting optimizado

## 🔧 Arquitectura Flux2

### Componentes Principales

```
Flux2Core
├── Double Stream Blocks (8 bloques)
│   ├── Image Stream
│   │   ├── Self-Attention
│   │   └── MLP
│   └── Text Stream
│       ├── Self-Attention
│       └── MLP
│   └── Cross-Attention (imagen-texto)
│
├── Single Stream Blocks (48 bloques)
│   ├── Self-Attention
│   └── MLP
│
└── Final Layer
    └── Adaptive LayerNorm + Linear
```

### Parámetros del Modelo

```python
Flux2Params(
    in_channels=128,
    context_in_dim=15360,  # CLIP text embedding
    hidden_size=6144,
    num_heads=48,
    depth=8,  # Double stream blocks
    depth_single_blocks=48,  # Single stream blocks
    axes_dim=[32, 32, 32, 32],  # RoPE dimensions
    theta=2000,  # RoPE base frequency
    mlp_ratio=3.0,
)
```

## 🚀 Uso

### Opción 1: Usar V2 (Arquitectura Oficial)

```python
from character_clothing_changer_ai.models import Flux2ClothingChangerModelV2

model = Flux2ClothingChangerModelV2(
    model_id="black-forest-labs/flux2-dev",
    use_inpainting=True,
    use_core_architecture=True,
)

result = model.change_clothing(
    image="character.jpg",
    clothing_description="a red elegant dress",
)
```

### Opción 2: Usar V1 (Compatible)

```python
from character_clothing_changer_ai.models import Flux2ClothingChangerModel

model = Flux2ClothingChangerModel(
    model_id="black-forest-labs/flux2-dev",
    use_inpainting=True,
)
```

## 📊 Diferencias entre V1 y V2

| Característica | V1 | V2 |
|---------------|----|----|
| Arquitectura | Wrapper sobre diffusers | Arquitectura oficial Flux2 |
| Core Implementation | No expuesta | `flux2_core.py` |
| Compatibilidad | ✅ | ✅ |
| Rendimiento | Optimizado | Optimizado + Arquitectura oficial |
| Flexibilidad | Alta | Muy Alta |

## 🔍 Detalles Técnicos

### Double Stream Blocks

Los bloques de doble stream permiten interacción entre imagen y texto:

1. **Image Stream**: Procesa tokens de imagen
2. **Text Stream**: Procesa tokens de texto
3. **Cross-Attention**: Combina ambos streams para atención cruzada

### Single Stream Blocks

Después de combinar imagen y texto, los bloques de single stream procesan la secuencia combinada.

### RoPE (Rotary Position Embedding)

- Embeddings posicionales rotatorios para mejor comprensión espacial
- 4 ejes dimensionales: [32, 32, 32, 32]
- Theta base: 2000

### Modulation

Normalización adaptativa que modula:
- Shift (desplazamiento)
- Scale (escala)
- Gate (puerta para residual connection)

## 🎯 Ventajas de la Arquitectura Oficial

1. **Precisión**: Implementación exacta del modelo oficial
2. **Compatibilidad**: Compatible con checkpoints oficiales
3. **Rendimiento**: Optimizaciones nativas de Flux2
4. **Mantenibilidad**: Fácil de actualizar con nuevas versiones
5. **Transparencia**: Código fuente visible y documentado

## 📚 Referencias

- **Repositorio Oficial**: https://github.com/black-forest-labs/flux2
- **Modelo Base**: https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py
- **Documentación Diffusers**: https://huggingface.co/docs/diffusers

## 🔄 Integración con el Sistema

El modelo V2 se integra perfectamente con:

- ✅ Sistema de embeddings (CharacterEncoder, ClothingEncoder)
- ✅ Generación de máscaras (MaskGenerator)
- ✅ Generación de safe tensors (ComfyUITensorGenerator)
- ✅ Sistema de caché (EmbeddingCache)
- ✅ Métricas de calidad (QualityMetrics)
- ✅ Mejora de prompts (PromptEnhancer)

## 🚀 Próximos Pasos

1. **Fine-tuning**: Entrenar el modelo en datos específicos de cambio de ropa
2. **Optimizaciones**: Aplicar optimizaciones específicas para inpainting
3. **ControlNet**: Integrar ControlNet para mejor control
4. **LoRA**: Soporte para LoRA adapters personalizados


