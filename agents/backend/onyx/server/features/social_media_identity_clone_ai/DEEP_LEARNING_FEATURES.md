# 🧠 Deep Learning Features - Social Media Identity Clone AI

## Nuevas Capacidades de Deep Learning

### 1. **Transformers Avanzados** ✅

#### Características
- Análisis de estilo de texto usando modelos transformer
- Generación de embeddings con sentence transformers
- Búsqueda semántica de contenido similar
- Análisis de sentimiento y características

**Uso:**
```python
from ml_advanced.transformer_service import get_transformer_service

transformer = get_transformer_service()

# Analizar estilo
result = transformer.analyze_text_style("💪 Contenido motivacional...")
print(result["style"], result["confidence"])

# Generar embeddings
embeddings = transformer.generate_embeddings(["texto 1", "texto 2"])

# Encontrar contenido similar
similar = transformer.find_similar_content(
    query_text="fitness motivation",
    candidate_texts=["texto 1", "texto 2", "texto 3"],
    top_k=3
)
```

### 2. **Fine-tuning con LoRA** ✅

#### Características
- Fine-tuning eficiente con LoRA (Low-Rank Adaptation)
- Personalización de modelos para identidades específicas
- Entrenamiento con mixed precision
- Gradient accumulation para batches grandes

**Uso:**
```python
from ml_advanced.lora_finetuning import get_lora_finetuner, LoRAConfig

finetuner = get_lora_finetuner()

# Preparar modelo
lora_config = LoRAConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"]
)
model, tokenizer = finetuner.prepare_model_for_lora(
    base_model_name="gpt2",
    lora_config=lora_config
)

# Fine-tune
train_texts = ["texto 1", "texto 2", ...]
result = finetuner.fine_tune(
    train_texts=train_texts,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5
)

# Generar texto personalizado
generated = finetuner.generate_text(
    prompt="Motivational post about fitness",
    max_length=100
)
```

### 3. **Modelos de Difusión** ✅

#### Características
- Generación de imágenes con Stable Diffusion
- Optimizaciones de GPU (attention slicing, xformers)
- Generación basada en identidad
- Control de seed para reproducibilidad

**Uso:**
```python
from ml_advanced.diffusion_service import get_diffusion_service

diffusion = get_diffusion_service()

# Generar imagen
image = diffusion.generate_image(
    prompt="A motivational fitness post with vibrant colors",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# Generar desde identidad
image = diffusion.generate_image_from_identity(
    identity_description="Fitness influencer, energetic style",
    content_style="authentic",
    additional_prompt="gym workout scene"
)
```

### 4. **Demo Interactivo con Gradio** ✅

#### Características
- Interfaz web interactiva
- Análisis de texto en tiempo real
- Generación de imágenes
- Visualización de resultados

**Uso:**
```bash
python ml_advanced/gradio_demo.py
```

O desde código:
```python
from ml_advanced.gradio_demo import launch_demo

launch_demo(share=False, server_port=7860)
```

## Optimizaciones de Rendimiento

### GPU y Mixed Precision
- Uso automático de GPU cuando está disponible
- Mixed precision training (FP16) para eficiencia
- Optimizaciones de memoria (attention slicing)

### Gradient Accumulation
- Permite entrenar con batches efectivos grandes
- Útil cuando la GPU tiene memoria limitada

### Model Optimization
- Attention slicing para reducir uso de memoria
- XFormers para atención eficiente
- Model quantization (opcional)

## Arquitectura de Deep Learning

```
ml_advanced/
├── transformer_service.py    # Análisis con transformers
├── lora_finetuning.py        # Fine-tuning con LoRA
├── diffusion_service.py       # Generación de imágenes
└── gradio_demo.py            # Demo interactivo
```

## Dependencias Adicionales

```txt
torch>=2.0.0
transformers>=4.35.0
diffusers>=0.21.0
peft>=0.6.0
accelerate>=0.24.0
sentence-transformers>=2.2.0
bitsandbytes>=0.41.0
gradio>=4.0.0
```

## Ejemplos de Uso Completo

### 1. Análisis Avanzado de Identidad

```python
# Usar transformers para análisis mejorado
transformer = get_transformer_service()

# Analizar todos los posts de una identidad
posts = ["post 1", "post 2", "post 3"]
analyses = [transformer.analyze_text_style(post) for post in posts]

# Generar perfil de estilo
style_profile = {
    "avg_sentiment": np.mean([a["sentiment_score"] for a in analyses]),
    "common_features": extract_common_features(analyses)
}
```

### 2. Personalización con Fine-tuning

```python
# Fine-tune modelo para identidad específica
finetuner = get_lora_finetuner()

# Recopilar textos de la identidad
identity_texts = extract_all_texts_from_identity(identity_id)

# Entrenar
finetuner.prepare_model_for_lora("gpt2")
result = finetuner.fine_tune(
    train_texts=identity_texts,
    num_epochs=5,
    batch_size=4
)

# Usar modelo personalizado
generated = finetuner.generate_text(
    prompt="New post about",
    max_length=150
)
```

### 3. Generación de Contenido Visual

```python
# Generar imágenes para posts
diffusion = get_diffusion_service()

# Generar imagen para post de Instagram
image = diffusion.generate_image(
    prompt="Fitness motivation post, vibrant colors, gym scene",
    height=1024,
    width=1024,
    num_inference_steps=50
)

# Guardar
image.save("generated_post.png")
```

## Mejores Prácticas

### 1. Fine-tuning
- Usar datasets balanceados
- Validar con conjunto de prueba
- Monitorear overfitting
- Guardar checkpoints regularmente

### 2. Generación de Imágenes
- Usar prompts descriptivos
- Experimentar con guidance_scale
- Ajustar num_inference_steps según necesidad
- Usar seeds para reproducibilidad

### 3. Performance
- Usar GPU cuando esté disponible
- Ajustar batch_size según memoria
- Usar gradient accumulation para batches grandes
- Monitorear uso de memoria

## Troubleshooting

### GPU Out of Memory
- Reducir batch_size
- Usar gradient_accumulation_steps
- Habilitar attention_slicing
- Usar mixed precision (FP16)

### Model Loading Issues
- Verificar que modelos estén descargados
- Verificar espacio en disco
- Verificar conexión a internet (primera vez)

### Slow Inference
- Usar GPU
- Reducir num_inference_steps
- Usar modelos más pequeños
- Habilitar optimizaciones (xformers)

## Próximas Mejoras

- [ ] Fine-tuning de modelos de difusión
- [ ] ControlNet para control preciso
- [ ] Modelos de lenguaje más grandes
- [ ] Multi-modal (texto + imagen)
- [ ] Optimización con TensorRT
- [ ] Distributed training




