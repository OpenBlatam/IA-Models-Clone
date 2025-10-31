# Deep Learning and Model Development System

Sistema completo para implementar flujos de trabajo de deep learning y mejores prácticas para desarrollo de modelos.

## 🚀 Características

### **Comprehensive Model Development**
- **Multiple Architectures**: Transformer, CNN, RNN, Hybrid models
- **Flexible Configuration**: Configuración configurable para diferentes tipos de modelos
- **Modular Design**: Diseño modular y reutilizable
- **Production Ready**: Sistema listo para producción

### **Advanced Training Features**
- **Mixed Precision Training**: Entrenamiento con precisión mixta (AMP)
- **Gradient Accumulation**: Acumulación de gradientes para batches grandes
- **Gradient Clipping**: Clipping de gradientes para estabilidad
- **Learning Rate Scheduling**: Programación automática de learning rate
- **Multiple Optimizers**: Adam, SGD, AdamW con configuración flexible

### **Monitoring and Logging**
- **TensorBoard Integration**: Integración completa con TensorBoard
- **Weights & Biases**: Soporte para experiment tracking con W&B
- **Real-time Metrics**: Métricas en tiempo real durante entrenamiento
- **Training Visualization**: Visualización completa del historial de entrenamiento

### **Model Architectures**
- **Transformer Models**: Modelos transformer para procesamiento de secuencias
- **CNN Models**: Modelos CNN para procesamiento de imágenes
- **RNN Models**: Modelos RNN/LSTM para secuencias
- **Hybrid Models**: Modelos híbridos combinando múltiples arquitecturas

## 📦 Instalación

```bash
pip install -r requirements_deep_learning_model_development.txt
```

## 🎯 Uso Básico

### Configuración del Sistema

```python
from deep_learning_model_development_system import (
    ModelDevelopmentConfig, DeepLearningModelDevelopmentSystem
)

# Configuración para modelo Transformer
config = ModelDevelopmentConfig(
    model_type="transformer",
    architecture_config={
        'vocab_size': 30000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1
    },
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    mixed_precision=True,
    gradient_accumulation_steps=2,
    use_tensorboard=True,
    use_wandb=False
)

# Crear sistema de desarrollo
dl_system = DeepLearningModelDevelopmentSystem(config)
```

### Entrenamiento de Modelo

```python
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# Crear dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Función de pérdida
criterion = CrossEntropyLoss()

# Entrenar modelo
history = dl_system.train_model(train_loader, val_loader, criterion)

# Evaluar modelo
metrics = dl_system.evaluate_model(test_loader, criterion)

# Visualizar historial de entrenamiento
dl_system.plot_training_history(history)
```

### Diferentes Tipos de Modelos

```python
# Modelo CNN
cnn_config = ModelDevelopmentConfig(
    model_type="cnn",
    architecture_config={
        'input_channels': 3,
        'num_classes': 1000,
        'base_channels': 64,
        'num_layers': 5
    },
    batch_size=64,
    learning_rate=1e-3
)

# Modelo RNN
rnn_config = ModelDevelopmentConfig(
    model_type="rnn",
    architecture_config={
        'input_size': 512,
        'hidden_size': 256,
        'num_layers': 2,
        'num_classes': 1000,
        'bidirectional': True
    },
    batch_size=32,
    learning_rate=1e-4
)

# Modelo Híbrido
hybrid_config = ModelDevelopmentConfig(
    model_type="hybrid",
    architecture_config={
        'transformer': {
            'vocab_size': 30000,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 3
        },
        'cnn': {
            'input_channels': 3,
            'num_classes': 1000,
            'base_channels': 32,
            'num_layers': 3
        },
        'fusion_method': 'concatenate'
    },
    batch_size=16,
    learning_rate=1e-4
)
```

## 🔧 Componentes Principales

### **DeepLearningModelDevelopmentSystem**
Sistema principal para desarrollo de modelos:

```python
from deep_learning_model_development_system import DeepLearningModelDevelopmentSystem

# Crear sistema
dl_system = DeepLearningModelDevelopmentSystem(config)

# Entrenar modelo
history = dl_system.train_model(train_loader, val_loader, criterion)

# Evaluar modelo
metrics = dl_system.evaluate_model(test_loader, criterion)

# Guardar checkpoint
dl_system._save_checkpoint('best_model.pth')

# Cargar checkpoint
dl_system.load_checkpoint('best_model.pth')

# Limpiar recursos
dl_system.cleanup()
```

### **ModelDevelopmentConfig**
Configuración completa para desarrollo de modelos:

```python
from deep_learning_model_development_system import ModelDevelopmentConfig

config = ModelDevelopmentConfig(
    # Arquitectura del modelo
    model_type="transformer",  # transformer, cnn, rnn, hybrid
    architecture_config={
        'vocab_size': 30000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6
    },
    
    # Configuración de entrenamiento
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100,
    weight_decay=1e-5,
    gradient_clip_norm=1.0,
    
    # Optimización
    optimizer_type="adamw",  # adam, sgd, adamw
    scheduler_type="cosine",  # step, cosine, plateau
    mixed_precision=True,
    gradient_accumulation_steps=2,
    
    # Datos
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    num_workers=4,
    pin_memory=True,
    
    # Monitoreo
    log_interval=100,
    eval_interval=1000,
    save_interval=5000,
    use_tensorboard=True,
    use_wandb=False
)
```

### **Model Architectures**

#### **TransformerModel**
```python
from deep_learning_model_development_system import TransformerModel

transformer = TransformerModel(
    vocab_size=30000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1
)

# Forward pass
input_tokens = torch.randint(0, 30000, (32, 100))
output_logits = transformer(input_tokens)
```

#### **CNNModel**
```python
from deep_learning_model_development_system import CNNModel

cnn = CNNModel(
    input_channels=3,
    num_classes=1000,
    base_channels=64,
    num_layers=5
)

# Forward pass
input_images = torch.randn(32, 3, 224, 224)
output_logits = cnn(input_images)
```

#### **RNNModel**
```python
from deep_learning_model_development_system import RNNModel

rnn = RNNModel(
    input_size=512,
    hidden_size=256,
    num_layers=2,
    num_classes=1000,
    bidirectional=True
)

# Forward pass
input_sequences = torch.randn(32, 100, 512)
output_logits = rnn(input_sequences)
```

#### **HybridModel**
```python
from deep_learning_model_development_system import HybridModel

hybrid = HybridModel(
    transformer_config={
        'vocab_size': 30000,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 3
    },
    cnn_config={
        'input_channels': 3,
        'num_classes': 1000,
        'base_channels': 32,
        'num_layers': 3
    },
    fusion_method='concatenate'
)
```

## 📊 Características Avanzadas

### **Mixed Precision Training**
```python
# Configurar precisión mixta
config = ModelDevelopmentConfig(
    mixed_precision=True,  # Habilitar AMP
    gradient_accumulation_steps=4
)

# El sistema automáticamente:
# - Usa autocast para forward pass
# - Escala la pérdida con GradScaler
# - Maneja la acumulación de gradientes
# - Aplica clipping de gradientes
```

### **Gradient Accumulation**
```python
# Para batches grandes que no caben en memoria
config = ModelDevelopmentConfig(
    batch_size=8,  # Batch size pequeño
    gradient_accumulation_steps=4,  # Acumular 4 pasos
    # Efectivo: batch_size = 8 * 4 = 32
)
```

### **Learning Rate Scheduling**
```python
# Diferentes tipos de schedulers
config = ModelDevelopmentConfig(
    scheduler_type="cosine",  # Cosine annealing
    # Alternativas: "step", "plateau"
)

# El sistema automáticamente:
# - Crea el scheduler apropiado
# - Actualiza el learning rate
# - Maneja ReduceLROnPlateau con métricas de validación
```

### **Advanced Monitoring**
```python
# Configurar monitoreo completo
config = ModelDevelopmentConfig(
    use_tensorboard=True,
    use_wandb=True,
    log_interval=100,
    eval_interval=1000,
    save_interval=5000
)

# El sistema automáticamente:
# - Registra métricas en TensorBoard
# - Sincroniza con Weights & Biases
# - Guarda checkpoints regulares
# - Monitorea el mejor modelo
```

## 📈 Visualización y Análisis

### **Training History Plots**
```python
# Visualizar historial completo de entrenamiento
history = dl_system.train_model(train_loader, val_loader, criterion)

dl_system.plot_training_history(
    history,
    save_path='training_history.png'
)

# Genera 4 gráficos:
# 1. Training vs Validation Loss
# 2. Learning Rate over time
# 3. Train-Val Loss Difference
# 4. Validation Loss Trend
```

### **TensorBoard Integration**
```python
# Monitoreo en tiempo real
config = ModelDevelopmentConfig(use_tensorboard=True)

# Métricas registradas automáticamente:
# - Loss/Train
# - Loss/Validation
# - Learning_Rate
# - Loss/Training_Step

# Acceder a TensorBoard:
# tensorboard --logdir=./runs
```

### **Weights & Biases Integration**
```python
# Experiment tracking
config = ModelDevelopmentConfig(use_wandb=True)

# El sistema automáticamente:
# - Inicializa W&B
# - Registra hiperparámetros
# - Logs métricas en tiempo real
# - Sincroniza checkpoints
```

## 🏗️ Arquitectura del Sistema

```
DeepLearningModelDevelopmentSystem
├── ModelDevelopmentConfig
│   ├── Model Architecture Configuration
│   ├── Training Configuration
│   ├── Optimization Configuration
│   ├── Data Configuration
│   └── Monitoring Configuration
├── Model Creation
│   ├── TransformerModel
│   ├── CNNModel
│   ├── RNNModel
│   └── HybridModel
├── Training Pipeline
│   ├── Mixed Precision Training
│   ├── Gradient Accumulation
│   ├── Learning Rate Scheduling
│   └── Checkpoint Management
├── Monitoring & Logging
│   ├── TensorBoard Integration
│   ├── Weights & Biases
│   ├── Real-time Metrics
│   └── Training Visualization
└── Model Evaluation
    ├── Test Set Evaluation
    ├── Metrics Calculation
    └── Performance Analysis
```

## 🚀 Casos de Uso

### **Large Language Models**
```python
# Configuración para LLMs grandes
llm_config = ModelDevelopmentConfig(
    model_type="transformer",
    architecture_config={
        'vocab_size': 50000,
        'd_model': 1024,
        'n_heads': 16,
        'n_layers': 12,
        'd_ff': 4096,
        'max_seq_len': 1024,
        'dropout': 0.1
    },
    batch_size=8,
    gradient_accumulation_steps=8,  # Efectivo: 64
    mixed_precision=True,
    learning_rate=1e-5
)
```

### **Computer Vision Models**
```python
# Configuración para modelos de visión
vision_config = ModelDevelopmentConfig(
    model_type="cnn",
    architecture_config={
        'input_channels': 3,
        'num_classes': 1000,
        'base_channels': 128,
        'num_layers': 6
    },
    batch_size=64,
    learning_rate=1e-3,
    optimizer_type="sgd",
    scheduler_type="step"
)
```

### **Sequence Models**
```python
# Configuración para modelos de secuencia
sequence_config = ModelDevelopmentConfig(
    model_type="rnn",
    architecture_config={
        'input_size': 512,
        'hidden_size': 512,
        'num_layers': 4,
        'num_classes': 1000,
        'bidirectional': True
    },
    batch_size=32,
    learning_rate=1e-4,
    scheduler_type="plateau"
)
```

### **Hybrid Models**
```python
# Configuración para modelos híbridos
hybrid_config = ModelDevelopmentConfig(
    model_type="hybrid",
    architecture_config={
        'transformer': {
            'vocab_size': 30000,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 3
        },
        'cnn': {
            'input_channels': 3,
            'num_classes': 1000,
            'base_channels': 32,
            'num_layers': 3
        },
        'fusion_method': 'attention'
    },
    batch_size=16,
    learning_rate=1e-4
)
```

## 📚 Mejores Prácticas

### **Model Development Workflow**
1. **Configuration**: Definir configuración completa del modelo
2. **Data Preparation**: Preparar dataloaders con splits apropiados
3. **Training**: Usar mixed precision y gradient accumulation
4. **Monitoring**: Monitorear métricas en tiempo real
5. **Evaluation**: Evaluar en conjunto de test
6. **Analysis**: Analizar resultados y visualizaciones

### **Performance Optimization**
- **Mixed Precision**: Usar AMP para acelerar entrenamiento
- **Gradient Accumulation**: Para batches grandes
- **Learning Rate Scheduling**: Para convergencia estable
- **Gradient Clipping**: Para estabilidad de entrenamiento
- **Checkpointing**: Guardar modelos regularmente

### **Monitoring Best Practices**
- **Real-time Logging**: Logs cada 100 pasos
- **Validation**: Evaluar cada época
- **Checkpointing**: Guardar cada 10 épocas
- **Visualization**: Gráficos automáticos de progreso
- **Experiment Tracking**: Usar TensorBoard y/o W&B

## 🔍 Troubleshooting

### **Common Issues**
1. **Out of Memory**: Reducir batch_size o usar gradient_accumulation_steps
2. **Training Instability**: Ajustar gradient_clip_norm y learning_rate
3. **Slow Convergence**: Verificar learning_rate y scheduler
4. **Overfitting**: Aumentar dropout y weight_decay

### **Performance Tips**
- Usar mixed precision para acelerar entrenamiento
- Ajustar num_workers para DataLoader
- Usar pin_memory=True para GPU
- Monitorear GPU memory usage

## 📈 Ejemplos de Uso

### **Entrenamiento Completo**
```python
def complete_training_workflow():
    """Flujo de trabajo completo de entrenamiento"""
    
    # 1. Configuración
    config = ModelDevelopmentConfig(
        model_type="transformer",
        architecture_config={
            'vocab_size': 30000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1
        },
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=100,
        mixed_precision=True,
        gradient_accumulation_steps=2,
        use_tensorboard=True
    )
    
    # 2. Crear sistema
    dl_system = DeepLearningModelDevelopmentSystem(config)
    
    # 3. Preparar datos
    train_loader = create_dataloader(train_dataset, batch_size=32)
    val_loader = create_dataloader(val_dataset, batch_size=32)
    test_loader = create_dataloader(test_dataset, batch_size=32)
    
    # 4. Entrenar modelo
    criterion = nn.CrossEntropyLoss()
    history = dl_system.train_model(train_loader, val_loader, criterion)
    
    # 5. Evaluar modelo
    metrics = dl_system.evaluate_model(test_loader, criterion)
    
    # 6. Visualizar resultados
    dl_system.plot_training_history(history, 'training_results.png')
    
    # 7. Limpiar recursos
    dl_system.cleanup()
    
    return metrics, history
```

### **Fine-tuning de Modelo Existente**
```python
def fine_tune_existing_model():
    """Fine-tuning de modelo pre-entrenado"""
    
    # 1. Cargar modelo existente
    dl_system = DeepLearningModelDevelopmentSystem(config)
    dl_system.load_checkpoint('pretrained_model.pth')
    
    # 2. Ajustar configuración para fine-tuning
    dl_system.config.learning_rate = 1e-5  # Learning rate más bajo
    dl_system.config.num_epochs = 20        # Menos épocas
    
    # 3. Entrenar con nuevos datos
    history = dl_system.train_model(new_train_loader, new_val_loader, criterion)
    
    # 4. Guardar modelo fine-tuneado
    dl_system._save_checkpoint('fine_tuned_model.pth')
    
    return history
```

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/DeepLearning`)
3. Commit tus cambios (`git commit -m 'Add deep learning features'`)
4. Push a la rama (`git push origin feature/DeepLearning`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Abre un issue en GitHub
- Consulta la documentación
- Revisa los ejemplos de uso

---

**Sistema de Deep Learning and Model Development implementado exitosamente** ✅

**Características implementadas:**
- ✅ Comprehensive model development
- ✅ Multiple architectures (Transformer, CNN, RNN, Hybrid)
- ✅ Advanced training features (AMP, gradient accumulation)
- ✅ Flexible configuration system
- ✅ Monitoring and logging (TensorBoard, W&B)
- ✅ Training visualization
- ✅ Checkpoint management
- ✅ Model evaluation
- ✅ Production ready code
- ✅ Best practices implementation

**SISTEMA COMPLETAMENTE FUNCIONAL PARA PRODUCCIÓN** 🚀


