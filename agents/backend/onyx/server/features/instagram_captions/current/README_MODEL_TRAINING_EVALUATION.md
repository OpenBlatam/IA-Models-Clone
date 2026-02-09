# Model Training and Evaluation System

Sistema completo de entrenamiento y evaluación de modelos implementando todas las mejores prácticas de deep learning.

## Características

- **Data Loading Eficiente**: DataLoader optimizado con splits automáticos
- **Splits de Datos**: Train/validation/test splits con cross-validation k-fold
- **Early Stopping**: Prevención de overfitting con restauración de mejores pesos
- **Learning Rate Scheduling**: Múltiples estrategias (cosine, step, exponential, plateau)
- **Métricas Completas**: Accuracy, precision, recall, F1, confusion matrix
- **Gradient Clipping**: Prevención de exploding gradients
- **NaN/Inf Handling**: Detección y manejo de valores problemáticos
- **Mixed Precision**: Entrenamiento con torch.cuda.amp para optimización
- **TensorBoard Integration**: Logging y visualización de métricas
- **Checkpointing**: Guardado automático del mejor modelo

## Instalación

```bash
pip install -r requirements_model_training_evaluation.txt
```

## Uso Básico

### Configuración de Datos

```python
from model_training_evaluation_system import DataSplitter, ModelTrainer, ModelEvaluator

# Split automático de datos
train_dataset, val_dataset, test_dataset = DataSplitter.train_val_test_split(
    dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# Cross-validation k-fold
kfold_splits = DataSplitter.k_fold_split(dataset, n_splits=5)
```

### Entrenamiento del Modelo

```python
# Crear trainer con todas las optimizaciones
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler_type="cosine",  # cosine, step, exponential, plateau
    early_stopping_patience=7,
    gradient_clip_norm=1.0,
    mixed_precision=True
)

# Entrenar modelo
metrics_history = trainer.train(num_epochs=100)
```

### Evaluación del Modelo

```python
# Evaluar modelo en test set
evaluator = ModelEvaluator(model)
test_metrics = evaluator.evaluate(test_loader)

print(f"Test Accuracy: {test_metrics['test_accuracy']:.2f}%")
print(f"F1 Score: {test_metrics['f1_score']:.4f}")
```

## Componentes Principales

### EarlyStopping

```python
early_stopping = EarlyStopping(
    patience=7,                    # Número de épocas sin mejora
    min_delta=0.001,              # Mejora mínima requerida
    restore_best_weights=True     # Restaurar mejores pesos
)
```

### LearningRateScheduler

```python
# Cosine annealing
scheduler = LearningRateScheduler(
    optimizer, 
    scheduler_type="cosine",
    T_max=100,        # Número total de épocas
    eta_min=1e-6      # Learning rate mínimo
)

# Reduce on plateau
scheduler = LearningRateScheduler(
    optimizer,
    scheduler_type="plateau",
    factor=0.1,       # Factor de reducción
    patience=10,      # Paciencia antes de reducir
    mode="min"        # Reducir cuando métrica baje
)
```

### MetricsTracker

```python
metrics_tracker = MetricsTracker(save_dir="./metrics")

# Actualizar métricas
metrics_tracker.update({
    "train_loss": 0.5,
    "val_loss": 0.6,
    "train_acc": 85.2,
    "val_acc": 82.1
}, epoch=10)

# Visualizar métricas
metrics_tracker.plot_metrics(save_path="./training_curves.png")
```

### GradientClipper

```python
# Clipping automático durante entrenamiento
GradientClipper.clip_gradients(model, max_norm=1.0)

# Monitorear norma de gradientes
grad_norm = GradientClipper.get_gradient_norm(model)
```

### NaNInfHandler

```python
# Verificar parámetros del modelo
if not NaNInfHandler.check_model_parameters(model):
    print("NaN/Inf detectado en parámetros")

# Verificar gradientes
if not NaNInfHandler.check_gradients(model):
    print("NaN/Inf detectado en gradientes")
```

## Configuraciones Avanzadas

### Mixed Precision Training

```python
trainer = ModelTrainer(
    # ... otros parámetros ...
    mixed_precision=True  # Usar torch.cuda.amp
)
```

### Cross-Validation

```python
# Entrenamiento con k-fold cross-validation
kfold_splits = DataSplitter.k_fold_split(dataset, n_splits=5)

for fold, (train_subset, val_subset) in enumerate(kfold_splits):
    print(f"Training fold {fold + 1}/5")
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    trainer = ModelTrainer(model, train_loader, val_loader, ...)
    metrics = trainer.train(num_epochs=50)
```

### Custom Schedulers

```python
# Scheduler personalizado
class CustomScheduler(LearningRateScheduler):
    def _create_scheduler(self, **kwargs):
        # Implementar lógica personalizada
        return custom_scheduler
```

## Métricas y Visualización

### Métricas Disponibles

- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase  
- **F1 Score**: Media armónica de precision y recall
- **Confusion Matrix**: Matriz de confusión visual

### Visualizaciones

```python
# Gráficos automáticos de entrenamiento
metrics_tracker.plot_metrics()

# Matriz de confusión
evaluator.evaluate(test_loader)  # Genera automáticamente confusion_matrix.png

# TensorBoard
tensorboard --logdir=./logs
```

## Optimizaciones de Rendimiento

### Mixed Precision

```python
# Automático con mixed_precision=True
# Reduce uso de memoria GPU y acelera entrenamiento
```

### Gradient Clipping

```python
# Prevenir exploding gradients
gradient_clip_norm=1.0  # Norma máxima de gradientes
```

### Early Stopping

```python
# Detener entrenamiento cuando no hay mejora
early_stopping_patience=7  # Épocas sin mejora
```

## Manejo de Errores

### NaN/Inf Detection

```python
# Verificación automática durante entrenamiento
# Salta batches problemáticos y continúa entrenamiento
```

### Gradient Monitoring

```python
# Monitoreo continuo de gradientes
# Logging automático en TensorBoard
```

## Logging y Checkpointing

### TensorBoard

```python
# Métricas automáticas en TensorBoard
# Loss, accuracy, learning rate, gradient norm
```

### Checkpoints

```python
# Guardado automático del mejor modelo
# Restauración de pesos óptimos
```

## Ejemplos de Uso

### Entrenamiento Básico

```python
# Dataset y modelo
dataset = YourDataset()
model = YourModel()

# Splits
train_data, val_data, test_data = DataSplitter.train_val_test_split(dataset)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    scheduler_type="cosine",
    early_stopping_patience=10,
    mixed_precision=True
)

# Entrenar
metrics = trainer.train(num_epochs=100)

# Evaluar
evaluator = ModelEvaluator(model)
test_metrics = evaluator.evaluate(test_loader)
```

### Entrenamiento con Cross-Validation

```python
# K-fold cross-validation
kfold_splits = DataSplitter.k_fold_split(dataset, n_splits=5)
cv_results = []

for fold, (train_subset, val_subset) in enumerate(kfold_splits):
    print(f"Fold {fold + 1}/5")
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_type="plateau"
    )
    
    metrics = trainer.train(num_epochs=50)
    cv_results.append(metrics)

# Promedio de resultados
avg_val_loss = np.mean([m['val_loss'][-1] for m in cv_results])
print(f"Average validation loss: {avg_val_loss:.4f}")
```

## Estructura del Proyecto

```
model_training_evaluation_system.py     # Sistema principal
requirements_model_training_evaluation.txt  # Dependencias
README_MODEL_TRAINING_EVALUATION.md     # Documentación
```

## Dependencias Principales

- **torch**: Framework de deep learning
- **scikit-learn**: Métricas y cross-validation
- **matplotlib/seaborn**: Visualización
- **tensorboard**: Logging y monitoreo
- **tqdm**: Barras de progreso

## Requisitos del Sistema

- **GPU**: NVIDIA con CUDA (recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **Python**: 3.8+

## Solución de Problemas

### Error de Memoria GPU

```python
# Reducir batch size
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Habilitar mixed precision
trainer = ModelTrainer(..., mixed_precision=True)
```

### Overfitting

```python
# Aumentar early stopping patience
trainer = ModelTrainer(..., early_stopping_patience=15)

# Usar scheduler plateau
trainer = ModelTrainer(..., scheduler_type="plateau")
```

### Gradientes Explosivos

```python
# Reducir gradient clipping
trainer = ModelTrainer(..., gradient_clip_norm=0.5)

# Verificar learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # LR más bajo
```

## Contribución

Para contribuir al sistema:

1. Fork del repositorio
2. Crear rama feature
3. Implementar cambios
4. Ejecutar tests
5. Crear Pull Request

## Licencia

Este proyecto está bajo la licencia MIT.

## Contacto

Para soporte técnico o preguntas, abrir un issue en el repositorio.


