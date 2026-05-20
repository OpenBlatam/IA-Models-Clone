# Object-Oriented Programming for Model Architectures and Functional Programming for Data Processing Pipelines

Sistema completo implementando programación orientada a objetos para arquitecturas de modelos y programación funcional para pipelines de procesamiento de datos con mejores prácticas.

## Características

- **Object-Oriented Model Architectures**: Clases abstractas, herencia, encapsulación para modelos
- **Functional Programming Data Pipelines**: Composición de funciones, pipelines funcionales, inmutabilidad
- **Design Patterns**: Factory Pattern, Builder Pattern, Strategy Pattern
- **Model Types**: Transformer, CNN, RNN, Diffusion, Hybrid models
- **Functional Transforms**: Text processing, image processing, data augmentation
- **Type Safety**: Generic programming, type hints, abstract base classes
- **Modular Design**: Componentes reutilizables, interfaces claras, separación de responsabilidades

## Instalación

```bash
pip install -r requirements_oop_fp.txt
```

## Uso Básico

### Object-Oriented Model Creation

```python
from oop_fp_system import ModelType, ModelFactory, ModelConfigBuilder

# Create Transformer model using factory
transformer_config = {
    'vocab_size': 30000,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 2048,
    'max_seq_len': 512,
    'dropout': 0.1,
    'enable_gradient_checkpointing': True,
    'enable_mixed_precision': True
}

transformer_model = ModelFactory.create_model(ModelType.TRANSFORMER, transformer_config)

# Create Hybrid model using builder pattern
hybrid_config = (ModelConfigBuilder()
                .set_model_type(ModelType.HYBRID)
                .set_transformer_config(vocab_size=30000, d_model=256, n_heads=4, n_layers=3)
                .set_cnn_config(input_channels=3, num_classes=1000, base_channels=32, num_layers=3)
                .set_optimization_config(enable_gradient_checkpointing=True)
                .build())

hybrid_model = ModelFactory.create_model(ModelType.HYBRID, hybrid_config)
```

### Functional Programming Data Processing

```python
from oop_fp_system import DataProcessor, TransformPipeline, FunctionalDataset

# Create data processor
data_processor = DataProcessor(DataConfig(batch_size=16, num_workers=2))

# Process data functionally
processed_data = data_processor.pipe(
    raw_data,
    lambda x: data_processor.map_batch(str.lower, x),
    lambda x: data_processor.map_batch(str.strip, x),
    lambda x: data_processor.map_batch(lambda s: ' '.join(s.split()), x)
)

# Create functional transform pipeline
pipeline = TransformPipeline()
pipeline.add_transform(lambda x: x.lower())
pipeline.add_transform(lambda x: x.strip())
pipeline.add_transform(lambda x: ' '.join(x.split()))

# Apply pipeline
processed = pipeline.apply("  Hello World  ")
```

## Componentes Principales

### Object-Oriented Model System

#### BaseModel (Abstract Base Class)

```python
from oop_fp_system import BaseModel

class BaseModel(ABC, nn.Module):
    """Abstract base class for all model architectures"""
    
    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        super().__init__()
        self.model_type = model_type
        self.config = config
        
        # Initialize model components
        self._initialize_components()
        self._setup_optimization()
    
    @abstractmethod
    def _initialize_components(self):
        """Initialize model components - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type.value,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "config": self.config
        }
```

#### TransformerModel

```python
class TransformerModel(BaseModel):
    """Object-oriented Transformer model architecture"""
    
    def _initialize_components(self):
        """Initialize Transformer components"""
        vocab_size = self.config.get('vocab_size', 30000)
        d_model = self.config.get('d_model', 512)
        n_heads = self.config.get('n_heads', 8)
        n_layers = self.config.get('n_layers', 6)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for Transformer model"""
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
```

#### CNNModel

```python
class CNNModel(BaseModel):
    """Object-oriented CNN model architecture"""
    
    def _initialize_components(self):
        """Initialize CNN components"""
        input_channels = self.config.get('input_channels', 3)
        num_classes = self.config.get('num_classes', 1000)
        base_channels = self.config.get('base_channels', 64)
        num_layers = self.config.get('num_layers', 4)
        
        # Build CNN layers dynamically
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_channels, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CNN model"""
        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flattened)
        return logits
```

#### DiffusionModel

```python
class DiffusionModel(BaseModel):
    """Object-oriented Diffusion model architecture"""
    
    def _initialize_components(self):
        """Initialize Diffusion model components"""
        input_channels = self.config.get('input_channels', 3)
        base_channels = self.config.get('base_channels', 128)
        time_dim = self.config.get('time_dim', 256)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # UNet backbone
        self.unet = UNet(
            input_channels=input_channels,
            base_channels=base_channels,
            time_dim=time_dim
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_train_timesteps=self.config.get('num_train_timesteps', 1000),
            beta_start=self.config.get('beta_start', 0.0001),
            beta_end=self.config.get('beta_end', 0.02)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass for Diffusion model"""
        time_emb = self.time_embedding(timesteps)
        noise_pred = self.unet(x, time_emb)
        return noise_pred
```

#### HybridModel

```python
class HybridModel(BaseModel):
    """Object-oriented Hybrid model architecture combining multiple approaches"""
    
    def _initialize_components(self):
        """Initialize Hybrid model components"""
        self.use_transformer = self.config.get('use_transformer', True)
        self.use_cnn = self.config.get('use_cnn', True)
        self.use_rnn = self.config.get('use_rnn', False)
        
        # Initialize sub-models
        if self.use_transformer:
            self.transformer = TransformerModel(self.config.get('transformer_config', {}))
        
        if self.use_cnn:
            self.cnn = CNNModel(self.config.get('cnn_config', {}))
        
        if self.use_rnn:
            self.rnn = RNNModel(self.config.get('rnn_config', {}))
        
        # Fusion layer
        fusion_dim = self.config.get('fusion_dim', 512)
        self.fusion_layer = nn.Linear(self._get_total_features(), fusion_dim)
        self.output_layer = nn.Linear(fusion_dim, self.config.get('num_classes', 1000))
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass for Hybrid model"""
        features = []
        
        # Extract features from each sub-model
        if self.use_transformer:
            transformer_features = self.transformer(*args, **kwargs)
            if len(transformer_features.shape) == 3:
                transformer_features = torch.mean(transformer_features, dim=1)
            features.append(transformer_features)
        
        if self.use_cnn:
            cnn_features = self.cnn(*args, **kwargs)
            features.append(cnn_features)
        
        if self.use_rnn:
            rnn_features = self.rnn(*args, **kwargs)
            features.append(rnn_features)
        
        # Concatenate and fuse features
        combined_features = torch.cat(features, dim=1)
        fused = self.fusion_layer(combined_features)
        fused = F.relu(fused)
        output = self.output_layer(fused)
        
        return output
```

### Functional Programming System

#### DataProcessor

```python
class DataProcessor:
    """Functional programming approach for data processing"""
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """Compose multiple functions: compose(f, g, h)(x) = f(g(h(x)))"""
        def compose_functions(x):
            for f in reversed(functions):
                x = f(x)
            return x
        return compose_functions
    
    @staticmethod
    def pipe(data: T, *functions: Callable) -> U:
        """Pipe data through multiple functions: pipe(data, f, g, h) = h(g(f(data)))"""
        return reduce(lambda x, f: f(x), functions, data)
    
    @staticmethod
    def curry(func: Callable, *args, **kwargs) -> Callable:
        """Curry a function with partial arguments"""
        return partial(func, *args, **kwargs)
    
    @staticmethod
    def map_batch(func: Callable, batch: List[T]) -> List[U]:
        """Apply function to each item in batch"""
        return [func(item) for item in batch]
    
    @staticmethod
    def filter_batch(predicate: Callable, batch: List[T]) -> List[T]:
        """Filter batch based on predicate"""
        return [item for item in batch if predicate(item)]
    
    @staticmethod
    def reduce_batch(func: Callable, batch: List[T], initial: U = None) -> U:
        """Reduce batch using function"""
        if initial is None:
            return reduce(func, batch)
        return reduce(func, batch, initial)
```

#### TransformPipeline

```python
class TransformPipeline:
    """Functional transform pipeline for data processing"""
    
    def __init__(self):
        self.transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable) -> 'TransformPipeline':
        """Add transform to pipeline"""
        self.transforms.append(transform)
        return self
    
    def compose(self) -> Callable:
        """Compose all transforms into single function"""
        if not self.transforms:
            return lambda x: x
        
        def composed_transform(x):
            for transform in self.transforms:
                x = transform(x)
            return x
        
        return composed_transform
    
    def apply(self, data: T) -> U:
        """Apply pipeline to data"""
        return self.compose()(data)
    
    def apply_batch(self, batch: List[T]) -> List[U]:
        """Apply pipeline to batch of data"""
        return [self.apply(item) for item in batch]
```

#### FunctionalDataset

```python
class FunctionalDataset(Dataset):
    """Functional programming approach to dataset creation"""
    
    def __init__(self, data: List[T], transform_pipeline: Optional[Callable] = None):
        self.data = data
        self.transform_pipeline = transform_pipeline or (lambda x: x)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> U:
        item = self.data[idx]
        return self.transform_pipeline(item)
    
    def map(self, func: Callable) -> 'FunctionalDataset':
        """Apply function to all data items"""
        new_data = [func(item) for item in self.data]
        return FunctionalDataset(new_data, self.transform_pipeline)
    
    def filter(self, predicate: Callable) -> 'FunctionalDataset':
        """Filter data based on predicate"""
        new_data = [item for item in self.data if predicate(item)]
        return FunctionalDataset(new_data, self.transform_pipeline)
    
    def batch(self, batch_size: int) -> List[List[T]]:
        """Create batches from data"""
        return [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]
```

## Design Patterns Implementados

### Factory Pattern

```python
class ModelFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def create_model(model_type: ModelType, config: Dict[str, Any]) -> BaseModel:
        """Create model instance based on type"""
        if model_type == ModelType.TRANSFORMER:
            return TransformerModel(config)
        elif model_type == ModelType.CNN:
            return CNNModel(config)
        elif model_type == ModelType.RNN:
            return RNNModel(config)
        elif model_type == ModelType.DIFFUSION:
            return DiffusionModel(config)
        elif model_type == ModelType.HYBRID:
            return HybridModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Usage
transformer_model = ModelFactory.create_model(ModelType.TRANSFORMER, config)
```

### Builder Pattern

```python
class ModelConfigBuilder:
    """Builder pattern for complex model configurations"""
    
    def __init__(self):
        self.config = {}
    
    def set_model_type(self, model_type: ModelType) -> 'ModelConfigBuilder':
        """Set model type"""
        self.config['model_type'] = model_type
        return self
    
    def set_transformer_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set Transformer-specific configuration"""
        if 'transformer_config' not in self.config:
            self.config['transformer_config'] = {}
        self.config['transformer_config'].update(kwargs)
        return self
    
    def set_cnn_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set CNN-specific configuration"""
        if 'cnn_config' not in self.config:
            self.config['cnn_config'] = {}
        self.config['cnn_config'].update(kwargs)
        return self
    
    def set_optimization_config(self, **kwargs) -> 'ModelConfigBuilder':
        """Set optimization configuration"""
        self.config.update(kwargs)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build final configuration"""
        return self.config.copy()

# Usage
hybrid_config = (ModelConfigBuilder()
                .set_model_type(ModelType.HYBRID)
                .set_transformer_config(vocab_size=30000, d_model=256, n_heads=4, n_layers=3)
                .set_cnn_config(input_channels=3, num_classes=1000, base_channels=32, num_layers=3)
                .set_optimization_config(enable_gradient_checkpointing=True)
                .build())
```

## Functional Data Transforms

### Text Processing

```python
def text_tokenize(text: str, tokenizer: Callable) -> List[int]:
    """Functional text tokenization"""
    return tokenizer(text)

def text_pad(tokens: List[int], max_length: int, pad_token: int = 0) -> List[int]:
    """Functional text padding"""
    if len(tokens) >= max_length:
        return tokens[:max_length]
    return tokens + [pad_token] * (max_length - len(tokens))

def text_truncate(tokens: List[int], max_length: int) -> List[int]:
    """Functional text truncation"""
    return tokens[:max_length]

# Usage
text_pipeline = TransformPipeline()
text_pipeline.add_transform(lambda x: x.lower())
text_pipeline.add_transform(lambda x: x.strip())
text_pipeline.add_transform(lambda x: ' '.join(x.split()))

processed_text = text_pipeline.apply("  Hello World  ")
# Result: "hello world"
```

### Image Processing

```python
def image_resize(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Functional image resizing"""
    return F.interpolate(image.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

def image_normalize(image: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Functional image normalization"""
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor

def image_augment(image: torch.Tensor, augmentation_type: str) -> torch.Tensor:
    """Functional image augmentation"""
    if augmentation_type == "horizontal_flip" and torch.rand(1) > 0.5:
        return torch.flip(image, [-1])
    elif augmentation_type == "vertical_flip" and torch.rand(1) > 0.5:
        return torch.flip(image, [-2])
    elif augmentation_type == "rotation" and torch.rand(1) > 0.5:
        angle = torch.rand(1) * 30 - 15
        return F.rotate(image, angle.item())
    
    return image

# Usage
image_pipeline = TransformPipeline()
image_pipeline.add_transform(lambda x: image_resize(x, (224, 224)))
image_pipeline.add_transform(lambda x: image_normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
image_pipeline.add_transform(lambda x: image_augment(x, "horizontal_flip"))

processed_image = image_pipeline.apply(input_image)
```

## Advanced Usage Examples

### Complex Model Configuration

```python
# Create complex hybrid model configuration
complex_config = (ModelConfigBuilder()
                .set_model_type(ModelType.HYBRID)
                .set_transformer_config(
                    vocab_size=50000,
                    d_model=768,
                    n_heads=12,
                    n_layers=12,
                    d_ff=3072,
                    max_seq_len=1024,
                    dropout=0.1
                )
                .set_cnn_config(
                    input_channels=3,
                    num_classes=1000,
                    base_channels=64,
                    num_layers=6
                )
                .set_rnn_config(
                    input_size=768,
                    hidden_size=512,
                    num_layers=3,
                    num_classes=1000,
                    dropout=0.1
                )
                .set_optimization_config(
                    enable_gradient_checkpointing=True,
                    enable_mixed_precision=True,
                    fusion_dim=1024
                )
                .build())

# Create model
hybrid_model = ModelFactory.create_model(ModelType.HYBRID, complex_config)
```

### Functional Data Processing Pipeline

```python
# Create comprehensive text processing pipeline
def create_advanced_text_pipeline() -> TransformPipeline:
    pipeline = TransformPipeline()
    
    # Basic preprocessing
    pipeline.add_transform(lambda x: x.lower())
    pipeline.add_transform(lambda x: x.strip())
    pipeline.add_transform(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation
    
    # Tokenization
    pipeline.add_transform(lambda x: x.split())
    pipeline.add_transform(lambda x: [word for word in x if len(word) > 2])  # Filter short words
    
    # Normalization
    pipeline.add_transform(lambda x: ' '.join(x))
    pipeline.add_transform(lambda x: x[:1000])  # Truncate to 1000 characters
    
    return pipeline

# Create image processing pipeline with augmentation
def create_advanced_image_pipeline() -> TransformPipeline:
    pipeline = TransformPipeline()
    
    # Resize
    pipeline.add_transform(lambda x: image_resize(x, (256, 256)))
    
    # Augmentation
    pipeline.add_transform(lambda x: image_augment(x, "horizontal_flip"))
    pipeline.add_transform(lambda x: image_augment(x, "rotation"))
    
    # Normalization
    pipeline.add_transform(lambda x: image_normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    # Final resize
    pipeline.add_transform(lambda x: image_resize(x, (224, 224)))
    
    return pipeline

# Apply pipelines
text_pipeline = create_advanced_text_pipeline()
image_pipeline = create_advanced_image_pipeline()

# Process data
processed_texts = [text_pipeline.apply(text) for text in text_data]
processed_images = [image_pipeline.apply(image) for image in image_data]
```

### Functional Dataset Operations

```python
# Create functional dataset
dataset = FunctionalDataset(raw_data, text_pipeline.compose())

# Apply functional operations
filtered_dataset = dataset.filter(lambda x: len(x) > 10)  # Filter long texts
mapped_dataset = dataset.map(lambda x: f"Processed: {x}")  # Add prefix
batched_dataset = dataset.batch(32)  # Create batches

# Chain operations
final_dataset = (dataset
                .filter(lambda x: len(x) > 10)
                .map(lambda x: x.upper())
                .map(lambda x: f"Processed: {x}"))

# Create DataLoader
dataloader = DataLoader(
    final_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```

### Model Checkpointing and Loading

```python
# Save model checkpoint
checkpoint_path = "model_checkpoint.pt"
model.save_checkpoint(checkpoint_path, {
    "training_epoch": 100,
    "validation_loss": 0.045,
    "learning_rate": 0.001,
    "optimizer_state": optimizer.state_dict()
})

# Load model checkpoint
loaded_model = ModelFactory.create_model(ModelType.TRANSFORMER, config)
checkpoint_info = loaded_model.load_checkpoint(checkpoint_path)

print(f"Model loaded from epoch {checkpoint_info['current_epoch']}")
print(f"Training steps: {checkpoint_info['total_steps']}")
print(f"Additional info: {checkpoint_info['additional_info']}")
```

## Estructura del Proyecto

```
oop_fp_system.py     # Sistema principal
requirements_oop_fp.txt  # Dependencias
README_OOP_FP.md     # Documentación
```

## Dependencias Principales

- **torch**: Framework de deep learning
- **numpy**: Computación numérica
- **PyYAML**: Procesamiento de archivos YAML

## Requisitos del Sistema

- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **CUDA**: Recomendado para modelos grandes

## Solución de Problemas

### Error de Importación

```python
# Verificar instalación de dependencias
try:
    import torch
    import numpy as np
    import yaml
    print("Todas las dependencias están instaladas")
except ImportError as e:
    print(f"Error de importación: {e}")
    print("Instalar dependencias con: pip install -r requirements_oop_fp.txt")
```

### Error de Configuración

```python
# Verificar configuración del modelo
def validate_config(config: Dict[str, Any]) -> bool:
    required_keys = ['model_type', 'vocab_size', 'd_model']
    
    for key in required_keys:
        if key not in config:
            print(f"Configuración incompleta: falta {key}")
            return False
    
    return True

# Validar antes de crear modelo
if validate_config(config):
    model = ModelFactory.create_model(ModelType.TRANSFORMER, config)
else:
    print("Configuración inválida")
```

### Error de Memoria

```python
# Habilitar optimizaciones de memoria
config = {
    'enable_gradient_checkpointing': True,
    'enable_mixed_precision': True,
    'd_model': 256,  # Reducir tamaño del modelo
    'n_layers': 4,   # Reducir número de capas
}

# Limpiar memoria GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
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


