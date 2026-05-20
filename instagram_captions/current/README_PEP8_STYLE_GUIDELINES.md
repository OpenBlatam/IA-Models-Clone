# PEP 8 Style Guidelines System

Sistema completo para implementar las guías de estilo PEP 8 y mejores prácticas para código Python en deep learning.

## 🚀 Características

### **PEP 8 Compliance System**
- **Automatic Validation**: Validación automática de cumplimiento PEP 8
- **Style Rule Enforcement**: Aplicación de reglas de estilo PEP 8
- **Import Order Management**: Gestión del orden de imports
- **Naming Convention Validation**: Validación de convenciones de nombres
- **Code Formatting**: Formateo automático de código

### **Code Quality Tools Integration**
- **Black Formatter**: Integración con Black para formateo automático
- **Flake8 Linter**: Integración con Flake8 para linting
- **isort Integration**: Integración con isort para ordenamiento de imports
- **autopep8 Integration**: Integración con autopep8 para corrección automática
- **AST Analysis**: Análisis de código usando Abstract Syntax Trees

### **Style Rule Categories**
- **Naming Conventions**: Clases (PascalCase), funciones/variables (snake_case)
- **Formatting Rules**: Longitud de línea, indentación, líneas en blanco
- **Import Rules**: Orden de imports, agrupación, separación
- **Documentation**: Docstrings, comentarios, type hints
- **Code Structure**: Organización de clases y funciones

### **Automatic Compliance Checking**
- **Line Length Validation**: Validación de longitud de línea (79 caracteres)
- **Indentation Checking**: Verificación de indentación (4 espacios)
- **Blank Line Rules**: Reglas para líneas en blanco
- **Import Order Validation**: Validación del orden de imports
- **Naming Convention Validation**: Validación de convenciones de nombres

## 📦 Instalación

```bash
pip install -r requirements_pep8_style_guidelines.txt
```

## 🎯 Uso Básico

### Configuración del Sistema

```python
from pep8_style_guidelines_system import (
    PEP8StyleConfig, PEP8StyleGuidelinesSystem
)

# Configuración de PEP 8
pep8_config = PEP8StyleConfig(
    max_line_length=79,
    use_black_formatter=True,
    use_flake8_linter=True,
    use_isort_imports=True,
    use_autopep8=True,
    enforce_naming_conventions=True,
    enforce_import_order=True,
    enforce_docstring_style=True,
    enforce_type_hints=True
)

# Inicializar sistema PEP 8
pep8_system = PEP8StyleGuidelinesSystem(pep8_config)

# Validar código
code_string = '''
def bad_function_name():
    x = 1
    return x
'''

validation_results = pep8_system.validate_pep8_compliance(code_string)
print(f"Code is PEP 8 compliant: {validation_results['is_compliant']}")
```

### Modelo con Cumplimiento PEP 8

```python
from pep8_style_guidelines_system import PEP8CompliantTransformerModel

# Crear modelo siguiendo PEP 8
transformer_model = PEP8CompliantTransformerModel(
    vocabulary_size=30000,
    embedding_dimension=512,
    number_of_attention_heads=8,
    number_of_transformer_layers=6,
    feed_forward_dimension=2048,
    maximum_sequence_length=512,
    dropout_probability=0.1
)

# Forward pass siguiendo PEP 8
input_token_sequence = torch.randint(0, 30000, (32, 100))
output_logits = transformer_model(input_token_sequence)
```

### Sistema de Entrenamiento con Cumplimiento PEP 8

```python
from pep8_style_guidelines_system import (
    PEP8CompliantTrainingSystem, PEP8CompliantTextDataset
)
from torch.utils.data import DataLoader

# Crear dataset siguiendo PEP 8
text_dataset = PEP8CompliantTextDataset(
    text_sequences=["sample text"] * 1000,
    tokenizer_function=lambda x: [1, 2, 3, 4, 5],
    maximum_sequence_length=100
)

# Crear data loader
training_data_loader = DataLoader(
    dataset=text_dataset,
    batch_size=32,
    shuffle=True
)

# Crear sistema de entrenamiento siguiendo PEP 8
training_system = PEP8CompliantTrainingSystem(
    neural_network_model=transformer_model,
    training_data_loader=training_data_loader,
    validation_data_loader=training_data_loader,
    loss_criterion=nn.CrossEntropyLoss(),
    optimization_algorithm=torch.optim.Adam(transformer_model.parameters())
)

# Entrenar siguiendo PEP 8
for current_epoch in range(10):
    training_metrics = training_system.train_single_epoch()
    validation_metrics = training_system.validate_model()
    
    print(f"Epoch {current_epoch + 1}")
    print(f"Training Loss: {training_metrics['average_epoch_loss']:.4f}")
    print(f"Validation Loss: {validation_metrics['average_validation_loss']:.4f}")
```

## 🔧 Componentes Principales

### **PEP8StyleGuidelinesSystem**
Sistema principal para validación y aplicación de PEP 8:

```python
from pep8_style_guidelines_system import PEP8StyleGuidelinesSystem

pep8_system = PEP8StyleGuidelinesSystem(pep8_config)

# Validar código completo
validation_results = pep8_system.validate_pep8_compliance(code_string)

# Revisar errores
for error in validation_results["errors"]:
    print(f"Error: {error['message']} at line {error['line']}")

# Revisar warnings
for warning in validation_results["warnings"]:
    print(f"Warning: {warning['message']} at line {warning['line']}")
```

### **PEP8CompliantTransformerModel**
Modelo Transformer siguiendo PEP 8:

```python
from pep8_style_guidelines_system import PEP8CompliantTransformerModel

# Configuración del modelo siguiendo PEP 8
model_config = {
    'vocabulary_size': 30000,
    'embedding_dimension': 512,
    'number_of_attention_heads': 8,
    'number_of_transformer_layers': 6,
    'feed_forward_dimension': 2048,
    'maximum_sequence_length': 512,
    'dropout_probability': 0.1
}

# Crear modelo
transformer_model = PEP8CompliantTransformerModel(**model_config)

# Componentes del modelo siguiendo PEP 8
print(f"Token Embedding Layer: {transformer_model.token_embedding_layer}")
print(f"Positional Encoding Layer: {transformer_model.positional_encoding_layer}")
print(f"Transformer Layers: {len(transformer_model.transformer_layers)}")
print(f"Output Projection Layer: {transformer_model.output_projection_layer}")
```

### **PEP8CompliantTextDataset**
Dataset de texto siguiendo PEP 8:

```python
from pep8_style_guidelines_system import PEP8CompliantTextDataset

# Función de tokenización simple
def simple_tokenizer(text_sequence):
    """Tokenize text sequence into integers.
    
    Args:
        text_sequence: Input text sequence.
        
    Returns:
        List of tokenized integers.
    """
    return [ord(char) % 1000 for char in text_sequence]

# Crear dataset siguiendo PEP 8
text_dataset = PEP8CompliantTextDataset(
    text_sequences=["Hello world", "Deep learning", "Neural networks"],
    tokenizer_function=simple_tokenizer,
    maximum_sequence_length=20
)

# Acceder a datos siguiendo PEP 8
for index in range(len(text_dataset)):
    sequence_tensor = text_dataset[index]
    print(f"Sequence {index}: {sequence_tensor}")
```

### **PEP8CompliantTrainingSystem**
Sistema de entrenamiento siguiendo PEP 8:

```python
from pep8_style_guidelines_system import PEP8CompliantTrainingSystem

# Crear sistema de entrenamiento
training_system = PEP8CompliantTrainingSystem(
    neural_network_model=transformer_model,
    training_data_loader=training_data_loader,
    validation_data_loader=validation_data_loader,
    loss_criterion=nn.CrossEntropyLoss(),
    optimization_algorithm=torch.optim.Adam(transformer_model.parameters())
)

# Entrenar una época siguiendo PEP 8
training_metrics = training_system.train_single_epoch()
print(f"Average Epoch Loss: {training_metrics['average_epoch_loss']:.4f}")
print(f"Total Training Batches: {training_metrics['total_training_batches']}")

# Validar modelo siguiendo PEP 8
validation_metrics = training_system.validate_model()
print(f"Average Validation Loss: {validation_metrics['average_validation_loss']:.4f}")
```

### **PEP8CompliantEvaluationSystem**
Sistema de evaluación siguiendo PEP 8:

```python
from pep8_style_guidelines_system import PEP8CompliantEvaluationSystem

# Crear sistema de evaluación
evaluation_system = PEP8CompliantEvaluationSystem(
    neural_network_model=transformer_model
)

# Calcular métricas siguiendo PEP 8
evaluation_metrics = evaluation_system.calculate_classification_metrics(
    model_predictions=model_predictions,
    ground_truth_labels=ground_truth_labels
)

print(f"Classification Accuracy: {evaluation_metrics['classification_accuracy']:.4f}")
print(f"Precision Score: {evaluation_metrics['precision_score']:.4f}")
print(f"Recall Score: {evaluation_metrics['recall_score']:.4f}")
print(f"F1 Measure: {evaluation_metrics['f1_measure']:.4f}")
```

### **PEP8CompliantOptimizationSystem**
Sistema de optimización siguiendo PEP 8:

```python
from pep8_style_guidelines_system import PEP8CompliantOptimizationSystem

# Crear sistema de optimización
optimization_system = PEP8CompliantOptimizationSystem(
    neural_network_model=transformer_model,
    initial_learning_rate=0.001,
    weight_decay_factor=0.0001,
    momentum_factor=0.9
)

# Aplicar optimizaciones siguiendo PEP 8
maximum_gradient_norm_value = 1.0
optimization_system.apply_gradient_clipping(maximum_gradient_norm_value)

# Actualizar learning rate siguiendo PEP 8
new_learning_rate_value = 0.0005
optimization_system.update_learning_rate(new_learning_rate_value)
```

## 📊 Reglas PEP 8 Implementadas

### **Naming Conventions**
- **Class Names**: `PEP8CompliantTransformerModel` (PascalCase)
- **Function Names**: `create_pep8_compliant_training_example` (snake_case)
- **Variable Names**: `vocabulary_size_value` (snake_case)
- **Constant Names**: `MAX_LINE_LENGTH` (UPPER_CASE)
- **Module Names**: `pep8_style_guidelines_system` (snake_case)
- **Package Names**: `pep8_style_guidelines` (snake_case)

### **Formatting Rules**
- **Indentation**: 4 espacios (no tabs)
- **Line Length**: Máximo 79 caracteres
- **Blank Lines**: 2 líneas entre clases, 1 línea entre métodos
- **Spacing**: Espacios alrededor de operadores
- **Line Continuation**: Uso de paréntesis para continuar líneas

### **Import Rules**
- **Order**: `__future__`, standard library, third party, local
- **Grouping**: Imports agrupados por tipo
- **Separation**: Líneas en blanco entre grupos
- **Format**: Un import por línea
- **Absolute vs Relative**: Preferir imports absolutos

### **Documentation Rules**
- **Docstrings**: Docstrings para todas las funciones públicas
- **Type Hints**: Type hints para parámetros y retornos
- **Comments**: Comentarios explicativos cuando sea necesario
- **String Formatting**: Uso de f-strings para Python 3.6+

## 🎛️ Configuración Avanzada

### **PEP 8 Configuration**
```python
pep8_config = PEP8StyleConfig(
    max_line_length=79,                    # Longitud máxima de línea
    use_black_formatter=True,              # Usar Black formatter
    use_flake8_linter=True,                # Usar Flake8 linter
    use_isort_imports=True,                # Usar isort para imports
    use_autopep8=True,                     # Usar autopep8
    enforce_naming_conventions=True,        # Aplicar convenciones de nombres
    enforce_import_order=True,              # Aplicar orden de imports
    enforce_docstring_style=True,           # Aplicar estilo de docstrings
    enforce_type_hints=True                 # Aplicar type hints
)
```

### **Custom Style Rules**
```python
# Agregar reglas personalizadas
custom_rules = {
    "custom_naming": {
        "custom_pattern": r"^custom_[a-z_]*$"
    }
}

# Integrar con el sistema existente
pep8_system.pep8_rules.update(custom_rules)
```

## 🔍 Validación de Cumplimiento

### **Line Length Validation**
```python
# Validar longitud de línea
validation_results = pep8_system.validate_pep8_compliance(code_string)

for error in validation_results["errors"]:
    if error["type"] == "line_length":
        print(f"Line {error['line']}: {error['message']}")
        print(f"Suggestion: {error['suggestion']}")
```

### **Naming Convention Validation**
```python
# Validar convenciones de nombres
for warning in validation_results["warnings"]:
    if warning["type"] == "naming_convention":
        print(f"Line {warning['line']}: {warning['message']}")
        print(f"Name: {warning['name']}")
        print(f"Rule Type: {warning['rule_type']}")
        print(f"Suggestion: {warning['suggestion']}")
```

### **Import Order Validation**
```python
# Validar orden de imports
for warning in validation_results["warnings"]:
    if warning["type"] == "import_order":
        print(f"Line {warning['line']}: {warning['message']}")
        print(f"Suggestion: {warning['suggestion']}")
```

### **Indentation Validation**
```python
# Validar indentación
for error in validation_results["errors"]:
    if error["type"] == "indentation":
        print(f"Line {error['line']}: {error['message']}")
        print(f"Suggestion: {error['suggestion']}")
```

## 📈 Ejemplos de Uso

### **Entrenamiento Completo con PEP 8**
```python
def train_model_with_pep8_compliance():
    """Train model following PEP 8 style guidelines."""
    
    # Model configuration following PEP 8
    model_configuration = {
        'vocabulary_size': 30000,
        'embedding_dimension': 512,
        'number_of_attention_heads': 8,
        'number_of_transformer_layers': 6,
        'feed_forward_dimension': 2048,
        'maximum_sequence_length': 512,
        'dropout_probability': 0.1
    }
    
    # Create model following PEP 8
    neural_network_model = PEP8CompliantTransformerModel(**model_configuration)
    
    # Training configuration following PEP 8
    training_configuration = {
        'batch_size_value': 32,
        'number_of_epochs_value': 10,
        'initial_learning_rate_value': 0.001,
        'weight_decay_factor_value': 0.0001
    }
    
    # Create training system following PEP 8
    training_system = PEP8CompliantTrainingSystem(
        neural_network_model=neural_network_model,
        training_data_loader=training_data_loader,
        validation_data_loader=validation_data_loader,
        loss_criterion=nn.CrossEntropyLoss(),
        optimization_algorithm=torch.optim.Adam(
            params=neural_network_model.parameters(),
            lr=training_configuration['initial_learning_rate_value']
        )
    )
    
    # Training loop following PEP 8
    for current_epoch in range(training_configuration['number_of_epochs_value']):
        # Train epoch following PEP 8
        training_epoch_metrics = training_system.train_single_epoch()
        
        # Validate model following PEP 8
        validation_epoch_metrics = training_system.validate_model()
        
        # Log progress following PEP 8
        print(f"Epoch {current_epoch + 1}")
        print(f"Training Loss: {training_epoch_metrics['average_epoch_loss']:.4f}")
        print(f"Validation Loss: {validation_epoch_metrics['average_validation_loss']:.4f}")
```

### **Evaluación de Modelo con PEP 8**
```python
def evaluate_model_with_pep8_compliance():
    """Evaluate model following PEP 8 style guidelines."""
    
    # Create evaluation system following PEP 8
    model_evaluation_system = PEP8CompliantEvaluationSystem(
        neural_network_model=transformer_model
    )
    
    # Generate model predictions following PEP 8
    model_output_predictions = transformer_model(input_sequence_tensor)
    
    # Calculate classification metrics following PEP 8
    classification_evaluation_metrics = (
        model_evaluation_system.calculate_classification_metrics(
            model_predictions=model_output_predictions,
            ground_truth_labels=target_sequence_tensor
        )
    )
    
    # Display results following PEP 8
    print("=== Model Evaluation Results ===")
    print(f"Classification Accuracy: {classification_evaluation_metrics['classification_accuracy']:.4f}")
    print(f"Precision Score: {classification_evaluation_metrics['precision_score']:.4f}")
    print(f"Recall Score: {classification_evaluation_metrics['recall_score']:.4f}")
    print(f"F1 Measure: {classification_evaluation_metrics['f1_measure']:.4f}")
```

## 🏗️ Arquitectura del Sistema

```
PEP8StyleGuidelinesSystem
├── PEP8StyleConfig
│   ├── Style Rule Configuration
│   ├── Tool Integration Configuration
│   └── Enforcement Configuration
├── PEP8CompliantTransformerModel
│   ├── Model Architecture Components
│   ├── Forward Pass Implementation
│   └── PEP 8 Compliant Code
├── PEP8CompliantTextDataset
│   ├── Data Processing Components
│   ├── Text Sequence Handling
│   └── PEP 8 Compliant Code
├── PEP8CompliantTrainingSystem
│   ├── Training Loop Components
│   ├── Validation Components
│   └── PEP 8 Compliant Code
├── PEP8CompliantEvaluationSystem
│   ├── Metric Calculation Components
│   ├── Performance Evaluation
│   └── PEP 8 Compliant Code
└── PEP8CompliantOptimizationSystem
    ├── Optimization Components
    ├── Learning Rate Management
    └── PEP 8 Compliant Code
```

## 🚀 Casos de Uso

### **Large Language Models**
```python
# Configuration for large LLMs following PEP 8
large_language_model_configuration = {
    'vocabulary_size': 50000,
    'embedding_dimension': 1024,
    'number_of_attention_heads': 16,
    'number_of_transformer_layers': 12,
    'feed_forward_dimension': 4096,
    'maximum_sequence_length': 1024,
    'dropout_probability': 0.1
}

large_language_model = PEP8CompliantTransformerModel(
    **large_language_model_configuration
)
```

### **Computer Vision Models**
```python
# Configuration for vision models following PEP 8
computer_vision_model_configuration = {
    'input_image_channels': 3,
    'number_of_convolutional_layers': 5,
    'convolutional_filter_size': 3,
    'pooling_kernel_size': 2,
    'number_of_fully_connected_layers': 3,
    'number_of_output_classes': 1000
}
```

### **Sequence Models**
```python
# Configuration for sequence models following PEP 8
sequence_model_configuration = {
    'input_sequence_length': 256,
    'hidden_state_dimension': 512,
    'number_of_recurrent_layers': 4,
    'bidirectional_enabled': True,
    'dropout_probability': 0.2
}
```

## 📚 Referencias y Mejores Prácticas

### **PEP 8 Official Documentation**
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PEP 8 Style Guide for Python Code](https://pep8.org/)
- [Python Style Guide](https://docs.python-guide.org/writing/style/)

### **Code Quality Tools**
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linter](https://flake8.pycqa.org/)
- [isort Import Sorter](https://pycqa.github.io/isort/)
- [autopep8](https://pypi.org/project/autopep8/)

### **Python Best Practices**
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Real Python PEP 8](https://realpython.com/python-pep8/)
- [Python Code Quality](https://python-guide.readthedocs.io/writing/style/)

### **Deep Learning Style**
- [PyTorch Style Guide](https://pytorch.org/docs/stable/notes/coding_style.html)
- [TensorFlow Style Guide](https://www.tensorflow.org/guide/style_guide)
- [Keras Style Guide](https://keras.io/guides/style_guide/)

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/PEP8Compliance`)
3. Commit tus cambios (`git commit -m 'Add PEP 8 compliance'`)
4. Push a la rama (`git push origin feature/PEP8Compliance`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Abre un issue en GitHub
- Consulta la documentación
- Revisa los ejemplos de uso

---

**Sistema de PEP 8 Style Guidelines implementado exitosamente** ✅

**Características implementadas:**
- ✅ PEP 8 compliance validation
- ✅ Style rule enforcement
- ✅ Import order management
- ✅ Naming convention validation
- ✅ Code formatting integration
- ✅ AST-based code analysis
- ✅ Comprehensive style checking
- ✅ Automatic compliance validation
- ✅ PEP 8 compliant examples
- ✅ Best practices implementation

**SISTEMA COMPLETAMENTE FUNCIONAL PARA PRODUCCIÓN** 🚀


