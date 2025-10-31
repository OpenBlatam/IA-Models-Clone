# TruthGPT API Documentation

## Overview

TruthGPT API provides a TensorFlow-like interface for building, training, and optimizing neural networks using the TruthGPT framework. It offers familiar APIs that make it easy to transition from TensorFlow/Keras to TruthGPT.

## Features

- **TensorFlow-like API**: Familiar interface for TensorFlow/Keras users
- **PyTorch Backend**: Built on PyTorch for high performance
- **Modular Design**: Easy to extend and customize
- **Comprehensive Layers**: Dense, Conv2D, LSTM, GRU, and more
- **Multiple Optimizers**: Adam, SGD, RMSprop, and others
- **Various Loss Functions**: Crossentropy, MSE, MAE, and more
- **Easy Model Saving/Loading**: Simple model persistence

## Installation

```bash
# Install required dependencies
pip install torch torchvision numpy

# Add TruthGPT API to your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/truthgpt_api"
```

## Quick Start

### Basic Usage

```python
import truthgpt as tg

# Create a simple model
model = tg.Sequential([
    tg.layers.Dense(128, activation='relu'),
    tg.layers.Dropout(0.2),
    tg.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tg.optimizers.Adam(learning_rate=0.001),
    loss=tg.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)
```

### Advanced Usage

```python
import truthgpt as tg

# Create a CNN model
model = tg.Sequential([
    tg.layers.Conv2D(32, 3, activation='relu'),
    tg.layers.MaxPooling2D(2),
    tg.layers.Conv2D(64, 3, activation='relu'),
    tg.layers.MaxPooling2D(2),
    tg.layers.Flatten(),
    tg.layers.Dense(64, activation='relu'),
    tg.layers.Dropout(0.5),
    tg.layers.Dense(10, activation='softmax')
])

# Compile with different optimizer
model.compile(
    optimizer=tg.optimizers.Adam(learning_rate=0.001),
    loss=tg.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train with validation data
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)

# Save and load model
model.save('my_model.pth')
loaded_model = tg.load_model('my_model.pth', model_class=tg.Sequential)
```

## API Reference

### Models

#### Sequential
```python
tg.Sequential(layers=None, name=None)
```
Stack layers in a linear fashion.

#### Functional
```python
tg.Functional(inputs, outputs, name=None)
```
Create complex architectures with multiple inputs/outputs.

### Layers

#### Dense
```python
tg.layers.Dense(units, activation=None, use_bias=True)
```
Fully connected layer.

#### Conv2D
```python
tg.layers.Conv2D(filters, kernel_size, strides=(1,1), padding='valid')
```
2D convolutional layer.

#### LSTM
```python
tg.layers.LSTM(units, return_sequences=False)
```
LSTM layer for sequence processing.

#### Dropout
```python
tg.layers.Dropout(rate)
```
Dropout layer for regularization.

### Optimizers

#### Adam
```python
tg.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```
Adam optimizer.

#### SGD
```python
tg.optimizers.SGD(learning_rate=0.01, momentum=0.0)
```
Stochastic gradient descent optimizer.

### Loss Functions

#### SparseCategoricalCrossentropy
```python
tg.losses.SparseCategoricalCrossentropy(from_logits=False)
```
Sparse categorical crossentropy loss.

#### MeanSquaredError
```python
tg.losses.MeanSquaredError()
```
Mean squared error loss.

### Utilities

#### Data Utils
```python
tg.to_categorical(y, num_classes=None)
tg.normalize(x, axis=-1, order=2)
tg.get_data(filepath, test_split=0.2)
```

#### Model Utils
```python
tg.save_model(model, filepath)
tg.load_model(filepath, model_class=None)
```

## Examples

### Basic Classification
```python
import truthgpt as tg
import numpy as np

# Generate dummy data
x_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 3, 1000)

# Create model
model = tg.Sequential([
    tg.layers.Dense(128, activation='relu'),
    tg.layers.Dropout(0.2),
    tg.layers.Dense(3, activation='softmax')
])

# Compile and train
model.compile(
    optimizer=tg.optimizers.Adam(learning_rate=0.001),
    loss=tg.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### Image Classification
```python
import truthgpt as tg
import numpy as np

# Generate dummy image data
x_train = np.random.randn(1000, 32, 32, 3)
y_train = np.random.randint(0, 10, 1000)

# Create CNN model
model = tg.Sequential([
    tg.layers.Conv2D(32, 3, activation='relu'),
    tg.layers.MaxPooling2D(2),
    tg.layers.Conv2D(64, 3, activation='relu'),
    tg.layers.MaxPooling2D(2),
    tg.layers.Flatten(),
    tg.layers.Dense(64, activation='relu'),
    tg.layers.Dropout(0.5),
    tg.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(
    optimizer=tg.optimizers.Adam(learning_rate=0.001),
    loss=tg.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Experiment with different batch sizes for optimal performance
3. **Learning Rate**: Use learning rate scheduling for better convergence
4. **Regularization**: Use dropout and batch normalization for better generalization

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure TruthGPT API is in your Python path
2. **CUDA Errors**: Check if CUDA is properly installed
3. **Memory Issues**: Reduce batch size or use gradient checkpointing

### Getting Help

- Check the examples in the `examples/` directory
- Review the API reference above
- Look at the source code for detailed implementation

## Contributing

We welcome contributions! Please see the main TruthGPT repository for contribution guidelines.

## License

This project is licensed under the same license as the main TruthGPT project.









