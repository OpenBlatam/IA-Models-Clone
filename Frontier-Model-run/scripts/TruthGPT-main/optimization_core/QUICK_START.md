# 🚀 Quick Start Guide - TruthGPT Optimization Core

## Installation

```bash
# Install all dependencies
pip install -r modules/requirements.txt
pip install -r modules/ultra_advanced/requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 1. Basic Usage

### Create a Simple Model
```python
import torch
import torch.nn as nn
from modules.advanced_libraries import create_model_module, create_model_config

# Create configuration
config = create_model_config(
    model_type="transformer",
    model_name="bert-base-uncased",
    add_classification_head=True,
    num_labels=2
)

# Create model
model = create_model_module("transformer", config.__dict__)
print("Model created successfully!")
```

### Setup Data Processing
```python
from modules.data.data_processor import create_data_module, create_data_config

config = create_data_config(
    data_type="text",
    dataset_name="imdb",
    batch_size=32,
    max_length=512
)

data_module = create_data_module("text", config.__dict__)
print("Data processing configured!")
```

## 2. GPU Acceleration

```python
from modules.feed_forward.ultra_optimization.gpu_accelerator import (
    GPUAccelerator, GPUAcceleratorConfig
)

# Configure GPU accelerator
config = GPUAcceleratorConfig(
    device_id=0,
    use_mixed_precision=True,
    enable_tensor_cores=True
)

# Create accelerator
accelerator = GPUAccelerator(config)

# Use with your model
optimized_model = accelerator.optimize(model)
```

## 3. Quantum Machine Learning

```python
from modules.ultra_advanced.quantum_ml import (
    create_quantum_neural_network, create_quantum_config
)

# Create quantum configuration
config = create_quantum_config(
    n_qubits=4,
    n_layers=2,
    optimizer_type=QuantumOptimizerType.VQE
)

# Create quantum neural network
qnn = create_quantum_neural_network(config)

# Forward pass
x = torch.randn(2, 4)
output = qnn(x)
print(f"Quantum output shape: {output.shape}")
```

## 4. Federated Learning

```python
from modules.ultra_advanced.federated_learning import (
    create_federated_server, create_federated_config
)

# Create federated configuration
config = create_federated_config(
    strategy=FederatedStrategy.FEDAVG,
    num_clients=5,
    num_rounds=10,
    use_differential_privacy=True
)

# Create federated server
server = create_federated_server(global_model, config)

# Run federation
results = server.run_federation(client_data_loaders)
print(f"Federated training completed: {results}")
```

## 5. Differential Privacy

```python
from modules.ultra_advanced.differential_privacy import (
    create_private_model, create_privacy_config
)

# Create privacy configuration
config = create_privacy_config(
    epsilon=1.0,
    delta=1e-5,
    mechanism=PrivacyMechanism.GAUSSIAN,
    l2_norm_clip=1.0
)

# Create private model
private_model = create_private_model(base_model, config)

# Private training
metrics = private_model.private_training_step(batch)
print(f"Private training metrics: {metrics}")
```

## 6. Graph Neural Networks

```python
from modules.ultra_advanced.graph_neural_networks import (
    create_gnn, create_gnn_config
)

# Create GNN configuration
config = create_gnn_config(
    gnn_type=GNNType.GAT,
    input_dim=64,
    hidden_dim=128,
    output_dim=32,
    num_heads=8,
    num_layers=3
)

# Create GNN
gnn = create_gnn(config)

# Forward pass
x = torch.randn(2, 10, 64)  # [batch, nodes, features]
adj = torch.randn(2, 10, 10)  # [batch, nodes, nodes]
adj = (adj > 0.5).float()  # Binary adjacency

output = gnn(x, adj)
print(f"GNN output shape: {output.shape}")
```

## 7. Complete Training Pipeline

```python
from modules.training.trainer import create_trainer, create_training_config
from modules.optimization.optimizer import create_optimizer, create_optimization_config

# Create training configuration
train_config = create_training_config(
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    use_mixed_precision=True,
    use_wandb=True,
    early_stopping=True
)

# Create trainer
trainer = create_trainer(train_config, model, train_dataloader, val_dataloader)

# Train
history = trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(f"Training completed! Final metrics: {metrics}")
```

## 8. Configuration File (YAML)

Create `config.yaml`:
```yaml
model:
  type: transformer
  model_name: bert-base-uncased
  add_classification_head: true
  num_labels: 2

data:
  type: text
  dataset_name: imdb
  batch_size: 32
  max_length: 512

training:
  epochs: 10
  learning_rate: 1e-4
  use_mixed_precision: true
  use_wandb: true
  early_stopping: true

optimization:
  optimizer: adamw
  scheduler: cosine
  gradient_clip_norm: 1.0

gpu_accelerator:
  device_id: 0
  use_mixed_precision: true
  enable_tensor_cores: true

quantum:
  n_qubits: 4
  n_layers: 2
  optimizer_type: vqe

federated:
  strategy: fedavg
  num_clients: 5
  num_rounds: 10

privacy:
  epsilon: 1.0
  delta: 1e-5
  mechanism: gaussian

gnn:
  gnn_type: gat
  input_dim: 64
  hidden_dim: 128
```

## 9. Run Complete System

```python
from modules.advanced_libraries import ModularSystem

# Load configuration and create system
system = ModularSystem("config.yaml")

# Train
system.train(epochs=10)

# Evaluate
metrics = system.evaluate()
print(f"Metrics: {metrics}")

# Predict
predictions = system.predict(input_data)
print(f"Predictions: {predictions}")

# Monitor
monitoring_data = {"inference_time": 0.5}
results = system.monitor(monitoring_data)
print(f"Monitoring results: {results}")
```

## 10. Examples

Run the provided examples:

```bash
# Basic modular system
python modules/advanced_libraries.py

# GPU acceleration
python modules/feed_forward/ultra_optimization/gpu_accelerator.py

# Quantum ML
python modules/ultra_advanced/quantum_ml.py

# Federated learning
python modules/ultra_advanced/federated_learning.py

# Differential privacy
python modules/ultra_advanced/differential_privacy.py

# Graph neural networks
python modules/ultra_advanced/graph_neural_networks.py
```

## 📊 Monitoring & Logging

### Wandb
```python
import wandb

wandb.init(project="truthgpt-optimization")
wandb.watch(model)
wandb.log({"loss": loss.item()})
```

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
writer.add_scalar("Loss/Train", loss, epoch)
```

### MLflow
```python
import mlflow

mlflow.start_run()
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("loss", loss.item())
```

## 🎯 Best Practices

1. **Start Simple**: Begin with basic configurations
2. **Add Complexity**: Gradually add advanced features
3. **Monitor**: Track metrics and performance
4. **Optimize**: Use GPU acceleration and mixed precision
5. **Experiment**: Try different architectures and hyperparameters
6. **Deploy**: Use production-ready features for deployment

## 🚀 Next Steps

1. Read the comprehensive documentation
2. Explore the examples in each module
3. Configure your specific use case
4. Run experiments and track results
5. Deploy your optimized models

## 📚 Documentation

- **System Complete**: `modules/SYSTEM_COMPLETE.md`
- **Architecture Guide**: `modules/README.md`
- **Ultra-Advanced**: `modules/ultra_advanced/README.md`
- **GPU Acceleration**: See `gpu_accelerator.py`

## 🎉 You're Ready!

You now have a complete, production-ready deep learning system with cutting-edge capabilities. Start experimenting and building amazing AI models! 🚀
