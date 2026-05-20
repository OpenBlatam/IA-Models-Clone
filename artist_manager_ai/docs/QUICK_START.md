# Quick Start Guide - Artist Manager AI ML

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Create a Model

```python
from ml.factories import ModelFactory

factory = ModelFactory()
model = factory.create("event_duration", config={"input_dim": 32})
```

#### 2. Create Dataset

```python
from ml.factories import DataFactory

factory = DataFactory()
dataset = factory.create_dataset("event", events_data)
train_loader, val_loader, test_loader = factory.create_dataloaders(dataset)
```

#### 3. Train Model

```python
from ml.factories import TrainerFactory
import torch.nn as nn

trainer = TrainerFactory().create(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss()
)

history = trainer.train(num_epochs=100)
```

#### 4. Make Predictions

```python
from ml.prediction_service import PredictionService

service = PredictionService()
prediction = service.predict_event_duration(
    event_type="concert",
    historical_events=[],
    event_data=event_data
)
```

## 📚 Examples

See `examples/` directory for complete examples:
- `train_example.py`: Complete training pipeline
- `inference_example.py`: Inference examples

## 🧪 Testing

```bash
pytest tests/ml/
```

## 📖 Documentation

- `MODULAR_ARCHITECTURE.md`: Architecture overview
- `DEEP_LEARNING_IMPROVEMENTS.md`: Deep learning features
- `QUICK_START.md`: This guide




