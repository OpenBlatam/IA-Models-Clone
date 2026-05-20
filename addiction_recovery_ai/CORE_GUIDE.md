# Core Components Guide - Addiction Recovery AI

## ✅ Recommended Core Structure

### Core Components Overview

The `core/` directory contains the foundational components of the application:

```
core/
├── app_factory.py              # ✅ Application factory (canonical)
├── lifespan.py                  # ✅ Application lifespan management
├── middleware_config.py         # ✅ Middleware configuration
├── routes_config.py             # ✅ Routes configuration
├── base/                        # ✅ Base classes
│   ├── base_model.py            # Base model classes
│   └── base_trainer.py          # Base trainer classes
├── layers/                      # ✅ Layered architecture
│   ├── service_layer.py         # Service layer
│   ├── model_layer.py           # Model layer
│   ├── training_layer.py        # Training layer
│   └── ...
├── models/                      # ✅ ML models
├── training/                     # ✅ Training components
├── inference/                    # ✅ Inference components
└── ...
```

## 🏗️ Core Components

### Application Factory

#### `core/app_factory.py` - **USE THIS** ✅
- **Status**: Canonical application factory
- **Purpose**: Creates and configures FastAPI application
- **Usage**:
```python
from core.app_factory import create_app

app = create_app()
```

**Features:**
- Centralized app creation
- Automatic middleware setup
- Automatic routes setup
- Lifespan management
- CORS configuration
- OpenAPI customization

### Lifespan Management

#### `core/lifespan.py` - **USE THIS** ✅
- **Status**: Canonical lifespan manager
- **Purpose**: Manages application startup and shutdown
- **Features**: Resource initialization, cleanup

### Configuration

#### `core/middleware_config.py` - **USE THIS** ✅
- **Status**: Canonical middleware configuration
- **Purpose**: Sets up all middleware
- **Used by**: `app_factory.py`

#### `core/routes_config.py` - **USE THIS** ✅
- **Status**: Canonical routes configuration
- **Purpose**: Sets up all API routes
- **Used by**: `app_factory.py`

### Base Classes

#### `core/base/base_model.py` - **USE THIS** ✅
- **Status**: Canonical base model classes
- **Purpose**: Base classes for ML models
- **Classes**:
  - `BaseModel` - Base neural network model
  - `BasePredictor` - Base predictor class

#### `core/base/base_trainer.py` - **USE THIS** ✅
- **Status**: Canonical base trainer classes
- **Purpose**: Base classes for model training
- **Features**: Training loop, callbacks, optimization

### Core Components

#### `core/addiction_analyzer.py` - Core Analyzer
- **Status**: ✅ Active
- **Purpose**: Main addiction analysis component

#### `core/recovery_planner.py` - Recovery Planner
- **Status**: ✅ Active
- **Purpose**: Recovery planning component

#### `core/progress_tracker.py` - Progress Tracker
- **Status**: ✅ Active
- **Purpose**: Progress tracking component

#### `core/relapse_prevention.py` - Relapse Prevention
- **Status**: ✅ Active
- **Purpose**: Relapse prevention component

### Layers Architecture

#### `core/layers/` - Layered Architecture
- **Status**: ✅ Active
- **Purpose**: Organized by architectural layers
- **Layers**:
  - `service_layer.py` - Service layer
  - `model_layer.py` - Model layer
  - `training_layer.py` - Training layer
  - `data_layer.py` - Data layer
  - `inference_layer.py` - Inference layer
  - `interface_layer.py` - Interface layer

### Models

#### `core/models/` - ML Models
- **Status**: ✅ Active
- **Purpose**: Machine learning models
- **Models**:
  - `diffusion_models.py` - Diffusion models
  - `llm_coach.py` - LLM coaching model
  - `sentiment_analyzer.py` - Sentiment analysis
  - `quantized_models.py` - Quantized models
  - `fast_models.py` - Fast inference models

### Training

#### `core/training/` - Training Components
- **Status**: ✅ Active
- **Purpose**: Training infrastructure
- **Components**:
  - `callbacks/` - Training callbacks
  - Experiment tracking
  - Checkpointing

### Inference

#### `core/inference/` - Inference Components
- **Status**: ✅ Active
- **Purpose**: Inference infrastructure
- **Components**:
  - `predictors/` - Prediction components
  - Fast inference engines

## 📝 Usage Examples

### Creating Application
```python
from core.app_factory import create_app

app = create_app()
# App is fully configured with middleware, routes, etc.
```

### Using Base Classes
```python
from core.base.base_model import BaseModel, BasePredictor
from core.base.base_trainer import BaseTrainer

# Create custom model
class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Your model definition

# Create custom trainer
class MyTrainer(BaseTrainer):
    def train(self, ...):
        # Your training logic
```

### Using Core Components
```python
from core.addiction_analyzer import AddictionAnalyzer
from core.recovery_planner import RecoveryPlanner
from core.progress_tracker import ProgressTracker

analyzer = AddictionAnalyzer()
planner = RecoveryPlanner()
tracker = ProgressTracker()
```

### Using Layers
```python
from core.layers.service_layer import ServiceLayer
from core.layers.model_layer import ModelLayer

service_layer = ServiceLayer()
model_layer = ModelLayer()
```

## 🎯 Quick Reference

| Component | Purpose | Status | When to Use |
|-----------|---------|--------|-------------|
| `app_factory.py` | Create FastAPI app | ✅ Canonical | Always (entry point) |
| `lifespan.py` | Lifespan management | ✅ Canonical | Managed by app_factory |
| `base/base_model.py` | Base model classes | ✅ Canonical | Creating ML models |
| `base/base_trainer.py` | Base trainer classes | ✅ Canonical | Creating trainers |
| `addiction_analyzer.py` | Core analyzer | ✅ Active | Addiction analysis |
| `recovery_planner.py` | Recovery planning | ✅ Active | Recovery planning |
| `progress_tracker.py` | Progress tracking | ✅ Active | Progress tracking |
| `relapse_prevention.py` | Relapse prevention | ✅ Active | Relapse prevention |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `ENTRY_POINTS_GUIDE.md` for entry points
- See `API_GUIDE.md` for API structure
- See `SERVICES_GUIDE.md` for services






