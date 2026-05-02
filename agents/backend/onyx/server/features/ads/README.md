# AI-Powered Ad Generation System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

An advanced system for generating advertisements using diffusion models and transformers. This module provides end-to-end capabilities for creating, optimizing, and managing AI-driven ad content — both visual and textual.

## 🚀 Key Features

- **Diffusion Model Generation** — Integration with state-of-the-art diffusion models for visual and textual ad creation
- **Performance Optimization** — Mixed-precision training, gradient accumulation, and multi-GPU support
- **Advanced Tokenization** — Optimized tokenization pipeline for efficient text processing
- **Version Management** — Built-in version control for models and configurations
- **RESTful API** — Full API interface for integration with other services
- **Flexible Configuration** — Scenario-based configuration management

## 📁 Project Structure

```
ads/
├── api/                    # API endpoints
├── core/                   # Core system logic
├── config/                 # Configuration files
├── domain/                 # Domain models
├── infrastructure/         # Infrastructure & services
├── optimization/           # Optimizations & improvements
├── providers/              # Service providers
├── services/               # Business services
├── training/               # Model training
├── examples/               # Usage examples
└── docs/                   # Additional documentation
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r profiling_requirements_dev.txt
```

## 💻 Basic Usage

```python
from ads.diffusion_service import DiffusionService
from ads.api import create_app

# Initialize service
service = DiffusionService()

# Generate an ad
result = service.generate_ad(prompt="Innovative product for startups")

# Start the API
app = create_app()
```

## 📚 Additional Documentation

- [Advanced Diffusers Guide](ADVANCED_DIFFUSERS_GUIDE.md)
- [Diffusion Process Guide](DIFFUSION_PROCESS_GUIDE.md)
- [Multi-GPU Training Guide](MULTI_GPU_TRAINING_GUIDE.md)
- [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [Tokenization Guide](TOKENIZATION_GUIDE.md)

## 🧪 Testing

```bash
# Run basic tests
python test_basic.py

# Diffusion tests
python test_diffusion.py

# Optimization tests
python test_performance_optimization.py
```

## 🔗 Integration

This module integrates with:
- **[Integration System](../integration_system/README.md)** — Primary API Gateway
- **[Export IA](../export_ia/README.md)** — Export generated ads
- **[Business Agents](../business_agents/README.md)** — Automated business workflows

## 📊 Service Info

| Item | Value |
|------|-------|
| **Default Port** | Configurable in `config.py` |
| **Health Endpoint** | `/health` |
| **API Docs** | `/docs` |

---

[← Back to Main README](../README.md)
