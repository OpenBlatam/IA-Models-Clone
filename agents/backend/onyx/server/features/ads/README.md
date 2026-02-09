# Sistema de Generación de Anuncios con IA

## 📋 Descripción

Sistema avanzado para generación de anuncios usando modelos de difusión (diffusion models) y transformers. Este módulo proporciona capacidades completas para crear, optimizar y gestionar anuncios publicitarios mediante IA.

## 🚀 Características Principales

- **Generación de Anuncios con Diffusion Models**: Integración con modelos de difusión para crear contenido visual y textual
- **Optimización de Rendimiento**: Sistema optimizado con mixed precision training, gradient accumulation y multi-GPU
- **Tokenización Avanzada**: Sistema de tokenización optimizado para procesamiento eficiente
- **Gestión de Versiones**: Control de versiones integrado para modelos y configuraciones
- **API RESTful**: Interfaz API completa para integración con otros servicios
- **Sistema de Configuración**: Gestión flexible de configuraciones para diferentes escenarios

## 📁 Estructura

```
ads/
├── api/                    # Endpoints de la API
├── core/                   # Lógica central del sistema
├── config/                 # Configuraciones
├── domain/                 # Modelos de dominio
├── infrastructure/         # Infraestructura y servicios
├── optimization/           # Optimizaciones y mejoras
├── providers/              # Proveedores de servicios
├── services/               # Servicios de negocio
├── training/               # Entrenamiento de modelos
├── examples/               # Ejemplos de uso
└── docs/                   # Documentación adicional
```

## 🔧 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Para desarrollo
pip install -r profiling_requirements_dev.txt
```

## 💻 Uso Básico

```python
from ads.diffusion_service import DiffusionService
from ads.api import create_app

# Inicializar servicio
service = DiffusionService()

# Generar anuncio
result = service.generate_ad(prompt="Producto innovador para startups")

# Iniciar API
app = create_app()
```

## 📚 Documentación Adicional

- [Advanced Diffusers Guide](ADVANCED_DIFFUSERS_GUIDE.md)
- [Diffusion Process Guide](DIFFUSION_PROCESS_GUIDE.md)
- [Multi-GPU Training Guide](MULTI_GPU_TRAINING_GUIDE.md)
- [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [Tokenization Guide](TOKENIZATION_GUIDE.md)

## 🧪 Testing

```bash
# Ejecutar tests básicos
python test_basic.py

# Tests de difusión
python test_diffusion.py

# Tests de optimización
python test_performance_optimization.py
```

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: API Gateway principal
- **Export IA**: Exportación de anuncios generados
- **Business Agents**: Agentes de negocio para automatización

## 📊 Puerto

- Puerto por defecto: Configurable en `config.py`
- Health endpoint: `/health`
- API Docs: `/docs`



