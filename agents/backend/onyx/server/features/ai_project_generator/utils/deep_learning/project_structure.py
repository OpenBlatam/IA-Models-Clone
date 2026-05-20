"""
Project Structure Generator
===========================

Genera estructura de proyecto organizada para deep learning.
"""

from typing import Dict, List, Any
from pathlib import Path
from enum import Enum


class ProjectStructure:
    """Generador de estructura de proyecto."""
    
    @staticmethod
    def generate_structure(project_type: str) -> Dict[str, str]:
        """Genera estructura completa de proyecto."""
        structure = {
            "README.md": ProjectStructure._generate_readme(project_type),
            "requirements.txt": ProjectStructure._generate_requirements(),
            "setup.py": ProjectStructure._generate_setup(),
            "config.yaml": ProjectStructure._generate_config_yaml(project_type),
            ".gitignore": ProjectStructure._generate_gitignore(),
            ".env.example": ProjectStructure._generate_env_example(),
        }
        
        # Directorios
        structure.update({
            "src/__init__.py": "",
            "src/models/__init__.py": "",
            "src/data/__init__.py": "",
            "src/training/__init__.py": "",
            "src/utils/__init__.py": "",
            "src/evaluation/__init__.py": "",
            "notebooks/.gitkeep": "",
            "data/.gitkeep": "",
            "checkpoints/.gitkeep": "",
            "logs/.gitkeep": "",
            "scripts/train.sh": ProjectStructure._generate_train_script(),
            "scripts/evaluate.sh": ProjectStructure._generate_evaluate_script(),
        })
        
        return structure
    
    @staticmethod
    def _generate_readme(project_type: str) -> str:
        """Genera README.md."""
        return f'''# {project_type.upper()} Deep Learning Project

Proyecto de deep learning usando PyTorch, Transformers y Diffusers.

## 📋 Estructura del Proyecto

```
.
├── src/                    # Código fuente
│   ├── models/             # Arquitecturas de modelos
│   ├── data/               # Procesamiento de datos
│   ├── training/           # Scripts de entrenamiento
│   ├── evaluation/         # Scripts de evaluación
│   └── utils/              # Utilidades
├── notebooks/              # Jupyter notebooks
├── data/                   # Datos del proyecto
├── checkpoints/            # Modelos guardados
├── logs/                   # Logs de entrenamiento
├── scripts/                # Scripts de shell
└── config.yaml             # Configuración del proyecto
```

## 🚀 Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\\Scripts\\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e .
```

## 📊 Uso

### Entrenamiento

```bash
# Usando script de Python
python -m src.training.train --config config.yaml

# Usando script de shell
bash scripts/train.sh
```

### Evaluación

```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

### Demo

```bash
python -m src.demo --model checkpoints/best_model.pt
```

## 🔧 Configuración

Edita `config.yaml` para ajustar hiperparámetros y configuración del modelo.

## 📝 Notas

- Los checkpoints se guardan en `checkpoints/`
- Los logs se guardan en `logs/`
- Usa `wandb` o `tensorboard` para visualizar el entrenamiento

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.
'''
    
    @staticmethod
    def _generate_requirements() -> str:
        """Genera requirements.txt."""
        return '''# Core Deep Learning
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Transformers & LLMs
transformers>=4.35.0
tokenizers>=0.15.0
sentencepiece>=0.1.99

# Diffusion Models
diffusers>=0.24.0
xformers>=0.0.23
safetensors>=0.4.0

# Efficient Training & Fine-tuning
accelerate>=0.25.0
peft>=0.7.0
bitsandbytes>=0.41.0
trl>=0.7.0

# Data & Datasets
datasets>=2.14.0
huggingface-hub>=0.19.0
pillow>=10.1.0
opencv-python>=4.8.0

# Visualization & UI
gradio>=4.7.0
streamlit>=1.28.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Experiment Tracking
wandb>=0.16.0
tensorboard>=2.15.0
mlflow>=2.8.0

# Utilities
numpy>=1.26.0
pandas>=2.1.0
scipy>=1.11.0
tqdm>=4.66.0
pyyaml>=6.0.1
omegaconf>=2.3.0

# Performance & Optimization
ninja>=1.11.0
optimum>=1.14.0

# Monitoring & Profiling
psutil>=5.9.0
gpustat>=1.1.0

# Testing & Quality
pytest>=7.4.0
black>=23.11.0
flake8>=6.1.0
mypy>=1.7.0

# Jupyter & Development
jupyter>=1.0.0
ipywidgets>=8.1.0
rich>=13.7.0
'''
    
    @staticmethod
    def _generate_setup() -> str:
        """Genera setup.py."""
        return '''"""
Setup script for deep learning project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deep-learning-project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deep-learning-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
)
'''
    
    @staticmethod
    def _generate_config_yaml(project_type: str) -> str:
        """Genera config.yaml."""
        return f'''# Configuración del proyecto
project:
  name: "deep-learning-project"
  type: "{project_type}"
  version: "0.1.0"

# Modelo
model:
  type: "{project_type}"
  vocab_size: 50257
  d_model: 768
  nhead: 12
  num_layers: 12
  dim_feedforward: 3072
  max_seq_length: 512
  dropout: 0.1

# Entrenamiento
training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  num_epochs: 10
  warmup_steps: 1000
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  mixed_precision: true
  distributed: false
  use_lora: false
  use_8bit: false

# Datos
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  cache_dir: ".cache"

# Checkpoints
checkpoints:
  save_dir: "checkpoints"
  save_interval: 1
  keep_last_n: 3

# Logging
logging:
  log_dir: "logs"
  use_wandb: true
  wandb_project: "deep-learning-project"
  use_tensorboard: true
  log_interval: 100

# Evaluación
evaluation:
  metrics: ["loss", "accuracy", "f1"]
  eval_interval: 1
'''
    
    @staticmethod
    def _generate_gitignore() -> str:
        """Genera .gitignore."""
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data
data/raw/
data/processed/
*.csv
*.json
*.pkl
*.h5
*.hdf5

# Models & Checkpoints
checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors

# Logs
logs/
*.log
wandb/
tensorboard/

# Cache
.cache/
__pycache__/
.pytest_cache/
.mypy_cache/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db
'''
    
    @staticmethod
    def _generate_env_example() -> str:
        """Genera .env.example."""
        return '''# WandB Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=deep-learning-project
WANDB_ENTITY=your_entity

# Hugging Face
HF_TOKEN=your_huggingface_token_here
HF_CACHE_DIR=.cache

# CUDA
CUDA_VISIBLE_DEVICES=0

# Training
SEED=42
NUM_WORKERS=4
'''
    
    @staticmethod
    def _generate_train_script() -> str:
        """Genera script de entrenamiento."""
        return '''#!/bin/bash

# Script de entrenamiento
# Uso: bash scripts/train.sh

set -e

# Configuración
CONFIG_FILE="config.yaml"
OUTPUT_DIR="checkpoints"
LOG_DIR="logs"

# Crear directorios
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Entrenar
python -m src.training.train \\
    --config $CONFIG_FILE \\
    --output_dir $OUTPUT_DIR \\
    --log_dir $LOG_DIR \\
    "$@"

echo "Entrenamiento completado!"
'''
    
    @staticmethod
    def _generate_evaluate_script() -> str:
        """Genera script de evaluación."""
        return '''#!/bin/bash

# Script de evaluación
# Uso: bash scripts/evaluate.sh <checkpoint_path>

set -e

if [ -z "$1" ]; then
    echo "Error: Se requiere path del checkpoint"
    echo "Uso: bash scripts/evaluate.sh <checkpoint_path>"
    exit 1
fi

CHECKPOINT=$1
CONFIG_FILE="config.yaml"

python -m src.evaluation.evaluate \\
    --checkpoint $CHECKPOINT \\
    --config $CONFIG_FILE

echo "Evaluación completada!"
'''

