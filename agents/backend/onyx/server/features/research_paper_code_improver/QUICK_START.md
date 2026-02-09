# Quick Start Guide - Research Paper Code Improver

## 🚀 Inicio Rápido

### 1. Instalación

```bash
cd research_paper_code_improver
pip install -r requirements.txt
```

### 2. Configuración

Crear archivo `.env`:

```env
HOST=0.0.0.0
PORT=8030
GITHUB_TOKEN=tu_token_aqui
DEFAULT_MODEL=gpt-4
```

### 3. Ejecutar Servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8030`

## 📖 Uso Básico

### Subir PDF

```bash
curl -X POST "http://localhost:8030/api/research-paper-code-improver/papers/upload" \
  -F "file=@paper.pdf"
```

### Procesar Link

```bash
curl -X POST "http://localhost:8030/api/research-paper-code-improver/papers/link" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/pdf/2301.00001.pdf"}'
```

### Entrenar Modelo

```bash
curl -X POST "http://localhost:8030/api/research-paper-code-improver/training/train" \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 3,
    "model_name": "gpt-4",
    "use_all_papers": true
  }'
```

### Mejorar Código

```bash
curl -X POST "http://localhost:8030/api/research-paper-code-improver/code/improve" \
  -H "Content-Type: application/json" \
  -d '{
    "github_repo": "owner/repo",
    "file_path": "src/main.py",
    "branch": "main"
  }'
```

## 🔧 Ejemplos Python

```python
from core.paper_extractor import PaperExtractor
from core.model_trainer import ModelTrainer
from core.code_improver import CodeImprover

# 1. Extraer paper
extractor = PaperExtractor()
paper = extractor.extract_from_pdf("paper.pdf")

# 2. Entrenar modelo
trainer = ModelTrainer()
model_path = trainer.train_from_papers([paper], epochs=3)

# 3. Mejorar código
improver = CodeImprover(model_path=model_path)
result = improver.improve_code("owner/repo", "src/main.py")
print(result["improved_code"])
```

## 📚 Documentación Completa

Ver [README.md](README.md) para documentación completa.




