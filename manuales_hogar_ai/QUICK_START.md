# 🚀 Quick Start Guide

## Instalación Rápida

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Crear archivo `.env`:

```bash
OPENROUTER_API_KEY=tu-api-key-aqui
DATABASE_URL=sqlite+aiosqlite:///./manuales_hogar.db
```

### 3. Configurar Entorno

```bash
python scripts/setup_environment.py
```

### 4. Verificar Sistema

```bash
python scripts/check_system.py
```

### 5. Inicializar Base de Datos

```bash
python scripts/init_db.py
```

### 6. Ejecutar Servidor

```bash
uvicorn api:app --reload --port 8000
```

## Uso Rápido

### Generar Manual desde Texto

```bash
curl -X POST "http://localhost:8000/api/v1/manuales/generate-from-text" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_description": "Tengo una fuga de agua en el grifo",
    "category": "plomeria"
  }'
```

### Generar Manual desde Imagen

```bash
curl -X POST "http://localhost:8000/api/v1/manuales/generate-from-image" \
  -F "file=@problema.jpg" \
  -F "problem_description=Problema con el grifo"
```

### Búsqueda Semántica

```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "fuga de agua",
    "limit": 10
  }'
```

## Scripts Útiles

### Setup Environment
```bash
python scripts/setup_environment.py
```

### Check System
```bash
python scripts/check_system.py
```

### Benchmark
```bash
python scripts/benchmark.py
```

### Train Model
```bash
python scripts/train_model.py
```

### Gradio Demo
```bash
python scripts/gradio_demo.py
```

## Endpoints Principales

- `POST /api/v1/manuales/generate-from-text` - Generar desde texto
- `POST /api/v1/manuales/generate-from-image` - Generar desde imagen
- `GET /api/v1/manuals` - Listar manuales
- `POST /api/v1/search/semantic` - Búsqueda semántica
- `GET /api/v1/health/` - Health check

## Documentación Completa

Ver `README.md` para documentación completa.




