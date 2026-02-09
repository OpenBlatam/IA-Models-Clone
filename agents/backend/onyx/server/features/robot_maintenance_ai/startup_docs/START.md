# 🚀 Inicio Rápido - Robot Maintenance AI

## Iniciar Todo

### Windows

```cmd
cd agents\backend\onyx\server\features\robot_maintenance_ai
python main.py
```

O con uvicorn directamente:

```cmd
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Linux/Mac

```bash
cd agents/backend/onyx/server/features/robot_maintenance_ai
python main.py
```

O con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuración Inicial

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Instalar Modelo de spaCy (Opcional pero Recomendado)

```bash
python -m spacy download es_core_news_sm
```

### 3. Configurar API Key de OpenRouter

**Windows:**
```cmd
set OPENROUTER_API_KEY=tu-api-key-aqui
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="tu-api-key-aqui"
```

**O crear archivo `.env`:**
```
OPENROUTER_API_KEY=tu-api-key-aqui
```

## URLs Disponibles

Una vez iniciado el servidor:

- **API Base**: http://localhost:8000
- **Documentación Swagger**: http://localhost:8000/docs
- **Documentación ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/robot-maintenance/health

## Endpoints Principales

### Hacer una Pregunta de Mantenimiento

```bash
curl -X POST http://localhost:8000/api/robot-maintenance/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cómo cambio el aceite de un robot industrial?",
    "robot_type": "robots_industriales"
  }'
```

### Diagnosticar un Problema

```bash
curl -X POST http://localhost:8000/api/robot-maintenance/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "El robot hace ruidos extraños",
    "robot_type": "robots_industriales",
    "sensor_data": {
      "temperature": 85.0,
      "vibration": 6.5
    }
  }'
```

### Obtener Procedimiento de Mantenimiento

```bash
curl -X POST http://localhost:8000/api/robot-maintenance/procedure \
  -H "Content-Type: application/json" \
  -d '{
    "procedure": "lubricación",
    "robot_type": "robots_industriales",
    "difficulty": "intermedio"
  }'
```

### Predecir Mantenimiento con ML

```bash
curl -X POST http://localhost:8000/api/robot-maintenance/predict \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "robots_industriales",
    "sensor_data": {
      "temperature": 28.5,
      "vibration": 0.15,
      "runtime_hours": 8500
    }
  }'
```

### Generar Checklist

```bash
curl -X POST http://localhost:8000/api/robot-maintenance/checklist \
  -H "Content-Type: application/json" \
  -d '{
    "robot_type": "robots_industriales",
    "maintenance_type": "preventivo"
  }'
```

## Ejecutar Ejemplos

```bash
python examples/basic_usage.py
```

## Troubleshooting Rápido

### Error: "OPENROUTER_API_KEY must be set"

**Solución**: Configura la variable de entorno `OPENROUTER_API_KEY`

### Error: "Spanish spaCy model not found"

**Solución**: Ejecuta `python -m spacy download es_core_news_sm`

### Error: Puerto 8000 ya en uso

**Solución**: Cambia el puerto en `main.py` o usa otro puerto:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Error: Módulo no encontrado

**Solución**: Asegúrate de estar en el directorio correcto y que todas las dependencias estén instaladas:
```bash
pip install -r requirements.txt
```

## Próximos Pasos

1. **Lee el README.md** para documentación completa
2. **Revisa los ejemplos** en `examples/basic_usage.py`
3. **Explora la API** en http://localhost:8000/docs
4. **Consulta QUICK_REFERENCE.md** para referencia rápida

## Detener el Servidor

Presiona `Ctrl+C` en la terminal donde está corriendo el servidor.






