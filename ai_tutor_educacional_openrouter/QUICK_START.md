# ⚡ Quick Start Guide - AI Tutor Educacional

## 🚀 Inicio Rápido (5 minutos)

### 1. Configuración Inicial

```bash
# Clonar o navegar al directorio
cd ai_tutor_educacional_openrouter

# Ejecutar setup automático
python scripts/setup.py
```

### 2. Configurar API Key

Edita el archivo `.env` y agrega tu API key de Open Router:

```bash
OPENROUTER_API_KEY=tu-api-key-aqui
```

### 3. Iniciar Servidor

```bash
# Opción 1: Python directo
python main.py

# Opción 2: Docker
docker-compose up -d
```

### 4. Probar el Sistema

```bash
# Health check
curl http://localhost:8000/api/tutor/health

# Hacer una pregunta
curl -X POST http://localhost:8000/api/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es la fotosíntesis?", "subject": "ciencias"}'
```

## 📚 Uso Básico

### Python SDK

```python
from sdk import TutorClient

client = TutorClient(base_url="http://localhost:8000")

# Hacer pregunta
response = client.ask_question(
    question="¿Qué es la fotosíntesis?",
    subject="ciencias"
)

print(response["data"]["answer"])
```

### API REST

```bash
# Pregunta
POST /api/tutor/ask

# Explicar concepto
POST /api/tutor/explain

# Generar ejercicios
POST /api/tutor/exercises

# Generar quiz
POST /api/tutor/quiz
```

## 📖 Documentación Completa

- **API Docs**: http://localhost:8000/docs
- **README**: Ver README.md
- **Integración**: Ver INTEGRATION_GUIDE.md
- **Deployment**: Ver DEPLOYMENT.md

## 🆘 Problemas Comunes

### Error: OPENROUTER_API_KEY not set
**Solución**: Configura la variable en `.env`

### Error: Connection refused
**Solución**: Verifica que el servidor esté corriendo en el puerto 8000

### Error: Rate limit exceeded
**Solución**: Ajusta el rate limiting en la configuración

## ✅ Checklist de Inicio

- [ ] API key configurada en `.env`
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Servidor iniciado (`python main.py`)
- [ ] Health check exitoso
- [ ] Primera pregunta realizada exitosamente

## 🎯 Próximos Pasos

1. Explora la documentación API en `/docs`
2. Revisa los ejemplos en `examples/`
3. Configura webhooks para integraciones
4. Personaliza la configuración según tus necesidades

## 🎯 Ejemplos Rápidos

### Explicar un Concepto

```python
explanation = client.explain_concept(
    concept="derivadas",
    subject="matematicas",
    difficulty="avanzado"
)
```

### Generar Ejercicios

```python
exercises = client.generate_exercises(
    topic="algebra",
    subject="matematicas",
    num_exercises=5
)
```

### Generar Quiz

```python
quiz = client.generate_quiz(
    topic="biologia",
    subject="ciencias",
    num_questions=10
)
```

## 📚 Siguientes Pasos

- Lee el [README.md](README.md) para documentación completa
- Revisa [FEATURES.md](FEATURES.md) para todas las características
- Consulta [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) para integraciones
- Ve [DEPLOYMENT.md](DEPLOYMENT.md) para deployment en producción

¡Listo para usar! 🎉




