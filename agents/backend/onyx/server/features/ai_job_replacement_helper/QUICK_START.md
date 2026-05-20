# 🚀 Quick Start Guide

Guía rápida para empezar con AI Job Replacement Helper.

## Instalación Rápida

```bash
# 1. Navegar al directorio
cd ai_job_replacement_helper

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar servidor
python main.py
```

## Primeros Pasos

### 1. Verificar que el servidor funciona

```bash
curl http://localhost:8030/health
```

### 2. Buscar trabajos (estilo Tinder)

```bash
curl "http://localhost:8030/api/v1/jobs/search/user123?keywords=Python&location=Madrid"
```

### 3. Hacer swipe en un trabajo

```bash
curl -X POST "http://localhost:8030/api/v1/jobs/swipe/user123" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_1", "action": "like"}'
```

### 4. Ver tu progreso

```bash
curl "http://localhost:8030/api/v1/gamification/progress/user123"
```

### 5. Ver tu roadmap

```bash
curl "http://localhost:8030/api/v1/steps/roadmap/user123"
```

## Documentación de API

Una vez que el servidor esté corriendo, visita:
- **Swagger UI**: http://localhost:8030/docs
- **ReDoc**: http://localhost:8030/redoc

## Estructura de Usuario

Para usar el sistema, necesitas un `user_id`. Por ahora, puedes usar cualquier string como ID de usuario. En producción, esto vendría de un sistema de autenticación.

Ejemplos de `user_id`:
- `user123`
- `john_doe`
- `alice_smith`

## Flujo Típico

1. **Buscar trabajos**: `/api/v1/jobs/search/{user_id}`
2. **Hacer swipe**: `/api/v1/jobs/swipe/{user_id}` (like/dislike/save)
3. **Aplicar**: `/api/v1/jobs/apply/{user_id}?job_id={job_id}`
4. **Completar pasos**: `/api/v1/steps/complete/{user_id}`
5. **Ver progreso**: `/api/v1/gamification/progress/{user_id}`
6. **Obtener recomendaciones**: `/api/v1/recommendations/skills/{user_id}`

## Próximos Pasos

- Lee el [README.md](README.md) completo para más detalles
- Explora la documentación de la API en `/docs`
- Personaliza la configuración en `.env`




