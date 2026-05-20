# 🚀 Bulk TruthGPT - Guía de Inicio Rápido

## Instalación y Configuración

### 1. Ejecutar Setup Automático

```bash
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk_truthgpt
python setup.py
```

Este script:
- ✅ Crea todos los directorios necesarios
- ✅ Genera el archivo `.env` con configuración por defecto
- ✅ Verifica dependencias básicas

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Servicios Externos (Opcional)

#### Redis (Recomendado para cache y colas)
```bash
# Con Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine

# O instalar localmente
# Windows: choco install redis
# Linux: sudo apt install redis-server
```

#### PostgreSQL (Opcional, para almacenamiento persistente)
```bash
# Con Docker
docker run -d -p 5432:5432 \
  -e POSTGRES_DB=bulk_truthgpt \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  --name postgres postgres:15-alpine
```

### 4. Configurar Variables de Entorno

Edita el archivo `.env` y configura:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis (si está disponible)
REDIS_URL=redis://localhost:6379/0

# Database (si está disponible)
DATABASE_URL=postgresql://postgres:password@localhost:5432/bulk_truthgpt

# Security (generado automáticamente por setup.py)
SECRET_KEY=tu-clave-generada-aqui
```

## Inicio del Servidor

### Opción 1: Script de Inicio Rápido (Recomendado)

```bash
python start.py
```

### Opción 2: Uvicorn Directo

```bash
uvicorn bulk_truthgpt.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción 3: Docker Compose (Producción)

```bash
docker-compose up -d --build
```

## Verificar que Funciona

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Deberías ver:
```json
{
  "status": "healthy",
  "timestamp": "2024-..."
}
```

### 2. Documentación Interactiva

Abre en tu navegador:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Primer Uso

### Generar Documentos Masivamente

```bash
curl -X POST http://localhost:8000/api/v1/bulk/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explicar inteligencia artificial",
    "config": {
      "max_documents": 10,
      "max_tokens": 2000,
      "temperature": 0.7
    }
  }'
```

Respuesta:
```json
{
  "task_id": "uuid-aqui",
  "status": "processing",
  "message": "Generación iniciada"
}
```

### Consultar Estado

```bash
curl http://localhost:8000/api/v1/bulk/status/{task_id}
```

### Obtener Documentos Generados

```bash
curl http://localhost:8000/api/v1/bulk/documents/{task_id}?limit=10
```

## Estructura de Directorios

```
bulk_truthgpt/
├── storage/          # Documentos generados
├── templates/         # Plantillas de documentos
├── models/            # Modelos de IA
├── knowledge_base/    # Base de conocimiento
├── logs/             # Logs del sistema
├── cache/            # Cache temporal
├── .env              # Variables de entorno (crear con setup.py)
├── start.py          # Script de inicio rápido
├── setup.py          # Script de configuración
└── main.py           # Aplicación principal
```

## Troubleshooting

### Error: "Module not found"

```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: "Redis connection failed"

El sistema funcionará sin Redis, pero con funcionalidad limitada. Para activar Redis:
```bash
docker run -d -p 6379:6379 redis
```

### Error: "Port already in use"

Cambia el puerto en `.env`:
```env
API_PORT=8001
```

### Error: "Import error en bulk_truthgpt"

Asegúrate de estar en el directorio correcto:
```bash
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk_truthgpt
```

## Próximos Pasos

1. ✅ Sistema configurado y funcionando
2. 📖 Revisa la documentación completa: `API_QUICKSTART.md`
3. 🔧 Personaliza configuración en `.env`
4. 🚀 Integra con tu aplicación frontend
5. 📊 Monitorea métricas en `/metrics`

## Soporte

- 📖 Documentación completa: `README.md`
- 🔌 API Docs: http://localhost:8000/docs
- 🐛 Issues: Revisa los logs en `logs/`

---

**¡Listo para usar! 🎉**
































