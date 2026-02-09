# 🚀 Inicio Rápido - Bulk TruthGPT

## ✅ Sistema Listo

El sistema ha sido configurado y está listo para usar.

## 📋 Resumen de lo Configurado

### ✅ Directorios Creados
- `storage/` - Documentos generados
- `templates/` - Plantillas
- `models/` - Modelos de IA  
- `knowledge_base/` - Base de conocimiento
- `logs/` - Logs del sistema
- `cache/` - Cache temporal
- `temp/` - Archivos temporales

### ✅ Archivos de Configuración
- `setup.py` - Script de configuración automática
- `start.py` - Script de inicio rápido
- `verify_setup.py` - Script de verificación
- `.env` - Variables de entorno (creado desde env.example)

### ✅ Documentación
- `QUICKSTART.md` - Guía de inicio rápido completa
- `API_QUICKSTART.md` - Documentación de API
- `ESTADO_SISTEMA.md` - Estado actual del sistema

## 🚀 Iniciar el Sistema

### Opción 1: Script Simple (Recomendado)

```bash
python start.py
```

Esto iniciará el servidor en `http://localhost:8000`

### Opción 2: Uvicorn Directo

```bash
uvicorn bulk_truthgpt.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción 3: Desde main.py

```bash
python main.py
```

(El puerto por defecto en main.py es 8006 según el código)

## 📝 Verificar que Funciona

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Documentación Interactiva

Abre en tu navegador:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Verificar Configuración

```bash
python verify_setup.py
```

## 🔧 Configuración Mínima

El sistema puede funcionar sin servicios externos:

- ✅ **Sin Redis**: Funciona pero sin cache avanzado
- ✅ **Sin PostgreSQL**: Funciona pero sin almacenamiento persistente
- ✅ **Mínimo**: Solo Python y las dependencias en requirements.txt

## 📦 Instalar Dependencias (si no están instaladas)

```bash
pip install -r requirements.txt
```

## 🎯 Primer Uso

### Generar Documentos

```bash
curl -X POST http://localhost:8000/api/v1/bulk/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explicar inteligencia artificial",
    "config": {
      "max_documents": 10
    }
  }'
```

## 📚 Documentación Completa

- **QUICKSTART.md** - Guía paso a paso
- **API_QUICKSTART.md** - Referencia de API
- **ESTADO_SISTEMA.md** - Estado actual

## ⚠️ Notas

1. El archivo `.env` ya está creado con configuración por defecto
2. Si necesitas cambiar el puerto, edita `.env` o `start.py`
3. Para producción, cambia `DEBUG=false` en `.env`
4. El sistema funciona sin Redis/PostgreSQL, pero con funcionalidad limitada

---

**✅ Sistema listo y configurado para usar**

Para más detalles, consulta `QUICKSTART.md` o `ESTADO_SISTEMA.md`
































