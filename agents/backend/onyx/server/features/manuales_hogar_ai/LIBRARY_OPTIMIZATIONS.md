# 📚 Optimizaciones de Librerías - Manuales Hogar AI

## Resumen de Mejoras

Este documento describe las optimizaciones realizadas en las dependencias del proyecto.

## 🎯 Cambios Principales

### 1. **Versiones Actualizadas**

Todas las librerías han sido actualizadas a las versiones más recientes y estables:

- **FastAPI**: 0.104.0 → 0.115.0 (mejoras de rendimiento y seguridad)
- **Pydantic**: 2.0.0 → 2.9.0 (mejor validación y rendimiento)
- **SQLAlchemy**: 2.0.0 → 2.0.36 (bug fixes y optimizaciones)
- **httpx**: 0.25.0 → 0.27.0 (mejor soporte HTTP/2)
- **Redis**: 5.0.0 → 5.1.1 (mejoras de estabilidad)
- **Pillow**: 10.0.0 → 10.4.0 (seguridad y rendimiento)

### 2. **Nuevas Librerías Eficientes**

- **httpcore**: Agregado explícitamente para mejor rendimiento HTTP
- **uvicorn[standard]**: Incluye dependencias optimizadas (uvloop, httptools)

### 3. **Organización Mejorada**

Se han creado 3 archivos de requirements:

1. **requirements.txt**: Todas las dependencias (core + ML + monitoring)
2. **requirements-core.txt**: Solo dependencias esenciales (sin ML)
3. **requirements-ml.txt**: Solo dependencias ML (para instalación opcional)

### 4. **Dependencias Opcionales Claramente Marcadas**

- Librerías ML marcadas como opcionales
- Monitoring opcional en requirements-core.txt
- Optimizaciones avanzadas (flash-attn, tensorrt) comentadas

## 📦 Estructura de Archivos

```
requirements.txt          # Todas las dependencias
requirements-core.txt     # Solo core (producción sin ML)
requirements-ml.txt       # Solo ML (instalación opcional)
```

## 🚀 Instalación

### Instalación Completa (con ML)
```bash
pip install -r requirements.txt
```

### Instalación Core (sin ML - más rápida)
```bash
pip install -r requirements-core.txt
```

### Instalación ML Separada
```bash
pip install -r requirements-core.txt
pip install -r requirements-ml.txt
```

## 🔧 Mejoras de Rendimiento

### FastAPI 0.115.0
- Mejor manejo de requests asíncronos
- Optimizaciones de serialización JSON
- Mejor soporte para streaming

### Pydantic 2.9.0
- Validación más rápida
- Mejor manejo de tipos
- Menor uso de memoria

### httpx 0.27.0
- Mejor connection pooling
- Soporte HTTP/2 mejorado
- Menor latencia

### SQLAlchemy 2.0.36
- Mejor rendimiento de queries
- Optimizaciones de connection pooling
- Bug fixes importantes

## 📊 Comparación de Tamaño

### Antes (requirements.txt original)
- ~48 dependencias
- Incluía todas las librerías ML siempre
- ~2-3GB de espacio en disco

### Después (requirements-core.txt)
- ~15 dependencias esenciales
- Sin librerías ML pesadas
- ~200-300MB de espacio en disco

**Ahorro: ~85-90% de espacio cuando no se necesitan features ML**

## ⚠️ Notas Importantes

### Dependencias ML Pesadas
Las siguientes librerías son opcionales y muy pesadas:
- `torch` (~2GB)
- `transformers` (~500MB)
- `diffusers` (~300MB)
- `sentence-transformers` (~200MB)

Solo instálalas si realmente necesitas las features ML.

### Optimizaciones GPU
Para usar GPU, necesitas:
- `faiss-gpu` en lugar de `faiss-cpu`
- `onnxruntime-gpu` en lugar de `onnxruntime`
- `flash-attn` (requiere CUDA específico)
- `nvidia-tensorrt` (requiere CUDA específico)

### Compatibilidad
- Python 3.10+ requerido
- Todas las versiones son compatibles entre sí
- Pinned a versiones menores para estabilidad

## 🔄 Migración

Para migrar de la versión anterior:

1. **Backup del entorno actual:**
```bash
pip freeze > requirements-old.txt
```

2. **Instalar nueva versión:**
```bash
pip install -r requirements-core.txt
# O si necesitas ML:
pip install -r requirements.txt
```

3. **Verificar instalación:**
```bash
python -c "import fastapi; print(fastapi.__version__)"
```

## 📝 Mantenimiento

### Actualizar Dependencias
```bash
pip install --upgrade -r requirements-core.txt
```

### Verificar Vulnerabilidades
```bash
pip install safety
safety check -r requirements-core.txt
```

### Generar requirements-lock.txt
```bash
pip freeze > requirements-lock.txt
```

## 🎯 Recomendaciones

1. **Producción sin ML**: Usa `requirements-core.txt`
2. **Desarrollo con ML**: Usa `requirements.txt`
3. **Docker**: Usa multi-stage builds con requirements-core.txt
4. **CI/CD**: Instala solo lo necesario según el stage

## 📚 Referencias

- [FastAPI Changelog](https://fastapi.tiangolo.com/release-notes/)
- [Pydantic v2 Migration](https://docs.pydantic.dev/latest/migration/)
- [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/changelog/migration_20.html)

