# Guía de Dependencias - Dermatology AI

## 📦 Archivos de Dependencias

Este proyecto incluye múltiples archivos de requirements para diferentes casos de uso:

### 1. `requirements.txt` - Producción Completa
**Uso**: Instalación completa para producción con todas las características.

```bash
pip install -r requirements.txt
```

**Incluye**:
- ✅ Todas las dependencias de producción
- ✅ ML/AI completo (PyTorch, Transformers, etc.)
- ✅ Múltiples bases de datos
- ✅ Message brokers
- ✅ Herramientas de monitoreo
- ✅ Visualización

**Tamaño estimado**: ~2-3 GB

### 2. `requirements-optimized.txt` - Producción Optimizada
**Uso**: Instalación mínima para producción (solo inferencia, sin entrenamiento).

```bash
pip install -r requirements-optimized.txt
```

**Incluye**:
- ✅ Core FastAPI
- ✅ Autenticación y seguridad
- ✅ PostgreSQL (base de datos principal)
- ✅ Redis (caché)
- ✅ Observabilidad esencial
- ✅ Procesamiento de imágenes básico
- ✅ ONNX Runtime (inferencia optimizada)
- ❌ Sin herramientas de desarrollo
- ❌ Sin librerías de entrenamiento ML
- ❌ Sin bases de datos opcionales

**Tamaño estimado**: ~500 MB

### 3. `requirements-dev.txt` - Desarrollo
**Uso**: Instalación para desarrollo local.

```bash
pip install -r requirements-dev.txt
```

**Incluye**:
- ✅ Todo de `requirements.txt`
- ✅ Herramientas de testing (pytest, coverage, etc.)
- ✅ Linters y formatters (black, ruff, mypy)
- ✅ Herramientas de desarrollo (ipython, pre-commit)
- ✅ Debugging tools

### 4. `requirements-minimal.txt` - Mínimo
**Uso**: Instalación mínima para pruebas rápidas o CI/CD básico.

```bash
pip install -r requirements-minimal.txt
```

**Incluye**:
- ✅ Solo FastAPI core
- ✅ Dependencias esenciales
- ❌ Sin ML/AI
- ❌ Sin bases de datos
- ❌ Sin herramientas adicionales

**Tamaño estimado**: ~50 MB

### 5. `requirements-lock.txt` - Versiones Fijas
**Uso**: Producción con versiones exactas (reproducibilidad).

```bash
pip install -r requirements-lock.txt
```

**Nota**: Este archivo debe generarse con `pip-compile`:
```bash
pip install pip-tools
pip-compile requirements.txt
```

## 🚀 Instalación Rápida

### Producción (Completa)
```bash
pip install -r requirements.txt
```

### Producción (Optimizada - Recomendado)
```bash
pip install -r requirements-optimized.txt
```

### Desarrollo
```bash
pip install -r requirements-dev.txt
```

### Mínimo (Testing)
```bash
pip install -r requirements-minimal.txt
```

## 📊 Comparación de Archivos

| Característica | requirements.txt | requirements-optimized.txt | requirements-dev.txt | requirements-minimal.txt |
|---------------|------------------|---------------------------|---------------------|-------------------------|
| FastAPI Core | ✅ | ✅ | ✅ | ✅ |
| ML Training | ✅ | ❌ | ✅ | ❌ |
| ML Inference | ✅ | ✅ | ✅ | ❌ |
| Multiple DBs | ✅ | ❌ (solo PostgreSQL) | ✅ | ❌ |
| Message Brokers | ✅ | ❌ | ✅ | ❌ |
| Monitoring | ✅ | ✅ (esencial) | ✅ | ❌ |
| Dev Tools | ❌ | ❌ | ✅ | ❌ |
| Testing | ❌ | ❌ | ✅ | ✅ (básico) |
| Tamaño | ~2-3 GB | ~500 MB | ~3-4 GB | ~50 MB |

## 🔧 Gestión de Dependencias

### Actualizar Dependencias

```bash
# Ver dependencias desactualizadas
pip list --outdated

# Actualizar una dependencia específica
pip install --upgrade package-name

# Actualizar requirements-lock.txt
pip-compile --upgrade requirements.txt
```

### Verificar Vulnerabilidades

```bash
# Usando safety
pip install safety
safety check -r requirements.txt

# Usando pip-audit
pip install pip-audit
pip-audit -r requirements.txt
```

### Limpiar Dependencias

```bash
# Desinstalar paquetes no usados
pip-autoremove -r requirements.txt

# Ver dependencias no usadas
pip-check
```

## 🎯 Recomendaciones por Caso de Uso

### 🐳 Docker (Producción)
```dockerfile
# Usar requirements-optimized.txt para imágenes más pequeñas
COPY requirements-optimized.txt .
RUN pip install --no-cache-dir -r requirements-optimized.txt
```

### ☁️ AWS Lambda
```bash
# Usar requirements-optimized.txt
# Considerar usar Lambda Layers para dependencias grandes
pip install -r requirements-optimized.txt -t .
```

### 🧪 CI/CD
```bash
# Usar requirements-minimal.txt para tests rápidos
pip install -r requirements-minimal.txt
pytest
```

### 💻 Desarrollo Local
```bash
# Usar requirements-dev.txt
pip install -r requirements-dev.txt
```

### 🚀 Kubernetes
```bash
# Usar requirements-optimized.txt
# Considerar multi-stage builds
```

## 📦 Dependencias Opcionales

Algunas dependencias están comentadas y pueden activarse según necesidad:

### GraphQL
```bash
# Descomentar en requirements.txt:
# strawberry-graphql[fastapi]>=0.215.0
```

### gRPC
```bash
# Descomentar en requirements.txt:
# grpcio>=1.62.0
# grpcio-tools>=1.62.0
```

### GPU Support
```bash
# Para ONNX con GPU:
onnxruntime-gpu>=1.19.0
```

## 🔒 Seguridad

### Mejores Prácticas

1. **Revisar regularmente**:
   ```bash
   pip list --outdated
   safety check
   ```

2. **Fijar versiones críticas**:
   ```python
   # En requirements.txt, cambiar >= por == para seguridad
   cryptography==43.0.0  # En lugar de >=
   ```

3. **Usar requirements-lock.txt en producción**:
   ```bash
   pip install -r requirements-lock.txt
   ```

4. **Actualizar automáticamente**:
   ```bash
   # Usar Dependabot o Renovate
   ```

## 📈 Optimización

### Reducir Tamaño de Imagen Docker

1. Usar `requirements-optimized.txt`
2. Multi-stage builds
3. Eliminar cache de pip:
   ```dockerfile
   RUN pip install --no-cache-dir -r requirements-optimized.txt
   ```

### Acelerar Instalación

1. Usar mirrors locales:
   ```bash
   pip install -r requirements.txt -i https://pypi.org/simple
   ```

2. Cache de pip:
   ```bash
   pip install --cache-dir /cache -r requirements.txt
   ```

3. Instalación paralela:
   ```bash
   pip install -r requirements.txt --parallel
   ```

## 🐛 Troubleshooting

### Error: "No module named X"
```bash
# Verificar que requirements.txt incluye la dependencia
# Reinstalar:
pip install -r requirements.txt --force-reinstall
```

### Conflictos de Versiones
```bash
# Usar pip-tools para resolver:
pip-compile requirements.txt
pip install -r requirements-lock.txt
```

### Dependencias de Sistema
```bash
# Algunas dependencias requieren librerías del sistema:
# - opencv-python: libgl1, libglib2.0
# - Pillow: libjpeg, zlib
# - psycopg2: libpq-dev
```

## 📚 Recursos Adicionales

- [pip documentation](https://pip.pypa.io/)
- [pip-tools](https://github.com/jazzband/pip-tools)
- [Poetry](https://python-poetry.org/) - Alternativa moderna
- [pipenv](https://pipenv.pypa.io/) - Otra alternativa

## 🔄 Migración

### De requirements.txt a Poetry
```bash
poetry init
poetry add $(cat requirements.txt | grep -v '^#' | grep -v '^$' | cut -d'=' -f1)
```

### De requirements.txt a pipenv
```bash
pipenv install -r requirements.txt
```

## ✅ Checklist de Producción

Antes de desplegar, verificar:

- [ ] Todas las dependencias están actualizadas
- [ ] No hay vulnerabilidades conocidas (`safety check`)
- [ ] `requirements-lock.txt` está actualizado
- [ ] Versiones críticas están fijadas
- [ ] Dependencias opcionales están documentadas
- [ ] Tamaño de imagen Docker es aceptable
- [ ] Tests pasan con las dependencias instaladas



