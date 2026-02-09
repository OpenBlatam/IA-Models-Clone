# Guía de Dependencias - GitHub Autonomous Agent

## 📦 Estructura de Requirements

Este proyecto utiliza múltiples archivos de requirements para diferentes entornos:

- **`requirements.txt`** - Dependencias base (producción y desarrollo)
- **`requirements-dev.txt`** - Dependencias adicionales para desarrollo
- **`requirements-prod.txt`** - Dependencias adicionales para producción

---

## 🚀 Instalación Rápida

### Desarrollo Local
```bash
# Instalar dependencias base + desarrollo
pip install -r requirements.txt
pip install -r requirements-dev.txt

# O en un solo comando
pip install -r requirements.txt -r requirements-dev.txt
```

### Producción
```bash
# Instalar solo dependencias de producción
pip install -r requirements.txt
pip install -r requirements-prod.txt

# O en un solo comando
pip install -r requirements.txt -r requirements-prod.txt
```

### Solo Base (Mínimo)
```bash
# Solo dependencias esenciales
pip install -r requirements.txt
```

---

## 📋 Categorías de Dependencias

### Core Framework & API
- **FastAPI**: Framework web moderno y rápido
- **Uvicorn**: Servidor ASGI de alto rendimiento
- **Pydantic**: Validación de datos y modelos

### GitHub Integration
- **PyGithub**: Cliente oficial de GitHub API
- **GitPython**: Operaciones Git nativas
- **Dulwich**: Implementación Git pura (más rápida, opcional)

### Task Queue & Workers
- **Celery**: Sistema de cola de tareas distribuido
- **Redis**: Broker y backend para Celery
- **Flower**: Monitor web para Celery

### Database & ORM
- **SQLAlchemy**: ORM y toolkit SQL
- **Alembic**: Migraciones de base de datos
- **aiosqlite**: Driver async para SQLite
- **asyncpg**: Driver async para PostgreSQL (opcional)

### HTTP Clients
- **httpx**: Cliente HTTP async moderno
- **aiohttp**: Cliente HTTP async alternativo

### Security & Authentication
- **PyJWT**: JSON Web Tokens
- **cryptography**: Criptografía y hashing
- **passlib**: Hashing de contraseñas
- **slowapi**: Rate limiting

### Observability
- **structlog**: Logging estructurado
- **prometheus-client**: Métricas Prometheus
- **sentry-sdk**: Error tracking (producción)

---

## 🔧 Dependencias Opcionales

Algunas dependencias están marcadas como opcionales. Puedes instalarlas según necesidad:

### Análisis de Código
```bash
pip install tree-sitter tree-sitter-python
```

### Formateo de Código
```bash
pip install black ruff
```

### Retry Logic
```bash
pip install tenacity backoff
```

### Webhooks
```bash
pip install webhooks
```

---

## 🧪 Testing

Para desarrollo, instala dependencias de testing:

```bash
pip install -r requirements-dev.txt
```

Esto incluye:
- `pytest` - Framework de testing
- `pytest-asyncio` - Soporte async
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `faker` - Datos de prueba
- `freezegun` - Mock de fechas

---

## 📊 Gestión de Versiones

### Estrategia de Versionado

Usamos **versionado semántico** con rangos específicos:

```python
# Formato: >=mínima,<siguiente_mayor
fastapi>=0.115.0,<0.116.0
```

Esto asegura:
- ✅ Compatibilidad con versiones mínimas requeridas
- ✅ Prevención de breaking changes en actualizaciones automáticas
- ✅ Flexibilidad para actualizaciones de parches

### Actualizar Dependencias

```bash
# Verificar dependencias desactualizadas
pip list --outdated

# Actualizar una dependencia específica
pip install --upgrade fastapi

# Actualizar todas (cuidado con breaking changes)
pip install --upgrade -r requirements.txt
```

### Verificar Vulnerabilidades

```bash
# Con safety (incluido en dev)
safety check -r requirements.txt

# Con pip-audit (alternativa)
pip install pip-audit
pip-audit -r requirements.txt
```

---

## 🔒 Seguridad

### Dependencias Críticas de Seguridad

Estas dependencias son críticas y deben mantenerse actualizadas:

- `cryptography` - Criptografía
- `PyJWT` - Tokens JWT
- `passlib` - Hashing de contraseñas
- `fastapi` - Framework (actualizaciones de seguridad)

### Verificación Regular

```bash
# Ejecutar semanalmente
safety check -r requirements.txt
pip-audit -r requirements.txt
```

---

## 📈 Performance

### Dependencias Optimizadas

Para mejor performance en producción:

- **orjson** - JSON parser más rápido
- **hiredis** - Parser Redis más rápido
- **asyncpg** - Driver PostgreSQL async (vs psycopg2)
- **dulwich** - Git operations más rápidas

### Instalación de Optimizaciones

```bash
# Agregar a requirements-prod.txt o instalar manualmente
pip install orjson hiredis asyncpg dulwich
```

---

## 🐛 Troubleshooting

### Error: "No module named X"

1. Verifica que instalaste requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

2. Verifica que estás en el entorno virtual correcto:
   ```bash
   which python
   pip list
   ```

### Error: "Version conflict"

1. Actualiza pip:
   ```bash
   pip install --upgrade pip
   ```

2. Limpia cache:
   ```bash
   pip cache purge
   ```

3. Reinstala:
   ```bash
   pip uninstall -r requirements.txt -y
   pip install -r requirements.txt
   ```

### Error: "Compilation failed"

Algunas dependencias requieren compilación (cryptography, etc.):

```bash
# En Ubuntu/Debian
sudo apt-get install build-essential python3-dev libffi-dev libssl-dev

# En macOS
xcode-select --install
brew install openssl

# Luego reinstalar
pip install --upgrade --force-reinstall cryptography
```

---

## 📝 Mantenimiento

### Checklist Mensual

- [ ] Revisar dependencias desactualizadas
- [ ] Verificar vulnerabilidades de seguridad
- [ ] Actualizar dependencias críticas
- [ ] Probar actualizaciones en desarrollo
- [ ] Actualizar documentación si hay cambios

### Proceso de Actualización

1. **Desarrollo:**
   ```bash
   # Crear branch
   git checkout -b update-dependencies
   
   # Actualizar requirements
   pip install --upgrade package-name
   pip freeze > requirements-new.txt
   
   # Probar
   pytest
   ```

2. **Testing:**
   - Ejecutar suite completa de tests
   - Verificar que no hay regresiones
   - Revisar changelogs de dependencias

3. **Producción:**
   - Actualizar en staging primero
   - Monitorear errores
   - Actualizar producción gradualmente

---

## 🔗 Recursos

- [Python Package Index (PyPI)](https://pypi.org/)
- [Safety DB (Vulnerabilidades)](https://github.com/pyupio/safety-db)
- [Dependabot](https://dependabot.com/) - Actualizaciones automáticas
- [Renovate](https://renovatebot.com/) - Alternativa a Dependabot

---

**Última actualización:** 2024  
**Mantenido por:** Development Team




