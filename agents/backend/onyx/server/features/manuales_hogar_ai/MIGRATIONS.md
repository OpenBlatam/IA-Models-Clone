# Migraciones de Base de Datos con Alembic

Este proyecto utiliza Alembic para gestionar las migraciones de base de datos.

## Configuración

### Variables de Entorno

Configura las siguientes variables de entorno o en un archivo `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://usuario:password@localhost:5432/manuales_hogar
# O individualmente:
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=tu_password
DB_NAME=manuales_hogar
```

## Comandos de Alembic

### Inicializar Base de Datos

```bash
# Crear la primera migración
alembic revision --autogenerate -m "Initial migration"

# Aplicar migraciones
alembic upgrade head
```

### Crear Nueva Migración

```bash
# Crear migración automática (detecta cambios en modelos)
alembic revision --autogenerate -m "Descripción de los cambios"

# Crear migración manual
alembic revision -m "Descripción de los cambios"
```

### Aplicar Migraciones

```bash
# Aplicar todas las migraciones pendientes
alembic upgrade head

# Aplicar hasta una revisión específica
alembic upgrade <revision_id>

# Aplicar siguiente migración
alembic upgrade +1
```

### Revertir Migraciones

```bash
# Revertir última migración
alembic downgrade -1

# Revertir hasta una revisión específica
alembic downgrade <revision_id>

# Revertir todas las migraciones
alembic downgrade base
```

### Ver Estado

```bash
# Ver estado actual de migraciones
alembic current

# Ver historial de migraciones
alembic history

# Ver migraciones pendientes
alembic heads
```

## Estructura de Migraciones

Las migraciones se almacenan en `alembic/versions/` con el formato:
- `XXXX_descripcion.py`

## Modelos de Base de Datos

### Manual
Almacena los manuales generados con:
- Descripción del problema
- Categoría
- Contenido del manual
- Metadata (modelo usado, tokens, etc.)

### ManualCache
Cache persistente de manuales con:
- Clave de cache
- Contenido
- Fecha de expiración
- Contador de hits

### UsageStats
Estadísticas de uso con:
- Fecha
- Categoría
- Modelo usado
- Métricas (requests, tokens, imágenes)

## Ejemplo de Uso

```bash
# 1. Crear migración inicial
alembic revision --autogenerate -m "Create initial tables"

# 2. Revisar la migración generada en alembic/versions/

# 3. Aplicar migración
alembic upgrade head

# 4. Verificar estado
alembic current
```

## Notas

- Siempre revisa las migraciones generadas antes de aplicarlas
- Haz backup de la base de datos antes de aplicar migraciones en producción
- Las migraciones son versionadas y no deben modificarse una vez aplicadas




