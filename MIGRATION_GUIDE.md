# 🔄 Guía de Migración - Blatam Academy Features

## 📋 Tabla de Contenidos

- [Migración de Versiones](#migración-de-versiones)
- [Actualización de Configuración](#actualización-de-configuración)
- [Migración de Datos](#migración-de-datos)
- [Compatibilidad](#compatibilidad)
- [Rollback](#rollback)

## 🔄 Migración de Versiones

### De v1.x a v2.0

#### Cambios Principales

1. **Nueva API de KV Cache**
   - Cambio en métodos de inicialización
   - Nueva configuración requerida

**Antes (v1.x):**
```python
from bulk.core.kv_cache import SimpleKVCache

cache = SimpleKVCache(max_size=4096)
```

**Después (v2.0):**
```python
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    KVCacheConfig
)

config = KVCacheConfig(max_tokens=4096)
cache = UltraAdaptiveKVCacheEngine(config)
```

2. **Cambios en Variables de Entorno**

**Antes:**
```env
KV_CACHE_SIZE=4096
KV_CACHE_STRATEGY=lru
```

**Después:**
```env
KV_CACHE_MAX_TOKENS=4096
KV_CACHE_STRATEGY=adaptive
KV_CACHE_ENABLE_PERSISTENCE=true
KV_CACHE_PERSISTENCE_PATH=/data/cache
```

#### Script de Migración

```python
# scripts/migrate_v1_to_v2.py
import os
import json
from pathlib import Path

def migrate_config_v1_to_v2(old_config_path: str, new_config_path: str):
    """Migra configuración de v1 a v2."""
    
    with open(old_config_path) as f:
        old_config = json.load(f)
    
    new_config = {
        "kv_cache": {
            "max_tokens": old_config.get("kv_cache_size", 4096),
            "cache_strategy": old_config.get("kv_cache_strategy", "adaptive"),
            "enable_persistence": True,
            "persistence_path": "/data/cache",
            "use_compression": True,
            "compression_ratio": 0.3
        }
    }
    
    with open(new_config_path, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    print(f"✅ Configuración migrada de {old_config_path} a {new_config_path}")

if __name__ == "__main__":
    migrate_config_v1_to_v2(
        "config/old_config.json",
        "config/new_config.json"
    )
```

### De v2.0 a v2.1

#### Nuevas Características

1. **Nuevo sistema de prefetching**
   - Habilitado por defecto
   - Configurable vía `prefetch_size`

2. **Mejoras en persistencia**
   - Formato de almacenamiento actualizado
   - Migración automática al cargar

**No requiere cambios de código**, pero se recomienda:

```python
# Actualizar configuración para aprovechar nuevas características
config = KVCacheConfig(
    max_tokens=4096,
    enable_prefetch=True,  # Nuevo: habilitado por defecto
    prefetch_size=8,       # Nuevo: configurable
    enable_persistence=True,
    persistence_path="/data/cache"
)
```

## ⚙️ Actualización de Configuración

### Migración Automática de Config

```python
# scripts/migrate_config.py
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager

def migrate_config():
    """Migra configuración automáticamente."""
    
    manager = ConfigManager()
    
    # Detectar versión antigua
    if manager.detect_old_format():
        print("🔄 Detectada configuración antigua, migrando...")
        
        # Backup
        manager.backup_current_config()
        
        # Migrar
        manager.migrate_to_new_format()
        
        print("✅ Migración completada")
    else:
        print("ℹ️  Configuración ya está actualizada")

if __name__ == "__main__":
    migrate_config()
```

### Actualización Manual

1. **Backup de configuración actual**
```bash
cp .env .env.backup
cp config/kv_cache.yaml config/kv_cache.yaml.backup
```

2. **Actualizar variables de entorno**
```env
# Agregar nuevas variables
KV_CACHE_ENABLE_PERSISTENCE=true
KV_CACHE_PERSISTENCE_PATH=/data/cache
KV_CACHE_ENABLE_PREFETCH=true
```

3. **Verificar configuración**
```python
from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig

config = KVCacheConfig()
print(config)  # Verificar que todos los campos están presentes
```

## 💾 Migración de Datos

### Migración de Cache Persistente

```python
# scripts/migrate_cache_data.py
from pathlib import Path
import pickle
import torch

def migrate_cache_v1_to_v2(old_cache_path: str, new_cache_path: str):
    """Migra datos de cache de v1 a v2."""
    
    print(f"🔄 Migrando cache de {old_cache_path} a {new_cache_path}...")
    
    # Cargar cache antiguo
    with open(old_cache_path, 'rb') as f:
        old_cache = pickle.load(f)
    
    # Convertir a nuevo formato
    new_cache = {}
    for key, value in old_cache.items():
        # Adaptar formato si es necesario
        new_cache[key] = value
    
    # Guardar nuevo formato
    Path(new_cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_cache, new_cache_path)
    
    print(f"✅ Cache migrado: {len(new_cache)} entradas")

if __name__ == "__main__":
    migrate_cache_v1_to_v2(
        "/data/cache/old_cache.pkl",
        "/data/cache/new_cache.pt"
    )
```

### Migración de Base de Datos

```python
# scripts/migrate_database.py
import sqlalchemy
from sqlalchemy import text

def migrate_database():
    """Migra esquema de base de datos."""
    
    engine = sqlalchemy.create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Agregar nuevas columnas
        conn.execute(text("""
            ALTER TABLE cache_entries 
            ADD COLUMN IF NOT EXISTS session_id VARCHAR(255),
            ADD COLUMN IF NOT EXISTS metadata JSONB
        """))
        
        # Crear nuevos índices
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_session_id 
            ON cache_entries(session_id)
        """))
        
        conn.commit()
    
    print("✅ Base de datos migrada")

if __name__ == "__main__":
    migrate_database()
```

## 🔌 Compatibilidad

### Backward Compatibility

El sistema mantiene compatibilidad hacia atrás para:

- **v1.x API**: Funciona pero genera warnings
- **Configuraciones antiguas**: Migración automática
- **Cache antiguo**: Conversión automática al cargar

### Verificar Compatibilidad

```python
# scripts/check_compatibility.py
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine

def check_compatibility():
    """Verifica compatibilidad de versión."""
    
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Verificar versión
    version = engine.get_version()
    print(f"Versión actual: {version}")
    
    # Verificar características soportadas
    features = engine.get_supported_features()
    print(f"Características: {features}")
    
    # Verificar compatibilidad de config
    validation = engine.validate_configuration()
    if validation['is_valid']:
        print("✅ Configuración compatible")
    else:
        print(f"❌ Problemas de compatibilidad: {validation['issues']}")

if __name__ == "__main__":
    check_compatibility()
```

## ↩️ Rollback

### Rollback de Versión

```bash
# 1. Detener servicios
docker-compose down

# 2. Restaurar código anterior
git checkout v1.x

# 3. Restaurar configuración
cp .env.backup .env
cp config/kv_cache.yaml.backup config/kv_cache.yaml

# 4. Restaurar datos si es necesario
cp /backup/cache.old /data/cache/cache.pkl

# 5. Reiniciar
docker-compose up -d
```

### Rollback de Configuración

```python
# scripts/rollback_config.py
from bulk.core.ultra_adaptive_kv_cache_config_manager import ConfigManager

def rollback_config():
    """Reverte a configuración anterior."""
    
    manager = ConfigManager()
    
    # Listar backups disponibles
    backups = manager.list_backups()
    print("Backups disponibles:")
    for i, backup in enumerate(backups):
        print(f"{i+1}. {backup}")
    
    # Seleccionar backup
    choice = int(input("Selecciona backup (número): "))
    selected_backup = backups[choice - 1]
    
    # Restaurar
    manager.restore_from_backup(selected_backup)
    print(f"✅ Configuración restaurada desde {selected_backup}")

if __name__ == "__main__":
    rollback_config()
```

## 📋 Checklist de Migración

### Pre-Migración
- [ ] Backup completo del sistema
- [ ] Backup de base de datos
- [ ] Backup de cache persistente
- [ ] Backup de configuración
- [ ] Documentar versión actual
- [ ] Verificar requisitos de nueva versión

### Migración
- [ ] Actualizar código
- [ ] Migrar configuración
- [ ] Migrar datos si es necesario
- [ ] Actualizar variables de entorno
- [ ] Ejecutar tests

### Post-Migración
- [ ] Verificar funcionamiento
- [ ] Monitorear métricas
- [ ] Verificar logs
- [ ] Documentar problemas encontrados
- [ ] Actualizar documentación

---

**Más información:**
- [Changelog](CHANGELOG.md)
- [Release Notes](../releases/)
- [Troubleshooting](TROUBLESHOOTING_GUIDE.md)

