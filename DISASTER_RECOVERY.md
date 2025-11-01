# 🚨 Guía de Disaster Recovery - Blatam Academy Features

## 📋 Plan de Disaster Recovery

### Procedimientos de Emergencia

#### Escenario 1: Pérdida Completa del Sistema

```
1. Evaluar daño
   - Identificar qué componentes están afectados
   - Determinar tiempo de inactividad aceptable (RTO)
   - Determinar pérdida de datos aceptable (RPO)

2. Activar plan de recuperación
   - Notificar al equipo
   - Activar sitio de respaldo (si existe)

3. Restaurar desde backups
   - Database: Restaurar último backup
   - Cache: Restaurar desde persistencia
   - Configuración: Restaurar desde version control

4. Verificar funcionamiento
   - Health checks
   - Smoke tests
   - Verificar integridad de datos

5. Monitorear post-recuperación
   - Métricas clave
   - Errores
   - Performance
```

#### Escenario 2: Corrupción de Datos

```
1. Detener escrituras
   - Detener servicios que escriben datos

2. Evaluar corrupción
   - Verificar scope del problema
   - Identificar datos afectados

3. Restaurar datos limpios
   - Desde backup más reciente antes de corrupción
   - Verificar integridad antes de restaurar

4. Re-iniciar servicios gradualmente
   - Monitorear cada servicio
   - Verificar no hay más corrupción
```

#### Escenario 3: Ataque de Seguridad

```
1. Aislar sistema
   - Desconectar de red
   - Detener servicios afectados

2. Evaluar brecha
   - Identificar qué fue comprometido
   - Determinar datos expuestos

3. Contener
   - Cambiar todas las credenciales
   - Revocar tokens/keys
   - Bloquear IPs atacantes

4. Limpiar y restaurar
   - Limpiar sistemas comprometidos
   - Restaurar desde backups limpios

5. Notificar
   - Stakeholders
   - Autoridades (si aplica)

6. Post-mortem
   - Identificar causa raíz
   - Mejorar controles de seguridad
```

## 💾 Estrategias de Backup

### Backup Automatizado

```python
# backup_automation.py
import schedule
import time
from datetime import datetime
import os

def backup_database():
    """Backup automático de base de datos."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"/backup/db_backup_{timestamp}.sql"
    
    os.system(f"docker-compose exec -T postgres pg_dump -U postgres dbname > {backup_file}")
    
    # Comprimir
    os.system(f"gzip {backup_file}")
    
    # Limpiar backups antiguos (>30 días)
    cleanup_old_backups("/backup", days=30)
    
    print(f"✅ Database backed up: {backup_file}.gz")

def backup_cache():
    """Backup automático de cache."""
    from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig
    
    config = KVCacheConfig(enable_persistence=True, persistence_path="/data/cache")
    engine = UltraAdaptiveKVCacheEngine(config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"/backup/cache_backup_{timestamp}.pt"
    
    engine.persist(backup_path)
    
    # Comprimir
    os.system(f"gzip {backup_path}")
    
    print(f"✅ Cache backed up: {backup_path}.gz")

# Programar backups
schedule.every(6).hours.do(backup_database)
schedule.every(12).hours.do(backup_cache)

# Ejecutar en background
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Backup de Configuración

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/backup/config"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup .env
cp .env $BACKUP_DIR/env_backup_$DATE

# Backup configs
cp config/*.yaml $BACKUP_DIR/

# Backup docker-compose
cp docker-compose.yml $BACKUP_DIR/docker-compose_backup_$DATE.yml

echo "✅ Configuration backed up to $BACKUP_DIR"
```

## 🔄 Procedimientos de Restore

### Restore Completo del Sistema

```bash
#!/bin/bash
# restore_complete.sh

BACKUP_DATE=$1  # Ej: 20240115_120000

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: ./restore_complete.sh YYYYMMDD_HHMMSS"
    exit 1
fi

echo "🔄 Restoring system from backup: $BACKUP_DATE"

# 1. Detener servicios
docker-compose down

# 2. Restaurar database
echo "Restoring database..."
gunzip /backup/db_backup_${BACKUP_DATE}.sql.gz
docker-compose exec -T postgres psql -U postgres dbname < /backup/db_backup_${BACKUP_DATE}.sql

# 3. Restaurar cache
echo "Restoring cache..."
gunzip /backup/cache_backup_${BACKUP_DATE}.pt.gz
python -c "
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig
config = KVCacheConfig(enable_persistence=True, persistence_path='/data/cache')
engine = UltraAdaptiveKVCacheEngine(config)
engine.load('/backup/cache_backup_${BACKUP_DATE}.pt')
print('Cache restored')
"

# 4. Restaurar configuración
echo "Restoring configuration..."
cp /backup/config/env_backup_${BACKUP_DATE} .env
cp /backup/config/*.yaml config/

# 5. Reiniciar servicios
docker-compose up -d

# 6. Verificar
./scripts/health_check.sh

echo "✅ System restored from backup: $BACKUP_DATE"
```

### Restore Selectivo

```python
# restore_selective.py
from bulk.core.ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine, KVCacheConfig

def restore_cache_only(backup_path: str):
    """Restaurar solo cache."""
    config = KVCacheConfig(
        enable_persistence=True,
        persistence_path='/data/cache'
    )
    engine = UltraAdaptiveKVCacheEngine(config)
    
    # Descomprimir si es necesario
    if backup_path.endswith('.gz'):
        import gzip
        import shutil
        decompressed = backup_path[:-3]
        with gzip.open(backup_path, 'rb') as f_in:
            with open(decompressed, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        backup_path = decompressed
    
    engine.load(backup_path)
    print(f"✅ Cache restored from {backup_path}")
```

## 🔐 Backup de Seguridad

### Backup Encriptado

```python
# encrypted_backup.py
from cryptography.fernet import Fernet
import os

def create_encrypted_backup(backup_path: str, encryption_key: bytes):
    """Crear backup encriptado."""
    cipher = Fernet(encryption_key)
    
    # Leer backup
    with open(backup_path, 'rb') as f:
        backup_data = f.read()
    
    # Encriptar
    encrypted_data = cipher.encrypt(backup_data)
    
    # Guardar encriptado
    encrypted_path = f"{backup_path}.encrypted"
    with open(encrypted_path, 'wb') as f:
        f.write(encrypted_data)
    
    print(f"✅ Encrypted backup created: {encrypted_path}")

def restore_encrypted_backup(encrypted_path: str, encryption_key: bytes, output_path: str):
    """Restaurar desde backup encriptado."""
    cipher = Fernet(encryption_key)
    
    # Leer encriptado
    with open(encrypted_path, 'rb') as f:
        encrypted_data = f.read()
    
    # Desencriptar
    backup_data = cipher.decrypt(encrypted_data)
    
    # Guardar desencriptado
    with open(output_path, 'wb') as f:
        f.write(backup_data)
    
    print(f"✅ Backup decrypted to: {output_path}")
```

## 📊 RTO y RPO Objetivos

### Recovery Time Objective (RTO)

| Componente | RTO Objetivo | Estrategia |
|-----------|--------------|------------|
| Database | 15 minutos | Hot standby |
| Cache | 5 minutos | Persistencia + Warmup |
| Configuración | 2 minutos | Version control |
| Sistema completo | 30 minutos | Disaster recovery site |

### Recovery Point Objective (RPO)

| Componente | RPO Objetivo | Frecuencia Backup |
|------------|--------------|-------------------|
| Database | 1 hora | Cada hora |
| Cache | 6 horas | Cada 6 horas |
| Configuración | 24 horas | Diario |
| Logs | 7 días | Semanal |

## ✅ Checklist de Disaster Recovery

### Pre-Desastre
- [ ] Backups automatizados configurados
- [ ] Backups verificados (test restore)
- [ ] Plan de DR documentado
- [ ] Equipo entrenado en procedimientos
- [ ] Contactos de emergencia listados
- [ ] Sitio de respaldo configurado (si aplica)

### Durante Desastre
- [ ] Daño evaluado
- [ ] Equipo notificado
- [ ] Plan de recuperación activado
- [ ] Backups localizados
- [ ] Restauración en progreso

### Post-Desastre
- [ ] Sistema restaurado
- [ ] Funcionalidad verificada
- [ ] Datos verificados
- [ ] Monitoreo activo
- [ ] Post-mortem programado
- [ ] Mejoras identificadas

## 🔄 Test de Disaster Recovery

### Test Regular (Mensual)

```bash
#!/bin/bash
# test_dr.sh

echo "🧪 Testing Disaster Recovery Plan"

# 1. Crear backup de test
./backup_complete.sh

# 2. Simular desastre (crear VM/container de test)
docker-compose -f docker-compose.test.yml up -d

# 3. Restaurar en ambiente de test
./restore_complete.sh $(date +%Y%m%d_%H%M%S)

# 4. Verificar funcionalidad
./scripts/health_check.sh
pytest tests/integration/test_disaster_recovery.py

# 5. Limpiar
docker-compose -f docker-compose.test.yml down

echo "✅ DR test completed"
```

---

**Más información:**
- [Backup Procedures](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md#backup-and-restore)
- [Production Ready](bulk/PRODUCTION_READY.md)
- [Security Guide](SECURITY_GUIDE.md)

