# 📖 Runbook Operacional - Blatam Academy Features

## 🚨 Incidentes Críticos

### Incidente: Sistema Caído

**Severidad**: Crítica  
**Tiempo de respuesta**: < 5 minutos

#### Pasos Inmediatos

```
1. Verificar servicios
   docker-compose ps
   
2. Ver logs de errores
   docker-compose logs --tail=100 | grep -i error
   
3. Verificar recursos
   docker stats --no-stream
   
4. Reiniciar si es necesario
   docker-compose restart
```

#### Si el reinicio no funciona

```
1. Verificar configuración
   cat .env
   python -c "from bulk.core.ultra_adaptive_kv_cache_engine import KVCacheConfig; print(KVCacheConfig())"

2. Verificar base de datos
   docker-compose exec postgres psql -U postgres -c "SELECT 1"

3. Verificar Redis
   docker-compose exec redis redis-cli ping

4. Escalar si necesario
   Contactar: [escalation contact]
```

---

### Incidente: Alta Latencia

**Severidad**: Alta  
**Tiempo de respuesta**: < 15 minutos

#### Diagnóstico Rápido

```bash
# 1. Verificar métricas
curl http://localhost:9090/api/v1/query?query=kv_cache_latency_seconds

# 2. Verificar cache hit rate
python bulk/core/ultra_adaptive_kv_cache_cli.py stats

# 3. Verificar carga
docker stats
```

#### Soluciones Rápidas

```python
# Opción 1: Aumentar cache
config.max_tokens = 16384

# Opción 2: Habilitar prefetching
config.enable_prefetch = True
config.prefetch_size = 16

# Opción 3: Escalar horizontalmente
# Aumentar réplicas
```

---

### Incidente: Memoria Alta

**Severidad**: Media  
**Tiempo de respuesta**: < 30 minutos

#### Diagnóstico

```bash
# Verificar memoria
docker stats --no-stream
free -h
nvidia-smi  # Si hay GPU
```

#### Soluciones

```python
# Reducir cache size
config.max_tokens = 2048

# Habilitar compresión
config.use_compression = True
config.compression_ratio = 0.2

# Limpiar cache
engine.clear_cache()
```

---

### Incidente: Cache Hit Rate Bajo

**Severidad**: Media  
**Tiempo de respuesta**: < 1 hora

#### Diagnóstico

```python
stats = engine.get_stats()
hit_rate = stats['hit_rate']
print(f"Hit rate: {hit_rate:.2%}")
```

#### Soluciones

```python
# Aumentar cache size
config.max_tokens = 16384

# Cambiar a estrategia Adaptive
config.cache_strategy = CacheStrategy.ADAPTIVE

# Habilitar prefetching
config.enable_prefetch = True
```

---

## 🔄 Operaciones Diarias

### Morning Routine

```bash
# 1. Verificar servicios (30 seg)
docker-compose ps

# 2. Health check (30 seg)
./scripts/health_check.sh

# 3. Revisar logs de errores (1 min)
docker-compose logs --since 24h | grep -i error | tail -20

# 4. Verificar métricas clave (1 min)
# Prometheus dashboard
# O CLI: python bulk/core/ultra_adaptive_kv_cache_cli.py stats
```

### Evening Routine

```bash
# 1. Backup (si no está automatizado)
./backup_automation.sh

# 2. Verificar backups
ls -lh /backup/ | tail -5

# 3. Limpiar logs antiguos
docker-compose logs --tail=0

# 4. Verificar espacio en disco
df -h
```

## 🔧 Mantenimiento Regular

### Diario

- [ ] Health check
- [ ] Revisar logs de errores
- [ ] Verificar métricas clave
- [ ] Verificar backups

### Semanal

- [ ] Revisar métricas de la semana
- [ ] Limpiar backups antiguos
- [ ] Actualizar dependencias (si hay actualizaciones)
- [ ] Revisar seguridad

### Mensual

- [ ] Test de disaster recovery
- [ ] Revisar y optimizar configuración
- [ ] Auditoría de seguridad
- [ ] Revisar costos

## 📞 Escalación

### Nivel 1: Auto-resolución
- Consultar: [TROUBLESHOOTING_BY_SYMPTOM.md](TROUBLESHOOTING_BY_SYMPTOM.md)
- Tiempo: < 15 minutos

### Nivel 2: Consultar Documentación
- Consultar: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- Tiempo: < 1 hora

### Nivel 3: Escalar a Equipo
- Contactar: [team contact]
- Tiempo: < 4 horas

### Nivel 4: Escalación Crítica
- Contactar: [critical escalation]
- Tiempo: < 15 minutos

## 📊 Métricas de Monitoreo Diario

### Métricas Críticas (Check cada hora)

- Latencia P95 < 500ms
- Error rate < 1%
- Cache hit rate > 60%
- CPU < 80%
- Memory < 80%
- Disk space > 10%

### Métricas Importantes (Check cada 4 horas)

- Throughput > objetivo
- GPU utilization (si aplica)
- Database connections < límite
- Redis memory < límite

---

**Más información:**
- [Common Workflows](COMMON_WORKFLOWS.md)
- [Troubleshooting by Symptom](TROUBLESHOOTING_BY_SYMPTOM.md)
- [Production Ready](bulk/PRODUCTION_READY.md)

