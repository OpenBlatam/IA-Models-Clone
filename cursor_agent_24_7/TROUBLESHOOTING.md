# 🔧 Troubleshooting - Cursor Agent 24/7

## Problemas Comunes y Soluciones

### ❌ El agente no inicia

**Síntomas:**
- Error al ejecutar `python main.py`
- El agente se detiene inmediatamente

**Soluciones:**
1. Verificar dependencias:
```bash
pip install -r requirements.txt
```

2. Verificar Python version:
```bash
python --version  # Debe ser 3.10+
```

3. Verificar permisos:
```bash
chmod +x main.py
chmod -R 755 data/
```

4. Verificar logs:
```bash
tail -f logs/agent.log
```

### ❌ El puerto 8024 está ocupado

**Síntomas:**
- Error: `Address already in use`
- No se puede conectar al puerto

**Soluciones:**
1. Cambiar puerto:
```bash
python main.py --port 8025
```

2. Encontrar proceso usando el puerto:
```bash
# Linux/macOS
lsof -i :8024
kill -9 <PID>

# Windows
netstat -ano | findstr :8024
taskkill /PID <PID> /F
```

### ❌ Las tareas no se ejecutan

**Síntomas:**
- Tareas quedan en estado "pending"
- No hay resultados

**Soluciones:**
1. Verificar que el agente esté corriendo:
```bash
curl http://localhost:8024/api/status
```

2. Verificar health:
```bash
curl http://localhost:8024/api/health
```

3. Verificar logs:
```bash
tail -f logs/agent.log | grep ERROR
```

4. Verificar rate limiting:
```bash
curl http://localhost:8024/api/rate-limit/stats
```

### ❌ Comandos son rechazados

**Síntomas:**
- Error: "Command validation failed"
- Comandos bloqueados

**Soluciones:**
1. Verificar que el comando no esté en la lista de bloqueados
2. Usar comandos más simples
3. Revisar validación:
```python
from cursor_agent_24_7.core.validators import CommandValidator

validator = CommandValidator()
result = validator.validate("tu_comando")
print(result.errors)
```

### ❌ El agente consume mucha memoria

**Síntomas:**
- Uso de memoria alto
- Sistema lento

**Soluciones:**
1. Reducir `max_concurrent_tasks`:
```python
config = AgentConfig(max_concurrent_tasks=3)
```

2. Limpiar tareas antiguas:
```bash
python scripts/maintenance.py cleanup --days 7
```

3. Limpiar caché:
```bash
curl -X POST http://localhost:8024/api/cache/clear
```

4. Reducir historial:
```python
config = AgentConfig(...)
# Reducir max_history en métricas
```

### ❌ Backups no se crean

**Síntomas:**
- No hay backups en `./data/backups`
- Error al crear backup

**Soluciones:**
1. Verificar permisos:
```bash
chmod -R 755 data/
```

2. Verificar espacio en disco:
```bash
df -h
```

3. Crear backup manual:
```bash
curl -X POST http://localhost:8024/api/backups/create
```

### ❌ WebSocket no funciona

**Síntomas:**
- No se puede conectar vía WebSocket
- Errores de conexión

**Soluciones:**
1. Verificar que el servidor esté corriendo
2. Verificar URL: `ws://localhost:8024/ws`
3. Verificar firewall
4. Probar con cliente WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:8024/ws');
ws.onopen = () => console.log('Connected');
ws.onerror = (e) => console.error('Error:', e);
```

### ❌ El file watcher no detecta cambios

**Síntomas:**
- Comandos escritos en archivo no se detectan
- No hay respuesta

**Soluciones:**
1. Verificar que watchdog esté instalado:
```bash
pip install watchdog
```

2. Verificar ruta del archivo:
```bash
ls -la data/commands.txt
```

3. Verificar permisos de escritura:
```bash
touch data/commands.txt
echo "test" > data/commands.txt
```

4. Usar polling mode si watchdog no funciona

### ❌ Métricas no se muestran

**Síntomas:**
- `/api/metrics` retorna vacío
- No hay estadísticas

**Soluciones:**
1. Verificar que métricas estén habilitadas
2. Ejecutar algunas tareas primero
3. Verificar estado:
```bash
curl http://localhost:8024/api/status | jq .metrics
```

### ❌ Error de importación

**Síntomas:**
- `ImportError` o `ModuleNotFoundError`

**Soluciones:**
1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Verificar entorno virtual:
```bash
which python
pip list | grep cursor-agent
```

3. Reinstalar:
```bash
pip install --upgrade -r requirements.txt
```

## 🔍 Diagnóstico

### Verificar estado completo

```bash
# Health check
curl http://localhost:8024/api/health | jq

# Estado
curl http://localhost:8024/api/status | jq

# Métricas
curl http://localhost:8024/api/metrics | jq

# Eventos
curl http://localhost:8024/api/events?limit=10 | jq
```

### Ver logs

```bash
# Logs del agente
tail -f logs/agent.log

# Buscar errores
grep ERROR logs/agent.log

# Buscar warnings
grep WARNING logs/agent.log
```

### Verificar recursos

```bash
# CPU y memoria
top -p $(pgrep -f "python.*main.py")

# Espacio en disco
df -h

# Procesos
ps aux | grep python
```

## 🆘 Obtener Ayuda

1. Revisar logs: `logs/agent.log`
2. Verificar health: `/api/health`
3. Revisar documentación: `README.md`
4. Ver ejemplos: `EXAMPLES.md`
5. Revisar API: `API_REFERENCE.md`

## 📝 Reportar Problemas

Al reportar un problema, incluir:

1. Versión de Python: `python --version`
2. Sistema operativo: `uname -a`
3. Logs relevantes: `tail -n 100 logs/agent.log`
4. Estado del agente: `curl http://localhost:8024/api/status`
5. Health check: `curl http://localhost:8024/api/health`
6. Pasos para reproducir



