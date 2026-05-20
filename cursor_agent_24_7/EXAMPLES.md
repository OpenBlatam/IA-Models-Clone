# 📚 Ejemplos de Uso - Cursor Agent 24/7

## 🎯 Ejemplos Básicos

### 1. Comando Python Simple

```python
# Escribir en ./data/commands.txt o enviar por API
print("Hola desde el agente!")
```

### 2. Cálculos y Operaciones

```python
# Calcular algo
result = 10 * 20 + 5
print(f"Resultado: {result}")

# Trabajar con listas
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Cuadrados: {squared}")
```

### 3. Procesar Archivos

```python
# Leer y procesar archivos
from pathlib import Path

files = list(Path('.').glob('*.py'))
print(f"Encontrados {len(files)} archivos Python")

for file in files[:5]:
    size = file.stat().st_size
    print(f"  {file.name}: {size} bytes")
```

## 🔧 Ejemplos Intermedios

### 4. Llamadas API

```python
# Llamar a una API externa
import requests

response = requests.get('https://api.github.com/users/octocat')
data = response.json()
print(f"Usuario: {data['login']}")
print(f"Repositorios públicos: {data['public_repos']}")
```

### 5. Procesamiento de Datos

```python
# Procesar datos JSON
import json

data = {
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ]
}

for user in data["users"]:
    print(f"{user['name']} tiene {user['age']} años")
```

### 6. Trabajar con Fechas

```python
# Operaciones con fechas
from datetime import datetime, timedelta

now = datetime.now()
yesterday = now - timedelta(days=1)
print(f"Hoy: {now.strftime('%Y-%m-%d')}")
print(f"Ayer: {yesterday.strftime('%Y-%m-%d')}")
```

## 🚀 Ejemplos Avanzados

### 7. Análisis de Archivos

```python
# Analizar archivos en el proyecto
from pathlib import Path
import os

def analyze_project():
    total_files = 0
    total_size = 0
    by_extension = {}
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            filepath = Path(root) / file
            if filepath.is_file():
                total_files += 1
                size = filepath.stat().st_size
                total_size += size
                
                ext = filepath.suffix or 'no_ext'
                by_extension[ext] = by_extension.get(ext, 0) + 1
    
    print(f"Total archivos: {total_files}")
    print(f"Tamaño total: {total_size / 1024 / 1024:.2f} MB")
    print("\nPor extensión:")
    for ext, count in sorted(by_extension.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ext}: {count}")

analyze_project()
```

### 8. Monitoreo del Sistema

```python
# Monitorear recursos del sistema
import psutil

cpu_percent = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory()
disk = psutil.disk_usage('/')

print(f"CPU: {cpu_percent}%")
print(f"Memoria: {memory.percent}% ({memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB)")
print(f"Disco: {disk.percent}% ({disk.used / 1024**3:.2f} GB / {disk.total / 1024**3:.2f} GB)")
```

### 9. Generación de Reportes

```python
# Generar un reporte
from datetime import datetime

report = {
    "timestamp": datetime.now().isoformat(),
    "status": "running",
    "tasks_completed": 42,
    "tasks_failed": 3,
    "uptime_hours": 24.5
}

print("=== REPORTE DEL AGENTE ===")
for key, value in report.items():
    print(f"{key}: {value}")
```

## 🌐 Ejemplos con API

### 10. Enviar Comando por API

```bash
# Usando curl
curl -X POST http://localhost:8024/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"command": "print(\"Hola desde API!\")"}'
```

### 11. Obtener Estado

```bash
# Ver estado del agente
curl http://localhost:8024/api/status | jq
```

### 12. Ver Tareas

```bash
# Ver últimas 10 tareas
curl http://localhost:8024/api/tasks?limit=10 | jq
```

### 13. Ver Métricas

```bash
# Ver métricas del agente
curl http://localhost:8024/api/metrics | jq
```

### 14. Ver Notificaciones

```bash
# Ver notificaciones no leídas
curl http://localhost:8024/api/notifications?unread_only=true | jq
```

## 🔌 Ejemplos con WebSocket

### 15. Conexión WebSocket (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8024/ws');

ws.onopen = () => {
    console.log('Conectado al agente');
    
    // Enviar comando
    ws.send(JSON.stringify({
        type: 'command',
        command: 'print("Hola desde WebSocket!")'
    }));
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Mensaje recibido:', message);
};

ws.onerror = (error) => {
    console.error('Error:', error);
};

ws.onclose = () => {
    console.log('Desconectado');
};
```

### 16. Conexión WebSocket (Python)

```python
import asyncio
import websockets
import json

async def connect_agent():
    uri = "ws://localhost:8024/ws"
    
    async with websockets.connect(uri) as websocket:
        # Enviar comando
        await websocket.send(json.dumps({
            "type": "command",
            "command": "print('Hola desde Python WebSocket!')"
        }))
        
        # Recibir respuesta
        response = await websocket.recv()
        print(f"Respuesta: {response}")

asyncio.run(connect_agent())
```

## 🛠️ Ejemplos de Automatización

### 17. Tarea Programada

```python
# Ejecutar tarea cada cierto tiempo
import time
from datetime import datetime

def scheduled_task():
    print(f"[{datetime.now()}] Ejecutando tarea programada...")
    # Tu código aquí
    print("Tarea completada")

scheduled_task()
```

### 18. Procesamiento en Lote

```python
# Procesar múltiples items
items = ["item1", "item2", "item3", "item4", "item5"]

results = []
for item in items:
    # Procesar cada item
    result = f"Procesado: {item}"
    results.append(result)
    print(result)

print(f"\nTotal procesado: {len(results)} items")
```

### 19. Validación de Datos

```python
# Validar y procesar datos
def validate_email(email):
    return "@" in email and "." in email.split("@")[1]

emails = [
    "user@example.com",
    "invalid-email",
    "another@test.org"
]

valid_emails = [email for email in emails if validate_email(email)]
print(f"Emails válidos: {valid_emails}")
```

## 📊 Ejemplos de Análisis

### 20. Análisis de Logs

```python
# Analizar logs (ejemplo)
log_entries = [
    "2024-01-01 INFO: Task started",
    "2024-01-01 ERROR: Task failed",
    "2024-01-01 INFO: Task completed"
]

errors = [entry for entry in log_entries if "ERROR" in entry]
print(f"Total errores: {len(errors)}")
for error in errors:
    print(f"  - {error}")
```

### 21. Estadísticas de Tareas

```python
# Calcular estadísticas
tasks = [
    {"status": "completed", "duration": 5.2},
    {"status": "completed", "duration": 3.1},
    {"status": "failed", "duration": 1.5},
    {"status": "completed", "duration": 7.8}
]

completed = [t for t in tasks if t["status"] == "completed"]
avg_duration = sum(t["duration"] for t in completed) / len(completed) if completed else 0

print(f"Tareas completadas: {len(completed)}/{len(tasks)}")
print(f"Duración promedio: {avg_duration:.2f}s")
```

## 🔐 Ejemplos de Seguridad

### 22. Validación de Entrada

```python
# Validar entrada antes de procesar
def safe_eval(expression):
    # Solo permitir operaciones seguras
    allowed_chars = set("0123456789+-*/(). ")
    if all(c in allowed_chars for c in expression):
        try:
            return eval(expression)
        except:
            return None
    return None

result = safe_eval("10 + 20 * 2")
print(f"Resultado: {result}")
```

## 💡 Tips y Mejores Prácticas

1. **Usa try/except** para manejar errores
2. **Valida datos** antes de procesar
3. **Usa logging** para debugging
4. **Limpia recursos** después de usar
5. **Documenta comandos complejos**
6. **Prueba comandos** antes de ejecutarlos en producción

## 📚 Más Recursos

- Ver [USAGE.md](USAGE.md) para guía de uso completa
- Ver [README.md](README.md) para documentación general
- Ver [QUICK_START.md](QUICK_START.md) para inicio rápido



