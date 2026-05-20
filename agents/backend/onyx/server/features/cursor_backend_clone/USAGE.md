# 📖 Guía de Uso - Cursor Agent 24/7

## 🚀 Inicio Rápido

### 1. Iniciar el Agente

```bash
python main.py
```

O usar el script de inicio:

```bash
python start.py
```

### 2. Enviar Comandos

El agente monitorea un archivo de comandos. Por defecto está en: `./data/commands.txt`

#### Opción 1: Escribir directamente al archivo

```bash
# En otra terminal o desde tu editor
echo "print('Hola desde el agente!')" > ./data/commands.txt
```

#### Opción 2: Usar la API

```bash
curl -X POST http://localhost:8024/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"command": "print(\"Hola desde API!\")"}'
```

#### Opción 3: Usar la interfaz web

Abre `http://localhost:8024` y escribe comandos en el campo de texto.

## 📝 Tipos de Comandos

### Comandos Python

```python
# Escribir en commands.txt o enviar por API
print("Hola mundo")
x = 10 + 20
print(f"Resultado: {x}")
```

O con prefijo:

```
python: print("Hola")
py: import os; print(os.getcwd())
```

### Comandos Shell

```bash
shell: ls -la
sh: echo "Hello from shell"
```

### Llamadas API

```
http://api.example.com/data
https://jsonplaceholder.typicode.com/posts/1
```

## 🎯 Ejemplos Prácticos

### Ejemplo 1: Procesar Archivos

```python
# Comando en commands.txt
import os
from pathlib import Path

files = list(Path('.').glob('*.py'))
print(f"Encontrados {len(files)} archivos Python")
for f in files[:5]:
    print(f"  - {f}")
```

### Ejemplo 2: Ejecutar Script Externo

```bash
# Comando shell
shell: python scripts/process_data.py
```

### Ejemplo 3: Llamar API Externa

```
http://httpbin.org/get
```

### Ejemplo 4: Tarea Compleja

```python
# Comando Python complejo
import json
import requests

response = requests.get('https://api.github.com/users/octocat')
data = response.json()
print(f"Usuario: {data['login']}")
print(f"Repos: {data['public_repos']}")
```

## 📊 Ver Estado y Resultados

### Ver Estado del Agente

```bash
curl http://localhost:8024/api/status
```

### Ver Tareas

```bash
curl http://localhost:8024/api/tasks?limit=10
```

### Interfaz Web

Abre `http://localhost:8024` para ver:
- Estado del agente
- Lista de tareas
- Resultados de ejecución

## ⚙️ Configuración Avanzada

### Cambiar Archivo de Comandos

```python
from cursor_agent_24_7.core.agent import CursorAgent, AgentConfig

config = AgentConfig(
    command_file="./mi_archivo_comandos.txt",
    storage_path="./data/agent_state.json"
)

agent = CursorAgent(config)
```

### Monitorear Directorio

```python
config = AgentConfig(
    watch_directory="./comandos/",
    storage_path="./data/agent_state.json"
)
```

### Configurar Timeout

```python
config = AgentConfig(
    task_timeout=600.0,  # 10 minutos
    max_concurrent_tasks=10
)
```

## 🔄 Flujo de Trabajo

1. **Iniciar el agente**: `python main.py`
2. **Escribir comando**: En `./data/commands.txt` o vía API
3. **El agente detecta**: Automáticamente lee el comando
4. **Ejecuta**: Procesa el comando según su tipo
5. **Guarda resultado**: En el estado del agente
6. **Ver resultado**: En la API o interfaz web

## 🛠️ Troubleshooting

### El agente no detecta comandos

1. Verifica que el archivo existe: `./data/commands.txt`
2. Verifica permisos de escritura
3. Revisa los logs del agente

### Comandos no se ejecutan

1. Verifica que el agente esté en estado "running"
2. Revisa los logs para ver errores
3. Verifica que las dependencias estén instaladas

### Timeout en comandos

Aumenta el timeout en la configuración:

```python
config = AgentConfig(task_timeout=600.0)  # 10 minutos
```

## 💡 Tips

1. **Usa prefijos** para especificar el tipo de comando
2. **Limpia el archivo** después de escribir (el agente lo hace automáticamente)
3. **Revisa los logs** para debugging
4. **Usa la API** para integración con otros sistemas
5. **Monitorea el estado** regularmente

## 📚 Más Información

- Ver [README.md](README.md) para documentación completa
- Ver [QUICK_START.md](QUICK_START.md) para inicio rápido
- Ver [LIBRARIES.md](LIBRARIES.md) para información de librerías


