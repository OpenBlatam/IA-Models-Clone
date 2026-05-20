# 🚀 Inicio Rápido - Cursor Agent 24/7

## Instalación en 3 pasos

### 1. Instalar dependencias

```bash
cd agents/backend/onyx/server/features/cursor_agent_24_7
pip install -r requirements.txt
```

### 2. Iniciar el agente

```bash
python main.py
```

O usar el script de inicio:

```bash
python start.py
```

### 3. Abrir el panel de control

Abre tu navegador en: **http://localhost:8024**

## 🎮 Uso Básico

1. **Iniciar el agente**: Haz clic en el botón "▶️ Iniciar"
2. **Agregar comando**: Escribe un comando en el campo de texto y presiona Enter
3. **Ver tareas**: Las tareas aparecerán en la lista debajo
4. **Detener**: Haz clic en "⏹️ Detener" cuando quieras parar

## 📡 Usar la API directamente

### Iniciar el agente

```bash
curl -X POST http://localhost:8024/api/start
```

### Agregar una tarea

```bash
curl -X POST http://localhost:8024/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"command": "optimiza este código"}'
```

### Ver estado

```bash
curl http://localhost:8024/api/status
```

### Ver tareas

```bash
curl http://localhost:8024/api/tasks?limit=10
```

### Detener el agente

```bash
curl -X POST http://localhost:8024/api/stop
```

## 🔧 Configuración Avanzada

Copia `.env.example` a `.env` y edita las variables:

```bash
cp .env.example .env
# Edita .env con tus preferencias
```

## 🛠️ Ejecutar como Servicio

### Windows

```bash
# Usar NSSM o Task Scheduler
python main.py --mode service
```

### Linux

```bash
# Crear servicio systemd (ver README.md)
sudo systemctl start cursor-agent-24-7
```

### macOS

```bash
# Usar launchd (ver README.md)
launchctl load ~/Library/LaunchAgents/com.cursor.agent24-7.plist
```

## ❓ Problemas Comunes

### El puerto 8024 está ocupado

Cambia el puerto:

```bash
python main.py --port 8025
```

### El agente no guarda el estado

Verifica que el directorio `data/` tenga permisos de escritura:

```bash
mkdir -p data
chmod 755 data
```

### No se ejecutan las tareas

Asegúrate de que el agente esté en estado "running" antes de agregar tareas.

## 📚 Más Información

Ver [README.md](README.md) para documentación completa.


