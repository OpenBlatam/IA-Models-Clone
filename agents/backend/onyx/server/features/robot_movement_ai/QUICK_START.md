# Guía de Inicio Rápido - Robot Movement AI

## 🚀 Inicio Rápido en 5 Minutos

### 1. Instalación

```bash
cd robot_movement_ai
pip install -r requirements.txt
```

### 2. Configuración Básica

Crear archivo `.env`:

```env
ROBOT_IP=192.168.1.100
ROBOT_PORT=30001
ROBOT_BRAND=generic
FEEDBACK_FREQUENCY=1000
```

### 3. Iniciar Servidor

```bash
python -m robot_movement_ai.main
```

### 4. Probar API

```bash
# Estado del robot
curl http://localhost:8010/api/v1/status

# Chat - Mover robot
curl -X POST http://localhost:8010/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "move to (0.5, 0.3, 0.2)"}'
```

## 📝 Comandos de Chat Disponibles

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `move to (x, y, z)` | Mover a posición absoluta | `move to (0.5, 0.3, 0.2)` |
| `move relative (dx, dy, dz)` | Mover relativamente | `move relative (0.1, 0, -0.05)` |
| `stop` | Detener movimiento | `stop` |
| `go home` | Ir a posición home | `go home` |
| `status` | Estado del robot | `status` |

## 🔧 Configuración por Marca de Robot

### KUKA

```env
ROBOT_BRAND=kuka
ROBOT_IP=192.168.1.100
```

### ABB

```env
ROBOT_BRAND=abb
ROBOT_IP=192.168.1.100
```

### Universal Robots

```env
ROBOT_BRAND=universal_robots
ROBOT_IP=192.168.1.100
```

## 🌐 WebSocket para Chat en Tiempo Real

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8010/ws/chat"
    async with websockets.connect(uri) as websocket:
        # Enviar comando
        await websocket.send("move to (0.5, 0.3, 0.2)")
        
        # Recibir respuesta
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(chat())
```

## 📊 Monitoreo

- **API Docs**: http://localhost:8010/docs
- **Health Check**: http://localhost:8010/health
- **Status**: http://localhost:8010/api/v1/status

## ⚠️ Notas Importantes

1. **Modo Simulación**: Por defecto, el sistema corre en modo simulación si no hay conexión real al robot
2. **Seguridad**: Siempre verifica los límites de seguridad antes de usar con robots reales
3. **ROS**: La integración ROS es opcional y requiere ROS2 instalado

## 🆘 Solución de Problemas

### Puerto en Uso

```bash
python -m robot_movement_ai.main --port 8011
```

### Sin Conexión al Robot

El sistema funcionará en modo simulación. Los comandos se procesarán pero no se enviarán al hardware.

### Error de Importación

Asegúrate de instalar todas las dependencias:

```bash
pip install -r requirements.txt --upgrade
```






