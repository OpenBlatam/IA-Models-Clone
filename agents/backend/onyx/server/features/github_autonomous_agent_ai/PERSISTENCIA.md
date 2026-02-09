# 🔄 Persistencia del Agente - No se Detiene Automáticamente

## ⚠️ Comportamiento Importante

El **GitHub Autonomous Agent AI** está configurado para **NO detenerse automáticamente**. Solo se detendrá cuando:

1. **Presiones el botón "Detener" en la interfaz web**
2. **Llamas explícitamente al endpoint `/api/agent/stop`**
3. **Detienes el proceso manualmente desde el sistema operativo**

## ✅ Lo que NO detiene el agente:

- ❌ Errores en el procesamiento de tareas
- ❌ Interrupciones de teclado (KeyboardInterrupt)
- ❌ Excepciones no manejadas
- ❌ Cambios de estado inesperados
- ❌ Problemas de conexión temporales
- ❌ Reinicios del navegador o cierre de la pestaña del frontend

## 🔧 Configuración Actual

### Backend (Python)

El agente está configurado con:

1. **Loop principal persistente**: El loop `_run_loop()` continúa ejecutándose indefinidamente
2. **Auto-recuperación**: Si hay un error, el agente intenta recuperarse automáticamente
3. **Reinicio automático**: Si el agente se detiene inesperadamente, el servicio lo reinicia
4. **Manejo robusto de errores**: Los errores se registran pero no detienen el agente

### Frontend (Next.js)

El frontend:

1. **Procesa tareas continuamente**: Las tareas se procesan automáticamente sin parar
2. **Persistencia local**: Las tareas se guardan en localStorage y se reanudan al recargar
3. **Botón de parar**: Solo se detiene cuando presionas el botón "Detener" en cada tarea

## 🚀 Iniciar el Agente

### Modo Manual

```bash
# Iniciar como servicio persistente
python -m github_autonomous_agent_ai.main --mode service

# O usar el script batch
start_service.bat
```

### Modo API (con frontend)

```bash
# Iniciar servidor API
python -m github_autonomous_agent_ai.main --mode api --port 8025
```

## 💻 Ejecutar al Iniciar la Computadora

### Opción 1: Script de Inicio Automático (Windows)

1. Presiona `Win + R`
2. Escribe `shell:startup` y presiona Enter
3. Crea un acceso directo a `start_service.bat`

### Opción 2: Servicio de Windows (Recomendado)

Usa el script `install_as_service.bat` (requiere ejecutar como Administrador):

```bash
# Ejecutar como Administrador
install_as_service.bat
```

Luego sigue las instrucciones para instalar con NSSM o el Programador de Tareas.

### Opción 3: Tarea Programada (Windows)

1. Abre el **Programador de tareas** de Windows
2. Crea una **tarea nueva**
3. Configura:
   - **Desencadenador**: "Al iniciar sesión" o "Al iniciar el equipo"
   - **Acción**: Iniciar programa → `start_service.bat`
   - **Condiciones**: Marca "Iniciar la tarea aunque el usuario no haya iniciado sesión"
   - **Configuración**: Marca "Permitir ejecutar la tarea a petición"

### Opción 4: Servidor Remoto / Cloud

Para que el agente continúe ejecutándose incluso si tu computadora está apagada:

1. **VPS (Virtual Private Server)**: Contrata un VPS (DigitalOcean, AWS, etc.)
2. **Cloud Services**: Usa servicios como Heroku, Railway, Render, etc.
3. **Docker**: Ejecuta en un contenedor Docker en un servidor remoto

## 🛑 Detener el Agente

### Desde la Interfaz Web

1. Abre la interfaz web del agente
2. Ve a la sección de control del agente
3. Presiona el botón **"Detener"** o **"⏹️ Detener"**

### Desde la API

```bash
# Detener el agente
curl -X POST http://localhost:8025/api/agent/stop
```

### Desde el Sistema Operativo

```bash
# Encontrar el proceso
tasklist | findstr python

# Detener el proceso (Windows)
taskkill /PID <PID> /F

# O en Linux/Mac
kill <PID>
```

## 📝 Notas Importantes

1. **Persistencia de Tareas**: Las tareas se guardan en la base de datos SQLite, por lo que si el agente se reinicia, las tareas pendientes se reanudan automáticamente.

2. **Logs**: Revisa los logs para ver el estado del agente:
   ```bash
   # Los logs se muestran en la consola donde ejecutaste el servicio
   ```

3. **Recursos**: El agente consume recursos mientras está ejecutándose. Asegúrate de tener suficiente RAM y CPU disponibles.

4. **Seguridad**: Si ejecutas el agente en un servidor remoto, asegúrate de configurar correctamente el firewall y las credenciales.

## 🔍 Verificar que el Agente Está Ejecutándose

### Desde la Interfaz Web

- Ve a la página de control del agente
- Verifica que el estado muestre "RUNNING" o "Ejecutándose"

### Desde la API

```bash
# Verificar estado
curl http://localhost:8025/api/agent/status
```

### Desde el Sistema Operativo

```bash
# Windows
tasklist | findstr python

# Linux/Mac
ps aux | grep python
```

## 🐛 Solución de Problemas

### El agente se detiene automáticamente

1. Revisa los logs para ver qué error causó la detención
2. Verifica que los cambios en `core/agent.py` y `core/service.py` estén aplicados
3. Asegúrate de que no hay otro proceso que esté matando el agente

### El agente no inicia al arrancar la computadora

1. Verifica que el script de inicio tenga permisos de ejecución
2. Revisa el Programador de Tareas para ver si hay errores
3. Asegúrate de que Python esté en el PATH del sistema

### Las tareas no se procesan

1. Verifica que el agente esté en estado "RUNNING"
2. Revisa los logs para ver si hay errores
3. Verifica la conexión a GitHub y las credenciales

