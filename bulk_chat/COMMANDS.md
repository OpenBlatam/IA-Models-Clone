# Comandos Útiles - Bulk Chat

## 🚀 Inicio Rápido

### Scripts de Inicio

**Windows:**
```bash
start.bat
start.bat openai
start.bat mock 9000
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
./start.sh openai
./start.sh mock 9000
```

**Python (Multiplataforma):**
```bash
python -m bulk_chat.main
python -m bulk_chat.main --llm-provider mock
python -m bulk_chat.main --llm-provider openai --port 8006
```

### Script de Comandos (run.py)

```bash
# Iniciar servidor
python run.py server
python run.py server --provider openai --port 8006

# Verificar setup
python run.py verify

# Instalar dependencias
python run.py install

# Ver estado del sistema
python run.py status

# Ejecutar tests
python run.py test
```

## 📦 Instalación y Configuración

### Instalación

```bash
# Instalación automática
python install.py

# Instalación manual
pip install -r requirements.txt

# Verificar instalación
python verify_setup.py
```

### Configuración

```bash
# Crear archivo .env
cp .env.example .env

# Editar .env con tus API keys
# OPENAI_API_KEY=tu-api-key-aqui
```

## 🔧 Operaciones Comunes

### Iniciar Servidor

```bash
# Modo básico (mock)
python -m bulk_chat.main --llm-provider mock

# Con OpenAI
export OPENAI_API_KEY=tu-api-key
python -m bulk_chat.main --llm-provider openai

# Con opciones personalizadas
python -m bulk_chat.main \
  --llm-provider openai \
  --llm-model gpt-4 \
  --port 8006 \
  --host 0.0.0.0 \
  --debug
```

### Verificar Estado

```bash
# Verificar que el servidor está corriendo
curl http://localhost:8006/health

# Ver estado completo
python run.py status
```

### Testing

```bash
# Ejecutar todos los tests
python run.py test

# Ejecutar tests específicos
pytest tests/test_chat_engine.py -v

# Ejecutar con coverage
pytest --cov=bulk_chat tests/
```

## 🌐 API Endpoints Útiles

### Health Check

```bash
curl http://localhost:8006/health
```

### Crear Sesión

```bash
curl -X POST "http://localhost:8006/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Hola",
    "auto_continue": true
  }'
```

### Ver Mensajes

```bash
curl "http://localhost:8006/api/v1/chat/sessions/{session_id}/messages"
```

### Pausar/Reanudar

```bash
# Pausar
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/pause"

# Reanudar
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/resume"
```

### Documentación

```bash
# Abrir en navegador
http://localhost:8006/docs          # Swagger UI
http://localhost:8006/redoc        # ReDoc
http://localhost:8006/dashboard    # Dashboard
```

## 🛠️ Desarrollo

### Modo Debug

```bash
python -m bulk_chat.main --debug
```

### Logs

```bash
# Ver logs en tiempo real
tail -f logs/bulk_chat.log

# Buscar errores
grep ERROR logs/bulk_chat.log
```

### Limpiar Datos

```bash
# Limpiar sesiones
rm -rf sessions/*

# Limpiar backups
rm -rf backups/*

# Limpiar logs
rm -rf logs/*
```

## 📊 Monitoreo

### Métricas

```bash
# Métricas globales
curl http://localhost:8006/api/v1/chat/metrics

# Métricas de sesión
curl http://localhost:8006/api/v1/chat/sessions/{session_id}/metrics
```

### Performance

```bash
# Métricas de rendimiento
curl http://localhost:8006/api/v1/performance/metrics

# Operaciones lentas
curl http://localhost:8006/api/v1/performance/slow-operations
```

## 🔍 Troubleshooting

### Puerto en Uso

```bash
# Windows
netstat -ano | findstr :8006

# Linux/Mac
lsof -i :8006

# Usar otro puerto
python -m bulk_chat.main --port 8007
```

### Dependencias Faltantes

```bash
# Verificar
python verify_setup.py

# Reinstalar
pip install -r requirements.txt --upgrade
```

### Problemas de Importación

```bash
# Verificar path
python -c "import sys; print(sys.path)"

# Agregar al path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🚢 Producción

### Iniciar como Servicio

**systemd (Linux):**
```ini
[Unit]
Description=Bulk Chat Service
After=network.target

[Service]
Type=simple
User=bulk_chat
WorkingDirectory=/path/to/bulk_chat
ExecStart=/usr/bin/python3 -m bulk_chat.main --llm-provider openai
Restart=always

[Install]
WantedBy=multi-user.target
```

**Windows Service:**
Usar NSSM (Non-Sucking Service Manager) o crear un servicio con Python.

### Docker (Futuro)

```bash
# Build
docker build -t bulk-chat .

# Run
docker run -p 8006:8006 -e OPENAI_API_KEY=xxx bulk-chat
```

## 📝 Scripts Útiles

### Backup Manual

```bash
curl -X POST "http://localhost:8006/api/v1/chat/backup/create"
```

### Exportar Sesión

```bash
curl "http://localhost:8006/api/v1/chat/sessions/{session_id}/export/json" \
  -o session.json
```

### Limpiar Cache

```bash
curl -X POST "http://localhost:8006/api/v1/chat/cache/clear"
```

## 🎯 Atajos

### Alias Útiles (Linux/Mac)

Agregar a `~/.bashrc` o `~/.zshrc`:

```bash
alias bulk-chat-start='cd /path/to/bulk_chat && python -m bulk_chat.main'
alias bulk-chat-verify='cd /path/to/bulk_chat && python verify_setup.py'
alias bulk-chat-status='curl http://localhost:8006/health'
```

### PowerShell (Windows)

Agregar a perfil de PowerShell:

```powershell
function Start-BulkChat {
    cd C:\path\to\bulk_chat
    python -m bulk_chat.main
}

function Test-BulkChat {
    curl http://localhost:8006/health
}
```

---

**Nota**: Reemplaza `{session_id}` con un ID de sesión real en los ejemplos.
















