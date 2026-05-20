# Setup - Bulk Chat
# ==================

## ✅ Verificación Rápida

Antes de usar el sistema, verifica que todo esté listo:

### 1. Verificar Python

```bash
python --version
# Debe ser Python 3.8 o superior
```

### 2. Instalar Dependencias

```bash
cd C:\blatam-academy\agents\backend\onyx\server\features\bulk_chat
pip install -r requirements.txt
```

### 3. Configurar Variables de Entorno (Opcional)

Crea un archivo `.env` en el directorio raíz de `bulk_chat` con:

```env
# Configuración mínima necesaria
OPENAI_API_KEY=tu-api-key-aqui
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# Configuración opcional
CHAT_API_PORT=8006
AUTO_CONTINUE=true
RESPONSE_INTERVAL=2.0
```

**Nota**: Si no tienes una API key, puedes usar el modo `mock`:

```env
LLM_PROVIDER=mock
```

### 4. Crear Directorios Necesarios

```bash
mkdir sessions backups
```

### 5. Iniciar el Servidor

```bash
# Opción 1: Usando el módulo
python -m bulk_chat.main

# Opción 2: Usando el script
python start.py

# Opción 3: Con configuración personalizada
python -m bulk_chat.main --llm-provider mock --port 8006
```

### 6. Verificar que Funciona

En otra terminal:

```bash
# Health check
curl http://localhost:8006/health

# Debería responder:
# {"status":"healthy","service":"bulk_chat",...}
```

## 🚀 Inicio Rápido sin Configuración

Si quieres probarlo rápidamente sin configuración:

```bash
python -m bulk_chat.main --llm-provider mock
```

Esto iniciará el servidor en modo mock (sin necesidad de API keys).

## 📝 Próximos Pasos

1. Lee el [README.md](README.md) para entender todas las características
2. Revisa [QUICK_START.md](QUICK_START.md) para ejemplos de uso
3. Configura tu `.env` con tus API keys reales cuando estés listo

## ⚠️ Solución de Problemas

### Error: "No module named 'bulk_chat'"

Asegúrate de estar en el directorio correcto o ejecuta:

```bash
export PYTHONPATH="${PYTHONPATH}:C:\blatam-academy\agents\backend\onyx\server\features"
```

### Error: "Module not found"

Instala las dependencias:

```bash
pip install -r requirements.txt
```

### Puerto ya en uso

Cambia el puerto:

```bash
python -m bulk_chat.main --port 8007
```
















