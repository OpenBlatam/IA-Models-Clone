# ⚡ Comando Rápido - Cursor Agent 24/7

## 🚀 Iniciar con un Solo Comando

```bash
python run.py
```

¡Eso es todo! El agente iniciará en `http://localhost:8024`

## 📋 Variaciones del Comando

```bash
# Puerto personalizado
python run.py --port 8080

# Habilitar AWS
python run.py --aws

# Modo servicio
python run.py --mode service

# Todo junto
python run.py --port 8080 --aws --aws-region eu-west-1
```

## 🌐 URLs Después de Iniciar

- **API**: http://localhost:8024
- **Health**: http://localhost:8024/api/health
- **Docs**: http://localhost:8024/docs
- **Status**: http://localhost:8024/api/status

## 📚 Más Información

- [COMMANDS.md](COMMANDS.md) - Todos los comandos disponibles
- [README.md](README.md) - Documentación completa
- [INSTALL.md](INSTALL.md) - Instalación como comando global




