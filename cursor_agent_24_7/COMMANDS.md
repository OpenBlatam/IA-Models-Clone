# 🚀 Comandos - Cursor Agent 24/7

Guía rápida de comandos para iniciar y usar el Cursor Agent 24/7.

## ⚡ Inicio Rápido

### Opción 1: Comando Simple (Recomendado)

```bash
# Iniciar API en puerto 8024 (default)
python run.py

# O usando el CLI directamente
python cli.py start
```

### Opción 2: Comando con Opciones

```bash
# Puerto personalizado
python run.py --port 8080

# Modo servicio persistente
python run.py --mode service

# Habilitar AWS
python run.py --aws

# AWS en región específica
python run.py --aws --aws-region eu-west-1
```

### Opción 3: Usando main.py (Compatibilidad)

```bash
python main.py
python main.py --port 8080
python main.py --mode service
python main.py --aws
```

## 📋 Comandos Disponibles

### `start` - Iniciar el agente

```bash
# Básico
python run.py start

# Con opciones
python run.py start --host 0.0.0.0 --port 8024 --mode api

# AWS
python run.py start --aws --aws-region us-east-1
```

**Opciones:**
- `--host` / `-h`: Host (default: 0.0.0.0)
- `--port` / `-p`: Puerto (default: 8024)
- `--mode` / `-m`: Modo: `api` o `service` (default: api)
- `--aws`: Habilitar servicios AWS
- `--aws-region`: Región AWS (default: us-east-1)

### `health` - Verificar salud

```bash
python run.py health
python run.py health --url http://localhost:8024
```

### `version` - Ver versión

```bash
python run.py version
```

## 🌐 URLs Importantes

Una vez iniciado, el agente expone:

- **API**: `http://localhost:8024`
- **Health Check**: `http://localhost:8024/api/health`
- **API Docs**: `http://localhost:8024/docs`
- **Status**: `http://localhost:8024/api/status`

## 🔧 Variables de Entorno

Puedes configurar el agente con variables de entorno:

```bash
# AWS
export AWS_REGION=us-east-1
export DYNAMODB_TABLE_NAME=cursor-agent-state
export CACHE_TYPE=elasticache
export REDIS_ENDPOINT=your-redis-endpoint

# API
export API_HOST=0.0.0.0
export API_PORT=8024

# Agente
export AGENT_PERSISTENT_STORAGE=true
export AGENT_AUTO_RESTART=true
```

## 🐳 Docker

```bash
# Build
docker build -t cursor-agent-24-7 .

# Run
docker run -p 8024:8024 cursor-agent-24-7

# Con variables de entorno
docker run -p 8024:8024 \
  -e AWS_REGION=us-east-1 \
  -e DYNAMODB_TABLE_NAME=cursor-agent-state \
  cursor-agent-24-7
```

## ☁️ AWS

### Local con LocalStack

```bash
cd aws
docker-compose -f docker-compose.aws.yml up
```

### Despliegue en AWS

```bash
# Con Terraform
cd aws/terraform
terraform apply

# Con script
./aws/deploy.sh ecs
```

## 📝 Ejemplos Completos

### Desarrollo Local

```bash
# Iniciar API
python run.py

# En otro terminal, verificar
curl http://localhost:8024/api/health
```

### Producción Local

```bash
# Modo servicio
python run.py --mode service
```

### AWS Local (con LocalStack)

```bash
# Iniciar LocalStack
docker-compose -f aws/docker-compose.aws.yml up -d

# Iniciar agente con AWS
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_REGION=us-east-1
python run.py --aws
```

### AWS Producción

```bash
# Configurar variables
export AWS_REGION=us-east-1
export DYNAMODB_TABLE_NAME=cursor-agent-state
export REDIS_ENDPOINT=your-redis-endpoint

# Iniciar
python run.py --aws
```

## 🆘 Troubleshooting

### Puerto en uso

```bash
# Cambiar puerto
python run.py --port 8080
```

### Error de AWS

```bash
# Verificar credenciales
aws sts get-caller-identity

# Usar modo local sin AWS
python run.py  # Sin --aws
```

### Ver logs

```bash
# Logs en consola (default)
python run.py

# Ver logs de archivo
tail -f logs/agent.log
```

## 📚 Más Información

- [README.md](README.md) - Documentación general
- [AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md) - Despliegue en AWS
- [QUICK_START.md](QUICK_START.md) - Inicio rápido
- [API_REFERENCE.md](API_REFERENCE.md) - Referencia de API




