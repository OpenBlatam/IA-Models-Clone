# 📦 Instalación - Cursor Agent 24/7

Guía de instalación para usar el comando `cursor-agent` globalmente.

## 🚀 Instalación Rápida

### Opción 1: Instalación como Paquete (Recomendado)

```bash
# Instalar en modo desarrollo
pip install -e .

# O instalar globalmente
pip install .
```

Después de instalar, puedes usar:

```bash
# Comando global
cursor-agent start
cursor-agent start --port 8080
cursor-agent start --aws
cursor-agent health
cursor-agent version
```

### Opción 2: Uso Directo (Sin Instalación)

```bash
# Usar directamente
python run.py
python run.py --port 8080
python run.py --aws
```

### Opción 3: Instalación con Dependencias AWS

```bash
# Instalar con dependencias AWS
pip install -e . -r requirements-aws.txt
```

## 🔧 Verificación

```bash
# Verificar instalación
cursor-agent version

# Verificar salud (si está corriendo)
cursor-agent health
```

## 📝 Requisitos

- Python 3.10+
- pip
- (Opcional) AWS CLI para despliegue en AWS

## 🐳 Docker

Si prefieres usar Docker:

```bash
# Build
docker build -t cursor-agent-24-7 .

# Run
docker run -p 8024:8024 cursor-agent-24-7
```

## ☁️ AWS

Para usar con AWS, instala dependencias adicionales:

```bash
pip install -r requirements-aws.txt
```

Luego configura variables de entorno:

```bash
export AWS_REGION=us-east-1
export DYNAMODB_TABLE_NAME=cursor-agent-state
export REDIS_ENDPOINT=your-redis-endpoint
```

## 🆘 Troubleshooting

### Comando no encontrado

Si `cursor-agent` no se encuentra después de instalar:

```bash
# Verificar instalación
pip show cursor-agent-24-7

# Reinstalar
pip install --force-reinstall -e .

# Verificar PATH
echo $PATH
```

### Dependencias faltantes

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# O mínimas
pip install -r requirements-minimal.txt
```

## 📚 Siguiente Paso

Una vez instalado, ver [COMMANDS.md](COMMANDS.md) para ver cómo usar el agente.




