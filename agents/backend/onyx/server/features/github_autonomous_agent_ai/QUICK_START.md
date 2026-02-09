# Quick Start Guide

## Instalación Rápida

### 1. Instalar dependencias

```bash
cd github_autonomous_agent_ai
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y agregar tu GITHUB_TOKEN si es necesario
```

### 3. Iniciar el servidor API

```bash
python main.py --mode api
```

O usar el script:

```bash
python scripts/start_api.py
```

### 4. Iniciar el servicio (daemon)

```bash
python main.py --mode service
```

O usar el script:

```bash
python scripts/start_service.py
```

## Uso con Docker

```bash
docker-compose up -d
```

## Acceder al Frontend

1. Instalar dependencias del frontend:

```bash
cd frontend
npm install
```

2. Iniciar el servidor de desarrollo:

```bash
npm run dev
```

3. Abrir en el navegador: http://localhost:3000

## Ejemplo de Uso de la API

### Crear una tarea

```bash
curl -X POST "http://localhost:8025/api/tasks/" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "instruction": "Crear un archivo README.md",
    "metadata": {
      "file_path": "README.md",
      "file_content": "# Mi Proyecto"
    }
  }'
```

### Iniciar el agente

```bash
curl -X POST "http://localhost:8025/api/agent/start"
```

### Ver estado

```bash
curl "http://localhost:8025/api/agent/status"
```




