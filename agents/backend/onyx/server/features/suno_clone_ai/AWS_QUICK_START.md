# AWS Quick Start Guide

Guía rápida para desplegar Suno Clone AI en AWS.

## Instalación Rápida

```bash
# 1. Instalar Serverless Framework
npm install -g serverless
npm install serverless-python-requirements serverless-offline

# 2. Configurar AWS CLI
aws configure

# 3. Configurar variables de entorno
cp aws/.env.example .env
# Editar .env con tus credenciales

# 4. Desplegar
cd aws
serverless deploy --stage dev
```

## Estructura de Archivos AWS

```
aws/
├── lambda_handler.py      # Handler principal para Lambda
├── circuit_breaker.py     # Circuit breaker pattern
├── exceptions.py         # Excepciones AWS
├── serverless.yml        # Configuración Serverless Framework
├── AWS_DEPLOYMENT.md     # Guía completa de deployment
└── services/
    ├── dynamodb.py       # Servicio DynamoDB
    ├── s3_service.py      # Servicio S3
    ├── sqs_service.py     # Servicio SQS
    └── cloudwatch.py     # Servicio CloudWatch
```

## Características Implementadas

✅ **Lambda Handler** - FastAPI adaptado para Lambda con Mangum
✅ **Circuit Breakers** - Protección contra fallos en cascada
✅ **Retry Logic** - Reintentos automáticos con exponential backoff
✅ **OpenTelemetry** - Distributed tracing con AWS X-Ray
✅ **DynamoDB** - Almacenamiento serverless de metadatos
✅ **S3** - Almacenamiento de archivos de audio
✅ **SQS** - Procesamiento asíncrono de tareas
✅ **CloudWatch** - Logging y métricas
✅ **API Gateway** - Rate limiting y CORS configurados

## Uso Básico

### Desplegar

```bash
serverless deploy --stage dev
```

### Ver Logs

```bash
serverless logs -f api --tail
```

### Invocar Localmente

```bash
serverless offline start
```

### Eliminar Stack

```bash
serverless remove --stage dev
```

## Variables de Entorno Importantes

- `AWS_REGION`: Región de AWS (default: us-east-1)
- `DYNAMODB_TABLE_NAME`: Nombre de la tabla DynamoDB
- `S3_BUCKET_NAME`: Nombre del bucket S3
- `SQS_QUEUE_NAME`: Nombre de la cola SQS
- `ENABLE_TRACING`: Habilitar OpenTelemetry (default: true)
- `CIRCUIT_BREAKER_ENABLED`: Habilitar circuit breakers (default: true)

## Ejemplo de Uso en Código

### DynamoDB

```python
from aws.services.dynamodb import DynamoDBService
from config.settings import settings

db = DynamoDBService(
    table_name=settings.dynamodb_table_name,
    region_name=settings.aws_region
)

# Guardar
db.put_item({"id": "song-1", "title": "My Song"})

# Leer
song = db.get_item(key={"id": "song-1"})
```

### S3

```python
from aws.services.s3_service import S3Service

s3 = S3Service(bucket_name=settings.s3_bucket_name)

# Subir
with open("audio.mp3", "rb") as f:
    s3.upload_file(f, "audio/song.mp3")

# Descargar
audio = s3.download_file("audio/song.mp3")
```

### Circuit Breaker

```python
from aws.circuit_breaker import circuit_breaker

@circuit_breaker("external_api")
async def call_api():
    # Tu código aquí
    pass
```

### Retry

```python
from middleware.retry_middleware import retry_with_backoff

@retry_with_backoff(max_attempts=3)
async def process_data():
    # Tu código aquí
    pass
```

## Troubleshooting

### Cold Starts Largos

- Usar Lambda Layers para dependencias pesadas
- Habilitar Provisioned Concurrency
- Optimizar imports (lazy loading)

### Timeout Errors

- Aumentar `timeout` en `serverless.yml` (máx 900s)
- Mover tareas largas a SQS

### Memory Errors

- Aumentar `memorySize` en `serverless.yml`
- Optimizar modelos ML

## Próximos Pasos

1. Leer `AWS_DEPLOYMENT.md` para guía completa
2. Configurar CI/CD
3. Configurar monitoring y alertas
4. Optimizar costos según uso















