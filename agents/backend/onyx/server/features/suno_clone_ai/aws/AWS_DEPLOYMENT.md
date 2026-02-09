# AWS Serverless Deployment Guide

Esta guía explica cómo desplegar Suno Clone AI en AWS usando arquitectura serverless.

## Arquitectura

- **API Gateway** → **Lambda** (FastAPI con Mangum)
- **DynamoDB** para almacenamiento de metadatos
- **S3** para almacenamiento de archivos de audio
- **SQS** para procesamiento asíncrono
- **CloudWatch** para logging y métricas
- **X-Ray** para distributed tracing

## Prerrequisitos

1. **AWS CLI** instalado y configurado
2. **Serverless Framework** instalado:
   ```bash
   npm install -g serverless
   npm install serverless-python-requirements
   npm install serverless-offline  # Para desarrollo local
   ```
3. **Docker** (para empaquetar dependencias Python)
4. **Python 3.11**

## Configuración

### 1. Variables de Entorno

Crea un archivo `.env` o configura variables en AWS Systems Manager Parameter Store:

```bash
# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# DynamoDB
DYNAMODB_TABLE_NAME=suno-clone-ai-dev

# S3
S3_BUCKET_NAME=suno-clone-ai-dev-audio

# SQS
SQS_QUEUE_NAME=suno-clone-ai-dev-queue

# OpenTelemetry
OTLP_ENDPOINT=http://localhost:4317  # Para desarrollo local
ENABLE_TRACING=true

# Circuit Breaker
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60.0

# Retry
RETRY_MAX_ATTEMPTS=3
RETRY_INITIAL_WAIT=1.0
RETRY_MAX_WAIT=60.0
```

### 2. Configurar Serverless

El archivo `serverless.yml` ya está configurado. Puedes ajustar:
- `region`: Región de AWS
- `stage`: Ambiente (dev, staging, prod)
- `memorySize`: Memoria para Lambda (ajustar según modelos ML)
- `timeout`: Timeout de Lambda (máximo 15 minutos)

## Despliegue

### Desarrollo Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar localmente con serverless-offline
serverless offline start
```

### Desplegar a AWS

```bash
# Desplegar a dev
serverless deploy --stage dev

# Desplegar a producción
serverless deploy --stage prod

# Desplegar solo una función
serverless deploy function -f api

# Ver logs
serverless logs -f api --tail

# Eliminar stack
serverless remove --stage dev
```

## Optimizaciones para Lambda

### 1. Cold Start Minimization

- **Pre-inicialización**: El handler Lambda usa singleton pattern para reutilizar la app FastAPI
- **Layers**: Dependencias pesadas (torch, transformers) se empaquetan en Lambda Layers
- **Provisioned Concurrency**: Para funciones críticas, usar provisioned concurrency

```yaml
# En serverless.yml
functions:
  api:
    provisionedConcurrency: 2  # Mantener 2 instancias calientes
```

### 2. Memory Optimization

- Ajustar `memorySize` según el modelo ML usado
- Usar `slim: true` en pythonRequirements para reducir tamaño del paquete

### 3. Timeout Configuration

- Generación de música puede tomar tiempo → timeout de 15 minutos
- Para tareas largas, usar SQS + Lambda separado

## Integración con AWS Services

### DynamoDB

```python
from aws.services.dynamodb import DynamoDBService

# Inicializar
db = DynamoDBService(
    table_name=settings.dynamodb_table_name,
    region_name=settings.aws_region
)

# Usar
song = db.get_item(key={"id": "song-123"})
db.put_item(item={"id": "song-123", "title": "My Song"})
```

### S3

```python
from aws.services.s3_service import S3Service

# Inicializar
s3 = S3Service(
    bucket_name=settings.s3_bucket_name,
    region_name=settings.aws_region
)

# Subir archivo
with open("audio.mp3", "rb") as f:
    s3.upload_file(f, "audio/song-123.mp3", content_type="audio/mpeg")

# Obtener URL pre-firmada
url = s3.get_presigned_url("audio/song-123.mp3", expiration=3600)
```

### SQS

```python
from aws.services.sqs_service import SQSService

# Inicializar
sqs = SQSService(
    queue_name=settings.sqs_queue_name,
    region_name=settings.aws_region
)

# Enviar mensaje
sqs.send_message({
    "task": "generate_music",
    "prompt": "A happy song",
    "user_id": "user-123"
})

# Procesar mensajes (en Lambda handler)
messages = sqs.receive_messages(max_number=10)
for msg in messages:
    process_message(msg['Body'])
    sqs.delete_message(msg['ReceiptHandle'])
```

## Circuit Breaker

Protege servicios externos con circuit breakers:

```python
from aws.circuit_breaker import circuit_breaker, CircuitBreakerConfig

@circuit_breaker(
    "external_api",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        timeout=60.0
    )
)
async def call_external_api():
    # Tu código aquí
    pass
```

## Monitoring y Logging

### CloudWatch Logs

Los logs se envían automáticamente a CloudWatch. Ver logs:

```bash
# CLI
aws logs tail /aws/lambda/suno-clone-ai-dev-api --follow

# O con serverless
serverless logs -f api --tail
```

### CloudWatch Metrics

```python
from aws.services.cloudwatch import CloudWatchService

cloudwatch = CloudWatchService(region_name=settings.aws_region)

# Enviar métrica
cloudwatch.put_metric(
    namespace="SunoCloneAI",
    metric_name="MusicGenerationDuration",
    value=45.5,
    unit="Seconds",
    dimensions=[{"Name": "Model", "Value": "musicgen-medium"}]
)
```

### X-Ray Tracing

X-Ray está habilitado automáticamente. Ver traces en AWS Console → X-Ray.

## API Gateway Integration

### Rate Limiting

Configurar rate limiting en API Gateway:

```yaml
# En serverless.yml
functions:
  api:
    events:
      - httpApi:
          path: /{proxy+}
          method: ANY
          throttling:
            burstLimit: 100
            rateLimit: 50
```

### CORS

CORS está configurado en `serverless.yml`. Ajustar según necesidades:

```yaml
events:
  - httpApi:
      path: /{proxy+}
      method: ANY
      cors:
        allowedOrigins:
          - https://yourdomain.com
        allowedHeaders:
          - Content-Type
          - Authorization
```

## Costos

### Optimización de Costos

1. **DynamoDB**: Usar `PAY_PER_REQUEST` para desarrollo, `PROVISIONED` para producción con auto-scaling
2. **Lambda**: Ajustar `memorySize` y `reservedConcurrentExecutions`
3. **S3**: Configurar lifecycle policies para eliminar archivos antiguos
4. **CloudWatch**: Configurar retention period (14 días por defecto)

### Estimación Mensual (Dev)

- Lambda: ~$5-20 (dependiendo de uso)
- DynamoDB: ~$1-5
- S3: ~$1-10 (dependiendo de almacenamiento)
- API Gateway: ~$1-5
- CloudWatch: ~$1-3

**Total estimado: ~$10-45/mes para desarrollo**

## Troubleshooting

### Cold Starts Largos

- Usar Lambda Layers para dependencias pesadas
- Habilitar Provisioned Concurrency
- Optimizar imports (lazy loading)

### Timeout Errors

- Aumentar `timeout` en `serverless.yml`
- Mover tareas largas a SQS + Lambda separado
- Usar Step Functions para workflows complejos

### Memory Errors

- Aumentar `memorySize`
- Optimizar modelos ML (quantization, pruning)
- Usar modelos más pequeños

## Seguridad

### IAM Roles

Los roles IAM están configurados en `serverless.yml`. Ajustar según principio de menor privilegio.

### Secrets Management

Usar AWS Secrets Manager o Systems Manager Parameter Store para secrets:

```python
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

### VPC (Opcional)

Para acceso a recursos en VPC:

```yaml
functions:
  api:
    vpc:
      securityGroupIds:
        - sg-xxxxx
      subnetIds:
        - subnet-xxxxx
        - subnet-yyyyy
```

## CI/CD

### GitHub Actions Example

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - uses: actions/setup-node@v2
      
      - name: Install Serverless
        run: npm install -g serverless
        
      - name: Deploy
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: serverless deploy --stage prod
```

## Referencias

- [Serverless Framework Docs](https://www.serverless.com/framework/docs)
- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [FastAPI on Lambda](https://mangum.io/)
- [OpenTelemetry AWS](https://opentelemetry.io/docs/instrumentation/python/aws/)















