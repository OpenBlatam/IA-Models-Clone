# AWS Serverless Integration

Este módulo proporciona integración completa con servicios AWS para el despliegue serverless de Addiction Recovery AI.

## 📁 Estructura

```
aws/
├── __init__.py              # Exports principales
├── lambda_handler.py        # Handler para AWS Lambda
├── aws_services.py          # Wrappers para servicios AWS
├── circuit_breaker.py       # Implementación de Circuit Breaker
├── retry_handler.py         # Retry con exponential backoff
├── sam_template.yaml        # SAM template para deployment
├── deploy.sh                # Script de deployment
├── Dockerfile.lambda        # Dockerfile para Lambda
├── example_usage.py         # Ejemplos de uso
└── README.md                # Esta documentación
```

## 🚀 Inicio Rápido

### 1. Configurar Variables de Entorno

```bash
export AWS_REGION=us-east-1
export ENVIRONMENT=production
export DYNAMODB_TABLE_NAME=addiction-recovery-users
export S3_BUCKET_NAME=addiction-recovery-data
```

### 2. Deployment

```bash
# Opción 1: Script automático
chmod +x aws/deploy.sh
./aws/deploy.sh

# Opción 2: SAM CLI manual
sam build
sam deploy --guided
```

### 3. Usar Servicios AWS

```python
from aws.aws_services import DynamoDBService, S3Service

# DynamoDB
dynamodb = DynamoDBService()
user = dynamodb.get_item({"user_id": {"S": "user123"}})

# S3
s3 = S3Service()
s3.upload_file(file_content, "reports/report.pdf")
```

## 🔧 Componentes

### Lambda Handler

`lambda_handler.py` proporciona el punto de entrada para AWS Lambda:

- **Singleton pattern** para reutilizar app entre invocaciones
- **Cold start optimization** con preloading
- **Error handling** robusto
- **Logging** estructurado para CloudWatch

### AWS Services

`aws_services.py` incluye wrappers para:

- **DynamoDBService**: Operaciones CRUD con retry y circuit breaker
- **S3Service**: Upload/download de archivos
- **CloudWatchService**: Métricas y logging
- **SNSService**: Notificaciones
- **SQSService**: Colas para tareas asíncronas
- **SecretsManagerService**: Gestión de secretos
- **ParameterStoreService**: Configuración

### Circuit Breaker

`circuit_breaker.py` implementa el patrón Circuit Breaker:

- **Estados**: CLOSED, OPEN, HALF_OPEN
- **Failure threshold**: Configurable
- **Timeout**: Auto-recovery después de timeout
- **Thread-safe**: Para uso concurrente

### Retry Handler

`retry_handler.py` proporciona retry con exponential backoff:

- **Exponential backoff**: Delay aumenta exponencialmente
- **Jitter**: Evita thundering herd
- **Configurable**: Max retries, delays, etc.
- **Decorator pattern**: Fácil de usar

## 📊 Observabilidad

### CloudWatch Metrics

Métricas automáticas:
- `RequestCount`: Número de requests
- `ResponseTime`: Tiempo de respuesta (ms)
- `ErrorCount`: Número de errores
- `StatusCode`: Distribución de códigos HTTP

### X-Ray Tracing

Tracing distribuido habilitado:
- Service map
- Performance insights
- Error tracking

### Structured Logging

Logs estructurados en JSON:
```json
{
  "request_id": "abc123",
  "method": "POST",
  "path": "/recovery/assess",
  "status_code": 200,
  "duration_ms": 45.2
}
```

## 🔒 Seguridad

### IAM Roles

Roles con least privilege:
- DynamoDB: Read/Write a tabla específica
- S3: Read/Write a bucket específico
- SNS: Publish a topic específico
- SQS: Send/Receive de queue específica

### Secrets Management

Usar AWS Secrets Manager:
```python
secrets = SecretsManagerService()
config = secrets.get_secret()
api_key = config["OPENAI_API_KEY"]
```

### Parameter Store

Configuración en SSM Parameter Store:
```python
params = ParameterStoreService()
value = params.get_parameter("openai/api_key")
```

## 🧪 Testing

### Local Testing

```bash
# SAM Local
sam local start-api

# Docker
docker build -f aws/Dockerfile.lambda -t test .
docker run -p 9000:8080 test
```

### Unit Tests

```python
from aws.aws_services import DynamoDBService
from unittest.mock import Mock, patch

@patch('boto3.client')
def test_dynamodb(mock_client):
    service = DynamoDBService()
    # Test implementation
```

## 📈 Performance

### Cold Start Optimization

1. **Preload models**: Cargar modelos AI en cold start
2. **Lambda Layers**: Dependencias en layers
3. **Provisioned Concurrency**: Para funciones críticas
4. **Connection pooling**: Reutilizar conexiones

### Best Practices

- ✅ Usar singleton pattern para servicios
- ✅ Cachear conexiones entre invocaciones
- ✅ Minimizar imports pesados
- ✅ Usar async/await para I/O
- ✅ Implementar circuit breakers
- ✅ Retry con exponential backoff

## 🐛 Troubleshooting

### Errores Comunes

1. **ImportError**: Verificar que dependencias estén en Lambda Layer
2. **Timeout**: Aumentar timeout de Lambda
3. **Memory**: Aumentar memoria si necesario
4. **Permissions**: Verificar IAM roles

### Debugging

```python
# Habilitar logging detallado
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar configuración
from config.aws_settings import get_aws_settings
settings = get_aws_settings()
print(f"Region: {settings.aws_region}")
print(f"Is Lambda: {settings.is_lambda}")
```

## 📚 Recursos

- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [SAM Documentation](https://docs.aws.amazon.com/serverless-application-model/)
- [Mangum Documentation](https://mangum.io/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

## 🤝 Contribuir

Para agregar nuevos servicios AWS:

1. Crear clase en `aws_services.py`
2. Implementar circuit breaker y retry
3. Agregar tests
4. Documentar en este README















