# AWS Serverless Deployment Guide

Guía completa para desplegar Addiction Recovery AI en AWS usando arquitectura serverless.

## 🏗️ Arquitectura

```
┌─────────────┐
│ API Gateway │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Lambda    │ ◄─── FastAPI Application
└──────┬──────┘
       │
       ├──► DynamoDB (Users, Data)
       ├──► S3 (Files, Reports)
       ├──► ElastiCache Redis (Cache)
       ├──► SNS (Notifications)
       ├──► SQS (Background Tasks)
       └──► CloudWatch (Logs, Metrics)
```

## 📋 Prerequisitos

1. **AWS CLI** instalado y configurado
2. **SAM CLI** instalado
3. **Docker** (para builds locales)
4. **Python 3.11+**
5. **Credenciales AWS** configuradas

## 🚀 Deployment Rápido

### Opción 1: Usando SAM CLI (Recomendado)

```bash
# 1. Configurar variables de entorno
export STACK_NAME=addiction-recovery-ai
export ENVIRONMENT=production
export AWS_REGION=us-east-1

# 2. Ejecutar script de deployment
chmod +x aws/deploy.sh
./aws/deploy.sh
```

### Opción 2: Deployment Manual

```bash
# 1. Build Lambda layer
mkdir -p layer/python
pip install -r requirements.txt -t layer/python/

# 2. Build SAM application
sam build --template-file aws/sam_template.yaml

# 3. Deploy
sam deploy \
    --stack-name addiction-recovery-ai \
    --region us-east-1 \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides Environment=production
```

### Opción 3: Usando Docker

```bash
# Build Docker image
docker build -f aws/Dockerfile.lambda -t addiction-recovery-ai:latest .

# Tag and push to ECR
aws ecr create-repository --repository-name addiction-recovery-ai || true
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag addiction-recovery-ai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/addiction-recovery-ai:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/addiction-recovery-ai:latest
```

## ⚙️ Configuración

### Variables de Entorno

Crear archivo `.env` o configurar en AWS Systems Manager Parameter Store:

```env
# AWS Configuration
AWS_REGION=us-east-1
ENVIRONMENT=production

# DynamoDB
DYNAMODB_TABLE_NAME=addiction-recovery-users

# S3
S3_BUCKET_NAME=addiction-recovery-data

# Redis/ElastiCache
REDIS_ENDPOINT=your-redis-endpoint.cache.amazonaws.com
REDIS_PORT=6379

# SNS
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:notifications

# SQS
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/background-tasks

# Secrets Manager
SECRETS_MANAGER_SECRET_NAME=addiction-recovery-secrets

# OpenAI
OPENAI_API_KEY=your-openai-key

# Feature Flags
PRELOAD_MODELS=true
ENABLE_XRAY=true
```

### Secrets Manager

Almacenar secretos sensibles en AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
    --name addiction-recovery-secrets \
    --secret-string '{
        "OPENAI_API_KEY": "your-key",
        "SECRET_KEY": "your-secret",
        "DATABASE_URL": "your-db-url"
    }'
```

## 📊 Servicios AWS Utilizados

### 1. AWS Lambda
- **Runtime**: Python 3.11
- **Memory**: 512 MB (configurable)
- **Timeout**: 300 segundos
- **Concurrency**: 100 (configurable)

### 2. API Gateway
- **Type**: REST API
- **Stage**: production/staging/development
- **CORS**: Habilitado
- **Tracing**: X-Ray habilitado

### 3. DynamoDB
- **Table**: Users, Progress, Analytics
- **Billing**: Pay-per-request
- **Backup**: Point-in-time recovery habilitado

### 4. S3
- **Bucket**: Data storage, reports, backups
- **Versioning**: Habilitado
- **Lifecycle**: Transición a Glacier después de 30 días

### 5. ElastiCache Redis
- **Type**: Replication Group
- **Nodes**: 2 (Multi-AZ)
- **Failover**: Automático

### 6. CloudWatch
- **Logs**: Automáticos desde Lambda
- **Metrics**: Custom metrics
- **Alarms**: Error rate, latency

### 7. X-Ray
- **Tracing**: Habilitado
- **Sampling**: 100% (Lambda)

## 🔧 Optimizaciones para Cold Starts

### 1. Preload Models
```python
# En lambda_handler.py
if aws_settings.preload_models:
    # Preload AI models on cold start
    from core.ultra_fast_engine import create_ultra_fast_engine
    engine = create_ultra_fast_engine()
```

### 2. Lambda Layers
- Dependencias pesadas en Lambda Layer
- Reutilización entre invocaciones

### 3. Provisioned Concurrency
```yaml
# En sam_template.yaml
ProvisionedConcurrencyConfig:
  ProvisionedConcurrentExecutions: 10
```

### 4. Connection Pooling
- Reutilizar conexiones entre invocaciones
- Singleton pattern para servicios

## 🔒 Seguridad

### IAM Roles
- Least privilege principle
- Roles específicos por servicio

### API Gateway
- API Keys (opcional)
- AWS_IAM authentication
- Rate limiting

### Secrets
- AWS Secrets Manager
- Parameter Store (SSM)
- No hardcoded secrets

### VPC (Opcional)
```yaml
VpcConfig:
  SecurityGroupIds:
    - sg-xxxxx
  SubnetIds:
    - subnet-xxxxx
    - subnet-yyyyy
```

## 📈 Monitoreo

### CloudWatch Metrics
- RequestCount
- ResponseTime
- ErrorCount
- StatusCode distribution

### CloudWatch Alarms
- Error rate > 10 en 5 minutos
- Latency > 5 segundos

### X-Ray Tracing
- Distributed tracing
- Service map
- Performance insights

### Logs Estructurados
```json
{
  "request_id": "abc123",
  "method": "POST",
  "path": "/recovery/assess",
  "status_code": 200,
  "duration_ms": 45.2
}
```

## 🔄 CI/CD

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
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Deploy SAM application
        run: |
          pip install aws-sam-cli
          sam build
          sam deploy --no-confirm-changeset
```

## 🧪 Testing Local

### SAM Local

```bash
# Start local API
sam local start-api

# Invoke function locally
sam local invoke ServerlessFunction --event events/test-event.json
```

### Docker Local

```bash
# Run Lambda runtime locally
docker run -p 9000:8080 \
    -e AWS_LAMBDA_FUNCTION_NAME=test \
    -v $(pwd):/var/task \
    addiction-recovery-ai:latest
```

## 📝 Endpoints

Después del deployment, los endpoints estarán disponibles en:

```
https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/
```

Ejemplo:
```
https://abc123.execute-api.us-east-1.amazonaws.com/production/recovery/health
```

## 🐛 Troubleshooting

### Cold Start Lento
- Aumentar memoria de Lambda
- Usar Provisioned Concurrency
- Optimizar imports

### Errores de Conexión
- Verificar VPC configuration
- Verificar security groups
- Verificar IAM permissions

### Timeouts
- Aumentar timeout de Lambda
- Optimizar queries a DynamoDB
- Usar SQS para tareas largas

## 💰 Costos Estimados

### Free Tier (Primeros 12 meses)
- Lambda: 1M requests gratis
- API Gateway: 1M requests gratis
- DynamoDB: 25 GB storage gratis
- S3: 5 GB storage gratis

### Post Free Tier (Estimado)
- Lambda: $0.20 por 1M requests
- API Gateway: $3.50 por 1M requests
- DynamoDB: $0.25 por GB storage
- S3: $0.023 por GB storage

## 📚 Recursos Adicionales

- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [SAM Documentation](https://docs.aws.amazon.com/serverless-application-model/)
- [FastAPI on Lambda](https://fastapi.tiangolo.com/deployment/serverless/)
- [Mangum Documentation](https://mangum.io/)

## 🆘 Soporte

Para problemas o preguntas:
1. Revisar logs en CloudWatch
2. Verificar X-Ray traces
3. Consultar documentación AWS
4. Abrir issue en el repositorio















