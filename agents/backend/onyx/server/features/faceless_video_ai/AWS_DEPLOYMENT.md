# AWS Deployment Guide - Faceless Video AI

Esta guía describe cómo desplegar Faceless Video AI en AWS usando Docker, ECS, Lambda y otros servicios de AWS.

## 📋 Tabla de Contenidos

1. [Arquitectura AWS](#arquitectura-aws)
2. [Prerrequisitos](#prerrequisitos)
3. [Configuración Inicial](#configuración-inicial)
4. [Despliegue en ECS (Fargate)](#despliegue-en-ecs-fargate)
5. [Despliegue Serverless (Lambda)](#despliegue-serverless-lambda)
6. [Configuración de Servicios AWS](#configuración-de-servicios-aws)
7. [Monitoreo y Logging](#monitoreo-y-logging)
8. [Escalado Automático](#escalado-automático)
9. [Seguridad](#seguridad)
10. [Troubleshooting](#troubleshooting)

## 🏗️ Arquitectura AWS

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS API Gateway                          │
│              (Rate Limiting, Authentication)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  ECS Fargate │ │  Lambda      │ │  ECS Fargate │
│  (API)       │ │  (Serverless)│ │  (Workers)   │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  ElastiCache │ │  RDS          │ │  S3           │
│  (Redis)     │ │  (PostgreSQL)│ │  (Storage)   │
└──────────────┘ └──────────────┘ └──────────────┘
```

## 📦 Prerrequisitos

1. **AWS CLI** instalado y configurado
2. **Docker** instalado
3. **Cuenta AWS** con permisos para:
   - ECS (Elastic Container Service)
   - ECR (Elastic Container Registry)
   - Lambda
   - S3
   - Secrets Manager
   - CloudWatch
   - IAM (para crear roles)

## ⚙️ Configuración Inicial

### 1. Configurar Variables de Entorno

Crea un archivo `.env` con las siguientes variables:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id
AWS_S3_BUCKET=faceless-video-ai-storage

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
POSTGRES_HOST=your-rds-endpoint.rds.amazonaws.com
POSTGRES_DB=faceless_video
POSTGRES_USER=faceless
POSTGRES_PASSWORD=your-secure-password

# Redis
REDIS_URL=redis://your-elasticache-endpoint:6379/0

# Celery
CELERY_BROKER_URL=redis://your-elasticache-endpoint:6379/1
CELERY_RESULT_BACKEND=redis://your-elasticache-endpoint:6379/2

# AI Services (o usar AWS Secrets Manager)
OPENAI_API_KEY=your-key
STABILITY_AI_API_KEY=your-key
ELEVENLABS_API_KEY=your-key

# Security
JWT_SECRET_KEY=your-secret-key
CORS_ORIGINS=https://yourdomain.com

# Observability
ENABLE_OPENTELEMETRY=true
OPENTELEMETRY_ENDPOINT=your-otlp-endpoint
CLOUDWATCH_LOG_GROUP=/ecs/faceless-video-ai
```

### 2. Crear Recursos AWS Base

#### Crear S3 Bucket

```bash
aws s3 mb s3://faceless-video-ai-storage --region us-east-1
aws s3api put-bucket-versioning \
    --bucket faceless-video-ai-storage \
    --versioning-configuration Status=Enabled
```

#### Crear Secrets en AWS Secrets Manager

```bash
# Crear secret para API keys
aws secretsmanager create-secret \
    --name faceless-video-ai/secrets \
    --secret-string '{
        "OPENAI_API_KEY": "your-key",
        "STABILITY_AI_API_KEY": "your-key",
        "ELEVENLABS_API_KEY": "your-key",
        "JWT_SECRET_KEY": "your-secret-key"
    }' \
    --region us-east-1
```

#### Crear RDS PostgreSQL

```bash
aws rds create-db-instance \
    --db-instance-identifier faceless-video-ai-db \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --master-username faceless \
    --master-user-password your-secure-password \
    --allocated-storage 100 \
    --vpc-security-group-ids sg-xxxxx \
    --db-subnet-group-name your-subnet-group \
    --backup-retention-period 7 \
    --region us-east-1
```

#### Crear ElastiCache Redis

```bash
aws elasticache create-cache-cluster \
    --cache-cluster-id faceless-video-ai-redis \
    --cache-node-type cache.t3.micro \
    --engine redis \
    --num-cache-nodes 1 \
    --vpc-security-group-ids sg-xxxxx \
    --subnet-group-name your-subnet-group \
    --region us-east-1
```

## 🚀 Despliegue en ECS (Fargate)

### 1. Crear ECR Repository

```bash
aws ecr create-repository \
    --repository-name faceless-video-ai \
    --region us-east-1
```

### 2. Construir y Subir Imagen Docker

```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Construir imagen
docker build -t faceless-video-ai:latest .

# Tag para ECR
docker tag faceless-video-ai:latest \
    YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/faceless-video-ai:latest

# Push a ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/faceless-video-ai:latest
```

O usar el script de deployment:

```bash
chmod +x aws/deploy.sh
./aws/deploy.sh
```

### 3. Crear IAM Roles

#### Task Execution Role

```bash
aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach policies
aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

#### Task Role (para acceso a S3, etc.)

```bash
aws iam create-role \
    --role-name ecsTaskRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Crear policy para S3
aws iam put-role-policy \
    --role-name ecsTaskRole \
    --policy-name S3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::faceless-video-ai-storage/*"
        }]
    }'
```

### 4. Crear ECS Cluster

```bash
aws ecs create-cluster \
    --cluster-name faceless-video-ai-cluster \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy \
        capacityProvider=FARGATE,weight=1 \
        capacityProvider=FARGATE_SPOT,weight=1 \
    --region us-east-1
```

### 5. Registrar Task Definition

Edita `aws/ecs-task-definition.json` con tus valores y luego:

```bash
aws ecs register-task-definition \
    --cli-input-json file://aws/ecs-task-definition.json \
    --region us-east-1
```

### 6. Crear ECS Service

```bash
aws ecs create-service \
    --cluster faceless-video-ai-cluster \
    --service-name faceless-video-ai-service \
    --task-definition faceless-video-ai \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={
        subnets=[subnet-xxx,subnet-yyy],
        securityGroups=[sg-xxx],
        assignPublicIp=ENABLED
    }" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,
        containerName=api,containerPort=8000" \
    --region us-east-1
```

## ⚡ Despliegue Serverless (Lambda)

### 1. Crear Lambda Function

```bash
# Crear deployment package
zip -r lambda-deployment.zip . \
    -x "*.git*" \
    -x "*tests*" \
    -x "*.env*" \
    -x "*__pycache__*"

# Crear Lambda function
aws lambda create-function \
    --function-name faceless-video-ai-api \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://lambda-deployment.zip \
    --timeout 900 \
    --memory-size 3008 \
    --environment Variables={
        ENVIRONMENT=lambda,
        AWS_REGION=us-east-1
    } \
    --region us-east-1
```

### 2. Configurar API Gateway

```bash
# Crear API Gateway REST API
aws apigateway create-rest-api \
    --name faceless-video-ai-api \
    --region us-east-1

# Configurar integración con Lambda
# (Usar AWS Console o Terraform/CloudFormation para configuración completa)
```

## 🔧 Configuración de Servicios AWS

### CloudWatch Logs

Los logs se configuran automáticamente en la task definition. Para ver logs:

```bash
aws logs tail /ecs/faceless-video-ai --follow --region us-east-1
```

### Application Load Balancer

```bash
# Crear ALB
aws elbv2 create-load-balancer \
    --name faceless-video-ai-alb \
    --subnets subnet-xxx subnet-yyy \
    --security-groups sg-xxx \
    --region us-east-1

# Crear target group
aws elbv2 create-target-group \
    --name faceless-video-ai-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id vpc-xxx \
    --health-check-path /health/live \
    --region us-east-1
```

## 📊 Monitoreo y Logging

### Prometheus + Grafana

Si usas docker-compose localmente, Prometheus y Grafana están configurados.

Para producción, considera:
- **AWS Managed Prometheus** (AMP)
- **AWS Managed Grafana**
- **CloudWatch Metrics**

### CloudWatch Dashboards

Crea dashboards personalizados para:
- Request rate
- Error rate
- Latency (p50, p95, p99)
- Celery task queue length
- S3 storage usage

## 📈 Escalado Automático

### ECS Auto Scaling

```bash
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/faceless-video-ai-cluster/faceless-video-ai-service \
    --min-capacity 2 \
    --max-capacity 10 \
    --region us-east-1

# Crear scaling policy
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/faceless-video-ai-cluster/faceless-video-ai-service \
    --policy-name cpu-scaling-policy \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        }
    }' \
    --region us-east-1
```

## 🔒 Seguridad

### 1. Network Security

- Usa VPC privadas para contenedores
- Configura security groups restrictivos
- Usa NAT Gateway para acceso a internet

### 2. Secrets Management

- Usa AWS Secrets Manager para API keys
- Rota secrets regularmente
- Usa IAM roles en lugar de credenciales hardcodeadas

### 3. API Security

- Configura rate limiting en API Gateway
- Implementa OAuth2/JWT authentication
- Usa HTTPS/TLS everywhere
- Configura CORS apropiadamente

## 🐛 Troubleshooting

### Ver logs de ECS

```bash
aws logs tail /ecs/faceless-video-ai --follow
```

### Verificar estado de tasks

```bash
aws ecs list-tasks --cluster faceless-video-ai-cluster
aws ecs describe-tasks --cluster faceless-video-ai-cluster --tasks task-id
```

### Debugging Lambda

```bash
aws lambda invoke \
    --function-name faceless-video-ai-api \
    --payload '{"httpMethod": "GET", "path": "/health"}' \
    response.json
```

## 📚 Recursos Adicionales

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## ✅ Checklist de Deployment

- [ ] ECR repository creado
- [ ] Imagen Docker construida y subida
- [ ] IAM roles configurados
- [ ] Secrets en Secrets Manager
- [ ] RDS PostgreSQL creado
- [ ] ElastiCache Redis creado
- [ ] S3 bucket creado
- [ ] ECS cluster creado
- [ ] Task definition registrada
- [ ] ECS service creado y corriendo
- [ ] Load balancer configurado
- [ ] Health checks pasando
- [ ] CloudWatch logs funcionando
- [ ] Auto-scaling configurado
- [ ] Security groups configurados
- [ ] DNS/Route53 configurado (opcional)

---

**Nota**: Reemplaza todos los valores placeholder (YOUR_ACCOUNT_ID, sg-xxx, etc.) con tus valores reales de AWS.




