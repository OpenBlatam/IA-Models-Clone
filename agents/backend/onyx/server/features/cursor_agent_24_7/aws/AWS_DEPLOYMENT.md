# 🚀 Guía de Despliegue en AWS - Cursor Agent 24/7

Esta guía cubre el despliegue completo del Cursor Agent 24/7 en AWS usando múltiples opciones: ECS Fargate, Lambda, y EKS.

## 📋 Tabla de Contenidos

- [Arquitectura](#arquitectura)
- [Prerrequisitos](#prerrequisitos)
- [Opciones de Despliegue](#opciones-de-despliegue)
- [Despliegue con Terraform](#despliegue-con-terraform)
- [Despliegue Manual](#despliegue-manual)
- [Configuración de Servicios AWS](#configuración-de-servicios-aws)
- [Monitoreo y Logging](#monitoreo-y-logging)
- [Optimizaciones](#optimizaciones)
- [Troubleshooting](#troubleshooting)

## 🏗️ Arquitectura

### Opción 1: ECS Fargate (Recomendado para producción)

```
Internet → ALB → ECS Fargate (2+ tasks) → DynamoDB
                              ↓
                         ElastiCache Redis
                              ↓
                         CloudWatch Logs
```

### Opción 2: AWS Lambda (Serverless)

```
API Gateway → Lambda Function → DynamoDB
                      ↓
              ElastiCache Redis
                      ↓
              CloudWatch Logs
```

### Opción 3: EKS (Kubernetes)

```
Internet → Ingress → EKS Pods → DynamoDB
                            ↓
                      ElastiCache Redis
```

## ✅ Prerrequisitos

1. **AWS CLI** instalado y configurado
   ```bash
   aws --version
   aws configure
   ```

2. **Docker** instalado
   ```bash
   docker --version
   ```

3. **Terraform** (opcional, para IaC)
   ```bash
   terraform --version
   ```

4. **Permisos AWS** necesarios:
   - ECR (Elastic Container Registry)
   - ECS (Elastic Container Service)
   - DynamoDB
   - ElastiCache
   - CloudWatch
   - IAM (para crear roles)
   - VPC (para networking)

## 🚀 Opciones de Despliegue

### Opción 1: Despliegue con Terraform (Recomendado)

#### Paso 1: Configurar variables

Edita `aws/terraform/variables.tf` o crea `terraform.tfvars`:

```hcl
aws_region   = "us-east-1"
environment  = "prod"
desired_count = 2
```

#### Paso 2: Inicializar Terraform

```bash
cd aws/terraform
terraform init
```

#### Paso 3: Plan y aplicar

```bash
# Ver qué se va a crear
terraform plan

# Aplicar cambios
terraform apply
```

#### Paso 4: Obtener outputs

```bash
terraform output
```

### Opción 2: Despliegue Manual con Scripts

#### Usando el script de deployment

```bash
# Para ECS
./aws/deploy.sh ecs

# Para Lambda (ZIP)
./aws/deploy.sh lambda

# Para Lambda (Container)
./aws/deploy.sh lambda-container
```

#### Variables de entorno

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPOSITORY=cursor-agent-24-7
export IMAGE_TAG=latest
```

### Opción 3: Despliegue Manual Paso a Paso

#### 1. Crear ECR Repository

```bash
aws ecr create-repository \
    --repository-name cursor-agent-24-7 \
    --region us-east-1
```

#### 2. Build y Push Docker Image

```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t cursor-agent-24-7:latest .

# Tag
docker tag cursor-agent-24-7:latest \
    <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/cursor-agent-24-7:latest

# Push
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/cursor-agent-24-7:latest
```

#### 3. Crear DynamoDB Tables

```bash
# Tabla de estado
aws dynamodb create-table \
    --table-name cursor-agent-state \
    --attribute-definitions AttributeName=id,AttributeType=S \
    --key-schema AttributeName=id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1

# Tabla de caché
aws dynamodb create-table \
    --table-name cursor-agent-cache \
    --attribute-definitions AttributeName=key,AttributeType=S \
    --key-schema AttributeName=key,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-east-1
```

#### 4. Crear ECS Cluster y Service

Ver documentación de AWS ECS o usar Terraform.

## ⚙️ Configuración de Servicios AWS

### DynamoDB

El agente usa DynamoDB para:
- **Estado persistente**: `cursor-agent-state`
- **Caché**: `cursor-agent-cache` (opcional, usar Redis si está disponible)

### ElastiCache Redis

Para mejor rendimiento, usar Redis:

```bash
# Crear ElastiCache Redis (usar Terraform o AWS Console)
aws elasticache create-replication-group \
    --replication-group-id cursor-agent-redis \
    --description "Redis for cursor-agent" \
    --node-type cache.t3.micro \
    --num-cache-clusters 1
```

### CloudWatch

Logs automáticos desde ECS/Lambda. Configurar alertas:

```bash
# Crear alarmas (ejemplo)
aws cloudwatch put-metric-alarm \
    --alarm-name cursor-agent-high-error-rate \
    --alarm-description "Alert when error rate is high" \
    --metric-name Errors \
    --namespace AWS/ECS \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold
```

## 📊 Monitoreo y Logging

### CloudWatch Logs

Los logs se envían automáticamente a CloudWatch:

- **Log Group**: `/aws/cursor-agent-24-7`
- **Log Stream**: Por cada task/function

Ver logs:

```bash
aws logs tail /aws/cursor-agent-24-7 --follow
```

### Métricas

El agente expone métricas en:
- `/api/metrics` (formato Prometheus)
- CloudWatch Metrics (automático desde ECS)

### Health Checks

```bash
# Verificar salud
curl https://<alb-dns-name>/api/health

# Verificar estado
curl https://<alb-dns-name>/api/status
```

## 🎯 Optimizaciones

### Para Lambda (Cold Start)

1. **Usar container images** en lugar de ZIP
2. **Provisioned concurrency** para funciones críticas
3. **Lazy loading** de dependencias pesadas
4. **Mangum** optimizado para Lambda

### Para ECS

1. **Auto Scaling** basado en CPU/Memory
2. **Health checks** configurados
3. **Task placement** strategies
4. **Container insights** habilitado

### Para Redis

1. **Connection pooling**
2. **Pipelining** para múltiples comandos
3. **TTL** apropiado para caché

## 🔧 Variables de Entorno

Configurar en ECS Task Definition o Lambda:

```bash
AWS_REGION=us-east-1
DYNAMODB_TABLE_NAME=cursor-agent-state
CACHE_TYPE=elasticache  # o "dynamodb" o "memory"
REDIS_ENDPOINT=<redis-endpoint>
REDIS_PORT=6379
API_PORT=8024
AGENT_PERSISTENT_STORAGE=true
AGENT_AUTO_RESTART=false  # Lambda se reinicia automáticamente
```

## 🐛 Troubleshooting

### Problemas Comunes

#### 1. Error de conexión a DynamoDB

**Síntoma**: `ResourceNotFoundException` o `AccessDeniedException`

**Solución**:
- Verificar que la tabla existe
- Verificar permisos IAM del task/function role
- Verificar región AWS

#### 2. Cold starts lentos en Lambda

**Síntoma**: Primera request tarda mucho

**Solución**:
- Usar provisioned concurrency
- Reducir tamaño del package
- Usar container images

#### 3. ECS tasks no inician

**Síntoma**: Tasks en estado `STOPPED`

**Solución**:
- Verificar logs en CloudWatch
- Verificar permisos IAM
- Verificar configuración de red (security groups, subnets)
- Verificar health checks

#### 4. Redis no conecta

**Síntoma**: Errores de conexión a Redis

**Solución**:
- Verificar security groups (puerto 6379)
- Verificar que Redis está en la misma VPC
- Verificar endpoint correcto

## 📚 Recursos Adicionales

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Mangum Documentation](https://mangum.io/)

## 🔐 Seguridad

### Mejores Prácticas

1. **IAM Roles**: Usar roles con permisos mínimos necesarios
2. **Secrets**: Usar AWS Secrets Manager para credenciales
3. **VPC**: Colocar recursos en VPC privada cuando sea posible
4. **Encryption**: Habilitar encryption en DynamoDB y ElastiCache
5. **HTTPS**: Usar ALB con certificado SSL/TLS
6. **WAF**: Considerar AWS WAF para protección adicional

## 💰 Costos Estimados

### ECS Fargate (2 tasks, 1GB RAM cada uno)
- Compute: ~$30-50/mes
- ALB: ~$20/mes
- DynamoDB: Pay-per-request (muy bajo para uso moderado)
- ElastiCache: ~$15/mes (cache.t3.micro)
- CloudWatch: ~$5-10/mes

**Total estimado**: ~$70-95/mes

### Lambda
- Requests: $0.20 por 1M requests
- Compute: $0.0000166667 por GB-second
- DynamoDB: Pay-per-request
- API Gateway: $3.50 por 1M requests

**Total estimado**: Muy variable según uso

## 🎉 Siguiente Paso

Una vez desplegado, verifica:

1. ✅ Health check responde
2. ✅ Logs aparecen en CloudWatch
3. ✅ DynamoDB tables tienen datos
4. ✅ Redis conecta (si está habilitado)
5. ✅ Auto-scaling funciona (si está configurado)

¡Feliz deployment! 🚀




