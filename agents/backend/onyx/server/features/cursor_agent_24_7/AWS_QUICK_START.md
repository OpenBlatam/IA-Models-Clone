# 🚀 AWS Quick Start - Cursor Agent 24/7

Guía rápida para desplegar el Cursor Agent 24/7 en AWS.

## ⚡ Inicio Rápido (5 minutos)

### Opción 1: Terraform (Recomendado)

```bash
cd aws/terraform
terraform init
terraform plan
terraform apply
```

### Opción 2: Script de Deployment

```bash
# ECS Fargate
./aws/deploy.sh ecs

# Lambda (Container)
./aws/deploy.sh lambda-container
```

## 📋 Prerrequisitos

1. AWS CLI configurado: `aws configure`
2. Docker instalado
3. Permisos AWS (ECR, ECS, DynamoDB, CloudWatch, IAM)

## 🏗️ Arquitectura Desplegada

- **ECS Fargate**: 2 tasks con auto-scaling
- **Application Load Balancer**: Distribución de carga
- **DynamoDB**: Estado persistente y caché
- **ElastiCache Redis**: Caché de alto rendimiento
- **CloudWatch**: Logging y métricas

## 🔧 Configuración

### Variables de Entorno

```bash
export AWS_REGION=us-east-1
export DYNAMODB_TABLE_NAME=cursor-agent-state
export CACHE_TYPE=elasticache
export REDIS_ENDPOINT=<redis-endpoint>
```

### Verificar Despliegue

```bash
# Obtener URL del ALB
terraform output alb_dns_name

# Health check
curl https://<alb-dns-name>/api/health

# Ver logs
aws logs tail /aws/cursor-agent-24-7 --follow
```

## 📚 Documentación Completa

Ver [aws/AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md) para documentación detallada.

## 🎯 Características AWS

✅ **Stateless**: Estado en DynamoDB  
✅ **Auto-scaling**: ECS auto-scaling configurado  
✅ **High Availability**: Múltiples tasks en diferentes AZs  
✅ **Monitoring**: CloudWatch logs y métricas  
✅ **Circuit Breakers**: Protección contra fallos  
✅ **Retries**: Reintentos automáticos con backoff  
✅ **Security**: IAM roles, encryption, VPC  

## 💰 Costos Estimados

- **ECS Fargate**: ~$30-50/mes
- **ALB**: ~$20/mes
- **DynamoDB**: Pay-per-request (muy bajo)
- **ElastiCache**: ~$15/mes
- **CloudWatch**: ~$5-10/mes

**Total**: ~$70-95/mes para producción

## 🆘 Troubleshooting

```bash
# Ver estado de ECS
aws ecs describe-services --cluster cursor-agent-cluster --services cursor-agent-service

# Ver logs
aws logs tail /aws/cursor-agent-24-7 --follow

# Verificar DynamoDB
aws dynamodb describe-table --table-name cursor-agent-state
```

## 📞 Soporte

Para más información, ver:
- [AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md) - Guía completa
- [README.md](README.md) - Documentación general
- [DEPLOYMENT.md](DEPLOYMENT.md) - Despliegue general




