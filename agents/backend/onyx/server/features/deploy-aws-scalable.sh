#!/bin/bash

# Configuración
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="blatam-academy"
CLUSTER_NAME="blatam-cluster"
SERVICE_NAME="blatam-service"
MIN_CAPACITY=2
MAX_CAPACITY=20
DESIRED_CAPACITY=2
VPC_ID="vpc-03e5c5c0e0c917b81"
SUBNET_IDS="subnet-03512b444b7165046,subnet-001b17eae6a6cd3fa"

# Crear Application Load Balancer
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name blatam-alb \
  --subnets ${SUBNET_IDS//,/ } \
  --security-groups sg-03cbfa32324f3766c \
  --scheme internet-facing \
  --type application \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

# Crear target group
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
  --name blatam-tg \
  --protocol HTTP \
  --port 3000 \
  --vpc-id ${VPC_ID} \
  --target-type ip \
  --health-check-path /api/health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 2 \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

# Crear listener HTTP
aws elbv2 create-listener \
  --load-balancer-arn ${ALB_ARN} \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=${TARGET_GROUP_ARN}


# Crear repositorio ECR
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} || \
aws ecr create-repository --repository-name ${ECR_REPOSITORY}

# Autenticar Docker con ECR
aws ecr get-login-password --region ${AWS_REGION} | \
docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Construir y subir imagen
docker build -t ${ECR_REPOSITORY} .
docker tag ${ECR_REPOSITORY}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest

# Crear cluster ECS
aws ecs create-cluster --cluster-name ${CLUSTER_NAME}

# Registrar task definition
TASK_DEFINITION=$(aws ecs register-task-definition \
  --cli-input-json file://aws-task-definition.json \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

aws ecs update-service \
  --cluster ${CLUSTER_NAME} \
  --service ${SERVICE_NAME} \
  --load-balancers "targetGroupArn=${TARGET_GROUP_ARN},containerName=blatam-academy,containerPort=3000" \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[sg-03cbfa32324f3766c],assignPublicIp=ENABLED}"

# Configurar auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/${CLUSTER_NAME}/${SERVICE_NAME} \
  --min-capacity ${MIN_CAPACITY} \
  --max-capacity ${MAX_CAPACITY}

# Configurar políticas de auto-scaling
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/${CLUSTER_NAME}/${SERVICE_NAME} \
  --policy-name cpu-autoscaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 300
  }'

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/${CLUSTER_NAME}/${SERVICE_NAME} \
  --policy-name memory-autoscaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageMemoryUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 300
  }'

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/${CLUSTER_NAME}/${SERVICE_NAME} \
  --policy-name request-count-autoscaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 1000.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 300
  }'

ALB_DNS=$(aws elbv2 describe-load-balancers --load-balancer-arns ${ALB_ARN} --query 'LoadBalancers[0].DNSName' --output text)

echo "Despliegue completado. La aplicación estará disponible en:"
echo "http://${ALB_DNS}"
echo ""
echo "Para HostGator, crear registro CNAME:"
echo "  Nombre: blatam.org (o www.blatam.org)"
echo "  Valor: ${ALB_DNS}"       