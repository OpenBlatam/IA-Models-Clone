#!/bin/bash

# Configuración
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="blatam-academy"
CLUSTER_NAME="blatam-cluster"
SERVICE_NAME="blatam-service"
MIN_CAPACITY=2
MAX_CAPACITY=10
DESIRED_CAPACITY=2

# Crear repositorio ECR si no existe
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} || \
aws ecr create-repository --repository-name ${ECR_REPOSITORY}

# Autenticar Docker con ECR
aws ecr get-login-password --region ${AWS_REGION} | \
docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Construir la imagen
docker build -t ${ECR_REPOSITORY} .

# Etiquetar la imagen
docker tag ${ECR_REPOSITORY}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest

# Subir la imagen a ECR
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest

# Crear cluster ECS si no existe
aws ecs describe-clusters --clusters ${CLUSTER_NAME} || \
aws ecs create-cluster --cluster-name ${CLUSTER_NAME}

./update-task-definition.sh

# Registrar la definición de tarea
TASK_DEFINITION=$(aws ecs register-task-definition \
  --cli-input-json file://aws-task-definition-updated.json \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)

# Crear o actualizar el servicio con auto-scaling
aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} || \
aws ecs create-service \
  --cluster ${CLUSTER_NAME} \
  --service-name ${SERVICE_NAME} \
  --task-definition ${TASK_DEFINITION} \
  --desired-count ${DESIRED_CAPACITY} \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-03512b444b7165046,subnet-001b17eae6a6cd3fa],securityGroups=[sg-03cbfa32324f3766c],assignPublicIp=ENABLED}" \
  --deployment-configuration "maximumPercent=200,minimumHealthyPercent=100" \
  --enable-execute-command

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

echo "Despliegue completado. La aplicación estará disponible en:"
echo "https://blatam-academy.${AWS_REGION}.elasticbeanstalk.com"

# CONFIGURA ESTAS VARIABLES:
EC2_USER=ec2-user
EC2_HOST=ec2-18-206-225-74.compute-1.amazonaws.com
KEY_PATH=/Users/adan/blatam.pem   # <-- Cambia esto si tu llave está en otra ruta
PROJECT_DIR=next-saas-stripe-starter    