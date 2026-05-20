#!/bin/bash

TARGET_GROUP_ARN=$(aws elbv2 describe-target-groups --names blatam-tg --query 'TargetGroups[0].TargetGroupArn' --output text --region us-east-1)

echo "Target Group ARN: ${TARGET_GROUP_ARN}"

aws ecs update-service \
  --cluster blatam-cluster \
  --service blatam-service \
  --load-balancers '[{"targetGroupArn":"'${TARGET_GROUP_ARN}'","containerName":"blatam-academy","containerPort":3000}]' \
  --region us-east-1

echo "ECS service updated with ALB target group"

echo "Waiting for service to stabilize..."
aws ecs wait services-stable --cluster blatam-cluster --services blatam-service --region us-east-1

echo "Checking target health..."
aws elbv2 describe-target-health --target-group-arn ${TARGET_GROUP_ARN} --region us-east-1

ALB_DNS=$(aws elbv2 describe-load-balancers --names blatam-alb --query 'LoadBalancers[0].DNSName' --output text --region us-east-1)

echo ""
echo "=== ALB CONFIGURATION COMPLETE ==="
echo "ALB DNS Name: ${ALB_DNS}"
echo ""
echo "Para HostGator, crear registro CNAME:"
echo "  Nombre: blatam.org"
echo "  Valor: ${ALB_DNS}"
echo ""
echo "La aplicación estará disponible en: http://${ALB_DNS}"
