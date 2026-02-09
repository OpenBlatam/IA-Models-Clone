# AWS Deployment Guide for Robot Movement AI

This directory contains all the necessary files and configurations to deploy the Robot Movement AI system on AWS.

## 🏗️ Architecture Options

### Option 1: ECS/Fargate (Recommended for WebSocket and Long-Running Tasks)
- **Best for**: WebSocket connections, real-time feedback, long-running processes
- **Components**: ECS Fargate, Application Load Balancer, VPC, CloudWatch
- **Scaling**: Auto-scaling based on CPU/memory metrics
- **Cost**: Pay per use, no idle costs

### Option 2: AWS Lambda (Serverless)
- **Best for**: Stateless API endpoints, event-driven processing
- **Components**: Lambda, API Gateway, CloudWatch
- **Scaling**: Automatic, scales to zero
- **Limitations**: No WebSocket support, 15-minute timeout
- **Cost**: Pay per request, very cost-effective for low traffic

## 🚀 Advanced Features

This deployment includes advanced production-ready features:

- ✅ **OpenTelemetry** distributed tracing
- ✅ **Rate limiting** with Redis backend
- ✅ **Circuit breakers** for resilience
- ✅ **Redis caching** for performance
- ✅ **Structured logging** with request IDs
- ✅ **Security headers** (OWASP compliance)
- ✅ **OAuth2/JWT** authentication
- ✅ **Celery workers** for async tasks
- ✅ **Kafka** event streaming
- ✅ **Prometheus metrics** for monitoring
- ✅ **API Gateway** integration
- ✅ **WAF** for DDoS protection

See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) for detailed documentation.

## 📋 Prerequisites

1. **AWS CLI** installed and configured
   ```bash
   aws --version
   aws configure
   ```

2. **Docker** installed (for ECS deployment)
   ```bash
   docker --version
   ```

3. **Terraform** installed (optional, for infrastructure as code)
   ```bash
   terraform --version
   ```

4. **AWS SAM CLI** installed (for Lambda deployment)
   ```bash
   sam --version
   ```

5. **Required AWS Permissions**:
   - ECS (for Fargate deployment)
   - ECR (for container registry)
   - VPC (for networking)
   - IAM (for roles and policies)
   - Secrets Manager (for API keys)
   - CloudWatch (for logging)
   - S3 (for model storage)
   - MSK (for Kafka, optional)
   - WAF (for DDoS protection, optional)

## 🚀 Quick Start

### ECS/Fargate Deployment (Recommended)

1. **Deploy Infrastructure with Terraform**:
   ```bash
   cd aws/terraform
   terraform init
   terraform plan
   terraform apply
   ```

2. **Build and Deploy Application**:
   ```bash
   cd aws
   chmod +x deploy.sh
   ./deploy.sh ecs
   ```

3. **Get ALB DNS Name**:
   ```bash
   terraform output -json | jq -r '.alb_dns_name.value'
   ```

### Lambda Deployment (Serverless)

1. **Deploy with SAM**:
   ```bash
   cd aws
   chmod +x deploy.sh
   ./deploy.sh lambda
   ```

2. **Get API URL**:
   ```bash
   aws cloudformation describe-stacks \
     --stack-name robot-movement-ai \
     --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
     --output text
   ```

## 📁 File Structure

```
aws/
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Local development with Docker
├── lambda_handler.py          # Lambda handler for serverless
├── ecs_task_definition.json   # ECS task definition
├── ecs_service.json          # ECS service configuration
├── sam_template.yaml         # AWS SAM template for Lambda
├── deploy.sh                 # Deployment automation script
├── requirements-lambda.txt   # Additional Lambda dependencies
├── .dockerignore            # Docker ignore patterns
├── README.md                # This file
└── terraform/               # Infrastructure as Code
    ├── main.tf              # Main Terraform configuration
    ├── variables.tf         # Variable definitions
    └── outputs.tf           # Output values
```

## 🔧 Configuration

### Environment Variables

Set these in AWS Secrets Manager or ECS task definition:

- `ROBOT_IP`: Robot IP address
- `ROBOT_PORT`: Robot port (default: 30001)
- `ROBOT_BRAND`: Robot brand (kuka, abb, fanuc, universal_robots, generic)
- `ROS_ENABLED`: Enable ROS integration (true/false)
- `FEEDBACK_FREQUENCY`: Feedback frequency in Hz (default: 1000)
- `LLM_PROVIDER`: LLM provider (openai, anthropic, etc.)
- `OPENAI_API_KEY`: OpenAI API key (or ANTHROPIC_API_KEY)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AWS_REGION`: AWS region

### Secrets Manager

Create secrets in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name robot-movement-ai/secrets \
  --secret-string '{
    "openai_api_key": "your-key-here",
    "robot_ip": "192.168.1.100"
  }'
```

## 🔐 Security Best Practices

1. **Use Secrets Manager** for sensitive data (API keys, passwords)
2. **Enable VPC** for ECS tasks (private subnets)
3. **Use IAM Roles** instead of access keys
4. **Enable WAF** on API Gateway for DDoS protection
5. **Enable CloudTrail** for audit logging
6. **Use HTTPS** for all API endpoints
7. **Enable encryption** for S3 buckets and EBS volumes

## 📊 Monitoring and Logging

### CloudWatch Metrics

- ECS: CPU utilization, memory utilization
- ALB: Request count, response time, error rate
- Lambda: Invocations, duration, errors

### CloudWatch Logs

- ECS logs: `/ecs/robot-movement-ai`
- Lambda logs: `/aws/lambda/robot-movement-ai`

### Set up Alarms

```bash
# CPU utilization alarm
aws cloudwatch put-metric-alarm \
  --alarm-name robot-movement-ai-high-cpu \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## 🔄 CI/CD Integration

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
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Deploy
        run: |
          cd aws
          ./deploy.sh ecs
```

## 💰 Cost Optimization

1. **Use Fargate Spot** for non-critical workloads (up to 70% savings)
2. **Enable Auto Scaling** to scale down during low traffic
3. **Use S3 Intelligent-Tiering** for model storage
4. **Enable CloudWatch Logs retention** (default: forever, set to 14 days)
5. **Use Reserved Capacity** for predictable workloads
6. **Monitor and optimize** Lambda memory allocation

## 🐛 Troubleshooting

### ECS Tasks Not Starting

1. Check CloudWatch logs: `/ecs/robot-movement-ai`
2. Verify task definition and IAM roles
3. Check security group rules
4. Verify secrets are accessible

### Lambda Timeout Issues

1. Increase timeout in SAM template
2. Optimize cold start (reduce dependencies)
3. Use Lambda Provisioned Concurrency for critical paths

### API Gateway 502 Errors

1. Check Lambda/ECS health
2. Verify integration configuration
3. Check CloudWatch logs for errors

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [AWS SAM Documentation](https://docs.aws.amazon.com/serverless-application-model/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

## 🤝 Support

For issues or questions:
- Check CloudWatch logs
- Review AWS service health dashboard
- Contact: support@blatam-academy.com

## 📝 License

Copyright (c) 2025 Blatam Academy

