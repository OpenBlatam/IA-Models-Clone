# AWS Deployment Guide - Faceless Video AI

This guide provides comprehensive instructions for deploying Faceless Video AI on AWS using Docker containers, ECS/Fargate, and serverless options.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Deployment Options](#deployment-options)
4. [ECS/Fargate Deployment](#ecsfargate-deployment)
5. [Serverless (Lambda) Deployment](#serverless-lambda-deployment)
6. [Infrastructure as Code](#infrastructure-as-code)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- AWS CLI v2 installed and configured
- Docker installed
- Terraform or AWS CloudFormation knowledge
- kubectl (if using EKS)
- Git

### AWS Account Setup

1. **Create AWS Account** with appropriate permissions
2. **Configure AWS CLI**:
   ```bash
   aws configure
   ```
3. **Set up IAM Roles** with necessary permissions:
   - ECS Task Execution Role
   - ECS Task Role
   - Lambda Execution Role
   - S3 Access
   - Secrets Manager Access

### Required AWS Services

- Amazon ECR (Elastic Container Registry)
- Amazon ECS (Elastic Container Service) or AWS Fargate
- Amazon RDS (PostgreSQL)
- Amazon ElastiCache (Redis)
- Amazon S3 (Storage)
- AWS Secrets Manager
- Amazon CloudWatch (Logging & Monitoring)
- Application Load Balancer
- AWS API Gateway (for serverless)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Load Balancer                  │
│                    (HTTPS/HTTP Termination)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              ECS Fargate Service (API + Workers)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   API Task   │  │ Worker Task  │  │ Worker Task  │     │
│  │  (FastAPI)   │  │  (Celery)    │  │  (Celery)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     RDS      │  │  ElastiCache │  │      S3      │
│  PostgreSQL  │  │    Redis     │  │   Storage    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Deployment Options

### Option 1: ECS/Fargate (Recommended for Production)

**Pros:**
- Full control over container environment
- Better for long-running processes
- Supports background workers (Celery)
- Can handle large video processing tasks

**Cons:**
- Higher cost for always-on services
- More complex setup

### Option 2: AWS Lambda (Serverless)

**Pros:**
- Pay-per-use pricing
- Automatic scaling
- No server management
- Fast deployment

**Cons:**
- 15-minute timeout limit
- Cold start latency
- Limited for long-running video processing
- Need to offload heavy processing to other services

### Option 3: Hybrid Approach

- API Gateway + Lambda for API endpoints
- ECS Fargate for background workers
- Step Functions for orchestration

## ECS/Fargate Deployment

### Step 1: Build and Push Docker Image to ECR

```bash
cd agents/backend/onyx/server/features/faceless_video_ai

# Make scripts executable
chmod +x aws/deployment/build-and-push.sh
chmod +x aws/deployment/deploy-ecs.sh

# Build and push
./aws/deployment/build-and-push.sh
```

Or manually:

```bash
# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
ECR_REPOSITORY=faceless-video-ai

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create repository if it doesn't exist
aws ecr create-repository \
  --repository-name $ECR_REPOSITORY \
  --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true

# Build image
docker build -t $ECR_REPOSITORY:latest .

# Tag image
docker tag $ECR_REPOSITORY:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
```

### Step 2: Deploy Infrastructure with CloudFormation

```bash
# Update CloudFormation template with your VPC and subnet IDs
# Edit aws/deployment/cloudformation-template.yaml

# Deploy stack
aws cloudformation create-stack \
  --stack-name faceless-video-ai-infrastructure \
  --template-body file://aws/deployment/cloudformation-template.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=production \
    ParameterKey=VpcId,ParameterValue=vpc-xxxxxxxxx \
    ParameterKey=SubnetIds,ParameterValue=subnet-xxxxx,subnet-yyyyy \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for stack creation
aws cloudformation wait stack-create-complete \
  --stack-name faceless-video-ai-infrastructure
```

### Step 3: Store Secrets in AWS Secrets Manager

```bash
# Store API keys
aws secretsmanager create-secret \
  --name faceless-video-ai/production/openai-api-key \
  --secret-string '{"api_key": "your-openai-key"}'

aws secretsmanager create-secret \
  --name faceless-video-ai/production/stability-api-key \
  --secret-string '{"api_key": "your-stability-key"}'

aws secretsmanager create-secret \
  --name faceless-video-ai/production/elevenlabs-api-key \
  --secret-string '{"api_key": "your-elevenlabs-key"}'

# Store database URL
aws secretsmanager create-secret \
  --name faceless-video-ai/production/database-url \
  --secret-string 'postgresql://user:pass@host:5432/dbname'

# Store Redis URL
aws secretsmanager create-secret \
  --name faceless-video-ai/production/redis-url \
  --secret-string 'redis://host:6379/0'
```

### Step 4: Update ECS Task Definition

```bash
# Edit aws/deployment/ecs-task-definition.json
# Update:
# - ACCOUNT_ID with your AWS Account ID
# - REGION with your AWS region
# - Secret ARNs with actual secret ARNs

# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://aws/deployment/ecs-task-definition.json
```

### Step 5: Create ECS Service

```bash
# Edit aws/deployment/ecs-service-definition.json
# Update subnet IDs and security group IDs

# Create service
aws ecs create-service \
  --cli-input-json file://aws/deployment/ecs-service-definition.json
```

Or use the deployment script:

```bash
./aws/deployment/deploy-ecs.sh
```

### Step 6: Verify Deployment

```bash
# Check service status
aws ecs describe-services \
  --cluster faceless-video-ai-production \
  --services faceless-video-ai-service

# Get load balancer DNS
aws cloudformation describe-stacks \
  --stack-name faceless-video-ai-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
  --output text

# Test health endpoint
curl http://<load-balancer-dns>/health
```

## Serverless (Lambda) Deployment

### Step 1: Install Serverless Framework

```bash
npm install -g serverless
npm install --save-dev serverless-python-requirements
npm install --save-dev serverless-offline
```

### Step 2: Configure Serverless

```bash
cd aws/deployment

# Update serverless.yml with your AWS account details
# Set stage (dev, staging, production)

# Install Python dependencies for Lambda
pip install -r lambda-requirements.txt -t .
```

### Step 3: Deploy to Lambda

```bash
# Deploy
serverless deploy --stage production

# Deploy function only
serverless deploy function --function api --stage production

# View logs
serverless logs -f api --stage production --tail
```

### Step 4: Set up API Gateway

The Serverless Framework automatically creates API Gateway. You can also configure:

- Custom domain
- API keys
- Usage plans
- Rate limiting
- CORS

## Infrastructure as Code

### CloudFormation

The CloudFormation template (`cloudformation-template.yaml`) creates:

- VPC and networking (if not provided)
- ECS Cluster
- RDS PostgreSQL database
- ElastiCache Redis
- S3 bucket
- ECR repository
- Application Load Balancer
- Security groups
- IAM roles
- CloudWatch log groups
- Secrets Manager secrets

### Terraform (Alternative)

You can also use Terraform. Example structure:

```hcl
# terraform/main.tf
provider "aws" {
  region = "us-east-1"
}

module "faceless_video_ai" {
  source = "./modules/faceless-video-ai"
  
  environment = "production"
  vpc_id      = var.vpc_id
  subnet_ids  = var.subnet_ids
}
```

## CI/CD Pipeline

### GitHub Actions

The pipeline (`.github/workflows/deploy.yml`) includes:

1. **Test**: Run unit and integration tests
2. **Build**: Build Docker image
3. **Security Scan**: Scan for vulnerabilities
4. **Deploy**: Deploy to ECS

### Setup GitHub Secrets

```bash
# In GitHub repository settings > Secrets
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
```

### Manual CI/CD

```bash
# Run tests
pytest tests/

# Build and push
./aws/deployment/build-and-push.sh

# Deploy
./aws/deployment/deploy-ecs.sh
```

## Monitoring and Observability

### CloudWatch

- **Logs**: Automatic log aggregation from ECS tasks
- **Metrics**: Custom metrics via Prometheus
- **Alarms**: Set up alarms for errors, latency, etc.

### Prometheus + Grafana

Deploy with docker-compose:

```bash
docker-compose up -d prometheus grafana
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Key Metrics to Monitor

- API request rate and latency
- Video generation success/failure rate
- Queue size
- Active video generations
- Error rates by type
- Resource utilization (CPU, memory)
- S3 storage usage
- Database connection pool

## Security Best Practices

### 1. Secrets Management

- ✅ Use AWS Secrets Manager (not environment variables)
- ✅ Rotate secrets regularly
- ✅ Use IAM roles (not access keys in code)

### 2. Network Security

- ✅ Use VPC with private subnets
- ✅ Security groups with least privilege
- ✅ Enable VPC Flow Logs
- ✅ Use HTTPS/TLS everywhere

### 3. Container Security

- ✅ Scan images for vulnerabilities
- ✅ Use non-root user in containers
- ✅ Keep base images updated
- ✅ Minimize attack surface

### 4. Access Control

- ✅ IAM roles with least privilege
- ✅ Enable MFA for AWS console
- ✅ Use API keys with rate limiting
- ✅ Implement JWT authentication

### 5. Data Protection

- ✅ Encrypt data at rest (S3, RDS)
- ✅ Encrypt data in transit (TLS)
- ✅ Enable S3 versioning
- ✅ Regular backups

## Troubleshooting

### Common Issues

#### 1. ECS Task Fails to Start

```bash
# Check task logs
aws logs tail /ecs/faceless-video-ai --follow

# Check task definition
aws ecs describe-task-definition --task-definition faceless-video-ai

# Check service events
aws ecs describe-services \
  --cluster faceless-video-ai-production \
  --services faceless-video-ai-service
```

#### 2. Cannot Connect to Database

- Check security group rules
- Verify database endpoint
- Check credentials in Secrets Manager
- Verify subnet configuration

#### 3. High Memory Usage

- Increase task memory allocation
- Optimize video processing
- Use S3 for temporary files
- Implement streaming for large files

#### 4. Slow API Responses

- Check CloudWatch metrics
- Review database query performance
- Enable caching (Redis)
- Scale out ECS service
- Use CDN for static assets

#### 5. Lambda Timeout

- Increase Lambda timeout (max 15 min)
- Offload heavy processing to ECS workers
- Use Step Functions for long workflows
- Implement async processing

### Debug Commands

```bash
# Execute command in running container
aws ecs execute-command \
  --cluster faceless-video-ai-production \
  --task <task-id> \
  --container faceless-video-api \
  --command "/bin/sh" \
  --interactive

# View CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=faceless-video-ai-service \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average

# Check S3 bucket
aws s3 ls s3://faceless-video-ai-production-<account-id>/

# Test database connection
aws rds describe-db-instances \
  --db-instance-identifier faceless-video-ai-production
```

## Cost Optimization

### Tips

1. **Use Fargate Spot** for non-critical workloads
2. **Auto-scaling** based on demand
3. **S3 Lifecycle Policies** to archive old videos
4. **Reserved Capacity** for RDS (if predictable)
5. **CloudWatch Logs Retention** (set to 7-30 days)
6. **Delete unused resources**

### Estimated Monthly Costs (Production)

- ECS Fargate (2 tasks): ~$150-300
- RDS (db.t3.medium): ~$100-150
- ElastiCache (cache.t3.medium): ~$50-100
- S3 Storage (100GB): ~$2-5
- Data Transfer: ~$20-50
- CloudWatch: ~$10-30

**Total: ~$332-635/month** (varies by usage)

## Next Steps

1. Set up custom domain with Route 53
2. Configure SSL/TLS certificates (ACM)
3. Implement WAF rules
4. Set up automated backups
5. Configure disaster recovery
6. Implement blue-green deployments
7. Set up staging environment
8. Configure auto-scaling policies

## Support

For issues or questions:
- Check CloudWatch logs
- Review ECS service events
- Check GitHub Issues
- Contact DevOps team

---

**Last Updated**: 2024
**Version**: 1.0.0




