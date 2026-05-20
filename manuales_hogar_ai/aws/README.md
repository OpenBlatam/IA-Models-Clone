# AWS Deployment Guide for Manuales Hogar AI

This directory contains all the necessary files and scripts to deploy the Manuales Hogar AI service to AWS.

## Architecture Overview

The deployment uses:
- **ECS Fargate** for containerized application hosting
- **RDS PostgreSQL** for database
- **Application Load Balancer** for traffic distribution
- **CloudWatch** for logging and monitoring
- **Secrets Manager** for secure credential storage
- **Auto-scaling** based on CPU and memory utilization

## Prerequisites

1. **AWS CLI** installed and configured
2. **Docker** installed
3. **Python 3.11+** installed
4. **AWS CDK** installed (`npm install -g aws-cdk`)
5. **AWS Account** with appropriate permissions

## Required AWS Permissions

Your AWS user/role needs permissions for:
- ECS (create clusters, services, task definitions)
- RDS (create databases, security groups)
- EC2 (VPC, subnets, security groups, load balancers)
- ECR (create repositories, push images)
- CloudWatch (create log groups, alarms)
- Secrets Manager (create and manage secrets)
- IAM (create roles and policies)
- CloudFormation (deploy stacks)

## Environment Variables

Set the following environment variables:

```bash
export AWS_ACCOUNT_ID="your-aws-account-id"
export AWS_REGION="us-east-1"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export DB_USERNAME="admin"  # Optional, defaults to admin
export DB_PASSWORD="your-db-password"  # Optional, will be auto-generated if not set
export ENVIRONMENT="dev"  # dev, staging, or prod
```

## Quick Start

### Option 1: Using Deployment Scripts

#### Linux/Mac:
```bash
cd aws/scripts
chmod +x deploy.sh
./deploy.sh
```

#### Windows (PowerShell):
```powershell
cd aws/scripts
.\deploy.ps1 -AwsAccountId "your-account-id" -AwsRegion "us-east-1"
```

### Option 2: Manual Deployment

1. **Build and push Docker image:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository
aws ecr create-repository --repository-name manuales-hogar-ai --region us-east-1

# Build and push
docker build -t manuales-hogar-ai:latest .
docker tag manuales-hogar-ai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/manuales-hogar-ai:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/manuales-hogar-ai:latest
```

2. **Deploy CDK stack:**
```bash
cd aws/infrastructure
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Bootstrap CDK (first time only)
cdk bootstrap aws://<account-id>/us-east-1

# Deploy
cdk deploy
```

3. **Update ECS service with new image:**
```bash
aws ecs update-service \
  --cluster manuales-hogar-ai-cluster \
  --service manuales-hogar-ai-service \
  --force-new-deployment \
  --region us-east-1
```

## CI/CD with GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/deploy-aws.yml`) that automatically deploys on push to main/develop branches.

### Required GitHub Secrets

Configure these secrets in your GitHub repository:

- `AWS_DEPLOY_ROLE_ARN`: ARN of the IAM role for deployment
- `AWS_ACCOUNT_ID`: Your AWS account ID
- `OPENROUTER_API_KEY`: OpenRouter API key
- `DB_USERNAME`: Database username (optional)
- `DB_PASSWORD`: Database password (optional)

### Manual Deployment via GitHub Actions

1. Go to Actions tab in GitHub
2. Select "Deploy to AWS" workflow
3. Click "Run workflow"
4. Select environment (dev/staging/prod)
5. Click "Run workflow"

## Configuration

### CDK Stack Configuration

Edit `aws/infrastructure/app.py` to customize:
- VPC CIDR blocks
- Instance types and sizes
- Auto-scaling limits
- Database configuration
- Environment-specific settings

### Application Configuration

The application reads configuration from environment variables. Key variables:

- `OPENROUTER_API_KEY`: Stored in AWS Secrets Manager
- `DB_HOST`: Automatically set from RDS endpoint
- `DB_PORT`: Automatically set to 5432
- `DB_USER`: Retrieved from Secrets Manager
- `DB_PASSWORD`: Retrieved from Secrets Manager
- `DB_NAME`: Set to "manuales_hogar"

## Monitoring

### CloudWatch Logs

Logs are automatically sent to CloudWatch Log Group: `/ecs/manuales-hogar-ai`

View logs:
```bash
aws logs tail /ecs/manuales-hogar-ai --follow --region us-east-1
```

### CloudWatch Alarms

The stack creates alarms for:
- High CPU utilization (>80%)
- High memory utilization (>85%)
- Database connection issues

Alarms send notifications to an SNS topic.

### Metrics

Key metrics to monitor:
- ECS service CPU/Memory utilization
- Application Load Balancer request count and latency
- RDS database connections and CPU utilization
- ECS task count (auto-scaling)

## Scaling

The service is configured with auto-scaling:
- **Min tasks**: 2
- **Max tasks**: 10
- **CPU threshold**: 70%
- **Memory threshold**: 80%

Adjust these values in `aws/infrastructure/app.py`.

## Database Migrations

After deployment, run Alembic migrations:

```bash
# Get database endpoint from CDK outputs
DB_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name manuales-hogar-ai \
  --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
  --output text)

# Run migrations (from a container or local machine with DB access)
export DATABASE_URL="postgresql+asyncpg://admin:password@${DB_ENDPOINT}:5432/manuales_hogar"
alembic upgrade head
```

## Security Best Practices

1. **Secrets Management**: All sensitive data is stored in AWS Secrets Manager
2. **Network Security**: Database is in private subnet, only accessible from ECS tasks
3. **IAM Roles**: Least privilege principle for all IAM roles
4. **Encryption**: RDS encryption at rest enabled
5. **HTTPS**: Configure SSL certificate for ALB (add to CDK stack)

## Troubleshooting

### Service won't start
- Check CloudWatch logs: `/ecs/manuales-hogar-ai`
- Verify secrets are correctly configured
- Check security group rules
- Verify database connectivity

### High latency
- Check ALB metrics
- Review ECS task CPU/memory utilization
- Consider increasing task resources or count

### Database connection errors
- Verify security group allows traffic from ECS to RDS (port 5432)
- Check database credentials in Secrets Manager
- Verify database is in same VPC as ECS tasks

## Cost Optimization

- Use Fargate Spot for non-production environments
- Enable RDS automated backups only for production
- Set up CloudWatch log retention policies
- Use smaller instance types for development

## Cleanup

To remove all resources:

```bash
cd aws/infrastructure
cdk destroy
```

**Warning**: This will delete all resources including the database. Make sure to backup data first!

## Support

For issues or questions, contact the Blatam Academy team.




