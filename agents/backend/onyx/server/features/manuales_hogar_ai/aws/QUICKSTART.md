# Quick Start Guide - AWS Deployment

Get your Manuales Hogar AI service running on AWS in 15 minutes!

## Prerequisites Checklist

- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Docker installed and running
- [ ] Python 3.11+ installed
- [ ] AWS CDK installed (`npm install -g aws-cdk`)
- [ ] AWS account with billing enabled
- [ ] OpenRouter API key ready

## Step 1: Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter default region (e.g., us-east-1)
# Enter default output format (json)
```

## Step 2: Set Environment Variables

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION="us-east-1"
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
export ENVIRONMENT="dev"
```

## Step 3: Deploy Infrastructure

### Option A: Automated Script (Recommended)

**Linux/Mac:**
```bash
cd agents/backend/onyx/server/features/manuales_hogar_ai/aws/scripts
chmod +x deploy.sh
./deploy.sh
```

**Windows (PowerShell):**
```powershell
cd agents/backend/onyx/server/features/manuales_hogar_ai/aws/scripts
.\deploy.ps1 -AwsAccountId $env:AWS_ACCOUNT_ID -AwsRegion "us-east-1"
```

### Option B: Manual Steps

1. **Build and push Docker image:**
```bash
cd agents/backend/onyx/server/features/manuales_hogar_ai

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository
aws ecr create-repository --repository-name manuales-hogar-ai --region $AWS_REGION || true

# Build and push
docker build -t manuales-hogar-ai:latest .
docker tag manuales-hogar-ai:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/manuales-hogar-ai:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/manuales-hogar-ai:latest
```

2. **Deploy CDK stack:**
```bash
cd aws/infrastructure

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Bootstrap CDK (first time only)
cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION

# Deploy
cdk deploy --require-approval never
```

## Step 4: Initialize Database

```bash
cd aws/scripts
chmod +x init_db.sh
./init_db.sh
```

## Step 5: Get Your Service URL

```bash
aws cloudformation describe-stacks \
    --stack-name manuales-hogar-ai \
    --region $AWS_REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
    --output text
```

Your service will be available at: `http://<load-balancer-dns>`

## Step 6: Verify Deployment

```bash
# Health check
curl http://<load-balancer-dns>/api/v1/health

# Or use the script
cd aws/scripts
chmod +x health-check.sh
./health-check.sh
```

## Common Issues

### Issue: "CDK bootstrap required"

**Solution:**
```bash
cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION
```

### Issue: "ECR repository not found"

**Solution:**
```bash
aws ecr create-repository --repository-name manuales-hogar-ai --region $AWS_REGION
```

### Issue: "Insufficient permissions"

**Solution:** Ensure your AWS user has permissions for:
- ECS, RDS, EC2, ECR, CloudWatch, Secrets Manager, IAM, CloudFormation

### Issue: "Database connection failed"

**Solution:**
1. Wait 5-10 minutes for RDS to fully initialize
2. Check security groups allow traffic from ECS to RDS
3. Verify database credentials in Secrets Manager

## Next Steps

1. **Set up HTTPS:** Add SSL certificate to ALB
2. **Configure custom domain:** Point your domain to the load balancer
3. **Set up monitoring:** Configure CloudWatch dashboards
4. **Enable backups:** Configure RDS automated backups
5. **Set up CI/CD:** Configure GitHub Actions (see `.github/workflows/deploy-aws.yml`)

## Cost Estimate

For development environment:
- **ECS Fargate:** ~$30/month (2 tasks, 1 vCPU, 2GB RAM each)
- **RDS t3.micro:** ~$15/month
- **ALB:** ~$20/month
- **Data transfer:** ~$5/month
- **Total:** ~$70/month

For production, expect 2-3x these costs.

## Cleanup

To remove all resources and stop charges:

```bash
cd aws/infrastructure
cdk destroy
```

**Warning:** This deletes everything including the database!

## Getting Help

- Check logs: `aws/scripts/view-logs.sh`
- Review documentation: `aws/README.md`
- Check CloudWatch: AWS Console > CloudWatch > Log Groups > `/ecs/manuales-hogar-ai`




