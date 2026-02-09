# Quick Start Guide - AWS Deployment

This guide will help you deploy Robot Movement AI to AWS in under 30 minutes.

## 🎯 Prerequisites Checklist

- [ ] AWS Account with admin access
- [ ] AWS CLI installed and configured (`aws configure`)
- [ ] Docker installed and running
- [ ] Terraform installed (optional, for IaC)
- [ ] Basic knowledge of AWS services

## 🚀 Option 1: ECS/Fargate Deployment (Recommended)

### Step 1: Set Up Infrastructure (5 minutes)

```bash
cd aws/terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply infrastructure
terraform apply
```

**Note**: You'll need to provide:
- `openai_api_key`: Your OpenAI API key
- `robot_ip`: Your robot's IP address (or use default)

### Step 2: Build and Push Docker Image (5 minutes)

```bash
cd aws

# Make deploy script executable
chmod +x deploy.sh

# Build and deploy
./deploy.sh ecs
```

This script will:
1. Build the Docker image
2. Create ECR repository (if needed)
3. Push image to ECR
4. Update ECS service

### Step 3: Get Your API URL (1 minute)

```bash
# Get ALB DNS name
terraform output -json | jq -r '.alb_dns_name.value'
```

Your API will be available at: `http://<alb-dns-name>/`

### Step 4: Test the API (2 minutes)

```bash
# Health check
curl http://<alb-dns-name>/health

# Test chat endpoint
curl -X POST http://<alb-dns-name>/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "move to (0.5, 0.3, 0.2)"}'
```

## ⚡ Option 2: Lambda Deployment (Serverless)

### Step 1: Install SAM CLI

```bash
# macOS
brew install aws-sam-cli

# Linux
pip install aws-sam-cli

# Windows
# Download from: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
```

### Step 2: Deploy with SAM

```bash
cd aws

# Build and deploy
sam build
sam deploy --guided
```

**Note**: First time deployment will ask for:
- Stack name: `robot-movement-ai`
- AWS Region: `us-east-1`
- Parameter values (use defaults or customize)

### Step 3: Get API URL

```bash
aws cloudformation describe-stacks \
  --stack-name robot-movement-ai \
  --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
  --output text
```

## 🔧 Configuration

### Environment Variables

Set these in AWS Secrets Manager or ECS task definition:

```bash
# Create secret
aws secretsmanager create-secret \
  --name robot-movement-ai/secrets \
  --secret-string '{
    "openai_api_key": "sk-...",
    "robot_ip": "192.168.1.100"
  }'
```

### Update ECS Task Definition

Edit `aws/ecs_task_definition.json` and update:
- `ACCOUNT_ID`: Your AWS account ID
- `REGION`: Your AWS region
- Environment variables as needed

Then register the task definition:

```bash
aws ecs register-task-definition \
  --cli-input-json file://aws/ecs_task_definition.json
```

## 📊 Monitoring

### View Logs

```bash
# ECS logs
aws logs tail /ecs/robot-movement-ai --follow

# Lambda logs
aws logs tail /aws/lambda/robot-movement-ai --follow
```

### View Metrics

1. Go to AWS Console → CloudWatch
2. Navigate to Metrics → ECS (or Lambda)
3. Select your service/function

## 🐛 Troubleshooting

### Issue: ECS tasks not starting

**Solution**:
1. Check CloudWatch logs: `/ecs/robot-movement-ai`
2. Verify IAM roles have correct permissions
3. Check security group allows traffic on port 8010

### Issue: Lambda timeout

**Solution**:
1. Increase timeout in `sam_template.yaml`:
   ```yaml
   Timeout: 60  # Increase from 30
   ```
2. Optimize cold start (reduce dependencies)

### Issue: API returns 502

**Solution**:
1. Check health endpoint: `/health`
2. Verify ALB target group health checks
3. Check CloudWatch logs for errors

## 📈 Next Steps

1. **Set up Auto Scaling**: Configure ECS auto-scaling based on CPU/memory
2. **Enable HTTPS**: Add SSL certificate to ALB
3. **Set up Monitoring**: Create CloudWatch alarms
4. **Configure CI/CD**: Use GitHub Actions (see `.github/workflows/deploy.yml`)

## 💰 Cost Estimation

### ECS/Fargate (per month)
- Fargate: ~$50-100 (2 tasks, 2 vCPU, 4GB RAM)
- ALB: ~$20
- Data transfer: ~$10
- **Total**: ~$80-130/month

### Lambda (per month)
- First 1M requests: Free
- Next requests: $0.20 per 1M
- Compute: $0.0000166667 per GB-second
- **Total**: ~$5-20/month (low traffic)

## 🆘 Need Help?

- Check [README.md](README.md) for detailed documentation
- Review CloudWatch logs for errors
- AWS Support: https://aws.amazon.com/support

## ✅ Deployment Checklist

- [ ] Infrastructure deployed (VPC, ECS, ALB)
- [ ] Docker image built and pushed to ECR
- [ ] ECS service running and healthy
- [ ] API accessible via ALB DNS
- [ ] Health check endpoint responding
- [ ] Secrets configured in Secrets Manager
- [ ] CloudWatch logs visible
- [ ] Monitoring and alarms configured

---

**Congratulations!** 🎉 Your Robot Movement AI is now running on AWS!















