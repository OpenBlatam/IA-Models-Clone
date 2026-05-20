# AWS Deployment - Manuales Hogar AI

## Overview

This document provides a complete guide for deploying the Manuales Hogar AI service to AWS using a modern, scalable, serverless-ready architecture.

## Architecture

The deployment uses the following AWS services:

- **ECS Fargate**: Containerized application hosting (serverless containers)
- **RDS PostgreSQL**: Managed database service
- **Application Load Balancer**: Traffic distribution and health checks
- **CloudWatch**: Logging, monitoring, and alarms
- **Secrets Manager**: Secure credential storage
- **ECR**: Container image registry
- **VPC**: Isolated network environment
- **Auto-scaling**: Automatic scaling based on CPU and memory

## Quick Start

See [aws/QUICKSTART.md](aws/QUICKSTART.md) for a step-by-step deployment guide.

## Directory Structure

```
aws/
├── infrastructure/          # AWS CDK infrastructure code
│   ├── app.py              # Main CDK stack definition
│   ├── cdk.json            # CDK configuration
│   └── requirements.txt    # Python dependencies for CDK
├── scripts/                 # Deployment and utility scripts
│   ├── deploy.sh           # Main deployment script (Linux/Mac)
│   ├── deploy.ps1           # Main deployment script (Windows)
│   ├── init_db.sh          # Database initialization
│   ├── update-secrets.sh   # Update secrets in Secrets Manager
│   ├── health-check.sh     # Health check script
│   ├── view-logs.sh        # View CloudWatch logs
│   └── scale-service.sh    # Manual scaling script
├── terraform/              # Alternative Terraform config (optional)
├── README.md               # Detailed AWS deployment documentation
├── QUICKSTART.md           # Quick start guide
└── ENVIRONMENT.md          # Environment variables guide
```

## Key Features

### 1. Serverless-Ready
- Uses ECS Fargate (no EC2 instances to manage)
- Auto-scaling from 2 to 10 tasks
- Pay only for what you use

### 2. High Availability
- Multi-AZ deployment (can be enabled)
- Application Load Balancer with health checks
- Automatic task replacement on failure

### 3. Security
- Secrets stored in AWS Secrets Manager
- Database in private subnet
- Security groups with least privilege
- Encryption at rest for RDS

### 4. Monitoring
- CloudWatch logs for all containers
- CloudWatch alarms for CPU, memory, and database
- SNS notifications for alerts

### 5. CI/CD Ready
- GitHub Actions workflow included
- Automated deployments on push
- Environment-specific configurations

## Deployment Options

### Option 1: Automated Script (Recommended)

**Linux/Mac:**
```bash
cd aws/scripts
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```powershell
cd aws/scripts
.\deploy.ps1 -AwsAccountId "your-account-id" -AwsRegion "us-east-1"
```

### Option 2: GitHub Actions

Configure secrets in GitHub and push to main/develop branch. See `.github/workflows/deploy-aws.yml`.

### Option 3: Manual Deployment

Follow the step-by-step guide in [aws/QUICKSTART.md](aws/QUICKSTART.md).

## Configuration

### Environment Variables

See [aws/ENVIRONMENT.md](aws/ENVIRONMENT.md) for complete environment variable documentation.

Key variables:
- `OPENROUTER_API_KEY`: Stored in Secrets Manager
- `DB_HOST`, `DB_USER`, `DB_PASSWORD`: Automatically configured
- `ENVIRONMENT`: dev/staging/prod

### Scaling Configuration

Default auto-scaling:
- **Min tasks**: 2
- **Max tasks**: 10
- **CPU threshold**: 70%
- **Memory threshold**: 80%

Modify in `aws/infrastructure/app.py`.

### Database Configuration

- **Engine**: PostgreSQL 15.4
- **Instance**: t3.micro (dev) / t3.small+ (prod)
- **Storage**: 20GB (auto-scales to 100GB)
- **Backups**: 7 days retention

## Monitoring and Logging

### View Logs

```bash
cd aws/scripts
./view-logs.sh
```

Or via AWS CLI:
```bash
aws logs tail /ecs/manuales-hogar-ai --follow --region us-east-1
```

### Health Checks

```bash
cd aws/scripts
./health-check.sh
```

### CloudWatch Dashboards

Create custom dashboards in AWS Console:
- ECS service metrics
- ALB metrics
- RDS metrics
- Application logs

## Maintenance

### Update Application

1. Build new Docker image
2. Push to ECR
3. Update ECS service (automatic with deployment scripts)

### Update Secrets

```bash
cd aws/scripts
export OPENROUTER_API_KEY="new-key"
./update-secrets.sh
```

### Scale Service

```bash
cd aws/scripts
./scale-service.sh 5  # Scale to 5 tasks
```

### Database Migrations

```bash
cd aws/scripts
./init_db.sh
```

## Cost Optimization

### Development Environment
- Use t3.micro for RDS
- Single NAT gateway
- Reduced backup retention
- Estimated: ~$70/month

### Production Environment
- Use t3.small+ for RDS
- Multi-AZ for high availability
- Extended backup retention
- CloudWatch log retention policies
- Estimated: ~$150-200/month

### Cost Reduction Tips
1. Use Fargate Spot for non-critical workloads (up to 70% savings)
2. Schedule scaling down during off-hours
3. Use S3 for log archival instead of CloudWatch
4. Enable RDS automated backups only for production

## Security Best Practices

1. ✅ Secrets in Secrets Manager (not environment variables)
2. ✅ Database in private subnet
3. ✅ Security groups with least privilege
4. ✅ Encryption at rest for RDS
5. ✅ IAM roles with minimal permissions
6. ⚠️ Add HTTPS/SSL certificate (recommended)
7. ⚠️ Enable WAF on ALB (recommended for production)
8. ⚠️ Enable VPC Flow Logs (recommended)

## Troubleshooting

### Service Won't Start
1. Check CloudWatch logs: `/ecs/manuales-hogar-ai`
2. Verify secrets in Secrets Manager
3. Check security group rules
4. Verify database is accessible

### High Latency
1. Check ALB metrics
2. Review ECS task CPU/memory
3. Consider increasing task resources
4. Check database performance

### Database Connection Errors
1. Verify security groups (ECS → RDS on port 5432)
2. Check database credentials in Secrets Manager
3. Verify database is in same VPC
4. Check RDS status in console

## Next Steps

1. **Add HTTPS**: Configure SSL certificate for ALB
2. **Custom Domain**: Point your domain to the load balancer
3. **WAF**: Add Web Application Firewall
4. **Backup Strategy**: Configure automated backups
5. **Disaster Recovery**: Set up cross-region replication
6. **Cost Monitoring**: Set up billing alerts

## Support

For issues or questions:
- Check [aws/README.md](aws/README.md) for detailed documentation
- Review CloudWatch logs
- Check AWS service health dashboard
- Contact Blatam Academy team

## Related Documentation

- [README.md](README.md) - Main project documentation
- [aws/README.md](aws/README.md) - Detailed AWS deployment guide
- [aws/QUICKSTART.md](aws/QUICKSTART.md) - Quick start guide
- [aws/ENVIRONMENT.md](aws/ENVIRONMENT.md) - Environment variables
- [MIGRATIONS.md](MIGRATIONS.md) - Database migrations




