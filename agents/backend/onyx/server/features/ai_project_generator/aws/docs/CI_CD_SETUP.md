# CI/CD Setup Guide - Automatic Deployment to EC2

This guide explains how to set up automatic deployment to EC2 instances after pushing to the main branch.

## 📋 Overview

There are two main approaches for automatic deployment:

1. **GitHub Actions** (Recommended) - CI/CD pipeline that deploys on push
2. **Webhook/Polling** - EC2 instance listens for changes and deploys automatically

## 🚀 Option 1: GitHub Actions (Recommended)

### Prerequisites

1. GitHub repository with Actions enabled
2. AWS credentials stored as GitHub Secrets
3. SSH key for EC2 access stored as GitHub Secret

### Setup Steps

#### 1. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions, and add:

- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `EC2_SSH_PRIVATE_KEY` - Private SSH key for EC2 access

#### 2. Workflow File

The workflow file is already created at `.github/workflows/deploy-ec2.yml`.

It will:
- Trigger on push to `main` branch
- Get EC2 instance IPs from Terraform
- Run Ansible deployment
- Perform health checks

#### 3. First Deployment

```bash
# Push to main branch
git push origin main

# Check workflow status
# Go to GitHub → Actions tab
```

### Workflow Features

- ✅ Automatic trigger on push to main
- ✅ Gets instance IPs from Terraform output
- ✅ Runs Ansible deployment
- ✅ Health checks after deployment
- ✅ Manual trigger support

## 🔄 Option 2: Webhook/Polling (Alternative)

### Setup on EC2 Instance

#### 1. SSH into EC2 Instance

```bash
ssh -i your-key.pem ubuntu@<ec2-ip>
```

#### 2. Set Environment Variables

```bash
export GIT_REPO_URL="https://github.com/your-org/your-repo.git"
export GIT_BRANCH="main"
export PROJECT_NAME="ai-project-generator"
```

#### 3. Run Setup Script

```bash
cd agents/backend/onyx/server/features/ai_project_generator/aws/scripts
sudo bash setup_webhook_listener.sh
```

#### 4. Start Service

**Option A: Webhook Service (requires webhook URL)**
```bash
sudo systemctl start github-webhook-deploy
sudo systemctl status github-webhook-deploy
```

**Option B: Polling Service (checks every 5 minutes)**
```bash
sudo systemctl start github-webhook-deploy-polling
sudo systemctl status github-webhook-deploy-polling
```

### Configure GitHub Webhook (Optional)

1. Go to repository → Settings → Webhooks
2. Add webhook:
   - **Payload URL**: `http://<ec2-ip>:9000/webhook`
   - **Content type**: `application/json`
   - **Secret**: (optional, set `WEBHOOK_SECRET` env var)
   - **Events**: Just the `push` event

## 📝 Configuration

### Environment Variables

Set these on the EC2 instance or in GitHub Actions:

```bash
export GIT_REPO_URL="https://github.com/your-org/your-repo.git"
export GIT_BRANCH="main"
export PROJECT_NAME="ai-project-generator"
export WEBHOOK_SECRET="your-secret"  # Optional, for webhook verification
export WEBHOOK_PORT="9000"           # Optional, for HTTP webhook listener
```

### Ansible Variables

The deployment uses these Ansible variables:

- `project_name`: Project name (default: ai-project-generator)
- `environment`: Environment (default: production)
- `git_branch`: Git branch to deploy (default: main)
- `git_commit`: Commit SHA (auto-detected)

## 🔍 Monitoring

### GitHub Actions

- View workflow runs: GitHub → Actions tab
- View logs: Click on workflow run → View logs

### EC2 Service

```bash
# View service status
sudo systemctl status github-webhook-deploy

# View logs
sudo journalctl -u github-webhook-deploy -f

# View deployment logs
tail -f /var/log/github-deploy.log
```

## 🛠️ Troubleshooting

### GitHub Actions Issues

**Problem**: Cannot connect to EC2
- Check AWS credentials in GitHub Secrets
- Verify EC2 security group allows SSH from GitHub Actions IPs
- Check SSH key format (should be private key without passphrase)

**Problem**: Cannot find instances
- Verify Terraform outputs are correct
- Check instance tags match project name and environment
- Ensure instances are running

### Webhook/Polling Issues

**Problem**: Service not starting
```bash
# Check service status
sudo systemctl status github-webhook-deploy

# Check logs
sudo journalctl -u github-webhook-deploy -n 50
```

**Problem**: Deployment not triggering
- Check if new commits exist: `git log origin/main`
- Verify environment variables are set
- Check lock file: `/var/run/github-deploy.lock`

**Problem**: Health check failing
- Check application logs
- Verify application is running: `sudo systemctl status ai-project-generator`
- Check port 8020 is accessible

## 🔐 Security Considerations

### GitHub Actions

1. **Secrets**: Never commit secrets to repository
2. **SSH Keys**: Use dedicated deployment key
3. **IAM Roles**: Use IAM roles with least privilege
4. **Security Groups**: Restrict SSH access

### Webhook/Polling

1. **Webhook Secret**: Always use webhook secret for verification
2. **Firewall**: Restrict webhook port access
3. **SSH Keys**: Use key-based authentication
4. **Permissions**: Run services with minimal privileges

## 📊 Deployment Flow

### GitHub Actions Flow

```
Push to main
    ↓
GitHub Actions triggered
    ↓
Get EC2 instance IPs (from Terraform)
    ↓
SSH to instances
    ↓
Run Ansible deployment
    ↓
Health check
    ↓
Deployment complete
```

### Webhook/Polling Flow

```
New commit on main
    ↓
Webhook received OR Polling detects change
    ↓
Check if deployment needed
    ↓
Pull latest code
    ↓
Run Ansible deployment
    ↓
Health check
    ↓
Deployment complete
```

## 🎯 Best Practices

1. **Use GitHub Actions** for production deployments
2. **Use webhook/polling** for development/testing
3. **Test deployments** on staging before production
4. **Monitor deployments** via logs and health checks
5. **Rollback plan** - Keep previous versions available
6. **Blue-green deployments** - For zero-downtime
7. **Notifications** - Set up alerts for deployment failures

## 📚 Related Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Ansible Playbooks](../ansible/playbooks/)

## 🔗 Quick Reference

### GitHub Actions
- Workflow: `.github/workflows/deploy-ec2.yml`
- Manual trigger: GitHub → Actions → Deploy to EC2 → Run workflow

### Webhook Service
- Script: `scripts/github_webhook_deploy.sh`
- Setup: `scripts/setup_webhook_listener.sh`
- Service: `github-webhook-deploy`

### Polling Service
- Service: `github-webhook-deploy-polling`
- Interval: 5 minutes (configurable)

---

**Last Updated**: 2024  
**Version**: 1.0

