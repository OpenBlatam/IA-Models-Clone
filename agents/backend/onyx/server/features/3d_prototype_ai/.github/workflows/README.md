# GitHub Actions CI/CD Workflows

This directory contains GitHub Actions workflows for automated CI/CD of the 3D Prototype AI application.

## 📋 Workflows

### `deploy-to-ec2.yml` - Main Deployment Workflow

**Trigger**: Push to `main` or `master` branch

**Features**:
- ✅ **Smart Change Detection**: Only deploys if relevant files changed
- ✅ **Multi-stage Pipeline**: Validate → Test → Security → Build → Deploy
- ✅ **Automatic Backup**: Creates backup before deployment
- ✅ **Health Checks**: Verifies deployment with retries
- ✅ **Automatic Rollback**: Rolls back on failure
- ✅ **Post-deployment Tasks**: Updates tags, cleans up backups
- ✅ **Notifications**: Slack integration (optional)

**Stages**:
1. **Validate**: Code validation, linting, Terraform/Ansible checks
2. **Test**: Unit tests with coverage
3. **Security Scan**: Trivy and Bandit security scanning
4. **Build Docker**: Builds and tests Docker images
5. **Get Instance Info**: Finds EC2 instance
6. **Pre-deploy Check**: Verifies instance status
7. **Deploy**: Deploys application with backup
8. **Verify**: Health checks and verification
9. **Post-deploy**: Updates tags, cleanup
10. **Notify**: Sends notifications

### `ci.yml` - Continuous Integration

**Trigger**: Pull requests and feature branches

**Features**:
- ✅ Code linting (Black, Flake8, mypy)
- ✅ Unit tests with coverage
- ✅ Security scanning
- ✅ Infrastructure validation

## 🔧 Setup

### 1. Configure GitHub Secrets

Go to: **Repository → Settings → Secrets and variables → Actions**

#### Required Secrets

```bash
AWS_ACCESS_KEY_ID          # AWS access key ID
AWS_SECRET_ACCESS_KEY      # AWS secret access key
AWS_SSH_PRIVATE_KEY        # SSH private key for EC2 (entire key including -----BEGIN/END-----)
```

#### Optional Secrets

```bash
SLACK_WEBHOOK_URL          # Slack webhook for notifications
DOCKER_USERNAME            # Docker Hub username (if pushing images)
DOCKER_PASSWORD            # Docker Hub password
```

### 2. Configure SSH Key

#### Generate SSH Key (if needed)

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/github_deploy_key -N ""
```

#### Add Public Key to EC2

```bash
# Option 1: Using ssh-copy-id
ssh-copy-id -i ~/.ssh/github_deploy_key.pub ubuntu@<instance-ip>

# Option 2: Manually
cat ~/.ssh/github_deploy_key.pub | ssh ubuntu@<instance-ip> \
  "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

#### Add Private Key to GitHub Secrets

1. Copy the private key:
   ```bash
   cat ~/.ssh/github_deploy_key
   ```

2. Add to GitHub Secrets as `AWS_SSH_PRIVATE_KEY` (include the entire key with headers)

### 3. Configure AWS IAM Permissions

The AWS user needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeTags",
        "ec2:CreateTags"
      ],
      "Resource": "*"
    }
  ]
}
```

### 4. Tag EC2 Instance

Ensure your EC2 instance has these tags:

```bash
Project=3D-Prototype-AI
Environment=prod  # or staging
```

## 🚀 Usage

### Automatic Deployment

Simply push to `main`:

```bash
git add .
git commit -m "Update application"
git push origin main
```

The workflow will:
1. Validate code
2. Run tests
3. Scan for security issues
4. Build Docker images
5. Deploy to EC2
6. Verify deployment
7. Send notifications

### Manual Deployment

1. Go to **Actions** tab in GitHub
2. Select **"Deploy to EC2 on Push to Main"**
3. Click **"Run workflow"**
4. Select options:
   - Environment: `production` or `staging`
   - Skip tests: `false` (recommended)
   - Force deploy: `false` (only if needed)
5. Click **"Run workflow"**

### Using GitHub CLI

```bash
gh workflow run deploy-to-ec2.yml \
  -f environment=production \
  -f skip_tests=false \
  -f force_deploy=false
```

## 📊 Workflow Features

### Smart Change Detection

The workflow only deploys if relevant files changed:
- Files in `agents/backend/onyx/server/features/3d_prototype_ai/`
- Workflow files

### Automatic Backup

Before each deployment:
- Creates timestamped backup
- Stores in `/tmp/app_backup_*.tar.gz`
- Keeps last 5 backups

### Health Checks

After deployment:
- Retries up to 10 times
- Checks `/health` endpoint
- Verifies application is responding

### Automatic Rollback

On failure:
- Automatically restores from backup
- Restarts services
- Verifies rollback success

### Post-Deployment

After successful deployment:
- Updates EC2 tags with deployment info
- Cleans up old backups
- Sends notifications

## 🔍 Monitoring

### View Workflow Runs

1. Go to **Actions** tab
2. Select workflow
3. Click on a run to see details

### View Logs

- Each step has expandable logs
- Failed steps show error details
- Artifacts available for download

### Deployment History

Check EC2 instance tags:
```bash
aws ec2 describe-instances \
  --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].Tags'
```

Tags include:
- `LastDeployment`: Timestamp
- `LastDeploymentCommit`: Git commit SHA
- `LastDeploymentBy`: GitHub user
- `LastDeploymentBranch`: Branch name

## 🐛 Troubleshooting

### Workflow Fails at "Get Instance Info"

**Problem**: Cannot find EC2 instance

**Solutions**:
1. Verify instance tags:
   ```bash
   aws ec2 describe-instances --instance-ids <id> --query 'Reservations[0].Instances[0].Tags'
   ```
2. Check AWS credentials have EC2 permissions
3. Verify instance is running
4. Check environment tag matches

### SSH Connection Fails

**Problem**: Cannot connect to EC2

**Solutions**:
1. Verify SSH key in GitHub Secrets (include full key with headers)
2. Test SSH manually:
   ```bash
   ssh -i ~/.ssh/key.pem ubuntu@<instance-ip>
   ```
3. Check security group allows SSH
4. Verify instance is running

### Deployment Fails

**Problem**: Application deployment fails

**Solutions**:
1. Check deployment logs in GitHub Actions
2. SSH to instance and check logs:
   ```bash
   ssh ubuntu@<instance-ip>
   docker-compose logs  # if using Docker
   sudo journalctl -u 3d-prototype-ai  # if using systemd
   ```
3. Check disk space:
   ```bash
   df -h
   ```
4. Verify application code is correct

### Health Check Fails

**Problem**: Health check fails after deployment

**Solutions**:
1. Check application logs
2. Verify application is listening on port 8030
3. Check security group allows traffic
4. Review application configuration
5. Check if rollback was successful

## 🔒 Security Best Practices

1. **Secrets Management**: All secrets in GitHub Secrets
2. **SSH Keys**: Use dedicated deploy key, rotate regularly
3. **IAM Permissions**: Least privilege principle
4. **Backup Before Deploy**: Automatic backups
5. **Rollback Capability**: Automatic rollback on failure
6. **Health Checks**: Mandatory verification
7. **Audit Trail**: All deployments tagged and logged

## 📈 Performance Optimizations

- **Caching**: Pip and Docker layer caching
- **Parallel Jobs**: Tests and security scans run in parallel
- **Smart Triggers**: Only runs on relevant file changes
- **Timeout Limits**: Prevents hanging workflows
- **Artifact Retention**: Configurable retention periods

## 🎯 Best Practices

1. **Always test locally** before pushing to main
2. **Review PRs** before merging to main
3. **Monitor deployments** in Actions tab
4. **Check health** after deployment
5. **Review logs** if deployment fails
6. **Keep backups** for quick rollback
7. **Update documentation** with changes

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [SSH Key Management](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

**Last Updated**: 2024-01-XX
**Workflow Version**: 2.0.0
