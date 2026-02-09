# Enhanced CI/CD Features

## 🚀 New Features

### 1. Pre-Deployment Checks
- ✅ Change detection (only deploys if relevant files changed)
- ✅ Commit message validation (skip deployment with `[skip ci]`)
- ✅ Environment validation

### 2. Testing Pipeline
- ✅ Python linting (flake8, black)
- ✅ Type checking (mypy)
- ✅ Unit tests with coverage
- ✅ Code coverage reporting (Codecov)
- ✅ Security scanning (Trivy)

### 3. Enhanced Deployment
- ✅ Automatic backups before deployment
- ✅ Retry mechanism (3 attempts)
- ✅ Health checks with multiple attempts
- ✅ Rollback on failure
- ✅ Blue-green deployment support

### 4. Notifications
- ✅ Success/failure notifications
- ✅ Deployment status tracking
- ✅ Commit information in notifications

### 5. Artifact Management
- ✅ Deployment artifacts
- ✅ Commit tracking
- ✅ Artifact retention (7 days)

## 📋 Workflow Jobs

### 1. Pre-Deployment
- Checks if deployment is needed
- Validates commit messages
- Determines if relevant files changed

### 2. Test
- Runs linting
- Type checking
- Unit tests
- Coverage reporting

### 3. Security Scan
- Vulnerability scanning with Trivy
- SARIF upload to GitHub Security

### 4. Build
- Builds Docker images (if applicable)
- Creates deployment artifacts
- Uploads artifacts

### 5. Deploy
- Gets EC2 instance IPs
- Creates backups
- Runs Ansible deployment
- Health checks
- Automatic rollback on failure

### 6. Notify
- Sends success/failure notifications
- Includes commit information

## 🔧 Usage

### Automatic Deployment

Just push to main:
```bash
git push origin main
```

### Manual Deployment

Go to GitHub → Actions → Deploy to EC2 (Enhanced) → Run workflow

Options:
- **Environment**: staging or production
- **Skip tests**: Skip test phase
- **Force deploy**: Deploy even if tests fail

### Skip Deployment

Add to commit message:
```
[skip ci]
```
or
```
ci skip
```

## 🛡️ Rollback

### Automatic Rollback

If health check fails after deployment, automatic rollback is triggered.

### Manual Rollback

On EC2 instance:
```bash
sudo /opt/ai-project-generator/scripts/rollback.sh
```

List available backups:
```bash
sudo /opt/ai-project-generator/scripts/rollback.sh --list
```

Restore specific backup:
```bash
sudo /opt/ai-project-generator/scripts/rollback.sh /opt/backups/backup-1234567890.tar.gz
```

## 🔵 Blue-Green Deployment

### Deploy

```bash
sudo /opt/ai-project-generator/scripts/blue_green_deploy.sh deploy
```

### Rollback

```bash
sudo /opt/ai-project-generator/scripts/blue_green_deploy.sh rollback
```

### Status

```bash
sudo /opt/ai-project-generator/scripts/blue_green_deploy.sh status
```

## 📊 Monitoring

### GitHub Actions

View workflow runs:
- GitHub → Actions tab
- Click on workflow run
- View logs for each job

### Deployment Logs

On EC2 instance:
```bash
# Service logs
sudo journalctl -u ai-project-generator -f

# Deployment logs
tail -f /var/log/github-deploy.log

# Ansible logs
tail -f /var/log/ansible.log
```

## 🔐 Security Features

1. **Secrets Management**: All secrets stored in GitHub Secrets
2. **SSH Key Rotation**: Support for key rotation
3. **Security Scanning**: Trivy vulnerability scanning
4. **Access Control**: Environment-based access control
5. **Audit Logging**: All deployments logged

## 🎯 Best Practices

1. **Always test locally** before pushing
2. **Use staging environment** for testing
3. **Monitor deployments** via GitHub Actions
4. **Review security scans** regularly
5. **Keep backups** for quick rollback
6. **Use blue-green** for zero-downtime deployments
7. **Set up notifications** for deployment status

## 📚 Related Documentation

- [CI/CD Setup Guide](CI_CD_SETUP.md)
- [Automatic Deployment Summary](AUTOMATIC_DEPLOYMENT_SUMMARY.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

## 🔗 Quick Reference

### Workflow Files
- Enhanced: `.github/workflows/deploy-ec2-enhanced.yml`
- Basic: `.github/workflows/deploy-ec2.yml`

### Scripts
- Rollback: `aws/scripts/rollback.sh`
- Blue-Green: `aws/scripts/blue_green_deploy.sh`
- Webhook: `aws/scripts/github_webhook_deploy.sh`

### Playbooks
- Update: `aws/ansible/playbooks/update.yml`
- Webhook: `aws/ansible/playbooks/webhook_deploy.yml`

---

**Version**: 2.0  
**Status**: Production Ready ✅  
**Last Updated**: 2024

