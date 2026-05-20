# Automatic Deployment Summary

## 🚀 Quick Setup

### Option 1: GitHub Actions (Recommended - 5 minutes)

1. **Add GitHub Secrets**:
   - Go to repository → Settings → Secrets → Actions
   - Add: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `EC2_SSH_PRIVATE_KEY`

2. **Push to main**:
   ```bash
   git push origin main
   ```

3. **Done!** Check deployment status in GitHub → Actions

### Option 2: Webhook/Polling (Alternative - 10 minutes)

1. **SSH to EC2**:
   ```bash
   ssh -i key.pem ubuntu@<ec2-ip>
   ```

2. **Set environment variables**:
   ```bash
   export GIT_REPO_URL="https://github.com/your-org/your-repo.git"
   export GIT_BRANCH="main"
   ```

3. **Run setup**:
   ```bash
   cd agents/backend/onyx/server/features/ai_project_generator/aws/scripts
   sudo bash setup_webhook_listener.sh
   sudo systemctl start github-webhook-deploy
   ```

## 📋 What Happens on Push

1. **GitHub Actions** (if enabled):
   - Detects push to main
   - Gets EC2 instance IPs from Terraform
   - Runs Ansible deployment
   - Performs health check

2. **Webhook/Polling** (if enabled):
   - Detects new commit (webhook or polling)
   - Pulls latest code
   - Runs Ansible deployment
   - Performs health check

## 🔍 Monitoring

### GitHub Actions
- View: GitHub → Actions tab
- Logs: Click workflow run

### EC2 Service
```bash
# Status
sudo systemctl status github-webhook-deploy

# Logs
sudo journalctl -u github-webhook-deploy -f
tail -f /var/log/github-deploy.log
```

## ⚙️ Configuration

### Required Secrets (GitHub Actions)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `EC2_SSH_PRIVATE_KEY`

### Environment Variables (Webhook/Polling)
- `GIT_REPO_URL` - Repository URL
- `GIT_BRANCH` - Branch to deploy (default: main)
- `PROJECT_NAME` - Project name (default: ai-project-generator)

## 🎯 Features

✅ Automatic deployment on push to main  
✅ Health checks after deployment  
✅ Rollback support  
✅ Deployment logs  
✅ Manual trigger support  
✅ Multiple deployment methods  

## 📚 Full Documentation

See [CI_CD_SETUP.md](CI_CD_SETUP.md) for complete setup guide.

---

**Status**: ✅ Ready to use  
**Setup Time**: 5-10 minutes  
**Deployment Time**: ~5 minutes per deployment

