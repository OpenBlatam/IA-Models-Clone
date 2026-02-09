# Quick Start - EC2 Deployment

## 🚀 Automated Deployment (5 minutes)

### Step 1: Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions → New repository secret

Add these secrets:
- `EC2_HOST`: Your EC2 instance IP or domain (e.g., `18.206.225.74`)
- `EC2_USER`: SSH user (usually `ubuntu` or `ec2-user`)
- `EC2_SSH_KEY`: Your private SSH key content (paste the entire key including `-----BEGIN RSA PRIVATE KEY-----`)

### Step 2: Push to Main

```bash
git add .
git commit -m "Setup CI/CD for Suno Clone AI"
git push origin main
```

### Step 3: Monitor Deployment

1. Go to GitHub → Actions tab
2. Watch the "Deploy Suno Clone AI to EC2" workflow
3. Wait for completion (usually 5-10 minutes)

### Step 4: Verify

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-host

# Check if container is running
docker ps | grep suno-clone-ai

# Test health endpoint
curl http://localhost:8020/health
```

## 🛠️ Manual Deployment (Alternative)

### Option 1: Using Deployment Script

```bash
# 1. Build Docker image
cd agents/backend/onyx/server/features/suno_clone_ai
docker build -t suno-clone-ai:latest .

# 2. Save image
docker save suno-clone-ai:latest | gzip > suno-clone-ai.tar.gz

# 3. Transfer to EC2
scp -i ~/.ssh/your-key.pem suno-clone-ai.tar.gz deploy/deploy.sh ubuntu@your-ec2-host:~/suno-clone-ai/

# 4. Deploy
ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-host
cd ~/suno-clone-ai
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Using Ansible

```bash
cd agents/backend/onyx/server/features/suno_clone_ai/deploy/ansible

# Edit inventory.yml with your EC2 details
# Then run:
ansible-playbook playbook.yml
```

## ⚙️ Configuration

### Create .env file on EC2

```bash
# On EC2 instance
cd ~/suno-clone-ai
nano .env
```

Minimum required variables:
```env
API_HOST=0.0.0.0
API_PORT=8020
DEBUG=False
SECRET_KEY=your-secret-key-here
```

## 🔍 Troubleshooting

### Container won't start
```bash
docker logs suno-clone-ai --tail 50
```

### Port already in use
```bash
docker stop suno-clone-ai
docker rm suno-clone-ai
```

### Health check fails
```bash
curl http://localhost:8020/health
docker exec -it suno-clone-ai python -c "from main import app; print('OK')"
```

## 📚 Next Steps

- Read [DEPLOYMENT.md](../DEPLOYMENT.md) for detailed documentation
- Configure monitoring and alerts
- Set up SSL/TLS with reverse proxy
- Enable backups

## 🆘 Need Help?

1. Check [DEPLOYMENT.md](../DEPLOYMENT.md) troubleshooting section
2. Review GitHub Actions logs
3. Check container logs: `docker logs suno-clone-ai`




