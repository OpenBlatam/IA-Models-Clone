# 🚀 CI/CD Complete System - Music Analyzer AI

## 📋 Overview

Enterprise-grade CI/CD system for Music Analyzer AI with multi-cloud support, Kubernetes orchestration, and comprehensive testing.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Cloud CI/CD Pipeline                       │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Backend    │  │   Frontend   │  │   Security   │      │
│  │   Quality    │  │   Quality    │  │    Scan      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│                  ┌────────▼────────┐                       │
│                  │  Docker Build   │                       │
│                  └────────┬─────────┘                       │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         │                 │                 │              │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│  │ Kubernetes  │  │     AWS     │  │   Azure     │        │
│  │  Deployment │  │  Deployment  │  │ Deployment │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Workflow Files

### GitHub Actions

1. **`music-analyzer-ai-ci.yml`** - Complete CI/CD pipeline
   - Backend code quality & tests
   - Frontend code quality & tests
   - Security scanning
   - Docker builds
   - Multi-cloud deployment

### Azure Pipelines

2. **`azure-pipelines.yml`** - Azure DevOps pipeline
   - Multi-stage pipeline
   - Quality gates
   - Azure-native deployment

### Kubernetes

3. **Helm Chart** - Kubernetes deployment
   - Backend & Frontend services
   - PostgreSQL & Redis
   - Ingress configuration
   - Auto-scaling

### Terraform

4. **`terraform/main.tf`** - Infrastructure as Code
   - Multi-cloud support
   - AWS resources
   - Azure resources
   - Kubernetes resources

## 🚀 Quick Start

### 1. GitHub Actions (Recommended)

```bash
# Automatic on push
git push origin main

# Manual trigger
gh workflow run music-analyzer-ai-ci.yml \
  -f environment=production \
  -f deploy_backend=true \
  -f deploy_frontend=true
```

### 2. Azure Pipelines

```bash
# Configure Azure DevOps
az pipelines create \
  --name "Music Analyzer AI" \
  --repository <repo-url> \
  --branch main \
  --yaml-path agents/backend/onyx/server/features/music_analyzer_ai/deployment/azure-pipelines/azure-pipelines.yml
```

### 3. Kubernetes Deployment

```bash
# Using Helm
helm install music-analyzer-ai \
  ./deployment/kubernetes/helm-chart \
  --namespace production \
  --create-namespace \
  --set backend.image.tag=latest \
  --set frontend.image.tag=latest
```

### 4. Terraform Deployment

```bash
# Initialize
cd deployment/terraform
terraform init

# Plan
terraform plan -var="environment=production"

# Apply
terraform apply -var="environment=production"
```

## 🔐 Required Secrets

### GitHub Secrets

**AWS:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_EC2_HOST`
- `AWS_EC2_USER`
- `AWS_EC2_SSH_KEY`

**Azure:**
- `AZURE_CREDENTIALS`

**Kubernetes:**
- `KUBECONFIG`

**General:**
- `SLACK_WEBHOOK_URL`

### Azure DevOps Variables

Create variable group `music-analyzer-ai-variables`:
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `DATABASE_URL`
- `REDIS_URL`

## 📊 Features

### ✅ Backend (FastAPI)

- Code quality (Black, Flake8, Pylint, Mypy)
- Security scanning (Safety, Bandit, Semgrep)
- Unit tests (pytest with coverage)
- Docker build (multi-arch)
- Matrix testing (Python 3.10, 3.11, 3.12)

### ✅ Frontend (Next.js)

- Code quality (ESLint, TypeScript)
- Unit tests (Jest)
- Build verification
- Docker build (multi-arch)

### ✅ Deployment

- **Kubernetes**: Helm charts with auto-scaling
- **AWS**: ECS/EC2 deployment
- **Azure**: Container Instances/App Service
- **Multi-cloud**: Terraform IaC

### ✅ Monitoring

- Health checks
- Auto-scaling
- Resource limits
- Service monitoring

## 🛠️ Local Development

### Backend

```bash
cd agents/backend/onyx/server/features/music_analyzer_ai

# Run tests
pytest tests/ -v --cov

# Code quality
black .
flake8 .
mypy .

# Security
safety check
bandit -r .
```

### Frontend

```bash
cd agents/backend/onyx/server/features/music_analyzer_ai/frontend

# Run tests
npm test

# Code quality
npm run lint
npm run type-check

# Build
npm run build
```

## 📈 Monitoring

### Kubernetes

```bash
# Check pods
kubectl get pods -n production

# Check services
kubectl get svc -n production

# View logs
kubectl logs -f deployment/music-analyzer-ai-backend -n production
```

### AWS

```bash
# ECS service status
aws ecs describe-services \
  --cluster music-analyzer-ai-cluster \
  --services music-analyzer-ai-backend

# CloudWatch logs
aws logs tail /ecs/music-analyzer-ai --follow
```

### Azure

```bash
# Container status
az container show \
  --resource-group music-analyzer-ai-rg \
  --name music-analyzer-ai-backend

# Application Insights
az monitor app-insights query \
  --app music-analyzer-ai \
  --analytics-query "requests | summarize count() by bin(timestamp, 1h)"
```

## 🔄 Workflow Status

### Viewing Results

1. **GitHub Actions**
   - Repository → Actions
   - Select workflow run
   - View job logs and artifacts

2. **Azure DevOps**
   - Pipelines → Runs
   - View pipeline execution
   - Download artifacts

3. **Kubernetes**
   - `kubectl get all -n production`
   - Helm status: `helm status music-analyzer-ai`

## 🆘 Troubleshooting

### Common Issues

#### Tests Failing
```bash
# Run locally
pytest tests/ -v

# Check specific test
pytest tests/test_api/test_music_api.py::test_search_track -v
```

#### Docker Build Failing
```bash
# Build locally
docker build -t music-analyzer-ai-backend:test \
  -f deployment/Dockerfile .

# Check logs
docker build --progress=plain -t music-analyzer-ai-backend:test .
```

#### Kubernetes Deployment Issues
```bash
# Check pod status
kubectl describe pod <pod-name> -n production

# View events
kubectl get events -n production --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n production --previous
```

## 📚 Documentation

- **[README.md](../README.md)** - Project README
- **[DEPLOYMENT.md](README.md)** - Deployment guide
- **[ARCHITECTURE.md](../ARCHITECTURE_SUMMARY.md)** - Architecture overview

## ✅ Best Practices

1. **Branch Strategy**
   - `main` - Production
   - `develop` - Staging
   - `feature/*` - Features

2. **Commit Messages**
   - Use conventional commits
   - Link to issues
   - Clear descriptions

3. **Pull Requests**
   - Require CI checks
   - Code review required
   - Link to issues

4. **Deployment**
   - Test in staging first
   - Monitor after deployment
   - Have rollback plan

## 🎯 Features Summary

✅ **Multi-Cloud Support**
- AWS (ECS/EC2)
- Azure (Container Instances/App Service)
- Kubernetes (Any cloud)

✅ **Complete Testing**
- Unit tests
- Integration tests
- Security scans
- Performance tests

✅ **Infrastructure as Code**
- Terraform
- Helm charts
- ARM templates

✅ **Monitoring & Observability**
- Health checks
- Auto-scaling
- Logging
- Metrics

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: ✅ Production Ready




