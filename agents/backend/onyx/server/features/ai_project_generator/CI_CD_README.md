# CI/CD Pipeline Documentation

Complete CI/CD pipeline implementation for AI Project Generator with GitHub Actions and automated deployment scripts.

## 🚀 Overview

The CI/CD pipeline provides:
- **Continuous Integration**: Automated testing, linting, and security scanning
- **Continuous Deployment**: Automated deployment to staging and production
- **Release Management**: Automated release creation and artifact publishing
- **Security Scanning**: Comprehensive security checks
- **Nightly Testing**: Extended test coverage

## 📋 Pipeline Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Code Push/PR                         │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼────────┐
│   CI Pipeline  │          │ Security Scan   │
│                │          │                  │
│ - Lint         │          │ - Dependencies  │
│ - Format       │          │ - Code Scan     │
│ - Unit Tests   │          │ - Secrets       │
│ - Integration  │          │ - Infrastructure│
│ - Docker Build │          └──────────────────┘
└───────┬────────┘
        │
        │ (on main/develop)
        │
┌───────▼────────┐
│   CD Pipeline   │
│                 │
│ - Build Image   │
│ - Push to ECR   │
│ - Deploy Infra │
│ - Deploy App   │
│ - Health Check │
└─────────────────┘
```

## 🔧 GitHub Actions Workflows

### 1. CI Workflow (`ci.yml`)

**Triggers:**
- Push to main, develop, or feature branches
- Pull requests

**Jobs:**
- ✅ Lint and Format Check
- ✅ Security Scanning
- ✅ Unit Tests (Python 3.10, 3.11, 3.12)
- ✅ Integration Tests
- ✅ Docker Build
- ✅ Shell Scripts Validation
- ✅ Ansible Validation
- ✅ Terraform Validation

### 2. CD Workflow (`cd.yml`)

**Triggers:**
- Push to main or production branch
- Manual workflow dispatch

**Jobs:**
- ✅ Build and Push Docker Image
- ✅ Deploy to Staging
- ✅ Deploy to Production
- ✅ Health Check
- ✅ Rollback on Failure

### 3. Security Workflow (`security.yml`)

**Triggers:**
- Push to main or develop
- Pull requests
- Weekly schedule

**Jobs:**
- ✅ Dependency Vulnerability Scan
- ✅ Code Security Scan (Trivy)
- ✅ Secret Scanning (Gitleaks)
- ✅ Infrastructure Security Scan (Checkov)

### 4. Release Workflow (`release.yml`)

**Triggers:**
- Version tags (v*.*.*)
- Manual workflow dispatch

**Jobs:**
- ✅ Create GitHub Release
- ✅ Build Release Artifacts
- ✅ Publish Docker Image

### 5. Nightly Workflow (`nightly.yml`)

**Triggers:**
- Daily schedule (2 AM UTC)
- Manual workflow dispatch

**Jobs:**
- ✅ Full Test Suite
- ✅ Performance Tests
- ✅ Load Tests

## 📝 CI/CD Scripts

### Build Script (`scripts/ci_cd/build.sh`)

Builds Docker images and Python packages.

```bash
# Build Docker image
BUILD_TYPE=docker IMAGE_TAG=v1.0.0 ./scripts/ci_cd/build.sh

# Build Python package
BUILD_TYPE=python ./scripts/ci_cd/build.sh
```

### Test Script (`scripts/ci_cd/test.sh`)

Runs automated tests.

```bash
# Run all tests
./scripts/ci_cd/test.sh

# Run specific test type
TEST_TYPE=unit ./scripts/ci_cd/test.sh
TEST_TYPE=integration ./scripts/ci_cd/test.sh
```

### Deploy Script (`scripts/ci_cd/deploy.sh`)

Deploys application to AWS.

```bash
# Deploy to staging
ENVIRONMENT=staging IMAGE_TAG=v1.0.0 ./scripts/ci_cd/deploy.sh deploy

# Rollback
ENVIRONMENT=production ./scripts/ci_cd/deploy.sh rollback
```

## 🔐 Required Secrets

Configure in GitHub repository settings:

```
AWS_ACCESS_KEY_ID          # AWS access key
AWS_SECRET_ACCESS_KEY      # AWS secret key
SLACK_WEBHOOK_URL          # Optional: Slack notifications
```

## 🚀 Quick Start

### 1. Configure Secrets

```bash
# In GitHub repository settings
# Settings > Secrets and variables > Actions
# Add required secrets
```

### 2. Enable Workflows

Workflows are automatically enabled when pushed to repository.

### 3. Test Locally

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run CI checks locally
./scripts/ci_cd/test.sh all
./scripts/ci_cd/build.sh docker
```

### 4. Create Pull Request

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git commit -m "Add new feature"

# Push and create PR
git push origin feature/my-feature
```

## 📊 Workflow Status

View workflow status:
- GitHub Actions tab in repository
- Status badges in README
- Status checks on pull requests

## 🔄 Deployment Process

### Automatic Deployment

1. **Merge to develop** → Deploys to staging
2. **Merge to main** → Deploys to production
3. **Push tag** → Creates release

### Manual Deployment

```bash
# Deploy to staging
gh workflow run cd.yml -f environment=staging

# Deploy to production
gh workflow run cd.yml -f environment=production
```

## 🛡️ Security Features

- **Dependency Scanning**: Safety checks for vulnerable packages
- **Code Scanning**: Trivy for code vulnerabilities
- **Secret Scanning**: Gitleaks for exposed secrets
- **Infrastructure Scanning**: Checkov for IaC security
- **Container Scanning**: Trivy for Docker images

## 📈 Monitoring

### CI/CD Metrics

- Build success rate
- Test coverage
- Deployment frequency
- Mean time to recovery (MTTR)

### CloudWatch Integration

All deployments send metrics to CloudWatch:
- Deployment success/failure
- Deployment duration
- Application health

## 🆘 Troubleshooting

### CI Failures

1. Check workflow logs
2. Run tests locally
3. Fix linting issues
4. Update dependencies

### CD Failures

1. Check AWS credentials
2. Verify Terraform state
3. Check Ansible logs
4. Review application health

### Common Issues

**Tests fail:**
- Run tests locally first
- Check test environment
- Review test dependencies

**Deployment fails:**
- Verify AWS permissions
- Check infrastructure status
- Review deployment logs

**Build fails:**
- Check Dockerfile syntax
- Verify dependencies
- Review build logs

## 📚 Best Practices

1. **Run Pre-commit Hooks**: Catch issues early
2. **Small Commits**: Easier to debug
3. **Clear Messages**: Help with debugging
4. **Test Locally**: Verify before pushing
5. **Review Logs**: Understand failures
6. **Fix Quickly**: Don't let CI break

## 🔗 Related Documentation

- [GitHub Actions Workflows README](.github/workflows/README.md)
- [CI/CD Scripts README](scripts/ci_cd/README.md)
- [AWS Deployment Guide](aws/README.md)
- [Main README](README.md)

