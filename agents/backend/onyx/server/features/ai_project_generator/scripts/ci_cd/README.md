# CI/CD Scripts Documentation

Scripts for continuous integration and continuous deployment.

## 📋 Available Scripts

### 1. `build.sh` - Build Script
Builds Docker images and Python packages.

**Usage:**
```bash
# Build Docker image
BUILD_TYPE=docker IMAGE_TAG=v1.0.0 ./build.sh

# Build Python package
BUILD_TYPE=python ./build.sh

# Build all
BUILD_TYPE=all ./build.sh
```

**Environment Variables:**
- `BUILD_TYPE`: docker, python, or all
- `IMAGE_NAME`: Docker image name
- `IMAGE_TAG`: Docker image tag
- `DOCKERFILE`: Dockerfile path
- `PYTHON_VERSION`: Python version
- `BUILD_CACHE`: Enable build cache

### 2. `test.sh` - Test Script
Runs automated tests.

**Usage:**
```bash
# Run all tests
./test.sh

# Run specific test type
TEST_TYPE=unit ./test.sh
TEST_TYPE=integration ./test.sh
TEST_TYPE=e2e ./test.sh
TEST_TYPE=lint ./test.sh
TEST_TYPE=security ./test.sh
```

**Environment Variables:**
- `TEST_TYPE`: unit, integration, e2e, lint, security, or all
- `COVERAGE`: Enable coverage reports
- `PARALLEL`: Run tests in parallel
- `VERBOSE`: Verbose output

### 3. `deploy.sh` - Deployment Script
Deploys application to AWS.

**Usage:**
```bash
# Deploy to staging
ENVIRONMENT=staging IMAGE_TAG=v1.0.0 ./deploy.sh deploy

# Deploy to production
ENVIRONMENT=production IMAGE_TAG=v1.0.0 AUTO_APPROVE=true ./deploy.sh deploy

# Rollback
ENVIRONMENT=production ./deploy.sh rollback

# Validate only
./deploy.sh validate
```

**Environment Variables:**
- `ENVIRONMENT`: staging, production, or development
- `IMAGE_TAG`: Docker image tag to deploy
- `SKIP_TESTS`: Skip pre-deployment tests
- `AUTO_APPROVE`: Auto-approve Terraform apply

## 🔄 CI/CD Pipeline Flow

```
1. Code Push/PR
   └─> CI Workflow (ci.yml)
       ├─> Lint & Format
       ├─> Security Scan
       ├─> Unit Tests
       ├─> Integration Tests
       └─> Docker Build

2. Merge to Main
   └─> CD Workflow (cd.yml)
       ├─> Build & Push Image
       ├─> Deploy Infrastructure
       ├─> Deploy Application
       └─> Health Check

3. Tag Release
   └─> Release Workflow (release.yml)
       ├─> Create Release
       ├─> Build Artifacts
       └─> Publish Image
```

## 🛠️ Local Development

### Run CI Locally

```bash
# Run all CI checks
./scripts/ci_cd/test.sh all

# Run specific checks
./scripts/ci_cd/test.sh lint
./scripts/ci_cd/test.sh security
```

### Build Locally

```bash
# Build Docker image
./scripts/ci_cd/build.sh docker

# Test build
docker run --rm ai-project-generator:test
```

### Deploy Locally (Dry Run)

```bash
# Validate deployment
./scripts/ci_cd/deploy.sh validate

# Deploy to staging (dry run)
ENVIRONMENT=staging AUTO_APPROVE=false ./scripts/ci_cd/deploy.sh deploy
```

## 📊 Integration with GitHub Actions

These scripts are used by GitHub Actions workflows:

- `ci.yml` uses `test.sh` and `build.sh`
- `cd.yml` uses `deploy.sh`
- `release.yml` uses `build.sh`

## 🔧 Configuration

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Local Testing

Set up local test environment:
```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest tests/ -v
```

## 📝 Best Practices

1. **Run tests before pushing**: Use pre-commit hooks
2. **Test locally first**: Verify changes work locally
3. **Small commits**: Easier to debug failures
4. **Clear commit messages**: Help with debugging
5. **Review CI logs**: Understand failures
6. **Fix failures quickly**: Don't let CI break

## 🆘 Troubleshooting

### Build Failures
- Check Dockerfile syntax
- Verify dependencies
- Check build logs

### Test Failures
- Run tests locally
- Check test environment
- Review test output

### Deployment Failures
- Verify AWS credentials
- Check Terraform state
- Review Ansible logs
- Check application health

