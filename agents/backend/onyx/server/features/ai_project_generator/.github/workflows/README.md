# GitHub Actions CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and continuous deployment.

## 📋 Available Workflows

### 1. `ci.yml` - Continuous Integration
Runs on every push and pull request.

**Jobs:**
- Lint and Format Check (Black, Ruff, MyPy, Pylint)
- Security Scanning (Bandit, Safety)
- Unit Tests (Python 3.10, 3.11, 3.12)
- Integration Tests (with Redis service)
- Docker Build and Test
- Shell Scripts Validation (ShellCheck)
- Ansible Validation (ansible-lint)
- Terraform Validation

**Triggers:**
- Push to main, develop, or feature branches
- Pull requests to main or develop

### 2. `cd.yml` - Continuous Deployment
Deploys to staging and production environments.

**Jobs:**
- Build and Push Docker Image to ECR
- Deploy to Staging (on develop branch)
- Deploy to Production (on main branch)
- Deployment Notifications

**Triggers:**
- Push to main or production branch
- Manual workflow dispatch

**Environments:**
- `staging` - Staging environment
- `production` - Production environment (requires approval)

### 3. `security.yml` - Security Scanning
Comprehensive security scanning.

**Jobs:**
- Dependency Vulnerability Scan (Safety)
- Code Security Scan (Trivy)
- Secret Scanning (Gitleaks)
- Infrastructure Security Scan (Checkov)

**Triggers:**
- Push to main or develop
- Pull requests
- Weekly schedule (Sundays)

### 4. `release.yml` - Release Management
Creates releases and publishes artifacts.

**Jobs:**
- Create GitHub Release
- Build Release Artifacts (Python package)
- Publish Docker Image

**Triggers:**
- Push of version tags (v*.*.*)
- Manual workflow dispatch

### 5. `nightly.yml` - Nightly Build and Test
Comprehensive nightly testing.

**Jobs:**
- Full Test Suite (all Python versions)
- Performance Tests
- Load Tests

**Triggers:**
- Daily schedule (2 AM UTC)
- Manual workflow dispatch

## 🔧 Configuration

### Required Secrets

Configure these secrets in GitHub repository settings:

```
AWS_ACCESS_KEY_ID          # AWS access key
AWS_SECRET_ACCESS_KEY      # AWS secret key
SLACK_WEBHOOK_URL          # Slack webhook for notifications (optional)
```

### Environment Variables

Set in workflow files or repository secrets:
- `AWS_REGION`: AWS region (default: us-east-1)
- `ECR_REPOSITORY`: ECR repository name
- `PYTHON_VERSION`: Python version (default: 3.11)

## 🚀 Usage

### Manual Deployment

```bash
# Deploy to staging
gh workflow run cd.yml -f environment=staging

# Deploy to production
gh workflow run cd.yml -f environment=production
```

### Create Release

```bash
# Create and push tag
git tag v1.0.0
git push origin v1.0.0

# Or use workflow dispatch
gh workflow run release.yml -f version=v1.0.0
```

### Run Tests Locally

```bash
# Run CI tests
./scripts/ci_cd/test.sh

# Run specific test type
TEST_TYPE=unit ./scripts/ci_cd/test.sh
TEST_TYPE=integration ./scripts/ci_cd/test.sh
```

## 📊 Workflow Status

View workflow status:
- GitHub Actions tab in repository
- Workflow badges in README
- Status checks on pull requests

## 🔔 Notifications

Configure notifications:
- GitHub email notifications
- Slack webhook (if configured)
- SNS alerts (via deployment scripts)

## 🛠️ Troubleshooting

### Workflow Failures

1. Check workflow logs in GitHub Actions
2. Verify secrets are configured
3. Check AWS credentials
4. Review test failures
5. Check infrastructure status

### Common Issues

**Docker build fails:**
- Check Dockerfile syntax
- Verify dependencies in requirements.txt
- Check build cache

**Deployment fails:**
- Verify AWS credentials
- Check Terraform state
- Verify Ansible inventory
- Check application health

**Tests fail:**
- Review test output
- Check test dependencies
- Verify test environment

## 📝 Best Practices

1. **Branch Protection**: Enable branch protection on main
2. **Required Checks**: Require CI to pass before merge
3. **Approvals**: Require approval for production deployments
4. **Secrets**: Rotate secrets regularly
5. **Monitoring**: Monitor workflow success rates
6. **Documentation**: Keep workflows documented

## 🔄 Workflow Dependencies

```
ci.yml (always runs)
  └─> cd.yml (on main/develop)
      └─> security.yml (on schedule/PR)
          └─> release.yml (on tags)
              └─> nightly.yml (on schedule)
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS ECR Documentation](https://docs.aws.amazon.com/ecr/)
- [Terraform GitHub Actions](https://github.com/hashicorp/setup-terraform)
- [Ansible Best Practices](https://docs.ansible.com/ansible/latest/user_guide/playbooks_best_practices.html)

