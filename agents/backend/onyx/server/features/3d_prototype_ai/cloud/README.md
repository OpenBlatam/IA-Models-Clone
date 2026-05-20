# ☁️ AWS EC2 Deployment Guide

This folder contains all the necessary infrastructure and automation scripts to quickly deploy the 3D Prototype AI system on AWS EC2.

**✨ Enhanced with DevOps Best Practices**: This deployment package has been improved with robust error handling, validation, CI/CD pipelines, backup/rollback mechanisms, and comprehensive monitoring. See [IMPROVEMENTS.md](./IMPROVEMENTS.md) for details.

## 🚀 Quick Start

### Initial Setup

First, run the setup script to configure your environment:

```bash
cd cloud
./scripts/setup.sh
```

Or using Makefile:

```bash
make setup
```

This will:
- Make all scripts executable
- Create necessary directories
- Set up configuration files
- Check dependencies

### Prerequisites

- **Bash** 4.0+ (usually pre-installed)
- **AWS CLI** configured with appropriate credentials (for AWS deployment)
- **Terraform** >= 1.5.0 (optional, for IaC)
- **Ansible** >= 2.14.0 (optional, for configuration management)
- **SSH key pair** in AWS

### Verify Environment

Before deploying, verify your environment:

```bash
./scripts/check_environment.sh
```

Or:

```bash
make check-env
```

### Option 1: One-Click Deployment (Recommended)

```bash
cd cloud

# Validate configuration first (recommended)
./scripts/validate.sh

# Deploy
./scripts/deploy.sh --method terraform --region us-east-1
```

This script will:
1. Validate prerequisites and configuration
2. Launch an EC2 instance with all dependencies
3. Configure the server automatically
4. Deploy the application
5. Set up monitoring and logging
6. Provide you with the endpoint URL

### Option 2: Manual Step-by-Step

1. **Launch EC2 Instance**:
   ```bash
   ./scripts/launch_ec2.sh
   ```

2. **Configure Server**:
   ```bash
   ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml
   ```

3. **Deploy Application**:
   ```bash
   ./scripts/deploy_app.sh
   ```

## 📁 Folder Structure

```
cloud/
├── terraform/              # Infrastructure as Code (Terraform)
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── ec2.tf
├── ansible/                # Configuration Management
│   ├── playbooks/
│   ├── roles/
│   └── inventory/
├── scripts/                # Deployment Automation
│   ├── deploy.sh
│   ├── launch_ec2.sh
│   ├── deploy_app.sh
│   └── health_check.sh
├── cloudformation/         # Alternative IaC (CloudFormation)
│   └── stack.yaml
├── user_data/              # EC2 Initialization Scripts
│   └── init.sh
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the cloud folder:

```env
AWS_REGION=us-east-1
AWS_INSTANCE_TYPE=t3.large
AWS_KEY_NAME=your-key-pair
AWS_SECURITY_GROUP=your-sg-id
APP_PORT=8030
APP_HOST=0.0.0.0
```

### SSH Configuration

Ensure your SSH key is configured:

```bash
export AWS_KEY_PATH=~/.ssh/your-key.pem
chmod 400 $AWS_KEY_PATH
```

## 📊 Architecture

```
┌─────────────────┐
│   AWS EC2       │
│                 │
│  ┌───────────┐  │
│  │  Docker   │  │
│  │  Compose  │  │
│  └───────────┘  │
│       │         │
│  ┌───────────┐  │
│  │   App     │  │
│  │  :8030    │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │  Nginx    │  │
│  │  :80/443  │  │
│  └───────────┘  │
└─────────────────┘
```

## 🔒 Security

- Security groups configured for minimal exposure
- SSL/TLS termination at Nginx
- Firewall rules applied
- Secrets managed via AWS Secrets Manager
- IAM roles with least privilege

## 📈 Monitoring

- CloudWatch integration
- Prometheus metrics endpoint
- Health check endpoints
- Log aggregation

## 🔄 Automatic Deployment (CI/CD)

### GitHub Actions Integration

The project includes comprehensive GitHub Actions workflows for automated CI/CD:

#### Main Deployment Workflow (`deploy-to-ec2.yml`)

**Automatic Trigger**: Push to `main` branch

**Features**:
- ✅ **Smart Change Detection**: Only deploys if relevant files changed
- ✅ **Multi-stage Pipeline**: Validate → Test → Security → Build → Deploy
- ✅ **Automatic Backup**: Creates backup before each deployment
- ✅ **Health Checks**: Verifies deployment with retries
- ✅ **Automatic Rollback**: Rolls back on failure
- ✅ **Post-deployment Tasks**: Updates tags, cleans up
- ✅ **Notifications**: Slack integration (optional)

**Setup**:
1. Configure GitHub Secrets (see `.github/workflows/README.md`)
2. Push to `main` branch
3. Workflow runs automatically

**Manual Trigger**:
- Go to Actions → "Deploy to EC2 on Push to Main" → "Run workflow"

#### CI Workflow (`ci.yml`)

**Trigger**: Pull requests and feature branches

**Features**:
- Code linting and formatting checks
- Unit tests with coverage
- Security scanning
- Infrastructure validation

#### Nightly Workflow (`nightly.yml`)

**Trigger**: Daily at 2 AM UTC

**Features**:
- Comprehensive test suite across Python versions
- Security audits
- Dependency checks
- Performance tests

#### Cleanup Workflow (`cleanup.yml`)

**Trigger**: Weekly on Sundays

**Features**:
- Cleans up old workflow runs
- Maintains storage efficiency
- Keeps minimum runs for history

See [`.github/workflows/README.md`](../../.github/workflows/README.md) for detailed setup instructions.

## 🛠️ Advanced Scripts

### Deployment Scripts

- **`deployment_status.sh`**: Check current deployment status and health
- **`quick_deploy.sh`**: Fast deployment for rapid iterations
- **`compare_versions.sh`**: Compare local vs remote versions
- **`metrics.sh`**: Collect and display deployment metrics
- **`notify.sh`**: Send deployment notifications

### Enterprise Scripts

- **`cost_optimizer.sh`**: AWS cost analysis and optimization
- **`disaster_recovery.sh`**: Comprehensive disaster recovery management
- **`log_aggregator.sh`**: Centralized log management and analysis
- **`security_hardening.sh`**: Automated security hardening and compliance
- **`health_monitor.sh`**: Continuous health monitoring with alerting
- **`performance_test.sh`**: Load testing and performance benchmarking
- **`backup_manager.sh`**: Complete backup management system
- **`update_dependencies.sh`**: Dependency management and security checking
- **`auto_scaling.sh`**: Automatic resource scaling based on metrics
- **`dashboard.sh`**: Real-time deployment dashboard
- **`optimize.sh`**: System and application optimization
- **`report_generator.sh`**: Comprehensive report generation
- **`alert_manager.sh`**: Centralized alert management
- **`integration_test.sh`**: Integration testing suite
- **`api_manager.sh`**: REST API interface for deployment operations
- **`audit_trail.sh`**: Track and log all deployment activities
- **`secret_manager.sh`**: Secure secret management
- **`multi_region.sh`**: Multi-region deployment management
- **`blue_green.sh`**: Blue-green deployment strategy
- **`analytics.sh`**: Advanced analytics and insights
- **`canary_deploy.sh`**: Canary deployment strategy
- **`service_mesh.sh`**: Service mesh management
- **`ai_insights.sh`**: AI-powered insights and recommendations
- **`kubernetes.sh`**: Kubernetes deployment management
- **`auto_remediate.sh`**: Automated issue detection and remediation
- **`compliance_automation.sh`**: Automated compliance checks and remediation
- **`multi_cloud.sh`**: Multi-cloud deployment management
- **`advanced_monitoring.sh`**: Comprehensive monitoring and alerting
- **`test_framework.sh`**: Comprehensive testing suite
- **`cost_automation.sh`**: Automated cost optimization
- **`serverless.sh`**: Serverless function deployment
- **`ml_ops.sh`**: Machine learning model deployment
- **`edge_deploy.sh`**: Edge computing deployment
- **`chaos_engineering.sh`**: Chaos engineering experiments
- **`gitops.sh`**: GitOps deployment practices

### Usage Examples

```bash
# Check deployment status
./scripts/deployment_status.sh --ip <instance-ip> --key-path ~/.ssh/key.pem

# Quick deployment
./scripts/quick_deploy.sh --ip <instance-ip> --key-path ~/.ssh/key.pem

# Compare versions
./scripts/compare_versions.sh --ip <instance-ip> --key-path ~/.ssh/key.pem

# Collect metrics
./scripts/metrics.sh --ip <instance-ip> --key-path ~/.ssh/key.pem

# JSON metrics output
./scripts/metrics.sh --ip <instance-ip> --key-path ~/.ssh/key.pem --json > metrics.json

# Enterprise features
./scripts/cost_optimizer.sh analyze --days 30
./scripts/disaster_recovery.sh backup-now --ip <instance-ip>
./scripts/log_aggregator.sh collect --ip <instance-ip>
./scripts/security_hardening.sh audit --ip <instance-ip>
```

Or use Makefile shortcuts:

```bash
make status          # Check deployment status
make quick-deploy     # Quick deployment
make compare         # Compare versions
make metrics         # Collect metrics
make cost-analyze    # Analyze AWS costs
make dr-backup        # Disaster recovery backup
make logs-collect     # Collect logs
make security-audit   # Security audit
```

See [ADVANCED_FEATURES.md](./ADVANCED_FEATURES.md) and [IMPROVEMENTS_V3.md](./IMPROVEMENTS_V3.md) for more details.

## 🛠️ Maintenance

### Update Application

```bash
./scripts/update_app.sh
```

### View Logs

```bash
./scripts/view_logs.sh
```

### Health Check

```bash
./scripts/health_check.sh
```

### Fix Script Permissions

If scripts are not executable:

```bash
./scripts/fix_permissions.sh
```

Or:

```bash
make fix-perms
```

### Check Environment

Verify your environment is properly configured:

```bash
./scripts/check_environment.sh
```

Or:

```bash
make check-env
```

## 📝 Documentation

- [Terraform Guide](./terraform/README.md)
- [Ansible Guide](./ansible/README.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
- [DEPLOYMENT_BEST_PRACTICES.md](./DEPLOYMENT_BEST_PRACTICES.md) - Best practices guide
- [ADVANCED_FEATURES.md](./ADVANCED_FEATURES.md) - Advanced features documentation
- [FEATURES_SUMMARY.md](./FEATURES_SUMMARY.md) - Complete features overview
- [IMPROVEMENTS_V2.md](./IMPROVEMENTS_V2.md) - Advanced improvements and features
- [IMPROVEMENTS_V3.md](./IMPROVEMENTS_V3.md) - Enterprise features and improvements
- [IMPROVEMENTS_V4.md](./IMPROVEMENTS_V4.md) - Advanced operations and optimization
- [IMPROVEMENTS_V5.md](./IMPROVEMENTS_V5.md) - Reporting, alerts, and integration
- [IMPROVEMENTS_V6.md](./IMPROVEMENTS_V6.md) - API management, audit trail, and secrets
- [IMPROVEMENTS_V7.md](./IMPROVEMENTS_V7.md) - Multi-region, blue-green, and analytics
- [IMPROVEMENTS_V8.md](./IMPROVEMENTS_V8.md) - Canary, service mesh, and AI insights
- [IMPROVEMENTS_V9.md](./IMPROVEMENTS_V9.md) - Kubernetes, auto-remediation, and compliance
- [IMPROVEMENTS_V10.md](./IMPROVEMENTS_V10.md) - Multi-cloud, advanced monitoring, and testing
- [IMPROVEMENTS_V11.md](./IMPROVEMENTS_V11.md) - Serverless, ML Ops, edge, and GitOps
- [FINAL_SUMMARY.md](./FINAL_SUMMARY.md) - Complete system summary and quick reference

## 🆘 Support

For issues or questions, refer to the main project README or create an issue.

