# Makefile for 3D Prototype AI Cloud Deployment
# Provides convenient commands for deployment operations

.PHONY: help deploy launch terraform ansible health update logs clean validate test cleanup

# Default target
help:
	@echo "3D Prototype AI - Cloud Deployment Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make deploy          - Full deployment (Terraform + App)"
	@echo "  make launch          - Quick EC2 launch"
	@echo "  make terraform       - Deploy with Terraform"
	@echo "  make ansible         - Deploy with Ansible"
	@echo "  make validate        - Validate configuration"
	@echo "  make test            - Run script tests"
	@echo "  make health          - Check application health"
	@echo "  make update          - Update application"
	@echo "  make logs            - View application logs"
	@echo "  make cleanup         - Clean deployment artifacts"
	@echo "  make clean           - Clean Terraform state"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-up       - Start Docker containers"
	@echo "  make docker-down     - Stop Docker containers"
	@echo "  make fix-perms       - Fix script permissions"
	@echo "  make check-env       - Check environment"
	@echo "  make terraform-validate - Validate Terraform config"
	@echo "  make terraform-fmt   - Format Terraform files"
	@echo "  make auto-deploy      - Automatic deployment to EC2"
	@echo "  make quick-deploy     - Quick deployment (fast)"
	@echo "  make get-instance     - Get EC2 instance information"
	@echo "  make status           - Show deployment status"
	@echo "  make metrics          - Collect deployment metrics"
	@echo "  make compare          - Compare local vs remote versions"
	@echo "  make monitor          - Start health monitoring"
	@echo "  make perf-test        - Run performance tests"
	@echo "  make backup-create    - Create backup"
	@echo "  make backup-list      - List backups"
	@echo "  make backup-status    - Backup status"
	@echo "  make update-deps      - Update dependencies"
	@echo "  make cost-analyze     - Analyze AWS costs"
	@echo "  make dr-backup         - Disaster recovery backup"
	@echo "  make logs-collect     - Collect logs"
	@echo "  make security-audit   - Security audit"
	@echo "  make dashboard         - Display dashboard"
	@echo "  make auto-scale        - Auto-scaling management"
	@echo "  make optimize          - Optimize system/application"
	@echo "  make report            - Generate reports"
	@echo "  make alerts            - Manage alerts"
	@echo "  make integration-test  - Run integration tests"
	@echo "  make api-start         - Start API server"
	@echo "  make audit-view        - View audit trail"
	@echo "  make secrets-list      - List secrets"
	@echo "  make multi-region      - Multi-region deployment"
	@echo "  make blue-green        - Blue-green deployment"
	@echo "  make analytics         - Advanced analytics"
	@echo "  make canary            - Canary deployment"
	@echo "  make service-mesh      - Service mesh management"
	@echo "  make ai-insights       - AI-powered insights"
	@echo "  make kubernetes        - Kubernetes deployment"
	@echo "  make auto-remediate    - Automated remediation"
	@echo "  make compliance        - Compliance automation"
	@echo "  make multi-cloud       - Multi-cloud deployment"
	@echo "  make monitoring        - Advanced monitoring"
	@echo "  make test-all          - Run all tests"
	@echo "  make cost-auto         - Automated cost optimization"
	@echo "  make serverless        - Serverless deployment"
	@echo "  make ml-ops            - ML Ops management"
	@echo "  make edge              - Edge deployment"
	@echo "  make gitops            - GitOps deployment"
	@echo ""

# Full deployment
deploy:
	@echo "Starting full deployment..."
	@./scripts/deploy.sh

# Quick launch
launch:
	@echo "Launching EC2 instance..."
	@./scripts/launch_ec2.sh

# Terraform deployment
terraform:
	@echo "Deploying with Terraform..."
	@cd terraform && terraform init && terraform apply

# Terraform validation
terraform-validate:
	@echo "Validating Terraform configuration..."
	@cd terraform && ./validate.sh || terraform validate

# Terraform format
terraform-fmt:
	@echo "Formatting Terraform files..."
	@cd terraform && terraform fmt -recursive
	@echo "Terraform files formatted ✓"

# Automatic deployment
auto-deploy:
	@echo "Starting automatic deployment..."
	@./scripts/auto_deploy.sh

# Get instance information
get-instance:
	@echo "Getting EC2 instance information..."
	@./scripts/get_instance_info.sh

# Quick deployment
quick-deploy:
	@echo "Starting quick deployment..."
	@./scripts/quick_deploy.sh

# Deployment status
status:
	@echo "Checking deployment status..."
	@./scripts/deployment_status.sh

# Collect metrics
metrics:
	@echo "Collecting deployment metrics..."
	@./scripts/metrics.sh

# Compare versions
compare:
	@echo "Comparing local and remote versions..."
	@./scripts/compare_versions.sh

# Health monitoring
monitor:
	@echo "Starting health monitor..."
	@./scripts/health_monitor.sh

# Performance test
perf-test:
	@echo "Running performance tests..."
	@./scripts/performance_test.sh

# Backup management
backup-create:
	@echo "Creating backup..."
	@./scripts/backup_manager.sh create

backup-list:
	@echo "Listing backups..."
	@./scripts/backup_manager.sh list

backup-status:
	@echo "Backup status..."
	@./scripts/backup_manager.sh status

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	@./scripts/update_dependencies.sh --update

# Cost optimization
cost-analyze:
	@echo "Analyzing AWS costs..."
	@./scripts/cost_optimizer.sh analyze

# Disaster recovery
dr-backup:
	@echo "Creating disaster recovery backup..."
	@./scripts/disaster_recovery.sh backup-now

# Log aggregation
logs-collect:
	@echo "Collecting logs..."
	@./scripts/log_aggregator.sh collect

# Security
security-audit:
	@echo "Running security audit..."
	@./scripts/security_hardening.sh audit

# Dashboard
dashboard:
	@echo "Displaying dashboard..."
	@./scripts/dashboard.sh

# Auto-scaling
auto-scale:
	@echo "Checking auto-scaling..."
	@./scripts/auto_scaling.sh check

# Optimization
optimize:
	@echo "Optimizing system..."
	@./scripts/optimize.sh all

# Reports
report:
	@echo "Generating comprehensive report..."
	@./scripts/report_generator.sh comprehensive

# Alerts
alerts:
	@echo "Setting up alerts..."
	@./scripts/alert_manager.sh setup

# Integration tests
integration-test:
	@echo "Running integration tests..."
	@./scripts/integration_test.sh

# API management
api-start:
	@echo "Starting API server..."
	@./scripts/api_manager.sh start

api-stop:
	@echo "Stopping API server..."
	@./scripts/api_manager.sh stop

api-status:
	@echo "Checking API status..."
	@./scripts/api_manager.sh status

# Audit trail
audit-view:
	@echo "Viewing audit trail..."
	@./scripts/audit_trail.sh view

# Secret management
secrets-list:
	@echo "Listing secrets..."
	@./scripts/secret_manager.sh list

# Multi-region
multi-region:
	@echo "Multi-region deployment status..."
	@./scripts/multi_region.sh status

# Blue-green deployment
blue-green:
	@echo "Blue-green deployment status..."
	@./scripts/blue_green.sh status

# Analytics
analytics:
	@echo "Generating analytics report..."
	@./scripts/analytics.sh report

# Canary deployment
canary:
	@echo "Canary deployment status..."
	@./scripts/canary_deploy.sh status

# Service mesh
service-mesh:
	@echo "Service mesh status..."
	@./scripts/service_mesh.sh status

# AI insights
ai-insights:
	@echo "Generating AI-powered insights..."
	@./scripts/ai_insights.sh analyze

# Kubernetes
kubernetes:
	@echo "Kubernetes deployment status..."
	@./scripts/kubernetes.sh status

# Auto-remediation
auto-remediate:
	@echo "Checking for issues..."
	@./scripts/auto_remediate.sh check

# Compliance
compliance:
	@echo "Running compliance checks..."
	@./scripts/compliance_automation.sh check

# Multi-cloud
multi-cloud:
	@echo "Multi-cloud deployment status..."
	@./scripts/multi_cloud.sh status

# Advanced monitoring
monitoring:
	@echo "Starting advanced monitoring..."
	@./scripts/advanced_monitoring.sh start

# Testing framework
test-all:
	@echo "Running all tests..."
	@./scripts/test_framework.sh all

# Cost automation
cost-auto:
	@echo "Analyzing and optimizing costs..."
	@./scripts/cost_automation.sh analyze

# Serverless
serverless:
	@echo "Serverless function status..."
	@./scripts/serverless.sh status

# ML Ops
ml-ops:
	@echo "ML Ops status..."
	@./scripts/ml_ops.sh monitor

# Edge deployment
edge:
	@echo "Edge deployment status..."
	@./scripts/edge_deploy.sh status

# GitOps
gitops:
	@echo "GitOps status..."
	@./scripts/gitops.sh status

# Ansible deployment
ansible:
	@echo "Deploying with Ansible..."
	@cd ansible && ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml

# Health check
health:
	@echo "Checking application health..."
	@./scripts/health_check.sh

# Update application
update:
	@echo "Updating application..."
	@./scripts/update_app.sh

# View logs
logs:
	@echo "Viewing application logs..."
	@./scripts/view_logs.sh app

# Validate configuration
validate:
	@echo "Validating configuration..."
	@./scripts/validate.sh

# Run tests
test:
	@echo "Running script tests..."
	@./scripts/test_scripts.sh

# Cleanup artifacts
cleanup:
	@echo "Cleaning deployment artifacts..."
	@./scripts/cleanup.sh --all --force

# Clean Terraform
clean:
	@echo "Cleaning Terraform state..."
	@cd terraform && terraform destroy
	@cd terraform && rm -rf .terraform *.tfstate* *.tfplan

# Docker commands
docker-build:
	@echo "Building Docker images..."
	@cd .. && ./cloud/scripts/docker_utils.sh build -e production

docker-up:
	@echo "Starting Docker containers..."
	@cd .. && ./cloud/scripts/docker_utils.sh up -e production

docker-down:
	@echo "Stopping Docker containers..."
	@cd .. && ./cloud/scripts/docker_utils.sh down -e production

docker-logs:
	@echo "Viewing Docker logs..."
	@cd .. && ./cloud/scripts/docker_utils.sh logs

docker-health:
	@echo "Checking Docker container health..."
	@cd .. && ./cloud/scripts/docker_utils.sh health

# Fix permissions
fix-perms:
	@echo "Fixing script permissions..."
	@./scripts/fix_permissions.sh

# Check environment
check-env:
	@echo "Checking environment..."
	@./scripts/check_environment.sh

# Setup (first time)
setup:
	@echo "Setting up deployment environment..."
	@./scripts/setup.sh

