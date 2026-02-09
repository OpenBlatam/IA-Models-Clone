.PHONY: help init plan apply destroy deploy setup check-health update clean validate

# Default target
help:
	@echo "AI Project Generator - AWS Deployment"
	@echo ""
	@echo "Available targets:"
	@echo "  make init          - Initialize Terraform"
	@echo "  make plan          - Plan Terraform deployment"
	@echo "  make apply         - Apply Terraform configuration"
	@echo "  make destroy       - Destroy all infrastructure"
	@echo "  make deploy        - Deploy application with Ansible"
	@echo "  make setup         - Set up local environment"
	@echo "  make check-health  - Check application health"
	@echo "  make update        - Update application"
	@echo "  make clean         - Clean temporary files"
	@echo "  make validate      - Validate configurations"
	@echo ""
	@echo "Variables:"
	@echo "  ENVIRONMENT        - Environment name (default: production)"
	@echo "  DEPLOY_INFRA       - Deploy infrastructure (default: false)"
	@echo "  DEPLOY_APP         - Deploy application (default: true)"
	@echo "  APP_PORT           - Application port (default: 8020)"

ENVIRONMENT ?= production
DEPLOY_INFRA ?= false
DEPLOY_APP ?= true
APP_PORT ?= 8020

# Terraform targets
init:
	cd terraform && terraform init

plan:
	cd terraform && terraform plan

apply:
	cd terraform && terraform apply

destroy:
	cd terraform && terraform destroy

# Ansible targets
deploy:
	cd ansible && \
	ansible-playbook \
		-i inventory/ec2.ini \
		playbooks/deploy.yml \
		-e "environment=$(ENVIRONMENT)" \
		-e "app_port=$(APP_PORT)"

update:
	cd ansible && \
	ansible-playbook \
		-i inventory/ec2.ini \
		playbooks/update.yml \
		-e "environment=$(ENVIRONMENT)" \
		-e "app_port=$(APP_PORT)"

setup:
	cd ansible && \
	ansible-playbook \
		-i inventory/ec2.ini \
		playbooks/setup.yml

# Utility targets
check-health:
	@cd terraform && \
	LB_DNS=$$(terraform output -raw load_balancer_dns 2>/dev/null) && \
	if [ -n "$$LB_DNS" ]; then \
		echo "Checking health at http://$$LB_DNS/health"; \
		curl -f -s "http://$$LB_DNS/health" && echo "✅ Health check passed!" || echo "❌ Health check failed"; \
	else \
		echo "⚠️  Load balancer DNS not available"; \
	fi

setup-env:
	cd scripts && ./setup_environment.sh

# Combined deployment
full-deploy: init apply deploy

# Cleanup
clean:
	find . -type f -name "*.retry" -delete
	find . -type f -name "*.log" -delete
	cd terraform && rm -f *.tfplan

# Validation
validate:
	cd terraform && terraform validate
	cd terraform && terraform fmt -check
	cd ansible && ansible-playbook --syntax-check playbooks/*.yml

# Show outputs
outputs:
	cd terraform && terraform output

# Show instance IPs
show-ips:
	@cd terraform && \
	terraform output ec2_instance_public_ips 2>/dev/null || \
	terraform output ec2_instance_private_ips 2>/dev/null || \
	echo "No instances found"

