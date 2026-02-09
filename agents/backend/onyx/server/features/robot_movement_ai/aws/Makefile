.PHONY: help build push deploy-ecs deploy-lambda deploy-terraform clean test

# Configuration
PROJECT_NAME := robot-movement-ai
AWS_REGION := us-east-1
AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
ECR_REPO := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(PROJECT_NAME)
ECS_CLUSTER := $(PROJECT_NAME)-cluster
ECS_SERVICE := $(PROJECT_NAME)-service

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -f aws/Dockerfile -t $(PROJECT_NAME):latest ..

push: build ## Build and push Docker image to ECR
	@echo "Logging in to ECR..."
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REPO)
	@echo "Tagging and pushing image..."
	docker tag $(PROJECT_NAME):latest $(ECR_REPO):latest
	docker tag $(PROJECT_NAME):latest $(ECR_REPO):$$(date +%Y%m%d-%H%M%S)
	docker push $(ECR_REPO):latest
	docker push $(ECR_REPO):$$(date +%Y%m%d-%H%M%S)
	@echo "Image pushed successfully"

deploy-ecs: push ## Deploy to ECS/Fargate
	@echo "Updating ECS service..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service $(ECS_SERVICE) \
		--force-new-deployment \
		--region $(AWS_REGION)
	@echo "Waiting for service to stabilize..."
	aws ecs wait services-stable \
		--cluster $(ECS_CLUSTER) \
		--services $(ECS_SERVICE) \
		--region $(AWS_REGION)
	@echo "Deployment completed!"

deploy-lambda: ## Deploy to Lambda using SAM
	@echo "Building SAM application..."
	cd aws && sam build
	@echo "Deploying SAM application..."
	cd aws && sam deploy --guided

deploy-terraform: ## Deploy infrastructure with Terraform
	@echo "Initializing Terraform..."
	cd aws/terraform && terraform init
	@echo "Planning Terraform deployment..."
	cd aws/terraform && terraform plan
	@echo "Applying Terraform changes..."
	cd aws/terraform && terraform apply

terraform-destroy: ## Destroy Terraform infrastructure
	@echo "Destroying infrastructure..."
	cd aws/terraform && terraform destroy

test: ## Run tests
	@echo "Running tests..."
	pytest tests/ -v

logs-ecs: ## View ECS logs
	aws logs tail /ecs/$(PROJECT_NAME) --follow

logs-lambda: ## View Lambda logs
	aws logs tail /aws/lambda/$(PROJECT_NAME) --follow

status-ecs: ## Check ECS service status
	@echo "ECS Service Status:"
	aws ecs describe-services \
		--cluster $(ECS_CLUSTER) \
		--services $(ECS_SERVICE) \
		--region $(AWS_REGION) \
		--query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}' \
		--output table

clean: ## Clean up local Docker images
	@echo "Cleaning up Docker images..."
	docker rmi $(PROJECT_NAME):latest 2>/dev/null || true
	docker system prune -f

setup-ecr: ## Create ECR repository if it doesn't exist
	@echo "Checking ECR repository..."
	@aws ecr describe-repositories --repository-names $(PROJECT_NAME) --region $(AWS_REGION) 2>/dev/null || \
		aws ecr create-repository --repository-name $(PROJECT_NAME) --region $(AWS_REGION)

setup-secrets: ## Create secrets in Secrets Manager
	@echo "Creating secrets in Secrets Manager..."
	@read -p "OpenAI API Key: " OPENAI_KEY; \
	read -p "Robot IP: " ROBOT_IP; \
	aws secretsmanager create-secret \
		--name $(PROJECT_NAME)/secrets \
		--secret-string "{\"openai_api_key\":\"$$OPENAI_KEY\",\"robot_ip\":\"$$ROBOT_IP\"}" \
		--region $(AWS_REGION) || \
	aws secretsmanager update-secret \
		--secret-id $(PROJECT_NAME)/secrets \
		--secret-string "{\"openai_api_key\":\"$$OPENAI_KEY\",\"robot_ip\":\"$$ROBOT_IP\"}" \
		--region $(AWS_REGION)

get-alb-url: ## Get ALB DNS name
	@cd aws/terraform && terraform output -json | jq -r '.alb_dns_name.value' 2>/dev/null || \
		echo "Run 'make deploy-terraform' first to get ALB URL"

get-lambda-url: ## Get Lambda API URL
	@aws cloudformation describe-stacks \
		--stack-name $(PROJECT_NAME) \
		--query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
		--output text 2>/dev/null || \
		echo "Lambda not deployed. Run 'make deploy-lambda' first"















