.PHONY: help build push deploy test clean

# Configuration
AWS_REGION ?= us-east-1
AWS_ACCOUNT_ID ?= $(shell aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY ?= faceless-video-ai
IMAGE_TAG ?= latest
CLUSTER_NAME ?= faceless-video-ai-production
SERVICE_NAME ?= faceless-video-ai-service
STACK_NAME ?= faceless-video-ai-infrastructure

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t $(ECR_REPOSITORY):$(IMAGE_TAG) -f ../../Dockerfile ../..
	@echo "Build complete!"

build-dev: ## Build development Docker image
	@echo "Building development Docker image..."
	docker build -t $(ECR_REPOSITORY):dev -f ../../Dockerfile.dev ../..
	@echo "Build complete!"

test: ## Run tests
	@echo "Running tests..."
	cd ../.. && pytest tests/ -v --cov=. --cov-report=html
	@echo "Tests complete!"

lint: ## Run linters
	@echo "Running linters..."
	cd ../.. && \
	black --check . && \
	isort --check-only . && \
	flake8 .
	@echo "Linting complete!"

format: ## Format code
	@echo "Formatting code..."
	cd ../.. && \
	black . && \
	isort .
	@echo "Formatting complete!"

login-ecr: ## Login to ECR
	@echo "Logging in to ECR..."
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	@echo "Login successful!"

create-ecr: ## Create ECR repository
	@echo "Creating ECR repository..."
	aws ecr create-repository \
		--repository-name $(ECR_REPOSITORY) \
		--region $(AWS_REGION) \
		--image-scanning-configuration scanOnPush=true \
		--encryption-configuration encryptionType=AES256 || true
	@echo "Repository ready!"

tag: ## Tag Docker image for ECR
	@echo "Tagging image..."
	docker tag $(ECR_REPOSITORY):$(IMAGE_TAG) \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):$(IMAGE_TAG)
	docker tag $(ECR_REPOSITORY):$(IMAGE_TAG) \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):latest
	@echo "Tagging complete!"

push: login-ecr tag ## Push Docker image to ECR
	@echo "Pushing image to ECR..."
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):$(IMAGE_TAG)
	docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY):latest
	@echo "Push complete!"

deploy-infra: ## Deploy CloudFormation infrastructure
	@echo "Deploying infrastructure..."
	aws cloudformation deploy \
		--template-file cloudformation-template.yaml \
		--stack-name $(STACK_NAME) \
		--capabilities CAPABILITY_NAMED_IAM \
		--parameter-overrides \
			Environment=production
	@echo "Infrastructure deployed!"

update-task-def: ## Update ECS task definition
	@echo "Updating task definition..."
	@sed -i.bak "s/ACCOUNT_ID/$(AWS_ACCOUNT_ID)/g" ecs-task-definition.json
	@sed -i.bak "s/REGION/$(AWS_REGION)/g" ecs-task-definition.json
	aws ecs register-task-definition \
		--cli-input-json file://ecs-task-definition.json
	@mv ecs-task-definition.json.bak ecs-task-definition.json
	@echo "Task definition updated!"

deploy-ecs: push update-task-def ## Deploy to ECS
	@echo "Deploying to ECS..."
	@TASK_DEF_ARN=$$(aws ecs describe-task-definition \
		--task-definition faceless-video-ai \
		--query 'taskDefinition.taskDefinitionArn' \
		--output text); \
	aws ecs update-service \
		--cluster $(CLUSTER_NAME) \
		--service $(SERVICE_NAME) \
		--task-definition $$TASK_DEF_ARN \
		--force-new-deployment
	@echo "Deployment initiated!"

deploy: test build push deploy-ecs ## Full deployment pipeline

status: ## Check ECS service status
	@echo "Service status:"
	@aws ecs describe-services \
		--cluster $(CLUSTER_NAME) \
		--services $(SERVICE_NAME) \
		--query 'services[0].[serviceName,status,desiredCount,runningCount,pendingCount]' \
		--output table

logs: ## View CloudWatch logs
	@echo "Viewing logs (Ctrl+C to exit)..."
	aws logs tail /ecs/faceless-video-ai --follow

clean: ## Clean up local Docker images
	@echo "Cleaning up..."
	docker rmi $(ECR_REPOSITORY):$(IMAGE_TAG) || true
	docker rmi $(ECR_REPOSITORY):dev || true
	@echo "Cleanup complete!"

clean-ecr: ## Clean up old ECR images (keep last 10)
	@echo "Cleaning up ECR images..."
	aws ecr list-images \
		--repository-name $(ECR_REPOSITORY) \
		--region $(AWS_REGION) \
		--query 'imageIds[10:][].imageDigest' \
		--output text | \
	while read digest; do \
		aws ecr batch-delete-image \
			--repository-name $(ECR_REPOSITORY) \
			--region $(AWS_REGION) \
			--image-ids imageDigest=$$digest || true; \
	done
	@echo "ECR cleanup complete!"

secrets: ## List secrets in Secrets Manager
	@echo "Secrets:"
	@aws secretsmanager list-secrets \
		--filters Key=name,Values=faceless-video-ai \
		--query 'SecretList[].Name' \
		--output table

local: ## Run locally with docker-compose
	@echo "Starting local environment..."
	cd ../.. && docker-compose up -d
	@echo "Local environment running!"

local-down: ## Stop local environment
	@echo "Stopping local environment..."
	cd ../.. && docker-compose down
	@echo "Local environment stopped!"

local-logs: ## View local logs
	@echo "Viewing local logs..."
	cd ../.. && docker-compose logs -f




