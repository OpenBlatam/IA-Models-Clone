# Makefile for Docker operations

.PHONY: help build dev prod up down logs shell migrate test clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build Docker images
	docker-compose build

build-dev: ## Build development image
	docker build -f Dockerfile.dev -t manuales-hogar-ai:dev .

build-prod: ## Build production image
	docker build -f Dockerfile.prod -t manuales-hogar-ai:prod .

dev: ## Start development environment
	docker-compose up -d
	@echo "Development environment started!"
	@echo "API: http://localhost:8000"
	@echo "View logs: make logs"

prod: ## Start production environment
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Production environment started!"
	@echo "API: http://localhost:8000"

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

down-v: ## Stop and remove volumes
	docker-compose down -v

logs: ## View logs
	docker-compose logs -f

logs-app: ## View app logs
	docker-compose logs -f app

logs-db: ## View database logs
	docker-compose logs -f postgres

logs-redis: ## View Redis logs
	docker-compose logs -f redis

shell: ## Access app container shell
	docker-compose exec app bash

shell-db: ## Access database shell
	docker-compose exec postgres psql -U postgres -d manuales_hogar

migrate: ## Run database migrations
	docker-compose exec app alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create NAME=migration_name)
	docker-compose exec app alembic revision --autogenerate -m "$(NAME)"

test: ## Run tests
	docker-compose exec app pytest

clean: ## Remove containers, networks, and volumes
	docker-compose down -v
	docker system prune -f

restart: ## Restart all services
	docker-compose restart

ps: ## Show running containers
	docker-compose ps

health: ## Check service health
	@curl -f http://localhost:8000/api/v1/health && echo "Service is healthy!" || echo "Service is unhealthy!"

stats: ## Show container resource usage
	docker stats

rebuild: ## Rebuild and restart services
	docker-compose up -d --build

stop: ## Stop all services
	docker-compose stop

start: ## Start stopped services
	docker-compose start




