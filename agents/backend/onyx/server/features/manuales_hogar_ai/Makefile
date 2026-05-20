# Makefile for Manuales Hogar AI
# Usage: make [target]

.PHONY: help start stop restart logs shell migrate test clean build dev prod

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

start: ## Start all services (development)
	@./start.sh dev || python run.py dev

start-prod: ## Start all services (production)
	@./start.sh prod || python run.py prod

start-staging: ## Start all services (staging)
	@./start.sh staging || python run.py staging

status: ## Show service status
	@./status.sh || echo "Status script not available"

stop: ## Stop all services
	@./stop.sh || docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs
	docker-compose logs -f

logs-app: ## View app logs only
	docker-compose logs -f app

shell: ## Access app container shell
	docker-compose exec app bash

migrate: ## Run database migrations
	docker-compose exec app alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create NAME=migration_name)
	docker-compose exec app alembic revision --autogenerate -m "$(NAME)"

test: ## Run tests
	docker-compose exec app pytest

build: ## Build Docker images
	docker-compose build

dev: ## Start development environment
	docker-compose up -d

prod: ## Start production environment
	docker-compose -f docker-compose.prod.yml up -d

clean: ## Remove containers, networks, and volumes
	docker-compose down -v
	docker-compose -f docker-compose.prod.yml down -v
	docker system prune -f

clean-all: ## Remove everything including images
	@./scripts/clean.sh --all || docker-compose down -v

setup: ## Initial setup
	@./scripts/setup.sh || echo "Setup script not available"

check-health: ## Check service health
	@./scripts/check-health.sh || curl -f http://localhost:8000/api/v1/health

backup: ## Create backup
	@./scripts/backup.sh

backup-full: ## Create full backup
	@./scripts/backup.sh --full

restore: ## Restore from backup (usage: make restore BACKUP=backup_file)
	@./scripts/restore.sh $(BACKUP)

monitor: ## Real-time monitoring
	@./scripts/monitor.sh

test-api: ## Test all API endpoints
	@./scripts/test-api.sh

quick-test: ## Quick API validation
	@./scripts/quick-test.sh

update: ## Update application
	@./scripts/update.sh

update-full: ## Update with code pull and rebuild
	@./scripts/update.sh --pull --rebuild

diagnostics: ## Run system diagnostics
	@./scripts/diagnostics.sh

export-logs: ## Export logs to file
	@./scripts/export-logs.sh

export-logs-all: ## Export all logs
	@./scripts/export-logs.sh --all

performance-test: ## Performance testing
	@./scripts/performance-test.sh

security-check: ## Security validation
	@./scripts/security-check.sh

optimize: ## Optimize Docker and system
	@./scripts/optimize.sh

optimize-all: ## Full optimization
	@./scripts/optimize.sh --all

watch: ## Watch for changes and auto-restart
	@./scripts/watch.sh

validate-config: ## Validate configuration
	@./scripts/validate-config.sh

health-monitor: ## Continuous health monitoring
	@./scripts/health-monitor.sh

notify: ## Send notifications (usage: make notify WEBHOOK=url)
	@./scripts/notify.sh --webhook $(WEBHOOK)

ps: ## Show running containers
	docker-compose ps

health: ## Check service health
	@curl -f http://localhost:8000/api/v1/health && echo "Service is healthy!" || echo "Service is unhealthy!"

stats: ## Show container resource usage
	docker stats

rebuild: ## Rebuild and restart services
	docker-compose up -d --build

