.PHONY: help build dev prod up down logs clean test start stop

# Default target - start development environment
.DEFAULT_GOAL := start

help:
	@echo "Music Analyzer AI - Docker Commands"
	@echo ""
	@echo "Quick Start (ONE COMMAND):"
	@echo "  make start          - Start everything (development)"
	@echo "  make start ENV=prod - Start everything (production)"
	@echo ""
	@echo "Available commands:"
	@echo "  make start          - Start all services (default: dev)"
	@echo "  make stop           - Stop all services"
	@echo "  make build          - Build Docker image"
	@echo "  make dev            - Start development environment"
	@echo "  make prod           - Start production environment"
	@echo "  make up             - Start default environment"
	@echo "  make down           - Stop all services"
	@echo "  make logs           - View logs"
	@echo "  make clean          - Clean up containers and volumes"
	@echo "  make test           - Run tests in container"
	@echo "  make shell          - Open shell in container"
	@echo "  make health         - Check health status"

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build -f deployment/Dockerfile -t music-analyzer-ai:latest ..

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose -f deployment/docker-compose.dev.yml up -d
	@echo "Services started!"
	@echo "API: http://localhost:8010"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"

# Production environment
prod:
	@echo "Starting production environment..."
	@if [ ! -f .env ]; then \
		echo "Warning: .env file not found. Copying from .env.example..."; \
		cp deployment/.env.example .env; \
		echo "Please update .env with your configuration before starting production!"; \
	fi
	docker-compose -f deployment/docker-compose.prod.yml up -d
	@echo "Production services started!"

# Default environment
up:
	@echo "Starting default environment..."
	docker-compose -f deployment/docker-compose.yml up -d

# Stop services
down:
	@echo "Stopping services..."
	docker-compose -f deployment/docker-compose.yml down
	docker-compose -f deployment/docker-compose.dev.yml down
	docker-compose -f deployment/docker-compose.prod.yml down

# View logs
logs:
	docker-compose -f deployment/docker-compose.yml logs -f

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose -f deployment/docker-compose.yml down -v
	docker-compose -f deployment/docker-compose.dev.yml down -v
	docker-compose -f deployment/docker-compose.prod.yml down -v
	docker system prune -f

# Run tests
test:
	@echo "Running tests in container..."
	docker run --rm \
		-v $(PWD):/app \
		-w /app \
		music-analyzer-ai:latest \
		pytest tests/ -v

# Open shell in container
shell:
	@echo "Opening shell in container..."
	docker run --rm -it \
		-v $(PWD):/app \
		-w /app \
		-e SPOTIFY_CLIENT_ID=${SPOTIFY_CLIENT_ID} \
		-e SPOTIFY_CLIENT_SECRET=${SPOTIFY_CLIENT_SECRET} \
		music-analyzer-ai:latest \
		/bin/bash

# Check health
health:
	@echo "Checking health status..."
	@curl -s http://localhost:8010/health | python -m json.tool || echo "Service not available"

# One-command start (default)
start:
	@echo "🚀 Starting Music Analyzer AI..."
	@if [ -z "$(ENV)" ]; then \
		./start.sh dev; \
	else \
		./start.sh $(ENV); \
	fi

# One-command stop
stop:
	@echo "🛑 Stopping Music Analyzer AI..."
	@if [ -z "$(ENV)" ]; then \
		./stop.sh dev; \
	else \
		./stop.sh $(ENV); \
	fi

# Rebuild and restart
rebuild:
	@echo "Rebuilding and restarting..."
	make build
	make down
	make up

# Development rebuild
dev-rebuild:
	@echo "Rebuilding development environment..."
	make build
	docker-compose -f deployment/docker-compose.dev.yml down
	make dev

# Production rebuild
prod-rebuild:
	@echo "Rebuilding production environment..."
	make build
	docker-compose -f deployment/docker-compose.prod.yml down
	make prod

