# Makefile - Cursor Agent 24/7
# =============================
# Comandos útiles para desarrollo y producción

.PHONY: help build up down logs restart clean test

# Variables
COMPOSE_FILE = docker-compose.yml
COMPOSE_DEV = docker-compose.yml docker-compose.dev.yml
COMPOSE_PROD = docker-compose.yml docker-compose.prod.yml

help: ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Desarrollo
dev: ## Iniciar stack de desarrollo
	docker-compose -f $(COMPOSE_DEV) up

dev-build: ## Build e iniciar desarrollo
	docker-compose -f $(COMPOSE_DEV) up --build

dev-detach: ## Iniciar desarrollo en background
	docker-compose -f $(COMPOSE_DEV) up -d

dev-logs: ## Ver logs de desarrollo
	docker-compose -f $(COMPOSE_DEV) logs -f

# Producción
prod: ## Iniciar stack de producción
	docker-compose -f $(COMPOSE_PROD) up -d

prod-build: ## Build e iniciar producción
	docker-compose -f $(COMPOSE_PROD) build
	docker-compose -f $(COMPOSE_PROD) up -d

prod-logs: ## Ver logs de producción
	docker-compose -f $(COMPOSE_PROD) logs -f

# Básico
up: ## Iniciar servicios básicos
	docker-compose up -d

down: ## Detener servicios
	docker-compose down

restart: ## Reiniciar servicios
	docker-compose restart

logs: ## Ver logs
	docker-compose logs -f

# Build
build: ## Build imágenes
	docker-compose build

build-no-cache: ## Build sin cache
	docker-compose build --no-cache

# Limpieza
clean: ## Limpiar contenedores y volúmenes
	docker-compose down -v

clean-all: ## Limpiar todo (contenedores, imágenes, volúmenes)
	docker-compose down -v --rmi all
	docker system prune -f

# Servicios individuales
api: ## Iniciar solo API
	docker-compose up -d api redis

worker: ## Iniciar solo worker
	docker-compose up -d worker redis

worker-logs: ## Ver logs del worker
	docker-compose logs -f worker

# Monitoreo
monitoring: ## Iniciar stack de monitoreo (Prometheus, Grafana, Flower)
	docker-compose --profile monitoring up -d

flower: ## Iniciar Flower (monitor de Celery)
	docker-compose --profile monitoring up -d flower
	@echo "Flower disponible en: http://localhost:5555"

prometheus: ## Iniciar Prometheus
	docker-compose --profile monitoring up -d prometheus
	@echo "Prometheus disponible en: http://localhost:9090"

grafana: ## Iniciar Grafana
	docker-compose --profile monitoring up -d grafana
	@echo "Grafana disponible en: http://localhost:3000"

# Message Broker
rabbitmq: ## Iniciar RabbitMQ
	docker-compose --profile message-broker up -d rabbitmq
	@echo "RabbitMQ Management UI: http://localhost:15672"

# Reverse Proxy
nginx: ## Iniciar Nginx
	docker-compose --profile reverse-proxy up -d nginx

# Escalado
scale-worker: ## Escalar workers (uso: make scale-worker N=5)
	docker-compose up -d --scale worker=$(N)

scale-api: ## Escalar API (uso: make scale-api N=3)
	docker-compose up -d --scale api=$(N)

# Testing
test: ## Ejecutar tests
	docker-compose exec api pytest

test-coverage: ## Tests con coverage
	docker-compose exec api pytest --cov

# Shell
shell-api: ## Entrar al shell de API
	docker-compose exec api bash

shell-worker: ## Entrar al shell de worker
	docker-compose exec worker bash

shell-redis: ## Entrar al shell de Redis
	docker-compose exec redis redis-cli

# Health
health: ## Verificar salud de servicios
	@echo "API Health:"
	@curl -s http://localhost:8024/api/health | jq . || echo "API no disponible"
	@echo "\nRedis:"
	@docker-compose exec -T redis redis-cli ping || echo "Redis no disponible"
	@echo "\nCelery Workers:"
	@docker-compose exec -T worker celery -A core.celery_worker.celery_app inspect active || echo "Workers no disponibles"

# Status
status: ## Ver estado de servicios
	docker-compose ps

# Actualización
update: ## Actualizar imágenes y reiniciar
	docker-compose pull
	docker-compose up -d --build

# Desarrollo rápido
quick-start: ## Inicio rápido (API + Redis)
	docker-compose up -d api redis
	@echo "API disponible en: http://localhost:8024"
	@echo "Health check: http://localhost:8024/api/health"

# Producción completa
prod-full: ## Stack completo de producción
	docker-compose -f $(COMPOSE_PROD) up -d
	docker-compose --profile monitoring up -d
	@echo "Stack de producción iniciado"
	@echo "API: http://localhost:8024"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"




