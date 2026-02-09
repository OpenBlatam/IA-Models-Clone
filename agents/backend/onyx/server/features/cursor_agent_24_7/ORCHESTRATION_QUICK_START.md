# ⚡ Orquestación - Inicio Rápido

## 🐳 Docker Compose (Más Fácil)

```bash
# Desarrollo
docker-compose up

# Producción
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Con Makefile
make dev      # Desarrollo
make prod     # Producción
make monitoring  # Con monitoreo
```

## ☸️ Kubernetes

```bash
# Crear secrets primero
kubectl create secret generic cursor-agent-secrets \
  --from-literal=jwt-secret-key='your-secret' \
  --namespace=cursor-agent

# Desplegar
kubectl apply -f k8s/

# Ver estado
kubectl get pods -n cursor-agent
```

## ☁️ AWS ECS

```bash
# Con Terraform
cd aws/terraform
terraform apply

# O manualmente
aws ecs create-service --cluster cursor-agent-cluster ...
```

## 📚 Documentación Completa

Ver [ORCHESTRATION.md](ORCHESTRATION.md) para documentación detallada.




