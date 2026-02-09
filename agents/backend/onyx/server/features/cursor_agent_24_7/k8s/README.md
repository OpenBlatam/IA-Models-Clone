# ☸️ Kubernetes - Cursor Agent 24/7

Manifests de Kubernetes para desplegar el sistema completo.

## 🚀 Despliegue Rápido

```bash
# Aplicar todos los recursos
kubectl apply -f k8s/

# Ver estado
kubectl get pods -n cursor-agent
kubectl get services -n cursor-agent

# Ver logs
kubectl logs -f deployment/cursor-agent-api -n cursor-agent
```

## 📋 Archivos

- `namespace.yaml` - Namespace
- `configmap.yaml` - Configuración
- `secrets.yaml.example` - Secrets (crear con valores reales)
- `redis.yaml` - Redis deployment
- `api.yaml` - API deployment con HPA
- `worker.yaml` - Worker deployment con HPA
- `beat.yaml` - Beat scheduler
- `ingress.yaml` - Ingress con TLS

## 🔐 Secrets

Crear secrets antes de desplegar:

```bash
kubectl create secret generic cursor-agent-secrets \
  --from-literal=jwt-secret-key='your-secret-key' \
  --from-literal=redis-password='your-redis-password' \
  --from-literal=aws-access-key-id='your-access-key' \
  --from-literal=aws-secret-access-key='your-secret-key' \
  --namespace=cursor-agent
```

## 📈 Escalado

```bash
# Manual
kubectl scale deployment cursor-agent-api --replicas=5 -n cursor-agent

# Auto-scaling (HPA ya configurado)
kubectl get hpa -n cursor-agent
```

## 🔍 Monitoreo

```bash
# Ver métricas
kubectl top pods -n cursor-agent

# Ver eventos
kubectl get events -n cursor-agent --sort-by='.lastTimestamp'
```

## 🛠️ Troubleshooting

```bash
# Describir pod
kubectl describe pod <pod-name> -n cursor-agent

# Logs de todos los pods
kubectl logs -l app=cursor-agent-api -n cursor-agent

# Ejecutar comando en pod
kubectl exec -it <pod-name> -n cursor-agent -- bash
```




