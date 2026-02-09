# Kubernetes Deployment - Robot Movement AI v2.0

## 📋 Descripción

Configuración de Kubernetes para desplegar Robot Movement AI v2.0 en un cluster.

## 🚀 Deployment

### Prerrequisitos

- Kubernetes cluster configurado
- `kubectl` instalado y configurado
- Docker image disponible en registry

### Pasos

1. **Crear secrets**

```bash
kubectl create secret generic robot-secrets \
  --from-literal=database-url=postgresql://user:pass@host/db \
  --from-literal=secret-key=your-secret-key
```

2. **Aplicar deployment**

```bash
kubectl apply -f deployment.yaml
```

3. **Verificar**

```bash
kubectl get pods
kubectl get services
kubectl get hpa
```

## 📊 Características

- ✅ Deployment con 3 réplicas
- ✅ Service con LoadBalancer
- ✅ Horizontal Pod Autoscaler (3-10 pods)
- ✅ Health checks (liveness y readiness)
- ✅ Resource limits configurados
- ✅ Secrets management

## 🔧 Configuración

Editar `deployment.yaml` para ajustar:
- Número de réplicas
- Resource limits
- Variables de entorno
- Health check intervals

## 📚 Más Información

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)




