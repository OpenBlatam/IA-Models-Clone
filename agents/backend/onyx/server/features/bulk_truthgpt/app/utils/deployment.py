"""
Advanced deployment utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import os
import time
import logging
import subprocess
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import current_app
import docker
import kubernetes
from pathlib import Path

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Advanced deployment manager with comprehensive deployment utilities."""
    
    def __init__(self, app=None):
        """Initialize deployment manager with early returns."""
        self.app = app
        self.docker_client = None
        self.k8s_client = None
        self.deployment_config = {}
        
        if app:
            self.init_app(app)
    
    def init_app(self, app) -> None:
        """Initialize deployment manager with app."""
        self.app = app
        self.deployment_config = app.config.get('DEPLOYMENT_CONFIG', {})
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"⚠️ Docker client not available: {e}")
        
        # Initialize Kubernetes client
        try:
            kubernetes.config.load_incluster_config()
            self.k8s_client = kubernetes.client.ApiClient()
        except Exception:
            try:
                kubernetes.config.load_kube_config()
                self.k8s_client = kubernetes.client.ApiClient()
            except Exception as e:
                logger.warning(f"⚠️ Kubernetes client not available: {e}")
        
        app.logger.info("🚀 Deployment manager initialized")
    
    def build_docker_image(self, image_name: str, tag: str = 'latest', 
                          dockerfile_path: str = 'Dockerfile') -> bool:
        """Build Docker image with early returns."""
        if not image_name or not self.docker_client:
            return False
        
        try:
            image, build_logs = self.docker_client.images.build(
                path='.',
                tag=f"{image_name}:{tag}",
                dockerfile=dockerfile_path
            )
            
            logger.info(f"✅ Docker image built: {image_name}:{tag}")
            return True
        except Exception as e:
            logger.error(f"❌ Docker build error: {e}")
            return False
    
    def push_docker_image(self, image_name: str, tag: str = 'latest') -> bool:
        """Push Docker image with early returns."""
        if not image_name or not self.docker_client:
            return False
        
        try:
            self.docker_client.images.push(f"{image_name}:{tag}")
            logger.info(f"✅ Docker image pushed: {image_name}:{tag}")
            return True
        except Exception as e:
            logger.error(f"❌ Docker push error: {e}")
            return False
    
    def deploy_to_kubernetes(self, deployment_config: Dict[str, Any]) -> bool:
        """Deploy to Kubernetes with early returns."""
        if not deployment_config or not self.k8s_client:
            return False
        
        try:
            # Create deployment
            apps_v1 = kubernetes.client.AppsV1Api(self.k8s_client)
            apps_v1.create_namespaced_deployment(
                namespace=deployment_config.get('namespace', 'default'),
                body=deployment_config
            )
            
            logger.info(f"✅ Kubernetes deployment created: {deployment_config.get('metadata', {}).get('name')}")
            return True
        except Exception as e:
            logger.error(f"❌ Kubernetes deployment error: {e}")
            return False
    
    def scale_deployment(self, name: str, replicas: int, namespace: str = 'default') -> bool:
        """Scale deployment with early returns."""
        if not name or replicas < 0 or not self.k8s_client:
            return False
        
        try:
            apps_v1 = kubernetes.client.AppsV1Api(self.k8s_client)
            apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body={'spec': {'replicas': replicas}}
            )
            
            logger.info(f"✅ Deployment scaled: {name} to {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"❌ Scale deployment error: {e}")
            return False
    
    def get_deployment_status(self, name: str, namespace: str = 'default') -> Dict[str, Any]:
        """Get deployment status with early returns."""
        if not name or not self.k8s_client:
            return {}
        
        try:
            apps_v1 = kubernetes.client.AppsV1Api(self.k8s_client)
            deployment = apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
            
            return {
                'name': deployment.metadata.name,
                'namespace': deployment.metadata.namespace,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas,
                'available_replicas': deployment.status.available_replicas,
                'conditions': deployment.status.conditions
            }
        except Exception as e:
            logger.error(f"❌ Get deployment status error: {e}")
            return {}
    
    def rollback_deployment(self, name: str, namespace: str = 'default') -> bool:
        """Rollback deployment with early returns."""
        if not name or not self.k8s_client:
            return False
        
        try:
            apps_v1 = kubernetes.client.AppsV1Api(self.k8s_client)
            apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body={'spec': {'template': {'spec': {'containers': [{'name': name, 'image': 'previous-image'}]}}}}
            )
            
            logger.info(f"✅ Deployment rolled back: {name}")
            return True
        except Exception as e:
            logger.error(f"❌ Rollback deployment error: {e}")
            return False

# Global deployment manager instance
deployment_manager = DeploymentManager()

def init_deployment(app) -> None:
    """Initialize deployment with app."""
    deployment_manager.init_app(app)

def deploy_application(app_name: str, environment: str = 'production') -> bool:
    """Deploy application with early returns."""
    if not app_name:
        return False
    
    try:
        # Build Docker image
        if not deployment_manager.build_docker_image(app_name):
            return False
        
        # Push Docker image
        if not deployment_manager.push_docker_image(app_name):
            return False
        
        # Deploy to Kubernetes
        deployment_config = create_deployment_config(app_name, environment)
        if not deployment_manager.deploy_to_kubernetes(deployment_config):
            return False
        
        logger.info(f"✅ Application deployed: {app_name} to {environment}")
        return True
    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        return False

def create_deployment_config(app_name: str, environment: str = 'production') -> Dict[str, Any]:
    """Create deployment configuration with early returns."""
    if not app_name:
        return {}
    
    return {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': app_name,
            'namespace': environment
        },
        'spec': {
            'replicas': 3,
            'selector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': app_name
                    }
                },
                'spec': {
                    'containers': [{
                        'name': app_name,
                        'image': f"{app_name}:latest",
                        'ports': [{
                            'containerPort': 8000
                        }],
                        'env': [
                            {'name': 'FLASK_ENV', 'value': environment},
                            {'name': 'FLASK_HOST', 'value': '0.0.0.0'},
                            {'name': 'FLASK_PORT', 'value': '8000'}
                        ],
                        'resources': {
                            'requests': {
                                'memory': '1Gi',
                                'cpu': '500m'
                            },
                            'limits': {
                                'memory': '2Gi',
                                'cpu': '1000m'
                            }
                        }
                    }]
                }
            }
        }
    }

def create_service_config(app_name: str, environment: str = 'production') -> Dict[str, Any]:
    """Create service configuration with early returns."""
    if not app_name:
        return {}
    
    return {
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': f"{app_name}-service",
            'namespace': environment
        },
        'spec': {
            'selector': {
                'app': app_name
            },
            'ports': [{
                'port': 80,
                'targetPort': 8000
            }],
            'type': 'LoadBalancer'
        }
    }

def create_ingress_config(app_name: str, domain: str, environment: str = 'production') -> Dict[str, Any]:
    """Create ingress configuration with early returns."""
    if not app_name or not domain:
        return {}
    
    return {
        'apiVersion': 'networking.k8s.io/v1',
        'kind': 'Ingress',
        'metadata': {
            'name': f"{app_name}-ingress",
            'namespace': environment,
            'annotations': {
                'kubernetes.io/ingress.class': 'nginx',
                'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
            }
        },
        'spec': {
            'tls': [{
                'hosts': [domain],
                'secretName': f"{app_name}-tls"
            }],
            'rules': [{
                'host': domain,
                'http': {
                    'paths': [{
                        'path': '/',
                        'pathType': 'Prefix',
                        'backend': {
                            'service': {
                                'name': f"{app_name}-service",
                                'port': {
                                    'number': 80
                                }
                            }
                        }
                    }]
                }
            }]
        }
    }

def create_hpa_config(app_name: str, min_replicas: int = 2, max_replicas: int = 10) -> Dict[str, Any]:
    """Create HPA configuration with early returns."""
    if not app_name or min_replicas < 1 or max_replicas < min_replicas:
        return {}
    
    return {
        'apiVersion': 'autoscaling/v2',
        'kind': 'HorizontalPodAutoscaler',
        'metadata': {
            'name': f"{app_name}-hpa"
        },
        'spec': {
            'scaleTargetRef': {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'name': app_name
            },
            'minReplicas': min_replicas,
            'maxReplicas': max_replicas,
            'metrics': [{
                'type': 'Resource',
                'resource': {
                    'name': 'cpu',
                    'target': {
                        'type': 'Utilization',
                        'averageUtilization': 70
                    }
                }
            }]
        }
    }

def create_configmap_config(app_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create ConfigMap configuration with early returns."""
    if not app_name or not config_data:
        return {}
    
    return {
        'apiVersion': 'v1',
        'kind': 'ConfigMap',
        'metadata': {
            'name': f"{app_name}-config"
        },
        'data': config_data
    }

def create_secret_config(app_name: str, secret_data: Dict[str, str]) -> Dict[str, Any]:
    """Create Secret configuration with early returns."""
    if not app_name or not secret_data:
        return {}
    
    return {
        'apiVersion': 'v1',
        'kind': 'Secret',
        'metadata': {
            'name': f"{app_name}-secret"
        },
        'type': 'Opaque',
        'data': secret_data
    }

def create_persistent_volume_config(app_name: str, size: str = '10Gi') -> Dict[str, Any]:
    """Create PersistentVolume configuration with early returns."""
    if not app_name:
        return {}
    
    return {
        'apiVersion': 'v1',
        'kind': 'PersistentVolume',
        'metadata': {
            'name': f"{app_name}-pv"
        },
        'spec': {
            'capacity': {
                'storage': size
            },
            'accessModes': ['ReadWriteOnce'],
            'persistentVolumeReclaimPolicy': 'Retain',
            'storageClassName': 'standard'
        }
    }

def create_persistent_volume_claim_config(app_name: str, size: str = '10Gi') -> Dict[str, Any]:
    """Create PersistentVolumeClaim configuration with early returns."""
    if not app_name:
        return {}
    
    return {
        'apiVersion': 'v1',
        'kind': 'PersistentVolumeClaim',
        'metadata': {
            'name': f"{app_name}-pvc"
        },
        'spec': {
            'accessModes': ['ReadWriteOnce'],
            'resources': {
                'requests': {
                    'storage': size
                }
            }
        }
    }

def create_monitoring_config(app_name: str) -> Dict[str, Any]:
    """Create monitoring configuration with early returns."""
    if not app_name:
        return {}
    
    return {
        'apiVersion': 'v1',
        'kind': 'ServiceMonitor',
        'metadata': {
            'name': f"{app_name}-monitor",
            'labels': {
                'app': app_name
            }
        },
        'spec': {
            'selector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'endpoints': [{
                'port': 'metrics',
                'path': '/metrics'
            }]
        }
    }

def create_network_policy_config(app_name: str) -> Dict[str, Any]:
    """Create NetworkPolicy configuration with early returns."""
    if not app_name:
        return {}
    
    return {
        'apiVersion': 'networking.k8s.io/v1',
        'kind': 'NetworkPolicy',
        'metadata': {
            'name': f"{app_name}-netpol"
        },
        'spec': {
            'podSelector': {
                'matchLabels': {
                    'app': app_name
                }
            },
            'policyTypes': ['Ingress', 'Egress'],
            'ingress': [{
                'from': [{
                    'namespaceSelector': {
                        'matchLabels': {
                            'name': 'default'
                        }
                    }
                }]
            }],
            'egress': [{
                'to': [{
                    'namespaceSelector': {
                        'matchLabels': {
                            'name': 'default'
                        }
                    }
                }]
            }]
        }
    }

def create_complete_deployment(app_name: str, environment: str = 'production', 
                              domain: str = None, config_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create complete deployment configuration with early returns."""
    if not app_name:
        return {}
    
    deployment_config = {
        'deployment': create_deployment_config(app_name, environment),
        'service': create_service_config(app_name, environment),
        'hpa': create_hpa_config(app_name),
        'configmap': create_configmap_config(app_name, config_data or {}),
        'network_policy': create_network_policy_config(app_name),
        'monitoring': create_monitoring_config(app_name)
    }
    
    if domain:
        deployment_config['ingress'] = create_ingress_config(app_name, domain, environment)
    
    return deployment_config

def deploy_complete_application(app_name: str, environment: str = 'production',
                               domain: str = None, config_data: Dict[str, Any] = None) -> bool:
    """Deploy complete application with early returns."""
    if not app_name:
        return False
    
    try:
        # Create complete deployment configuration
        deployment_config = create_complete_deployment(app_name, environment, domain, config_data)
        
        # Deploy each component
        for component_name, component_config in deployment_config.items():
            if not deployment_manager.deploy_to_kubernetes(component_config):
                logger.error(f"❌ Failed to deploy {component_name}")
                return False
        
        logger.info(f"✅ Complete application deployed: {app_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Complete deployment error: {e}")
        return False

def get_deployment_health(app_name: str, namespace: str = 'default') -> Dict[str, Any]:
    """Get deployment health with early returns."""
    if not app_name:
        return {}
    
    try:
        status = deployment_manager.get_deployment_status(app_name, namespace)
        
        if not status:
            return {'status': 'unknown', 'message': 'Deployment not found'}
        
        ready_replicas = status.get('ready_replicas', 0)
        total_replicas = status.get('replicas', 0)
        
        if ready_replicas == total_replicas:
            return {'status': 'healthy', 'ready_replicas': ready_replicas, 'total_replicas': total_replicas}
        elif ready_replicas > 0:
            return {'status': 'degraded', 'ready_replicas': ready_replicas, 'total_replicas': total_replicas}
        else:
            return {'status': 'unhealthy', 'ready_replicas': ready_replicas, 'total_replicas': total_replicas}
    except Exception as e:
        logger.error(f"❌ Get deployment health error: {e}")
        return {'status': 'error', 'message': str(e)}

def scale_application(app_name: str, replicas: int, namespace: str = 'default') -> bool:
    """Scale application with early returns."""
    if not app_name or replicas < 0:
        return False
    
    return deployment_manager.scale_deployment(app_name, replicas, namespace)

def rollback_application(app_name: str, namespace: str = 'default') -> bool:
    """Rollback application with early returns."""
    if not app_name:
        return False
    
    return deployment_manager.rollback_deployment(app_name, namespace)









