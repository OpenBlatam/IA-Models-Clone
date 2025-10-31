"""
Kubernetes Support for Container Orchestration
Sistema Kubernetes para orquestación de contenedores ultra-optimizada
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)


class PodStatus(Enum):
    """Estados de pod"""
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class ServiceType(Enum):
    """Tipos de servicio Kubernetes"""
    CLUSTER_IP = "ClusterIP"
    NODE_PORT = "NodePort"
    LOAD_BALANCER = "LoadBalancer"
    EXTERNAL_NAME = "ExternalName"


class DeploymentStatus(Enum):
    """Estados de deployment"""
    AVAILABLE = "Available"
    PROGRESSING = "Progressing"
    REPLICA_FAILURE = "ReplicaFailure"


@dataclass
class PodInfo:
    """Información de pod"""
    name: str
    namespace: str
    status: PodStatus
    node_name: str
    ip: str
    created_at: float
    restart_count: int
    ready: bool
    containers: List[str]
    labels: Dict[str, str]
    annotations: Dict[str, str]


@dataclass
class ServiceInfo:
    """Información de servicio"""
    name: str
    namespace: str
    type: ServiceType
    cluster_ip: str
    external_ips: List[str]
    ports: List[Dict[str, Any]]
    selector: Dict[str, str]
    created_at: float
    labels: Dict[str, str]


@dataclass
class DeploymentInfo:
    """Información de deployment"""
    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    available_replicas: int
    unavailable_replicas: int
    status: DeploymentStatus
    created_at: float
    labels: Dict[str, str]
    selector: Dict[str, str]


@dataclass
class NodeInfo:
    """Información de nodo"""
    name: str
    status: str
    roles: List[str]
    version: str
    os_image: str
    kernel_version: str
    container_runtime: str
    kubelet_version: str
    capacity: Dict[str, str]
    allocatable: Dict[str, str]
    conditions: List[Dict[str, Any]]
    created_at: float


@dataclass
class NamespaceInfo:
    """Información de namespace"""
    name: str
    status: str
    created_at: float
    labels: Dict[str, str]
    annotations: Dict[str, str]


class KubernetesClient:
    """Cliente Kubernetes"""
    
    def __init__(self, api_server: str = "https://kubernetes.default.svc", 
                 token: Optional[str] = None, ca_cert: Optional[str] = None):
        self.api_server = api_server
        self.token = token
        self.ca_cert = ca_cert
        self.http_client = httpx.AsyncClient(
            base_url=api_server,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            verify=ca_cert if ca_cert else True,
            timeout=30.0
        )
    
    async def get_pods(self, namespace: str = "default") -> List[PodInfo]:
        """Obtener pods"""
        try:
            response = await self.http_client.get(f"/api/v1/namespaces/{namespace}/pods")
            data = response.json()
            
            pods = []
            for item in data.get("items", []):
                pod = PodInfo(
                    name=item["metadata"]["name"],
                    namespace=item["metadata"]["namespace"],
                    status=PodStatus(item["status"]["phase"]),
                    node_name=item["spec"].get("nodeName", ""),
                    ip=item["status"].get("podIP", ""),
                    created_at=time.mktime(time.strptime(
                        item["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    )),
                    restart_count=sum(
                        container.get("restartCount", 0) 
                        for container in item["status"].get("containerStatuses", [])
                    ),
                    ready=all(
                        container.get("ready", False)
                        for container in item["status"].get("containerStatuses", [])
                    ),
                    containers=[
                        container["name"] 
                        for container in item["spec"].get("containers", [])
                    ],
                    labels=item["metadata"].get("labels", {}),
                    annotations=item["metadata"].get("annotations", {})
                )
                pods.append(pod)
            
            return pods
            
        except Exception as e:
            logger.error(f"Error getting pods: {e}")
            return []
    
    async def get_services(self, namespace: str = "default") -> List[ServiceInfo]:
        """Obtener servicios"""
        try:
            response = await self.http_client.get(f"/api/v1/namespaces/{namespace}/services")
            data = response.json()
            
            services = []
            for item in data.get("items", []):
                service = ServiceInfo(
                    name=item["metadata"]["name"],
                    namespace=item["metadata"]["namespace"],
                    type=ServiceType(item["spec"]["type"]),
                    cluster_ip=item["spec"].get("clusterIP", ""),
                    external_ips=item["status"].get("loadBalancer", {}).get("ingress", []),
                    ports=[
                        {
                            "name": port.get("name", ""),
                            "port": port["port"],
                            "target_port": port.get("targetPort", ""),
                            "protocol": port.get("protocol", "TCP")
                        }
                        for port in item["spec"].get("ports", [])
                    ],
                    selector=item["spec"].get("selector", {}),
                    created_at=time.mktime(time.strptime(
                        item["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    )),
                    labels=item["metadata"].get("labels", {}),
                    annotations=item["metadata"].get("annotations", {})
                )
                services.append(service)
            
            return services
            
        except Exception as e:
            logger.error(f"Error getting services: {e}")
            return []
    
    async def get_deployments(self, namespace: str = "default") -> List[DeploymentInfo]:
        """Obtener deployments"""
        try:
            response = await self.http_client.get(f"/apis/apps/v1/namespaces/{namespace}/deployments")
            data = response.json()
            
            deployments = []
            for item in data.get("items", []):
                status = item["status"]
                deployment = DeploymentInfo(
                    name=item["metadata"]["name"],
                    namespace=item["metadata"]["namespace"],
                    replicas=item["spec"]["replicas"],
                    ready_replicas=status.get("readyReplicas", 0),
                    available_replicas=status.get("availableReplicas", 0),
                    unavailable_replicas=status.get("unavailableReplicas", 0),
                    status=DeploymentStatus.AVAILABLE if status.get("availableReplicas", 0) > 0 else DeploymentStatus.REPLICA_FAILURE,
                    created_at=time.mktime(time.strptime(
                        item["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    )),
                    labels=item["metadata"].get("labels", {}),
                    selector=item["spec"]["selector"]["matchLabels"]
                )
                deployments.append(deployment)
            
            return deployments
            
        except Exception as e:
            logger.error(f"Error getting deployments: {e}")
            return []
    
    async def get_nodes(self) -> List[NodeInfo]:
        """Obtener nodos"""
        try:
            response = await self.http_client.get("/api/v1/nodes")
            data = response.json()
            
            nodes = []
            for item in data.get("items", []):
                metadata = item["metadata"]
                status = item["status"]
                node_info = item["status"]["nodeInfo"]
                
                node = NodeInfo(
                    name=metadata["name"],
                    status="Ready" if any(
                        condition["type"] == "Ready" and condition["status"] == "True"
                        for condition in status.get("conditions", [])
                    ) else "NotReady",
                    roles=[
                        label.replace("node-role.kubernetes.io/", "")
                        for label in metadata.get("labels", {}).keys()
                        if label.startswith("node-role.kubernetes.io/")
                    ],
                    version=node_info["kubeletVersion"],
                    os_image=node_info["osImage"],
                    kernel_version=node_info["kernelVersion"],
                    container_runtime=node_info["containerRuntimeVersion"],
                    kubelet_version=node_info["kubeletVersion"],
                    capacity=status.get("capacity", {}),
                    allocatable=status.get("allocatable", {}),
                    conditions=status.get("conditions", []),
                    created_at=time.mktime(time.strptime(
                        metadata["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    ))
                )
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error getting nodes: {e}")
            return []
    
    async def get_namespaces(self) -> List[NamespaceInfo]:
        """Obtener namespaces"""
        try:
            response = await self.http_client.get("/api/v1/namespaces")
            data = response.json()
            
            namespaces = []
            for item in data.get("items", []):
                namespace = NamespaceInfo(
                    name=item["metadata"]["name"],
                    status=item["status"]["phase"],
                    created_at=time.mktime(time.strptime(
                        item["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ"
                    )),
                    labels=item["metadata"].get("labels", {}),
                    annotations=item["metadata"].get("annotations", {})
                )
                namespaces.append(namespace)
            
            return namespaces
            
        except Exception as e:
            logger.error(f"Error getting namespaces: {e}")
            return []
    
    async def scale_deployment(self, name: str, namespace: str, replicas: int) -> bool:
        """Escalar deployment"""
        try:
            patch_data = {
                "spec": {
                    "replicas": replicas
                }
            }
            
            response = await self.http_client.patch(
                f"/apis/apps/v1/namespaces/{namespace}/deployments/{name}",
                json=patch_data,
                headers={"Content-Type": "application/merge-patch+json"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error scaling deployment: {e}")
            return False
    
    async def delete_pod(self, name: str, namespace: str) -> bool:
        """Eliminar pod"""
        try:
            response = await self.http_client.delete(f"/api/v1/namespaces/{namespace}/pods/{name}")
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error deleting pod: {e}")
            return False
    
    async def get_pod_logs(self, name: str, namespace: str, container: Optional[str] = None) -> str:
        """Obtener logs de pod"""
        try:
            url = f"/api/v1/namespaces/{namespace}/pods/{name}/log"
            if container:
                url += f"?container={container}"
            
            response = await self.http_client.get(url)
            return response.text
            
        except Exception as e:
            logger.error(f"Error getting pod logs: {e}")
            return ""
    
    async def close(self):
        """Cerrar cliente"""
        await self.http_client.aclose()


class KubernetesManager:
    """Manager de Kubernetes"""
    
    def __init__(self):
        self.client: Optional[KubernetesClient] = None
        self.is_connected = False
        self.cluster_info: Dict[str, Any] = {}
    
    async def connect(self, api_server: str = "https://kubernetes.default.svc",
                     token: Optional[str] = None, ca_cert: Optional[str] = None):
        """Conectar a cluster Kubernetes"""
        try:
            self.client = KubernetesClient(api_server, token, ca_cert)
            
            # Verificar conexión
            response = await self.client.http_client.get("/api/v1")
            if response.status_code == 200:
                self.is_connected = True
                self.cluster_info = response.json()
                logger.info("Connected to Kubernetes cluster")
            else:
                raise Exception(f"Failed to connect to Kubernetes: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error connecting to Kubernetes: {e}")
            raise
    
    async def disconnect(self):
        """Desconectar de cluster"""
        try:
            if self.client:
                await self.client.close()
                self.client = None
                self.is_connected = False
                logger.info("Disconnected from Kubernetes cluster")
        except Exception as e:
            logger.error(f"Error disconnecting from Kubernetes: {e}")
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cluster"""
        if not self.client or not self.is_connected:
            return {"error": "Not connected to Kubernetes cluster"}
        
        try:
            # Obtener información básica
            namespaces = await self.client.get_namespaces()
            nodes = await self.client.get_nodes()
            
            # Obtener pods de todos los namespaces
            all_pods = []
            for namespace in namespaces:
                pods = await self.client.get_pods(namespace.name)
                all_pods.extend(pods)
            
            # Obtener servicios de todos los namespaces
            all_services = []
            for namespace in namespaces:
                services = await self.client.get_services(namespace.name)
                all_services.extend(services)
            
            # Obtener deployments de todos los namespaces
            all_deployments = []
            for namespace in namespaces:
                deployments = await self.client.get_deployments(namespace.name)
                all_deployments.extend(deployments)
            
            # Calcular estadísticas
            total_pods = len(all_pods)
            running_pods = sum(1 for pod in all_pods if pod.status == PodStatus.RUNNING)
            failed_pods = sum(1 for pod in all_pods if pod.status == PodStatus.FAILED)
            
            total_nodes = len(nodes)
            ready_nodes = sum(1 for node in nodes if node.status == "Ready")
            
            total_services = len(all_services)
            total_deployments = len(all_deployments)
            
            return {
                "cluster_info": self.cluster_info,
                "namespaces": {
                    "total": len(namespaces),
                    "list": [ns.name for ns in namespaces]
                },
                "nodes": {
                    "total": total_nodes,
                    "ready": ready_nodes,
                    "not_ready": total_nodes - ready_nodes,
                    "list": [
                        {
                            "name": node.name,
                            "status": node.status,
                            "roles": node.roles,
                            "version": node.version
                        }
                        for node in nodes
                    ]
                },
                "pods": {
                    "total": total_pods,
                    "running": running_pods,
                    "failed": failed_pods,
                    "pending": sum(1 for pod in all_pods if pod.status == PodStatus.PENDING),
                    "succeeded": sum(1 for pod in all_pods if pod.status == PodStatus.SUCCEEDED)
                },
                "services": {
                    "total": total_services,
                    "cluster_ip": sum(1 for svc in all_services if svc.type == ServiceType.CLUSTER_IP),
                    "load_balancer": sum(1 for svc in all_services if svc.type == ServiceType.LOAD_BALANCER),
                    "node_port": sum(1 for svc in all_services if svc.type == ServiceType.NODE_PORT)
                },
                "deployments": {
                    "total": total_deployments,
                    "available": sum(1 for dep in all_deployments if dep.status == DeploymentStatus.AVAILABLE),
                    "replica_failure": sum(1 for dep in all_deployments if dep.status == DeploymentStatus.REPLICA_FAILURE)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster stats: {e}")
            return {"error": str(e)}
    
    async def get_namespace_resources(self, namespace: str) -> Dict[str, Any]:
        """Obtener recursos de namespace específico"""
        if not self.client or not self.is_connected:
            return {"error": "Not connected to Kubernetes cluster"}
        
        try:
            pods = await self.client.get_pods(namespace)
            services = await self.client.get_services(namespace)
            deployments = await self.client.get_deployments(namespace)
            
            return {
                "namespace": namespace,
                "pods": [
                    {
                        "name": pod.name,
                        "status": pod.status.value,
                        "node_name": pod.node_name,
                        "ip": pod.ip,
                        "ready": pod.ready,
                        "restart_count": pod.restart_count,
                        "containers": pod.containers
                    }
                    for pod in pods
                ],
                "services": [
                    {
                        "name": service.name,
                        "type": service.type.value,
                        "cluster_ip": service.cluster_ip,
                        "ports": service.ports,
                        "selector": service.selector
                    }
                    for service in services
                ],
                "deployments": [
                    {
                        "name": deployment.name,
                        "replicas": deployment.replicas,
                        "ready_replicas": deployment.ready_replicas,
                        "available_replicas": deployment.available_replicas,
                        "status": deployment.status.value
                    }
                    for deployment in deployments
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting namespace resources: {e}")
            return {"error": str(e)}
    
    async def scale_deployment(self, name: str, namespace: str, replicas: int) -> Dict[str, Any]:
        """Escalar deployment"""
        if not self.client or not self.is_connected:
            return {"error": "Not connected to Kubernetes cluster"}
        
        try:
            success = await self.client.scale_deployment(name, namespace, replicas)
            return {
                "success": success,
                "deployment": name,
                "namespace": namespace,
                "replicas": replicas
            }
        except Exception as e:
            logger.error(f"Error scaling deployment: {e}")
            return {"error": str(e)}
    
    async def get_pod_logs(self, name: str, namespace: str, container: Optional[str] = None) -> Dict[str, Any]:
        """Obtener logs de pod"""
        if not self.client or not self.is_connected:
            return {"error": "Not connected to Kubernetes cluster"}
        
        try:
            logs = await self.client.get_pod_logs(name, namespace, container)
            return {
                "pod": name,
                "namespace": namespace,
                "container": container,
                "logs": logs
            }
        except Exception as e:
            logger.error(f"Error getting pod logs: {e}")
            return {"error": str(e)}


# Instancia global del manager Kubernetes
kubernetes_manager = KubernetesManager()


# Router para endpoints Kubernetes
kubernetes_router = APIRouter()


@kubernetes_router.post("/kubernetes/connect")
async def connect_kubernetes_endpoint(connection_data: dict):
    """Conectar a cluster Kubernetes"""
    try:
        api_server = connection_data.get("api_server", "https://kubernetes.default.svc")
        token = connection_data.get("token")
        ca_cert = connection_data.get("ca_cert")
        
        await kubernetes_manager.connect(api_server, token, ca_cert)
        
        return {
            "message": "Connected to Kubernetes cluster successfully",
            "api_server": api_server,
            "cluster_info": kubernetes_manager.cluster_info
        }
        
    except Exception as e:
        logger.error(f"Error connecting to Kubernetes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to Kubernetes: {str(e)}")


@kubernetes_router.post("/kubernetes/disconnect")
async def disconnect_kubernetes_endpoint():
    """Desconectar de cluster Kubernetes"""
    try:
        await kubernetes_manager.disconnect()
        return {"message": "Disconnected from Kubernetes cluster successfully"}
    except Exception as e:
        logger.error(f"Error disconnecting from Kubernetes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disconnect from Kubernetes: {str(e)}")


@kubernetes_router.get("/kubernetes/stats")
async def get_kubernetes_stats_endpoint():
    """Obtener estadísticas del cluster Kubernetes"""
    try:
        stats = await kubernetes_manager.get_cluster_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting Kubernetes stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Kubernetes stats: {str(e)}")


@kubernetes_router.get("/kubernetes/namespaces")
async def get_kubernetes_namespaces_endpoint():
    """Obtener namespaces de Kubernetes"""
    try:
        if not kubernetes_manager.client or not kubernetes_manager.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to Kubernetes cluster")
        
        namespaces = await kubernetes_manager.client.get_namespaces()
        return {
            "namespaces": [
                {
                    "name": ns.name,
                    "status": ns.status,
                    "created_at": ns.created_at,
                    "labels": ns.labels
                }
                for ns in namespaces
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Kubernetes namespaces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Kubernetes namespaces: {str(e)}")


@kubernetes_router.get("/kubernetes/namespaces/{namespace}/resources")
async def get_namespace_resources_endpoint(namespace: str):
    """Obtener recursos de namespace específico"""
    try:
        resources = await kubernetes_manager.get_namespace_resources(namespace)
        return resources
    except Exception as e:
        logger.error(f"Error getting namespace resources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get namespace resources: {str(e)}")


@kubernetes_router.get("/kubernetes/nodes")
async def get_kubernetes_nodes_endpoint():
    """Obtener nodos de Kubernetes"""
    try:
        if not kubernetes_manager.client or not kubernetes_manager.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to Kubernetes cluster")
        
        nodes = await kubernetes_manager.client.get_nodes()
        return {
            "nodes": [
                {
                    "name": node.name,
                    "status": node.status,
                    "roles": node.roles,
                    "version": node.version,
                    "os_image": node.os_image,
                    "kernel_version": node.kernel_version,
                    "container_runtime": node.container_runtime,
                    "capacity": node.capacity,
                    "allocatable": node.allocatable
                }
                for node in nodes
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Kubernetes nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Kubernetes nodes: {str(e)}")


@kubernetes_router.post("/kubernetes/deployments/{name}/scale")
async def scale_deployment_endpoint(name: str, scale_data: dict):
    """Escalar deployment"""
    try:
        namespace = scale_data.get("namespace", "default")
        replicas = scale_data.get("replicas", 1)
        
        if replicas < 0:
            raise HTTPException(status_code=400, detail="Replicas must be non-negative")
        
        result = await kubernetes_manager.scale_deployment(name, namespace, replicas)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scaling deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scale deployment: {str(e)}")


@kubernetes_router.get("/kubernetes/pods/{name}/logs")
async def get_pod_logs_endpoint(name: str, namespace: str = "default", container: Optional[str] = None):
    """Obtener logs de pod"""
    try:
        result = await kubernetes_manager.get_pod_logs(name, namespace, container)
        return result
    except Exception as e:
        logger.error(f"Error getting pod logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pod logs: {str(e)}")


@kubernetes_router.delete("/kubernetes/pods/{name}")
async def delete_pod_endpoint(name: str, namespace: str = "default"):
    """Eliminar pod"""
    try:
        if not kubernetes_manager.client or not kubernetes_manager.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to Kubernetes cluster")
        
        success = await kubernetes_manager.client.delete_pod(name, namespace)
        
        if success:
            return {"message": f"Pod {name} deleted successfully", "pod": name, "namespace": namespace}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete pod")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting pod: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete pod: {str(e)}")


# Funciones de utilidad para integración
async def connect_kubernetes(api_server: str = "https://kubernetes.default.svc",
                           token: Optional[str] = None, ca_cert: Optional[str] = None):
    """Conectar a Kubernetes"""
    await kubernetes_manager.connect(api_server, token, ca_cert)


async def disconnect_kubernetes():
    """Desconectar de Kubernetes"""
    await kubernetes_manager.disconnect()


async def get_kubernetes_cluster_stats() -> Dict[str, Any]:
    """Obtener estadísticas del cluster"""
    return await kubernetes_manager.get_cluster_stats()


async def scale_kubernetes_deployment(name: str, namespace: str, replicas: int) -> Dict[str, Any]:
    """Escalar deployment"""
    return await kubernetes_manager.scale_deployment(name, namespace, replicas)


def is_kubernetes_connected() -> bool:
    """Verificar si está conectado a Kubernetes"""
    return kubernetes_manager.is_connected


logger.info("Kubernetes support module loaded successfully")

