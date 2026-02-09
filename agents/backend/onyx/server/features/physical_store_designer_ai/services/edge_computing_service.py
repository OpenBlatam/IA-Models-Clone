"""
Edge Computing Service - Integración con edge computing
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EdgeDeviceType(str, Enum):
    """Tipos de dispositivos edge"""
    IOT_GATEWAY = "iot_gateway"
    EDGE_SERVER = "edge_server"
    MOBILE_DEVICE = "mobile_device"
    EMBEDDED = "embedded"


class EdgeComputingService:
    """Servicio para edge computing"""
    
    def __init__(self):
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.deployments: Dict[str, List[Dict[str, Any]]] = {}
        self.sync_status: Dict[str, Dict[str, Any]] = {}
    
    def register_edge_device(
        self,
        store_id: str,
        device_name: str,
        device_type: EdgeDeviceType,
        location: str,
        capabilities: List[str],
        connection_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Registrar dispositivo edge"""
        
        device_id = f"edge_{store_id}_{len(self.devices.get(store_id, [])) + 1}"
        
        device = {
            "device_id": device_id,
            "store_id": store_id,
            "name": device_name,
            "type": device_type.value,
            "location": location,
            "capabilities": capabilities,
            "connection_info": connection_info,
            "is_online": True,
            "registered_at": datetime.now().isoformat(),
            "last_sync": datetime.now().isoformat()
        }
        
        if store_id not in self.devices:
            self.devices[store_id] = {}
        
        self.devices[store_id][device_id] = device
        
        return device
    
    def deploy_to_edge(
        self,
        device_id: str,
        application_name: str,
        application_code: str,
        dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Desplegar aplicación a dispositivo edge"""
        
        device = self._find_device(device_id)
        
        if not device:
            raise ValueError(f"Dispositivo {device_id} no encontrado")
        
        deployment_id = f"deploy_{device_id}_{len(self.deployments.get(device_id, [])) + 1}"
        
        deployment = {
            "deployment_id": deployment_id,
            "device_id": device_id,
            "application_name": application_name,
            "status": "deployed",
            "deployed_at": datetime.now().isoformat(),
            "dependencies": dependencies or [],
            "note": "En producción, esto desplegaría código real al dispositivo edge"
        }
        
        if device_id not in self.deployments:
            self.deployments[device_id] = []
        
        self.deployments[device_id].append(deployment)
        
        return deployment
    
    def sync_edge_data(
        self,
        device_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sincronizar datos desde edge"""
        
        device = self._find_device(device_id)
        
        if not device:
            raise ValueError(f"Dispositivo {device_id} no encontrado")
        
        sync = {
            "sync_id": f"sync_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "device_id": device_id,
            "data": data,
            "synced_at": datetime.now().isoformat(),
            "status": "success"
        }
        
        device["last_sync"] = datetime.now().isoformat()
        self.sync_status[device_id] = sync
        
        return sync
    
    def get_edge_analytics(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Obtener analytics de dispositivos edge"""
        
        devices = self.devices.get(store_id, {})
        
        analytics = {
            "store_id": store_id,
            "total_devices": len(devices),
            "online_devices": len([d for d in devices.values() if d["is_online"]]),
            "devices_by_type": {},
            "total_deployments": 0,
            "last_syncs": []
        }
        
        for device in devices.values():
            device_type = device["type"]
            analytics["devices_by_type"][device_type] = analytics["devices_by_type"].get(device_type, 0) + 1
            
            deployments = self.deployments.get(device["device_id"], [])
            analytics["total_deployments"] += len(deployments)
            
            if device["last_sync"]:
                analytics["last_syncs"].append({
                    "device": device["name"],
                    "last_sync": device["last_sync"]
                })
        
        return analytics
    
    def _find_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Encontrar dispositivo"""
        for store_devices in self.devices.values():
            if device_id in store_devices:
                return store_devices[device_id]
        return None




