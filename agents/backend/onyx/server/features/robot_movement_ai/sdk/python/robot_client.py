"""
SDK de Python para Robot Movement AI API v2.0
Cliente fácil de usar para interactuar con la API
"""

import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not installed. Install with: pip install httpx")


@dataclass
class RobotStatus:
    """Estado de un robot"""
    id: str
    name: str
    status: str
    position: Dict[str, float]
    orientation: Dict[str, float]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class MovementResult:
    """Resultado de un movimiento"""
    movement_id: str
    robot_id: str
    status: str
    duration: Optional[float] = None
    error_message: Optional[str] = None


class RobotClient:
    """Cliente para interactuar con Robot Movement AI API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8010",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Inicializar cliente
        
        Args:
            base_url: URL base de la API
            api_key: API key para autenticación (opcional)
            timeout: Timeout para peticiones en segundos
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    def _get_headers(self) -> Dict[str, str]:
        """Obtener headers para peticiones"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud de la API"""
        response = await self._client.get(
            f"{self.base_url}/health",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def get_robot(self, robot_id: str) -> RobotStatus:
        """Obtener información de un robot"""
        response = await self._client.get(
            f"{self.base_url}/api/v2/robots/{robot_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        data = response.json()
        return RobotStatus(**data)
    
    async def list_robots(self) -> List[RobotStatus]:
        """Listar todos los robots"""
        response = await self._client.get(
            f"{self.base_url}/api/v2/robots",
            headers=self._get_headers()
        )
        response.raise_for_status()
        data = response.json()
        return [RobotStatus(**robot) for robot in data]
    
    async def move_robot(
        self,
        robot_id: str,
        target_x: float,
        target_y: float,
        target_z: float,
        speed: Optional[float] = None
    ) -> MovementResult:
        """Mover robot a posición"""
        payload = {
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z
        }
        if speed:
            payload["speed"] = speed
        
        response = await self._client.post(
            f"{self.base_url}/api/v2/robots/{robot_id}/move",
            json=payload,
            headers=self._get_headers()
        )
        response.raise_for_status()
        data = response.json()
        return MovementResult(**data)
    
    async def get_movement_history(
        self,
        robot_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener historial de movimientos"""
        response = await self._client.get(
            f"{self.base_url}/api/v2/robots/{robot_id}/movements",
            params={"limit": limit},
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def chat_command(self, message: str) -> Dict[str, Any]:
        """Enviar comando mediante chat"""
        response = await self._client.post(
            f"{self.base_url}/api/v2/chat",
            json={"message": message},
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema"""
        response = await self._client.get(
            f"{self.base_url}/health/metrics",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.text
    
    async def close(self):
        """Cerrar cliente"""
        await self._client.aclose()
    
    async def __aenter__(self):
        """Context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()


# Funciones helper para uso síncrono
def create_client(base_url: str = "http://localhost:8010", api_key: Optional[str] = None) -> RobotClient:
    """Crear cliente de forma síncrona"""
    return RobotClient(base_url=base_url, api_key=api_key)


# Ejemplo de uso
if __name__ == "__main__":
    async def main():
        async with RobotClient() as client:
            # Health check
            health = await client.health_check()
            print(f"API Health: {health['status']}")
            
            # Listar robots
            robots = await client.list_robots()
            print(f"Found {len(robots)} robots")
            
            # Mover robot (si existe)
            if robots:
                robot = robots[0]
                result = await client.move_robot(
                    robot.id,
                    target_x=0.5,
                    target_y=0.3,
                    target_z=0.2
                )
                print(f"Movement result: {result.status}")
    
    asyncio.run(main())




