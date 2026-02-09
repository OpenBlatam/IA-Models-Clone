#!/usr/bin/env python3
"""
Script de load testing para Robot Movement AI v2.0
Usa Locust para pruebas de carga
"""

import asyncio
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from locust import HttpUser, task, between
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Warning: locust not installed. Install with: pip install locust")


if LOCUST_AVAILABLE:
    class RobotMovementAIUser(HttpUser):
        """Usuario de Locust para load testing"""
        
        wait_time = between(1, 3)  # Esperar entre 1 y 3 segundos entre tareas
        
        def on_start(self):
            """Ejecutar al inicio de cada usuario"""
            # Health check inicial
            self.client.get("/health")
        
        @task(3)
        def health_check(self):
            """Health check (peso 3)"""
            self.client.get("/health")
        
        @task(2)
        def get_robots(self):
            """Listar robots (peso 2)"""
            self.client.get("/api/v2/robots")
        
        @task(1)
        def move_robot(self):
            """Mover robot (peso 1)"""
            self.client.post(
                "/api/v2/robots/robot-1/move",
                json={
                    "target_x": 0.5,
                    "target_y": 0.3,
                    "target_z": 0.2
                }
            )
        
        @task(1)
        def get_metrics(self):
            """Obtener métricas (peso 1)"""
            self.client.get("/health/metrics")


def run_load_test(host: str = "http://localhost:8010", users: int = 10, spawn_rate: int = 2, duration: str = "1m"):
    """
    Ejecutar load test
    
    Args:
        host: URL del servidor
        users: Número de usuarios concurrentes
        spawn_rate: Tasa de spawn de usuarios
        duration: Duración del test (ej: "1m", "5m")
    """
    if not LOCUST_AVAILABLE:
        print("Error: locust is required for load testing")
        print("Install with: pip install locust")
        return
    
    import subprocess
    
    cmd = [
        "locust",
        "-f", __file__,
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", duration,
        "--headless"
    ]
    
    print(f"Running load test: {users} users, {duration} duration")
    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test for Robot Movement AI")
    parser.add_argument("--host", default="http://localhost:8010", help="Server URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, default=2, help="Spawn rate")
    parser.add_argument("--duration", default="1m", help="Test duration (e.g., '1m', '5m')")
    
    args = parser.parse_args()
    
    run_load_test(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration
    )




