"""
Monitor en tiempo real de la API BUL
Monitorea el estado de la API y genera alertas
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import deque
import sys

BASE_URL = "http://localhost:8000"
CHECK_INTERVAL = 5  # segundos
ALERT_THRESHOLD_RESPONSE_TIME = 1000  # ms
ALERT_THRESHOLD_ERROR_RATE = 0.1  # 10%
HISTORY_SIZE = 100

class Monitor:
    """Monitor de la API en tiempo real."""
    
    def __init__(self):
        self.response_times = deque(maxlen=HISTORY_SIZE)
        self.error_count = 0
        self.total_checks = 0
        self.start_time = datetime.now()
        self.alerts: List[Dict[str, Any]] = []
        self.is_running = False
    
    def check_health(self) -> Dict[str, Any]:
        """Verifica el estado de salud de la API."""
        start = time.time()
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            response_time = (time.time() - start) * 1000  # ms
            
            self.total_checks += 1
            self.response_times.append(response_time)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                self.error_count += 1
                return {
                    "status": "unhealthy",
                    "response_code": response.status_code,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self.error_count += 1
            response_time = (time.time() - start) * 1000
            return {
                "status": "error",
                "error": str(e),
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la API."""
        try:
            response = requests.get(f"{BASE_URL}/api/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calcula métricas del monitoreo."""
        if not self.response_times:
            return {}
        
        response_times_list = list(self.response_times)
        avg_response_time = sum(response_times_list) / len(response_times_list)
        max_response_time = max(response_times_list)
        min_response_time = min(response_times_list)
        
        error_rate = self.error_count / self.total_checks if self.total_checks > 0 else 0
        
        uptime = datetime.now() - self.start_time
        
        return {
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "error_rate": error_rate,
            "total_checks": self.total_checks,
            "error_count": self.error_count,
            "uptime": str(uptime),
            "checks_per_minute": self.total_checks / (uptime.total_seconds() / 60) if uptime.total_seconds() > 0 else 0
        }
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verifica condiciones de alerta."""
        new_alerts = []
        
        if metrics.get("avg_response_time", 0) > ALERT_THRESHOLD_RESPONSE_TIME:
            alert = {
                "type": "HIGH_RESPONSE_TIME",
                "severity": "WARNING",
                "message": f"Tiempo de respuesta promedio alto: {metrics['avg_response_time']:.2f}ms",
                "timestamp": datetime.now().isoformat()
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        if metrics.get("error_rate", 0) > ALERT_THRESHOLD_ERROR_RATE:
            alert = {
                "type": "HIGH_ERROR_RATE",
                "severity": "CRITICAL",
                "message": f"Tasa de error alta: {metrics['error_rate']*100:.1f}%",
                "timestamp": datetime.now().isoformat()
            }
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        return new_alerts
    
    def print_status(self, health: Dict[str, Any], metrics: Dict[str, Any], api_stats: Dict[str, Any]):
        """Imprime el estado actual."""
        status = health.get("status", "unknown")
        status_icon = "✅" if status == "healthy" else "❌"
        
        print(f"\r{status_icon} Status: {status.upper()} | "
              f"Resp: {health.get('response_time', 0):.0f}ms | "
              f"Avg: {metrics.get('avg_response_time', 0):.0f}ms | "
              f"Errors: {self.error_count}/{self.total_checks} | "
              f"Uptime: {metrics.get('uptime', '0:00:00')}        ", end="", flush=True)
    
    def run(self, duration_minutes: int = None):
        """Ejecuta el monitor."""
        self.is_running = True
        end_time = None
        
        if duration_minutes:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        print("🔍 Monitor de API BUL iniciado")
        print(f"Endpoint: {BASE_URL}")
        print(f"Intervalo: {CHECK_INTERVAL}s")
        if duration_minutes:
            print(f"Duración: {duration_minutes} minutos")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while self.is_running:
                if end_time and datetime.now() >= end_time:
                    break
                
                # Verificar salud
                health = self.check_health()
                
                # Obtener estadísticas de API
                api_stats = self.get_stats()
                
                # Calcular métricas
                metrics = self.calculate_metrics()
                
                # Verificar alertas
                alerts = self.check_alerts(metrics)
                
                # Imprimir estado
                self.print_status(health, metrics, api_stats)
                
                # Mostrar alertas
                if alerts:
                    print("\n⚠️  ALERTAS:")
                    for alert in alerts:
                        severity_icon = "🔴" if alert["severity"] == "CRITICAL" else "🟡"
                        print(f"  {severity_icon} {alert['message']}")
                    print()
                
                time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Monitor detenido por el usuario")
        
        finally:
            self.is_running = False
            self.print_final_report()
    
    def print_final_report(self):
        """Imprime reporte final."""
        metrics = self.calculate_metrics()
        uptime = datetime.now() - self.start_time
        
        print("\n" + "="*70)
        print("  REPORTE FINAL DE MONITOREO")
        print("="*70)
        print(f"Tiempo total: {uptime}")
        print(f"Total de checks: {self.total_checks}")
        print(f"Errores: {self.error_count}")
        print(f"Tasa de error: {metrics.get('error_rate', 0)*100:.2f}%")
        print(f"Tiempo promedio de respuesta: {metrics.get('avg_response_time', 0):.2f}ms")
        print(f"Tiempo mínimo: {metrics.get('min_response_time', 0):.2f}ms")
        print(f"Tiempo máximo: {metrics.get('max_response_time', 0):.2f}ms")
        print(f"Checks por minuto: {metrics.get('checks_per_minute', 0):.2f}")
        print(f"Total de alertas: {len(self.alerts)}")
        print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor de API BUL")
    parser.add_argument("--duration", type=int, help="Duración en minutos (opcional)")
    parser.add_argument("--interval", type=int, default=5, help="Intervalo de verificación en segundos")
    
    args = parser.parse_args()
    
    CHECK_INTERVAL = args.interval
    
    monitor = Monitor()
    monitor.run(duration_minutes=args.duration)
































