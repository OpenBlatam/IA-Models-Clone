"""
MOEA Monitor - Monitoreo en tiempo real
========================================
Monitor para observar el estado del sistema MOEA en tiempo real
"""
import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class MOEAMonitor:
    """Monitor del sistema MOEA"""
    
    def __init__(self, base_url: str = "http://localhost:8000", interval: float = 2.0):
        self.base_url = base_url
        self.interval = interval
        self.running = False
        self.stats_history: List[Dict] = []
    
    def get_health(self) -> Optional[Dict]:
        """Obtener estado de salud del servidor"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_stats(self) -> Optional[Dict]:
        """Obtener estadísticas del sistema"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_queue(self) -> Optional[Dict]:
        """Obtener estado de la cola"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/queue", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def format_stats(self, stats: Dict) -> str:
        """Formatear estadísticas para mostrar"""
        lines = []
        lines.append(f"📊 Estadísticas del Sistema")
        lines.append(f"   Proyectos procesados: {stats.get('processed_count', 0)}")
        lines.append(f"   En cola: {stats.get('queue_size', 0)}")
        lines.append(f"   Tiempo promedio: {stats.get('avg_time', 0):.2f}s")
        return "\n".join(lines)
    
    def display_dashboard(self):
        """Mostrar dashboard en tiempo real"""
        import os
        import sys
        
        print("\033[2J\033[H")  # Limpiar pantalla
        print("=" * 70)
        print("MOEA System Monitor - Dashboard en Tiempo Real".center(70))
        print("=" * 70)
        print(f"URL: {self.base_url} | Intervalo: {self.interval}s | Presiona Ctrl+C para salir")
        print("=" * 70)
        print()
        
        # Health check
        health = self.get_health()
        if health:
            print("✅ Servidor: OPERATIVO")
        else:
            print("❌ Servidor: NO DISPONIBLE")
            return
        
        print()
        
        # Estadísticas
        stats = self.get_stats()
        if stats:
            print(self.format_stats(stats))
            self.stats_history.append({
                "timestamp": datetime.now().isoformat(),
                "stats": stats
            })
        
        print()
        
        # Cola
        queue = self.get_queue()
        if queue:
            queue_size = queue.get('queue_size', 0)
            print(f"📋 Cola de Proyectos: {queue_size}")
            if queue_size > 0:
                queue_items = queue.get('queue', [])[:5]
                for i, item in enumerate(queue_items, 1):
                    status = item.get('status', 'unknown')
                    desc = item.get('description', '')[:50]
                    print(f"   {i}. [{status}] {desc}...")
        
        print()
        print("=" * 70)
        print(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def monitor_loop(self):
        """Loop principal de monitoreo"""
        self.running = True
        
        try:
            while self.running:
                self.display_dashboard()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoreo detenido por el usuario")
            self.running = False
    
    def save_history(self, filename: str = "moea_monitor_history.json"):
        """Guardar historial de estadísticas"""
        if self.stats_history:
            with open(filename, 'w') as f:
                json.dump(self.stats_history, f, indent=2)
            print(f"✅ Historial guardado en: {filename}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA System Monitor")
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='URL base de la API'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Intervalo de actualización en segundos'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Guardar historial al salir'
    )
    
    args = parser.parse_args()
    
    monitor = MOEAMonitor(args.url, args.interval)
    
    try:
        monitor.monitor_loop()
    finally:
        if args.save:
            monitor.save_history()


if __name__ == "__main__":
    main()

