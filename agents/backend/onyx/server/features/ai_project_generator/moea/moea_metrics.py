"""
MOEA Metrics Collector - Recolector de métricas
==============================================
Recolecta y analiza métricas del sistema MOEA
"""
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict


class MOEAMetricsCollector:
    """Recolector de métricas MOEA"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics_history: List[Dict] = []
        self.metrics_file = Path("moea_metrics.json")
    
    def collect_metrics(self) -> Dict:
        """Recolectar métricas actuales"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "health": self._check_health(),
            "stats": self._get_stats(),
            "queue": self._get_queue()
        }
        
        self.metrics_history.append(metrics)
        self._save_metrics()
        
        return metrics
    
    def _check_health(self) -> Dict:
        """Verificar salud"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return {
                "status": "ok" if response.status_code == 200 else "error",
                "code": response.status_code
            }
        except:
            return {"status": "error", "code": 0}
    
    def _get_stats(self) -> Dict:
        """Obtener estadísticas"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def _get_queue(self) -> Dict:
        """Obtener cola"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/queue", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def _save_metrics(self):
        """Guardar métricas"""
        # Mantener solo últimas 1000 entradas
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def load_metrics(self) -> List[Dict]:
        """Cargar métricas históricas"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
            except:
                self.metrics_history = []
        return self.metrics_history
    
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Obtener resumen de métricas"""
        self.load_metrics()
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not recent_metrics:
            return {"error": "No hay métricas recientes"}
        
        summary = {
            "period_hours": hours,
            "total_samples": len(recent_metrics),
            "health": {
                "ok_count": sum(1 for m in recent_metrics if m.get("health", {}).get("status") == "ok"),
                "error_count": sum(1 for m in recent_metrics if m.get("health", {}).get("status") == "error")
            },
            "stats": {
                "avg_processed": sum(m.get("stats", {}).get("processed_count", 0) for m in recent_metrics) / len(recent_metrics),
                "avg_queue_size": sum(m.get("queue", {}).get("queue_size", 0) for m in recent_metrics) / len(recent_metrics),
                "avg_time": sum(m.get("stats", {}).get("avg_time", 0) for m in recent_metrics) / len(recent_metrics)
            }
        }
        
        return summary
    
    def export_metrics(self, output_file: str = "moea_metrics_export.json"):
        """Exportar métricas"""
        self.load_metrics()
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_metrics": len(self.metrics_history),
            "metrics": self.metrics_history
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_file


def main():
    """Función principal"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="MOEA Metrics Collector")
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='URL base de la API'
    )
    parser.add_argument(
        '--collect',
        action='store_true',
        help='Recolectar métricas ahora'
    )
    parser.add_argument(
        '--summary',
        type=int,
        metavar='HOURS',
        help='Mostrar resumen de últimas N horas'
    )
    parser.add_argument(
        '--export',
        help='Exportar métricas a archivo'
    )
    parser.add_argument(
        '--continuous',
        type=int,
        metavar='INTERVAL',
        help='Recolectar continuamente cada N segundos'
    )
    
    args = parser.parse_args()
    
    collector = MOEAMetricsCollector(args.url)
    
    if args.collect:
        metrics = collector.collect_metrics()
        print("✅ Métricas recolectadas")
        print(json.dumps(metrics, indent=2))
    
    if args.summary:
        summary = collector.get_metrics_summary(args.summary)
        print(f"\n📊 Resumen de últimas {args.summary} horas:")
        print(json.dumps(summary, indent=2))
    
    if args.export:
        output = collector.export_metrics(args.export)
        print(f"✅ Métricas exportadas a: {output}")
    
    if args.continuous:
        print(f"🔄 Recolectando métricas cada {args.continuous} segundos...")
        print("   Presiona Ctrl+C para detener\n")
        try:
            while True:
                collector.collect_metrics()
                print(f"✅ Métricas recolectadas: {datetime.now().strftime('%H:%M:%S')}")
                time.sleep(args.continuous)
        except KeyboardInterrupt:
            print("\n\n⚠️  Recolección detenida")


if __name__ == "__main__":
    main()

