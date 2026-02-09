"""
MOEA Analytics - Sistema de análisis avanzado
==============================================
Análisis estadístico y reportes del sistema MOEA
"""
import json
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


class MOEAAnalytics:
    """Sistema de análisis MOEA"""
    
    def __init__(self, metrics_file: str = "moea_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> List[Dict]:
        """Cargar métricas"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def analyze_performance(self, hours: int = 24) -> Dict:
        """Analizar performance del sistema"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not recent_metrics:
            return {"error": "No hay métricas suficientes"}
        
        # Extraer datos
        processed_counts = [m.get("stats", {}).get("processed_count", 0) for m in recent_metrics]
        queue_sizes = [m.get("queue", {}).get("queue_size", 0) for m in recent_metrics]
        avg_times = [m.get("stats", {}).get("avg_time", 0) for m in recent_metrics if m.get("stats", {}).get("avg_time", 0) > 0]
        health_statuses = [m.get("health", {}).get("status") for m in recent_metrics]
        
        analysis = {
            "period_hours": hours,
            "total_samples": len(recent_metrics),
            "uptime_percentage": (health_statuses.count("ok") / len(health_statuses) * 100) if health_statuses else 0,
            "processed_projects": {
                "total": sum(processed_counts),
                "average": statistics.mean(processed_counts) if processed_counts else 0,
                "max": max(processed_counts) if processed_counts else 0,
                "min": min(processed_counts) if processed_counts else 0,
                "median": statistics.median(processed_counts) if processed_counts else 0
            },
            "queue": {
                "average_size": statistics.mean(queue_sizes) if queue_sizes else 0,
                "max_size": max(queue_sizes) if queue_sizes else 0,
                "min_size": min(queue_sizes) if queue_sizes else 0,
                "median_size": statistics.median(queue_sizes) if queue_sizes else 0
            },
            "performance": {
                "avg_time": statistics.mean(avg_times) if avg_times else 0,
                "min_time": min(avg_times) if avg_times else 0,
                "max_time": max(avg_times) if avg_times else 0,
                "median_time": statistics.median(avg_times) if avg_times else 0,
                "std_dev": statistics.stdev(avg_times) if len(avg_times) > 1 else 0
            }
        }
        
        return analysis
    
    def analyze_trends(self, days: int = 7) -> Dict:
        """Analizar tendencias"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not recent_metrics:
            return {"error": "No hay métricas suficientes"}
        
        # Agrupar por día
        daily_data = defaultdict(lambda: {"processed": [], "queue": [], "time": []})
        
        for metric in recent_metrics:
            date = datetime.fromisoformat(metric["timestamp"]).date()
            daily_data[date.isoformat()]["processed"].append(
                metric.get("stats", {}).get("processed_count", 0)
            )
            daily_data[date.isoformat()]["queue"].append(
                metric.get("queue", {}).get("queue_size", 0)
            )
            if metric.get("stats", {}).get("avg_time", 0) > 0:
                daily_data[date.isoformat()]["time"].append(
                    metric.get("stats", {}).get("avg_time", 0)
                )
        
        trends = {}
        for date, data in sorted(daily_data.items()):
            trends[date] = {
                "avg_processed": statistics.mean(data["processed"]) if data["processed"] else 0,
                "avg_queue": statistics.mean(data["queue"]) if data["queue"] else 0,
                "avg_time": statistics.mean(data["time"]) if data["time"] else 0
            }
        
        return {
            "period_days": days,
            "trends": trends
        }
    
    def generate_report(self, output_file: str = "moea_analytics_report.json") -> str:
        """Generar reporte completo"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "performance_24h": self.analyze_performance(24),
            "performance_7d": self.analyze_performance(168),
            "trends_7d": self.analyze_trends(7),
            "summary": self._generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file
    
    def _generate_summary(self) -> Dict:
        """Generar resumen"""
        if not self.metrics:
            return {"error": "No hay métricas"}
        
        total_samples = len(self.metrics)
        health_ok = sum(1 for m in self.metrics if m.get("health", {}).get("status") == "ok")
        
        return {
            "total_samples": total_samples,
            "uptime_percentage": (health_ok / total_samples * 100) if total_samples > 0 else 0,
            "date_range": {
                "first": self.metrics[0]["timestamp"] if self.metrics else None,
                "last": self.metrics[-1]["timestamp"] if self.metrics else None
            }
        }


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Analytics")
    parser.add_argument(
        '--metrics-file',
        default='moea_metrics.json',
        help='Archivo de métricas'
    )
    parser.add_argument(
        '--performance',
        type=int,
        metavar='HOURS',
        help='Analizar performance de últimas N horas'
    )
    parser.add_argument(
        '--trends',
        type=int,
        metavar='DAYS',
        help='Analizar tendencias de últimos N días'
    )
    parser.add_argument(
        '--report',
        help='Generar reporte completo'
    )
    
    args = parser.parse_args()
    
    analytics = MOEAAnalytics(args.metrics_file)
    
    if args.performance:
        analysis = analytics.analyze_performance(args.performance)
        print(f"\n📊 Performance Analysis ({args.performance}h):")
        print(json.dumps(analysis, indent=2))
    
    if args.trends:
        trends = analytics.analyze_trends(args.trends)
        print(f"\n📈 Trends Analysis ({args.trends}d):")
        print(json.dumps(trends, indent=2))
    
    if args.report:
        report_file = analytics.generate_report(args.report)
        print(f"\n✅ Reporte generado: {report_file}")
    
    if not any([args.performance, args.trends, args.report]):
        # Análisis por defecto
        analysis = analytics.analyze_performance(24)
        print("\n📊 Performance Analysis (24h):")
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()

