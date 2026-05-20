"""
Visualizador de Métricas en Tiempo Real
Crea gráficos y visualizaciones de las métricas
"""

import requests
import time
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, List
import sys

BASE_URL = "http://localhost:8000"

class MetricsVisualizer:
    """Visualizador de métricas."""
    
    def __init__(self):
        self.response_times = deque(maxlen=100)
        self.error_counts = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.start_time = datetime.now()
    
    def collect_metrics(self, duration_seconds: int = 60):
        """Recolecta métricas durante un período."""
        print("📊 Recolectando métricas...")
        print(f"   Duración: {duration_seconds} segundos\n")
        
        end_time = datetime.now() + timedelta(seconds=duration_seconds)
        
        try:
            while datetime.now() < end_time:
                try:
                    start = time.time()
                    response = requests.get(f"{BASE_URL}/api/health", timeout=5)
                    response_time = (time.time() - start) * 1000
                    
                    self.response_times.append(response_time)
                    self.error_counts.append(0 if response.status_code == 200 else 1)
                    self.timestamps.append(datetime.now())
                    
                    avg = sum(self.response_times) / len(self.response_times) if self.response_times else 0
                    print(f"\r⏱ Avg: {avg:.0f}ms | "
                          f"Current: {response_time:.0f}ms | "
                          f"Errors: {sum(self.error_counts)}/{len(self.error_counts)}        ", 
                          end="", flush=True)
                    
                    time.sleep(1)
                except Exception as e:
                    self.error_counts.append(1)
                    self.response_times.append(0)
                    self.timestamps.append(datetime.now())
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹ Recolección detenida")
    
    def generate_text_chart(self, data: List[float], width: int = 50, height: int = 10) -> str:
        """Genera un gráfico de texto."""
        if not data:
            return "No hay datos"
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val > min_val else 1
        
        chart = []
        for y in range(height, 0, -1):
            line = ""
            threshold = min_val + (range_val * (height - y) / height)
            for x in range(width):
                if x < len(data):
                    if data[x] >= threshold:
                        line += "█"
                    else:
                        line += " "
                else:
                    line += " "
            chart.append(line)
        
        return "\n".join(chart)
    
    def print_visualization(self):
        """Imprime visualización de métricas."""
        if not self.response_times:
            print("No hay datos para visualizar")
            return
        
        print("\n" + "="*70)
        print("  📊 VISUALIZACIÓN DE MÉTRICAS")
        print("="*70 + "\n")
        
        times_list = list(self.response_times)
        avg_time = sum(times_list) / len(times_list) if times_list else 0
        min_time = min(times_list) if times_list else 0
        max_time = max(times_list) if times_list else 0
        
        error_rate = sum(self.error_counts) / len(self.error_counts) * 100 if self.error_counts else 0
        
        print(f"📈 Estadísticas:")
        print(f"   Tiempo promedio: {avg_time:.2f}ms")
        print(f"   Tiempo mínimo: {min_time:.2f}ms")
        print(f"   Tiempo máximo: {max_time:.2f}ms")
        print(f"   Tasa de error: {error_rate:.2f}%")
        print(f"   Total de muestras: {len(times_list)}\n")
        
        print("📉 Gráfico de Tiempo de Respuesta:")
        print("-" * 70)
        chart = self.generate_text_chart(times_list, width=60, height=15)
        print(chart)
        print("-" * 70)
        print(f"   Min: {min_time:.0f}ms  Max: {max_time:.0f}ms  Avg: {avg_time:.0f}ms")
        print()
        
        if sum(self.error_counts) > 0:
            print("❌ Gráfico de Errores:")
            print("-" * 70)
            error_chart = self.generate_text_chart(list(self.error_counts), width=60, height=5)
            print(error_chart)
            print("-" * 70)
            print()
        
        print("⏱ Timeline:")
        print("-" * 70)
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"   Duración total: {duration:.0f}s")
        print(f"   Muestras por minuto: {len(times_list) / (duration / 60) if duration > 0 else 0:.1f}")
        print()
        
        print("="*70)

def run_visualizer(duration_seconds: int = 60):
    """Ejecuta el visualizador."""
    visualizer = MetricsVisualizer()
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Servidor no disponible")
            return
    except:
        print("❌ No se puede conectar al servidor")
        return
    
    print("📊 Visualizador de Métricas - API BUL")
    print("="*70)
    
    visualizer.collect_metrics(duration_seconds)
    visualizer.print_visualization()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualizador de Métricas")
    parser.add_argument("--duration", type=int, default=60, help="Duración en segundos")
    
    args = parser.parse_args()
    
    run_visualizer(args.duration)
































