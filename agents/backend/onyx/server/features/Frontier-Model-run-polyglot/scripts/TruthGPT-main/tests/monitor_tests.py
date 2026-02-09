#!/usr/bin/env python3
"""
Monitor de Tests
Monitorea ejecuciones de tests en tiempo real y genera alertas
"""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import signal


class TestMonitor:
    """Monitor de tests en tiempo real"""
    
    def __init__(self, base_path: Path, watch_interval: int = 60):
        self.base_path = Path(base_path)
        self.watch_interval = watch_interval
        self.running = True
        self.stats = {
            'total_runs': 0,
            'successful': 0,
            'failed': 0,
            'last_run': None,
            'history': []
        }
        self.alerts = []
        
        # Configurar handler para Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Manejar señales de interrupción"""
        print("\n\n🛑 Deteniendo monitor...")
        self.running = False
    
    def check_test_health(self) -> Dict:
        """Verificar salud de los tests"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        # Verificar estructura
        try:
            result = subprocess.run(
                ['python', 'validate_structure.py'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                health['status'] = 'warning'
                health['issues'].append('Estructura de tests inválida')
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f'Error validando estructura: {e}')
        
        # Verificar archivos críticos
        critical_files = ['conftest.py', 'run_tests.py', 'pytest.ini']
        for file in critical_files:
            if not (self.base_path / file).exists():
                health['status'] = 'error'
                health['issues'].append(f'Archivo crítico faltante: {file}')
        
        return health
    
    def run_quick_check(self) -> Dict:
        """Ejecutar verificación rápida de tests"""
        print(f"🔍 Ejecutando verificación rápida... ({datetime.now().strftime('%H:%M:%S')})")
        
        start_time = time.time()
        result = subprocess.run(
            ['python', 'run_tests.py', 'unit', '-v', '--tb=no', '-q'],
            cwd=self.base_path,
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.time() - start_time
        
        run_result = {
            'timestamp': datetime.now().isoformat(),
            'exit_code': result.returncode,
            'elapsed': elapsed,
            'success': result.returncode == 0,
            'output_lines': len(result.stdout.splitlines())
        }
        
        # Actualizar estadísticas
        self.stats['total_runs'] += 1
        if run_result['success']:
            self.stats['successful'] += 1
        else:
            self.stats['failed'] += 1
            self._add_alert('test_failure', f"Tests fallaron en {elapsed:.2f}s")
        
        self.stats['last_run'] = run_result
        self.stats['history'].append(run_result)
        
        # Mantener solo últimos 100 runs
        if len(self.stats['history']) > 100:
            self.stats['history'] = self.stats['history'][-100:]
        
        return run_result
    
    def _add_alert(self, alert_type: str, message: str):
        """Agregar alerta"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': 'warning' if alert_type == 'test_failure' else 'info'
        }
        self.alerts.append(alert)
        
        # Mantener solo últimos 50 alertas
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
        
        # Imprimir alerta
        icon = "⚠️" if alert['severity'] == 'warning' else "ℹ️"
        print(f"\n{alert} {alert['message']}")
    
    def check_performance_degradation(self):
        """Verificar degradación de rendimiento"""
        if len(self.stats['history']) < 5:
            return
        
        recent_runs = self.stats['history'][-5:]
        avg_recent = sum(r['elapsed'] for r in recent_runs) / len(recent_runs)
        
        if len(self.stats['history']) >= 10:
            older_runs = self.stats['history'][-10:-5]
            avg_older = sum(r['elapsed'] for r in older_runs) / len(older_runs)
            
            if avg_recent > avg_older * 1.5:  # 50% más lento
                self._add_alert(
                    'performance_degradation',
                    f"Rendimiento degradado: {avg_recent:.2f}s vs {avg_older:.2f}s promedio"
                )
    
    def monitor(self, duration: Optional[int] = None):
        """Monitorear tests continuamente"""
        print("🚀 Iniciando monitor de tests...")
        print(f"   Intervalo: {self.watch_interval}s")
        if duration:
            print(f"   Duración: {duration}s")
        print("   Presiona Ctrl+C para detener\n")
        
        start_time = time.time()
        iteration = 0
        
        while self.running:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"📊 Iteración #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            # Verificar salud
            health = self.check_test_health()
            print(f"🏥 Salud: {health['status'].upper()}")
            if health['issues']:
                for issue in health['issues']:
                    print(f"   ⚠️  {issue}")
            
            # Ejecutar verificación rápida
            run_result = self.run_quick_check()
            status_icon = "✅" if run_result['success'] else "❌"
            print(f"{status_icon} Verificación: {run_result['elapsed']:.2f}s")
            
            # Verificar degradación de rendimiento
            self.check_performance_degradation()
            
            # Mostrar estadísticas
            print(f"\n📈 Estadísticas:")
            print(f"   Total ejecuciones: {self.stats['total_runs']}")
            print(f"   Exitosas: {self.stats['successful']}")
            print(f"   Fallidas: {self.stats['failed']}")
            if self.stats['total_runs'] > 0:
                success_rate = (self.stats['successful'] / self.stats['total_runs']) * 100
                print(f"   Tasa de éxito: {success_rate:.1f}%")
            
            # Verificar duración
            if duration and (time.time() - start_time) >= duration:
                print("\n⏰ Tiempo de monitoreo completado")
                break
            
            # Esperar antes de siguiente iteración
            if self.running:
                print(f"\n⏳ Esperando {self.watch_interval}s hasta próxima verificación...")
                time.sleep(self.watch_interval)
        
        # Resumen final
        self._print_summary()
    
    def _print_summary(self):
        """Imprimir resumen final"""
        print("\n" + "="*60)
        print("📊 RESUMEN FINAL")
        print("="*60)
        print(f"Total ejecuciones: {self.stats['total_runs']}")
        print(f"Exitosas: {self.stats['successful']}")
        print(f"Fallidas: {self.stats['failed']}")
        
        if self.stats['total_runs'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_runs']) * 100
            print(f"Tasa de éxito: {success_rate:.1f}%")
        
        if self.stats['history']:
            avg_time = sum(r['elapsed'] for r in self.stats['history']) / len(self.stats['history'])
            print(f"Tiempo promedio: {avg_time:.2f}s")
        
        if self.alerts:
            print(f"\n⚠️  Alertas generadas: {len(self.alerts)}")
            for alert in self.alerts[-10:]:  # Últimas 10
                print(f"   [{alert['timestamp']}] {alert['message']}")
    
    def save_stats(self, output_path: Path):
        """Guardar estadísticas"""
        data = {
            'stats': self.stats,
            'alerts': self.alerts,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Estadísticas guardadas en: {output_path}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor de tests')
    parser.add_argument('--interval', type=int, default=60,
                       help='Intervalo entre verificaciones (segundos)')
    parser.add_argument('--duration', type=int,
                       help='Duración del monitoreo (segundos)')
    parser.add_argument('--output', type=Path,
                       help='Archivo para guardar estadísticas')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    monitor = TestMonitor(args.base_path, args.interval)
    
    try:
        monitor.monitor(args.duration)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitor interrumpido por usuario")
    finally:
        if args.output:
            monitor.save_stats(args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

