#!/usr/bin/env python3
"""
Scheduler de Tests
Programa y ejecuta tests en horarios específicos
"""

import sys
import schedule
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json


class TestScheduler:
    """Scheduler de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.running = True
        self.jobs = []
    
    def schedule_daily(self, time_str: str, category: str = 'all'):
        """Programar ejecución diaria"""
        print(f"📅 Programando ejecución diaria a las {time_str}...")
        
        def job():
            print(f"\n🕐 Ejecutando tests programados ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})...")
            subprocess.run(
                ['python', 'run_tests.py', category, '-v'],
                cwd=self.base_path
            )
        
        schedule.every().day.at(time_str).do(job)
        self.jobs.append(('daily', time_str, category))
        print(f"   ✅ Programado: Diario a las {time_str}")
    
    def schedule_hourly(self, category: str = 'unit'):
        """Programar ejecución cada hora"""
        print("📅 Programando ejecución cada hora...")
        
        def job():
            print(f"\n🕐 Ejecutando tests programados ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})...")
            subprocess.run(
                ['python', 'run_tests.py', category, '-q'],
                cwd=self.base_path
            )
        
        schedule.every().hour.do(job)
        self.jobs.append(('hourly', None, category))
        print("   ✅ Programado: Cada hora")
    
    def schedule_weekly(self, day: str, time_str: str, category: str = 'all'):
        """Programar ejecución semanal"""
        print(f"📅 Programando ejecución semanal los {day} a las {time_str}...")
        
        def job():
            print(f"\n🕐 Ejecutando tests programados ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})...")
            subprocess.run(
                ['python', 'run_tests.py', category, '-v'],
                cwd=self.base_path
            )
        
        day_map = {
            'monday': schedule.every().monday,
            'tuesday': schedule.every().tuesday,
            'wednesday': schedule.every().wednesday,
            'thursday': schedule.every().thursday,
            'friday': schedule.every().friday,
            'saturday': schedule.every().saturday,
            'sunday': schedule.every().sunday
        }
        
        if day.lower() in day_map:
            day_map[day.lower()].at(time_str).do(job)
            self.jobs.append(('weekly', f"{day} {time_str}", category))
            print(f"   ✅ Programado: Semanal los {day} a las {time_str}")
        else:
            print(f"   ❌ Día inválido: {day}")
    
    def list_jobs(self):
        """Listar jobs programados"""
        print("\n📋 Jobs programados:")
        for i, (job_type, time_info, category) in enumerate(self.jobs, 1):
            print(f"   {i}. {job_type}: {time_info or 'N/A'} - {category}")
        
        print(f"\n   Total: {len(self.jobs)} jobs")
    
    def run_scheduler(self):
        """Ejecutar scheduler"""
        print("🚀 Iniciando scheduler de tests...")
        print("   Presiona Ctrl+C para detener\n")
        
        self.list_jobs()
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Verificar cada minuto
        except KeyboardInterrupt:
            print("\n\n🛑 Deteniendo scheduler...")
            self.running = False


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scheduler de tests')
    parser.add_argument('--daily', type=str, metavar='TIME',
                       help='Programar ejecución diaria (formato: HH:MM)')
    parser.add_argument('--hourly', action='store_true',
                       help='Programar ejecución cada hora')
    parser.add_argument('--weekly', nargs=2, metavar=('DAY', 'TIME'),
                       help='Programar ejecución semanal (día y hora)')
    parser.add_argument('--category', default='all',
                       help='Categoría de tests')
    parser.add_argument('--list', action='store_true',
                       help='Listar jobs programados')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    scheduler = TestScheduler(args.base_path)
    
    if args.list:
        scheduler.list_jobs()
        return 0
    
    if args.daily:
        scheduler.schedule_daily(args.daily, args.category)
    
    if args.hourly:
        scheduler.schedule_hourly(args.category)
    
    if args.weekly:
        scheduler.schedule_weekly(args.weekly[0], args.weekly[1], args.category)
    
    if scheduler.jobs:
        scheduler.run_scheduler()
    else:
        print("No se programaron jobs. Usa --daily, --hourly o --weekly")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

