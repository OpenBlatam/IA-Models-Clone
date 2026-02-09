"""
MOEA Scheduler - Programador de tareas
======================================
Sistema para programar tareas automáticas del sistema MOEA
"""
import json
import subprocess
import schedule
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class MOEAScheduler:
    """Programador de tareas MOEA"""
    
    def __init__(self, config_file: str = ".moea_scheduler.json"):
        self.config_file = Path(config_file)
        self.tasks: List[Dict] = []
        self.load_tasks()
    
    def load_tasks(self):
        """Cargar tareas"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.tasks = data.get('tasks', [])
            except:
                self.tasks = []
        else:
            self.tasks = []
    
    def save_tasks(self):
        """Guardar tareas"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "tasks": self.tasks
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_task(
        self,
        name: str,
        command: str,
        schedule_type: str,
        schedule_value: str,
        enabled: bool = True
    ) -> bool:
        """Agregar tarea"""
        task = {
            "id": f"task_{len(self.tasks) + 1}",
            "name": name,
            "command": command,
            "schedule_type": schedule_type,
            "schedule_value": schedule_value,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "run_count": 0
        }
        
        self.tasks.append(task)
        self.save_tasks()
        return True
    
    def remove_task(self, task_id: str) -> bool:
        """Eliminar tarea"""
        self.tasks = [t for t in self.tasks if t['id'] != task_id]
        self.save_tasks()
        return True
    
    def enable_task(self, task_id: str) -> bool:
        """Habilitar tarea"""
        for task in self.tasks:
            if task['id'] == task_id:
                task['enabled'] = True
                self.save_tasks()
                return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """Deshabilitar tarea"""
        for task in self.tasks:
            if task['id'] == task_id:
                task['enabled'] = False
                self.save_tasks()
                return True
        return False
    
    def run_task(self, task: Dict):
        """Ejecutar tarea"""
        print(f"▶️  Ejecutando tarea: {task['name']}")
        
        try:
            result = subprocess.run(
                task['command'],
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            task['last_run'] = datetime.now().isoformat()
            task['run_count'] = task.get('run_count', 0) + 1
            task['last_result'] = {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:500],
                "stderr": result.stderr[:500]
            }
            
            self.save_tasks()
            
            if result.returncode == 0:
                print(f"✅ Tarea completada: {task['name']}")
            else:
                print(f"❌ Tarea falló: {task['name']}")
            
        except Exception as e:
            print(f"❌ Error ejecutando tarea: {e}")
            task['last_run'] = datetime.now().isoformat()
            task['last_result'] = {"success": False, "error": str(e)}
            self.save_tasks()
    
    def setup_schedules(self):
        """Configurar schedules"""
        for task in self.tasks:
            if not task.get('enabled', True):
                continue
            
            schedule_type = task['schedule_type']
            schedule_value = task['schedule_value']
            
            if schedule_type == "interval":
                # Ejecutar cada N segundos
                seconds = int(schedule_value)
                schedule.every(seconds).seconds.do(self.run_task, task)
            
            elif schedule_type == "daily":
                # Ejecutar diariamente a una hora
                schedule.every().day.at(schedule_value).do(self.run_task, task)
            
            elif schedule_type == "hourly":
                # Ejecutar cada hora
                schedule.every().hour.do(self.run_task, task)
            
            elif schedule_type == "weekly":
                # Ejecutar semanalmente
                day, time_str = schedule_value.split('@')
                schedule.every().week.at(time_str).do(self.run_task, task)
    
    def run(self):
        """Ejecutar scheduler"""
        self.setup_schedules()
        
        print("⏰ MOEA Scheduler iniciado")
        print(f"   Tareas programadas: {len([t for t in self.tasks if t.get('enabled', True)])}")
        print("   Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⚠️  Scheduler detenido")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Scheduler")
    subparsers = parser.add_subparsers(dest='command', help='Comandos')
    
    # Comando add
    add_parser = subparsers.add_parser('add', help='Agregar tarea')
    add_parser.add_argument('name', help='Nombre de la tarea')
    add_parser.add_argument('command', help='Comando a ejecutar')
    add_parser.add_argument('--type', choices=['interval', 'daily', 'hourly', 'weekly'], required=True)
    add_parser.add_argument('--value', required=True, help='Valor del schedule')
    
    # Comando list
    list_parser = subparsers.add_parser('list', help='Listar tareas')
    
    # Comando remove
    remove_parser = subparsers.add_parser('remove', help='Eliminar tarea')
    remove_parser.add_argument('task_id', help='ID de la tarea')
    
    # Comando enable/disable
    enable_parser = subparsers.add_parser('enable', help='Habilitar tarea')
    enable_parser.add_argument('task_id', help='ID de la tarea')
    
    disable_parser = subparsers.add_parser('disable', help='Deshabilitar tarea')
    disable_parser.add_argument('task_id', help='ID de la tarea')
    
    # Comando run
    run_parser = subparsers.add_parser('run', help='Ejecutar scheduler')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    scheduler = MOEAScheduler()
    
    if args.command == 'add':
        scheduler.add_task(args.name, args.command, args.type, args.value)
        print(f"✅ Tarea agregada: {args.name}")
    
    elif args.command == 'list':
        tasks = scheduler.tasks
        print(f"\n📋 Tareas programadas: {len(tasks)}\n")
        for task in tasks:
            status = "✅ Habilitada" if task.get('enabled', True) else "❌ Deshabilitada"
            print(f"  {task['id']}: {task['name']} [{status}]")
            print(f"    Comando: {task['command']}")
            print(f"    Schedule: {task['schedule_type']} = {task['schedule_value']}")
            if task.get('last_run'):
                print(f"    Última ejecución: {task['last_run']}")
            print()
    
    elif args.command == 'remove':
        if scheduler.remove_task(args.task_id):
            print(f"✅ Tarea eliminada: {args.task_id}")
        else:
            print(f"❌ Tarea no encontrada: {args.task_id}")
    
    elif args.command == 'enable':
        if scheduler.enable_task(args.task_id):
            print(f"✅ Tarea habilitada: {args.task_id}")
        else:
            print(f"❌ Tarea no encontrada: {args.task_id}")
    
    elif args.command == 'disable':
        if scheduler.disable_task(args.task_id):
            print(f"✅ Tarea deshabilitada: {args.task_id}")
        else:
            print(f"❌ Tarea no encontrada: {args.task_id}")
    
    elif args.command == 'run':
        scheduler.run()


if __name__ == "__main__":
    main()

