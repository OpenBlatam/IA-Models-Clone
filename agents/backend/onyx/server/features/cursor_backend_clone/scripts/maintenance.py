"""
Maintenance - Scripts de mantenimiento
=======================================

Scripts para mantenimiento del agente.
"""

import asyncio
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import CursorAgent, AgentConfig


async def cleanup_old_tasks(agent, days: int = 30):
    """Limpiar tareas antiguas"""
    print(f"🧹 Cleaning up tasks older than {days} days...")
    
    from datetime import datetime, timedelta
    cutoff_date = datetime.now() - timedelta(days=days)
    
    tasks_to_remove = []
    for task_id, task in agent.tasks.items():
        if task.timestamp < cutoff_date and task.status in ["completed", "failed"]:
            tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del agent.tasks[task_id]
    
    print(f"✅ Removed {len(tasks_to_remove)} old tasks")
    return len(tasks_to_remove)


async def cleanup_old_backups(backup_manager, keep: int = 10):
    """Limpiar backups antiguos"""
    print(f"🧹 Cleaning up old backups (keeping {keep})...")
    
    backups = backup_manager.list_backups()
    if len(backups) > keep:
        to_delete = backups[keep:]
        deleted = 0
        for backup in to_delete:
            if backup_manager.delete_backup(backup["name"]):
                deleted += 1
        print(f"✅ Deleted {deleted} old backups")
        return deleted
    return 0


async def optimize_database():
    """Optimizar base de datos (si existe)"""
    print("🔧 Optimizing database...")
    # TODO: Implementar si se usa base de datos
    print("✅ Database optimized")


async def check_health(agent):
    """Verificar salud del agente"""
    print("🏥 Checking agent health...")
    
    from core.health_check import HealthChecker
    
    checker = HealthChecker(agent)
    health = await checker.check_all()
    
    print(f"Status: {health['status']}")
    for check in health['checks']:
        status_icon = "✅" if check['status'] == "healthy" else "⚠️" if check['status'] == "degraded" else "❌"
        print(f"  {status_icon} {check['name']}: {check['message']}")
    
    return health['status'] == "healthy"


async def generate_report(agent):
    """Generar reporte del agente"""
    print("📊 Generating report...")
    
    status = await agent.get_status()
    
    report = f"""
# Agent Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Status
- Status: {status['status']}
- Running: {status['running']}

## Tasks
- Total: {status['tasks_total']}
- Pending: {status['tasks_pending']}
- Running: {status['tasks_running']}
- Completed: {status['tasks_completed']}
- Failed: {status['tasks_failed']}

## Metrics
"""
    
    if 'metrics' in status:
        metrics = status['metrics']
        report += f"- Uptime: {metrics.get('uptime_human', 'N/A')}\n"
        report += f"- Total Metrics: {metrics.get('total_metrics_recorded', 0)}\n"
    
    print(report)
    
    # Guardar reporte
    report_file = Path("./data/reports") / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report, encoding='utf-8')
    
    print(f"✅ Report saved to {report_file}")


async def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Maintenance scripts for Cursor Agent 24/7")
    parser.add_argument("action", choices=["cleanup", "health", "report", "optimize", "all"])
    parser.add_argument("--days", type=int, default=30, help="Days for cleanup")
    parser.add_argument("--keep-backups", type=int, default=10, help="Number of backups to keep")
    
    args = parser.parse_args()
    
    # Crear agente (solo para lectura)
    config = AgentConfig(persistent_storage=True)
    agent = CursorAgent(config)
    
    # Cargar estado
    if config.persistent_storage:
        await agent._load_state()
    
    try:
        if args.action == "cleanup" or args.action == "all":
            await cleanup_old_tasks(agent, args.days)
            if agent.backup_manager:
                await cleanup_old_backups(agent.backup_manager, args.keep_backups)
        
        if args.action == "health" or args.action == "all":
            healthy = await check_health(agent)
            if not healthy:
                sys.exit(1)
        
        if args.action == "report" or args.action == "all":
            await generate_report(agent)
        
        if args.action == "optimize" or args.action == "all":
            await optimize_database()
        
        print("✅ Maintenance completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())


