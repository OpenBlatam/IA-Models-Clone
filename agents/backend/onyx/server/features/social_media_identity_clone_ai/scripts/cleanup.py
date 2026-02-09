"""
Script para limpiar datos antiguos
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.base import get_db_session
from db.models import GeneratedContentModel, TaskModel, AlertModel
from sqlalchemy import func

def cleanup_old_content(days: int = 90):
    """Limpia contenido generado antiguo"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_session() as db:
        count = db.query(GeneratedContentModel).filter(
            GeneratedContentModel.generated_at < cutoff_date
        ).delete()
        db.commit()
        
        print(f"🗑️  Eliminados {count} contenidos antiguos (>{days} días)")

def cleanup_old_tasks(days: int = 30):
    """Limpia tareas completadas antiguas"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_session() as db:
        count = db.query(TaskModel).filter(
            TaskModel.status == "completed",
            TaskModel.completed_at < cutoff_date
        ).delete()
        db.commit()
        
        print(f"🗑️  Eliminadas {count} tareas completadas antiguas (>{days} días)")

def cleanup_resolved_alerts(days: int = 7):
    """Limpia alertas resueltas antiguas"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    with get_db_session() as db:
        count = db.query(AlertModel).filter(
            AlertModel.resolved_at.isnot(None),
            AlertModel.resolved_at < cutoff_date
        ).delete()
        db.commit()
        
        print(f"🗑️  Eliminadas {count} alertas resueltas antiguas (>{days} días)")

def main():
    """Ejecuta limpieza"""
    print("🧹 Iniciando limpieza de datos antiguos...\n")
    
    try:
        cleanup_old_content(days=90)
        cleanup_old_tasks(days=30)
        cleanup_resolved_alerts(days=7)
        
        print("\n✅ Limpieza completada")
    except Exception as e:
        print(f"❌ Error en limpieza: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




