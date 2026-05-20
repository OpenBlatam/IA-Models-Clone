#!/usr/bin/env python3
"""
Start Celery Beat - Iniciar scheduler de Celery
================================================

Script para iniciar Celery Beat para tareas programadas.
"""

import sys
import os
from pathlib import Path

# Agregar directorio al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.celery_worker import celery_app

if __name__ == "__main__":
    # Configurar beat scheduler
    beat = celery_app.Beat(
        loglevel="info"
    )
    
    print("⏰ Starting Celery Beat scheduler...")
    print(f"   Broker: {celery_app.conf.broker_url}")
    
    beat.run()




