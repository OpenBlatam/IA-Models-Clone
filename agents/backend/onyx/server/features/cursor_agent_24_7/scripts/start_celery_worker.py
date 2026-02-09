#!/usr/bin/env python3
"""
Start Celery Worker - Iniciar worker de Celery
==============================================

Script para iniciar workers Celery para procesar tareas en background.
"""

import sys
import os
from pathlib import Path

# Agregar directorio al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.celery_worker import celery_app

if __name__ == "__main__":
    # Configurar worker
    worker = celery_app.Worker(
        loglevel="info",
        concurrency=4,
        queues=["default", "tasks", "heavy", "notifications"]
    )
    
    print("🚀 Starting Celery worker...")
    print(f"   Broker: {celery_app.conf.broker_url}")
    print(f"   Backend: {celery_app.conf.result_backend}")
    print(f"   Queues: default, tasks, heavy, notifications")
    
    worker.start()




