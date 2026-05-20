#!/usr/bin/env python3
"""
Ejemplo de Operaciones Masivas - Bulk Chat
==========================================

Este script muestra cómo usar las operaciones masivas del sistema.
"""

import asyncio
import requests
from typing import List, Dict, Any

BASE_URL = "http://localhost:8006"


def bulk_create_sessions(count: int = 10, initial_messages: List[str] = None) -> List[str]:
    """Crear múltiples sesiones en lote."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/create"
    data = {
        "count": count,
        "initial_messages": initial_messages or ["Hola", "Hello", "Hi"],
        "auto_continue": True,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Creadas {result['created']} sesiones")
    return result.get("session_ids", [])


def bulk_pause_sessions(session_ids: List[str], reason: str = "Bulk pause"):
    """Pausar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/pause"
    data = {
        "session_ids": session_ids,
        "reason": reason,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Pausadas {result['processed']} sesiones en {result['duration']:.2f}s")


def bulk_resume_sessions(session_ids: List[str]):
    """Reanudar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/resume"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Reanudadas {result['processed']} sesiones en {result['duration']:.2f}s")


def bulk_stop_sessions(session_ids: List[str]):
    """Detener múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/stop"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Detenidas {result['processed']} sesiones en {result['duration']:.2f}s")


def bulk_send_messages(session_ids: List[str], message: str):
    """Enviar mensaje a múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/messages/send"
    data = {
        "session_ids": session_ids,
        "message": message,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Enviado a {result['processed']} sesiones en {result['duration']:.2f}s")


def bulk_export_sessions(session_ids: List[str], format: str = "json"):
    """Exportar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/export/sessions"
    data = {
        "session_ids": session_ids,
        "format": format,
        "compress": False,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    job_id = result["job_id"]
    print(f"✅ Export iniciado. Job ID: {job_id}")
    
    # Verificar estado
    import time
    while True:
        status_url = f"{BASE_URL}/api/v1/bulk/export/status/{job_id}"
        status = requests.get(status_url).json()
        
        print(f"📊 Progreso: {status['progress']:.1f}% ({status['processed']}/{status['total']})")
        
        if status["status"] == "completed":
            print(f"✅ Export completado!")
            break
        elif status["status"] == "failed":
            print(f"❌ Export falló: {status.get('errors', [])}")
            break
        
        time.sleep(1)
    
    return job_id


def bulk_analyze_sessions(session_ids: List[str]):
    """Analizar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/analytics/sessions"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Analizadas {result['analyzed']} sesiones")
    return result.get("results", [])


def bulk_cleanup_sessions(days_old: int = 30, dry_run: bool = True):
    """Limpiar sesiones antiguas."""
    url = f"{BASE_URL}/api/v1/bulk/cleanup/sessions"
    data = {
        "days_old": days_old,
        "dry_run": dry_run
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    mode = "DRY RUN" if dry_run else "REAL"
    print(f"✅ {mode}: {result['processed']} sesiones procesadas")
    return result


def main():
    """Ejemplo principal de operaciones masivas."""
    print("=" * 60)
    print("🚀 Ejemplo de Operaciones Masivas - Bulk Chat")
    print("=" * 60)
    print()
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print("✅ Servidor está corriendo\n")
    except Exception as e:
        print(f"❌ Servidor no está disponible: {e}")
        print(f"   Inicia el servidor con: python -m bulk_chat.main --llm-provider mock")
        return
    
    # 1. Crear múltiples sesiones
    print("📝 Creando 10 sesiones en lote...")
    session_ids = bulk_create_sessions(count=10)
    print(f"   Sesiones creadas: {len(session_ids)}\n")
    
    if not session_ids:
        print("❌ No se pudieron crear sesiones")
        return
    
    # Esperar un poco para que generen respuestas
    print("⏳ Esperando respuestas automáticas (5 segundos)...")
    import time
    time.sleep(5)
    print()
    
    # 2. Enviar mensaje a todas las sesiones
    print("📨 Enviando mensaje a todas las sesiones...")
    bulk_send_messages(session_ids, "Mensaje desde operación masiva")
    print()
    
    # 3. Pausar todas las sesiones
    print("⏸️  Pausando todas las sesiones...")
    bulk_pause_sessions(session_ids, "Bulk pause example")
    print()
    
    time.sleep(2)
    
    # 4. Reanudar todas las sesiones
    print("▶️  Reanudando todas las sesiones...")
    bulk_resume_sessions(session_ids)
    print()
    
    time.sleep(3)
    
    # 5. Analizar sesiones
    print("📊 Analizando sesiones...")
    analyses = bulk_analyze_sessions(session_ids[:5])  # Analizar solo las primeras 5
    print(f"   Análisis completados: {len(analyses)}\n")
    
    # 6. Exportar sesiones
    print("📦 Exportando sesiones...")
    export_job_id = bulk_export_sessions(session_ids[:3], format="json")  # Exportar solo 3
    print()
    
    # 7. Limpieza (dry run)
    print("🧹 Limpieza de sesiones antiguas (dry run)...")
    bulk_cleanup_sessions(days_old=30, dry_run=True)
    print()
    
    # 8. Detener todas las sesiones
    print("🛑 Deteniendo todas las sesiones...")
    bulk_stop_sessions(session_ids)
    print()
    
    print("✨ Ejemplo de operaciones masivas completado!")
    print(f"\n💡 Puedes ver más información en: {BASE_URL}/docs")


if __name__ == "__main__":
    main()
















