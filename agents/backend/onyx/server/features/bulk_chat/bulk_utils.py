#!/usr/bin/env python3
"""
Utilidades para Operaciones Masivas - Bulk Chat
===============================================

Script CLI para ejecutar operaciones masivas fácilmente.
"""

import sys
import argparse
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

BASE_URL = "http://localhost:8006"


def print_result(result: Dict[str, Any], title: str = "Resultado"):
    """Imprimir resultado formateado."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def bulk_create(count: int, messages: Optional[List[str]] = None):
    """Crear múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/create"
    data = {
        "count": count,
        "initial_messages": messages or ["Hola"],
        "auto_continue": True,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result, f"✅ Creadas {result['created']} sesiones")
    return result.get("session_ids", [])


def bulk_pause(session_ids: List[str], reason: str = "Bulk pause"):
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
    
    print_result(result, f"✅ Pausadas {result['processed']} sesiones")
    return result


def bulk_resume(session_ids: List[str]):
    """Reanudar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/resume"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result, f"✅ Reanudadas {result['processed']} sesiones")
    return result


def bulk_stop(session_ids: List[str]):
    """Detener múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/stop"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result, f"✅ Detenidas {result['processed']} sesiones")
    return result


def bulk_delete(session_ids: List[str]):
    """Eliminar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/sessions/delete"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result, f"✅ Eliminadas {result['processed']} sesiones")
    return result


def bulk_export(session_ids: List[str], format: str = "json"):
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
    
    # Monitorear progreso
    import time
    while True:
        status_url = f"{BASE_URL}/api/v1/bulk/export/status/{job_id}"
        status = requests.get(status_url).json()
        
        print(f"📊 Progreso: {status['progress']:.1f}% ({status['processed']}/{status['total']})")
        
        if status["status"] == "completed":
            print(f"✅ Export completado!")
            break
        elif status["status"] == "failed":
            print(f"❌ Export falló")
            break
        
        time.sleep(1)
    
    return job_id


def bulk_analyze(session_ids: List[str]):
    """Analizar múltiples sesiones."""
    url = f"{BASE_URL}/api/v1/bulk/analytics/sessions"
    data = {
        "session_ids": session_ids,
        "parallel": True
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result, f"✅ Analizadas {result['analyzed']} sesiones")
    return result


def bulk_cleanup(days_old: int = 30, dry_run: bool = True):
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
    print_result(result, f"✅ {mode}: {result['processed']} sesiones procesadas")
    return result


def bulk_test_load(concurrent: int = 100, duration: int = 60):
    """Ejecutar test de carga."""
    url = f"{BASE_URL}/api/v1/bulk/testing/load-test"
    data = {
        "concurrent_sessions": concurrent,
        "duration": duration,
        "operations_per_session": 10
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result.get("results", {}), "📊 Resultados del Test de Carga")
    return result


def bulk_test_stress(max_sessions: int = 1000, ramp_up: int = 60):
    """Ejecutar test de estrés."""
    url = f"{BASE_URL}/api/v1/bulk/testing/stress-test"
    data = {
        "max_sessions": max_sessions,
        "ramp_up_seconds": ramp_up
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    result = response.json()
    
    print_result(result.get("results", {}), "📊 Resultados del Test de Estrés")
    return result


def check_server():
    """Verificar que el servidor esté corriendo."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk Chat - Utilidades para Operaciones Masivas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python bulk_utils.py create --count 10
  python bulk_utils.py pause --session-ids id1 id2 id3
  python bulk_utils.py export --session-ids id1 id2 --format json
  python bulk_utils.py cleanup --days 30 --dry-run
  python bulk_utils.py test-load --concurrent 100 --duration 60
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Create
    create_parser = subparsers.add_parser('create', help='Crear múltiples sesiones')
    create_parser.add_argument('--count', type=int, default=10, help='Número de sesiones')
    create_parser.add_argument('--messages', nargs='+', help='Mensajes iniciales')
    
    # Pause
    pause_parser = subparsers.add_parser('pause', help='Pausar sesiones')
    pause_parser.add_argument('--session-ids', nargs='+', required=True, help='IDs de sesiones')
    pause_parser.add_argument('--reason', default='Bulk pause', help='Razón')
    
    # Resume
    resume_parser = subparsers.add_parser('resume', help='Reanudar sesiones')
    resume_parser.add_argument('--session-ids', nargs='+', required=True, help='IDs de sesiones')
    
    # Stop
    stop_parser = subparsers.add_parser('stop', help='Detener sesiones')
    stop_parser.add_argument('--session-ids', nargs='+', required=True, help='IDs de sesiones')
    
    # Delete
    delete_parser = subparsers.add_parser('delete', help='Eliminar sesiones')
    delete_parser.add_argument('--session-ids', nargs='+', required=True, help='IDs de sesiones')
    
    # Export
    export_parser = subparsers.add_parser('export', help='Exportar sesiones')
    export_parser.add_argument('--session-ids', nargs='+', required=True, help='IDs de sesiones')
    export_parser.add_argument('--format', default='json', choices=['json', 'markdown', 'csv', 'html', 'txt'], help='Formato')
    
    # Analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analizar sesiones')
    analyze_parser.add_argument('--session-ids', nargs='+', required=True, help='IDs de sesiones')
    
    # Cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Limpiar sesiones antiguas')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Días de antigüedad')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Ejecutar sin eliminar')
    
    # Test Load
    load_parser = subparsers.add_parser('test-load', help='Test de carga')
    load_parser.add_argument('--concurrent', type=int, default=100, help='Sesiones concurrentes')
    load_parser.add_argument('--duration', type=int, default=60, help='Duración en segundos')
    
    # Test Stress
    stress_parser = subparsers.add_parser('test-stress', help='Test de estrés')
    stress_parser.add_argument('--max-sessions', type=int, default=1000, help='Máximo de sesiones')
    stress_parser.add_argument('--ramp-up', type=int, default=60, help='Ramp-up en segundos')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Verificar servidor
    if not check_server():
        print(f"❌ Servidor no está disponible en {BASE_URL}")
        print("   Inicia el servidor con: python -m bulk_chat.main --llm-provider mock")
        sys.exit(1)
    
    # Ejecutar comando
    try:
        if args.command == 'create':
            bulk_create(args.count, args.messages)
        elif args.command == 'pause':
            bulk_pause(args.session_ids, args.reason)
        elif args.command == 'resume':
            bulk_resume(args.session_ids)
        elif args.command == 'stop':
            bulk_stop(args.session_ids)
        elif args.command == 'delete':
            bulk_delete(args.session_ids)
        elif args.command == 'export':
            bulk_export(args.session_ids, args.format)
        elif args.command == 'analyze':
            bulk_analyze(args.session_ids)
        elif args.command == 'cleanup':
            bulk_cleanup(args.days, args.dry_run)
        elif args.command == 'test-load':
            bulk_test_load(args.concurrent, args.duration)
        elif args.command == 'test-stress':
            bulk_test_stress(args.max_sessions, args.ramp_up)
        else:
            parser.print_help()
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error de API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Respuesta: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
















