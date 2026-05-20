#!/usr/bin/env python3
"""
Script de Prueba de Generación
===============================

Script simple para probar que el sistema genera contenido correctamente.
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Verificar que el servidor está funcionando."""
    print("🔍 Verificando salud del servidor...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor funcionando correctamente")
            print(f"   Respuesta: {response.json()}")
            return True
        else:
            print(f"❌ Servidor respondió con código: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servidor")
        print("   Asegúrate de que el servidor esté corriendo:")
        print("   python start.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_bulk_generate():
    """Probar generación masiva básica."""
    print("\n📝 Probando generación masiva básica...")
    
    payload = {
        "query": "Explicar las ventajas de la inteligencia artificial en la medicina moderna",
        "config": {
            "max_documents": 3,
            "max_tokens": 1000,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/bulk/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Generación iniciada exitosamente")
            print(f"   Task ID: {result.get('task_id', 'N/A')}")
            print(f"   Estado: {result.get('status', 'N/A')}")
            return result.get('task_id')
        else:
            print(f"❌ Error en generación: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def check_task_status(task_id):
    """Verificar estado de una tarea."""
    print(f"\n📊 Verificando estado de tarea {task_id}...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/bulk/status/{task_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Estado: {status.get('status', 'N/A')}")
            print(f"   Progreso: {status.get('progress', 0)}%")
            print(f"   Documentos generados: {status.get('documents_generated', 0)}")
            return status
        else:
            print(f"⚠️  Estado no disponible: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Error verificando estado: {e}")
        return None

def get_documents(task_id, limit=5):
    """Obtener documentos generados."""
    print(f"\n📄 Obteniendo documentos generados (limit={limit})...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/bulk/documents/{task_id}",
            params={"limit": limit, "offset": 0},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            documents = data.get('documents', [])
            print(f"✅ Obtenidos {len(documents)} documentos")
            
            for i, doc in enumerate(documents[:3], 1):
                print(f"\n   Documento {i}:")
                print(f"   - ID: {doc.get('id', 'N/A')}")
                print(f"   - Título: {doc.get('title', 'Sin título')}")
                content = doc.get('content', '')
                if content:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"   - Contenido: {preview}")
                print(f"   - Calidad: {doc.get('quality_score', 'N/A')}")
            
            return documents
        else:
            print(f"⚠️  No se pudieron obtener documentos: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️  Error obteniendo documentos: {e}")
        return []

def test_bulk_ai_query():
    """Probar query de Bulk AI."""
    print("\n🤖 Probando Bulk AI Query...")
    
    payload = {
        "query": "Historia breve de la programación",
        "max_documents": 2,
        "enable_continuous": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/bulk-ai/process-query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query procesada exitosamente")
            print(f"   Task ID: {result.get('task_id', 'N/A')}")
            return result.get('task_id')
        else:
            print(f"⚠️  Respuesta: {response.status_code}")
            print(f"   {response.text[:200]}")
            return None
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return None

def main():
    """Función principal."""
    print("=" * 60)
    print("🧪 Prueba de Generación - Bulk TruthGPT")
    print("=" * 60)
    print()
    
    # Verificar salud
    if not test_health():
        print("\n❌ El servidor no está disponible. Inicia el servidor primero:")
        print("   python start.py")
        return 1
    
    # Probar generación básica
    task_id = test_bulk_generate()
    
    if task_id:
        # Esperar un poco
        print("\n⏳ Esperando 5 segundos...")
        time.sleep(5)
        
        # Verificar estado
        status = check_task_status(task_id)
        
        # Intentar obtener documentos
        if status and status.get('status') in ['completed', 'processing']:
            documents = get_documents(task_id, limit=3)
            if documents:
                print(f"\n✅ ¡Generación exitosa! Se generaron documentos.")
    
    # Probar Bulk AI
    print("\n" + "=" * 60)
    bulk_task_id = test_bulk_ai_query()
    
    if bulk_task_id:
        time.sleep(3)
        status = check_task_status(bulk_task_id)
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")
    print("=" * 60)
    print("\n💡 Para más ejemplos, consulta:")
    print("   - EJEMPLOS_GENERACION.md")
    print("   - http://localhost:8000/docs")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
































