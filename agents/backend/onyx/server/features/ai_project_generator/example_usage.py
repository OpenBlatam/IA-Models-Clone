"""
Ejemplo de uso del AI Project Generator
========================================

Este script muestra cómo usar el generador de proyectos de IA.
"""

import asyncio
import requests
import time
from pathlib import Path

# URL del servidor (ajustar si es necesario)
API_URL = "http://localhost:8020"


def example_generate_project():
    """Ejemplo de generación de proyecto"""
    print("🚀 Generando proyecto de IA...")
    
    response = requests.post(
        f"{API_URL}/api/v1/generate",
        json={
            "description": "Un sistema de chat con IA que responde preguntas sobre programación y ayuda a los desarrolladores",
            "project_name": "programming_assistant",
            "author": "Blatam Academy",
            "priority": 5
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Proyecto agregado a la cola!")
        print(f"   ID: {data['project_id']}")
        print(f"   Estado: {data['status']}")
        print(f"   Mensaje: {data['message']}")
        return data['project_id']
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def example_check_status(project_id: str):
    """Ejemplo de verificación de estado"""
    print(f"\n📊 Verificando estado del proyecto {project_id}...")
    
    response = requests.get(f"{API_URL}/api/v1/project/{project_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Estado: {data.get('status', 'unknown')}")
        print(f"   Creado: {data.get('created_at', 'unknown')}")
        if 'completed_at' in data:
            print(f"   Completado: {data['completed_at']}")
        if 'result' in data:
            print(f"   Directorio: {data['result'].get('project_dir', 'unknown')}")
    else:
        print(f"❌ Error: {response.status_code}")


def example_get_queue():
    """Ejemplo de obtener la cola"""
    print("\n📋 Obteniendo cola de proyectos...")
    
    response = requests.get(f"{API_URL}/api/v1/queue")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Proyectos en cola: {data.get('queue_size', 0)}")
        for project in data.get('queue', [])[:5]:  # Primeros 5
            print(f"   - {project['id']}: {project['description'][:50]}...")
    else:
        print(f"❌ Error: {response.status_code}")


def example_get_generator_status():
    """Ejemplo de obtener estado del generador"""
    print("\n🔍 Estado del generador:")
    
    response = requests.get(f"{API_URL}/api/v1/status")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Corriendo: {data.get('is_running', False)}")
        print(f"   Cola: {data.get('queue_size', 0)} proyectos")
        print(f"   Procesados: {data.get('processed_count', 0)} proyectos")
    else:
        print(f"❌ Error: {response.status_code}")


def main():
    """Función principal"""
    print("=" * 60)
    print("AI Project Generator - Ejemplo de Uso")
    print("=" * 60)
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code != 200:
            print(f"❌ El servidor no está respondiendo correctamente")
            print(f"   Asegúrate de que el servidor esté corriendo en {API_URL}")
            return
    except requests.exceptions.RequestException:
        print(f"❌ No se puede conectar al servidor en {API_URL}")
        print(f"   Inicia el servidor con: python main.py")
        return
    
    print("✅ Servidor conectado\n")
    
    # Generar un proyecto
    project_id = example_generate_project()
    
    if project_id:
        # Esperar un poco
        print("\n⏳ Esperando 3 segundos...")
        time.sleep(3)
        
        # Verificar estado
        example_check_status(project_id)
        
        # Ver cola
        example_get_queue()
        
        # Ver estado del generador
        example_get_generator_status()
        
        print("\n" + "=" * 60)
        print("✅ Ejemplo completado!")
        print(f"   Puedes verificar el estado del proyecto con:")
        print(f"   curl {API_URL}/api/v1/project/{project_id}")
        print("=" * 60)


if __name__ == "__main__":
    main()


