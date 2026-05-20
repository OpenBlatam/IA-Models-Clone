"""
Ejemplo de Uso de API - Cómo usar los endpoints
===============================================

Ejemplos de cómo usar la API una vez iniciada.
"""

import requests

# URL base
BASE_URL = "http://localhost:8020"

# 1. Health Check
def check_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health:", response.json())

# 2. Crear Proyecto
def create_project():
    response = requests.post(
        f"{BASE_URL}/api/v1/projects",
        json={
            "description": "Un sistema de chat con IA que responde preguntas sobre programación",
            "project_name": "programming_chat_ai",
            "author": "Blatam Academy"
        }
    )
    print("Project created:", response.json())
    return response.json()["project_id"]

# 3. Obtener Proyecto
def get_project(project_id: str):
    response = requests.get(f"{BASE_URL}/api/v1/projects/{project_id}")
    print("Project:", response.json())

# 4. Generar Proyecto
def generate_project():
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        json={
            "description": "Un analizador de imágenes con IA",
            "project_name": "image_analyzer",
            "author": "Blatam Academy"
        }
    )
    print("Generation:", response.json())
    return response.json().get("project_id") or response.json().get("task_id")

# 5. Listar Proyectos
def list_projects():
    response = requests.get(f"{BASE_URL}/api/v1/projects")
    print("Projects:", response.json())

# 6. Métricas (si está habilitado)
def get_metrics():
    response = requests.get(f"{BASE_URL}/metrics")
    print("Metrics:", response.text[:500])  # Primeros 500 caracteres

if __name__ == "__main__":
    # Ejecutar ejemplos
    print("=== Health Check ===")
    check_health()
    
    print("\n=== Crear Proyecto ===")
    project_id = create_project()
    
    print("\n=== Obtener Proyecto ===")
    get_project(project_id)
    
    print("\n=== Generar Proyecto ===")
    generate_project()
    
    print("\n=== Listar Proyectos ===")
    list_projects()
    
    print("\n=== Métricas ===")
    get_metrics()















