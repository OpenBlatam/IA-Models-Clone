"""
Ejemplos de uso del GitHub Autonomous Agent.

Este archivo contiene ejemplos de cómo usar las diferentes
funcionalidades del agente.
"""

import asyncio
from typing import Dict, Any

# Ejemplos de imports (ajustar según tu implementación)
# from core.github_client import GitHubClient
# from core.task_processor import TaskProcessor
# from core.worker import WorkerManager
# from api.schemas import TaskCreate, TaskResponse


# ============================================================================
# Ejemplo 1: Cliente de GitHub Básico
# ============================================================================

async def example_github_client():
    """Ejemplo de uso básico del cliente de GitHub."""
    print("=== Ejemplo 1: Cliente de GitHub ===\n")
    
    # Inicializar cliente
    # client = GitHubClient(token="your_token_here")
    
    # Obtener información de un repositorio
    # repo = await client.get_repository("owner", "repo-name")
    # print(f"Repositorio: {repo['name']}")
    # print(f"Descripción: {repo.get('description', 'Sin descripción')}")
    
    # Listar issues
    # issues = await client.list_issues("owner", "repo-name")
    # print(f"\nTotal de issues: {len(issues)}")
    
    # Crear un issue
    # new_issue = await client.create_issue(
    #     owner="owner",
    #     repo="repo-name",
    #     title="Nuevo issue desde agente",
    #     body="Este issue fue creado automáticamente"
    # )
    # print(f"\nIssue creado: {new_issue['number']}")
    
    print("✅ Ejemplo completado\n")


# ============================================================================
# Ejemplo 2: Procesar Tarea
# ============================================================================

async def example_process_task():
    """Ejemplo de procesamiento de tarea."""
    print("=== Ejemplo 2: Procesar Tarea ===\n")
    
    # Inicializar procesador
    # processor = TaskProcessor()
    
    # Crear tarea
    # task_data = {
    #     "repository": "owner/repo-name",
    #     "instruction": "Crear un nuevo archivo README.md",
    #     "priority": "high"
    # }
    
    # Procesar tarea
    # result = await processor.process_task(task_data)
    # print(f"Tarea procesada: {result['task_id']}")
    # print(f"Estado: {result['status']}")
    
    print("✅ Ejemplo completado\n")


# ============================================================================
# Ejemplo 3: Worker Manager
# ============================================================================

async def example_worker_manager():
    """Ejemplo de uso del Worker Manager."""
    print("=== Ejemplo 3: Worker Manager ===\n")
    
    # Inicializar manager
    # manager = WorkerManager()
    
    # Iniciar workers
    # await manager.start()
    # print("Workers iniciados")
    
    # Agregar tarea a la cola
    # task = {
    #     "id": "task-123",
    #     "type": "github_operation",
    #     "data": {"action": "create_file", "path": "test.py"}
    # }
    # await manager.add_task(task)
    # print(f"Tarea agregada: {task['id']}")
    
    # Esperar un poco
    # await asyncio.sleep(5)
    
    # Detener workers
    # await manager.stop()
    # print("Workers detenidos")
    
    print("✅ Ejemplo completado\n")


# ============================================================================
# Ejemplo 4: API REST
# ============================================================================

async def example_api_usage():
    """Ejemplo de uso de la API REST."""
    print("=== Ejemplo 4: API REST ===\n")
    
    import httpx
    
    base_url = "http://localhost:8030/api/v1"
    
    async with httpx.AsyncClient() as client:
        # Health check
        response = await client.get(f"{base_url}/../health")
        print(f"Health: {response.json()}")
        
        # Crear tarea
        # task_data = {
        #     "repository": "owner/repo",
        #     "instruction": "Agregar documentación"
        # }
        # response = await client.post(f"{base_url}/tasks", json=task_data)
        # task = response.json()
        # print(f"Tarea creada: {task['id']}")
        
        # Obtener estado de tarea
        # task_id = task['id']
        # response = await client.get(f"{base_url}/tasks/{task_id}")
        # status = response.json()
        # print(f"Estado: {status['status']}")
    
    print("✅ Ejemplo completado\n")


# ============================================================================
# Ejemplo 5: Flujo Completo
# ============================================================================

async def example_complete_flow():
    """Ejemplo de flujo completo."""
    print("=== Ejemplo 5: Flujo Completo ===\n")
    
    # 1. Inicializar componentes
    # client = GitHubClient(token="your_token")
    # processor = TaskProcessor()
    # manager = WorkerManager()
    
    # 2. Iniciar workers
    # await manager.start()
    
    # 3. Crear tarea
    # task = {
    #     "repository": "owner/repo",
    #     "instruction": "Crear archivo de configuración",
    #     "files": [
    #         {
    #             "path": "config.json",
    #             "content": '{"key": "value"}'
    #         }
    #     ]
    # }
    
    # 4. Procesar tarea
    # result = await processor.process_task(task)
    
    # 5. Monitorear progreso
    # while result['status'] != 'completed':
    #     await asyncio.sleep(1)
    #     result = await processor.get_task_status(result['task_id'])
    #     print(f"Progreso: {result['progress']}%")
    
    # 6. Obtener resultado
    # final_result = await processor.get_task_result(result['task_id'])
    # print(f"Resultado: {final_result}")
    
    # 7. Limpiar
    # await manager.stop()
    
    print("✅ Ejemplo completado\n")


# ============================================================================
# Ejemplo 6: Manejo de Errores
# ============================================================================

async def example_error_handling():
    """Ejemplo de manejo de errores."""
    print("=== Ejemplo 6: Manejo de Errores ===\n")
    
    try:
        # client = GitHubClient(token="invalid_token")
        # await client.get_repository("owner", "repo")
        pass
    except Exception as e:
        print(f"Error capturado: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        # Manejar error apropiadamente
    
    print("✅ Ejemplo completado\n")


# ============================================================================
# Ejecutar Ejemplos
# ============================================================================

async def main():
    """Ejecutar todos los ejemplos."""
    print("=" * 60)
    print("Ejemplos de Uso - GitHub Autonomous Agent")
    print("=" * 60)
    print()
    
    # Ejecutar ejemplos
    await example_github_client()
    await example_process_task()
    await example_worker_manager()
    await example_api_usage()
    await example_complete_flow()
    await example_error_handling()
    
    print("=" * 60)
    print("Todos los ejemplos completados")
    print("=" * 60)


if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())




