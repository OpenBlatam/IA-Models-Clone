"""
Ejemplo de uso del Autonomous Long-Term Agent
"""

import asyncio
import requests
import time
from typing import Dict, Any


BASE_URL = "http://localhost:8001/api/v1"


def start_agent(instruction: str = "Operate autonomously and continuously learn") -> str:
    """Iniciar un nuevo agente"""
    response = requests.post(
        f"{BASE_URL}/agents/start",
        json={"instruction": instruction}
    )
    response.raise_for_status()
    result = response.json()
    print(f"✅ Agente iniciado: {result['message']}")
    
    # Extraer agent_id del mensaje
    agent_id = result['message'].split()[-1]
    return agent_id


def start_parallel_agents(count: int, instruction: str = "Operate autonomously") -> list:
    """Iniciar múltiples agentes en paralelo"""
    response = requests.post(
        f"{BASE_URL}/agents/parallel",
        json={"count": count, "instruction": instruction}
    )
    response.raise_for_status()
    result = response.json()
    print(f"✅ {result['total']} agentes iniciados en paralelo")
    return result['agent_ids']


def get_agent_status(agent_id: str) -> Dict[str, Any]:
    """Obtener estado de un agente"""
    response = requests.get(f"{BASE_URL}/agents/{agent_id}/status")
    response.raise_for_status()
    return response.json()


def add_task(agent_id: str, instruction: str, metadata: Dict[str, Any] = None) -> str:
    """Agregar tarea a un agente"""
    response = requests.post(
        f"{BASE_URL}/agents/{agent_id}/tasks",
        json={"instruction": instruction, "metadata": metadata or {}}
    )
    response.raise_for_status()
    result = response.json()
    print(f"✅ Tarea agregada: {result['task_id']}")
    return result['task_id']


def list_tasks(agent_id: str, status: str = None) -> list:
    """Listar tareas de un agente"""
    url = f"{BASE_URL}/agents/{agent_id}/tasks"
    if status:
        url += f"?status={status}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def list_agents() -> list:
    """Listar todos los agentes activos"""
    response = requests.get(f"{BASE_URL}/agents")
    response.raise_for_status()
    result = response.json()
    return result['agents']


def stop_agent(agent_id: str):
    """Detener un agente"""
    response = requests.post(f"{BASE_URL}/agents/{agent_id}/stop")
    response.raise_for_status()
    print(f"⏹️  Agente {agent_id} detenido")


def stop_all_agents():
    """Detener todos los agentes"""
    response = requests.post(f"{BASE_URL}/agents/stop-all")
    response.raise_for_status()
    print(f"⏹️  {response.json()['message']}")


def example_single_agent():
    """Ejemplo: Agente simple"""
    print("\n" + "="*60)
    print("Ejemplo 1: Agente Simple")
    print("="*60)
    
    # Iniciar agente
    agent_id = start_agent("Aprender sobre Python y machine learning")
    
    # Agregar algunas tareas
    add_task(agent_id, "Investigar las mejores prácticas de Python")
    add_task(agent_id, "Analizar frameworks de machine learning")
    add_task(agent_id, "Estudiar técnicas de deep learning")
    
    # Esperar un poco
    print("\n⏳ Esperando 5 segundos para que el agente procese...")
    time.sleep(5)
    
    # Ver estado
    status = get_agent_status(agent_id)
    print(f"\n📊 Estado del agente:")
    print(f"  - Tareas completadas: {status['metrics']['tasks_completed']}")
    print(f"  - Tokens usados: {status['metrics']['total_tokens_used']}")
    print(f"  - Uptime: {status['metrics']['uptime_seconds']:.2f} segundos")
    print(f"  - Tamaño de cola: {status['queue_size']}")
    
    # Listar tareas
    tasks = list_tasks(agent_id)
    print(f"\n📋 Tareas ({len(tasks)}):")
    for task in tasks[:5]:  # Mostrar solo las primeras 5
        print(f"  - {task['task_id']}: {task['status']} - {task['instruction'][:50]}...")
    
    # Detener agente
    print("\n⏹️  Deteniendo agente...")
    stop_agent(agent_id)


def example_parallel_agents():
    """Ejemplo: Múltiples agentes en paralelo"""
    print("\n" + "="*60)
    print("Ejemplo 2: Agentes Paralelos")
    print("="*60)
    
    # Iniciar 3 agentes en paralelo
    agent_ids = start_parallel_agents(
        count=3,
        instruction="Investigar diferentes aspectos de IA"
    )
    
    # Agregar tareas específicas a cada agente
    tasks_per_agent = [
        "Investigar modelos de lenguaje grandes",
        "Estudiar computer vision",
        "Analizar reinforcement learning"
    ]
    
    for agent_id, task in zip(agent_ids, tasks_per_agent):
        add_task(agent_id, task)
    
    # Esperar un poco
    print("\n⏳ Esperando 5 segundos...")
    time.sleep(5)
    
    # Ver estado de todos los agentes
    print("\n📊 Estado de todos los agentes:")
    agents = list_agents()
    for agent in agents:
        print(f"\n  Agente {agent['agent_id'][:8]}...")
        print(f"    - Estado: {agent['status']}")
        print(f"    - Tareas completadas: {agent['metrics']['tasks_completed']}")
        print(f"    - Tokens usados: {agent['metrics']['total_tokens_used']}")
    
    # Detener todos los agentes
    print("\n⏹️  Deteniendo todos los agentes...")
    stop_all_agents()


def example_continuous_operation():
    """Ejemplo: Operación continua"""
    print("\n" + "="*60)
    print("Ejemplo 3: Operación Continua")
    print("="*60)
    
    # Iniciar agente
    agent_id = start_agent("Operar continuamente y aprender")
    
    print("\n⚠️  El agente correrá continuamente hasta que lo detengas")
    print("   Presiona Ctrl+C para detener")
    
    try:
        # Agregar tareas periódicamente
        for i in range(5):
            add_task(agent_id, f"Tarea {i+1}: Procesar información sobre IA")
            time.sleep(2)
            
            # Mostrar estado periódicamente
            status = get_agent_status(agent_id)
            print(f"\n📊 Estado (iteración {i+1}):")
            print(f"  - Tareas completadas: {status['metrics']['tasks_completed']}")
            print(f"  - Tareas fallidas: {status['metrics']['tasks_failed']}")
            print(f"  - Uptime: {status['metrics']['uptime_seconds']:.2f}s")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupción recibida")
    
    finally:
        # Detener agente
        print("\n⏹️  Deteniendo agente...")
        stop_agent(agent_id)


if __name__ == "__main__":
    print("🚀 Ejemplos de uso del Autonomous Long-Term Agent")
    print("=" * 60)
    print("\nAsegúrate de que el servidor esté corriendo en http://localhost:8001")
    print("Inicia el servidor con: python -m autonomous_long_term_agent.main\n")
    
    try:
        # Ejecutar ejemplos
        example_single_agent()
        time.sleep(2)
        
        example_parallel_agents()
        time.sleep(2)
        
        # Descomentar para probar operación continua
        # example_continuous_operation()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar al servidor")
        print("   Asegúrate de que el servidor esté corriendo en http://localhost:8001")
    except Exception as e:
        print(f"\n❌ Error: {e}")




