"""
Script de prueba para el agente
"""

import asyncio
import sys
from pathlib import Path

# Agregar el directorio al path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent import CursorAgent, AgentConfig


async def test_agent():
    """Probar el agente"""
    print("🧪 Testing Cursor Agent 24/7...")
    
    # Crear configuración
    config = AgentConfig(
        check_interval=1.0,
        max_concurrent_tasks=3,
        task_timeout=30.0,
        persistent_storage=True,
        storage_path="./data/test_agent_state.json"
    )
    
    # Crear agente
    agent = CursorAgent(config)
    
    try:
        # Iniciar agente
        print("🚀 Starting agent...")
        await agent.start()
        
        # Agregar algunas tareas de prueba
        print("📝 Adding test tasks...")
        task1 = await agent.add_task("print('Task 1: Hello from agent!')")
        task2 = await agent.add_task("print('Task 2: Testing execution')")
        task3 = await agent.add_task("shell: echo 'Task 3: Shell command'")
        
        print(f"✅ Tasks added: {task1}, {task2}, {task3}")
        
        # Esperar a que se ejecuten
        print("⏳ Waiting for tasks to complete...")
        await asyncio.sleep(5)
        
        # Ver estado
        status = await agent.get_status()
        print(f"📊 Status: {status}")
        
        # Ver tareas
        tasks = await agent.get_tasks(limit=10)
        print(f"📋 Tasks: {len(tasks)} total")
        for task in tasks:
            print(f"  - {task['id']}: {task['status']} - {task['command'][:50]}")
        
        # Detener agente
        print("🛑 Stopping agent...")
        await agent.stop()
        
        print("✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(test_agent())


