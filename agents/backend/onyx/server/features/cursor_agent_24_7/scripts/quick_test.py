"""
Quick Test - Prueba rápida del agente
======================================

Script para probar rápidamente el agente.
"""

import asyncio
import sys
from pathlib import Path

# Agregar al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import CursorAgent, AgentConfig


async def quick_test():
    """Prueba rápida"""
    print("🧪 Quick Test - Cursor Agent 24/7")
    print("=" * 50)
    
    # Crear agente
    config = AgentConfig(
        check_interval=0.5,
        max_concurrent_tasks=2,
        task_timeout=10.0,
        persistent_storage=False  # No guardar en test
    )
    
    agent = CursorAgent(config)
    
    try:
        # Iniciar
        print("\n1️⃣ Starting agent...")
        await agent.start()
        await asyncio.sleep(1)
        
        # Agregar tareas
        print("\n2️⃣ Adding test tasks...")
        task1 = await agent.add_task("print('Test 1: Hello!')")
        task2 = await agent.add_task("print('Test 2: World!')")
        print(f"   ✅ Task 1: {task1}")
        print(f"   ✅ Task 2: {task2}")
        
        # Esperar ejecución
        print("\n3️⃣ Waiting for execution...")
        await asyncio.sleep(3)
        
        # Ver estado
        print("\n4️⃣ Checking status...")
        status = await agent.get_status()
        print(f"   Status: {status['status']}")
        print(f"   Tasks: {status['tasks_total']} total")
        print(f"   Completed: {status['tasks_completed']}")
        
        # Ver tareas
        print("\n5️⃣ Getting tasks...")
        tasks = await agent.get_tasks(limit=5)
        for task in tasks:
            print(f"   - {task['id'][:12]}...: {task['status']} - {task['command'][:30]}")
        
        # Detener
        print("\n6️⃣ Stopping agent...")
        await agent.stop()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(quick_test())



