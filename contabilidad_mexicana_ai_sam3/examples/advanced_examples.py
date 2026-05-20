"""
Advanced examples for Contabilidad Mexicana AI SAM3
==================================================
"""

import asyncio
from contabilidad_mexicana_ai_sam3 import ContadorSAM3Agent, ContadorSAM3Config
from contabilidad_mexicana_ai_sam3.utils import (
    format_tax_calculation_result,
    format_fiscal_advice,
    validate_calculation_data,
)


async def example_batch_calculations():
    """Example: Process multiple tax calculations in parallel."""
    config = ContadorSAM3Config()
    agent = ContadorSAM3Agent(config=config, max_parallel_tasks=5)
    
    # Submit multiple tasks
    tasks = []
    calculations = [
        {"regimen": "RESICO", "tipo_impuesto": "ISR", "datos": {"ingresos": 100000, "gastos": 30000}},
        {"regimen": "RESICO", "tipo_impuesto": "ISR", "datos": {"ingresos": 200000, "gastos": 50000}},
        {"regimen": "PFAE", "tipo_impuesto": "ISR", "datos": {"ingresos": 500000, "gastos": 200000}},
    ]
    
    for calc in calculations:
        # Validate data first
        is_valid, error = validate_calculation_data(calc["datos"])
        if not is_valid:
            print(f"Error validating data: {error}")
            continue
        
        task_id = await agent.calcular_impuestos(
            regimen=calc["regimen"],
            tipo_impuesto=calc["tipo_impuesto"],
            datos=calc["datos"],
            priority=5
        )
        tasks.append((task_id, calc))
    
    print(f"Submitted {len(tasks)} tasks")
    
    # Wait for all results
    results = []
    while tasks:
        for task_id, calc in tasks[:]:
            status = await agent.get_task_status(task_id)
            if status["status"] == "completed":
                result = await agent.get_task_result(task_id)
                results.append((calc, result))
                tasks.remove((task_id, calc))
                print(f"Completed calculation for {calc['regimen']}")
            elif status["status"] == "failed":
                print(f"Failed calculation for {calc['regimen']}: {status.get('error')}")
                tasks.remove((task_id, calc))
        
        if tasks:
            await asyncio.sleep(1)
    
    # Display results
    for calc, result in results:
        print(f"\n{calc['regimen']} - {calc['tipo_impuesto']}:")
        print(format_tax_calculation_result(result.get("resultado", {})))
    
    await agent.close()


async def example_priority_queue():
    """Example: Demonstrate task priority."""
    config = ContadorSAM3Config()
    agent = ContadorSAM3Agent(config=config)
    
    # Submit tasks with different priorities
    low_priority = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos": 50000},
        priority=1
    )
    
    high_priority = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={"ingresos": 200000},
        priority=10
    )
    
    print(f"Low priority task: {low_priority}")
    print(f"High priority task: {high_priority}")
    
    # High priority should be processed first
    await asyncio.sleep(2)
    
    status_low = await agent.get_task_status(low_priority)
    status_high = await agent.get_task_status(high_priority)
    
    print(f"\nLow priority status: {status_low['status']}")
    print(f"High priority status: {status_high['status']}")
    
    await agent.close()


async def example_comprehensive_advice():
    """Example: Get comprehensive fiscal advice."""
    config = ContadorSAM3Config()
    agent = ContadorSAM3Agent(config=config)
    
    # Submit advice request
    task_id = await agent.asesoria_fiscal(
        pregunta="¿Qué régimen fiscal es mejor para un freelancer con ingresos de $800,000 anuales?",
        contexto={
            "regimen_actual": "RESICO",
            "ingresos_anuales": 800000,
            "gastos_anuales": 200000,
            "actividad": "Desarrollo de software"
        },
        priority=8
    )
    
    print(f"Task submitted: {task_id}")
    
    # Wait for result
    while True:
        status = await agent.get_task_status(task_id)
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print("\n" + format_fiscal_advice(result.get("asesoria", {})))
            break
        elif status["status"] == "failed":
            print(f"Task failed: {status.get('error')}")
            break
        await asyncio.sleep(2)
    
    await agent.close()


async def example_continuous_mode():
    """Example: Run agent in continuous mode (simulated)."""
    config = ContadorSAM3Config()
    agent = ContadorSAM3Agent(config=config)
    
    # Submit tasks
    tasks = []
    for i in range(3):
        task_id = await agent.calcular_impuestos(
            regimen="RESICO",
            tipo_impuesto="ISR",
            datos={"ingresos": 100000 + i * 10000, "gastos": 30000},
            priority=i
        )
        tasks.append(task_id)
    
    print(f"Submitted {len(tasks)} tasks")
    
    # Start agent processing (in real scenario, this would run continuously)
    # For this example, we'll just wait for results
    import time
    start_time = time.time()
    
    while tasks:
        for task_id in tasks[:]:
            status = await agent.get_task_status(task_id)
            if status["status"] == "completed":
                result = await agent.get_task_result(task_id)
                print(f"Task {task_id} completed in {time.time() - start_time:.2f}s")
                tasks.remove(task_id)
            elif status["status"] == "failed":
                print(f"Task {task_id} failed")
                tasks.remove(task_id)
        
        if tasks:
            await asyncio.sleep(1)
    
    await agent.close()


async def main():
    """Run advanced examples."""
    print("=== Advanced Examples ===\n")
    
    print("1. Batch Calculations")
    print("-" * 50)
    await example_batch_calculations()
    
    print("\n2. Priority Queue")
    print("-" * 50)
    await example_priority_queue()
    
    print("\n3. Comprehensive Advice")
    print("-" * 50)
    await example_comprehensive_advice()
    
    print("\n4. Continuous Mode Simulation")
    print("-" * 50)
    await example_continuous_mode()


if __name__ == "__main__":
    asyncio.run(main())
