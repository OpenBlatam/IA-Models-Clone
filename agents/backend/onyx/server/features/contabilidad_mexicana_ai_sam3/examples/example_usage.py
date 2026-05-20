"""
Example usage of Contabilidad Mexicana AI SAM3
==============================================
"""

import asyncio
import os
from contabilidad_mexicana_ai_sam3 import ContadorSAM3Agent, ContadorSAM3Config


async def example_calcular_impuestos():
    """Example: Calculate taxes."""
    config = ContadorSAM3Config()
    agent = ContadorSAM3Agent(config=config)
    
    # Submit task
    task_id = await agent.calcular_impuestos(
        regimen="RESICO",
        tipo_impuesto="ISR",
        datos={
            "ingresos": 100000,
            "gastos": 30000,
            "periodo": "2024-01"
        }
    )
    
    print(f"Task submitted: {task_id}")
    
    # Wait for result
    while True:
        status = await agent.get_task_status(task_id)
        print(f"Status: {status['status']}")
        
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print(f"\nResult:\n{result}")
            break
        elif status["status"] == "failed":
            print(f"Task failed: {status.get('error')}")
            break
        
        await asyncio.sleep(2)
    
    await agent.close()


async def example_asesoria_fiscal():
    """Example: Get fiscal advice."""
    config = ContadorSAM3Config()
    agent = ContadorSAM3Agent(config=config)
    
    task_id = await agent.asesoria_fiscal(
        pregunta="¿Puedo deducir gastos de home office en RESICO?",
        contexto={
            "regimen": "RESICO",
            "ingresos_anuales": 500000
        }
    )
    
    print(f"Task submitted: {task_id}")
    
    # Wait for result
    while True:
        status = await agent.get_task_status(task_id)
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print(f"\nAdvice:\n{result['asesoria']}")
            break
        elif status["status"] == "failed":
            print(f"Task failed: {status.get('error')}")
            break
        await asyncio.sleep(2)
    
    await agent.close()


async def main():
    """Run examples."""
    print("=== Contabilidad Mexicana AI SAM3 Examples ===\n")
    
    print("1. Tax Calculation Example")
    print("-" * 50)
    await example_calcular_impuestos()
    
    print("\n2. Fiscal Advice Example")
    print("-" * 50)
    await example_asesoria_fiscal()


if __name__ == "__main__":
    asyncio.run(main())
