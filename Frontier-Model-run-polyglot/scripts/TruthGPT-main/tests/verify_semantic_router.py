import sys
import os
import asyncio
import logging

# Configurar path para importar módulos correctamente
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization_core.agents.client import AgentClient
from optimization_core.configs.loader import load_config
from optimization_core.models import build_model

logging.basicConfig(level=logging.DEBUG)

async def main():
    print("Iniciando prueba de Router Semántico (Swarm)...")
    
    # Cargar LLM
    config_path = os.path.join(os.path.dirname(__file__), '..', 'optimization_core', 'configs', 'llm_default.yaml')
    if not os.path.exists(config_path):
        print(f"Buscando fallback: {config_path}")
        # Intentar ruta local de dev
        config_path = "optimization_core/configs/llm_default.yaml"
        
    cfg = load_config(config_path, overrides=None)
    llm = build_model(cfg.model.family, cfg.dict())
    
    # Iniciar cliente en modo swarm
    client = AgentClient(llm_engine=llm, use_swarm=True)
    
    prompts = [
        "Por favor escribe un script en Python que descargue un video de youtube.",
        "Necesito 5 tweets virales para promocionar mi nuevo curso de programación.",
        "Toma estos datos de ventas [10, 20, 30] y dime cuál es la media y varianza.",
        "Cuál es el sentido de la vida?"
    ]
    
    for idx, prompt in enumerate(prompts):
        print(f"\n========================================================")
        print(f"PRUEBA {idx+1}")
        print(f"QUERY: {prompt}")
        print(f"========================================================")
        try:
            # Observar los logs DEBUg para ver el Chain of Thought
            response = await client.run(user_id=f"test_user_{idx}", prompt=prompt)
            print(f"\n[RESPUESTA AGENTE]: {response[:200]}...\n")
        except Exception as e:
            print(f"Error procesando query: {e}")

if __name__ == "__main__":
    asyncio.run(main())
