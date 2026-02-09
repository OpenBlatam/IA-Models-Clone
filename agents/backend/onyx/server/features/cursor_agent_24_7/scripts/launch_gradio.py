"""
Launch Gradio - Lanzar interfaz Gradio
=======================================

Script para lanzar la interfaz Gradio del agente.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import CursorAgent, AgentConfig
from core.gradio_interface import GradioInterface


async def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio interface for Cursor Agent 24/7")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    
    args = parser.parse_args()
    
    # Crear agente
    config = AgentConfig(persistent_storage=True)
    agent = CursorAgent(config)
    
    # Inicializar agente
    await agent.start()
    
    # Crear interfaz Gradio
    gradio_interface = GradioInterface(agent)
    interface = gradio_interface.create_interface()
    
    print(f"🚀 Launching Gradio interface on http://{args.host}:{args.port}")
    if args.share:
        print("📡 Creating public link...")
    
    # Lanzar
    interface.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port
    )


if __name__ == "__main__":
    asyncio.run(main())



