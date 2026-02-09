"""
Ejemplo de uso del sistema Bulk Chat
=====================================

Este script muestra cómo usar el sistema de chat continuo programáticamente.
"""

import asyncio
import logging
from bulk_chat.core.chat_engine import ContinuousChatEngine
from bulk_chat.config.chat_config import ChatConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def mock_llm_provider(messages, **kwargs):
    """Proveedor de LLM simulado para el ejemplo."""
    await asyncio.sleep(1)  # Simular latencia
    
    last_message = messages[-1]["content"] if messages else ""
    response_number = len([m for m in messages if m["role"] == "assistant"])
    
    responses = [
        f"Entiendo que quieres saber sobre {last_message[:50]}. Déjame explicarte...",
        f"Además, es importante mencionar que...",
        f"Otra cosa relevante es...",
        f"También deberías considerar que...",
        f"Finalmente, cabe destacar...",
    ]
    
    return responses[response_number % len(responses)]


async def main():
    """Ejemplo principal."""
    print("🚀 Iniciando ejemplo de Bulk Chat\n")
    
    # Crear configuración
    config = ChatConfig()
    config.auto_continue = True
    config.response_interval = 2.0
    config.max_consecutive_responses = 5
    
    # Crear motor de chat
    engine = ContinuousChatEngine(
        llm_provider=mock_llm_provider,
        auto_continue=config.auto_continue,
        response_interval=config.response_interval,
        max_consecutive_responses=config.max_consecutive_responses,
    )
    
    # Crear sesión
    print("📝 Creando sesión de chat...")
    session = await engine.create_session(
        user_id="demo_user",
        initial_message="Explícame sobre inteligencia artificial",
        auto_continue=True,
    )
    
    print(f"✅ Sesión creada: {session.session_id}\n")
    
    # Iniciar chat continuo
    print("🔄 Iniciando chat continuo...")
    await engine.start_continuous_chat(session.session_id)
    print("✅ Chat iniciado. Generando respuestas automáticamente...\n")
    
    # Esperar y mostrar mensajes
    print("⏳ Esperando respuestas (10 segundos)...\n")
    await asyncio.sleep(10)
    
    # Mostrar mensajes generados
    print(f"\n📊 Mensajes generados: {len(session.messages)}\n")
    for i, msg in enumerate(session.messages, 1):
        role_emoji = "👤" if msg.role == "user" else "🤖"
        print(f"{i}. {role_emoji} [{msg.role}]: {msg.content[:100]}...")
    
    # Pausar el chat
    print("\n⏸️  Pausando el chat...")
    await engine.pause_session(session.session_id, "Pausado por usuario")
    print("✅ Chat pausado\n")
    
    # Esperar un poco
    await asyncio.sleep(2)
    
    # Reanudar
    print("▶️  Reanudando el chat...")
    await engine.resume_session(session.session_id)
    print("✅ Chat reanudado\n")
    
    # Esperar más respuestas
    print("⏳ Esperando más respuestas (5 segundos)...\n")
    await asyncio.sleep(5)
    
    # Mostrar estado final
    print(f"\n📊 Total de mensajes: {len(session.messages)}")
    print(f"📊 Estado: {session.state.value}")
    print(f"📊 Pausado: {session.is_paused}\n")
    
    # Detener
    print("🛑 Deteniendo el chat...")
    await engine.stop_session(session.session_id)
    print("✅ Chat detenido\n")
    
    print("✨ Ejemplo completado!")


if __name__ == "__main__":
    asyncio.run(main())
































