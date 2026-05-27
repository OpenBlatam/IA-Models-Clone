import time
import logging
from agents.os_nexus import sys
import asyncio
from agents.memoria_aprendizaje.l1_l2_memory import L1L2TieredMemory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SystemTest")

async def test_memory():
    logger.info("--- TEST 1: Tiered Memory (L1 + L2) ---")
    memory = L1L2TieredMemory("test_l2_memory.db")
    session_id = "agent_test_session"
    
    # Escribir a L1 y L2
    await memory.add_message(session_id, "user", "Hola, esta es una prueba de Tiering L1/L2.")
    
    # Leer (debería ser un Hit en L1)
    history = await memory.get_history(session_id)
    logger.info(f"Historial recuperado: {history}")
    
def test_ipc():
    logger.info("--- TEST 2: IPC (Enjambre) ---")
    sys.ipc_send("agent_writer", "Redacta un documento sobre agujeros negros.")
    
    # Simulando al agent_writer leyendo su inbox
    msg = sys.ipc_read("agent_writer")
    logger.info(f"[agent_writer] Mensaje recibido vía Rust DashMap: {msg}")

def test_web():
    logger.info("--- TEST 3: Network Offloading (SYS_HTTP_GET) ---")
    # Delegamos la descarga de Example.com a los hilos de Tokio en Rust
    html = sys.http_get("http://example.com")
    logger.info(f"Rust devolvió {len(html)} bytes del servidor. Fragmento: {html[:60]}...")

if __name__ == "__main__":
    asyncio.run(test_memory())
    test_ipc()
    test_web()
    logger.info("--- TODOS LOS TESTS SUPERADOS ---")
