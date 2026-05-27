import time
import logging
from agents.os_nexus import sys, SysCallError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NexusClientTest")

if __name__ == "__main__":
    logger.info("🧠 Iniciando Mente (Python Brain) con 'libc' os_nexus...")
    
    try:
        # Prueba 1: Ping básico
        logger.info("Test 1: Ping")
        logger.info(f"Respuesta: {sys.ping()}")
        time.sleep(0.5)
        
        # Prueba 2: L1 Cache Write
        logger.info("Test 2: L1 Cache Write")
        sys.mem_write("test_agent_status", "active_via_os_nexus")
        logger.info("Escritura en memoria exitosa.")
        time.sleep(0.5)
        
        # Prueba 3: L1 Cache Read
        logger.info("Test 3: L1 Cache Read")
        val = sys.mem_read("test_agent_status")
        logger.info(f"Leído de memoria: {val}")
        time.sleep(0.5)
        
        # Prueba 4: VFS Jail Write
        logger.info("Test 4: VFS Jail Write")
        sys.vfs_write("test_libc.txt", "Hola VFS desde la libc de Python!")
        logger.info("Archivo escrito en jaula VFS.")
        time.sleep(0.5)
        
        # Prueba 5: VFS Jail Read
        logger.info("Test 5: VFS Jail Read")
        file_val = sys.vfs_read("test_libc.txt")
        logger.info(f"Contenido del archivo: {file_val}")
        time.sleep(0.5)
        
        # Prueba 6: VFS Escape Attempt (Should fail with SysCallError)
        logger.info("Test 6: VFS Escape Attempt")
        try:
            sys.vfs_read("../../../Windows/System32/cmd.exe")
        except SysCallError as e:
            logger.info(f"VFS Escape Bloqueado exitosamente. Razón: {e}")
        time.sleep(0.5)

        # Prueba 7: Solicitud simulada de Sandbox
        logger.info("Test 7: Sandbox Execution")
        output = sys.execute_code("print('Hello World from os_nexus!')")
        logger.info(f"Ejecución exitosa, Output: {output.strip()}")

    except SysCallError as e:
        logger.error(f"SysCall Error: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
