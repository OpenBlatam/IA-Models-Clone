from typing import List, Dict, Any, Optional
import json
import logging
from .sqlite_memory import SQLiteMemory
from agents.os_nexus import sys

logger = logging.getLogger(__name__)

class L1L2TieredMemory(SQLiteMemory):
    """
    Jerarquía de Memoria Avanzada (Tiering):
    - L1 Cache (RAM): Utiliza el DashMap del Kernel de Rust vía `os_nexus.mem_write` para velocidad instantánea (lecturas concurrentes en nanosegundos).
    - L2 Cache (Disco): Utiliza SQLite para persistencia a largo plazo y snapshots.
    """
    
    def __init__(self, db_path: str = "memory.db"):
        super().__init__(db_path)
        logger.info("🧠 L1/L2 Tiered Memory Activada. L1: Rust DashMap, L2: SQLite")

    async def add_message(self, session_id: str, role: str, content: str, meta: Optional[Dict] = None) -> None:
        """Escribe el mensaje en L1 y delega a SQLite (L2) para persistencia."""
        # 1. Escritura en L2 (Persistencia Disco - Heredado de SQLiteMemory)
        await super().add_message(session_id, role, content, meta)
        
        # 2. Refrescar L1 (Caché RAM en Rust)
        # Leemos los últimos mensajes de L2 y los volcamos a L1 para acceso instantáneo
        recent_history = await self.get_history(session_id, limit=20)
        
        # Convertimos a JSON y escribimos en L1
        l1_key = f"context:{session_id}"
        l1_value = json.dumps(recent_history)
        
        try:
            sys.mem_write(l1_key, l1_value)
            logger.debug(f"[L1 Cache] Contexto actualizado para session_id={session_id}")
        except Exception as e:
            logger.error(f"[L1 Cache] Fallo al escribir en la caché rápida de Rust: {e}")

    async def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Intenta leer de L1 primero. Si falla (Cache Miss), va a L2 y actualiza L1."""
        l1_key = f"context:{session_id}"
        
        try:
            l1_data = sys.mem_read(l1_key)
            if l1_data:
                logger.debug(f"[L1 Cache] HIT (Cache acertada) para session_id={session_id}")
                history = json.loads(l1_data)
                return history[-limit:]
        except Exception as e:
            logger.debug(f"[L1 Cache] MISS o Error: {e}")

        # L1 Cache Miss -> Leemos de L2 (Disco)
        logger.debug(f"[L2 Cache] Leyendo desde disco (SQLite) para session_id={session_id}")
        history = await super().get_history(session_id, limit)
        
        # Actualizamos L1
        try:
            sys.mem_write(l1_key, json.dumps(history))
        except Exception:
            pass
            
        return history
