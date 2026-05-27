import json
import logging
import hashlib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PythonMetaLearner:
    """
    Sistema Inmune para las herramientas de Python.
    Mantiene memoria a largo plazo sobre qué llamadas a herramientas fallan consistentemente.
    """
    def __init__(self, memory_file: str = "python_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory = self._load_memory()

    def _load_memory(self) -> dict:
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"No se pudo cargar la memoria meta-learning: {e}")
        return {}

    def _save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando memoria meta-learning: {e}")

    def _hash_tool_call(self, tool_name: str, tool_input: str) -> str:
        content = f"{tool_name}:{tool_input}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def should_break_circuit(self, tool_name: str, tool_input: str) -> bool:
        """
        Devuelve True si esta combinación de herramienta + input ha fallado 3 veces o más
        sin haber tenido éxito nunca, activando el Circuit Breaker.
        """
        tool_hash = self._hash_tool_call(tool_name, str(tool_input))
        entry = self.memory.get(tool_hash)
        
        if entry:
            if entry.get("failures", 0) >= 3 and entry.get("successes", 0) == 0:
                return True
        return False

    def record_result(self, tool_name: str, tool_input: str, success: bool):
        """
        Registra el resultado de una ejecución en la memoria muscular de Python.
        """
        tool_hash = self._hash_tool_call(tool_name, str(tool_input))
        
        if tool_hash not in self.memory:
            self.memory[tool_hash] = {
                "tool_name": tool_name,
                "input_preview": str(tool_input)[:50],
                "successes": 0,
                "failures": 0
            }
            
        if success:
            self.memory[tool_hash]["successes"] += 1
            # Si tuvo éxito, se perdona el historial de fracasos pasados
            self.memory[tool_hash]["failures"] = 0
        else:
            self.memory[tool_hash]["failures"] += 1
            
        self._save_memory()
