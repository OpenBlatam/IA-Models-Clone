import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CoALAMemoryManager:
    """
    Gestor de Memoria basado en el paper "Cognitive Architectures for Language Agents" (CoALA).
    Divide la memoria en 4 lóbulos cognitivos para evitar la confusión de contexto en los LLMs.
    """
    def __init__(self, storage_dir: str = "coala_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.working_memory_file = self.storage_dir / "working.json"
        self.episodic_memory_file = self.storage_dir / "episodic.json"
        self.procedural_memory_file = self.storage_dir / "procedural.json"
        
        self.working_memory = self._load_json(self.working_memory_file, {"tasks": [], "scratchpad": ""})
        self.episodic_memory = self._load_json(self.episodic_memory_file, [])
        self.procedural_memory = self._load_json(self.procedural_memory_file, {})

    def _load_json(self, file_path: Path, default: Any) -> Any:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando {file_path.name}: {e}")
        return default

    def _save_json(self, file_path: Path, data: Any):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando {file_path.name}: {e}")

    # --- Working Memory (Short-Term / Scratchpad) ---
    def update_working_memory(self, scratchpad: str):
        self.working_memory["scratchpad"] = scratchpad
        self._save_json(self.working_memory_file, self.working_memory)

    def get_working_memory(self) -> str:
        return self.working_memory.get("scratchpad", "")

    # --- Episodic Memory (Experience / Reflections) ---
    def add_episode(self, task: str, action_taken: str, result: str, reflection: str):
        episode = {
            "task": task,
            "action": action_taken,
            "result": result,
            "reflection": reflection
        }
        self.episodic_memory.append(episode)
        # Mantener solo los últimos 20 episodios para no desbordar el contexto
        if len(self.episodic_memory) > 20:
            self.episodic_memory = self.episodic_memory[-20:]
        self._save_json(self.episodic_memory_file, self.episodic_memory)

    def get_episodic_memory_summary(self) -> str:
        if not self.episodic_memory:
            return "No hay recuerdos pasados."
        
        summary = "Reflexiones de episodios pasados:\n"
        # Traer solo los 5 más recientes para el prompt
        for ep in self.episodic_memory[-5:]:
            summary += f"- Tarea: {ep['task']} | Aprendizaje: {ep['reflection']}\n"
        return summary

    # --- Procedural Memory (Tool Knowledge & Rules) ---
    def add_procedure(self, tool_name: str, usage_rules: str):
        self.procedural_memory[tool_name] = usage_rules
        self._save_json(self.procedural_memory_file, self.procedural_memory)

    def get_procedural_memory(self) -> str:
        if not self.procedural_memory:
            return "Las herramientas deben usarse con precisión según su esquema JSON."
        
        rules = "Reglas procedimentales estrictas:\n"
        for tool, rule in self.procedural_memory.items():
            rules += f"- {tool}: {rule}\n"
        return rules

    # --- Semantic Memory (Factual Knowledge / RAG) ---
    # En la arquitectura CoALA, la Semantic Memory suele delegarse a un VectorDB (Pinecone/Chroma)
    # o a un "RAG" agent. Proveeremos un endpoint simulado para conectar el RAG existente.
    def get_semantic_memory(self, semantic_context: str) -> str:
        if not semantic_context:
            return "Memoria Semántica: No hay conocimiento enciclopédico inyectado."
        return f"Memoria Semántica Activa:\n{semantic_context}"

    def get_full_coala_context(self, semantic_context: str = "") -> str:
        """
        Formatea los 4 lóbulos para inyectarlos en el LLM.
        """
        return f"""
=========================================
🧠 [CoALA COGNITIVE ARCHITECTURE]
=========================================
--- 1. WORKING MEMORY (Scratchpad) ---
{self.get_working_memory()}

--- 2. EPISODIC MEMORY (Reflections) ---
{self.get_episodic_memory_summary()}

--- 3. PROCEDURAL MEMORY (Skills/Tools) ---
{self.get_procedural_memory()}

--- 4. SEMANTIC MEMORY (Factual Knowledge) ---
{self.get_semantic_memory(semantic_context)}
=========================================
"""
