# truthgpt_advanced_memory.py - Mejora de TruthGPT con memoria estructurada, logs y mejora de inyección de código
import sqlite3
import time
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

class StructuredMemory:
    """Memoria jerárquica con segmentos, logs y puntuaciones de utilidad."""
    def __init__(self, db_path: str = "truthgpt_adv_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.segment_cache = defaultdict(list)
    
    def _init_db(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                description TEXT,
                created_at REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER,
                content TEXT,
                timestamp REAL,
                source TEXT,
                utility_score REAL DEFAULT 1.0,
                embedding BLOB,
                FOREIGN KEY(segment_id) REFERENCES segments(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                action TEXT,
                details TEXT,
                result TEXT
            )
        """)
        self.conn.commit()
    
    def _create_segment(self, name: str, description: str = "") -> int:
        c = self.conn.cursor()
        try:
            c.execute("INSERT INTO segments (name, description, created_at) VALUES (?, ?, ?)",
                      (name, description, time.time()))
            self.conn.commit()
            return c.lastrowid
        except sqlite3.IntegrityError:
            c.execute("SELECT id FROM segments WHERE name = ?", (name,))
            return c.fetchone()[0]
    
    def store(self, content: str, source: str = "system", segment: str = "general") -> int:
        seg_id = self._create_segment(segment)
        emb = self._mock_embed(content)
        c = self.conn.cursor()
        c.execute("INSERT INTO memory_entries (segment_id, content, timestamp, source, embedding) VALUES (?, ?, ?, ?, ?)",
                  (seg_id, content, time.time(), source, emb.tobytes()))
        self.conn.commit()
        entry_id = c.lastrowid
        self.segment_cache[segment].append(entry_id)
        return entry_id
    
    def _mock_embed(self, text: str) -> bytes:
        rng = np.random.RandomState(hash(text) & 0xFFFFFFFF)
        return rng.randn(384).astype(np.float32).tobytes()
    
    def retrieve(self, query: str, top_k: int = 5, segment: Optional[str] = None) -> List[Dict]:
        query_emb = np.frombuffer(self._mock_embed(query), dtype=np.float32)
        c = self.conn.cursor()
        if segment:
            seg_id = self._create_segment(segment)
            c.execute("SELECT id, content, timestamp, source, utility_score, embedding FROM memory_entries WHERE segment_id=?", (seg_id,))
        else:
            c.execute("SELECT id, content, timestamp, source, utility_score, embedding FROM memory_entries")
        rows = c.fetchall()
        scores = []
        for row in rows:
            stored_emb = np.frombuffer(row[5], dtype=np.float32)
            sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb) + 1e-8)
            scores.append((sim * row[4], row))
        scores.sort(key=lambda x: x[0], reverse=True)
        result = []
        for s, row in scores[:top_k]:
            result.append({
                "id": row[0],
                "content": row[1],
                "timestamp": row[2],
                "source": row[3],
                "score": round(s, 4)
            })
        return result
    
    def log_action(self, action: str, details: str = "", result: str = "success"):
        c = self.conn.cursor()
        c.execute("INSERT INTO action_logs (timestamp, action, details, result) VALUES (?, ?, ?, ?)",
                  (time.time(), action, details, result))
        self.conn.commit()
    
    def get_recent_logs(self, n: int = 10) -> List[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT timestamp, action, details, result FROM action_logs ORDER BY id DESC LIMIT ?", (n,))
        rows = c.fetchall()
        return [{"time": r[0], "action": r[1], "details": r[2], "result": r[3]} for r in rows]
    
    def improve_code_injection(self, current_code: str) -> str:
        """Sugiere mejoras basadas en patrones almacenados."""
        improved = f"""import logging
logger = logging.getLogger(__name__)

def safe_execute():
    try:
        # Código original:
        {current_code}
        logger.info("Code executed successfully")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise
"""
        self.log_action("improve_code_injection", details=f"Injected code: {current_code[:50]}...")
        return improved
    
    def close(self):
        self.conn.close()

# --- Ejemplo de uso ---
if __name__ == "__main__":
    sm = StructuredMemory("truthgpt_demo.db")
    # Almacenar memorias
    sm.store("El usuario prefiere café solo", source="human", segment="preferences")
    sm.store("TruthGPT debe ser modular", source="system", segment="rules")
    sm.store("Inyección de código necesita validación", source="analysis", segment="security")
    
    # Recuperar
    results = sm.retrieve("café", segment="preferences")
    print("Resultados de búsqueda:")
    for r in results:
        print(f"  - {r['content']} (score: {r['score']})")
    
    # Logging de acciones
    sm.log_action("search_papers", details="arXiv query for structured memory", result="found 10 papers")
    sm.log_action("refactor", details="modularize memory system", result="success")
    
    # Mejorar inyección de código
    old_code = "print('Hello World')"
    improved = sm.improve_code_injection(old_code)
    print("\nCódigo mejorado para inyección:")
    print(improved)
    
    sm.close()