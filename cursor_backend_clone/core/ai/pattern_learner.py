"""
Pattern Learner - Sistema de aprendizaje de patrones
=====================================================

Aprende patrones de comandos exitosos para mejorar la ejecución.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PatternLearner:
    """Aprende patrones de comandos exitosos"""
    
    def __init__(self, storage_path: str = "./data/patterns.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.patterns: Dict[str, Dict] = {}
        self.command_history: List[Dict] = []
        self.success_rates: Dict[str, float] = {}
        self._max_history = 1000
        
    async def record_command(
        self,
        command: str,
        success: bool,
        execution_time: float,
        result: Optional[str] = None
    ):
        """Registrar un comando ejecutado"""
        # Agregar a historial
        self.command_history.append({
            "command": command,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "result_length": len(result) if result else 0
        })
        
        # Mantener historial limitado
        if len(self.command_history) > self._max_history:
            self.command_history = self.command_history[-self._max_history:]
        
        # Actualizar patrones
        await self._update_patterns(command, success, execution_time)
        
        # Guardar periódicamente
        if len(self.command_history) % 10 == 0:
            await self.save()
    
    async def _update_patterns(self, command: str, success: bool, execution_time: float):
        """Actualizar patrones aprendidos"""
        # Extraer palabras clave del comando
        keywords = self._extract_keywords(command)
        
        for keyword in keywords:
            if keyword not in self.patterns:
                self.patterns[keyword] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time": 0.0,
                    "last_seen": None
                }
            
            pattern = self.patterns[keyword]
            pattern["count"] += 1
            if success:
                pattern["success_count"] += 1
            pattern["total_time"] += execution_time
            pattern["last_seen"] = datetime.now().isoformat()
        
        # Calcular tasa de éxito
        if keywords:
            for keyword in keywords:
                pattern = self.patterns[keyword]
                self.success_rates[keyword] = pattern["success_count"] / pattern["count"]
    
    def _extract_keywords(self, command: str) -> List[str]:
        """Extraer palabras clave del comando"""
        # Palabras comunes a ignorar
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "must", "can"
        }
        
        # Extraer palabras
        words = command.lower().split()
        keywords = [
            word.strip(".,!?;:()[]{}'\"")
            for word in words
            if len(word) > 2 and word.lower() not in stop_words
        ]
        
        return keywords[:10]  # Limitar a 10 palabras clave
    
    async def predict_success(self, command: str) -> Tuple[float, Dict]:
        """Predecir probabilidad de éxito"""
        keywords = self._extract_keywords(command)
        
        if not keywords:
            return 0.5, {}
        
        # Calcular probabilidad basada en patrones
        probabilities = []
        pattern_info = {}
        
        for keyword in keywords:
            if keyword in self.success_rates:
                prob = self.success_rates[keyword]
                probabilities.append(prob)
                pattern_info[keyword] = {
                    "success_rate": prob,
                    "count": self.patterns[keyword]["count"]
                }
        
        if probabilities:
            avg_prob = sum(probabilities) / len(probabilities)
        else:
            avg_prob = 0.5
        
        return avg_prob, pattern_info
    
    async def suggest_improvements(self, command: str) -> List[str]:
        """Sugerir mejoras para un comando"""
        suggestions = []
        
        # Analizar comando
        keywords = self._extract_keywords(command)
        
        # Buscar comandos similares exitosos
        similar_successful = [
            h for h in self.command_history[-100:]
            if h["success"] and any(kw in h["command"].lower() for kw in keywords)
        ]
        
        if similar_successful:
            # Encontrar patrones comunes en comandos exitosos
            common_patterns = self._find_common_patterns(similar_successful)
            if common_patterns:
                suggestions.append(f"Similar successful commands used: {', '.join(common_patterns[:3])}")
        
        # Sugerir basado en tasa de éxito
        low_success_keywords = [
            kw for kw in keywords
            if kw in self.success_rates and self.success_rates[kw] < 0.3
        ]
        
        if low_success_keywords:
            suggestions.append(
                f"Warning: Keywords '{', '.join(low_success_keywords)}' have low success rate"
            )
        
        return suggestions
    
    def _find_common_patterns(self, commands: List[Dict]) -> List[str]:
        """Encontrar patrones comunes en comandos"""
        # Extraer palabras clave comunes
        all_keywords = []
        for cmd in commands:
            keywords = self._extract_keywords(cmd["command"])
            all_keywords.extend(keywords)
        
        # Contar frecuencia
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        
        # Retornar más comunes
        return [kw for kw, count in keyword_counts.most_common(5)]
    
    async def get_statistics(self) -> Dict:
        """Obtener estadísticas de aprendizaje"""
        total_commands = len(self.command_history)
        successful = sum(1 for h in self.command_history if h["success"])
        
        avg_time = (
            sum(h["execution_time"] for h in self.command_history) / total_commands
            if total_commands > 0 else 0.0
        )
        
        return {
            "total_commands": total_commands,
            "successful_commands": successful,
            "failed_commands": total_commands - successful,
            "success_rate": successful / total_commands if total_commands > 0 else 0.0,
            "average_execution_time": avg_time,
            "patterns_learned": len(self.patterns),
            "top_patterns": self._get_top_patterns(10)
        }
    
    def _get_top_patterns(self, n: int) -> List[Dict]:
        """Obtener top N patrones"""
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return [
            {
                "keyword": keyword,
                "count": pattern["count"],
                "success_rate": pattern["success_count"] / pattern["count"],
                "avg_time": pattern["total_time"] / pattern["count"]
            }
            for keyword, pattern in sorted_patterns[:n]
        ]
    
    async def save(self):
        """Guardar patrones"""
        try:
            data = {
                "patterns": self.patterns,
                "success_rates": self.success_rates,
                "command_history": self.command_history[-100:]  # Guardar solo últimos 100
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    async def load(self):
        """Cargar patrones"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = data.get("patterns", {})
                    self.success_rates = data.get("success_rates", {})
                    self.command_history = data.get("command_history", [])
                logger.info(f"✅ Loaded {len(self.patterns)} patterns")
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")
    
    def clear(self):
        """Limpiar todos los patrones"""
        self.patterns.clear()
        self.success_rates.clear()
        self.command_history.clear()


