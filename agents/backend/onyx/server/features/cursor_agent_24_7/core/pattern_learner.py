"""
Pattern Learner - Sistema de aprendizaje de patrones
=====================================================

Aprende patrones de comandos exitosos para mejorar la ejecución
y predecir la probabilidad de éxito de nuevos comandos.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# Palabras comunes a ignorar en extracción de keywords
STOP_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "may", "might", "must", "can"
}


@dataclass
class PatternData:
    """
    Datos de un patrón aprendido.
    
    Attributes:
        count: Número de veces que se ha visto este patrón.
        success_count: Número de veces que fue exitoso.
        total_time: Tiempo total de ejecución acumulado.
        last_seen: Timestamp ISO de la última vez que se vio.
    """
    count: int = 0
    success_count: int = 0
    total_time: float = 0.0
    last_seen: Optional[str] = None


@dataclass
class CommandHistoryEntry:
    """
    Entrada en el historial de comandos.
    
    Attributes:
        command: Comando ejecutado.
        success: Si fue exitoso.
        execution_time: Tiempo de ejecución en segundos.
        timestamp: Timestamp ISO de ejecución.
        result_length: Longitud del resultado.
    """
    command: str
    success: bool
    execution_time: float
    timestamp: str
    result_length: int


class PatternLearner:
    """
    Sistema de aprendizaje de patrones de comandos.
    
    Aprende de comandos ejecutados para predecir probabilidad de éxito
    y sugerir mejoras basadas en patrones históricos.
    """
    
    def __init__(
        self,
        storage_path: str = "./data/patterns.json",
        max_history: int = 1000
    ) -> None:
        """
        Inicializar pattern learner.
        
        Args:
            storage_path: Ruta donde guardar los patrones (default: "./data/patterns.json").
            max_history: Máximo número de comandos en historial (default: 1000).
        
        Raises:
            ValueError: Si max_history es inválido.
        """
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        
        self.storage_path: Path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.patterns: Dict[str, PatternData] = {}
        self.command_history: List[CommandHistoryEntry] = []
        self.success_rates: Dict[str, float] = {}
        self._max_history: int = max_history
    
    async def record_command(
        self,
        command: str,
        success: bool,
        execution_time: float,
        result: Optional[str] = None
    ) -> None:
        """
        Registrar un comando ejecutado.
        
        Args:
            command: Comando ejecutado.
            success: Si fue exitoso.
            execution_time: Tiempo de ejecución en segundos.
            result: Resultado del comando (opcional).
        
        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")
        if execution_time < 0:
            raise ValueError("execution_time must be non-negative")
        
        # Agregar a historial
        entry = CommandHistoryEntry(
            command=command,
            success=success,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            result_length=len(result) if result else 0
        )
        self.command_history.append(entry)
        
        # Mantener historial limitado
        if len(self.command_history) > self._max_history:
            self.command_history = self.command_history[-self._max_history:]
        
        # Actualizar patrones
        await self._update_patterns(command, success, execution_time)
        
        # Guardar periódicamente
        if len(self.command_history) % 10 == 0:
            await self.save()
    
    async def _update_patterns(
        self,
        command: str,
        success: bool,
        execution_time: float
    ) -> None:
        """
        Actualizar patrones aprendidos basados en un comando.
        
        Args:
            command: Comando ejecutado.
            success: Si fue exitoso.
            execution_time: Tiempo de ejecución.
        """
        keywords = self._extract_keywords(command)
        
        for keyword in keywords:
            if keyword not in self.patterns:
                self.patterns[keyword] = PatternData()
            
            pattern = self.patterns[keyword]
            pattern.count += 1
            if success:
                pattern.success_count += 1
            pattern.total_time += execution_time
            pattern.last_seen = datetime.now().isoformat()
            
            # Calcular tasa de éxito
            self.success_rates[keyword] = (
                pattern.success_count / pattern.count
                if pattern.count > 0
                else 0.0
            )
    
    def _extract_keywords(self, command: str, max_keywords: int = 10) -> List[str]:
        """
        Extraer palabras clave del comando.
        
        Args:
            command: Comando a analizar.
            max_keywords: Máximo número de keywords a retornar (default: 10).
        
        Returns:
            Lista de palabras clave extraídas.
        """
        if not command or not command.strip():
            return []
        
        # Extraer palabras
        words = command.lower().split()
        keywords = [
            word.strip(".,!?;:()[]{}'\"")
            for word in words
            if len(word) > 2 and word.lower() not in STOP_WORDS
        ]
        
        # Remover duplicados manteniendo orden
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
                if len(unique_keywords) >= max_keywords:
                    break
        
        return unique_keywords
    
    async def predict_success(
        self,
        command: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predecir probabilidad de éxito de un comando.
        
        Args:
            command: Comando a predecir.
        
        Returns:
            Tupla con (probabilidad, información de patrones).
        """
        if not command or not command.strip():
            return 0.5, {}
        
        keywords = self._extract_keywords(command)
        
        if not keywords:
            return 0.5, {}
        
        # Calcular probabilidad basada en patrones
        probabilities: List[float] = []
        pattern_info: Dict[str, Any] = {}
        
        for keyword in keywords:
            if keyword in self.success_rates:
                prob = self.success_rates[keyword]
                probabilities.append(prob)
                pattern_info[keyword] = {
                    "success_rate": prob,
                    "count": self.patterns[keyword].count
                }
        
        if probabilities:
            avg_prob = sum(probabilities) / len(probabilities)
        else:
            avg_prob = 0.5
        
        return avg_prob, pattern_info
    
    async def suggest_improvements(self, command: str) -> List[str]:
        """
        Sugerir mejoras para un comando basado en patrones históricos.
        
        Args:
            command: Comando a analizar.
        
        Returns:
            Lista de sugerencias de mejora.
        """
        if not command or not command.strip():
            return []
        
        suggestions: List[str] = []
        keywords = self._extract_keywords(command)
        
        if not keywords:
            return suggestions
        
        # Buscar comandos similares exitosos
        recent_history = self.command_history[-100:]
        similar_successful = [
            entry
            for entry in recent_history
            if entry.success and any(
                kw in entry.command.lower() for kw in keywords
            )
        ]
        
        if similar_successful:
            common_patterns = self._find_common_patterns(similar_successful)
            if common_patterns:
                suggestions.append(
                    f"Similar successful commands used: "
                    f"{', '.join(common_patterns[:3])}"
                )
        
        # Sugerir basado en tasa de éxito
        low_success_keywords = [
            kw for kw in keywords
            if kw in self.success_rates and self.success_rates[kw] < 0.3
        ]
        
        if low_success_keywords:
            suggestions.append(
                f"Warning: Keywords '{', '.join(low_success_keywords)}' "
                f"have low success rate"
            )
        
        return suggestions
    
    def _find_common_patterns(
        self,
        commands: List[CommandHistoryEntry]
    ) -> List[str]:
        """
        Encontrar patrones comunes en comandos exitosos.
        
        Args:
            commands: Lista de comandos a analizar.
        
        Returns:
            Lista de keywords más comunes.
        """
        if not commands:
            return []
        
        all_keywords: List[str] = []
        for entry in commands:
            keywords = self._extract_keywords(entry.command)
            all_keywords.extend(keywords)
        
        # Contar frecuencia
        keyword_counts = Counter(all_keywords)
        
        # Retornar más comunes
        return [kw for kw, count in keyword_counts.most_common(5)]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de aprendizaje.
        
        Returns:
            Diccionario con estadísticas.
        """
        total_commands = len(self.command_history)
        successful = sum(1 for entry in self.command_history if entry.success)
        
        avg_time = (
            sum(entry.execution_time for entry in self.command_history) / total_commands
            if total_commands > 0
            else 0.0
        )
        
        return {
            "total_commands": total_commands,
            "successful_commands": successful,
            "failed_commands": total_commands - successful,
            "success_rate": (
                successful / total_commands
                if total_commands > 0
                else 0.0
            ),
            "average_execution_time": avg_time,
            "patterns_learned": len(self.patterns),
            "top_patterns": self._get_top_patterns(10)
        }
    
    def _get_top_patterns(self, n: int) -> List[Dict[str, Any]]:
        """
        Obtener top N patrones por frecuencia.
        
        Args:
            n: Número de patrones a retornar.
        
        Returns:
            Lista de diccionarios con información de patrones.
        """
        if n <= 0:
            return []
        
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1].count,
            reverse=True
        )
        
        return [
            {
                "keyword": keyword,
                "count": pattern.count,
                "success_rate": (
                    pattern.success_count / pattern.count
                    if pattern.count > 0
                    else 0.0
                ),
                "avg_time": (
                    pattern.total_time / pattern.count
                    if pattern.count > 0
                    else 0.0
                )
            }
            for keyword, pattern in sorted_patterns[:n]
        ]
    
    async def save(self) -> None:
        """
        Guardar patrones en disco.
        
        Raises:
            RuntimeError: Si hay error al guardar.
        """
        try:
            data = {
                "patterns": {
                    keyword: {
                        "count": pattern.count,
                        "success_count": pattern.success_count,
                        "total_time": pattern.total_time,
                        "last_seen": pattern.last_seen
                    }
                    for keyword, pattern in self.patterns.items()
                },
                "success_rates": self.success_rates,
                "command_history": [
                    {
                        "command": entry.command,
                        "success": entry.success,
                        "execution_time": entry.execution_time,
                        "timestamp": entry.timestamp,
                        "result_length": entry.result_length
                    }
                    for entry in self.command_history[-100:]  # Guardar solo últimos 100
                ]
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Patterns saved to {self.storage_path}")
        
        except Exception as e:
            logger.error(f"Error saving patterns: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save patterns: {e}") from e
    
    async def load(self) -> None:
        """
        Cargar patrones desde disco.
        
        Si hay error al cargar, continúa con patrones vacíos.
        """
        try:
            if not self.storage_path.exists():
                logger.debug(f"Pattern file not found: {self.storage_path}")
                return
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Cargar patrones
            patterns_data = data.get("patterns", {})
            self.patterns = {
                keyword: PatternData(
                    count=pattern.get("count", 0),
                    success_count=pattern.get("success_count", 0),
                    total_time=pattern.get("total_time", 0.0),
                    last_seen=pattern.get("last_seen")
                )
                for keyword, pattern in patterns_data.items()
            }
            
            # Cargar success rates
            self.success_rates = data.get("success_rates", {})
            
            # Cargar historial
            history_data = data.get("command_history", [])
            self.command_history = [
                CommandHistoryEntry(
                    command=entry["command"],
                    success=entry["success"],
                    execution_time=entry["execution_time"],
                    timestamp=entry["timestamp"],
                    result_length=entry.get("result_length", 0)
                )
                for entry in history_data
            ]
            
            logger.info(f"✅ Loaded {len(self.patterns)} patterns")
        
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}", exc_info=True)
            # Continuar con patrones vacíos
    
    def clear(self) -> None:
        """Limpiar todos los patrones y el historial."""
        self.patterns.clear()
        self.success_rates.clear()
        self.command_history.clear()
        logger.info("Patterns cleared")
