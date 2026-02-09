"""
ML Learner - Sistema de aprendizaje automático mejorado
========================================================
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MLLearner:
    """
    Sistema de aprendizaje automático que mejora basándose en feedback y uso.
    """
    
    def __init__(self, learning_dir: str = "data/learning"):
        """
        Inicializar sistema de aprendizaje.
        
        Args:
            learning_dir: Directorio para almacenar datos de aprendizaje
        """
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_data: Dict[str, Any] = {
            "patterns": defaultdict(int),
            "successful_improvements": [],
            "failed_improvements": [],
            "paper_effectiveness": defaultdict(float),
            "code_patterns": defaultdict(list)
        }
        self._load_learning_data()
    
    def learn_from_improvement(
        self,
        improvement_result: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ):
        """
        Aprende de una mejora realizada.
        
        Args:
            improvement_result: Resultado de la mejora
            feedback: Feedback del usuario (opcional)
        """
        try:
            # Extraer información de la mejora
            papers_used = improvement_result.get("papers_used", [])
            improvements_applied = improvement_result.get("improvements_applied", 0)
            suggestions = improvement_result.get("suggestions", [])
            
            # Determinar si fue exitosa
            is_successful = False
            if feedback:
                rating = feedback.get("rating", 0)
                is_successful = rating >= 4
            else:
                # Usar heurística: si hay mejoras aplicadas, probablemente exitosa
                is_successful = improvements_applied > 0
            
            # Aprender de papers usados
            for paper in papers_used:
                paper_title = paper.get("title", "")
                if paper_title:
                    if is_successful:
                        self.learning_data["paper_effectiveness"][paper_title] += 0.1
                    else:
                        self.learning_data["paper_effectiveness"][paper_title] -= 0.05
                    
                    # Mantener en rango [0, 1]
                    self.learning_data["paper_effectiveness"][paper_title] = max(
                        0, min(1, self.learning_data["paper_effectiveness"][paper_title])
                    )
            
            # Aprender de sugerencias exitosas
            for suggestion in suggestions:
                suggestion_type = suggestion.get("type", "")
                if suggestion_type and is_successful:
                    self.learning_data["patterns"][suggestion_type] += 1
            
            # Guardar en historial
            if is_successful:
                self.learning_data["successful_improvements"].append({
                    "timestamp": datetime.now().isoformat(),
                    "papers_used": [p.get("title", "") for p in papers_used],
                    "improvements_count": improvements_applied
                })
            else:
                self.learning_data["failed_improvements"].append({
                    "timestamp": datetime.now().isoformat(),
                    "papers_used": [p.get("title", "") for p in papers_used],
                    "improvements_count": improvements_applied
                })
            
            # Mantener historial limitado
            if len(self.learning_data["successful_improvements"]) > 1000:
                self.learning_data["successful_improvements"] = \
                    self.learning_data["successful_improvements"][-1000:]
            
            if len(self.learning_data["failed_improvements"]) > 500:
                self.learning_data["failed_improvements"] = \
                    self.learning_data["failed_improvements"][-500:]
            
            self._save_learning_data()
            
            logger.info(f"Aprendizaje registrado: {'exitoso' if is_successful else 'fallido'}")
            
        except Exception as e:
            logger.error(f"Error en aprendizaje: {e}")
    
    def get_best_papers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene los papers más efectivos basándose en aprendizaje.
        
        Args:
            limit: Número de papers a retornar
            
        Returns:
            Lista de papers más efectivos
        """
        sorted_papers = sorted(
            self.learning_data["paper_effectiveness"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "paper": paper,
                "effectiveness_score": score,
                "usage_count": self._get_paper_usage_count(paper)
            }
            for paper, score in sorted_papers
        ]
    
    def get_effective_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene los patrones más efectivos.
        
        Args:
            limit: Número de patrones
            
        Returns:
            Lista de patrones más efectivos
        """
        sorted_patterns = sorted(
            self.learning_data["patterns"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {
                "pattern": pattern,
                "success_count": count
            }
            for pattern, count in sorted_patterns
        ]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de aprendizaje"""
        total_improvements = len(self.learning_data["successful_improvements"]) + \
                           len(self.learning_data["failed_improvements"])
        
        success_rate = 0
        if total_improvements > 0:
            success_rate = len(self.learning_data["successful_improvements"]) / total_improvements
        
        return {
            "total_improvements_learned": total_improvements,
            "successful": len(self.learning_data["successful_improvements"]),
            "failed": len(self.learning_data["failed_improvements"]),
            "success_rate": round(success_rate * 100, 2),
            "papers_tracked": len(self.learning_data["paper_effectiveness"]),
            "patterns_learned": len(self.learning_data["patterns"])
        }
    
    def _get_paper_usage_count(self, paper_title: str) -> int:
        """Obtiene número de veces que se usó un paper"""
        count = 0
        for improvement in self.learning_data["successful_improvements"]:
            if paper_title in improvement.get("papers_used", []):
                count += 1
        for improvement in self.learning_data["failed_improvements"]:
            if paper_title in improvement.get("papers_used", []):
                count += 1
        return count
    
    def _load_learning_data(self):
        """Carga datos de aprendizaje desde disco"""
        learning_file = self.learning_dir / "learning_data.json"
        
        if learning_file.exists():
            try:
                with open(learning_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.learning_data.update(data)
                logger.info("Datos de aprendizaje cargados")
            except Exception as e:
                logger.error(f"Error cargando datos de aprendizaje: {e}")
    
    def _save_learning_data(self):
        """Guarda datos de aprendizaje en disco"""
        try:
            learning_file = self.learning_dir / "learning_data.json"
            with open(learning_file, "w", encoding="utf-8") as f:
                json.dump(self.learning_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando datos de aprendizaje: {e}")




