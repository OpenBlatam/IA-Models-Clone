"""
Sistema de historial y tracking de análisis de piel
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class AnalysisRecord:
    """Registro de un análisis"""
    id: str
    timestamp: str
    user_id: Optional[str]
    analysis_type: str  # "image" o "video"
    quality_scores: Dict
    conditions: List[Dict]
    skin_type: str
    recommendations_priority: List[str]
    image_hash: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return asdict(self)


class HistoryTracker:
    """Sistema de tracking de historial de análisis"""
    
    def __init__(self, storage_dir: str = "history"):
        """
        Inicializa el tracker
        
        Args:
            storage_dir: Directorio para almacenar historial
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.user_records: Dict[str, List[str]] = {}  # user_id -> [record_ids]
    
    def save_analysis(self, analysis_result: Dict, user_id: Optional[str] = None,
                     image_hash: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> str:
        """
        Guarda un análisis en el historial
        
        Args:
            analysis_result: Resultado del análisis
            user_id: ID del usuario (opcional)
            image_hash: Hash de la imagen (opcional)
            metadata: Metadatos adicionales
            
        Returns:
            ID del registro creado
        """
        # Generar ID único
        record_id = self._generate_id(analysis_result, user_id)
        
        # Crear registro
        record = AnalysisRecord(
            id=record_id,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            analysis_type=analysis_result.get("analysis_type", "image"),
            quality_scores=analysis_result.get("quality_scores", {}),
            conditions=analysis_result.get("conditions", []),
            skin_type=analysis_result.get("skin_type", "unknown"),
            recommendations_priority=analysis_result.get("recommendations_priority", []),
            image_hash=image_hash,
            metadata=metadata or {}
        )
        
        # Guardar en archivo
        record_file = self.storage_dir / f"{record_id}.json"
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Actualizar índice de usuario
        if user_id:
            if user_id not in self.user_records:
                self.user_records[user_id] = []
            if record_id not in self.user_records[user_id]:
                self.user_records[user_id].append(record_id)
        
        return record_id
    
    def get_user_history(self, user_id: str, limit: int = 50) -> List[AnalysisRecord]:
        """
        Obtiene historial de un usuario
        
        Args:
            user_id: ID del usuario
            limit: Límite de registros a retornar
            
        Returns:
            Lista de registros ordenados por fecha (más reciente primero)
        """
        if user_id not in self.user_records:
            return []
        
        records = []
        for record_id in self.user_records[user_id][-limit:]:
            record = self.get_record(record_id)
            if record:
                records.append(record)
        
        # Ordenar por timestamp
        records.sort(key=lambda x: x.timestamp, reverse=True)
        return records
    
    def get_record(self, record_id: str) -> Optional[AnalysisRecord]:
        """
        Obtiene un registro específico
        
        Args:
            record_id: ID del registro
            
        Returns:
            Registro o None si no existe
        """
        record_file = self.storage_dir / f"{record_id}.json"
        if not record_file.exists():
            return None
        
        try:
            with open(record_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return AnalysisRecord(**data)
        except Exception:
            return None
    
    def compare_analyses(self, record_id1: str, record_id2: str) -> Dict:
        """
        Compara dos análisis
        
        Args:
            record_id1: ID del primer análisis
            record_id2: ID del segundo análisis
            
        Returns:
            Diccionario con comparación
        """
        record1 = self.get_record(record_id1)
        record2 = self.get_record(record_id2)
        
        if not record1 or not record2:
            raise ValueError("Uno o ambos registros no existen")
        
        scores1 = record1.quality_scores
        scores2 = record2.quality_scores
        
        # Calcular diferencias
        differences = {}
        for key in scores1:
            if key in scores2:
                diff = scores2[key] - scores1[key]
                differences[key] = {
                    "before": scores1[key],
                    "after": scores2[key],
                    "difference": round(diff, 2),
                    "improvement": diff > 0
                }
        
        # Comparar condiciones
        conditions1 = {c.get("name"): c for c in record1.conditions}
        conditions2 = {c.get("name"): c for c in record2.conditions}
        
        conditions_comparison = {
            "resolved": [],  # Condiciones que desaparecieron
            "improved": [],  # Condiciones que mejoraron
            "worsened": [],  # Condiciones que empeoraron
            "new": []  # Condiciones nuevas
        }
        
        for name, cond2 in conditions2.items():
            if name not in conditions1:
                conditions_comparison["new"].append(cond2)
            else:
                cond1 = conditions1[name]
                if cond2.get("severity") != cond1.get("severity"):
                    if self._severity_improved(cond1.get("severity"), cond2.get("severity")):
                        conditions_comparison["improved"].append({
                            "name": name,
                            "before": cond1,
                            "after": cond2
                        })
                    else:
                        conditions_comparison["worsened"].append({
                            "name": name,
                            "before": cond1,
                            "after": cond2
                        })
        
        for name, cond1 in conditions1.items():
            if name not in conditions2:
                conditions_comparison["resolved"].append(cond1)
        
        # Calcular score general de mejora
        overall_improvement = sum(
            diff["difference"] for diff in differences.values()
        ) / len(differences) if differences else 0
        
        return {
            "record1": {
                "id": record_id1,
                "timestamp": record1.timestamp,
                "overall_score": scores1.get("overall_score", 0)
            },
            "record2": {
                "id": record_id2,
                "timestamp": record2.timestamp,
                "overall_score": scores2.get("overall_score", 0)
            },
            "score_differences": differences,
            "overall_improvement": round(overall_improvement, 2),
            "conditions_comparison": conditions_comparison,
            "time_span_days": self._days_between(record1.timestamp, record2.timestamp)
        }
    
    def get_progress_timeline(self, user_id: str, metric: str = "overall_score") -> List[Dict]:
        """
        Obtiene línea de tiempo de progreso para una métrica
        
        Args:
            user_id: ID del usuario
            metric: Métrica a trackear
            
        Returns:
            Lista de puntos en el tiempo
        """
        history = self.get_user_history(user_id, limit=100)
        
        timeline = []
        for record in history:
            score = record.quality_scores.get(metric, 0)
            timeline.append({
                "timestamp": record.timestamp,
                "value": score,
                "record_id": record.id
            })
        
        return timeline
    
    def _generate_id(self, analysis_result: Dict, user_id: Optional[str]) -> str:
        """Genera ID único para el registro"""
        data = json.dumps({
            "scores": analysis_result.get("quality_scores", {}),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
    
    def _severity_improved(self, severity1: str, severity2: str) -> bool:
        """Determina si la severidad mejoró"""
        severity_levels = {"mild": 1, "moderate": 2, "severe": 3}
        level1 = severity_levels.get(severity1, 2)
        level2 = severity_levels.get(severity2, 2)
        return level2 < level1
    
    def _days_between(self, timestamp1: str, timestamp2: str) -> float:
        """Calcula días entre dos timestamps"""
        dt1 = datetime.fromisoformat(timestamp1)
        dt2 = datetime.fromisoformat(timestamp2)
        delta = abs(dt2 - dt1)
        return delta.total_seconds() / 86400  # días






