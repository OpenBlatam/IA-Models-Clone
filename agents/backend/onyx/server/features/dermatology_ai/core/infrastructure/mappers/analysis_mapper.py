from datetime import datetime
from typing import Dict, Any

from ...domain.entities import Analysis, SkinMetrics, Condition, SkinType, AnalysisStatus


class AnalysisMapper:
    
    @staticmethod
    def to_dict(analysis: Analysis) -> Dict[str, Any]:
        return {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "image_url": analysis.image_url,
            "video_url": analysis.video_url,
            "metrics": analysis.metrics.to_dict() if analysis.metrics else None,
            "conditions": [
                {
                    "name": c.name,
                    "confidence": c.confidence,
                    "severity": c.severity,
                    "description": c.description
                }
                for c in analysis.conditions
            ],
            "skin_type": analysis.skin_type.value if analysis.skin_type else None,
            "status": analysis.status.value,
            "created_at": analysis.created_at.isoformat(),
            "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
            "metadata": analysis.metadata,
        }
    
    @staticmethod
    def to_entity(data: Dict[str, Any]) -> Analysis:
        metrics = None
        if data.get("metrics"):
            metrics = SkinMetrics(**data["metrics"])
        
        conditions = [
            Condition(**c) for c in data.get("conditions", [])
        ]
        
        return Analysis(
            id=data["id"],
            user_id=data["user_id"],
            image_url=data.get("image_url"),
            video_url=data.get("video_url"),
            metrics=metrics,
            conditions=conditions,
            skin_type=SkinType(data["skin_type"]) if data.get("skin_type") else None,
            status=AnalysisStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {})
        )
    
    @staticmethod
    def to_update_dict(analysis: Analysis) -> Dict[str, Any]:
        return {
            "metrics": analysis.metrics.to_dict() if analysis.metrics else None,
            "conditions": [
                {
                    "name": c.name,
                    "confidence": c.confidence,
                    "severity": c.severity,
                    "description": c.description
                }
                for c in analysis.conditions
            ],
            "skin_type": analysis.skin_type.value if analysis.skin_type else None,
            "status": analysis.status.value,
            "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
            "metadata": analysis.metadata,
        }















