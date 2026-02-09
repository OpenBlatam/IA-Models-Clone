"""
Learning analyzer for tracking student progress and adapting teaching.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class LearningAnalyzer:
    """
    Analyzes student learning patterns and adapts teaching approach.
    """
    
    def __init__(self):
        self.student_profiles: Dict[str, Dict] = {}
        self.topic_mastery: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.learning_paths: Dict[str, List[str]] = defaultdict(list)
    
    def update_student_profile(
        self,
        student_id: str,
        subject: str,
        topic: str,
        performance: float,
        difficulty: str
    ):
        """Update student learning profile."""
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = {
                "subjects": defaultdict(dict),
                "overall_performance": 0.0,
                "learning_style": "balanced",
                "last_updated": datetime.now().isoformat()
            }
        
        profile = self.student_profiles[student_id]
        profile["subjects"][subject][topic] = {
            "performance": performance,
            "difficulty": difficulty,
            "last_practiced": datetime.now().isoformat()
        }
        
        self.topic_mastery[student_id][f"{subject}:{topic}"] = performance
        profile["last_updated"] = datetime.now().isoformat()
    
    def get_recommended_difficulty(
        self,
        student_id: str,
        subject: str,
        topic: str
    ) -> str:
        """Get recommended difficulty level for student."""
        mastery = self.topic_mastery[student_id].get(f"{subject}:{topic}", 0.5)
        
        if mastery < 0.4:
            return "basico"
        elif mastery < 0.7:
            return "intermedio"
        else:
            return "avanzado"
    
    def get_learning_path(self, student_id: str, subject: str) -> List[str]:
        """Get recommended learning path for student."""
        if student_id in self.learning_paths:
            return self.learning_paths[student_id]
        
        return []
    
    def get_strengths_and_weaknesses(
        self,
        student_id: str
    ) -> Dict[str, Dict[str, List[str]]]:
        """Analyze student strengths and weaknesses."""
        if student_id not in self.student_profiles:
            return {"strengths": {}, "weaknesses": {}}
        
        strengths = defaultdict(list)
        weaknesses = defaultdict(list)
        
        for subject, topics in self.student_profiles[student_id]["subjects"].items():
            for topic, data in topics.items():
                performance = data.get("performance", 0.0)
                if performance >= 0.7:
                    strengths[subject].append(topic)
                elif performance < 0.4:
                    weaknesses[subject].append(topic)
        
        return {
            "strengths": dict(strengths),
            "weaknesses": dict(weaknesses)
        }






