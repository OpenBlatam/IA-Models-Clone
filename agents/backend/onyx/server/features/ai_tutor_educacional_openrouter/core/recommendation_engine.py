"""
Intelligent recommendation engine for personalized learning paths.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Provides intelligent recommendations for personalized learning.
    """
    
    def __init__(self, learning_analyzer):
        self.learning_analyzer = learning_analyzer
        self.recommendation_cache: Dict[str, List[Dict]] = {}
    
    def get_learning_path_recommendation(
        self,
        student_id: str,
        subject: str,
        current_level: str = "intermedio"
    ) -> List[Dict[str, Any]]:
        """
        Get recommended learning path for a subject.
        
        Args:
            student_id: Student identifier
            subject: Subject area
            current_level: Current difficulty level
        
        Returns:
            List of recommended topics in order
        """
        profile = self.learning_analyzer.student_profiles.get(student_id, {})
        subject_data = profile.get("subjects", {}).get(subject, {})
        
        # Get topics the student has studied
        studied_topics = set(subject_data.keys())
        
        # Define learning paths by subject (simplified)
        learning_paths = self._get_default_learning_paths()
        subject_path = learning_paths.get(subject, [])
        
        # Filter out already studied topics
        recommended = [
            topic for topic in subject_path
            if topic["topic"] not in studied_topics
        ]
        
        # Sort by difficulty matching current level
        level_order = {"basico": 0, "intermedio": 1, "avanzado": 2}
        current_level_order = level_order.get(current_level, 1)
        
        recommended.sort(
            key=lambda x: abs(level_order.get(x.get("difficulty", "intermedio"), 1) - current_level_order)
        )
        
        return recommended[:10]  # Return top 10 recommendations
    
    def get_next_topic_recommendation(
        self,
        student_id: str,
        subject: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next recommended topic to study.
        
        Args:
            student_id: Student identifier
            subject: Subject area
        
        Returns:
            Next recommended topic
        """
        recommendations = self.get_learning_path_recommendation(student_id, subject)
        return recommendations[0] if recommendations else None
    
    def get_practice_recommendations(
        self,
        student_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get practice recommendations based on weaknesses.
        
        Args:
            student_id: Student identifier
        
        Returns:
            Practice recommendations by subject
        """
        strengths_weaknesses = self.learning_analyzer.get_strengths_and_weaknesses(student_id)
        weaknesses = strengths_weaknesses.get("weaknesses", {})
        
        recommendations = {}
        
        for subject, topics in weaknesses.items():
            recommendations[subject] = [
                {
                    "topic": topic,
                    "type": "practice",
                    "priority": "high",
                    "reason": "Área de mejora identificada",
                    "suggested_exercises": 5,
                    "difficulty": self.learning_analyzer.get_recommended_difficulty(
                        student_id, subject, topic
                    )
                }
                for topic in topics[:5]  # Top 5 weak topics per subject
            ]
        
        return recommendations
    
    def get_resource_recommendations(
        self,
        student_id: str,
        topic: str,
        subject: str
    ) -> List[Dict[str, Any]]:
        """
        Get recommended learning resources for a topic.
        
        Args:
            student_id: Student identifier
            topic: Topic to study
            subject: Subject area
        
        Returns:
            List of recommended resources
        """
        difficulty = self.learning_analyzer.get_recommended_difficulty(
            student_id, subject, topic
        )
        
        resources = [
            {
                "type": "explanation",
                "title": f"Explicación de {topic}",
                "description": f"Conceptos fundamentales de {topic}",
                "difficulty": difficulty,
                "estimated_time": "15 minutos"
            },
            {
                "type": "exercises",
                "title": f"Ejercicios de práctica: {topic}",
                "description": f"Ejercicios para practicar {topic}",
                "difficulty": difficulty,
                "estimated_time": "30 minutos",
                "num_exercises": 10
            },
            {
                "type": "quiz",
                "title": f"Quiz: {topic}",
                "description": f"Evalúa tu conocimiento de {topic}",
                "difficulty": difficulty,
                "estimated_time": "20 minutos",
                "num_questions": 10
            }
        ]
        
        return resources
    
    def get_adaptive_difficulty(
        self,
        student_id: str,
        subject: str,
        topic: str,
        recent_performance: float
    ) -> str:
        """
        Recommend adaptive difficulty based on performance.
        
        Args:
            student_id: Student identifier
            subject: Subject area
            topic: Topic
            recent_performance: Recent performance score (0.0-1.0)
        
        Returns:
            Recommended difficulty level
        """
        current_difficulty = self.learning_analyzer.get_recommended_difficulty(
            student_id, subject, topic
        )
        
        # Adjust based on recent performance
        if recent_performance >= 0.9:
            # Excellent performance, suggest more challenging
            if current_difficulty == "basico":
                return "intermedio"
            elif current_difficulty == "intermedio":
                return "avanzado"
        elif recent_performance < 0.5:
            # Poor performance, suggest easier
            if current_difficulty == "avanzado":
                return "intermedio"
            elif current_difficulty == "intermedio":
                return "basico"
        
        return current_difficulty
    
    def _get_default_learning_paths(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get default learning paths by subject."""
        return {
            "matematicas": [
                {"topic": "aritmetica basica", "difficulty": "basico", "order": 1},
                {"topic": "algebra elemental", "difficulty": "basico", "order": 2},
                {"topic": "geometria basica", "difficulty": "basico", "order": 3},
                {"topic": "ecuaciones lineales", "difficulty": "intermedio", "order": 4},
                {"topic": "ecuaciones cuadraticas", "difficulty": "intermedio", "order": 5},
                {"topic": "trigonometria", "difficulty": "intermedio", "order": 6},
                {"topic": "calculo diferencial", "difficulty": "avanzado", "order": 7},
                {"topic": "calculo integral", "difficulty": "avanzado", "order": 8},
            ],
            "programacion": [
                {"topic": "variables y tipos", "difficulty": "basico", "order": 1},
                {"topic": "estructuras de control", "difficulty": "basico", "order": 2},
                {"topic": "funciones", "difficulty": "intermedio", "order": 3},
                {"topic": "estructuras de datos", "difficulty": "intermedio", "order": 4},
                {"topic": "programacion orientada a objetos", "difficulty": "avanzado", "order": 5},
                {"topic": "algoritmos", "difficulty": "avanzado", "order": 6},
            ],
            "ciencias": [
                {"topic": "celulas", "difficulty": "basico", "order": 1},
                {"topic": "fotosintesis", "difficulty": "basico", "order": 2},
                {"topic": "genetica basica", "difficulty": "intermedio", "order": 3},
                {"topic": "ecosistemas", "difficulty": "intermedio", "order": 4},
                {"topic": "evolucion", "difficulty": "avanzado", "order": 5},
            ]
        }






