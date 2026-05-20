"""
Quiz generator for creating assessments and practice tests.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class QuizGenerator:
    """
    Generates quizzes and assessments for students.
    """
    
    def __init__(self, tutor):
        self.tutor = tutor
    
    async def generate_quiz(
        self,
        topic: str,
        subject: str,
        difficulty: str = "intermedio",
        num_questions: int = 10,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete quiz with questions and answers.
        
        Args:
            topic: Topic for the quiz
            subject: Subject area
            difficulty: Difficulty level
            num_questions: Number of questions
            question_types: Types of questions (multiple_choice, true_false, short_answer)
        
        Returns:
            Quiz with questions, answers, and explanations
        """
        if question_types is None:
            question_types = ["multiple_choice", "short_answer"]
        
        prompt = f"""Genera un quiz completo sobre {topic} en el área de {subject}.

Requisitos:
- Número de preguntas: {num_questions}
- Nivel de dificultad: {difficulty}
- Tipos de preguntas: {', '.join(question_types)}
- Incluye respuestas correctas
- Incluye explicaciones para cada respuesta
- Formato: JSON con estructura clara

Estructura esperada:
{{
  "quiz_title": "Título del quiz",
  "topic": "{topic}",
  "subject": "{subject}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "id": 1,
      "type": "multiple_choice",
      "question": "Pregunta aquí",
      "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
      "correct_answer": "Opción A",
      "explanation": "Explicación de por qué es correcta"
    }}
  ]
}}"""

        try:
            response = await self.tutor.ask_question(
                question=prompt,
                subject=subject,
                difficulty=difficulty
            )
            
            return {
                "quiz": response["answer"],
                "topic": topic,
                "subject": subject,
                "difficulty": difficulty,
                "num_questions": num_questions,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            raise
    
    async def generate_practice_test(
        self,
        topics: List[str],
        subject: str,
        difficulty: str = "intermedio",
        questions_per_topic: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a practice test covering multiple topics.
        
        Args:
            topics: List of topics to cover
            subject: Subject area
            difficulty: Difficulty level
            questions_per_topic: Questions per topic
        
        Returns:
            Practice test with questions from all topics
        """
        all_questions = []
        
        for topic in topics:
            quiz = await self.generate_quiz(
                topic=topic,
                subject=subject,
                difficulty=difficulty,
                num_questions=questions_per_topic
            )
            all_questions.append(quiz)
        
        return {
            "test_title": f"Práctica de {subject}",
            "topics": topics,
            "subject": subject,
            "difficulty": difficulty,
            "total_questions": len(topics) * questions_per_topic,
            "sections": all_questions,
            "generated_at": datetime.now().isoformat()
        }






