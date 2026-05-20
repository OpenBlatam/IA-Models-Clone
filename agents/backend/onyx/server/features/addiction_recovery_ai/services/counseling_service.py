"""
Servicio de consejería - Proporciona coaching y apoyo personalizado
"""

from typing import Dict, List, Optional
from datetime import datetime
import json


class CounselingService:
    """Servicio de consejería y coaching"""
    
    def __init__(self, openai_client=None):
        """
        Inicializa el servicio de consejería
        
        Args:
            openai_client: Cliente de OpenAI (opcional)
        """
        self.openai_client = openai_client
    
    def create_coaching_session(
        self,
        user_id: str,
        topic: str,
        current_situation: str,
        questions: Optional[List[str]] = None
    ) -> Dict:
        """
        Crea una sesión de coaching personalizada
        
        Args:
            user_id: ID del usuario
            topic: Tema de la sesión
            current_situation: Situación actual del usuario
            questions: Preguntas específicas (opcional)
        
        Returns:
            Sesión de coaching
        """
        session = {
            "user_id": user_id,
            "topic": topic,
            "created_at": datetime.now().isoformat(),
            "guidance": self._generate_guidance(topic, current_situation),
            "questions_to_consider": questions or self._generate_questions(topic),
            "action_items": self._generate_action_items(topic, current_situation),
            "encouragement": self._generate_encouragement()
        }
        
        # Si hay cliente de OpenAI, mejorar con IA
        if self.openai_client:
            enhanced = self._enhance_with_ai(topic, current_situation, questions)
            session.update(enhanced)
        
        return session
    
    def _generate_guidance(self, topic: str, situation: str) -> str:
        """Genera guía básica para el tema"""
        guidance_templates = {
            "cravings": "Los cravings son temporales. Recuerda que la mayoría pasan en 15-20 minutos. Usa técnicas de distracción y respiración.",
            "relapse": "Si has tenido una recaída, no es el fin del mundo. Aprende de la experiencia y continúa. La recuperación es un proceso.",
            "motivation": "Recuerda tus razones para estar sobrio. Escribe una lista y léela cuando sientas que pierdes motivación.",
            "stress": "El estrés es un trigger común. Desarrolla técnicas de manejo de estrés como ejercicio, meditación o hablar con alguien.",
            "social": "Las situaciones sociales pueden ser desafiantes. Ten un plan antes de ir, lleva apoyo si es posible, y ten una salida preparada."
        }
        
        topic_lower = topic.lower()
        for key, guidance in guidance_templates.items():
            if key in topic_lower:
                return guidance
        
        return "Continúa trabajando en tu recuperación. Cada día es una oportunidad de crecimiento."
    
    def _generate_questions(self, topic: str) -> List[str]:
        """Genera preguntas reflexivas"""
        return [
            "¿Qué puedo hacer ahora mismo para mejorar mi situación?",
            "¿Qué he aprendido de experiencias pasadas?",
            "¿Quién puedo contactar para apoyo?",
            "¿Qué estrategias han funcionado mejor para mí?",
            "¿Cómo puedo cuidarme mejor hoy?"
        ]
    
    def _generate_action_items(self, topic: str, situation: str) -> List[Dict]:
        """Genera elementos de acción"""
        return [
            {
                "action": "Contactar sistema de apoyo",
                "priority": "high",
                "description": "Llama o contacta a alguien de tu red de apoyo"
            },
            {
                "action": "Practicar técnica de relajación",
                "priority": "medium",
                "description": "Usa respiración profunda o meditación"
            },
            {
                "action": "Revisar plan de recuperación",
                "priority": "medium",
                "description": "Revisa y ajusta tu plan si es necesario"
            }
        ]
    
    def _generate_encouragement(self) -> str:
        """Genera mensaje de aliento"""
        encouragements = [
            "Estás haciendo un trabajo increíble. Cada día cuenta.",
            "La recuperación es un viaje, no un destino. Sigue adelante.",
            "Eres más fuerte de lo que crees. Has llegado hasta aquí.",
            "Cada momento de resistencia te hace más fuerte.",
            "No estás solo en esto. Hay personas que te apoyan."
        ]
        
        import random
        return random.choice(encouragements)
    
    def _enhance_with_ai(
        self,
        topic: str,
        situation: str,
        questions: Optional[List[str]]
    ) -> Dict:
        """Mejora la sesión usando IA"""
        if not self.openai_client:
            return {}
        
        try:
            prompt = f"""
            Eres un consejero experto en adicciones. Proporciona apoyo compasivo y práctico.
            
            Tema: {topic}
            Situación: {situation}
            
            Proporciona:
            1. Un mensaje de apoyo personalizado
            2. Estrategias específicas para esta situación
            3. Preguntas reflexivas adicionales
            
            Responde en formato JSON con: support_message, specific_strategies, reflective_questions
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un consejero compasivo y experto en recuperación de adicciones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            try:
                ai_data = json.loads(ai_response)
                return {
                    "ai_support_message": ai_data.get("support_message", ""),
                    "ai_specific_strategies": ai_data.get("specific_strategies", []),
                    "ai_reflective_questions": ai_data.get("reflective_questions", [])
                }
            except json.JSONDecodeError:
                return {
                    "ai_enhanced_guidance": ai_response
                }
        except Exception as e:
            return {}

