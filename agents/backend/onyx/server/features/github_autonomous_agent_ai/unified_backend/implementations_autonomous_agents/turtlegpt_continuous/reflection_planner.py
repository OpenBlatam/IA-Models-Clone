"""
Reflection and Planning Module

Implements reflection and planning based on Generative Agents paper.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..common.memory import EpisodicMemory, SemanticMemory

logger = logging.getLogger(__name__)


class ReflectionPlanner:
    """
    Handles reflection and planning based on Generative Agents paper.
    
    Paper concepts:
    - Reflection: Periodic reflection on experiences to form insights
    - Planning: Generate plans based on memory and current situation
    """
    
    def __init__(
        self,
        llm_client: Any,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        reflection_threshold: int = 5,
        planning_horizon: int = 3
    ):
        """
        Initialize reflection planner.
        
        Args:
            llm_client: LLM client for reflection and planning
            episodic_memory: Episodic memory instance
            semantic_memory: Semantic memory instance
            reflection_threshold: Threshold for triggering reflection
            planning_horizon: Number of actions in plan
        """
        self.llm_client = llm_client
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.reflection_threshold = reflection_threshold
        self.planning_horizon = planning_horizon
        
        self.last_reflection: Optional[datetime] = None
        self.insights: List[str] = []
        self.current_plan: List[Dict[str, Any]] = []
    
    def should_reflect(self) -> bool:
        """
        Determine if should reflect (Generative Agents paper).
        
        The paper describes that agents reflect periodically when
        there are enough new episodic memories.
        """
        recent_experiences = self.episodic_memory.get_recent(n=self.reflection_threshold)
        return len(recent_experiences) >= self.reflection_threshold
    
    async def reflect_on_experiences(self) -> None:
        """
        Reflect on recent experiences (Generative Agents paper).
        
        The paper describes that reflection allows agents to:
        - Understand patterns in their experiences
        - Form higher-level knowledge
        - Update semantic memory with insights
        """
        logger.info("Reflecting on recent experiences...")
        
        # Get recent experiences
        recent_experiences = self.episodic_memory.get_recent(n=10)
        
        if not recent_experiences:
            return
        
        # Build reflection prompt
        experiences_text = "\n".join([
            f"- {exp.content}" for exp in recent_experiences[:5]
        ])
        
        reflection_prompt = f"""Eres un agente autónomo que reflexiona sobre sus experiencias recientes.

Experiencias recientes:
{experiences_text}

Genera insights o patrones que observes en estas experiencias. ¿Qué puedes aprender? ¿Qué patrones identificas?"""
        
        try:
            response = await self.llm_client.generate_text(
                prompt=reflection_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            insight = response.get("generated_text", "")
            if insight:
                self.insights.append(insight)
                # Save insight to semantic memory
                insight_key = f"insight_{datetime.now().timestamp()}"
                self.semantic_memory.add_fact(
                    key=insight_key,
                    value=f"Insight: {insight}",
                    relationships=["reflection", "insight"]
                )
                logger.info(f"Generated insight: {insight[:100]}...")
            
            self.last_reflection = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during reflection: {e}", exc_info=True)
    
    async def generate_plan_from_memory(self) -> None:
        """
        Generate plan based on memory and context (Generative Agents paper).
        
        Creates an action plan based on:
        - Recent episodic memory
        - Reflection insights
        - Pending tasks
        """
        logger.debug("Generating plan from memory...")
        
        # Get relevant context
        recent_memories = self.episodic_memory.get_recent(count=5)
        recent_insights = self.insights[-3:] if self.insights else []
        
        # Build planning prompt
        context_parts = []
        if recent_memories:
            context_parts.append("Memorias recientes:")
            context_parts.extend([f"- {m.get('content', '')}" for m in recent_memories[:3]])
        
        if recent_insights:
            context_parts.append("\nInsights recientes:")
            context_parts.extend([f"- {insight}" for insight in recent_insights])
        
        context_text = "\n".join(context_parts) if context_parts else "No hay contexto reciente."
        
        planning_prompt = f"""Eres un agente autónomo que funciona continuamente. Genera un plan de {self.planning_horizon} acciones próximas basado en tu contexto.

Contexto:
{context_text}

Genera un plan de acciones que el agente debería realizar. Sé específico y práctico."""
        
        try:
            response = await self.llm_client.generate_text(
                prompt=planning_prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            plan_text = response.get("generated_text", "")
            
            # Parse plan (simplified - in production use more robust parsing)
            plan_items = []
            for line in plan_text.split("\n"):
                line = line.strip()
                if line and (line.startswith("-") or line[0].isdigit()):
                    action_text = line.lstrip("- ").lstrip("0123456789. ")
                    if action_text:
                        plan_items.append({
                            "action": action_text,
                            "description": action_text,
                            "status": "pending"
                        })
            
            if plan_items:
                self.current_plan = plan_items[:self.planning_horizon]
                logger.info(f"Generated plan with {len(self.current_plan)} actions")
            
        except Exception as e:
            logger.error(f"Error during planning: {e}", exc_info=True)
    
    def cleanup_insights(self, max_keep: int = 20) -> None:
        """
        Cleanup old insights.
        
        Args:
            max_keep: Maximum insights to keep
        """
        if len(self.insights) > max_keep:
            self.insights = self.insights[-max_keep:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get reflection and planning status."""
        return {
            "insights_count": len(self.insights),
            "last_reflection": self.last_reflection.isoformat() if self.last_reflection else None,
            "current_plan_size": len(self.current_plan),
            "planning_horizon": self.planning_horizon
        }



