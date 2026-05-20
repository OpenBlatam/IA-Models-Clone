"""
Prompt Builder Module
=====================

Construye prompts estructurados para diferentes operaciones del agente.
"""

from typing import Dict, Any, Optional, List


class PromptBuilder:
    """
    Constructor de prompts para el agente.
    
    Centraliza la construcción de prompts para:
    - Thinking/Reasoning
    - Reflection
    - Planning
    - Tool usage
    - Evaluation
    """
    
    @staticmethod
    def build_thinking_prompt(
        task: str,
        context: Optional[Dict[str, Any]] = None,
        memory_context: Optional[str] = None
    ) -> str:
        """
        Construir prompt para pensar sobre una tarea.
        
        Args:
            task: Descripción de la tarea
            context: Contexto adicional
            memory_context: Contexto de memoria relevante
            
        Returns:
            Prompt formateado
        """
        prompt = f"""Eres un agente autónomo inteligente. Analiza la siguiente tarea y genera un plan de acción.

Tarea: {task}"""
        
        if memory_context:
            prompt += f"\n\nMemoria relevante:\n{memory_context}"
        
        if context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
            prompt += f"\n\nContexto adicional:\n{context_str}"
        
        prompt += "\n\nGenera un plan de acción paso a paso para completar esta tarea. Sé específico y detallado."
        
        return prompt
    
    @staticmethod
    def build_react_thought_prompt(
        task: str,
        observation: str,
        available_tools: List[str]
    ) -> str:
        """
        Construir prompt para el paso Thought de ReAct.
        
        Args:
            task: Tarea actual
            observation: Observación previa
            available_tools: Herramientas disponibles
            
        Returns:
            Prompt formateado
        """
        return f"""You are an autonomous agent. Current task: {task}

Previous observations:
{observation}

Available tools: {', '.join(available_tools)}

Think about what action to take next. Consider:
- What information do you need?
- What tools are available?
- What is the next step?

Format: Thought: [your reasoning]"""
    
    @staticmethod
    def build_react_action_prompt(thought: str, available_tools: List[str]) -> str:
        """
        Construir prompt para el paso Action de ReAct.
        
        Args:
            thought: Pensamiento generado
            available_tools: Herramientas disponibles
            
        Returns:
            Prompt formateado
        """
        return f"""Based on this thought: {thought}

What action should be taken? Available tools: {', '.join(available_tools)}

Format: Action: [tool_name]([parameters])"""
    
    @staticmethod
    def build_lats_candidates_prompt(goal: str, current_state: str) -> str:
        """
        Construir prompt para generar candidatos en LATS.
        
        Args:
            goal: Objetivo a alcanzar
            current_state: Estado actual
            
        Returns:
            Prompt formateado
        """
        return f"""Goal: {goal}
Current state: {current_state}

Generate 3 candidate actions to progress toward the goal.
Format each as: Action: [description]"""
    
    @staticmethod
    def build_lats_evaluation_prompt(
        goal: str,
        current_state: str,
        candidate_action: str
    ) -> str:
        """
        Construir prompt para evaluar candidatos en LATS.
        
        Args:
            goal: Objetivo
            current_state: Estado actual
            candidate_action: Acción candidata
            
        Returns:
            Prompt formateado
        """
        return f"""Goal: {goal}
Current state: {current_state}
Candidate action: {candidate_action}

Rate how well this action helps achieve the goal (0.0 to 1.0).
Respond with only a number."""
    
    @staticmethod
    def build_tot_thoughts_prompt(problem: str, current_state: str) -> str:
        """
        Construir prompt para generar pensamientos en Tree of Thoughts.
        
        Args:
            problem: Problema a resolver
            current_state: Estado actual
            
        Returns:
            Prompt formateado
        """
        return f"""Problem: {problem}
Current state: {current_state}

Generate 3 intermediate thoughts/steps to progress toward solving this problem.
Format each as: Thought: [description]"""
    
    @staticmethod
    def build_tot_evaluation_prompt(
        problem: str,
        current_state: str,
        thought: str
    ) -> str:
        """
        Construir prompt para evaluar pensamientos en Tree of Thoughts.
        
        Args:
            problem: Problema
            current_state: Estado actual
            thought: Pensamiento a evaluar
            
        Returns:
            Prompt formateado
        """
        return f"""Problem: {problem}
Current state: {current_state}
New thought: {thought}

Rate how promising this thought is for solving the problem (0.0 to 1.0).
Respond with only a number."""
    
    @staticmethod
    def build_theory_of_mind_intentions_prompt(
        agent_id: str,
        observed_actions: List[Dict[str, Any]]
    ) -> str:
        """
        Construir prompt para inferir intenciones (Theory of Mind).
        
        Args:
            agent_id: ID del agente
            observed_actions: Acciones observadas
            
        Returns:
            Prompt formateado
        """
        actions_str = "\n".join([str(a) for a in observed_actions[-5:]])
        return f"""Agent {agent_id} has performed these actions:
{actions_str}

What are this agent's likely intentions? What goals is it pursuing?
Format: Intentions: [list of intentions]"""
    
    @staticmethod
    def build_theory_of_mind_prediction_prompt(
        agent_id: str,
        intentions: List[str],
        recent_actions: List[Dict[str, Any]]
    ) -> str:
        """
        Construir prompt para predecir acciones (Theory of Mind).
        
        Args:
            agent_id: ID del agente
            intentions: Intenciones inferidas
            recent_actions: Acciones recientes
            
        Returns:
            Prompt formateado
        """
        return f"""Agent {agent_id} has these intentions: {', '.join(intentions)}
Recent actions: {str(recent_actions[-3:])}

Predict the next action this agent is likely to take.
Format: Prediction: [action description]"""
    
    @staticmethod
    def build_toolformer_learning_prompt(
        tool_name: str,
        successful_examples: List[Dict[str, Any]]
    ) -> str:
        """
        Construir prompt para aprender uso de herramientas (Toolformer).
        
        Args:
            tool_name: Nombre de la herramienta
            successful_examples: Ejemplos exitosos
            
        Returns:
            Prompt formateado
        """
        examples_str = "\n".join([
            f"Input: {ex['input']}, Output: {ex['output']}"
            for ex in successful_examples[:3]
        ])
        
        return f"""Tool: {tool_name}

Successful usage examples:
{examples_str}

Learn patterns for when and how to use this tool effectively.
Format: Rules: [learned rules]"""
