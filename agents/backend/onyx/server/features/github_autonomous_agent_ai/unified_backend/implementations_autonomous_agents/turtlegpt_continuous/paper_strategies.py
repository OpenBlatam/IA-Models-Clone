"""
Paper Strategies Module
========================

Módulo que contiene las implementaciones de estrategias de diferentes papers.
Cada estrategia puede ser usada independientemente según la prioridad de la tarea.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ReactStrategy:
    """Estrategia ReAct: Thought → Action → Observation (paper: ReAct)."""
    
    def __init__(self, llm_client, tool_registry, agent):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.agent = agent
        self.reasoning_history: List[Dict[str, Any]] = []
    
    async def execute(self, task: str) -> Dict[str, Any]:
        """Ejecutar ciclo ReAct."""
        max_iterations = 10
        observation = f"Task: {task}"
        
        for iteration in range(max_iterations):
            # Thought: Razonar sobre qué hacer usando PromptBuilder
            thought_prompt = PromptBuilder.build_react_thought_prompt(
                task=task,
                observation=observation,
                available_tools=self.tool_registry.list_tools()
            )
            
            try:
                thought_response = await self.llm_client.generate_text(
                    prompt=thought_prompt,
                    max_tokens=500,
                    temperature=0.7
                )
                thought = thought_response.get("generated_text", "")
                self.reasoning_history.append({
                    "iteration": iteration,
                    "thought": thought,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # Action: Determinar y ejecutar acción usando PromptBuilder
                action_prompt = PromptBuilder.build_react_action_prompt(
                    thought=thought,
                    available_tools=self.tool_registry.list_tools()
                )
                
                action_response = await self.llm_client.generate_text(
                    prompt=action_prompt,
                    max_tokens=200,
                    temperature=0.3
                )
                action_text = action_response.get("generated_text", "")
                
                # Parse action (simplificado)
                if "Action:" in action_text:
                    action_text = action_text.split("Action:")[-1].strip()
                
                # Execute action
                action_result = self.agent.act({
                    "type": "tool_call",
                    "tool": "search",  # Simplified
                    "args": {"query": task}
                })
                
                # Observation: Procesar resultado
                observation = f"Action result: {action_result}"
                self.agent.observe(observation)
                
                # Check if task is complete
                if "complete" in str(action_result).lower() or iteration >= max_iterations - 1:
                    break
                    
            except Exception as e:
                logger.error(f"Error in ReAct cycle: {e}")
                break
        
        return {
            "task": task,
            "iterations": iteration + 1,
            "final_observation": observation,
            "reasoning_steps": len(self.reasoning_history)
        }


class LATSStrategy:
    """Estrategia LATS: Tree search unificando reasoning, acting y planning (paper: LATS)."""
    
    def __init__(self, llm_client, agent):
        self.llm_client = llm_client
        self.agent = agent
        self.search_tree: Optional[Dict[str, Any]] = None
    
    async def execute(self, goal: str) -> Dict[str, Any]:
        """Ejecutar búsqueda LATS."""
        # Inicializar árbol de búsqueda
        tree = {
            "root": {
                "state": goal,
                "value": 0.0,
                "depth": 0,
                "children": []
            }
        }
        self.search_tree = tree
        
        max_depth = 5
        best_path = []
        
        # Búsqueda en árbol
        current_node = tree["root"]
        for depth in range(max_depth):
            # Generar candidatos de acción usando PromptBuilder
            candidates_prompt = PromptBuilder.build_lats_candidates_prompt(
                goal=goal,
                current_state=current_node['state']
            )
            
            try:
                candidates_response = await self.llm_client.generate_text(
                    prompt=candidates_prompt,
                    max_tokens=300,
                    temperature=0.7
                )
                candidates_text = candidates_response.get("generated_text", "")
                
                # Evaluar cada candidato
                best_candidate = None
                best_value = -1.0
                
                for candidate in candidates_text.split("\n")[:3]:
                    if "Action:" in candidate:
                        # Evaluar candidato usando PromptBuilder
                        eval_prompt = PromptBuilder.build_lats_evaluation_prompt(
                            goal=goal,
                            current_state=current_node['state'],
                            candidate_action=candidate
                        )
                        
                        eval_response = await self.llm_client.generate_text(
                            prompt=eval_prompt,
                            max_tokens=10,
                            temperature=0.1
                        )
                        
                        try:
                            value = float(eval_response.get("generated_text", "0.5").strip())
                            if value > best_value:
                                best_value = value
                                best_candidate = candidate
                        except ValueError:
                            continue
                
                if best_candidate:
                    # Ejecutar mejor acción
                    action_result = self.agent.act({
                        "type": "process",
                        "description": best_candidate
                    })
                    
                    # Nuevo estado
                    new_state = f"{current_node['state']} → {action_result.get('status', 'executed')}"
                    
                    # Agregar al árbol
                    new_node = {
                        "state": new_state,
                        "value": best_value,
                        "depth": depth + 1,
                        "action": best_candidate,
                        "children": []
                    }
                    current_node["children"].append(new_node)
                    best_path.append(best_candidate)
                    current_node = new_node
                    
                    # Verificar si se alcanzó el objetivo
                    if best_value > 0.8:
                        break
                        
            except Exception as e:
                logger.error(f"Error in LATS search: {e}")
                break
        
        return {
            "goal": goal,
            "path": best_path,
            "final_state": current_node["state"],
            "depth": current_node["depth"]
        }


class TreeOfThoughtsStrategy:
    """Estrategia Tree of Thoughts: Búsqueda deliberada en árbol (paper: Tree of Thoughts)."""
    
    def __init__(self, llm_client, agent, strategy: str = "bfs"):
        self.llm_client = llm_client
        self.agent = agent
        self.strategy = strategy
    
    async def execute(self, problem: str) -> Dict[str, Any]:
        """Resolver problema usando Tree of Thoughts."""
        max_depth = 5
        beam_width = 3
        
        # Inicializar árbol
        root_state = f"Problem: {problem}"
        tree_nodes = [{"state": root_state, "value": 0.5, "depth": 0, "path": []}]
        
        best_solution = None
        best_value = 0.0
        
        for depth in range(max_depth):
            next_level = []
            
            for node in tree_nodes[:beam_width]:  # Beam search
                # Generar pensamientos candidatos
                thoughts_prompt = f"""Problem: {problem}
Current state: {node['state']}

Generate 3 intermediate thoughts/steps to progress toward solving this problem.
Format each as: Thought: [description]"""
                
                try:
                    thoughts_response = await self.llm_client.generate_text(
                        prompt=thoughts_prompt,
                        max_tokens=400,
                        temperature=0.7
                    )
                    thoughts_text = thoughts_response.get("generated_text", "")
                    
                    # Parse thoughts
                    thoughts = []
                    for line in thoughts_text.split("\n"):
                        if "Thought:" in line:
                            thought = line.split("Thought:")[-1].strip()
                            if thought:
                                thoughts.append(thought)
                    
                    # Evaluar cada pensamiento
                    for thought in thoughts[:3]:
                        # Evaluar valor del pensamiento
                        eval_prompt = f"""Problem: {problem}
Current state: {node['state']}
New thought: {thought}

Rate how promising this thought is for solving the problem (0.0 to 1.0).
Respond with only a number."""
                        
                        eval_response = await self.llm_client.generate_text(
                            prompt=eval_prompt,
                            max_tokens=10,
                            temperature=0.1
                        )
                        
                        try:
                            value = float(eval_response.get("generated_text", "0.5").strip())
                            new_state = f"{node['state']} → {thought}"
                            new_path = node['path'] + [thought]
                            
                            next_level.append({
                                "state": new_state,
                                "value": value,
                                "depth": depth + 1,
                                "path": new_path
                            })
                            
                            if value > best_value:
                                best_value = value
                                best_solution = {
                                    "problem": problem,
                                    "solution_path": new_path,
                                    "value": value,
                                    "depth": depth + 1
                                }
                        except ValueError:
                            continue
                            
                except Exception as e:
                    logger.error(f"Error in ToT solve: {e}")
                    break
            
            if not next_level:
                break
            
            # Ordenar por valor y mantener los mejores
            next_level.sort(key=lambda x: x["value"], reverse=True)
            tree_nodes = next_level[:beam_width]
            
            # Si encontramos una solución muy buena, detener
            if best_value > 0.85:
                break
        
        return best_solution or {
            "problem": problem,
            "solution_path": [],
            "value": 0.0,
            "status": "no_solution_found"
        }


class TheoryOfMindStrategy:
    """Estrategia Theory of Mind: Modelado de otros agentes (paper: Theory of Mind)."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.agent_models: Dict[str, Dict[str, Any]] = {}
    
    async def model_agent(self, agent_id: str, observed_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Modelar otro agente usando Theory of Mind."""
        # Crear o actualizar modelo del agente
        if agent_id not in self.agent_models:
            self.agent_models[agent_id] = {
                "beliefs": {},
                "desires": [],
                "intentions": [],
                "action_history": [],
                "behavioral_patterns": {}
            }
        
        model = self.agent_models[agent_id]
        model["action_history"].extend(observed_actions)
        
        # Analizar patrones de comportamiento
        if len(model["action_history"]) >= 3:
            # Inferir intenciones
            intentions_prompt = f"""Agent {agent_id} has performed these actions:
{chr(10).join([str(a) for a in observed_actions[-5:]])}

What are this agent's likely intentions? What goals is it pursuing?
Format: Intentions: [list of intentions]"""
            
            try:
                intentions_response = await self.llm_client.generate_text(
                    prompt=intentions_prompt,
                    max_tokens=200,
                    temperature=0.5
                )
                intentions_text = intentions_response.get("generated_text", "")
                
                # Parse intentions
                if "Intentions:" in intentions_text:
                    intentions = [
                        i.strip() for i in intentions_text.split("Intentions:")[-1].split(",")
                    ]
                    model["intentions"] = intentions[:5]  # Keep top 5
                
                # Predecir siguiente acción
                prediction_prompt = f"""Agent {agent_id} has these intentions: {', '.join(model['intentions'])}
Recent actions: {str(observed_actions[-3:])}

Predict the next action this agent is likely to take.
Format: Prediction: [action description]"""
                
                prediction_response = await self.llm_client.generate_text(
                    prompt=prediction_prompt,
                    max_tokens=100,
                    temperature=0.3
                )
                prediction = prediction_response.get("generated_text", "")
                
                model["predicted_next_action"] = prediction
                
            except Exception as e:
                logger.error(f"Error in Theory of Mind modeling: {e}")
        
        return {
            "agent_id": agent_id,
            "model": model,
            "intentions": model.get("intentions", []),
            "predicted_action": model.get("predicted_next_action", "")
        }


class PersonalityStrategy:
    """Estrategia Personality-Driven: Decisiones basadas en personalidad (paper: Personality-Driven)."""
    
    def __init__(self, personality_profile: Optional[Dict[str, Any]] = None):
        self.personality_profile = personality_profile
    
    def apply_to_decision(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aplicar personalidad a decisión."""
        if not self.personality_profile or not options:
            return max(options, key=lambda x: x.get("value", 0.0)) if options else {}
        
        # Calcular fit de personalidad para cada opción
        for option in options:
            personality_fit = 0.5  # Default
            
            # Ajustar según rasgos de personalidad
            if self.personality_profile.get("openness", 0.5) > 0.7:
                if option.get("innovative", False):
                    personality_fit += 0.2
            
            if self.personality_profile.get("conscientiousness", 0.5) > 0.7:
                if option.get("structured", False):
                    personality_fit += 0.2
            
            if self.personality_profile.get("extraversion", 0.5) > 0.7:
                if option.get("social", False):
                    personality_fit += 0.15
            
            option["personality_fit"] = min(1.0, personality_fit)
            option["final_score"] = (
                option.get("value", 0.0) * 0.7 +
                option["personality_fit"] * 0.3
            )
        
        # Seleccionar opción con mejor score final
        return max(options, key=lambda x: x.get("final_score", 0.0))


class ToolformerStrategy:
    """Estrategia Toolformer: Auto-aprendizaje de herramientas (paper: Toolformer)."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.learned_tools: Dict[str, Any] = {}
    
    async def learn_tool_usage(self, tool_name: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aprender uso de herramienta."""
        # Analizar ejemplos para aprender patrones
        successful_patterns = []
        failed_patterns = []
        
        for example in examples:
            if example.get("success", False):
                successful_patterns.append({
                    "input": example.get("input"),
                    "output": example.get("output"),
                    "context": example.get("context", {})
                })
            else:
                failed_patterns.append({
                    "input": example.get("input"),
                    "error": example.get("error")
                })
        
        # Generar reglas de uso aprendidas
        if successful_patterns:
            learning_prompt = f"""Tool: {tool_name}

Successful usage examples:
{chr(10).join([f"Input: {p['input']}, Output: {p['output']}" for p in successful_patterns[:3]])}

Learn patterns for when and how to use this tool effectively.
Format: Rules: [learned rules]"""
            
            try:
                learning_response = await self.llm_client.generate_text(
                    prompt=learning_prompt,
                    max_tokens=300,
                    temperature=0.5
                )
                learned_rules = learning_response.get("generated_text", "")
                
                self.learned_tools[tool_name] = {
                    "rules": learned_rules,
                    "success_count": len(successful_patterns),
                    "failure_count": len(failed_patterns),
                    "confidence": len(successful_patterns) / (len(successful_patterns) + len(failed_patterns) + 1)
                }
                
                logger.info(f"Learned tool usage patterns for {tool_name}")
                
                return {
                    "tool_name": tool_name,
                    "learned_rules": learned_rules,
                    "confidence": self.learned_tools[tool_name]["confidence"]
                }
            except Exception as e:
                logger.error(f"Error learning tool usage: {e}")
        
        return {"status": "no_patterns_learned"}
