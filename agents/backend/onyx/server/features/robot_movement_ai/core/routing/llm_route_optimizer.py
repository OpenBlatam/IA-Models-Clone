"""
LLM-based Route Optimizer
=========================

Módulo de optimización de rutas usando LLMs para generar explicaciones,
recomendaciones y análisis avanzados.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers no disponible, funcionalidad LLM limitada")


@dataclass
class RouteExplanation:
    """Explicación de ruta generada por LLM."""
    route_id: str
    explanation: str
    reasoning: str
    alternatives: List[str]
    recommendations: List[str]
    confidence: float


class LLMRouteOptimizer:
    """
    Optimizador de rutas basado en LLM.
    
    Usa modelos de lenguaje para generar explicaciones, recomendaciones
    y análisis avanzados de rutas.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Inicializar optimizador LLM.
        
        Args:
            model_name: Nombre del modelo (local o API)
            device: Dispositivo ('cuda', 'cpu', o None)
            use_api: Usar API externa (OpenAI, etc.)
            api_key: Clave de API si se usa API externa
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_key = api_key
        
        # Detectar dispositivo
        if device is None and TRANSFORMERS_AVAILABLE:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device) if device else None
        
        # Inicializar pipeline local si no se usa API
        self.pipeline = None
        if not use_api and TRANSFORMERS_AVAILABLE:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if self.device and self.device.type == "cuda" else -1,
                    max_length=300,
                    do_sample=True,
                    temperature=0.7
                )
                logger.info(f"LLM pipeline inicializado con modelo: {model_name}")
            except Exception as e:
                logger.warning(f"Error inicializando LLM pipeline: {e}")
        
        # Inicializar cliente de API si se usa
        self.api_client = None
        if use_api:
            try:
                if "openai" in model_name.lower() or "gpt" in model_name.lower():
                    import openai
                    if api_key:
                        openai.api_key = api_key
                    self.api_client = "openai"
                elif "anthropic" in model_name.lower() or "claude" in model_name.lower():
                    import anthropic
                    if api_key:
                        self.api_client = anthropic.Anthropic(api_key=api_key)
                    else:
                        self.api_client = "anthropic"
                logger.info(f"API client inicializado: {self.api_client}")
            except ImportError:
                logger.warning("Librerías de API no disponibles")
    
    def generate_route_explanation(
        self,
        route: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RouteExplanation:
        """
        Generar explicación de ruta usando LLM.
        
        Args:
            route: Información de la ruta
            context: Contexto adicional
            
        Returns:
            Explicación generada
        """
        route_id = route.get("route_id", "unknown")
        
        # Construir prompt
        prompt = self._build_explanation_prompt(route, context)
        
        # Generar texto
        generated_text = self._generate_text(prompt, max_length=200)
        
        # Parsear respuesta
        explanation = self._parse_explanation(generated_text)
        
        return RouteExplanation(
            route_id=route_id,
            explanation=explanation.get("explanation", "No explanation available"),
            reasoning=explanation.get("reasoning", ""),
            alternatives=explanation.get("alternatives", []),
            recommendations=explanation.get("recommendations", []),
            confidence=explanation.get("confidence", 0.5)
        )
    
    def optimize_route_with_llm(
        self,
        route: Dict[str, Any],
        constraints: List[str],
        objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Optimizar ruta usando análisis de LLM.
        
        Args:
            route: Ruta actual
            constraints: Restricciones
            objectives: Objetivos de optimización
            
        Returns:
            Recomendaciones de optimización
        """
        prompt = f"""
        Analyze the following route and provide optimization recommendations:
        
        Route Details:
        - Distance: {route.get('total_distance', 0)}
        - Time: {route.get('total_time', 0)}
        - Cost: {route.get('total_cost', 0)}
        - Path: {route.get('path', [])}
        
        Constraints: {', '.join(constraints)}
        Objectives: {', '.join(objectives)}
        
        Provide specific recommendations to optimize this route:
        1. Identify bottlenecks
        2. Suggest alternative paths
        3. Recommend parameter adjustments
        4. Estimate potential improvements
        
        Recommendations:
        """
        
        generated_text = self._generate_text(prompt, max_length=300)
        
        return {
            "original_route": route,
            "recommendations": self._parse_recommendations(generated_text),
            "optimization_score": self._calculate_optimization_score(route, generated_text)
        }
    
    def compare_routes(
        self,
        routes: List[Dict[str, Any]],
        criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Comparar múltiples rutas usando LLM.
        
        Args:
            routes: Lista de rutas a comparar
            criteria: Criterios de comparación
            
        Returns:
            Comparación detallada
        """
        prompt = f"""
        Compare the following routes based on the criteria: {', '.join(criteria)}
        
        Routes:
        """
        
        for i, route in enumerate(routes):
            prompt += f"""
        Route {i + 1}:
        - Distance: {route.get('total_distance', 0)}
        - Time: {route.get('total_time', 0)}
        - Cost: {route.get('total_cost', 0)}
        - Strategy: {route.get('strategy', 'unknown')}
        """
        
        prompt += """
        
        Provide a detailed comparison:
        1. Best route for each criterion
        2. Overall best route
        3. Trade-offs between routes
        4. Recommendations
        
        Comparison:
        """
        
        generated_text = self._generate_text(prompt, max_length=400)
        
        return {
            "routes": routes,
            "comparison": generated_text,
            "best_route": self._extract_best_route(generated_text, routes),
            "scores": self._calculate_route_scores(routes, criteria)
        }
    
    def _build_explanation_prompt(
        self,
        route: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Construir prompt para explicación."""
        prompt = f"""
        Explain why this route was selected:
        
        Route Information:
        - Distance: {route.get('total_distance', 0)} units
        - Time: {route.get('total_time', 0)} units
        - Cost: {route.get('total_cost', 0)} units
        - Strategy: {route.get('strategy', 'unknown')}
        - Path: {len(route.get('path', []))} nodes
        """
        
        if context:
            prompt += f"""
        Context:
        - Constraints: {context.get('constraints', [])}
        - Preferences: {context.get('preferences', [])}
        - Historical Performance: {context.get('historical', {})}
        """
        
        prompt += """
        
        Provide:
        1. Clear explanation of route selection
        2. Reasoning behind the choice
        3. Alternative routes considered
        4. Recommendations for improvement
        
        Explanation:
        """
        
        return prompt
    
    def _generate_text(
        self,
        prompt: str,
        max_length: int = 200
    ) -> str:
        """Generar texto usando LLM."""
        if self.use_api and self.api_client:
            return self._generate_with_api(prompt, max_length)
        elif self.pipeline:
            try:
                result = self.pipeline(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                return result[0]["generated_text"]
            except Exception as e:
                logger.error(f"Error generando texto: {e}")
                return "Error generating text"
        else:
            # Fallback: respuesta básica
            return f"Analysis for: {prompt[:100]}..."
    
    def _generate_with_api(
        self,
        prompt: str,
        max_length: int
    ) -> str:
        """Generar texto usando API externa."""
        try:
            if self.api_client == "openai":
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length
                )
                return response.choices[0].message.content
            elif isinstance(self.api_client, type) and "anthropic" in str(type(self.api_client)):
                response = self.api_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_length,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"Error con API: {e}")
        
        return "API generation failed"
    
    def _parse_explanation(self, text: str) -> Dict[str, Any]:
        """Parsear explicación generada."""
        # Análisis básico del texto
        lines = text.split("\n")
        
        explanation = ""
        reasoning = ""
        alternatives = []
        recommendations = []
        
        current_section = None
        for line in lines:
            line_lower = line.lower()
            if "explanation" in line_lower or "explain" in line_lower:
                current_section = "explanation"
            elif "reasoning" in line_lower or "reason" in line_lower:
                current_section = "reasoning"
            elif "alternative" in line_lower:
                current_section = "alternatives"
            elif "recommendation" in line_lower or "suggest" in line_lower:
                current_section = "recommendations"
            
            if current_section == "explanation":
                explanation += line + " "
            elif current_section == "reasoning":
                reasoning += line + " "
            elif current_section == "alternatives" and line.strip().startswith("-"):
                alternatives.append(line.strip())
            elif current_section == "recommendations" and line.strip().startswith("-"):
                recommendations.append(line.strip())
        
        return {
            "explanation": explanation.strip() or text[:200],
            "reasoning": reasoning.strip(),
            "alternatives": alternatives[:5],
            "recommendations": recommendations[:5],
            "confidence": 0.7  # Valor por defecto
        }
    
    def _parse_recommendations(self, text: str) -> List[str]:
        """Parsear recomendaciones."""
        recommendations = []
        for line in text.split("\n"):
            if line.strip().startswith(("-", "1.", "2.", "3.", "4.", "5.")):
                recommendations.append(line.strip())
        return recommendations[:10]
    
    def _calculate_optimization_score(
        self,
        route: Dict[str, Any],
        recommendations: str
    ) -> float:
        """Calcular score de optimización."""
        # Score basado en keywords positivos
        positive_keywords = ["improve", "optimize", "reduce", "better", "faster", "cheaper"]
        negative_keywords = ["cannot", "impossible", "limited", "constraint"]
        
        text_lower = recommendations.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        score = min(1.0, (positive_count - negative_count * 0.5) / 5.0)
        return max(0.0, score)
    
    def _extract_best_route(
        self,
        comparison_text: str,
        routes: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extraer mejor ruta del texto de comparación."""
        # Buscar menciones de "best", "optimal", "recommended"
        text_lower = comparison_text.lower()
        for i, route in enumerate(routes):
            route_mention = f"route {i + 1}"
            if route_mention in text_lower and ("best" in text_lower or "optimal" in text_lower):
                return route
        return routes[0] if routes else None
    
    def _calculate_route_scores(
        self,
        routes: List[Dict[str, Any]],
        criteria: List[str]
    ) -> List[float]:
        """Calcular scores para cada ruta."""
        scores = []
        for route in routes:
            # Score basado en métricas normalizadas
            distance_score = 1.0 / (1.0 + route.get('total_distance', 1.0))
            time_score = 1.0 / (1.0 + route.get('total_time', 1.0))
            cost_score = 1.0 / (1.0 + route.get('total_cost', 1.0))
            
            # Score combinado
            combined_score = (distance_score + time_score + cost_score) / 3.0
            scores.append(combined_score)
        
        return scores




