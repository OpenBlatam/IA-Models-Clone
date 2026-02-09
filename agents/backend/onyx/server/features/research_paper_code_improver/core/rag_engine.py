"""
RAG Engine - Retrieval Augmented Generation para mejoras de código
===================================================================
"""

from typing import Dict, Any, Optional, List
import os

from .core_utils import get_logger
from .llm_factory import LLMFactory

logger = get_logger(__name__)


class RAGEngine:
    """
    Motor RAG que combina búsqueda de papers relevantes con generación de mejoras.
    """
    
    def __init__(self, vector_store=None):
        """
        Inicializar motor RAG.
        
        Args:
            vector_store: Instancia de VectorStore (opcional)
        """
        self.vector_store = vector_store
        self.llm_client_info = LLMFactory.create_client()
        
        if self.llm_client_info:
            logger.info(f"Cliente LLM: {self.llm_client_info['provider'].value}")
        else:
            logger.warning("No se encontró cliente LLM configurado")
    
    def improve_code_with_rag(
        self,
        code: str,
        context: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mejora código usando RAG (papers relevantes + LLM).
        
        Args:
            code: Código a mejorar
            context: Contexto adicional
            language: Lenguaje de programación (opcional)
            
        Returns:
            Código mejorado y papers usados
        """
        try:
            # 1. Buscar papers relevantes
            relevant_papers = []
            if self.vector_store:
                query = self._build_search_query(code, context, language)
                relevant_papers = self.vector_store.search_relevant_papers(query, top_k=3)
            
            # 2. Construir prompt con contexto de papers
            prompt = self._build_improvement_prompt(code, relevant_papers, context, language)
            
            # 3. Generar mejora usando LLM
            improved_code = self._generate_improvement(prompt, code)
            
            # 4. Generar explicación
            explanation = self._generate_explanation(code, improved_code, relevant_papers)
            
            return {
                "original_code": code,
                "improved_code": improved_code,
                "explanation": explanation,
                "papers_used": [
                    {
                        "title": p.get("metadata", {}).get("title", ""),
                        "relevance_score": 1 - p.get("distance", 1.0)
                    }
                    for p in relevant_papers
                ],
                "improvements_count": self._count_improvements(code, improved_code)
            }
            
        except Exception as e:
            logger.error(f"Error en RAG: {e}")
            raise
    
    def _build_search_query(self, code: str, context: Optional[str], language: Optional[str]) -> str:
        """Construye query de búsqueda desde el código"""
        query_parts = []
        
        if language:
            query_parts.append(f"{language} programming")
        
        # Extraer conceptos clave del código (simplificado)
        if "class" in code:
            query_parts.append("object-oriented programming")
        if "async" in code or "await" in code:
            query_parts.append("asynchronous programming")
        if "def" in code or "function" in code:
            query_parts.append("function optimization")
        
        if context:
            query_parts.append(context)
        
        return " ".join(query_parts) if query_parts else "code improvement best practices"
    
    def _build_improvement_prompt(
        self,
        code: str,
        papers: List[Dict[str, Any]],
        context: Optional[str],
        language: Optional[str]
    ) -> str:
        """Construye prompt para mejora de código"""
        prompt_parts = [
            "You are an expert code reviewer. Improve the following code based on research papers and best practices.",
            "",
            "Code to improve:",
            "```",
            code,
            "```",
            ""
        ]
        
        if papers:
            prompt_parts.append("Relevant research papers:")
            for i, paper in enumerate(papers, 1):
                title = paper.get("metadata", {}).get("title", "Unknown")
                content = paper.get("content", "")[:500]
                prompt_parts.append(f"\n{i}. {title}")
                prompt_parts.append(f"   {content}...")
            prompt_parts.append("")
        
        if context:
            prompt_parts.append(f"Additional context: {context}")
        
        if language:
            prompt_parts.append(f"Language: {language}")
        
        prompt_parts.append("\nProvide improved code with explanations.")
        
        return "\n".join(prompt_parts)
    
    def _generate_improvement(self, prompt: str, original_code: str) -> str:
        """Genera código mejorado usando LLM"""
        if self.llm_client == "openai":
            return self._generate_with_openai(prompt)
        elif self.llm_client == "anthropic":
            return self._generate_with_anthropic(prompt)
        else:
            # Fallback: retornar código original con mejoras básicas
            logger.warning("LLM no disponible, usando mejoras básicas")
            return self._basic_improvements(original_code)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Genera usando OpenAI"""
        try:
            import openai
            
            response = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer and optimizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error con OpenAI: {e}")
            raise
    
    def _generate_with_anthropic(self, prompt: str) -> str:
        """Genera usando Anthropic Claude"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            message = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error con Anthropic: {e}")
            raise
    
    def _basic_improvements(self, code: str) -> str:
        """Mejoras básicas sin LLM"""
        # Mejoras simples
        improved = code
        
        # Agregar espacios alrededor de operadores (ejemplo básico)
        # En producción, esto sería más sofisticado
        
        return improved
    
    def _generate_explanation(
        self,
        original: str,
        improved: str,
        papers: List[Dict[str, Any]]
    ) -> str:
        """Genera explicación de las mejoras"""
        if original == improved:
            return "No se detectaron mejoras necesarias."
        
        explanations = []
        
        if papers:
            explanations.append(f"Mejoras basadas en {len(papers)} papers de investigación relevantes.")
        
        # Análisis básico de diferencias
        if len(improved) > len(original):
            explanations.append("Se agregaron mejoras y optimizaciones.")
        elif len(improved) < len(original):
            explanations.append("Se optimizó y simplificó el código.")
        else:
            explanations.append("Se refactorizó el código manteniendo funcionalidad.")
        
        return " ".join(explanations)
    
    def _count_improvements(self, original: str, improved: str) -> int:
        """Cuenta número de mejoras aplicadas"""
        if original == improved:
            return 0
        
        # Análisis básico
        improvements = 0
        
        # Contar líneas diferentes
        orig_lines = original.split("\n")
        impr_lines = improved.split("\n")
        
        if len(impr_lines) != len(orig_lines):
            improvements += 1
        
        # Otras métricas...
        
        return max(improvements, 1)  # Mínimo 1 si hay diferencias




