"""
Evaluation Metrics - Métricas para evaluar respuestas de LLMs.

Sigue principios de evaluación de modelos de deep learning.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Resultado de una métrica."""
    name: str
    value: float
    description: str = ""
    threshold: Optional[float] = None
    passed: bool = True
    
    def __post_init__(self):
        if self.threshold is not None:
            self.passed = self.value >= self.threshold


class EvaluationMetrics:
    """
    Calculadora de métricas para evaluar respuestas de LLMs.
    
    Sigue principios de evaluación de modelos.
    """
    
    def __init__(self):
        """Inicializar calculadora de métricas."""
        pass
    
    def calculate_all_metrics(
        self,
        response: str,
        reference: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, MetricResult]:
        """
        Calcular todas las métricas disponibles.
        
        Args:
            response: Respuesta a evaluar
            reference: Respuesta de referencia (opcional)
            expected_keywords: Palabras clave esperadas (opcional)
            
        Returns:
            Diccionario con todas las métricas
        """
        metrics = {}
        
        # Métricas básicas
        metrics["length"] = self.length_metric(response)
        metrics["readability"] = self.readability_metric(response)
        metrics["structure"] = self.structure_metric(response)
        
        # Métricas de contenido
        if expected_keywords:
            metrics["keyword_coverage"] = self.keyword_coverage_metric(
                response, expected_keywords
            )
        
        # Métricas de comparación
        if reference:
            metrics["similarity"] = self.similarity_metric(response, reference)
            metrics["rouge_l"] = self.rouge_l_metric(response, reference)
        
        # Métricas específicas de código
        if self._has_code(response):
            metrics["code_quality"] = self.code_quality_metric(response)
            metrics["code_completeness"] = self.code_completeness_metric(response)
        
        return metrics
    
    def length_metric(self, response: str) -> MetricResult:
        """Métrica de longitud."""
        length = len(response)
        word_count = len(response.split())
        
        # Score basado en longitud razonable (100-5000 caracteres)
        if length < 50:
            score = length / 50.0
        elif length > 5000:
            score = max(0.0, 1.0 - (length - 5000) / 10000.0)
        else:
            score = 1.0
        
        return MetricResult(
            name="length",
            value=score,
            description=f"Longitud: {length} caracteres, {word_count} palabras",
            threshold=0.3
        )
    
    def readability_metric(self, response: str) -> MetricResult:
        """Métrica de legibilidad."""
        sentences = re.split(r'[.!?]+', response)
        words = response.split()
        
        if not sentences or not words:
            return MetricResult(
                name="readability",
                value=0.0,
                description="No se puede calcular legibilidad"
            )
        
        # Longitud promedio de oraciones
        avg_sentence_length = len(words) / len(sentences)
        
        # Longitud promedio de palabras
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Score: penalizar oraciones muy largas o palabras muy largas
        score = 1.0
        if avg_sentence_length > 25:
            score -= 0.3
        if avg_word_length > 6:
            score -= 0.2
        
        return MetricResult(
            name="readability",
            value=max(0.0, score),
            description=f"Oraciones: {len(sentences)}, Promedio: {avg_sentence_length:.1f} palabras/oración"
        )
    
    def structure_metric(self, response: str) -> MetricResult:
        """Métrica de estructura."""
        has_headers = bool(re.search(r'^#+\s', response, re.MULTILINE))
        has_lists = bool(re.search(r'^[\*\-\+]\s', response, re.MULTILINE))
        has_code_blocks = "```" in response
        has_paragraphs = len(re.split(r'\n\s*\n', response)) > 1
        
        structure_elements = sum([
            has_headers,
            has_lists,
            has_code_blocks,
            has_paragraphs
        ])
        
        score = structure_elements / 4.0
        
        return MetricResult(
            name="structure",
            value=score,
            description=f"Elementos estructurales: {structure_elements}/4"
        )
    
    def keyword_coverage_metric(
        self,
        response: str,
        keywords: List[str]
    ) -> MetricResult:
        """Métrica de cobertura de palabras clave."""
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        coverage = found / len(keywords) if keywords else 0.0
        
        return MetricResult(
            name="keyword_coverage",
            value=coverage,
            description=f"Palabras clave encontradas: {found}/{len(keywords)}",
            threshold=0.5
        )
    
    def similarity_metric(
        self,
        response: str,
        reference: str
    ) -> MetricResult:
        """Métrica de similitud simple (basada en palabras comunes)."""
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        if not response_words or not reference_words:
            return MetricResult(
                name="similarity",
                value=0.0,
                description="No se puede calcular similitud"
            )
        
        intersection = response_words & reference_words
        union = response_words | reference_words
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return MetricResult(
            name="similarity",
            value=jaccard,
            description=f"Similitud Jaccard: {jaccard:.2f}"
        )
    
    def rouge_l_metric(
        self,
        response: str,
        reference: str
    ) -> MetricResult:
        """
        Métrica ROUGE-L simplificada (Longest Common Subsequence).
        """
        def lcs_length(s1, s2):
            """Calcular longitud de LCS."""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        response_words = response.split()
        reference_words = reference.split()
        
        if not response_words or not reference_words:
            return MetricResult(
                name="rouge_l",
                value=0.0,
                description="No se puede calcular ROUGE-L"
            )
        
        lcs = lcs_length(response_words, reference_words)
        precision = lcs / len(response_words) if response_words else 0.0
        recall = lcs / len(reference_words) if reference_words else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return MetricResult(
            name="rouge_l",
            value=f1,
            description=f"ROUGE-L F1: {f1:.2f} (P: {precision:.2f}, R: {recall:.2f})"
        )
    
    def code_quality_metric(self, response: str) -> MetricResult:
        """Métrica de calidad de código."""
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', response, re.DOTALL)
        
        if not code_blocks:
            return MetricResult(
                name="code_quality",
                value=0.0,
                description="No se encontraron bloques de código"
            )
        
        # Indicadores de calidad
        has_comments = any('#' in code or '//' in code for code in code_blocks)
        has_docstrings = any('"""' in code or "'''" in code for code in code_blocks)
        has_error_handling = any(
            'try' in code or 'except' in code or 'catch' in code
            for code in code_blocks
        )
        
        quality_score = sum([has_comments, has_docstrings, has_error_handling]) / 3.0
        
        return MetricResult(
            name="code_quality",
            value=quality_score,
            description=f"Calidad: {quality_score:.2f} (comentarios, docstrings, error handling)"
        )
    
    def code_completeness_metric(self, response: str) -> MetricResult:
        """Métrica de completitud de código."""
        code_blocks = re.findall(r'```[\w]*\n(.*?)```', response, re.DOTALL)
        
        if not code_blocks:
            return MetricResult(
                name="code_completeness",
                value=0.0,
                description="No se encontraron bloques de código"
            )
        
        incomplete_patterns = [
            r'def\s+\w+\s*\([^)]*$',  # Función sin cerrar
            r'class\s+\w+[^:]*$',  # Clase sin dos puntos
            r'\{[^}]*$',  # Llave sin cerrar
            r'\[[^\]]*$',  # Corchete sin cerrar
            r'TODO',
            r'FIXME',
            r'\.\.\.'  # Placeholder
        ]
        
        incomplete_count = sum(
            1 for code in code_blocks
            for pattern in incomplete_patterns
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE)
        )
        
        completeness = 1.0 - (incomplete_count / len(code_blocks))
        
        return MetricResult(
            name="code_completeness",
            value=max(0.0, completeness),
            description=f"Completitud: {completeness:.2f} (incompleto: {incomplete_count})"
        )
    
    def _has_code(self, text: str) -> bool:
        """Verificar si el texto contiene código."""
        return "```" in text or bool(re.search(r'\b(def|class|function|const|let|var)\b', text))
    
    def get_overall_score(
        self,
        metrics: Dict[str, MetricResult]
    ) -> float:
        """
        Calcular score general promediando métricas.
        
        Args:
            metrics: Diccionario de métricas
            
        Returns:
            Score general (0.0 - 1.0)
        """
        if not metrics:
            return 0.0
        
        values = [m.value for m in metrics.values()]
        return sum(values) / len(values)


# Instancia global
_evaluation_metrics = EvaluationMetrics()


def get_evaluation_metrics() -> EvaluationMetrics:
    """Obtener instancia global de evaluación."""
    return _evaluation_metrics



