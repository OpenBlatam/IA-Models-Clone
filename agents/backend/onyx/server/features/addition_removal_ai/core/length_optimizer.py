"""
Length Optimizer - Sistema de análisis y optimización de longitud
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LengthOptimizer:
    """Optimizador de longitud"""

    def __init__(self):
        """Inicializar optimizador"""
        # Longitudes óptimas según tipo de contenido
        self.optimal_lengths = {
            "title": {"min": 30, "max": 60, "optimal": 50},
            "meta_description": {"min": 120, "max": 160, "optimal": 140},
            "paragraph": {"min": 50, "max": 200, "optimal": 100},
            "sentence": {"min": 10, "max": 25, "optimal": 15},
            "article": {"min": 300, "max": 2000, "optimal": 800},
            "blog_post": {"min": 500, "max": 2500, "optimal": 1200},
            "social_media": {"min": 50, "max": 280, "optimal": 150}
        }

    def analyze(
        self,
        content: str,
        content_type: str = "article"
    ) -> Dict[str, Any]:
        """
        Analizar longitud del contenido.

        Args:
            content: Contenido
            content_type: Tipo de contenido

        Returns:
            Análisis de longitud
        """
        char_count = len(content)
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Obtener longitudes óptimas
        optimal = self.optimal_lengths.get(content_type, self.optimal_lengths["article"])
        
        # Evaluar longitud
        length_evaluation = self._evaluate_length(char_count, optimal)
        
        # Análisis de párrafos
        paragraph_analysis = self._analyze_paragraphs(content)
        
        # Análisis de oraciones
        sentence_analysis = self._analyze_sentences(content)
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "optimal_length": optimal,
            "length_evaluation": length_evaluation,
            "paragraph_analysis": paragraph_analysis,
            "sentence_analysis": sentence_analysis,
            "suggestions": self._generate_length_suggestions(
                char_count,
                optimal,
                length_evaluation
            )
        }

    def _evaluate_length(
        self,
        current_length: int,
        optimal: Dict[str, int]
    ) -> Dict[str, Any]:
        """Evaluar longitud"""
        min_len = optimal["min"]
        max_len = optimal["max"]
        optimal_len = optimal["optimal"]
        
        if current_length < min_len:
            status = "too_short"
            score = current_length / min_len
        elif current_length > max_len:
            status = "too_long"
            score = max_len / current_length
        elif current_length == optimal_len:
            status = "optimal"
            score = 1.0
        else:
            status = "acceptable"
            # Score basado en proximidad al óptimo
            if current_length < optimal_len:
                score = 0.7 + (current_length - min_len) / (optimal_len - min_len) * 0.3
            else:
                score = 1.0 - (current_length - optimal_len) / (max_len - optimal_len) * 0.3
        
        return {
            "status": status,
            "score": score,
            "min": min_len,
            "max": max_len,
            "optimal": optimal_len,
            "current": current_length
        }

    def _analyze_paragraphs(self, content: str) -> Dict[str, Any]:
        """Analizar párrafos"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {"avg_length": 0, "count": 0, "issues": []}
        
        lengths = [len(p) for p in paragraphs]
        avg_length = sum(lengths) / len(lengths)
        
        issues = []
        optimal_para = self.optimal_lengths["paragraph"]
        
        for i, para in enumerate(paragraphs):
            para_len = len(para)
            if para_len < optimal_para["min"]:
                issues.append({
                    "paragraph": i + 1,
                    "issue": "too_short",
                    "length": para_len,
                    "suggestion": "Considera expandir este párrafo"
                })
            elif para_len > optimal_para["max"]:
                issues.append({
                    "paragraph": i + 1,
                    "issue": "too_long",
                    "length": para_len,
                    "suggestion": "Considera dividir este párrafo"
                })
        
        return {
            "avg_length": avg_length,
            "count": len(paragraphs),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "issues": issues
        }

    def _analyze_sentences(self, content: str) -> Dict[str, Any]:
        """Analizar oraciones"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if not sentences:
            return {"avg_length": 0, "count": 0, "issues": []}
        
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        issues = []
        optimal_sent = self.optimal_lengths["sentence"]
        
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            if word_count < optimal_sent["min"]:
                issues.append({
                    "sentence": i + 1,
                    "issue": "too_short",
                    "word_count": word_count,
                    "suggestion": "Considera combinar con otra oración"
                })
            elif word_count > optimal_sent["max"]:
                issues.append({
                    "sentence": i + 1,
                    "issue": "too_long",
                    "word_count": word_count,
                    "suggestion": "Considera dividir esta oración"
                })
        
        return {
            "avg_length": avg_length,
            "count": len(sentences),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "issues": issues
        }

    def _generate_length_suggestions(
        self,
        current_length: int,
        optimal: Dict[str, int],
        evaluation: Dict[str, Any]
    ) -> List[str]:
        """Generar sugerencias de longitud"""
        suggestions = []
        status = evaluation["status"]
        
        if status == "too_short":
            diff = optimal["min"] - current_length
            suggestions.append(f"El contenido es muy corto. Agrega aproximadamente {diff} caracteres.")
            suggestions.append("Considera expandir con más detalles, ejemplos o explicaciones.")
        elif status == "too_long":
            diff = current_length - optimal["max"]
            suggestions.append(f"El contenido es muy largo. Reduce aproximadamente {diff} caracteres.")
            suggestions.append("Considera eliminar información redundante o dividir en secciones.")
        elif status == "optimal":
            suggestions.append("La longitud del contenido es óptima.")
        else:
            suggestions.append("La longitud del contenido es aceptable.")
        
        return suggestions

    def optimize_length(
        self,
        content: str,
        target_length: int,
        method: str = "expand"
    ) -> Dict[str, Any]:
        """
        Optimizar longitud del contenido.

        Args:
            content: Contenido
            target_length: Longitud objetivo
            method: Método (expand, reduce, maintain)

        Returns:
            Contenido optimizado
        """
        current_length = len(content)
        
        if method == "expand" and current_length < target_length:
            # Expandir contenido
            expansion_needed = target_length - current_length
            expanded = self._expand_content(content, expansion_needed)
            return {
                "original_length": current_length,
                "target_length": target_length,
                "new_length": len(expanded),
                "content": expanded,
                "method": "expand"
            }
        elif method == "reduce" and current_length > target_length:
            # Reducir contenido
            reduction_needed = current_length - target_length
            reduced = self._reduce_content(content, reduction_needed)
            return {
                "original_length": current_length,
                "target_length": target_length,
                "new_length": len(reduced),
                "content": reduced,
                "method": "reduce"
            }
        else:
            return {
                "original_length": current_length,
                "target_length": target_length,
                "new_length": current_length,
                "content": content,
                "method": "maintain",
                "message": "No se requiere optimización"
            }

    def _expand_content(self, content: str, expansion_needed: int) -> str:
        """Expandir contenido"""
        # Agregar detalles adicionales
        expansion_phrases = [
            " Es importante mencionar que",
            " Además, cabe destacar que",
            " Vale la pena señalar que",
            " Es relevante considerar que"
        ]
        
        expanded = content
        import random
        while len(expanded) < len(content) + expansion_needed:
            phrase = random.choice(expansion_phrases)
            expanded += phrase + " este aspecto requiere atención adicional."
            
            if len(expanded) >= len(content) + expansion_needed:
                break
        
        return expanded[:len(content) + expansion_needed]

    def _reduce_content(self, content: str, reduction_needed: int) -> str:
        """Reducir contenido"""
        # Eliminar palabras redundantes y frases innecesarias
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Eliminar oraciones más cortas primero
        sentences.sort(key=len)
        
        reduced = content
        removed = 0
        
        for sentence in sentences:
            if removed >= reduction_needed:
                break
            
            if sentence in reduced:
                reduced = reduced.replace(sentence + '.', '', 1)
                removed += len(sentence) + 1
        
        return reduced






