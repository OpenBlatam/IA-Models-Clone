"""
Model Comparator Service
========================

Servicio para comparar modelos de IA.
Responsabilidad única: Comparar modelos y calcular similitudes.
"""

import re
from typing import Dict, Any, List, Tuple
from ..entities import HistoryEntry, ComparisonResult


class ModelComparator:
    """
    Servicio para comparar modelos de IA.
    
    Responsabilidad única: Comparar modelos de IA y calcular
    similitudes, diferencias de calidad y métricas de comparación.
    """
    
    def __init__(self):
        """Inicializar el comparador de modelos."""
        pass
    
    def compare(self, entry1: HistoryEntry, entry2: HistoryEntry) -> ComparisonResult:
        """
        Comparar dos entradas de historial.
        
        Args:
            entry1: Primera entrada de historial
            entry2: Segunda entrada de historial
            
        Returns:
            Resultado de la comparación
        """
        # Calcular similitud de contenido
        similarity = self._calculate_similarity(entry1.content, entry2.content)
        
        # Calcular diferencia de calidad
        quality_diff = abs(entry1.quality - entry2.quality)
        
        # Crear resultado de comparación
        result = ComparisonResult.create(
            model_a=entry1.model,
            model_b=entry2.model,
            similarity=similarity,
            quality_diff=quality_diff,
            details=self._create_comparison_details(entry1, entry2)
        )
        
        return result
    
    def compare_multiple(self, entries: List[HistoryEntry]) -> List[ComparisonResult]:
        """
        Comparar múltiples entradas entre sí.
        
        Args:
            entries: Lista de entradas de historial
            
        Returns:
            Lista de resultados de comparación
        """
        results = []
        
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                result = self.compare(entries[i], entries[j])
                results.append(result)
        
        return results
    
    def find_most_similar(self, target_entry: HistoryEntry, entries: List[HistoryEntry]) -> Tuple[HistoryEntry, float]:
        """
        Encontrar la entrada más similar a la entrada objetivo.
        
        Args:
            target_entry: Entrada objetivo
            entries: Lista de entradas para comparar
            
        Returns:
            Tupla con la entrada más similar y su puntuación de similitud
        """
        if not entries:
            raise ValueError("No entries to compare")
        
        best_entry = None
        best_similarity = 0.0
        
        for entry in entries:
            if entry.id == target_entry.id:
                continue  # Skip self-comparison
            
            similarity = self._calculate_similarity(target_entry.content, entry.content)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry
        
        return best_entry, best_similarity
    
    def find_most_different(self, target_entry: HistoryEntry, entries: List[HistoryEntry]) -> Tuple[HistoryEntry, float]:
        """
        Encontrar la entrada más diferente a la entrada objetivo.
        
        Args:
            target_entry: Entrada objetivo
            entries: Lista de entradas para comparar
            
        Returns:
            Tupla con la entrada más diferente y su puntuación de similitud
        """
        if not entries:
            raise ValueError("No entries to compare")
        
        worst_entry = None
        worst_similarity = 1.0
        
        for entry in entries:
            if entry.id == target_entry.id:
                continue  # Skip self-comparison
            
            similarity = self._calculate_similarity(target_entry.content, entry.content)
            
            if similarity < worst_similarity:
                worst_similarity = similarity
                worst_entry = entry
        
        return worst_entry, worst_similarity
    
    def calculate_similarity_matrix(self, entries: List[HistoryEntry]) -> List[List[float]]:
        """
        Calcular matriz de similitud para múltiples entradas.
        
        Args:
            entries: Lista de entradas de historial
            
        Returns:
            Matriz de similitud
        """
        n = len(entries)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0  # Self-similarity
                else:
                    similarity = self._calculate_similarity(entries[i].content, entries[j].content)
                    matrix[i][j] = similarity
        
        return matrix
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calcular similitud entre dos contenidos.
        
        Args:
            content1: Primer contenido
            content2: Segundo contenido
            
        Returns:
            Puntuación de similitud (0.0 - 1.0)
        """
        if not content1 or not content2:
            return 0.0
        
        # Extraer palabras
        words1 = set(re.findall(r'\b[a-zA-Z]+\b', content1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]+\b', content2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calcular similitud de Jaccard
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_comparison_details(self, entry1: HistoryEntry, entry2: HistoryEntry) -> Dict[str, Any]:
        """
        Crear detalles de comparación.
        
        Args:
            entry1: Primera entrada
            entry2: Segunda entrada
            
        Returns:
            Diccionario con detalles de comparación
        """
        return {
            'model_a_quality': entry1.quality,
            'model_b_quality': entry2.quality,
            'model_a_words': entry1.words,
            'model_b_words': entry2.words,
            'model_a_readability': entry1.readability,
            'model_b_readability': entry2.readability,
            'model_a_sentiment': entry1.sentiment,
            'model_b_sentiment': entry2.sentiment,
            'quality_winner': entry1.model if entry1.quality > entry2.quality else entry2.model,
            'readability_winner': entry1.model if entry1.readability > entry2.readability else entry2.model,
            'sentiment_winner': entry1.model if entry1.sentiment > entry2.sentiment else entry2.model,
            'word_count_diff': abs(entry1.words - entry2.words),
            'readability_diff': abs(entry1.readability - entry2.readability),
            'sentiment_diff': abs(entry1.sentiment - entry2.sentiment)
        }
    
    def get_comparison_summary(self, result: ComparisonResult) -> Dict[str, Any]:
        """
        Obtener resumen de comparación.
        
        Args:
            result: Resultado de comparación
            
        Returns:
            Resumen de comparación
        """
        return {
            'models': f"{result.model_a} vs {result.model_b}",
            'similarity_level': self._get_similarity_level(result.similarity),
            'quality_difference_level': self._get_quality_diff_level(result.quality_diff),
            'winner': result.get_winner(),
            'is_high_similarity': result.is_high_similarity(),
            'is_significant_quality_diff': result.is_significant_quality_diff(),
            'summary': self._generate_summary(result)
        }
    
    def _get_similarity_level(self, similarity: float) -> str:
        """
        Obtener nivel de similitud.
        
        Args:
            similarity: Puntuación de similitud
            
        Returns:
            Nivel de similitud
        """
        if similarity >= 0.8:
            return "very_high"
        elif similarity >= 0.6:
            return "high"
        elif similarity >= 0.4:
            return "medium"
        elif similarity >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _get_quality_diff_level(self, quality_diff: float) -> str:
        """
        Obtener nivel de diferencia de calidad.
        
        Args:
            quality_diff: Diferencia de calidad
            
        Returns:
            Nivel de diferencia de calidad
        """
        if quality_diff >= 0.3:
            return "significant"
        elif quality_diff >= 0.2:
            return "moderate"
        elif quality_diff >= 0.1:
            return "minor"
        else:
            return "negligible"
    
    def _generate_summary(self, result: ComparisonResult) -> str:
        """
        Generar resumen de comparación.
        
        Args:
            result: Resultado de comparación
            
        Returns:
            Resumen en texto
        """
        similarity_level = self._get_similarity_level(result.similarity)
        quality_diff_level = self._get_quality_diff_level(result.quality_diff)
        winner = result.get_winner()
        
        summary = f"Models {result.model_a} and {result.model_b} show {similarity_level} similarity "
        summary += f"with {quality_diff_level} quality difference."
        
        if winner:
            summary += f" {winner} performs better in terms of quality."
        else:
            summary += " Both models perform similarly in terms of quality."
        
        return summary




