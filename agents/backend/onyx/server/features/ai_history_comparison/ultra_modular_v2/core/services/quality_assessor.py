"""
Quality Assessor Service
========================

Servicio para evaluar la calidad de contenido.
Responsabilidad única: Evaluar calidad y generar recomendaciones.
"""

from typing import Dict, Any, List, Tuple
from ..entities import HistoryEntry


class QualityAssessor:
    """
    Servicio para evaluar la calidad de contenido.
    
    Responsabilidad única: Evaluar la calidad de contenido de IA
    y generar recomendaciones para mejorar la calidad.
    """
    
    def __init__(self):
        """Inicializar el evaluador de calidad."""
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.0
        }
        
        self.metric_weights = {
            'quality': 0.4,
            'readability': 0.3,
            'sentiment': 0.2,
            'word_count': 0.1
        }
    
    def assess(self, entry: HistoryEntry) -> Dict[str, Any]:
        """
        Evaluar la calidad de una entrada.
        
        Args:
            entry: Entrada de historial a evaluar
            
        Returns:
            Evaluación de calidad completa
        """
        # Calcular puntuación de calidad general
        overall_quality = self._calculate_overall_quality(entry)
        
        # Determinar nivel de calidad
        quality_level = self._determine_quality_level(overall_quality)
        
        # Identificar fortalezas y debilidades
        strengths = self._identify_strengths(entry)
        weaknesses = self._identify_weaknesses(entry)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(weaknesses)
        
        # Calcular puntuaciones por categoría
        category_scores = self._calculate_category_scores(entry)
        
        return {
            'overall_quality': overall_quality,
            'quality_level': quality_level,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'category_scores': category_scores,
            'improvement_potential': self._calculate_improvement_potential(entry),
            'quality_breakdown': self._create_quality_breakdown(entry)
        }
    
    def compare_quality(self, entry1: HistoryEntry, entry2: HistoryEntry) -> Dict[str, Any]:
        """
        Comparar calidad entre dos entradas.
        
        Args:
            entry1: Primera entrada
            entry2: Segunda entrada
            
        Returns:
            Comparación de calidad
        """
        assessment1 = self.assess(entry1)
        assessment2 = self.assess(entry2)
        
        return {
            'entry1': {
                'id': entry1.id,
                'model': entry1.model,
                'quality': assessment1['overall_quality'],
                'level': assessment1['quality_level']
            },
            'entry2': {
                'id': entry2.id,
                'model': entry2.model,
                'quality': assessment2['overall_quality'],
                'level': assessment2['quality_level']
            },
            'comparison': {
                'quality_difference': abs(assessment1['overall_quality'] - assessment2['overall_quality']),
                'winner': entry1.model if assessment1['overall_quality'] > assessment2['overall_quality'] else entry2.model,
                'strengths_comparison': self._compare_strengths(assessment1['strengths'], assessment2['strengths']),
                'weaknesses_comparison': self._compare_weaknesses(assessment1['weaknesses'], assessment2['weaknesses'])
            }
        }
    
    def assess_batch(self, entries: List[HistoryEntry]) -> Dict[str, Any]:
        """
        Evaluar calidad de múltiples entradas.
        
        Args:
            entries: Lista de entradas a evaluar
            
        Returns:
            Evaluación en lote
        """
        if not entries:
            return {'error': 'No entries provided'}
        
        assessments = [self.assess(entry) for entry in entries]
        
        # Calcular estadísticas agregadas
        quality_scores = [assessment['overall_quality'] for assessment in assessments]
        quality_levels = [assessment['quality_level'] for assessment in assessments]
        
        return {
            'total_entries': len(entries),
            'average_quality': sum(quality_scores) / len(quality_scores),
            'quality_distribution': self._calculate_quality_distribution(quality_levels),
            'best_entry': self._find_best_entry(entries, assessments),
            'worst_entry': self._find_worst_entry(entries, assessments),
            'assessments': assessments
        }
    
    def _calculate_overall_quality(self, entry: HistoryEntry) -> float:
        """
        Calcular puntuación de calidad general.
        
        Args:
            entry: Entrada de historial
            
        Returns:
            Puntuación de calidad general
        """
        # Puntuación base de calidad
        base_quality = entry.quality
        
        # Ajustes basados en métricas adicionales
        readability_adjustment = entry.readability * 0.3
        sentiment_adjustment = entry.sentiment * 0.2
        word_count_adjustment = min(1.0, entry.words / 100) * 0.1
        
        # Calcular puntuación final
        overall_quality = (
            base_quality * self.metric_weights['quality'] +
            readability_adjustment * self.metric_weights['readability'] +
            sentiment_adjustment * self.metric_weights['sentiment'] +
            word_count_adjustment * self.metric_weights['word_count']
        )
        
        return min(1.0, overall_quality)
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """
        Determinar nivel de calidad.
        
        Args:
            quality_score: Puntuación de calidad
            
        Returns:
            Nivel de calidad
        """
        if quality_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif quality_score >= self.quality_thresholds['good']:
            return 'good'
        elif quality_score >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_strengths(self, entry: HistoryEntry) -> List[str]:
        """
        Identificar fortalezas de la entrada.
        
        Args:
            entry: Entrada de historial
            
        Returns:
            Lista de fortalezas
        """
        strengths = []
        
        if entry.quality >= 0.8:
            strengths.append('high_overall_quality')
        
        if entry.readability >= 0.7:
            strengths.append('high_readability')
        
        if entry.sentiment >= 0.6:
            strengths.append('positive_sentiment')
        
        if entry.words >= 50:
            strengths.append('adequate_length')
        
        if entry.words >= 100:
            strengths.append('substantial_content')
        
        if entry.sentiment >= 0.8:
            strengths.append('very_positive_sentiment')
        
        if entry.readability >= 0.9:
            strengths.append('excellent_readability')
        
        return strengths
    
    def _identify_weaknesses(self, entry: HistoryEntry) -> List[str]:
        """
        Identificar debilidades de la entrada.
        
        Args:
            entry: Entrada de historial
            
        Returns:
            Lista de debilidades
        """
        weaknesses = []
        
        if entry.quality < 0.4:
            weaknesses.append('low_overall_quality')
        
        if entry.readability < 0.5:
            weaknesses.append('low_readability')
        
        if entry.sentiment < 0.4:
            weaknesses.append('negative_sentiment')
        
        if entry.words < 20:
            weaknesses.append('too_short')
        
        if entry.words > 500:
            weaknesses.append('too_long')
        
        if entry.sentiment < 0.2:
            weaknesses.append('very_negative_sentiment')
        
        if entry.readability < 0.3:
            weaknesses.append('very_low_readability')
        
        return weaknesses
    
    def _generate_recommendations(self, weaknesses: List[str]) -> List[str]:
        """
        Generar recomendaciones basadas en debilidades.
        
        Args:
            weaknesses: Lista de debilidades
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        if 'low_readability' in weaknesses or 'very_low_readability' in weaknesses:
            recommendations.append('Improve sentence structure and word choice for better readability')
        
        if 'negative_sentiment' in weaknesses or 'very_negative_sentiment' in weaknesses:
            recommendations.append('Consider using more positive language and tone')
        
        if 'too_short' in weaknesses:
            recommendations.append('Add more detail and context to provide comprehensive information')
        
        if 'too_long' in weaknesses:
            recommendations.append('Consider condensing content for better focus and clarity')
        
        if 'low_overall_quality' in weaknesses:
            recommendations.append('Review and improve overall content quality across all metrics')
        
        if not recommendations:
            recommendations.append('Content quality is good, consider minor refinements for excellence')
        
        return recommendations
    
    def _calculate_category_scores(self, entry: HistoryEntry) -> Dict[str, float]:
        """
        Calcular puntuaciones por categoría.
        
        Args:
            entry: Entrada de historial
            
        Returns:
            Puntuaciones por categoría
        """
        return {
            'overall_quality': entry.quality,
            'readability': entry.readability,
            'sentiment': entry.sentiment,
            'content_length': min(1.0, entry.words / 100),
            'completeness': min(1.0, entry.words / 50)
        }
    
    def _calculate_improvement_potential(self, entry: HistoryEntry) -> float:
        """
        Calcular potencial de mejora.
        
        Args:
            entry: Entrada de historial
            
        Returns:
            Potencial de mejora (0.0 - 1.0)
        """
        current_quality = self._calculate_overall_quality(entry)
        max_possible_quality = 1.0
        
        return max_possible_quality - current_quality
    
    def _create_quality_breakdown(self, entry: HistoryEntry) -> Dict[str, Any]:
        """
        Crear desglose de calidad.
        
        Args:
            entry: Entrada de historial
            
        Returns:
            Desglose de calidad
        """
        return {
            'quality_score': entry.quality,
            'readability_score': entry.readability,
            'sentiment_score': entry.sentiment,
            'word_count': entry.words,
            'quality_grade': self._get_quality_grade(entry.quality),
            'readability_grade': self._get_quality_grade(entry.readability),
            'sentiment_grade': self._get_quality_grade(entry.sentiment)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """
        Obtener calificación de calidad.
        
        Args:
            score: Puntuación de calidad
            
        Returns:
            Calificación (A, B, C, D, F)
        """
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _compare_strengths(self, strengths1: List[str], strengths2: List[str]) -> Dict[str, Any]:
        """
        Comparar fortalezas entre dos entradas.
        
        Args:
            strengths1: Fortalezas de la primera entrada
            strengths2: Fortalezas de la segunda entrada
            
        Returns:
            Comparación de fortalezas
        """
        common_strengths = set(strengths1).intersection(set(strengths2))
        unique_strengths1 = set(strengths1) - set(strengths2)
        unique_strengths2 = set(strengths2) - set(strengths1)
        
        return {
            'common_strengths': list(common_strengths),
            'unique_to_entry1': list(unique_strengths1),
            'unique_to_entry2': list(unique_strengths2),
            'strength_count_comparison': {
                'entry1': len(strengths1),
                'entry2': len(strengths2)
            }
        }
    
    def _compare_weaknesses(self, weaknesses1: List[str], weaknesses2: List[str]) -> Dict[str, Any]:
        """
        Comparar debilidades entre dos entradas.
        
        Args:
            weaknesses1: Debilidades de la primera entrada
            weaknesses2: Debilidades de la segunda entrada
            
        Returns:
            Comparación de debilidades
        """
        common_weaknesses = set(weaknesses1).intersection(set(weaknesses2))
        unique_weaknesses1 = set(weaknesses1) - set(weaknesses2)
        unique_weaknesses2 = set(weaknesses2) - set(weaknesses1)
        
        return {
            'common_weaknesses': list(common_weaknesses),
            'unique_to_entry1': list(unique_weaknesses1),
            'unique_to_entry2': list(unique_weaknesses2),
            'weakness_count_comparison': {
                'entry1': len(weaknesses1),
                'entry2': len(weaknesses2)
            }
        }
    
    def _calculate_quality_distribution(self, quality_levels: List[str]) -> Dict[str, int]:
        """
        Calcular distribución de niveles de calidad.
        
        Args:
            quality_levels: Lista de niveles de calidad
            
        Returns:
            Distribución de niveles de calidad
        """
        distribution = {}
        for level in quality_levels:
            distribution[level] = distribution.get(level, 0) + 1
        return distribution
    
    def _find_best_entry(self, entries: List[HistoryEntry], assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Encontrar la mejor entrada.
        
        Args:
            entries: Lista de entradas
            assessments: Lista de evaluaciones
            
        Returns:
            Mejor entrada
        """
        best_index = max(range(len(assessments)), key=lambda i: assessments[i]['overall_quality'])
        return {
            'entry': entries[best_index].to_dict(),
            'assessment': assessments[best_index]
        }
    
    def _find_worst_entry(self, entries: List[HistoryEntry], assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Encontrar la peor entrada.
        
        Args:
            entries: Lista de entradas
            assessments: Lista de evaluaciones
            
        Returns:
            Peor entrada
        """
        worst_index = min(range(len(assessments)), key=lambda i: assessments[i]['overall_quality'])
        return {
            'entry': entries[worst_index].to_dict(),
            'assessment': assessments[worst_index]
        }




