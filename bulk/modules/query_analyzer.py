"""
Query Analyzer Module
====================

Analyzes business queries to determine appropriate processing.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class Complexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class QueryAnalysis:
    """Result of query analysis."""
    
    def __init__(self, 
                 primary_area: str,
                 secondary_areas: List[str],
                 document_types: List[str],
                 complexity: Complexity,
                 priority: int,
                 confidence: float):
        self.primary_area = primary_area
        self.secondary_areas = secondary_areas
        self.document_types = document_types
        self.complexity = complexity
        self.priority = priority
        self.confidence = confidence

class QueryAnalyzer:
    """Analyzes business queries to determine processing requirements."""
    
    def __init__(self):
        self.business_areas = {
            'marketing': {
                'keywords': ['marketing', 'campaign', 'brand', 'advertising', 'promotion', 'social media', 'content'],
                'document_types': ['strategy', 'campaign', 'content', 'analysis']
            },
            'sales': {
                'keywords': ['sales', 'proposal', 'client', 'customer', 'revenue', 'pipeline', 'forecast'],
                'document_types': ['proposal', 'presentation', 'playbook', 'forecast']
            },
            'operations': {
                'keywords': ['operations', 'process', 'workflow', 'procedure', 'manual', 'efficiency'],
                'document_types': ['manual', 'procedure', 'workflow', 'report']
            },
            'hr': {
                'keywords': ['hr', 'human resources', 'employee', 'training', 'policy', 'recruitment'],
                'document_types': ['policy', 'training', 'job_description', 'evaluation']
            },
            'finance': {
                'keywords': ['finance', 'budget', 'financial', 'cost', 'revenue', 'investment', 'forecast'],
                'document_types': ['budget', 'forecast', 'analysis', 'report']
            },
            'legal': {
                'keywords': ['legal', 'contract', 'compliance', 'policy', 'agreement', 'terms'],
                'document_types': ['contract', 'policy', 'compliance', 'agreement']
            },
            'technical': {
                'keywords': ['technical', 'documentation', 'system', 'software', 'development', 'architecture'],
                'document_types': ['documentation', 'specification', 'guide', 'troubleshooting']
            },
            'content': {
                'keywords': ['content', 'writing', 'blog', 'article', 'whitepaper', 'case study'],
                'document_types': ['article', 'blog', 'whitepaper', 'case_study']
            },
            'strategy': {
                'keywords': ['strategy', 'plan', 'roadmap', 'initiative', 'business plan', 'goals'],
                'document_types': ['plan', 'roadmap', 'initiative', 'assessment']
            },
            'customer_service': {
                'keywords': ['customer service', 'support', 'help', 'faq', 'training', 'satisfaction'],
                'document_types': ['faq', 'guide', 'policy', 'training']
            }
        }
        
        self.complexity_indicators = {
            Complexity.SIMPLE: ['create', 'write', 'generate', 'make'],
            Complexity.MEDIUM: ['develop', 'design', 'plan', 'analyze'],
            Complexity.COMPLEX: ['comprehensive', 'detailed', 'complete', 'strategic', 'advanced']
        }
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a business query."""
        
        logger.info(f"Analyzing query: {query[:50]}...")
        
        query_lower = query.lower()
        
        # Find business areas
        area_scores = self._calculate_area_scores(query_lower)
        primary_area, secondary_areas = self._determine_areas(area_scores)
        
        # Determine document types
        document_types = self._determine_document_types(query_lower, primary_area)
        
        # Assess complexity
        complexity = self._assess_complexity(query_lower)
        
        # Calculate priority
        priority = self._calculate_priority(complexity, area_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(area_scores, primary_area)
        
        return QueryAnalysis(
            primary_area=primary_area,
            secondary_areas=secondary_areas,
            document_types=document_types,
            complexity=complexity,
            priority=priority,
            confidence=confidence
        )
    
    def _calculate_area_scores(self, query: str) -> Dict[str, float]:
        """Calculate relevance scores for each business area."""
        scores = {}
        
        for area, config in self.business_areas.items():
            score = 0
            keywords = config['keywords']
            
            for keyword in keywords:
                if keyword in query:
                    score += 1
                    # Boost score for exact matches
                    if f" {keyword} " in f" {query} ":
                        score += 0.5
            
            scores[area] = score / len(keywords) if keywords else 0
        
        return scores
    
    def _determine_areas(self, scores: Dict[str, float]) -> Tuple[str, List[str]]:
        """Determine primary and secondary business areas."""
        
        sorted_areas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_area = sorted_areas[0][0] if sorted_areas[0][1] > 0 else 'general'
        
        secondary_areas = [
            area for area, score in sorted_areas[1:3] 
            if score > 0.1
        ]
        
        return primary_area, secondary_areas
    
    def _determine_document_types(self, query: str, primary_area: str) -> List[str]:
        """Determine appropriate document types."""
        
        if primary_area not in self.business_areas:
            return ['report']
        
        available_types = self.business_areas[primary_area]['document_types']
        
        # Check for specific document type mentions
        found_types = []
        for doc_type in available_types:
            if doc_type in query:
                found_types.append(doc_type)
        
        # If no specific types found, return default
        if not found_types:
            found_types = [available_types[0]] if available_types else ['report']
        
        return found_types
    
    def _assess_complexity(self, query: str) -> Complexity:
        """Assess query complexity."""
        
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    return complexity
        
        # Default to medium complexity
        return Complexity.MEDIUM
    
    def _calculate_priority(self, complexity: Complexity, area_scores: Dict[str, float]) -> int:
        """Calculate processing priority (1-5, where 5 is highest)."""
        
        base_priority = {
            Complexity.SIMPLE: 3,
            Complexity.MEDIUM: 2,
            Complexity.COMPLEX: 1
        }
        
        priority = base_priority[complexity]
        
        # Boost priority for high-scoring areas
        max_score = max(area_scores.values()) if area_scores else 0
        if max_score > 0.5:
            priority += 1
        
        return min(priority, 5)
    
    def _calculate_confidence(self, area_scores: Dict[str, float], primary_area: str) -> float:
        """Calculate confidence in the analysis."""
        
        if not area_scores:
            return 0.0
        
        primary_score = area_scores.get(primary_area, 0)
        total_score = sum(area_scores.values())
        
        if total_score == 0:
            return 0.0
        
        confidence = primary_score / total_score
        
        # Boost confidence if primary area has high score
        if primary_score > 0.3:
            confidence += 0.2
        
        return min(confidence, 1.0)

