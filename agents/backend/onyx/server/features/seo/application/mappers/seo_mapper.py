from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, List
from domain.entities.seo_analysis import SEOAnalysis
from application.dto.analyze_url_response import AnalyzeURLResponse, AnalyzeURLResponseDomain
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO Mapper
Object mapper for converting between domain entities and DTOs
"""



class SEOMapper:
    """
    SEO data mapper
    
    This mapper converts between domain entities and DTOs,
    ensuring proper data transformation and validation.
    """
    
    def to_response(self, seo_analysis: SEOAnalysis, cache_hit: bool = False) -> AnalyzeURLResponse:
        """
        Convert SEO analysis to response DTO
        
        Args:
            seo_analysis: SEO analysis domain entity
            cache_hit: Whether result was from cache
            
        Returns:
            AnalyzeURLResponse: Response DTO
        """
        score = seo_analysis.get_score()
        
        return AnalyzeURLResponse(
            url=str(seo_analysis.url),
            title=seo_analysis.title,
            description=seo_analysis.description,
            keywords=seo_analysis.keywords,
            meta_tags=seo_analysis.meta_tags.to_dict(),
            links=seo_analysis.links,
            content_length=seo_analysis.content_length,
            processing_time=seo_analysis.processing_time,
            cache_hit=cache_hit,
            score=score.value,
            grade=score.get_grade().value,
            level=score.get_level(),
            issues=seo_analysis.get_issues(),
            recommendations=seo_analysis.get_recommendations(),
            timestamp=seo_analysis.created_at.isoformat(),
            created_at=seo_analysis.created_at.isoformat()
        )
    
    def to_domain_response(self, seo_analysis: SEOAnalysis, cache_hit: bool = False) -> AnalyzeURLResponseDomain:
        """
        Convert SEO analysis to domain response
        
        Args:
            seo_analysis: SEO analysis domain entity
            cache_hit: Whether result was from cache
            
        Returns:
            AnalyzeURLResponseDomain: Domain response
        """
        score = seo_analysis.get_score()
        
        return AnalyzeURLResponseDomain(
            url=str(seo_analysis.url),
            title=seo_analysis.title,
            description=seo_analysis.description,
            keywords=seo_analysis.keywords,
            meta_tags=seo_analysis.meta_tags.to_dict(),
            links=seo_analysis.links,
            content_length=seo_analysis.content_length,
            processing_time=seo_analysis.processing_time,
            cache_hit=cache_hit,
            score=score.value,
            grade=score.get_grade().value,
            level=score.get_level(),
            issues=seo_analysis.get_issues(),
            recommendations=seo_analysis.get_recommendations(),
            timestamp=seo_analysis.created_at,
            created_at=seo_analysis.created_at
        )
    
    def to_dict(self, seo_analysis: SEOAnalysis) -> Dict[str, Any]:
        """
        Convert SEO analysis to dictionary
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            Dict[str, Any]: Analysis as dictionary
        """
        score = seo_analysis.get_score()
        
        return {
            "url": str(seo_analysis.url),
            "title": seo_analysis.title,
            "description": seo_analysis.description,
            "keywords": seo_analysis.keywords,
            "meta_tags": seo_analysis.meta_tags.to_dict(),
            "links": seo_analysis.links,
            "content_length": seo_analysis.content_length,
            "processing_time": seo_analysis.processing_time,
            "score": score.value,
            "grade": score.get_grade().value,
            "level": score.get_level(),
            "color": score.get_color(),
            "priority": score.get_priority(),
            "issues": seo_analysis.get_issues(),
            "recommendations": seo_analysis.get_recommendations(),
            "summary": seo_analysis.get_summary(),
            "created_at": seo_analysis.created_at.isoformat()
        }
    
    def to_summary(self, seo_analysis: SEOAnalysis) -> Dict[str, Any]:
        """
        Convert SEO analysis to summary
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            Dict[str, Any]: Analysis summary
        """
        return seo_analysis.get_summary()
    
    def to_issues_report(self, seo_analysis: SEOAnalysis) -> Dict[str, Any]:
        """
        Convert SEO analysis to issues report
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            Dict[str, Any]: Issues report
        """
        score = seo_analysis.get_score()
        
        return {
            "url": str(seo_analysis.url),
            "score": score.value,
            "grade": score.get_grade().value,
            "level": score.get_level(),
            "priority": score.get_priority(),
            "issues_count": len(seo_analysis.get_issues()),
            "issues": seo_analysis.get_issues(),
            "recommendations_count": len(seo_analysis.get_recommendations()),
            "recommendations": seo_analysis.get_recommendations(),
            "improvement_potential": score.get_improvement_potential(),
            "improvement_percentage": score.get_improvement_percentage()
        }
    
    def to_performance_report(self, seo_analysis: SEOAnalysis) -> Dict[str, Any]:
        """
        Convert SEO analysis to performance report
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            Dict[str, Any]: Performance report
        """
        score = seo_analysis.get_score()
        
        return {
            "url": str(seo_analysis.url),
            "score": score.value,
            "grade": score.get_grade().value,
            "level": score.get_level(),
            "color": score.get_color(),
            "processing_time": seo_analysis.processing_time,
            "content_length": seo_analysis.content_length,
            "links_count": len(seo_analysis.links),
            "meta_tags_count": len(seo_analysis.meta_tags.tags),
            "has_title": bool(seo_analysis.title),
            "has_description": bool(seo_analysis.description),
            "has_keywords": bool(seo_analysis.keywords),
            "created_at": seo_analysis.created_at.isoformat()
        }
    
    def to_comparison_data(self, seo_analysis: SEOAnalysis) -> Dict[str, Any]:
        """
        Convert SEO analysis to comparison data
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            Dict[str, Any]: Comparison data
        """
        score = seo_analysis.get_score()
        
        return {
            "url": str(seo_analysis.url),
            "score": score.value,
            "grade": score.get_grade().value,
            "level": score.get_level(),
            "content_length": seo_analysis.content_length,
            "links_count": len(seo_analysis.links),
            "meta_tags_count": len(seo_analysis.meta_tags.tags),
            "issues_count": len(seo_analysis.get_issues()),
            "processing_time": seo_analysis.processing_time,
            "created_at": seo_analysis.created_at.isoformat()
        }
    
    def to_export_data(self, seo_analysis: SEOAnalysis, format_type: str = "json") -> Any:
        """
        Convert SEO analysis to export data
        
        Args:
            seo_analysis: SEO analysis domain entity
            format_type: Export format type
            
        Returns:
            Any: Export data
        """
        if format_type == "json":
            return self.to_dict(seo_analysis)
        elif format_type == "csv":
            return self._to_csv_row(seo_analysis)
        elif format_type == "xml":
            return self._to_xml(seo_analysis)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _to_csv_row(self, seo_analysis: SEOAnalysis) -> List[str]:
        """
        Convert SEO analysis to CSV row
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            List[str]: CSV row data
        """
        score = seo_analysis.get_score()
        
        return [
            str(seo_analysis.url),
            seo_analysis.title or "",
            seo_analysis.description or "",
            seo_analysis.keywords or "",
            str(len(seo_analysis.links)),
            str(seo_analysis.content_length),
            str(seo_analysis.processing_time),
            str(score.value),
            score.get_grade().value,
            score.get_level(),
            str(len(seo_analysis.get_issues())),
            str(len(seo_analysis.get_recommendations())),
            seo_analysis.created_at.isoformat()
        ]
    
    def _to_xml(self, seo_analysis: SEOAnalysis) -> str:
        """
        Convert SEO analysis to XML
        
        Args:
            seo_analysis: SEO analysis domain entity
            
        Returns:
            str: XML representation
        """
        score = seo_analysis.get_score()
        
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<seo_analysis>',
            f'  <url>{seo_analysis.url}</url>',
            f'  <title>{seo_analysis.title or ""}</title>',
            f'  <description>{seo_analysis.description or ""}</description>',
            f'  <keywords>{seo_analysis.keywords or ""}</keywords>',
            f'  <content_length>{seo_analysis.content_length}</content_length>',
            f'  <processing_time>{seo_analysis.processing_time}</processing_time>',
            f'  <score>{score.value}</score>',
            f'  <grade>{score.get_grade().value}</grade>',
            f'  <level>{score.get_level()}</level>',
            '  <links>',
        ]
        
        for link in seo_analysis.links:
            xml_parts.append(f'    <link>{link}</link>')
        
        xml_parts.extend([
            '  </links>',
            '  <meta_tags>',
        ])
        
        for name, value in seo_analysis.meta_tags.tags.items():
            xml_parts.append(f'    <meta name="{name}">{value}</meta>')
        
        xml_parts.extend([
            '  </meta_tags>',
            '  <issues>',
        ])
        
        for issue in seo_analysis.get_issues():
            xml_parts.append(f'    <issue>{issue}</issue>')
        
        xml_parts.extend([
            '  </issues>',
            '  <recommendations>',
        ])
        
        for recommendation in seo_analysis.get_recommendations():
            xml_parts.append(f'    <recommendation>{recommendation}</recommendation>')
        
        xml_parts.extend([
            '  </recommendations>',
            f'  <created_at>{seo_analysis.created_at.isoformat()}</created_at>',
            '</seo_analysis>'
        ])
        
        return '\n'.join(xml_parts) 