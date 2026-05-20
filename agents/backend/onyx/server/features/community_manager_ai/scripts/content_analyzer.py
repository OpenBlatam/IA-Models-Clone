"""
Content Analyzer Script - Analizador de Contenido
==================================================

Script para analizar contenido y sugerir mejoras.
"""

import logging
from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from community_manager_ai import CommunityManager
from community_manager_ai.services.content_generator import ContentGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_content(content: str, platform: str) -> Dict[str, Any]:
    """
    Analizar contenido y proporcionar sugerencias
    
    Args:
        content: Contenido a analizar
        platform: Plataforma objetivo
        
    Returns:
        Dict con análisis y sugerencias
    """
    generator = ContentGenerator()
    
    analysis = {
        "original_content": content,
        "length": len(content),
        "platform": platform,
        "suggestions": []
    }
    
    # Análisis de longitud
    platform_limits = {
        "twitter": 280,
        "facebook": 5000,
        "instagram": 2200,
        "linkedin": 3000,
        "tiktok": 2200
    }
    
    limit = platform_limits.get(platform.lower(), 5000)
    
    if len(content) > limit:
        analysis["suggestions"].append({
            "type": "length",
            "message": f"Contenido excede el límite de {limit} caracteres para {platform}",
            "action": "Truncar o acortar el contenido"
        })
    
    # Optimizar contenido
    optimized = generator.optimize_for_platform(content, platform)
    if optimized != content:
        analysis["optimized_content"] = optimized
        analysis["suggestions"].append({
            "type": "optimization",
            "message": "Contenido optimizado para la plataforma",
            "optimized": optimized
        })
    
    # Generar hashtags
    hashtags = generator.generate_hashtags(content, platform)
    if hashtags:
        analysis["suggested_hashtags"] = hashtags
        analysis["suggestions"].append({
            "type": "hashtags",
            "message": "Hashtags sugeridos",
            "hashtags": hashtags
        })
    
    return analysis


def analyze_scheduled_posts(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Analizar posts programados
    
    Args:
        limit: Número máximo de posts a analizar
        
    Returns:
        Lista de análisis
    """
    manager = CommunityManager()
    posts = manager.scheduler.get_all_posts(status="scheduled")[:limit]
    
    analyses = []
    
    for post in posts:
        content = post.get("content", "")
        platforms = post.get("platforms", [])
        
        for platform in platforms:
            analysis = analyze_content(content, platform)
            analysis["post_id"] = post.get("id")
            analyses.append(analysis)
    
    return analyses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analizador de contenido")
    parser.add_argument(
        "--content",
        type=str,
        help="Contenido a analizar"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="facebook",
        help="Plataforma objetivo"
    )
    parser.add_argument(
        "--analyze-scheduled",
        action="store_true",
        help="Analizar posts programados"
    )
    
    args = parser.parse_args()
    
    if args.analyze_scheduled:
        analyses = analyze_scheduled_posts()
        for analysis in analyses:
            print(f"\nPost ID: {analysis['post_id']}")
            print(f"Plataforma: {analysis['platform']}")
            print(f"Sugerencias: {len(analysis['suggestions'])}")
            for suggestion in analysis['suggestions']:
                print(f"  - {suggestion['message']}")
    elif args.content:
        analysis = analyze_content(args.content, args.platform)
        print(f"\nAnálisis para {args.platform}:")
        print(f"Longitud: {analysis['length']} caracteres")
        print(f"Sugerencias: {len(analysis['suggestions'])}")
        for suggestion in analysis['suggestions']:
            print(f"  - {suggestion['message']}")
    else:
        parser.print_help()




