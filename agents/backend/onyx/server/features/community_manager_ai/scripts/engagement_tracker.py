"""
Engagement Tracker Script - Rastreador de Engagement
=====================================================

Script para rastrear y analizar engagement en publicaciones.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from community_manager_ai import CommunityManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def track_engagement(
    post_id: str,
    platform: str,
    manager: CommunityManager
) -> Dict[str, Any]:
    """
    Rastrear engagement de un post
    
    Args:
        post_id: ID del post
        platform: Plataforma del post
        manager: Instancia de CommunityManager
        
    Returns:
        Dict con métricas de engagement
    """
    try:
        analytics = manager.social_connector.get_analytics(platform, post_id)
        
        # Calcular engagement rate (si hay reach disponible)
        reach = analytics.get("reach", 0)
        likes = analytics.get("likes", 0)
        comments = analytics.get("comments", 0)
        shares = analytics.get("shares", 0) or analytics.get("retweets", 0)
        
        total_engagement = likes + comments + shares
        
        engagement_rate = (total_engagement / reach * 100) if reach > 0 else 0
        
        return {
            "post_id": post_id,
            "platform": platform,
            "metrics": analytics,
            "total_engagement": total_engagement,
            "engagement_rate": round(engagement_rate, 2),
            "tracked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error rastreando engagement de {post_id}: {e}")
        return {
            "post_id": post_id,
            "platform": platform,
            "error": str(e)
        }


def track_recent_posts(
    days: int = 7,
    manager: Optional[CommunityManager] = None
) -> List[Dict[str, Any]]:
    """
    Rastrear engagement de posts recientes
    
    Args:
        days: Número de días hacia atrás
        manager: Instancia de CommunityManager
        
    Returns:
        Lista de métricas de engagement
    """
    if not manager:
        manager = CommunityManager()
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    all_posts = manager.scheduler.get_all_posts(status="published")
    
    recent_posts = [
        post for post in all_posts
        if post.get("published_at") and 
        datetime.fromisoformat(post.get("published_at")) >= cutoff_date
    ]
    
    results = []
    
    for post in recent_posts:
        post_id = post.get("id")
        platforms = post.get("platforms", [])
        results_data = post.get("results", {})
        
        for platform in platforms:
            platform_result = results_data.get(platform, {})
            platform_post_id = platform_result.get("post_id")
            
            if platform_post_id:
                engagement = track_engagement(platform_post_id, platform, manager)
                results.append(engagement)
    
    return results


def generate_engagement_report(days: int = 7) -> Dict[str, Any]:
    """
    Generar reporte de engagement
    
    Args:
        days: Número de días a analizar
        
    Returns:
        Dict con reporte completo
    """
    manager = CommunityManager()
    results = track_recent_posts(days, manager)
    
    if not results:
        return {
            "period_days": days,
            "total_posts": 0,
            "message": "No hay posts para analizar"
        }
    
    total_engagement = sum(r.get("total_engagement", 0) for r in results)
    avg_engagement_rate = sum(r.get("engagement_rate", 0) for r in results) / len(results)
    
    platform_stats = {}
    for result in results:
        platform = result.get("platform")
        if platform not in platform_stats:
            platform_stats[platform] = {
                "posts": 0,
                "total_engagement": 0,
                "avg_engagement_rate": 0
            }
        
        platform_stats[platform]["posts"] += 1
        platform_stats[platform]["total_engagement"] += result.get("total_engagement", 0)
        platform_stats[platform]["avg_engagement_rate"] += result.get("engagement_rate", 0)
    
    # Calcular promedios por plataforma
    for platform, stats in platform_stats.items():
        if stats["posts"] > 0:
            stats["avg_engagement_rate"] = round(stats["avg_engagement_rate"] / stats["posts"], 2)
    
    return {
        "period_days": days,
        "total_posts": len(results),
        "total_engagement": total_engagement,
        "average_engagement_rate": round(avg_engagement_rate, 2),
        "platform_stats": platform_stats,
        "generated_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Rastreador de engagement")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Número de días a analizar (default: 7)"
    )
    parser.add_argument(
        "--post-id",
        type=str,
        help="ID específico del post a rastrear"
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Plataforma del post (requerido con --post-id)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output en formato JSON"
    )
    
    args = parser.parse_args()
    
    manager = CommunityManager()
    
    if args.post_id and args.platform:
        result = track_engagement(args.post_id, args.platform, manager)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nEngagement para post {args.post_id} en {args.platform}:")
            print(f"Total engagement: {result.get('total_engagement', 0)}")
            print(f"Engagement rate: {result.get('engagement_rate', 0)}%")
    else:
        report = generate_engagement_report(args.days)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"\nReporte de Engagement (últimos {args.days} días):")
            print(f"Total posts: {report.get('total_posts', 0)}")
            print(f"Total engagement: {report.get('total_engagement', 0)}")
            print(f"Promedio engagement rate: {report.get('average_engagement_rate', 0)}%")
            print("\nPor plataforma:")
            for platform, stats in report.get('platform_stats', {}).items():
                print(f"  {platform}:")
                print(f"    Posts: {stats['posts']}")
                print(f"    Total engagement: {stats['total_engagement']}")
                print(f"    Avg engagement rate: {stats['avg_engagement_rate']}%")




