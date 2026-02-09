"""
Pure functions for analytics and reporting logic
Functional programming approach - no classes
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from utils.cache import cache_result


def calculate_trend(
    data_points: List[Dict[str, Any]],
    value_field: str,
    date_field: str
) -> Dict[str, Any]:
    """
    Calculate trend from data points
    
    Args:
        data_points: List of data points with values and dates
        value_field: Field name containing the value
        date_field: Field name containing the date
    
    Returns:
        Dictionary with trend analysis
    """
    if not data_points or len(data_points) < 2:
        return {
            "trend": "insufficient_data",
            "direction": "stable",
            "change_percentage": 0.0
        }
    
    # Sort by date
    sorted_points = sorted(
        data_points,
        key=lambda x: _parse_date(x.get(date_field))
    )
    
    first_value = sorted_points[0].get(value_field, 0)
    last_value = sorted_points[-1].get(value_field, 0)
    
    if first_value == 0:
        change_percentage = 100.0 if last_value > 0 else 0.0
    else:
        change_percentage = ((last_value - first_value) / first_value) * 100
    
    if change_percentage > 5:
        direction = "increasing"
        trend = "positive"
    elif change_percentage < -5:
        direction = "decreasing"
        trend = "negative"
    else:
        direction = "stable"
        trend = "neutral"
    
    return {
        "trend": trend,
        "direction": direction,
        "change_percentage": round(change_percentage, 2),
        "first_value": first_value,
        "last_value": last_value,
        "data_points_count": len(data_points)
    }


def calculate_average_over_period(
    data_points: List[Dict[str, Any]],
    value_field: str,
    period_days: int = 7
) -> float:
    """
    Calculate average value over a period
    
    Args:
        data_points: List of data points
        value_field: Field name containing the value
        period_days: Number of days to average over
    
    Returns:
        Average value
    """
    if not data_points:
        return 0.0
    
    cutoff_date = datetime.now() - timedelta(days=period_days)
    
    recent_points = [
        point for point in data_points
        if _parse_date(point.get("date")) >= cutoff_date
    ]
    
    if not recent_points:
        return 0.0
    
    values = [
        point.get(value_field, 0)
        for point in recent_points
        if isinstance(point.get(value_field), (int, float))
    ]
    
    if not values:
        return 0.0
    
    return sum(values) / len(values)


def identify_patterns(
    data_points: List[Dict[str, Any]],
    pattern_type: str = "daily"
) -> List[Dict[str, Any]]:
    """
    Identify patterns in data
    
    Args:
        data_points: List of data points
        pattern_type: Type of pattern to identify (daily, weekly, monthly)
    
    Returns:
        List of identified patterns
    """
    if not data_points:
        return []
    
    patterns = []
    
    if pattern_type == "daily":
        # Group by hour of day
        hourly_data = {}
        for point in data_points:
            date = _parse_date(point.get("date"))
            if date:
                hour = date.hour
                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(point)
        
        for hour, points in hourly_data.items():
            if len(points) >= 3:  # Need at least 3 occurrences
                patterns.append({
                    "type": "daily",
                    "hour": hour,
                    "frequency": len(points),
                    "description": f"Pattern at {hour}:00"
                })
    
    return patterns


def generate_insights(
    user_id: str,
    progress_data: Dict[str, Any],
    trends: Dict[str, Any]
) -> List[str]:
    """
    Generate personalized insights
    
    Args:
        user_id: User ID
        progress_data: User progress data
        trends: Trend analysis
    
    Returns:
        List of insight messages
    """
    insights = []
    
    days_sober = progress_data.get("days_sober", 0)
    
    if days_sober >= 30:
        insights.append(f"¡Increíble! Has estado sobrio por {days_sober} días. Este es un logro significativo.")
    
    if trends.get("trend") == "positive":
        insights.append("Tu progreso muestra una tendencia positiva. ¡Sigue así!")
    
    if trends.get("trend") == "negative":
        insights.append("Has experimentado algunos desafíos recientemente. Considera revisar tu plan de recuperación.")
    
    streak = progress_data.get("streak_days", 0)
    if streak >= 7:
        insights.append(f"Tienes una racha de {streak} días. ¡Mantén el impulso!")
    
    return insights


@cache_result(ttl=600, key_prefix="analytics")
def generate_comprehensive_analytics(
    user_id: str,
    progress_data: Dict[str, Any],
    entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comprehensive analytics
    
    Returns:
        Dictionary with comprehensive analytics
    """
    trends = calculate_trend(entries, "cravings_level", "date")
    patterns = identify_patterns(entries, "daily")
    insights = generate_insights(user_id, progress_data, trends)
    
    return {
        "user_id": user_id,
        "progress_summary": progress_data,
        "trends": trends,
        "patterns": patterns,
        "insights": insights,
        "generated_at": datetime.now().isoformat()
    }


def _parse_date(date_value: Any) -> datetime | None:
    """Parse date value to datetime"""
    if date_value is None:
        return None
    
    if isinstance(date_value, datetime):
        return date_value
    
    if isinstance(date_value, str):
        try:
            return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        except ValueError:
            return None
    
    return None

