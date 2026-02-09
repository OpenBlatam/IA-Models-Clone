"""
Performance Blueprint
====================

Ultra-advanced performance monitoring with Flask-RESTful.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from flask import request, jsonify, g, current_app
from flask_restful import Resource, Api, reqparse
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, validate
from app import db, cache
from models import PerformanceMetric, OptimizationSession, User
from utils.decorators import rate_limit, monitor_performance, handle_errors
from utils.exceptions import PerformanceError

# Create blueprint and API
performance_bp = Blueprint('performance', __name__)
api = Api(performance_bp)

# Schemas
class PerformanceMetricSchema(Schema):
    """Performance metric schema."""
    metric_name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    metric_value = fields.Float(required=True)
    metric_unit = fields.Str(validate=validate.Length(max=20))
    metadata = fields.Dict()

class PerformanceStatsSchema(Schema):
    """Performance stats schema."""
    start_date = fields.DateTime()
    end_date = fields.DateTime()
    metric_names = fields.List(fields.Str())
    aggregation = fields.Str(validate=validate.OneOf(['avg', 'sum', 'min', 'max', 'count']))

# Initialize schemas
performance_metric_schema = PerformanceMetricSchema()
performance_stats_schema = PerformanceStatsSchema()

class PerformanceMetricsResource(Resource):
    """Performance metrics resource."""
    
    @jwt_required()
    @rate_limit(limit="100 per hour")
    @monitor_performance("performance_metrics_retrieval")
    def get(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            JSON response with performance metrics
        """
        try:
            # Get current user
            current_user_id = get_jwt_identity()
            
            # Parse query parameters
            parser = reqparse.RequestParser()
            parser.add_argument('session_id', type=str, help='Session ID')
            parser.add_argument('metric_name', type=str, help='Metric name')
            parser.add_argument('start_date', type=str, help='Start date')
            parser.add_argument('end_date', type=str, help='End date')
            parser.add_argument('limit', type=int, default=100, help='Limit')
            parser.add_argument('offset', type=int, default=0, help='Offset')
            args = parser.parse_args()
            
            # Build query
            query = PerformanceMetric.query
            
            if args['session_id']:
                query = query.filter_by(session_id=args['session_id'])
            
            if args['metric_name']:
                query = query.filter_by(metric_name=args['metric_name'])
            
            if args['start_date']:
                start_date = datetime.fromisoformat(args['start_date'])
                query = query.filter(PerformanceMetric.timestamp >= start_date)
            
            if args['end_date']:
                end_date = datetime.fromisoformat(args['end_date'])
                query = query.filter(PerformanceMetric.timestamp <= end_date)
            
            # Execute query with pagination
            metrics = query.order_by(PerformanceMetric.timestamp.desc()).offset(args['offset']).limit(args['limit']).all()
            
            return {
                'metrics': [{
                    'id': str(metric.id),
                    'session_id': str(metric.session_id),
                    'metric_name': metric.metric_name,
                    'metric_value': metric.metric_value,
                    'metric_unit': metric.metric_unit,
                    'timestamp': metric.timestamp.isoformat(),
                    'metadata': metric.metadata
                } for metric in metrics],
                'pagination': {
                    'limit': args['limit'],
                    'offset': args['offset'],
                    'total': query.count()
                }
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Performance metrics retrieval error: {str(e)}")
            raise PerformanceError(f"Failed to retrieve performance metrics: {str(e)}")
    
    @jwt_required()
    @rate_limit(limit="50 per hour")
    @monitor_performance("performance_metric_creation")
    def post(self) -> Dict[str, Any]:
        """
        Create performance metric.
        
        Returns:
            JSON response with created metric
        """
        try:
            # Get current user
            current_user_id = get_jwt_identity()
            
            # Parse request data
            parser = reqparse.RequestParser()
            parser.add_argument('session_id', type=str, required=True, help='Session ID is required')
            parser.add_argument('metric_name', type=str, required=True, help='Metric name is required')
            parser.add_argument('metric_value', type=float, required=True, help='Metric value is required')
            parser.add_argument('metric_unit', type=str, help='Metric unit')
            parser.add_argument('metadata', type=dict, help='Metadata')
            args = parser.parse_args()
            
            # Validate session belongs to user
            session = OptimizationSession.query.filter_by(
                id=args['session_id'],
                user_id=current_user_id
            ).first()
            
            if not session:
                return {'error': 'Session not found'}, 404
            
            # Create performance metric
            metric = PerformanceMetric(
                session_id=args['session_id'],
                metric_name=args['metric_name'],
                metric_value=args['metric_value'],
                metric_unit=args.get('metric_unit'),
                metadata=args.get('metadata', {})
            )
            
            db.session.add(metric)
            db.session.commit()
            
            current_app.logger.info(f"Performance metric created: {metric.id}")
            
            return {
                'id': str(metric.id),
                'session_id': str(metric.session_id),
                'metric_name': metric.metric_name,
                'metric_value': metric.metric_value,
                'metric_unit': metric.metric_unit,
                'timestamp': metric.timestamp.isoformat(),
                'metadata': metric.metadata
            }, 201
            
        except Exception as e:
            current_app.logger.error(f"Performance metric creation error: {str(e)}")
            raise PerformanceError(f"Failed to create performance metric: {str(e)}")

class PerformanceStatsResource(Resource):
    """Performance statistics resource."""
    
    @jwt_required()
    @rate_limit(limit="100 per hour")
    @monitor_performance("performance_stats_calculation")
    def get(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            JSON response with performance statistics
        """
        try:
            # Get current user
            current_user_id = get_jwt_identity()
            
            # Parse query parameters
            parser = reqparse.RequestParser()
            parser.add_argument('session_id', type=str, help='Session ID')
            parser.add_argument('metric_name', type=str, help='Metric name')
            parser.add_argument('start_date', type=str, help='Start date')
            parser.add_argument('end_date', type=str, help='End date')
            parser.add_argument('aggregation', type=str, default='avg', help='Aggregation type')
            args = parser.parse_args()
            
            # Build query
            query = PerformanceMetric.query
            
            if args['session_id']:
                query = query.filter_by(session_id=args['session_id'])
            
            if args['metric_name']:
                query = query.filter_by(metric_name=args['metric_name'])
            
            if args['start_date']:
                start_date = datetime.fromisoformat(args['start_date'])
                query = query.filter(PerformanceMetric.timestamp >= start_date)
            
            if args['end_date']:
                end_date = datetime.fromisoformat(args['end_date'])
                query = query.filter(PerformanceMetric.timestamp <= end_date)
            
            # Calculate statistics based on aggregation type
            if args['aggregation'] == 'avg':
                stats = query.with_entities(
                    PerformanceMetric.metric_name,
                    db.func.avg(PerformanceMetric.metric_value).label('value')
                ).group_by(PerformanceMetric.metric_name).all()
            elif args['aggregation'] == 'sum':
                stats = query.with_entities(
                    PerformanceMetric.metric_name,
                    db.func.sum(PerformanceMetric.metric_value).label('value')
                ).group_by(PerformanceMetric.metric_name).all()
            elif args['aggregation'] == 'min':
                stats = query.with_entities(
                    PerformanceMetric.metric_name,
                    db.func.min(PerformanceMetric.metric_value).label('value')
                ).group_by(PerformanceMetric.metric_name).all()
            elif args['aggregation'] == 'max':
                stats = query.with_entities(
                    PerformanceMetric.metric_name,
                    db.func.max(PerformanceMetric.metric_value).label('value')
                ).group_by(PerformanceMetric.metric_name).all()
            elif args['aggregation'] == 'count':
                stats = query.with_entities(
                    PerformanceMetric.metric_name,
                    db.func.count(PerformanceMetric.metric_value).label('value')
                ).group_by(PerformanceMetric.metric_name).all()
            else:
                return {'error': 'Invalid aggregation type'}, 400
            
            return {
                'statistics': [{
                    'metric_name': stat.metric_name,
                    'value': float(stat.value),
                    'aggregation': args['aggregation']
                } for stat in stats],
                'parameters': {
                    'session_id': args['session_id'],
                    'metric_name': args['metric_name'],
                    'start_date': args['start_date'],
                    'end_date': args['end_date'],
                    'aggregation': args['aggregation']
                }
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Performance stats calculation error: {str(e)}")
            raise PerformanceError(f"Failed to calculate performance statistics: {str(e)}")

class PerformanceHealthResource(Resource):
    """Performance health resource."""
    
    @rate_limit(limit="1000 per hour")
    @monitor_performance("performance_health_check")
    def get(self) -> Dict[str, Any]:
        """
        Get performance health status.
        
        Returns:
            JSON response with performance health
        """
        try:
            # Get system performance metrics
            cpu_usage = _get_cpu_usage()
            memory_usage = _get_memory_usage()
            disk_usage = _get_disk_usage()
            network_latency = _get_network_latency()
            
            # Calculate overall health score
            health_score = _calculate_health_score(cpu_usage, memory_usage, disk_usage, network_latency)
            
            # Determine health status
            if health_score >= 0.9:
                status = 'excellent'
            elif health_score >= 0.7:
                status = 'good'
            elif health_score >= 0.5:
                status = 'fair'
            else:
                status = 'poor'
            
            return {
                'status': status,
                'health_score': health_score,
                'metrics': {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'network_latency': network_latency
                },
                'timestamp': datetime.utcnow().isoformat(),
                'recommendations': _get_performance_recommendations(health_score)
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Performance health check error: {str(e)}")
            raise PerformanceError(f"Failed to check performance health: {str(e)}")

# Register resources
api.add_resource(PerformanceMetricsResource, '/metrics')
api.add_resource(PerformanceStatsResource, '/stats')
api.add_resource(PerformanceHealthResource, '/health')

# Helper functions
def _get_cpu_usage() -> float:
    """Get CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0

def _get_memory_usage() -> float:
    """Get memory usage percentage."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0

def _get_disk_usage() -> float:
    """Get disk usage percentage."""
    try:
        import psutil
        return psutil.disk_usage('/').percent
    except ImportError:
        return 0.0

def _get_network_latency() -> float:
    """Get network latency in milliseconds."""
    try:
        import time
        start_time = time.time()
        # This would implement actual network latency check
        return (time.time() - start_time) * 1000
    except Exception:
        return 0.0

def _calculate_health_score(cpu_usage: float, memory_usage: float, disk_usage: float, network_latency: float) -> float:
    """Calculate overall health score."""
    try:
        # Normalize metrics (lower is better)
        cpu_score = max(0, 1 - (cpu_usage / 100))
        memory_score = max(0, 1 - (memory_usage / 100))
        disk_score = max(0, 1 - (disk_usage / 100))
        network_score = max(0, 1 - (network_latency / 1000))  # Assuming 1000ms is max acceptable
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # CPU, Memory, Disk, Network
        scores = [cpu_score, memory_score, disk_score, network_score]
        
        return sum(weight * score for weight, score in zip(weights, scores))
    except Exception:
        return 0.0

def _get_performance_recommendations(health_score: float) -> List[str]:
    """Get performance recommendations based on health score."""
    recommendations = []
    
    if health_score < 0.5:
        recommendations.extend([
            'Consider scaling up resources',
            'Optimize database queries',
            'Enable caching',
            'Review application performance'
        ])
    elif health_score < 0.7:
        recommendations.extend([
            'Monitor resource usage',
            'Consider optimization',
            'Review performance metrics'
        ])
    else:
        recommendations.append('System performance is good')
    
    return recommendations

# Import datetime for type hints
from datetime import datetime









