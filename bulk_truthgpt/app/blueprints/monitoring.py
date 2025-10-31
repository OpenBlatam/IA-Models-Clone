"""
Monitoring blueprint for Ultimate Enhanced Supreme Production system
"""

from flask import Blueprint, request, jsonify, current_app
from flask_restful import Resource, Api
from marshmallow import Schema, fields, ValidationError
from app.utils.decorators import performance_monitor, error_handler, validate_request
from app.services.monitoring_service import MonitoringService
from app.models.monitoring import SystemMetrics, PerformanceMetrics, HealthStatus
import time

monitoring_bp = Blueprint('monitoring', __name__)
api = Api(monitoring_bp)

# Schemas
class MonitoringQuerySchema(Schema):
    """Monitoring query schema."""
    start_time = fields.DateTime(missing=None)
    end_time = fields.DateTime(missing=None)
    metric_types = fields.List(fields.Str(), missing=None)
    aggregation_level = fields.Str(missing='minute', validate=lambda x: x in ['second', 'minute', 'hour', 'day'])
    include_details = fields.Bool(missing=False)

class AlertConfigSchema(Schema):
    """Alert configuration schema."""
    metric_name = fields.Str(required=True)
    threshold_value = fields.Float(required=True)
    threshold_type = fields.Str(required=True, validate=lambda x: x in ['greater_than', 'less_than', 'equals'])
    alert_level = fields.Str(required=True, validate=lambda x: x in ['info', 'warning', 'critical'])
    enabled = fields.Bool(missing=True)
    notification_channels = fields.List(fields.Str(), missing=[])

# Initialize schemas
monitoring_query_schema = MonitoringQuerySchema()
alert_config_schema = AlertConfigSchema()

# Initialize service
monitoring_service = MonitoringService()

class SystemMetrics(Resource):
    """System metrics resource."""
    
    @performance_monitor
    @error_handler
    def get(self):
        """Get system metrics."""
        start_time = time.perf_counter()
        
        metrics = monitoring_service.get_system_metrics()
        
        response_time = time.perf_counter() - start_time
        
        return {
            'success': True,
            'message': 'System metrics retrieved successfully',
            'data': metrics.to_dict(),
            'response_time': response_time
        }

class PerformanceMetrics(Resource):
    """Performance metrics resource."""
    
    @performance_monitor
    @error_handler
    @validate_request(monitoring_query_schema)
    def get(self):
        """Get performance metrics."""
        start_time = time.perf_counter()
        
        # Get query parameters
        query_params = request.args.to_dict()
        
        # Get performance metrics
        metrics = monitoring_service.get_performance_metrics(query_params)
        
        response_time = time.perf_counter() - start_time
        
        return {
            'success': True,
            'message': 'Performance metrics retrieved successfully',
            'data': metrics,
            'response_time': response_time
        }

class HealthStatus(Resource):
    """Health status resource."""
    
    @performance_monitor
    @error_handler
    def get(self):
        """Get health status."""
        start_time = time.perf_counter()
        
        health_status = monitoring_service.get_health_status()
        
        response_time = time.perf_counter() - start_time
        
        return {
            'success': True,
            'message': 'Health status retrieved successfully',
            'data': {
                'status': health_status.value,
                'timestamp': time.time(),
                'details': monitoring_service.get_health_details()
            },
            'response_time': response_time
        }

class Alerts(Resource):
    """Alerts resource."""
    
    @performance_monitor
    @error_handler
    def get(self):
        """Get active alerts."""
        start_time = time.perf_counter()
        
        alerts = monitoring_service.get_active_alerts()
        
        response_time = time.perf_counter() - start_time
        
        return {
            'success': True,
            'message': 'Active alerts retrieved successfully',
            'data': alerts,
            'response_time': response_time
        }
    
    @performance_monitor
    @error_handler
    @validate_request(alert_config_schema)
    def post(self):
        """Create alert configuration."""
        start_time = time.perf_counter()
        
        # Get request data
        request_data = request.get_json()
        
        # Create alert configuration
        alert_config = monitoring_service.create_alert_config(request_data)
        
        response_time = time.perf_counter() - start_time
        
        return {
            'success': True,
            'message': 'Alert configuration created successfully',
            'data': alert_config,
            'response_time': response_time
        }

class MonitoringDashboard(Resource):
    """Monitoring dashboard resource."""
    
    @performance_monitor
    @error_handler
    def get(self):
        """Get monitoring dashboard data."""
        start_time = time.perf_counter()
        
        dashboard_data = monitoring_service.get_dashboard_data()
        
        response_time = time.perf_counter() - start_time
        
        return {
            'success': True,
            'message': 'Monitoring dashboard data retrieved successfully',
            'data': dashboard_data,
            'response_time': response_time
        }

# Register resources
api.add_resource(SystemMetrics, '/monitoring/system-metrics')
api.add_resource(PerformanceMetrics, '/monitoring/performance-metrics')
api.add_resource(HealthStatus, '/monitoring/health-status')
api.add_resource(Alerts, '/monitoring/alerts')
api.add_resource(MonitoringDashboard, '/monitoring/dashboard')









