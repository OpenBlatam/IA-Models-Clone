from fastapi import APIRouter
from fastapi.responses import RedirectResponse


router = APIRouter(prefix="/docs", tags=["docs"])


@router.get("")
async def redirect_to_swagger():
    return RedirectResponse(url="/docs")


@router.get("/redoc")
async def redirect_to_redoc():
    return RedirectResponse(url="/redoc")

"""
API Documentation Blueprint
===========================

Ultra-advanced API documentation with Flask-RESTX and Swagger.
"""

import logging
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, g, current_app
from flask_restx import Api, Resource, Namespace, fields, reqparse
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from models import User, OptimizationSession, PerformanceMetric
from utils.decorators import rate_limit, monitor_performance, handle_errors

# Create blueprint and API
api_docs_bp = Blueprint('api_docs', __name__)
api = Api(
    api_docs_bp,
    title='Bulk TruthGPT API',
    version='1.0.0',
    description='Ultra-advanced optimization system with Flask best practices',
    doc='/docs/',
    prefix='/api/v1'
)

# Create namespaces
auth_ns = Namespace('auth', description='Authentication operations')
optimization_ns = Namespace('optimization', description='Optimization operations')
performance_ns = Namespace('performance', description='Performance operations')
security_ns = Namespace('security', description='Security operations')
ml_ns = Namespace('ml', description='Machine Learning operations')
ai_ns = Namespace('ai', description='AI operations')
quantum_ns = Namespace('quantum', description='Quantum computing operations')
edge_ns = Namespace('edge', description='Edge computing operations')

# Add namespaces to API
api.add_namespace(auth_ns)
api.add_namespace(optimization_ns)
api.add_namespace(performance_ns)
api.add_namespace(security_ns)
api.add_namespace(ml_ns)
api.add_namespace(ai_ns)
api.add_namespace(quantum_ns)
api.add_namespace(edge_ns)

# Define models
user_model = api.model('User', {
    'id': fields.String(required=True, description='User ID'),
    'username': fields.String(required=True, description='Username'),
    'email': fields.String(required=True, description='Email address'),
    'is_active': fields.Boolean(description='User active status'),
    'is_admin': fields.Boolean(description='User admin status'),
    'created_at': fields.DateTime(description='Creation timestamp'),
    'last_login': fields.DateTime(description='Last login timestamp')
})

login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

register_model = api.model('Register', {
    'username': fields.String(required=True, description='Username'),
    'email': fields.String(required=True, description='Email address'),
    'password': fields.String(required=True, description='Password'),
    'confirm_password': fields.String(required=True, description='Confirm password')
})

optimization_session_model = api.model('OptimizationSession', {
    'id': fields.String(required=True, description='Session ID'),
    'session_name': fields.String(required=True, description='Session name'),
    'optimization_type': fields.String(required=True, description='Optimization type'),
    'status': fields.String(description='Session status'),
    'parameters': fields.Raw(description='Session parameters'),
    'results': fields.Raw(description='Optimization results'),
    'metrics': fields.Raw(description='Performance metrics'),
    'created_at': fields.DateTime(description='Creation timestamp'),
    'updated_at': fields.DateTime(description='Update timestamp'),
    'completed_at': fields.DateTime(description='Completion timestamp')
})

performance_metric_model = api.model('PerformanceMetric', {
    'id': fields.String(required=True, description='Metric ID'),
    'session_id': fields.String(required=True, description='Session ID'),
    'metric_name': fields.String(required=True, description='Metric name'),
    'metric_value': fields.Float(required=True, description='Metric value'),
    'metric_unit': fields.String(description='Metric unit'),
    'timestamp': fields.DateTime(description='Metric timestamp'),
    'metadata': fields.Raw(description='Metric metadata')
})

error_model = api.model('Error', {
    'error': fields.String(required=True, description='Error message'),
    'details': fields.String(description='Error details')
})

success_model = api.model('Success', {
    'message': fields.String(required=True, description='Success message'),
    'data': fields.Raw(description='Response data')
})

# Authentication endpoints
@auth_ns.route('/login')
class LoginResource(Resource):
    """User login resource."""
    
    @api.expect(login_model)
    @api.marshal_with(user_model, code=200)
    @api.marshal_with(error_model, code=400)
    @api.marshal_with(error_model, code=401)
    @rate_limit(limit="5 per minute")
    @monitor_performance("user_login")
    @handle_errors
    def post(self):
        """
        User login endpoint.
        
        Authenticate user with username and password.
        Returns access and refresh tokens.
        """
        try:
            # Parse request data
            parser = reqparse.RequestParser()
            parser.add_argument('username', type=str, required=True, help='Username is required')
            parser.add_argument('password', type=str, required=True, help='Password is required')
            args = parser.parse_args()
            
            # Find user
            user = User.query.filter_by(username=args['username']).first()
            if not user or not user.check_password(args['password']):
                return {'error': 'Invalid username or password'}, 401
            
            # Check if user is active
            if not user.is_active:
                return {'error': 'Account is disabled'}, 401
            
            # Create tokens
            from flask_jwt_extended import create_access_token, create_refresh_token
            access_token = create_access_token(identity=str(user.id))
            refresh_token = create_refresh_token(identity=str(user.id))
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'user': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email,
                    'is_active': user.is_active,
                    'is_admin': user.is_admin,
                    'last_login': user.last_login.isoformat() if user.last_login else None
                }
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Login error: {str(e)}")
            return {'error': 'Internal server error'}, 500

@auth_ns.route('/register')
class RegisterResource(Resource):
    """User registration resource."""
    
    @api.expect(register_model)
    @api.marshal_with(success_model, code=201)
    @api.marshal_with(error_model, code=400)
    @rate_limit(limit="3 per minute")
    @monitor_performance("user_registration")
    @handle_errors
    def post(self):
        """
        User registration endpoint.
        
        Register a new user with username, email, and password.
        """
        try:
            # Parse request data
            parser = reqparse.RequestParser()
            parser.add_argument('username', type=str, required=True, help='Username is required')
            parser.add_argument('email', type=str, required=True, help='Email is required')
            parser.add_argument('password', type=str, required=True, help='Password is required')
            parser.add_argument('confirm_password', type=str, required=True, help='Confirm password is required')
            args = parser.parse_args()
            
            # Validate passwords match
            if args['password'] != args['confirm_password']:
                return {'error': 'Passwords do not match'}, 400
            
            # Check if user already exists
            if User.query.filter_by(username=args['username']).first():
                return {'error': 'Username already exists'}, 400
            
            if User.query.filter_by(email=args['email']).first():
                return {'error': 'Email already exists'}, 400
            
            # Create new user
            user = User(
                username=args['username'],
                email=args['email'],
                password_hash=User.hash_password(args['password'])
            )
            
            db.session.add(user)
            db.session.commit()
            
            return {
                'message': 'User registered successfully',
                'data': {
                    'id': str(user.id),
                    'username': user.username,
                    'email': user.email
                }
            }, 201
            
        except Exception as e:
            current_app.logger.error(f"Registration error: {str(e)}")
            return {'error': 'Internal server error'}, 500

# Optimization endpoints
@optimization_ns.route('/sessions')
class OptimizationSessionsResource(Resource):
    """Optimization sessions resource."""
    
    @api.marshal_with(optimization_session_model, code=200)
    @api.marshal_with(error_model, code=401)
    @jwt_required()
    @rate_limit(limit="100 per hour")
    @monitor_performance("optimization_sessions_list")
    @handle_errors
    def get(self):
        """
        List optimization sessions.
        
        Get paginated list of optimization sessions for the authenticated user.
        """
        try:
            # Get current user
            current_user_id = get_jwt_identity()
            
            # Parse query parameters
            parser = reqparse.RequestParser()
            parser.add_argument('page', type=int, default=1, help='Page number')
            parser.add_argument('per_page', type=int, default=10, help='Items per page')
            parser.add_argument('type', type=str, help='Optimization type filter')
            parser.add_argument('status', type=str, help='Status filter')
            args = parser.parse_args()
            
            # Build query
            query = OptimizationSession.query.filter_by(user_id=current_user_id)
            
            if args['type']:
                query = query.filter_by(optimization_type=args['type'])
            
            if args['status']:
                query = query.filter_by(status=args['status'])
            
            # Paginate results
            sessions = query.order_by(OptimizationSession.created_at.desc()).paginate(
                page=args['page'], per_page=args['per_page'], error_out=False
            )
            
            return {
                'sessions': [{
                    'id': str(session.id),
                    'session_name': session.session_name,
                    'optimization_type': session.optimization_type,
                    'status': session.status,
                    'parameters': session.parameters,
                    'results': session.results,
                    'metrics': session.metrics,
                    'created_at': session.created_at.isoformat(),
                    'updated_at': session.updated_at.isoformat(),
                    'completed_at': session.completed_at.isoformat() if session.completed_at else None
                } for session in sessions.items],
                'pagination': {
                    'page': args['page'],
                    'per_page': args['per_page'],
                    'total': sessions.total,
                    'pages': sessions.pages,
                    'has_next': sessions.has_next,
                    'has_prev': sessions.has_prev
                }
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Optimization sessions list error: {str(e)}")
            return {'error': 'Internal server error'}, 500

# Performance endpoints
@performance_ns.route('/metrics')
class PerformanceMetricsResource(Resource):
    """Performance metrics resource."""
    
    @api.marshal_with(performance_metric_model, code=200)
    @api.marshal_with(error_model, code=401)
    @jwt_required()
    @rate_limit(limit="100 per hour")
    @monitor_performance("performance_metrics_retrieval")
    @handle_errors
    def get(self):
        """
        Get performance metrics.
        
        Retrieve performance metrics with optional filtering.
        """
        try:
            # Get current user
            current_user_id = get_jwt_identity()
            
            # Parse query parameters
            parser = reqparse.RequestParser()
            parser.add_argument('session_id', type=str, help='Session ID filter')
            parser.add_argument('metric_name', type=str, help='Metric name filter')
            parser.add_argument('start_date', type=str, help='Start date filter')
            parser.add_argument('end_date', type=str, help='End date filter')
            parser.add_argument('limit', type=int, default=100, help='Limit results')
            parser.add_argument('offset', type=int, default=0, help='Offset results')
            args = parser.parse_args()
            
            # Build query
            query = PerformanceMetric.query
            
            if args['session_id']:
                query = query.filter_by(session_id=args['session_id'])
            
            if args['metric_name']:
                query = query.filter_by(metric_name=args['metric_name'])
            
            if args['start_date']:
                from datetime import datetime
                start_date = datetime.fromisoformat(args['start_date'])
                query = query.filter(PerformanceMetric.timestamp >= start_date)
            
            if args['end_date']:
                from datetime import datetime
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
            return {'error': 'Internal server error'}, 500

# Health check endpoint
@api.route('/health')
class HealthResource(Resource):
    """Health check resource."""
    
    @api.marshal_with(success_model, code=200)
    @rate_limit(limit="1000 per hour")
    @monitor_performance("health_check")
    @handle_errors
    def get(self):
        """
        Health check endpoint.
        
        Check system health and return status information.
        """
        try:
            # Get system metrics
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Calculate health score
            health_score = (100 - cpu_usage) * (100 - memory_usage) * (100 - disk_usage) / 10000
            
            if health_score >= 0.9:
                status = 'excellent'
            elif health_score >= 0.7:
                status = 'good'
            elif health_score >= 0.5:
                status = 'fair'
            else:
                status = 'poor'
            
            return {
                'message': 'System is healthy',
                'data': {
                    'status': status,
                    'health_score': health_score,
                    'metrics': {
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'disk_usage': disk_usage
                    },
                    'timestamp': g.get('request_start_time')
                }
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"Health check error: {str(e)}")
            return {'error': 'Internal server error'}, 500

# API documentation endpoint
@api.route('/docs')
class DocsResource(Resource):
    """API documentation resource."""
    
    @api.marshal_with(success_model, code=200)
    @rate_limit(limit="100 per hour")
    @monitor_performance("api_docs")
    @handle_errors
    def get(self):
        """
        API documentation endpoint.
        
        Get API documentation and available endpoints.
        """
        try:
            return {
                'message': 'API documentation available',
                'data': {
                    'swagger_ui': '/api/v1/docs/',
                    'openapi_spec': '/api/v1/swagger.json',
                    'version': '1.0.0',
                    'title': 'Bulk TruthGPT API',
                    'description': 'Ultra-advanced optimization system with Flask best practices'
                }
            }, 200
            
        except Exception as e:
            current_app.logger.error(f"API docs error: {str(e)}")
            return {'error': 'Internal server error'}, 500




