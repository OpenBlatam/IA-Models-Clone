"""
Logger setup for Ultimate Enhanced Supreme Production system
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from flask import current_app

def setup_logger(app=None):
    """Setup application logger."""
    if app is None:
        app = current_app
    
    if not app.debug and not app.testing:
        # Production logging
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        # File handler for general logs
        file_handler = RotatingFileHandler(
            'logs/ultimate_enhanced_supreme.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        # File handler for error logs
        error_handler = RotatingFileHandler(
            'logs/ultimate_enhanced_supreme_error.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        error_handler.setLevel(logging.ERROR)
        app.logger.addHandler(error_handler)
        
        # File handler for performance logs
        performance_handler = RotatingFileHandler(
            'logs/ultimate_enhanced_supreme_performance.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        performance_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        performance_handler.setLevel(logging.INFO)
        app.logger.addHandler(performance_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('👑 Ultimate Enhanced Supreme Production system startup')
    else:
        # Development logging
        app.logger.setLevel(logging.DEBUG)
        app.logger.info('🔧 Ultimate Enhanced Supreme Production system startup (development)')

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)