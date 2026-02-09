"""
Gunicorn Configuration
=====================

Ultra-advanced Gunicorn configuration for production deployment.
"""

import os
import multiprocessing
from typing import Dict, Any

# Server socket
bind = os.getenv('GUNICORN_BIND', '0.0.0.0:8000')
backlog = int(os.getenv('GUNICORN_BACKLOG', 2048))

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'sync')
worker_connections = int(os.getenv('GUNICORN_WORKER_CONNECTIONS', 1000))
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', 100))

# Timeout settings
timeout = int(os.getenv('GUNICORN_TIMEOUT', 30))
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', 2))
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', 30))

# Logging
accesslog = os.getenv('GUNICORN_ACCESSLOG', '-')
errorlog = os.getenv('GUNICORN_ERRORLOG', '-')
loglevel = os.getenv('GUNICORN_LOGLEVEL', 'info')
access_log_format = os.getenv('GUNICORN_ACCESS_LOG_FORMAT', 
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s')

# Process naming
proc_name = os.getenv('GUNICORN_PROC_NAME', 'bulk_truthgpt')

# Server mechanics
daemon = os.getenv('GUNICORN_DAEMON', 'False').lower() == 'true'
pidfile = os.getenv('GUNICORN_PIDFILE', '/tmp/gunicorn.pid')
user = os.getenv('GUNICORN_USER', None)
group = os.getenv('GUNICORN_GROUP', None)
tmp_upload_dir = os.getenv('GUNICORN_TMP_UPLOAD_DIR', None)

# SSL
keyfile = os.getenv('GUNICORN_KEYFILE', None)
certfile = os.getenv('GUNICORN_CERTFILE', None)

# Security
limit_request_line = int(os.getenv('GUNICORN_LIMIT_REQUEST_LINE', 4094))
limit_request_fields = int(os.getenv('GUNICORN_LIMIT_REQUEST_FIELDS', 100))
limit_request_field_size = int(os.getenv('GUNICORN_LIMIT_REQUEST_FIELD_SIZE', 8190))

# Performance
preload_app = os.getenv('GUNICORN_PRELOAD_APP', 'True').lower() == 'true'
reload = os.getenv('GUNICORN_RELOAD', 'False').lower() == 'true'
reload_extra_files = os.getenv('GUNICORN_RELOAD_EXTRA_FILES', '').split(',')
reload_engine = os.getenv('GUNICORN_RELOAD_ENGINE', 'auto')

# Worker lifecycle
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Bulk TruthGPT server...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Bulk TruthGPT server...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Bulk TruthGPT server is ready. PID: %s", server.pid)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info("Worker received SIGABRT signal")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.info("Processing request: %s %s", req.method, req.uri)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.info("Request processed: %s %s - %s", req.method, req.uri, resp.status)

def child_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info("Worker exited (pid: %s)", worker.pid)

def max_requests_jitter_handler(server, worker):
    """Called when a worker has processed max_requests requests."""
    server.log.info("Worker max requests reached (pid: %s)", worker.pid)

# Custom configuration
def get_config() -> Dict[str, Any]:
    """Get Gunicorn configuration."""
    return {
        'bind': bind,
        'backlog': backlog,
        'workers': workers,
        'worker_class': worker_class,
        'worker_connections': worker_connections,
        'max_requests': max_requests,
        'max_requests_jitter': max_requests_jitter,
        'timeout': timeout,
        'keepalive': keepalive,
        'graceful_timeout': graceful_timeout,
        'accesslog': accesslog,
        'errorlog': errorlog,
        'loglevel': loglevel,
        'access_log_format': access_log_format,
        'proc_name': proc_name,
        'daemon': daemon,
        'pidfile': pidfile,
        'user': user,
        'group': group,
        'tmp_upload_dir': tmp_upload_dir,
        'keyfile': keyfile,
        'certfile': certfile,
        'limit_request_line': limit_request_line,
        'limit_request_fields': limit_request_fields,
        'limit_request_field_size': limit_request_field_size,
        'preload_app': preload_app,
        'reload': reload,
        'reload_extra_files': reload_extra_files,
        'reload_engine': reload_engine
    }









