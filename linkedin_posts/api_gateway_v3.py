"""
🚀 Next-Generation API Gateway for Ultra-Optimized LinkedIn Posts Optimization v3.0
=================================================================================

FastAPI-based API gateway with real-time monitoring, A/B testing, and multi-language support.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Import the next-generation service
try:
    from ultra_optimized_linkedin_optimizer_v3 import (
        create_nextgen_service,
        ContentData,
        ContentType,
        OptimizationStrategy,
        Language,
        ABTestConfig
    )
except ImportError:
    # Fallback for development
    print("Warning: Next-generation service not available, using mock service")
    from typing import Protocol
    
    class MockService:
        async def optimize_linkedin_post(self, *args, **kwargs):
            return {"status": "mock", "message": "Service not available"}
    
    create_nextgen_service = lambda: MockService()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
OPTIMIZATION_REQUESTS = Counter('linkedin_optimization_requests_total', 'Total optimization requests', ['strategy', 'language'])
OPTIMIZATION_DURATION = Histogram('linkedin_optimization_duration_seconds', 'Optimization duration in seconds', ['strategy'])
ACTIVE_AB_TESTS = Gauge('linkedin_active_ab_tests', 'Number of active A/B tests')
REAL_TIME_LEARNING_INSIGHTS = Gauge('linkedin_learning_insights_total', 'Total learning insights generated')

class NextGenAPIGateway:
    """Next-generation API gateway for LinkedIn optimization v3.0."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Next-Generation LinkedIn Optimizer v3.0 API",
            description="Revolutionary API for ultra-optimized LinkedIn posts optimization",
            version="3.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize service
        self.service = create_nextgen_service()
        
        # Setup middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
        
        # Setup static files and templates
        self.setup_web_interface()
        
        # Background tasks
        self.background_tasks = []
        
        logger.info("Next-Generation API Gateway v3.0 initialized successfully")
    
    def setup_middleware(self):
        """Setup API middleware."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                "Request processed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def setup_routes(self):
        """Setup API routes."""
        
        # Health check endpoints
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": "3.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "Next-Generation LinkedIn Optimizer v3.0"
            }
        
        @self.app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if service is ready
                await self.service.get_performance_trends()
                return {"status": "ready"}
            except Exception as e:
                logger.error("Service not ready", error=str(e))
                raise HTTPException(status_code=503, detail="Service not ready")
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return StreamingResponse(
                generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        # Core optimization endpoints
        @self.app.post("/api/v3/optimize")
        async def optimize_post(
            request: Dict[str, Any],
            background_tasks: BackgroundTasks
        ):
            """Optimize LinkedIn post with next-generation features."""
            try:
                start_time = time.time()
                
                # Extract parameters
                content = request.get("content")
                strategy = request.get("strategy", "ENGAGEMENT")
                target_language = request.get("target_language")
                enable_ab_testing = request.get("enable_ab_testing", False)
                enable_learning = request.get("enable_learning", True)
                
                # Validate content
                if not content:
                    raise HTTPException(status_code=400, detail="Content is required")
                
                # Create content data
                if isinstance(content, str):
                    content_data = ContentData(
                        id=f"api_{int(time.time())}",
                        content=content,
                        content_type=ContentType.POST,
                        language=Language.ENGLISH
                    )
                else:
                    content_data = ContentData(**content)
                
                # Parse strategy
                try:
                    optimization_strategy = OptimizationStrategy(strategy.upper())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
                
                # Parse target language
                target_lang = None
                if target_language:
                    try:
                        target_lang = Language(target_language.upper())
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid language: {target_language}")
                
                # Optimize
                result = await self.service.optimize_linkedin_post(
                    content_data,
                    optimization_strategy,
                    target_language=target_lang,
                    enable_ab_testing=enable_ab_testing,
                    enable_learning=enable_learning
                )
                
                # Update metrics
                duration = time.time() - start_time
                OPTIMIZATION_REQUESTS.labels(strategy=strategy, language=target_language or "en").inc()
                OPTIMIZATION_DURATION.labels(strategy=strategy).observe(duration)
                
                # Background task for analytics
                background_tasks.add_task(self.log_optimization, result, strategy, duration)
                
                logger.info(
                    "Post optimized successfully",
                    strategy=strategy,
                    duration=duration,
                    optimization_score=result.optimization_score
                )
                
                return {
                    "status": "success",
                    "version": "3.0.0",
                    "result": result,
                    "processing_time": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error("Optimization failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # Batch optimization endpoint
        @self.app.post("/api/v3/optimize/batch")
        async def optimize_batch(
            request: Dict[str, Any],
            background_tasks: BackgroundTasks
        ):
            """Optimize multiple LinkedIn posts in batch."""
            try:
                start_time = time.time()
                
                contents = request.get("contents", [])
                strategy = request.get("strategy", "ENGAGEMENT")
                max_workers = request.get("max_workers", 8)
                
                if not contents:
                    raise HTTPException(status_code=400, detail="Contents list is required")
                
                # Parse strategy
                try:
                    optimization_strategy = OptimizationStrategy(strategy.upper())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
                
                # Process batch
                results = []
                for content in contents:
                    if isinstance(content, str):
                        content_data = ContentData(
                            id=f"batch_{int(time.time())}_{len(results)}",
                            content=content,
                            content_type=ContentType.POST
                        )
                    else:
                        content_data = ContentData(**content)
                    
                    result = await self.service.optimize_linkedin_post(
                        content_data,
                        optimization_strategy
                    )
                    results.append(result)
                
                duration = time.time() - start_time
                
                # Background task for analytics
                background_tasks.add_task(self.log_batch_optimization, results, strategy, duration)
                
                return {
                    "status": "success",
                    "version": "3.0.0",
                    "results": results,
                    "total_posts": len(results),
                    "processing_time": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error("Batch optimization failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # A/B Testing endpoints
        @self.app.post("/api/v3/ab-test/create")
        async def create_ab_test(request: Dict[str, Any]):
            """Create a new A/B test."""
            try:
                config = ABTestConfig(**request)
                test_id = self.service.ab_testing_engine.create_test(config)
                
                # Update metrics
                ACTIVE_AB_TESTS.inc()
                
                return {
                    "status": "success",
                    "test_id": test_id,
                    "message": "A/B test created successfully"
                }
                
            except Exception as e:
                logger.error("A/B test creation failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v3/ab-test/{test_id}/results")
        async def get_ab_test_results(test_id: str):
            """Get A/B test results."""
            try:
                results = self.service.ab_testing_engine.get_test_results(test_id)
                return {
                    "status": "success",
                    "test_id": test_id,
                    "results": results
                }
                
            except Exception as e:
                logger.error("Failed to get A/B test results", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # Real-time Learning endpoints
        @self.app.get("/api/v3/learning/insights")
        async def get_learning_insights():
            """Get real-time learning insights."""
            try:
                insights = await self.service.get_learning_insights()
                return {
                    "status": "success",
                    "insights": insights,
                    "total_insights": len(insights)
                }
                
            except Exception as e:
                logger.error("Failed to get learning insights", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v3/learning/trends")
        async def get_learning_trends():
            """Get performance trends from learning engine."""
            try:
                trends = await self.service.get_performance_trends()
                return {
                    "status": "success",
                    "trends": trends
                }
                
            except Exception as e:
                logger.error("Failed to get learning trends", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # Multi-language endpoints
        @self.app.get("/api/v3/languages/supported")
        async def get_supported_languages():
            """Get list of supported languages."""
            languages = [lang.value for lang in Language]
            return {
                "status": "success",
                "languages": languages,
                "total_languages": len(languages)
            }
        
        @self.app.post("/api/v3/languages/translate")
        async def translate_content(request: Dict[str, Any]):
            """Translate content to target language."""
            try:
                content = request.get("content")
                source_lang = request.get("source_language", "ENGLISH")
                target_lang = request.get("target_language")
                
                if not content or not target_lang:
                    raise HTTPException(status_code=400, detail="Content and target language are required")
                
                # Parse languages
                try:
                    source = Language(source_lang.upper())
                    target = Language(target_lang.upper())
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid language: {e}")
                
                # Translate
                translated = await self.service.multi_language_optimizer.translate_content(
                    content, source, target
                )
                
                return {
                    "status": "success",
                    "original_content": content,
                    "translated_content": translated,
                    "source_language": source_lang,
                    "target_language": target_lang
                }
                
            except Exception as e:
                logger.error("Translation failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # Performance monitoring endpoints
        @self.app.get("/api/v3/performance/stats")
        async def get_performance_stats():
            """Get performance statistics."""
            try:
                stats = self.service.monitor.get_stats()
                return {
                    "status": "success",
                    "stats": stats
                }
                
            except Exception as e:
                logger.error("Failed to get performance stats", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v3/performance/alerts")
        async def get_performance_alerts():
            """Get performance alerts."""
            try:
                alerts = self.service.monitor.get_performance_alerts()
                return {
                    "status": "success",
                    "alerts": alerts
                }
                
            except Exception as e:
                logger.error("Failed to get performance alerts", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # System status endpoints
        @self.app.get("/api/v3/system/status")
        async def get_system_status():
            """Get comprehensive system status."""
            try:
                # Get various system metrics
                performance_stats = self.service.monitor.get_stats()
                active_tests = len(self.service.ab_testing_engine.active_tests)
                learning_insights = len(await self.service.get_learning_insights())
                
                return {
                    "status": "success",
                    "system": {
                        "version": "3.0.0",
                        "uptime": performance_stats.get("total_uptime", 0),
                        "active_optimizations": performance_stats.get("operations", {}),
                        "active_ab_tests": active_tests,
                        "learning_insights": learning_insights,
                        "cache_size": len(self.service.cache),
                        "error_count": len(self.service.error_log)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error("Failed to get system status", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint for real-time updates
        @self.app.websocket("/ws/v3/optimizations")
        async def websocket_endpoint(websocket):
            """WebSocket endpoint for real-time optimization updates."""
            try:
                await websocket.accept()
                logger.info("WebSocket connection established")
                
                # Send initial connection message
                await websocket.send_text(json.dumps({
                    "type": "connection",
                    "message": "Connected to Next-Generation LinkedIn Optimizer v3.0",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                # Keep connection alive and send periodic updates
                while True:
                    await asyncio.sleep(30)  # Send update every 30 seconds
                    
                    # Get system status
                    try:
                        status = await get_system_status()
                        await websocket.send_text(json.dumps({
                            "type": "status_update",
                            "data": status,
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                    except Exception as e:
                        logger.error("Failed to send status update", error=str(e))
                        
            except Exception as e:
                logger.error("WebSocket error", error=str(e))
            finally:
                logger.info("WebSocket connection closed")
    
    def setup_web_interface(self):
        """Setup web interface with static files and templates."""
        try:
            # Create static directory
            static_dir = Path(__file__).parent / "static"
            static_dir.mkdir(exist_ok=True)
            
            # Create templates directory
            templates_dir = Path(__file__).parent / "templates"
            templates_dir.mkdir(exist_ok=True)
            
            # Mount static files
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            
            # Create basic HTML template
            html_template = self.create_html_template()
            with open(templates_dir / "index.html", "w") as f:
                f.write(html_template)
            
            # Create basic CSS
            css_content = self.create_css_styles()
            with open(static_dir / "styles.css", "w") as f:
                f.write(css_content)
            
            # Create basic JavaScript
            js_content = self.create_javascript()
            with open(static_dir / "app.js", "w") as f:
                f.write(js_content)
            
            logger.info("Web interface setup completed")
            
        except Exception as e:
            logger.warning("Web interface setup failed, continuing without it", error=str(e))
    
    def create_html_template(self) -> str:
        """Create HTML template for web interface."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Next-Generation LinkedIn Optimizer v3.0</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🚀 Next-Generation LinkedIn Optimizer v3.0</h1>
            <p>Revolutionary AI-powered content optimization with real-time learning</p>
        </header>
        
        <main class="main">
            <section class="optimization-section">
                <h2>Content Optimization</h2>
                <form id="optimizationForm" class="optimization-form">
                    <div class="form-group">
                        <label for="content">LinkedIn Content:</label>
                        <textarea id="content" name="content" rows="6" placeholder="Enter your LinkedIn post content here..." required></textarea>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="strategy">Optimization Strategy:</label>
                            <select id="strategy" name="strategy">
                                <option value="ENGAGEMENT">Engagement</option>
                                <option value="REACH">Reach</option>
                                <option value="CLICKS">Clicks</option>
                                <option value="SHARES">Shares</option>
                                <option value="COMMENTS">Comments</option>
                                <option value="BRAND_AWARENESS">Brand Awareness</option>
                                <option value="LEAD_GENERATION">Lead Generation</option>
                                <option value="CONVERSION">Conversion</option>
                                <option value="RETENTION">Retention</option>
                                <option value="INFLUENCE">Influence</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="targetLanguage">Target Language:</label>
                            <select id="targetLanguage" name="targetLanguage">
                                <option value="">Keep Original</option>
                                <option value="SPANISH">Spanish</option>
                                <option value="FRENCH">French</option>
                                <option value="GERMAN">German</option>
                                <option value="PORTUGUESE">Portuguese</option>
                                <option value="ITALIAN">Italian</option>
                                <option value="DUTCH">Dutch</option>
                                <option value="RUSSIAN">Russian</option>
                                <option value="CHINESE">Chinese</option>
                                <option value="JAPANESE">Japanese</option>
                                <option value="KOREAN">Korean</option>
                                <option value="ARABIC">Arabic</option>
                                <option value="HINDI">Hindi</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group checkbox-group">
                            <label>
                                <input type="checkbox" id="enableAbTesting" name="enableAbTesting">
                                Enable A/B Testing
                            </label>
                        </div>
                        
                        <div class="form-group checkbox-group">
                            <label>
                                <input type="checkbox" id="enableLearning" name="enableLearning" checked>
                                Enable Real-time Learning
                            </label>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn-optimize">🚀 Optimize Content</button>
                </form>
            </section>
            
            <section class="results-section" id="resultsSection" style="display: none;">
                <h2>Optimization Results</h2>
                <div id="resultsContent" class="results-content"></div>
            </section>
            
            <section class="features-section">
                <h2>Revolutionary Features v3.0</h2>
                <div class="features-grid">
                    <div class="feature-card">
                        <h3>🧠 Real-Time Learning</h3>
                        <p>Continuous model improvement from every optimization</p>
                    </div>
                    <div class="feature-card">
                        <h3>🧪 A/B Testing</h3>
                        <p>Automated testing for content variants</p>
                    </div>
                    <div class="feature-card">
                        <h3>🌍 Multi-Language</h3>
                        <p>13+ languages with cultural adaptation</p>
                    </div>
                    <div class="feature-card">
                        <h3>⚡ Distributed Processing</h3>
                        <p>Scalable computing with Ray integration</p>
                    </div>
                </div>
            </section>
            
            <section class="status-section">
                <h2>System Status</h2>
                <div id="systemStatus" class="status-content">
                    <p>Loading system status...</p>
                </div>
            </section>
        </main>
        
        <footer class="footer">
            <p>&copy; 2024 Next-Generation LinkedIn Optimizer v3.0 - Revolutionary AI-Powered Content Optimization</p>
        </footer>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>"""
    
    def create_css_styles(self) -> str:
        """Create CSS styles for web interface."""
        return """/* Next-Generation LinkedIn Optimizer v3.0 - Styles */

:root {
    --primary-color: #0077b5;
    --secondary-color: #00a0dc;
    --accent-color: #ff6b35;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --error-color: #dc3545;
    --dark-color: #2c3e50;
    --light-color: #ecf0f1;
    --text-color: #333;
    --border-color: #ddd;
    --shadow: 0 2px 10px rgba(0,0,0,0.1);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    text-align: center;
    color: white;
    margin-bottom: 40px;
}

.header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

.main {
    background: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: var(--shadow);
    margin-bottom: 30px;
}

.optimization-section {
    margin-bottom: 40px;
}

.optimization-section h2 {
    color: var(--primary-color);
    margin-bottom: 20px;
    font-size: 1.8rem;
}

.optimization-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.form-group {
    display: flex;
    flex-direction: column;
}

.form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.form-group label {
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--dark-color);
}

.form-group input,
.form-group select,
.form-group textarea {
    padding: 12px;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    font-size: 16px;
    transition: border-color 0.3s ease;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(0, 119, 181, 0.1);
}

.checkbox-group {
    flex-direction: row;
    align-items: center;
    gap: 10px;
}

.checkbox-group input[type="checkbox"] {
    width: 20px;
    height: 20px;
}

.btn-optimize {
    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 15px 30px;
    border: none;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.btn-optimize:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 119, 181, 0.3);
}

.btn-optimize:active {
    transform: translateY(0);
}

.results-section {
    margin-bottom: 40px;
}

.results-section h2 {
    color: var(--success-color);
    margin-bottom: 20px;
}

.results-content {
    background: var(--light-color);
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid var(--success-color);
}

.features-section {
    margin-bottom: 40px;
}

.features-section h2 {
    color: var(--primary-color);
    margin-bottom: 20px;
    text-align: center;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
}

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: var(--shadow);
    text-align: center;
    border: 2px solid transparent;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    border-color: var(--primary-color);
}

.feature-card h3 {
    color: var(--primary-color);
    margin-bottom: 10px;
}

.status-section {
    margin-bottom: 40px;
}

.status-section h2 {
    color: var(--primary-color);
    margin-bottom: 20px;
}

.status-content {
    background: var(--light-color);
    padding: 20px;
    border-radius: 8px;
}

.footer {
    text-align: center;
    color: white;
    opacity: 0.8;
}

/* Responsive Design */
@media (max-width: 768px) {
    .form-row {
        grid-template-columns: 1fr;
    }
    
    .features-grid {
        grid-template-columns: 1fr;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .container {
        padding: 10px;
    }
    
    .main {
        padding: 20px;
    }
}

/* Loading Animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-radius: 50%;
    border-top-color: var(--primary-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Success/Error States */
.success {
    border-color: var(--success-color) !important;
    background-color: rgba(40, 167, 69, 0.1);
}

.error {
    border-color: var(--error-color) !important;
    background-color: rgba(220, 53, 69, 0.1);
}"""
    
    def create_javascript(self) -> str:
        """Create JavaScript for web interface."""
        return """// Next-Generation LinkedIn Optimizer v3.0 - JavaScript

class LinkedInOptimizerApp {
    constructor() {
        this.initializeEventListeners();
        this.loadSystemStatus();
        this.setupWebSocket();
    }
    
    initializeEventListeners() {
        const form = document.getElementById('optimizationForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleOptimization(e));
        }
    }
    
    async handleOptimization(event) {
        event.preventDefault();
        
        const form = event.target;
        const submitButton = form.querySelector('.btn-optimize');
        const originalText = submitButton.textContent;
        
        try {
            // Show loading state
            submitButton.textContent = '🔄 Optimizing...';
            submitButton.disabled = true;
            
            // Get form data
            const formData = new FormData(form);
            const content = formData.get('content');
            const strategy = formData.get('strategy');
            const targetLanguage = formData.get('targetLanguage');
            const enableAbTesting = formData.get('enableAbTesting') === 'on';
            const enableLearning = formData.get('enableLearning') === 'on';
            
            // Prepare request
            const request = {
                content: content,
                strategy: strategy,
                target_language: targetLanguage || null,
                enable_ab_testing: enableAbTesting,
                enable_learning: enableLearning
            };
            
            // Make API call
            const response = await fetch('/api/v3/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Display results
            this.displayResults(result);
            
            // Show success message
            this.showNotification('✅ Content optimized successfully!', 'success');
            
        } catch (error) {
            console.error('Optimization failed:', error);
            this.showNotification(`❌ Optimization failed: ${error.message}`, 'error');
        } finally {
            // Reset button
            submitButton.textContent = originalText;
            submitButton.disabled = false;
        }
    }
    
    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');
        
        if (!resultsSection || !resultsContent) return;
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Format results
        const resultData = result.result;
        const html = `
            <div class="result-item">
                <h3>🎯 Optimization Results</h3>
                <div class="result-metrics">
                    <div class="metric">
                        <strong>Optimization Score:</strong> ${resultData.optimization_score?.toFixed(1)}%
                    </div>
                    <div class="metric">
                        <strong>Confidence Score:</strong> ${resultData.confidence_score?.toFixed(1)}%
                    </div>
                    <div class="metric">
                        <strong>Processing Time:</strong> ${result.processing_time?.toFixed(3)}s
                    </div>
                    <div class="metric">
                        <strong>Model Used:</strong> ${resultData.model_used || 'N/A'}
                    </div>
                </div>
                
                ${resultData.improvements ? `
                <div class="improvements">
                    <h4>🚀 Improvements Applied:</h4>
                    <ul>
                        ${resultData.improvements.map(imp => `<li>${imp}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
                
                ${resultData.optimized_content ? `
                <div class="optimized-content">
                    <h4>✨ Optimized Content:</h4>
                    <div class="content-preview">
                        <p><strong>Original:</strong> ${resultData.original_content?.content || 'N/A'}</p>
                        <p><strong>Optimized:</strong> ${resultData.optimized_content?.content || 'N/A'}</p>
                    </div>
                </div>
                ` : ''}
                
                ${resultData.language_optimizations ? `
                <div class="language-optimizations">
                    <h4>🌐 Language Optimizations:</h4>
                    <p>${JSON.stringify(resultData.language_optimizations, null, 2)}</p>
                </div>
                ` : ''}
                
                ${resultData.ab_test_results ? `
                <div class="ab-test-results">
                    <h4>🧪 A/B Test Results:</h4>
                    <p>Test ID: ${resultData.ab_test_results.test_id}</p>
                    <p>Variant: ${resultData.ab_test_results.variant}</p>
                </div>
                ` : ''}
            </div>
        `;
        
        resultsContent.innerHTML = html;
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/v3/system/status');
            if (response.ok) {
                const status = await response.json();
                this.updateSystemStatus(status);
            }
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }
    
    updateSystemStatus(status) {
        const statusElement = document.getElementById('systemStatus');
        if (!statusElement) return;
        
        const systemData = status.system;
        const html = `
            <div class="status-grid">
                <div class="status-item">
                    <strong>Version:</strong> ${systemData.version}
                </div>
                <div class="status-item">
                    <strong>Uptime:</strong> ${this.formatUptime(systemData.uptime)}
                </div>
                <div class="status-item">
                    <strong>Active A/B Tests:</strong> ${systemData.active_ab_tests}
                </div>
                <div class="status-item">
                    <strong>Learning Insights:</strong> ${systemData.learning_insights}
                </div>
                <div class="status-item">
                    <strong>Cache Size:</strong> ${systemData.cache_size}
                </div>
                <div class="status-item">
                    <strong>Error Count:</strong> ${systemData.error_count}
                </div>
            </div>
        `;
        
        statusElement.innerHTML = html;
    }
    
    formatUptime(seconds) {
        if (!seconds) return 'N/A';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours}h ${minutes}m ${secs}s`;
    }
    
    setupWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/v3/optimizations`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                // Try to reconnect after 5 seconds
                setTimeout(() => this.setupWebSocket(), 5000);
            };
            
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'connection':
                console.log('WebSocket connected:', data.message);
                break;
            case 'status_update':
                this.updateSystemStatus(data.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
            max-width: 300px;
        `;
        
        // Set background color based on type
        switch (type) {
            case 'success':
                notification.style.backgroundColor = '#28a745';
                break;
            case 'error':
                notification.style.backgroundColor = '#dc3545';
                break;
            case 'warning':
                notification.style.backgroundColor = '#ffc107';
                notification.style.color = '#333';
                break;
            default:
                notification.style.backgroundColor = '#0077b5';
        }
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.linkedinOptimizerApp = new LinkedInOptimizerApp();
});

// Auto-refresh system status every 30 seconds
setInterval(() => {
    if (window.linkedinOptimizerApp) {
        window.linkedinOptimizerApp.loadSystemStatus();
    }
}, 30000);"""
    
    async def log_optimization(self, result: Dict[str, Any], strategy: str, duration: float):
        """Log optimization for analytics."""
        try:
            logger.info(
                "Optimization logged for analytics",
                strategy=strategy,
                duration=duration,
                optimization_score=result.get('optimization_score', 0),
                model_used=result.get('model_used', 'unknown')
            )
        except Exception as e:
            logger.error("Failed to log optimization", error=str(e))
    
    async def log_batch_optimization(self, results: List[Dict[str, Any]], strategy: str, duration: float):
        """Log batch optimization for analytics."""
        try:
            logger.info(
                "Batch optimization logged for analytics",
                strategy=strategy,
                duration=duration,
                total_posts=len(results),
                average_score=sum(r.get('optimization_score', 0) for r in results) / len(results) if results else 0
            )
        except Exception as e:
            logger.error("Failed to log batch optimization", error=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the API gateway."""
        logger.info(f"Starting Next-Generation API Gateway v3.0 on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

def main():
    """Main function to run the API gateway."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Next-Generation LinkedIn Optimizer v3.0 API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Create and run API gateway
    gateway = NextGenAPIGateway()
    gateway.run(host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
