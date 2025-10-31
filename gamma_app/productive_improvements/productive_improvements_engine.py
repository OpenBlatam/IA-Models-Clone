"""
Gamma App - Productive Improvements Engine
Only real, productive improvements that actually work
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ProductiveImprovementType(Enum):
    """Productive improvement types"""
    BUG_FIX = "bug_fix"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USER_EXPERIENCE = "user_experience"
    MAINTENANCE = "maintenance"

@dataclass
class ProductiveImprovement:
    """Productive improvement representation"""
    improvement_id: str
    title: str
    description: str
    improvement_type: ProductiveImprovementType
    effort_minutes: int
    impact: str  # "low", "medium", "high"
    status: str  # "pending", "done"
    created_at: datetime
    completed_at: Optional[datetime] = None
    notes: str = ""

class ProductiveImprovementsEngine:
    """
    Productive improvements engine for real, working improvements
    """
    
    def __init__(self):
        """Initialize productive improvements engine"""
        self.improvements: Dict[str, ProductiveImprovement] = {}
        self.total_improvements = 0
        self.completed_improvements = 0
        
        # Load productive improvements that work
        self._load_productive_improvements()
    
    def _load_productive_improvements(self):
        """Load productive improvements that work"""
        productive_improvements = [
            {
                "title": "Fix 404 error page",
                "description": "Create a proper 404 error page instead of default server error",
                "type": "bug_fix",
                "effort_minutes": 15,
                "impact": "medium",
                "implementation": "Create custom 404.html template and configure in FastAPI"
            },
            {
                "title": "Add loading spinner",
                "description": "Show loading spinner when API calls are in progress",
                "type": "user_experience",
                "effort_minutes": 30,
                "impact": "high",
                "implementation": "Add CSS spinner and JavaScript to show/hide during API calls"
            },
            {
                "title": "Improve error messages",
                "description": "Make error messages more user-friendly and helpful",
                "type": "user_experience",
                "effort_minutes": 20,
                "impact": "high",
                "implementation": "Replace technical error messages with user-friendly ones"
            },
            {
                "title": "Add request logging",
                "description": "Log all incoming requests for debugging",
                "type": "maintenance",
                "effort_minutes": 10,
                "impact": "medium",
                "implementation": "Add middleware to log request method, URL, and timestamp"
            },
            {
                "title": "Add environment variables",
                "description": "Move hardcoded values to environment variables",
                "type": "maintenance",
                "effort_minutes": 15,
                "impact": "medium",
                "implementation": "Create .env file and use python-dotenv to load variables"
            },
            {
                "title": "Add basic health check",
                "description": "Add /health endpoint to check if app is running",
                "type": "maintenance",
                "effort_minutes": 5,
                "impact": "medium",
                "implementation": "Create simple endpoint that returns {'status': 'ok'}"
            },
            {
                "title": "Add CORS headers",
                "description": "Allow frontend to make requests to API",
                "type": "bug_fix",
                "effort_minutes": 10,
                "impact": "high",
                "implementation": "Add CORS middleware to allow requests from frontend domain"
            },
            {
                "title": "Add input validation",
                "description": "Validate user input before processing",
                "type": "security",
                "effort_minutes": 25,
                "impact": "high",
                "implementation": "Add Pydantic models to validate request data"
            },
            {
                "title": "Add rate limiting",
                "description": "Prevent too many requests from same IP",
                "type": "security",
                "effort_minutes": 20,
                "impact": "medium",
                "implementation": "Use slowapi to limit requests per IP address"
            },
            {
                "title": "Add database connection pooling",
                "description": "Reuse database connections instead of creating new ones",
                "type": "performance",
                "effort_minutes": 30,
                "impact": "high",
                "implementation": "Configure connection pool in database settings"
            },
            {
                "title": "Add basic authentication",
                "description": "Protect API endpoints with simple username/password",
                "type": "security",
                "effort_minutes": 45,
                "impact": "high",
                "implementation": "Add HTTP Basic Auth to protect sensitive endpoints"
            },
            {
                "title": "Add request timeout",
                "description": "Set timeout for external API calls",
                "type": "performance",
                "effort_minutes": 15,
                "impact": "medium",
                "implementation": "Add timeout parameter to HTTP requests"
            },
            {
                "title": "Add error retry logic",
                "description": "Retry failed requests automatically",
                "type": "bug_fix",
                "effort_minutes": 25,
                "impact": "medium",
                "implementation": "Add retry decorator for external API calls"
            },
            {
                "title": "Add response compression",
                "description": "Compress API responses to reduce bandwidth",
                "type": "performance",
                "effort_minutes": 10,
                "impact": "medium",
                "implementation": "Add gzip compression middleware"
            },
            {
                "title": "Add request ID tracking",
                "description": "Add unique ID to each request for debugging",
                "type": "maintenance",
                "effort_minutes": 15,
                "impact": "medium",
                "implementation": "Generate UUID for each request and log it"
            },
            {
                "title": "Add basic metrics",
                "description": "Track basic app metrics like request count",
                "type": "maintenance",
                "effort_minutes": 20,
                "impact": "low",
                "implementation": "Add simple counter for requests and errors"
            },
            {
                "title": "Add graceful shutdown",
                "description": "Handle server shutdown properly",
                "type": "maintenance",
                "effort_minutes": 15,
                "impact": "medium",
                "implementation": "Add signal handlers for graceful shutdown"
            },
            {
                "title": "Add request size limit",
                "description": "Limit size of incoming requests",
                "type": "security",
                "effort_minutes": 10,
                "impact": "medium",
                "implementation": "Set max request size in FastAPI configuration"
            },
            {
                "title": "Add response caching",
                "description": "Cache responses for frequently requested data",
                "type": "performance",
                "effort_minutes": 30,
                "impact": "high",
                "implementation": "Add simple in-memory cache for GET requests"
            },
            {
                "title": "Add database query optimization",
                "description": "Optimize slow database queries",
                "type": "performance",
                "effort_minutes": 45,
                "impact": "high",
                "implementation": "Add database indexes and optimize query structure"
            }
        ]
        
        for improvement_data in productive_improvements:
            self.create_improvement(
                title=improvement_data["title"],
                description=improvement_data["description"],
                improvement_type=ProductiveImprovementType(improvement_data["type"]),
                effort_minutes=improvement_data["effort_minutes"],
                impact=improvement_data["impact"]
            )
    
    def create_improvement(self, title: str, description: str, 
                          improvement_type: ProductiveImprovementType,
                          effort_minutes: int, impact: str) -> str:
        """Create a productive improvement"""
        try:
            improvement_id = f"pi_{int(time.time() * 1000)}"
            
            improvement = ProductiveImprovement(
                improvement_id=improvement_id,
                title=title,
                description=description,
                improvement_type=improvement_type,
                effort_minutes=effort_minutes,
                impact=impact,
                status="pending",
                created_at=datetime.now()
            )
            
            self.improvements[improvement_id] = improvement
            self.total_improvements += 1
            
            return improvement_id
            
        except Exception as e:
            print(f"Failed to create improvement: {e}")
            raise
    
    def get_quick_wins(self) -> List[Dict[str, Any]]:
        """Get quick wins that can be done in under 30 minutes"""
        return [
            {
                "title": "Add favicon",
                "description": "Add favicon to prevent 404 errors",
                "effort_minutes": 5,
                "impact": "low",
                "implementation": "Add favicon.ico to static files"
            },
            {
                "title": "Add robots.txt",
                "description": "Add robots.txt for SEO",
                "effort_minutes": 5,
                "impact": "low",
                "implementation": "Create robots.txt file in static directory"
            },
            {
                "title": "Add security headers",
                "description": "Add basic security headers",
                "effort_minutes": 10,
                "impact": "medium",
                "implementation": "Add middleware to set security headers"
            },
            {
                "title": "Add request logging",
                "description": "Log all requests for debugging",
                "effort_minutes": 10,
                "impact": "medium",
                "implementation": "Add simple logging middleware"
            },
            {
                "title": "Add health check",
                "description": "Add basic health check endpoint",
                "effort_minutes": 5,
                "impact": "medium",
                "implementation": "Create /health endpoint"
            },
            {
                "title": "Add error handling",
                "description": "Add global error handler",
                "effort_minutes": 15,
                "impact": "high",
                "implementation": "Add exception handler for common errors"
            },
            {
                "title": "Add input sanitization",
                "description": "Sanitize user input",
                "effort_minutes": 20,
                "impact": "high",
                "implementation": "Add input cleaning functions"
            },
            {
                "title": "Add response time logging",
                "description": "Log response times for monitoring",
                "effort_minutes": 15,
                "impact": "medium",
                "implementation": "Add timing middleware"
            }
        ]
    
    def get_high_impact_improvements(self) -> List[Dict[str, Any]]:
        """Get high impact improvements that users will notice"""
        return [
            {
                "title": "Fix slow page loads",
                "description": "Optimize database queries and add caching",
                "effort_minutes": 60,
                "impact": "high",
                "implementation": "Profile slow queries, add indexes, implement caching"
            },
            {
                "title": "Improve error messages",
                "description": "Make error messages helpful for users",
                "effort_minutes": 30,
                "impact": "high",
                "implementation": "Replace technical errors with user-friendly messages"
            },
            {
                "title": "Add loading states",
                "description": "Show loading indicators during operations",
                "effort_minutes": 45,
                "impact": "high",
                "implementation": "Add loading spinners and progress indicators"
            },
            {
                "title": "Fix broken links",
                "description": "Find and fix all broken internal links",
                "effort_minutes": 30,
                "impact": "high",
                "implementation": "Audit all links and fix 404s"
            },
            {
                "title": "Add form validation",
                "description": "Validate forms before submission",
                "effort_minutes": 40,
                "impact": "high",
                "implementation": "Add client-side and server-side validation"
            },
            {
                "title": "Improve mobile experience",
                "description": "Make app work better on mobile devices",
                "effort_minutes": 90,
                "impact": "high",
                "implementation": "Add responsive design and mobile optimizations"
            }
        ]
    
    def mark_improvement_done(self, improvement_id: str, notes: str = "") -> bool:
        """Mark an improvement as completed"""
        try:
            if improvement_id in self.improvements:
                improvement = self.improvements[improvement_id]
                improvement.status = "done"
                improvement.completed_at = datetime.now()
                improvement.notes = notes
                self.completed_improvements += 1
                return True
            return False
        except Exception as e:
            print(f"Failed to mark improvement as done: {e}")
            return False
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get improvement statistics"""
        try:
            total_effort = sum(imp.effort_minutes for imp in self.improvements.values())
            completed_effort = sum(
                imp.effort_minutes for imp in self.improvements.values() 
                if imp.status == "done"
            )
            
            return {
                "total_improvements": self.total_improvements,
                "completed_improvements": self.completed_improvements,
                "pending_improvements": self.total_improvements - self.completed_improvements,
                "total_effort_minutes": total_effort,
                "completed_effort_minutes": completed_effort,
                "remaining_effort_minutes": total_effort - completed_effort,
                "completion_rate": (self.completed_improvements / self.total_improvements * 100) if self.total_improvements > 0 else 0
            }
        except Exception as e:
            print(f"Failed to get improvement stats: {e}")
            return {}
    
    def get_improvements_by_type(self, improvement_type: ProductiveImprovementType) -> List[ProductiveImprovement]:
        """Get improvements by type"""
        return [
            imp for imp in self.improvements.values() 
            if imp.improvement_type == improvement_type
        ]
    
    def get_improvements_by_impact(self, impact: str) -> List[ProductiveImprovement]:
        """Get improvements by impact level"""
        return [
            imp for imp in self.improvements.values() 
            if imp.impact == impact
        ]
    
    def get_quick_wins_available(self) -> List[Dict[str, Any]]:
        """Get available quick wins"""
        quick_wins = self.get_quick_wins()
        return [
            win for win in quick_wins 
            if win["effort_minutes"] <= 30
        ]
    
    def get_high_impact_available(self) -> List[Dict[str, Any]]:
        """Get available high impact improvements"""
        high_impact = self.get_high_impact_improvements()
        return [
            imp for imp in high_impact 
            if imp["impact"] == "high"
        ]
    
    def export_improvements(self) -> Dict[str, Any]:
        """Export all improvements to JSON"""
        try:
            improvements_data = []
            for improvement in self.improvements.values():
                improvements_data.append({
                    "improvement_id": improvement.improvement_id,
                    "title": improvement.title,
                    "description": improvement.description,
                    "improvement_type": improvement.improvement_type.value,
                    "effort_minutes": improvement.effort_minutes,
                    "impact": improvement.impact,
                    "status": improvement.status,
                    "created_at": improvement.created_at.isoformat(),
                    "completed_at": improvement.completed_at.isoformat() if improvement.completed_at else None,
                    "notes": improvement.notes
                })
            
            return {
                "improvements": improvements_data,
                "stats": self.get_improvement_stats(),
                "exported_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Failed to export improvements: {e}")
            return {}
    
    def import_improvements(self, data: Dict[str, Any]) -> bool:
        """Import improvements from JSON"""
        try:
            if "improvements" in data:
                for imp_data in data["improvements"]:
                    improvement = ProductiveImprovement(
                        improvement_id=imp_data["improvement_id"],
                        title=imp_data["title"],
                        description=imp_data["description"],
                        improvement_type=ProductiveImprovementType(imp_data["improvement_type"]),
                        effort_minutes=imp_data["effort_minutes"],
                        impact=imp_data["impact"],
                        status=imp_data["status"],
                        created_at=datetime.fromisoformat(imp_data["created_at"]),
                        completed_at=datetime.fromisoformat(imp_data["completed_at"]) if imp_data["completed_at"] else None,
                        notes=imp_data["notes"]
                    )
                    self.improvements[improvement.improvement_id] = improvement
                
                self.total_improvements = len(self.improvements)
                self.completed_improvements = len([imp for imp in self.improvements.values() if imp.status == "done"])
                
                return True
            return False
        except Exception as e:
            print(f"Failed to import improvements: {e}")
            return False

# Global productive improvements engine instance
productive_improvements_engine = None

def get_productive_improvements_engine() -> ProductiveImprovementsEngine:
    """Get productive improvements engine instance"""
    global productive_improvements_engine
    if not productive_improvements_engine:
        productive_improvements_engine = ProductiveImprovementsEngine()
    return productive_improvements_engine













