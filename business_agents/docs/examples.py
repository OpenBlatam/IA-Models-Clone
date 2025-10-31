"""
API Examples
============

Comprehensive examples for API documentation.
"""

from typing import Dict, Any, List

class ExampleGenerator:
    """Generates comprehensive examples for API documentation."""
    
    @staticmethod
    def get_agent_examples() -> Dict[str, Any]:
        """Get agent-related examples."""
        return {
            "agent_response": {
                "summary": "Business Agent Response",
                "description": "Example response for a business agent",
                "value": {
                    "id": "marketing_001",
                    "name": "Marketing Strategy Agent",
                    "business_area": "marketing",
                    "description": "Handles marketing strategy, campaign planning, and brand management",
                    "capabilities": [
                        {
                            "name": "campaign_planning",
                            "description": "Plan and create marketing campaigns",
                            "input_types": ["target_audience", "budget", "objectives"],
                            "output_types": ["campaign_plan", "timeline", "budget_breakdown"],
                            "parameters": {},
                            "estimated_duration": 600,
                            "required_permissions": [],
                            "tags": ["marketing", "planning", "campaign"]
                        }
                    ],
                    "is_active": True,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "metadata": {
                        "version": "1.0.0",
                        "author": "system",
                        "tags": ["marketing", "strategy", "campaigns"]
                    }
                }
            },
            "agent_list": {
                "summary": "Agent List Response",
                "description": "Example response for listing agents",
                "value": {
                    "agents": [
                        {
                            "id": "marketing_001",
                            "name": "Marketing Strategy Agent",
                            "business_area": "marketing",
                            "description": "Handles marketing strategy and campaign planning",
                            "capabilities": [],
                            "is_active": True,
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z"
                        },
                        {
                            "id": "sales_001",
                            "name": "Sales Operations Agent",
                            "business_area": "sales",
                            "description": "Manages sales processes and lead qualification",
                            "capabilities": [],
                            "is_active": True,
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z"
                        }
                    ],
                    "total": 2,
                    "business_area": None,
                    "is_active": None
                }
            },
            "capability_execution": {
                "summary": "Capability Execution Request",
                "description": "Example request for executing an agent capability",
                "value": {
                    "agent_id": "marketing_001",
                    "capability_name": "campaign_planning",
                    "inputs": {
                        "target_audience": "tech professionals",
                        "budget": 50000,
                        "objectives": "increase_brand_awareness"
                    },
                    "parameters": {
                        "campaign_duration": "3 months",
                        "channels": ["social_media", "content_marketing"]
                    }
                }
            },
            "capability_execution_response": {
                "summary": "Capability Execution Response",
                "description": "Example response for capability execution",
                "value": {
                    "status": "completed",
                    "agent_id": "marketing_001",
                    "capability": "campaign_planning",
                    "result": {
                        "target_audience": "tech professionals",
                        "budget": 50000,
                        "objectives": "increase_brand_awareness",
                        "channels": ["social_media", "content_marketing"],
                        "timeline": "3 months",
                        "expected_reach": 500000,
                        "success_metrics": ["impressions", "clicks", "conversions"]
                    },
                    "error": None,
                    "execution_time": 45,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
        }
    
    @staticmethod
    def get_workflow_examples() -> Dict[str, Any]:
        """Get workflow-related examples."""
        return {
            "workflow_response": {
                "summary": "Workflow Response",
                "description": "Example response for a workflow",
                "value": {
                    "id": "workflow_001",
                    "name": "Marketing Campaign Workflow",
                    "description": "Complete marketing campaign workflow from planning to execution",
                    "business_area": "marketing",
                    "status": "active",
                    "created_by": "user_001",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "variables": {
                        "campaign_budget": 50000,
                        "target_audience": "tech professionals"
                    },
                    "metadata": {
                        "version": "1.0.0",
                        "tags": ["marketing", "campaign", "automation"]
                    },
                    "steps": [
                        {
                            "id": "step_001",
                            "name": "Campaign Planning",
                            "step_type": "task",
                            "description": "Plan the marketing campaign",
                            "agent_type": "marketing_001",
                            "parameters": {
                                "planning_duration": "1 week"
                            },
                            "conditions": None,
                            "next_steps": ["step_002"],
                            "parallel_steps": [],
                            "max_retries": 3,
                            "timeout": 3600,
                            "status": "pending",
                            "created_at": None,
                            "completed_at": None,
                            "error_message": None
                        }
                    ]
                }
            },
            "workflow_creation": {
                "summary": "Workflow Creation Request",
                "description": "Example request for creating a workflow",
                "value": {
                    "name": "Customer Onboarding Workflow",
                    "description": "Automated customer onboarding process",
                    "business_area": "sales",
                    "steps": [
                        {
                            "name": "Lead Qualification",
                            "step_type": "task",
                            "description": "Qualify the lead",
                            "agent_type": "sales_001",
                            "parameters": {
                                "qualification_criteria": "budget > 10000"
                            },
                            "conditions": None,
                            "next_steps": ["step_002"],
                            "parallel_steps": [],
                            "max_retries": 3,
                            "timeout": 1800
                        },
                        {
                            "name": "Send Welcome Email",
                            "step_type": "notification",
                            "description": "Send welcome email to customer",
                            "agent_type": "email_agent",
                            "parameters": {
                                "template": "welcome_template"
                            },
                            "conditions": {
                                "lead_qualified": True
                            },
                            "next_steps": [],
                            "parallel_steps": [],
                            "max_retries": 2,
                            "timeout": 300
                        }
                    ],
                    "variables": {
                        "customer_tier": "premium",
                        "onboarding_duration": "7 days"
                    },
                    "metadata": {
                        "category": "customer_management",
                        "priority": "high"
                    }
                }
            },
            "workflow_execution": {
                "summary": "Workflow Execution Response",
                "description": "Example response for workflow execution",
                "value": {
                    "workflow_id": "workflow_001",
                    "status": "completed",
                    "execution_results": {
                        "step_001": {
                            "status": "completed",
                            "result": {
                                "campaign_plan": "Generated comprehensive campaign plan",
                                "budget_allocation": "Distributed across channels"
                            },
                            "execution_time": 45.5
                        },
                        "step_002": {
                            "status": "completed",
                            "result": {
                                "content_created": "Created campaign content",
                                "assets_generated": 15
                            },
                            "execution_time": 120.0
                        }
                    },
                    "error": None,
                    "executed_at": "2024-01-01T00:00:00Z",
                    "duration": 165.5
                }
            }
        }
    
    @staticmethod
    def get_document_examples() -> Dict[str, Any]:
        """Get document-related examples."""
        return {
            "document_generation_request": {
                "summary": "Document Generation Request",
                "description": "Example request for generating a document",
                "value": {
                    "document_type": "business_plan",
                    "title": "Q1 2024 Marketing Strategy",
                    "description": "Comprehensive marketing strategy for Q1 2024",
                    "business_area": "marketing",
                    "variables": {
                        "quarter": "Q1 2024",
                        "budget": 100000,
                        "target_audience": "enterprise customers",
                        "objectives": ["increase_awareness", "generate_leads", "improve_conversion"]
                    },
                    "template_id": "business_plan_template",
                    "format": "pdf",
                    "priority": "high",
                    "deadline": "2024-01-15T00:00:00Z"
                }
            },
            "document_response": {
                "summary": "Document Response",
                "description": "Example response for a generated document",
                "value": {
                    "id": "doc_001",
                    "request_id": "req_001",
                    "title": "Q1 2024 Marketing Strategy",
                    "content": "# Q1 2024 Marketing Strategy\n\n## Executive Summary\n...",
                    "format": "pdf",
                    "file_path": "/generated_documents/q1_2024_marketing_strategy_20240101_120000.pdf",
                    "size_bytes": 2048576,
                    "created_at": "2024-01-01T00:00:00Z",
                    "metadata": {
                        "template_used": "business_plan_template",
                        "generation_time": 45.5,
                        "business_area": "marketing"
                    }
                }
            },
            "document_list": {
                "summary": "Document List Response",
                "description": "Example response for listing documents",
                "value": {
                    "documents": [
                        {
                            "id": "doc_001",
                            "request_id": "req_001",
                            "title": "Q1 2024 Marketing Strategy",
                            "content": None,
                            "format": "pdf",
                            "file_path": "/generated_documents/q1_2024_marketing_strategy.pdf",
                            "size_bytes": 2048576,
                            "created_at": "2024-01-01T00:00:00Z",
                            "metadata": {
                                "business_area": "marketing",
                                "document_type": "business_plan"
                            }
                        },
                        {
                            "id": "doc_002",
                            "request_id": "req_002",
                            "title": "Sales Performance Report",
                            "content": None,
                            "format": "docx",
                            "file_path": "/generated_documents/sales_performance_report.docx",
                            "size_bytes": 1024768,
                            "created_at": "2024-01-01T00:00:00Z",
                            "metadata": {
                                "business_area": "sales",
                                "document_type": "report"
                            }
                        }
                    ],
                    "total": 2,
                    "business_area": None,
                    "document_type": None
                }
            }
        }
    
    @staticmethod
    def get_system_examples() -> Dict[str, Any]:
        """Get system-related examples."""
        return {
            "health_check": {
                "summary": "Health Check Response",
                "description": "Example response for system health check",
                "value": {
                    "status": "healthy",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "version": "1.0.0",
                    "components": {
                        "database": "healthy",
                        "cache": "healthy",
                        "queue": "healthy",
                        "storage": "healthy"
                    },
                    "metrics": {
                        "uptime": "99.9%",
                        "response_time": "45ms",
                        "active_connections": 150,
                        "memory_usage": "65%",
                        "cpu_usage": "25%"
                    }
                }
            },
            "system_info": {
                "summary": "System Information Response",
                "description": "Example response for system information",
                "value": {
                    "system": {
                        "name": "Business Agents System",
                        "version": "1.0.0",
                        "environment": "production",
                        "uptime": "15 days, 3 hours",
                        "started_at": "2023-12-17T21:00:00Z"
                    },
                    "capabilities": {
                        "agent_execution": True,
                        "workflow_automation": True,
                        "document_generation": True,
                        "real_time_monitoring": True,
                        "caching": True,
                        "rate_limiting": True
                    },
                    "business_areas": [
                        {
                            "value": "marketing",
                            "name": "Marketing",
                            "agents_count": 3,
                            "description": "Marketing strategy, campaigns, and brand management"
                        },
                        {
                            "value": "sales",
                            "name": "Sales",
                            "agents_count": 2,
                            "description": "Sales processes, lead generation, and customer acquisition"
                        }
                    ],
                    "workflow_templates": {
                        "marketing": 5,
                        "sales": 3,
                        "operations": 2,
                        "hr": 1
                    },
                    "configuration": {
                        "max_concurrent_agents": 10,
                        "default_timeout": 300,
                        "cache_ttl": 3600,
                        "rate_limit": 100
                    }
                }
            },
            "system_metrics": {
                "summary": "System Metrics Response",
                "description": "Example response for system metrics",
                "value": {
                    "agents": {
                        "total": 15,
                        "active": 12,
                        "executions_today": 245,
                        "average_execution_time": 45.5,
                        "success_rate": 98.5
                    },
                    "workflows": {
                        "total": 25,
                        "active": 20,
                        "executions_today": 89,
                        "average_duration": 180.0,
                        "success_rate": 95.2
                    },
                    "business_areas": {
                        "marketing": {
                            "agents": 3,
                            "workflows": 8,
                            "executions_today": 120
                        },
                        "sales": {
                            "agents": 2,
                            "workflows": 5,
                            "executions_today": 85
                        }
                    },
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }
        }
    
    @staticmethod
    def get_error_examples() -> Dict[str, Any]:
        """Get error response examples."""
        return {
            "validation_error": {
                "summary": "Validation Error",
                "description": "Example validation error response",
                "value": {
                    "error": "validation_error",
                    "message": "Invalid input data",
                    "details": {
                        "field": "email",
                        "issue": "Invalid email format",
                        "provided": "invalid-email",
                        "expected": "valid email address"
                    },
                    "timestamp": "2024-01-01T00:00:00Z",
                    "request_id": "req_123456"
                }
            },
            "not_found": {
                "summary": "Not Found Error",
                "description": "Example not found error response",
                "value": {
                    "error": "not_found",
                    "message": "Resource not found",
                    "details": {
                        "resource_type": "agent",
                        "resource_id": "nonexistent_agent",
                        "suggestion": "Check the agent ID or list available agents"
                    },
                    "timestamp": "2024-01-01T00:00:00Z",
                    "request_id": "req_123456"
                }
            },
            "rate_limit": {
                "summary": "Rate Limit Exceeded",
                "description": "Example rate limit exceeded response",
                "value": {
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "details": {
                        "limit": 100,
                        "window": "1 minute",
                        "retry_after": 60,
                        "current_usage": 100
                    },
                    "timestamp": "2024-01-01T00:00:00Z",
                    "request_id": "req_123456"
                }
            },
            "internal_error": {
                "summary": "Internal Server Error",
                "description": "Example internal server error response",
                "value": {
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "details": {
                        "error_id": "err_789012",
                        "support_reference": "REF-2024-001"
                    },
                    "timestamp": "2024-01-01T00:00:00Z",
                    "request_id": "req_123456"
                }
            }
        }

def get_examples() -> Dict[str, Any]:
    """Get all examples for API documentation."""
    generator = ExampleGenerator()
    
    return {
        "agents": generator.get_agent_examples(),
        "workflows": generator.get_workflow_examples(),
        "documents": generator.get_document_examples(),
        "system": generator.get_system_examples(),
        "errors": generator.get_error_examples()
    }
