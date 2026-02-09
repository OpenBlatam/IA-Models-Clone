"""
Mock content templates for AI document generation.

Predefined content structures for different document types.
"""

from typing import List, Dict, Any


MOCK_CONTENT_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "proposal": [
        {
            "title": "Executive Summary",
            "content": "This proposal outlines a comprehensive solution to address the challenges identified in your request. Our approach combines innovative strategies with proven methodologies to deliver exceptional results that align with your organizational goals and objectives.",
            "level": 1,
            "metadata": {"word_count": 45, "key_points": ["Solution overview", "Value proposition", "Expected outcomes"]}
        },
        {
            "title": "Problem Statement",
            "content": "The current situation presents several key challenges that require immediate attention. These challenges impact operational efficiency, stakeholder satisfaction, and long-term strategic objectives. Through careful analysis, we have identified the root causes and developed targeted solutions.",
            "level": 1,
            "metadata": {"word_count": 48, "key_points": ["Current challenges", "Impact analysis", "Root causes"]}
        },
        {
            "title": "Proposed Solution",
            "content": "Our proposed solution addresses the identified challenges through a multi-phased approach that ensures sustainable results. The solution includes strategic planning, implementation support, and ongoing optimization to maximize value delivery.",
            "level": 1,
            "metadata": {"word_count": 42, "key_points": ["Multi-phased approach", "Implementation strategy", "Value maximization"]}
        },
        {
            "title": "Methodology",
            "content": "We employ a proven methodology that combines industry best practices with customized approaches tailored to your specific needs. Our process includes discovery, design, implementation, and optimization phases.",
            "level": 1,
            "metadata": {"word_count": 38, "key_points": ["Best practices", "Customized approach", "Four-phase process"]}
        },
        {
            "title": "Timeline",
            "content": "The implementation timeline is designed to minimize disruption while ensuring rapid value delivery. Key milestones include project initiation, design completion, implementation, testing, and go-live phases.",
            "level": 1,
            "metadata": {"word_count": 35, "key_points": ["Minimal disruption", "Rapid delivery", "Key milestones"]}
        },
        {
            "title": "Budget and Pricing",
            "content": "Our pricing structure is transparent and value-based, ensuring you receive maximum return on investment. The investment includes all necessary resources, tools, and ongoing support throughout the engagement.",
            "level": 1,
            "metadata": {"word_count": 37, "key_points": ["Transparent pricing", "Value-based", "ROI focus"]}
        }
    ],
    "technical": [
        {
            "title": "Overview",
            "content": "This technical document provides comprehensive information about the system architecture, implementation details, and operational procedures. It serves as a reference guide for developers, system administrators, and technical stakeholders.",
            "level": 1,
            "metadata": {"word_count": 40, "key_points": ["System architecture", "Implementation details", "Operational procedures"]}
        },
        {
            "title": "System Architecture",
            "content": "The system is built using modern, scalable architecture patterns that ensure high availability, performance, and maintainability. Key components include the application layer, business logic layer, data access layer, and infrastructure components.",
            "level": 1,
            "metadata": {"word_count": 42, "key_points": ["Scalable architecture", "High availability", "Component layers"]}
        },
        {
            "title": "Installation Guide",
            "content": "The installation process is straightforward and automated where possible. Prerequisites include system requirements, dependencies, and configuration settings. Follow the step-by-step instructions for successful deployment.",
            "level": 1,
            "metadata": {"word_count": 35, "key_points": ["Automated process", "Prerequisites", "Step-by-step"]}
        },
        {
            "title": "Configuration",
            "content": "System configuration involves setting up environment variables, database connections, API endpoints, and security parameters. Each configuration option is documented with examples and best practices.",
            "level": 1,
            "metadata": {"word_count": 33, "key_points": ["Environment setup", "Database configuration", "Security parameters"]}
        },
        {
            "title": "API Reference",
            "content": "The API provides comprehensive endpoints for all system operations. Each endpoint is documented with request/response formats, authentication requirements, and usage examples.",
            "level": 1,
            "metadata": {"word_count": 30, "key_points": ["Comprehensive endpoints", "Request/response formats", "Usage examples"]}
        }
    ],
    "report": [
        {
            "title": "Executive Summary",
            "content": "This report provides a comprehensive analysis of the requested topic, including key findings, insights, and recommendations. The analysis is based on thorough research and industry best practices to ensure accuracy and relevance.",
            "level": 1,
            "metadata": {"word_count": 42, "key_points": ["Comprehensive analysis", "Key findings", "Recommendations"]}
        },
        {
            "title": "Introduction",
            "content": "The introduction sets the context for this analysis, outlining the scope, objectives, and methodology used. This foundation ensures that all subsequent sections build upon a clear understanding of the subject matter.",
            "level": 1,
            "metadata": {"word_count": 38, "key_points": ["Context setting", "Scope definition", "Methodology"]}
        },
        {
            "title": "Findings and Analysis",
            "content": "Our analysis reveals several key insights that are critical for understanding the current situation and future opportunities. These findings are supported by data, research, and industry expertise to ensure reliability and actionable insights.",
            "level": 1,
            "metadata": {"word_count": 42, "key_points": ["Key insights", "Data support", "Actionable results"]}
        },
        {
            "title": "Recommendations",
            "content": "Based on our analysis, we recommend a strategic approach that addresses the identified challenges while capitalizing on opportunities. These recommendations are prioritized and include implementation guidance.",
            "level": 1,
            "metadata": {"word_count": 35, "key_points": ["Strategic approach", "Prioritized actions", "Implementation guidance"]}
        },
        {
            "title": "Conclusion",
            "content": "In conclusion, this analysis provides a clear path forward based on comprehensive research and industry expertise. The recommendations outlined will help achieve the desired outcomes while minimizing risks and maximizing opportunities.",
            "level": 1,
            "metadata": {"word_count": 37, "key_points": ["Clear path forward", "Risk minimization", "Opportunity maximization"]}
        }
    ]
}


def get_mock_content_for_type(doc_type: str) -> List[Dict[str, Any]]:
    """Get mock content template for a document type."""
    return MOCK_CONTENT_TEMPLATES.get(doc_type, MOCK_CONTENT_TEMPLATES["report"])


def detect_document_type_from_prompt(prompt: str) -> str:
    """Detect document type from prompt text."""
    prompt_lower = prompt.lower()
    
    type_keywords = {
        "proposal": "proposal",
        "manual": "manual",
        "technical": "technical"
    }
    
    for keyword, doc_type in type_keywords.items():
        if keyword in prompt_lower:
            return doc_type
    
    return "report"

