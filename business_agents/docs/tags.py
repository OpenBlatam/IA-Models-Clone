"""
API Tags Metadata
=================

Tags metadata for API documentation organization.
"""

from typing import List, Dict, Any

def get_tags_metadata() -> List[Dict[str, Any]]:
    """Get comprehensive tags metadata for API documentation."""
    return [
        {
            "name": "Agents",
            "description": """
# Business Agents

Manage and execute business agents across different domains.

## Key Features

- **Agent Management**: Create, update, and manage business agents
- **Capability Execution**: Execute specific agent capabilities with custom inputs
- **Business Area Organization**: Agents organized by business domains
- **Real-time Execution**: Execute agent capabilities with real-time feedback

## Business Areas

- **Marketing**: Campaign planning, brand management, content strategy
- **Sales**: Lead qualification, customer acquisition, sales processes  
- **Operations**: Process optimization, workflow automation
- **HR**: Employee lifecycle, recruitment, performance management
- **Finance**: Budget analysis, financial reporting, cost optimization
- **Legal**: Contract review, compliance monitoring, document analysis
- **Technical**: System documentation, API specifications, technical guides
- **Content**: Content creation, editorial workflows, publishing

## Usage Examples

### List All Agents
```bash
curl -X GET "https://api.business-agents.com/v1/agents" \\
  -H "X-API-Key: your-api-key"
```

### Execute Agent Capability
```bash
curl -X POST "https://api.business-agents.com/v1/agents/marketing_001/execute" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_id": "marketing_001",
    "capability_name": "campaign_planning",
    "inputs": {
      "target_audience": "tech professionals",
      "budget": 50000,
      "objectives": "increase_brand_awareness"
    }
  }'
```

## Rate Limits

- **List Operations**: 100 requests/minute
- **Execution Operations**: 50 requests/minute
- **Burst Limit**: 10 requests/second
            """,
            "externalDocs": {
                "description": "Agent Documentation",
                "url": "https://docs.business-agents.com/agents"
            }
        },
        {
            "name": "Workflows",
            "description": """
# Workflow Management

Design, execute, and manage complex business workflows.

## Key Features

- **Workflow Design**: Create multi-step workflows with conditional logic
- **Agent Integration**: Integrate business agents into workflow steps
- **Parallel Execution**: Execute multiple steps in parallel
- **Status Tracking**: Real-time workflow execution status
- **Template System**: Pre-built workflow templates for common processes

## Workflow Types

- **Sequential**: Steps execute one after another
- **Parallel**: Multiple steps execute simultaneously
- **Conditional**: Steps execute based on conditions
- **Loop**: Steps repeat based on conditions

## Step Types

- **Task**: Execute an agent capability
- **Notification**: Send notifications or alerts
- **Decision**: Make decisions based on data
- **Integration**: Integrate with external systems
- **Custom**: Execute custom business logic

## Usage Examples

### Create Workflow
```bash
curl -X POST "https://api.business-agents.com/v1/workflows" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Customer Onboarding",
    "description": "Automated customer onboarding process",
    "business_area": "sales",
    "steps": [
      {
        "name": "Qualify Lead",
        "step_type": "task",
        "agent_type": "sales_001",
        "parameters": {"criteria": "budget > 10000"}
      }
    ]
  }'
```

### Execute Workflow
```bash
curl -X POST "https://api.business-agents.com/v1/workflows/workflow_001/execute" \\
  -H "X-API-Key: your-api-key"
```

## Rate Limits

- **List Operations**: 100 requests/minute
- **Execution Operations**: 25 requests/minute
- **Burst Limit**: 5 requests/second
            """,
            "externalDocs": {
                "description": "Workflow Documentation",
                "url": "https://docs.business-agents.com/workflows"
            }
        },
        {
            "name": "Documents",
            "description": """
# Document Generation

Generate business documents in multiple formats with AI assistance.

## Key Features

- **Multi-Format Support**: Generate documents in PDF, DOCX, PPTX, HTML, Markdown, JSON
- **Template System**: Use pre-built templates or create custom ones
- **AI-Powered Content**: AI-assisted content generation and enhancement
- **Variable Substitution**: Dynamic content with variable replacement
- **Batch Processing**: Generate multiple documents efficiently

## Document Types

- **Business Plans**: Strategic business planning documents
- **Reports**: Performance reports and analytics
- **Proposals**: Business proposals and presentations
- **Manuals**: User guides and technical documentation
- **Contracts**: Legal documents and agreements
- **Presentations**: PowerPoint presentations and slides
- **Spreadsheets**: Data analysis and reporting

## Supported Formats

- **PDF**: Professional documents with formatting
- **DOCX**: Microsoft Word documents
- **PPTX**: PowerPoint presentations
- **HTML**: Web-ready documents
- **Markdown**: Developer-friendly format
- **JSON**: Structured data format

## Usage Examples

### Generate Document
```bash
curl -X POST "https://api.business-agents.com/v1/documents/generate" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "document_type": "business_plan",
    "title": "Q1 2024 Marketing Strategy",
    "business_area": "marketing",
    "variables": {
      "budget": 100000,
      "target_audience": "enterprise customers"
    },
    "format": "pdf"
  }'
```

### Download Document
```bash
curl -X GET "https://api.business-agents.com/v1/documents/doc_001/download" \\
  -H "X-API-Key: your-api-key" \\
  -O "marketing_strategy.pdf"
```

## Rate Limits

- **Generation**: 20 requests/minute
- **Download**: 100 requests/minute
- **Burst Limit**: 3 requests/second
            """,
            "externalDocs": {
                "description": "Document Documentation",
                "url": "https://docs.business-agents.com/documents"
            }
        },
        {
            "name": "System",
            "description": """
# System Management

Monitor system health, performance, and configuration.

## Key Features

- **Health Monitoring**: Real-time system health checks
- **Performance Metrics**: System performance and usage statistics
- **Configuration Management**: System configuration and settings
- **Business Area Overview**: Information about available business areas
- **System Information**: Detailed system capabilities and status

## Monitoring Endpoints

- **Health Check**: System health status and component status
- **System Info**: Detailed system information and capabilities
- **Metrics**: Performance metrics and usage statistics
- **Business Areas**: Available business areas and agent counts

## Health Status

- **Healthy**: All systems operational
- **Degraded**: Some systems experiencing issues
- **Unhealthy**: Critical systems down

## Usage Examples

### Health Check
```bash
curl -X GET "https://api.business-agents.com/v1/system/health" \\
  -H "X-API-Key: your-api-key"
```

### System Information
```bash
curl -X GET "https://api.business-agents.com/v1/system/info" \\
  -H "X-API-Key: your-api-key"
```

### System Metrics
```bash
curl -X GET "https://api.business-agents.com/v1/system/metrics" \\
  -H "X-API-Key: your-api-key"
```

## Rate Limits

- **Health Check**: 1000 requests/minute
- **System Info**: 100 requests/minute
- **Metrics**: 100 requests/minute
- **Burst Limit**: 50 requests/second
            """,
            "externalDocs": {
                "description": "System Documentation",
                "url": "https://docs.business-agents.com/system"
            }
        },
        {
            "name": "Health",
            "description": """
# Health Monitoring

Monitor system health and component status.

## Health Checks

The health endpoint provides real-time information about system status:

- **Overall Status**: System-wide health status
- **Component Status**: Individual component health
- **Performance Metrics**: Key performance indicators
- **Timestamp**: Last health check time

## Component Monitoring

- **Database**: Database connectivity and performance
- **Cache**: Redis cache system status
- **Queue**: Background task queue status
- **Storage**: File storage system status
- **External APIs**: Third-party service connectivity

## Response Codes

- **200 OK**: System is healthy
- **503 Service Unavailable**: System is unhealthy or degraded

## Usage

```bash
curl -X GET "https://api.business-agents.com/v1/health"
```

No authentication required for health checks.
            """
        }
    ]
