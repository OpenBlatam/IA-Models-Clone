"""
AI Document Type Classifier Engine
==================================

This module provides an AI-powered system that can identify document types
from a single query and export appropriate template designs.

Supported document types:
- Novel (Fiction)
- Contract (Legal)
- Design (Technical/Architectural)
- Business Plan
- Academic Paper
- Technical Manual
- Marketing Material
- User Manual
- Report
- Proposal
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import openai
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Enumeration of supported document types"""
    NOVEL = "novel"
    CONTRACT = "contract"
    DESIGN = "design"
    BUSINESS_PLAN = "business_plan"
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_MANUAL = "technical_manual"
    MARKETING_MATERIAL = "marketing_material"
    USER_MANUAL = "user_manual"
    REPORT = "report"
    PROPOSAL = "proposal"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    """Result of document type classification"""
    document_type: DocumentType
    confidence: float
    keywords: List[str]
    reasoning: str
    template_suggestions: List[str]

@dataclass
class TemplateDesign:
    """Template design structure"""
    name: str
    document_type: DocumentType
    sections: List[Dict[str, Any]]
    formatting: Dict[str, Any]
    metadata: Dict[str, Any]

class DocumentClassifierEngine:
    """
    AI-powered document type classifier that can identify document types
    from text queries and provide template designs.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the document classifier engine.
        
        Args:
            openai_api_key: OpenAI API key for AI classification
        """
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Document type patterns and keywords
        self.document_patterns = {
            DocumentType.NOVEL: {
                "keywords": [
                    "novel", "fiction", "story", "character", "plot", "chapter",
                    "narrative", "protagonist", "antagonist", "dialogue", "scene",
                    "setting", "theme", "conflict", "resolution", "climax"
                ],
                "patterns": [
                    r"\b(chapter|scene|character|plot|story)\b",
                    r"\b(novel|fiction|narrative)\b",
                    r"\b(protagonist|antagonist|hero|villain)\b"
                ]
            },
            DocumentType.CONTRACT: {
                "keywords": [
                    "contract", "agreement", "terms", "conditions", "party",
                    "obligation", "liability", "indemnity", "clause", "section",
                    "whereas", "hereby", "witnesseth", "legal", "binding"
                ],
                "patterns": [
                    r"\b(contract|agreement|terms|conditions)\b",
                    r"\b(party|parties|obligation|liability)\b",
                    r"\b(whereas|hereby|witnesseth)\b"
                ]
            },
            DocumentType.DESIGN: {
                "keywords": [
                    "design", "blueprint", "specification", "technical", "drawing",
                    "architecture", "engineering", "dimensions", "materials",
                    "components", "assembly", "CAD", "model", "prototype"
                ],
                "patterns": [
                    r"\b(design|blueprint|specification|technical)\b",
                    r"\b(architecture|engineering|dimensions)\b",
                    r"\b(CAD|model|prototype|assembly)\b"
                ]
            },
            DocumentType.BUSINESS_PLAN: {
                "keywords": [
                    "business", "plan", "strategy", "market", "revenue", "profit",
                    "investment", "funding", "executive", "summary", "financial",
                    "projection", "milestone", "objective", "goal"
                ],
                "patterns": [
                    r"\b(business plan|strategy|market analysis)\b",
                    r"\b(revenue|profit|investment|funding)\b",
                    r"\b(executive summary|financial projection)\b"
                ]
            },
            DocumentType.ACADEMIC_PAPER: {
                "keywords": [
                    "research", "study", "analysis", "methodology", "results",
                    "conclusion", "abstract", "introduction", "literature",
                    "review", "hypothesis", "data", "findings", "citation"
                ],
                "patterns": [
                    r"\b(research|study|analysis|methodology)\b",
                    r"\b(abstract|introduction|conclusion)\b",
                    r"\b(literature review|hypothesis|findings)\b"
                ]
            },
            DocumentType.TECHNICAL_MANUAL: {
                "keywords": [
                    "manual", "instruction", "procedure", "step", "guide",
                    "technical", "installation", "configuration", "troubleshooting",
                    "maintenance", "operation", "safety", "warning", "caution"
                ],
                "patterns": [
                    r"\b(manual|instruction|procedure|guide)\b",
                    r"\b(installation|configuration|troubleshooting)\b",
                    r"\b(maintenance|operation|safety)\b"
                ]
            },
            DocumentType.MARKETING_MATERIAL: {
                "keywords": [
                    "marketing", "campaign", "promotion", "advertisement", "brand",
                    "customer", "target", "audience", "message", "value",
                    "proposition", "call to action", "CTA", "conversion"
                ],
                "patterns": [
                    r"\b(marketing|campaign|promotion|advertisement)\b",
                    r"\b(brand|customer|target audience)\b",
                    r"\b(call to action|CTA|conversion)\b"
                ]
            },
            DocumentType.USER_MANUAL: {
                "keywords": [
                    "user", "manual", "guide", "tutorial", "how to", "getting",
                    "started", "quick start", "overview", "features", "interface",
                    "navigation", "help", "support", "FAQ"
                ],
                "patterns": [
                    r"\b(user manual|guide|tutorial|how to)\b",
                    r"\b(getting started|quick start|overview)\b",
                    r"\b(interface|navigation|help|support)\b"
                ]
            },
            DocumentType.REPORT: {
                "keywords": [
                    "report", "analysis", "findings", "recommendations", "summary",
                    "executive", "overview", "data", "statistics", "trends",
                    "performance", "metrics", "KPI", "dashboard"
                ],
                "patterns": [
                    r"\b(report|analysis|findings|recommendations)\b",
                    r"\b(executive summary|overview|data)\b",
                    r"\b(performance|metrics|KPI|dashboard)\b"
                ]
            },
            DocumentType.PROPOSAL: {
                "keywords": [
                    "proposal", "suggestion", "recommendation", "plan", "project",
                    "initiative", "budget", "timeline", "scope", "deliverables",
                    "stakeholder", "approval", "implementation"
                ],
                "patterns": [
                    r"\b(proposal|suggestion|recommendation)\b",
                    r"\b(project|initiative|budget|timeline)\b",
                    r"\b(scope|deliverables|stakeholder)\b"
                ]
            }
        }
        
        # Load template designs
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[DocumentType, List[TemplateDesign]]:
        """Load template designs from configuration files"""
        templates = {}
        templates_dir = Path(__file__).parent / "templates"
        
        for doc_type in DocumentType:
            if doc_type == DocumentType.UNKNOWN:
                continue
                
            template_file = templates_dir / f"{doc_type.value}_templates.yaml"
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                    templates[doc_type] = [
                        TemplateDesign(**template) for template in template_data
                    ]
            else:
                # Create default templates
                templates[doc_type] = self._create_default_templates(doc_type)
        
        return templates
    
    def _create_default_templates(self, doc_type: DocumentType) -> List[TemplateDesign]:
        """Create default templates for document types"""
        default_templates = {
            DocumentType.NOVEL: [
                TemplateDesign(
                    name="Standard Novel",
                    document_type=doc_type,
                    sections=[
                        {"name": "Title Page", "required": True},
                        {"name": "Copyright", "required": True},
                        {"name": "Dedication", "required": False},
                        {"name": "Table of Contents", "required": True},
                        {"name": "Chapters", "required": True, "repeatable": True},
                        {"name": "Epilogue", "required": False},
                        {"name": "About the Author", "required": False}
                    ],
                    formatting={
                        "font": "Times New Roman",
                        "size": 12,
                        "line_spacing": 1.5,
                        "margins": {"top": 1, "bottom": 1, "left": 1.25, "right": 1}
                    },
                    metadata={"pages_per_chapter": 10, "total_chapters": 20}
                )
            ],
            DocumentType.CONTRACT: [
                TemplateDesign(
                    name="Standard Contract",
                    document_type=doc_type,
                    sections=[
                        {"name": "Header", "required": True},
                        {"name": "Parties", "required": True},
                        {"name": "Recitals", "required": True},
                        {"name": "Terms and Conditions", "required": True},
                        {"name": "Obligations", "required": True},
                        {"name": "Termination", "required": True},
                        {"name": "Signatures", "required": True}
                    ],
                    formatting={
                        "font": "Arial",
                        "size": 11,
                        "line_spacing": 1.2,
                        "margins": {"top": 1, "bottom": 1, "left": 1, "right": 1}
                    },
                    metadata={"legal_format": True, "page_numbers": True}
                )
            ],
            DocumentType.DESIGN: [
                TemplateDesign(
                    name="Technical Design Document",
                    document_type=doc_type,
                    sections=[
                        {"name": "Title Page", "required": True},
                        {"name": "Executive Summary", "required": True},
                        {"name": "Design Overview", "required": True},
                        {"name": "Technical Specifications", "required": True},
                        {"name": "Drawings and Diagrams", "required": True},
                        {"name": "Materials List", "required": True},
                        {"name": "Implementation Plan", "required": True}
                    ],
                    formatting={
                        "font": "Arial",
                        "size": 10,
                        "line_spacing": 1.15,
                        "margins": {"top": 0.75, "bottom": 0.75, "left": 0.75, "right": 0.75}
                    },
                    metadata={"include_diagrams": True, "technical_format": True}
                )
            ]
        }
        
        return default_templates.get(doc_type, [])
    
    def classify_document(self, query: str, use_ai: bool = True) -> ClassificationResult:
        """
        Classify document type from a text query.
        
        Args:
            query: Text query describing the document
            use_ai: Whether to use AI for classification (requires OpenAI API)
            
        Returns:
            ClassificationResult with document type and confidence
        """
        query_lower = query.lower()
        
        if use_ai and self.openai_api_key:
            return self._classify_with_ai(query)
        else:
            return self._classify_with_patterns(query_lower)
    
    def _classify_with_ai(self, query: str) -> ClassificationResult:
        """Classify document using AI"""
        try:
            prompt = f"""
            Analyze the following document description and classify it into one of these types:
            - novel (fiction, story, narrative)
            - contract (legal agreement, terms)
            - design (technical, architectural, engineering)
            - business_plan (business strategy, market analysis)
            - academic_paper (research, study, analysis)
            - technical_manual (instructions, procedures, guides)
            - marketing_material (campaigns, promotions, advertisements)
            - user_manual (user guides, tutorials, help)
            - report (analysis, findings, recommendations)
            - proposal (suggestions, recommendations, project plans)
            
            Document description: "{query}"
            
            Respond with a JSON object containing:
            - document_type: the classified type
            - confidence: confidence score (0.0 to 1.0)
            - keywords: list of relevant keywords found
            - reasoning: brief explanation of the classification
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            return ClassificationResult(
                document_type=DocumentType(result_data["document_type"]),
                confidence=result_data["confidence"],
                keywords=result_data["keywords"],
                reasoning=result_data["reasoning"],
                template_suggestions=self._get_template_suggestions(
                    DocumentType(result_data["document_type"])
                )
            )
            
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            return self._classify_with_patterns(query.lower())
    
    def _classify_with_patterns(self, query_lower: str) -> ClassificationResult:
        """Classify document using pattern matching"""
        scores = {}
        matched_keywords = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            keywords = []
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    score += 1
                    keywords.append(keyword)
            
            # Check regex patterns
            for pattern in patterns["patterns"]:
                matches = re.findall(pattern, query_lower)
                score += len(matches) * 0.5
            
            scores[doc_type] = score
            matched_keywords[doc_type] = keywords
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            best_type = DocumentType.UNKNOWN
            confidence = 0.0
            reasoning = "No clear document type indicators found"
        else:
            best_type = max(scores, key=scores.get)
            max_score = scores[best_type]
            total_possible = len(self.document_patterns[best_type]["keywords"]) + len(self.document_patterns[best_type]["patterns"])
            confidence = min(max_score / total_possible, 1.0)
            reasoning = f"Matched {len(matched_keywords[best_type])} keywords and patterns"
        
        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            keywords=matched_keywords.get(best_type, []),
            reasoning=reasoning,
            template_suggestions=self._get_template_suggestions(best_type)
        )
    
    def _get_template_suggestions(self, doc_type: DocumentType) -> List[str]:
        """Get template suggestions for a document type"""
        if doc_type in self.templates:
            return [template.name for template in self.templates[doc_type]]
        return []
    
    def get_templates(self, document_type: DocumentType) -> List[TemplateDesign]:
        """Get all templates for a specific document type"""
        return self.templates.get(document_type, [])
    
    def export_template(self, template: TemplateDesign, format: str = "json") -> str:
        """
        Export a template design in the specified format.
        
        Args:
            template: TemplateDesign object to export
            format: Export format ("json", "yaml", "markdown")
            
        Returns:
            Exported template as string
        """
        if format == "json":
            return json.dumps({
                "name": template.name,
                "document_type": template.document_type.value,
                "sections": template.sections,
                "formatting": template.formatting,
                "metadata": template.metadata
            }, indent=2)
        
        elif format == "yaml":
            return yaml.dump({
                "name": template.name,
                "document_type": template.document_type.value,
                "sections": template.sections,
                "formatting": template.formatting,
                "metadata": template.metadata
            }, default_flow_style=False)
        
        elif format == "markdown":
            md = f"# {template.name}\n\n"
            md += f"**Document Type:** {template.document_type.value}\n\n"
            md += "## Sections\n\n"
            for section in template.sections:
                md += f"- **{section['name']}**"
                if section.get('required'):
                    md += " (Required)"
                if section.get('repeatable'):
                    md += " (Repeatable)"
                md += "\n"
            
            md += "\n## Formatting\n\n"
            for key, value in template.formatting.items():
                md += f"- **{key}:** {value}\n"
            
            md += "\n## Metadata\n\n"
            for key, value in template.metadata.items():
                md += f"- **{key}:** {value}\n"
            
            return md
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_template(self, template: TemplateDesign, filename: Optional[str] = None) -> str:
        """
        Save a template to a file.
        
        Args:
            template: TemplateDesign to save
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if not filename:
            filename = f"{template.document_type.value}_{template.name.lower().replace(' ', '_')}.yaml"
        
        templates_dir = Path(__file__).parent / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        filepath = templates_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.export_template(template, "yaml"))
        
        return str(filepath)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the classifier
    classifier = DocumentClassifierEngine()
    
    # Test queries
    test_queries = [
        "I want to write a science fiction novel about space exploration",
        "Create a service agreement contract for web development",
        "Design a new mobile app interface with user experience focus",
        "Write a business plan for a startup company",
        "Research paper on machine learning algorithms",
        "User manual for a new software application"
    ]
    
    print("AI Document Type Classifier - Test Results")
    print("=" * 50)
    
    for query in test_queries:
        result = classifier.classify_document(query, use_ai=False)
        print(f"\nQuery: {query}")
        print(f"Type: {result.document_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Keywords: {', '.join(result.keywords)}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Templates: {', '.join(result.template_suggestions)}")
        
        # Export a template if available
        if result.template_suggestions:
            templates = classifier.get_templates(result.document_type)
            if templates:
                print(f"\nExported Template ({templates[0].name}):")
                print(classifier.export_template(templates[0], "markdown"))
                print("-" * 30)



























