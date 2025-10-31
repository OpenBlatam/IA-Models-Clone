"""
Document Comparison Service
==========================

Advanced document comparison and difference analysis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import difflib
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ComparisonType(str, Enum):
    """Comparison type."""
    CONTENT = "content"
    STRUCTURE = "structure"
    METADATA = "metadata"
    STYLE = "style"
    COMPREHENSIVE = "comprehensive"


class ChangeType(str, Enum):
    """Change type."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"
    UNCHANGED = "unchanged"


@dataclass
class DocumentChange:
    """Document change."""
    change_id: str
    change_type: ChangeType
    field_name: str
    old_value: Any
    new_value: Any
    position: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    description: str = ""


@dataclass
class ComparisonResult:
    """Comparison result."""
    comparison_id: str
    document1_id: str
    document2_id: str
    comparison_type: ComparisonType
    changes: List[DocumentChange]
    similarity_score: float
    summary: Dict[str, Any]
    created_at: datetime
    processing_time: float


@dataclass
class ContentDiff:
    """Content difference."""
    diff_type: str
    old_text: str
    new_text: str
    position: Dict[str, Any]
    context: str


class DocumentComparisonService:
    """Document comparison service."""
    
    def __init__(self):
        self.comparison_cache: Dict[str, ComparisonResult] = {}
        self.diff_algorithms = {
            "unified": self._unified_diff,
            "context": self._context_diff,
            "html": self._html_diff,
            "word": self._word_diff
        }
    
    async def compare_documents(
        self,
        document1_id: str,
        document2_id: str,
        document1_content: str,
        document2_content: str,
        document1_metadata: Dict[str, Any] = None,
        document2_metadata: Dict[str, Any] = None,
        comparison_type: ComparisonType = ComparisonType.COMPREHENSIVE
    ) -> ComparisonResult:
        """Compare two documents."""
        
        start_time = datetime.now()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(
                document1_content, document2_content, comparison_type
            )
            
            # Check cache
            if cache_key in self.comparison_cache:
                result = self.comparison_cache[cache_key]
                result.comparison_id = str(uuid4())  # New ID for this comparison
                return result
            
            changes = []
            
            # Perform different types of comparisons
            if comparison_type in [ComparisonType.CONTENT, ComparisonType.COMPREHENSIVE]:
                content_changes = await self._compare_content(
                    document1_content, document2_content
                )
                changes.extend(content_changes)
            
            if comparison_type in [ComparisonType.STRUCTURE, ComparisonType.COMPREHENSIVE]:
                structure_changes = await self._compare_structure(
                    document1_content, document2_content
                )
                changes.extend(structure_changes)
            
            if comparison_type in [ComparisonType.METADATA, ComparisonType.COMPREHENSIVE]:
                if document1_metadata and document2_metadata:
                    metadata_changes = await self._compare_metadata(
                        document1_metadata, document2_metadata
                    )
                    changes.extend(metadata_changes)
            
            if comparison_type in [ComparisonType.STYLE, ComparisonType.COMPREHENSIVE]:
                style_changes = await self._compare_style(
                    document1_content, document2_content
                )
                changes.extend(style_changes)
            
            # Calculate similarity score
            similarity_score = await self._calculate_similarity(
                document1_content, document2_content
            )
            
            # Generate summary
            summary = await self._generate_comparison_summary(changes, similarity_score)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ComparisonResult(
                comparison_id=str(uuid4()),
                document1_id=document1_id,
                document2_id=document2_id,
                comparison_type=comparison_type,
                changes=changes,
                similarity_score=similarity_score,
                summary=summary,
                created_at=datetime.now(),
                processing_time=processing_time
            )
            
            # Cache result
            self.comparison_cache[cache_key] = result
            
            logger.info(f"Document comparison completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing documents: {str(e)}")
            raise
    
    async def _compare_content(
        self,
        content1: str,
        content2: str
    ) -> List[DocumentChange]:
        """Compare document content."""
        
        changes = []
        
        try:
            # Calculate content hash
            hash1 = hashlib.md5(content1.encode()).hexdigest()
            hash2 = hashlib.md5(content2.encode()).hexdigest()
            
            if hash1 == hash2:
                return changes  # No changes
            
            # Generate unified diff
            diff = list(difflib.unified_diff(
                content1.splitlines(keepends=True),
                content2.splitlines(keepends=True),
                fromfile="Document 1",
                tofile="Document 2",
                lineterm=""
            ))
            
            # Parse diff into changes
            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    changes.append(DocumentChange(
                        change_id=str(uuid4()),
                        change_type=ChangeType.ADDED,
                        field_name="content",
                        old_value="",
                        new_value=line[1:].rstrip(),
                        description="Content added"
                    ))
                elif line.startswith('-') and not line.startswith('---'):
                    changes.append(DocumentChange(
                        change_id=str(uuid4()),
                        change_type=ChangeType.REMOVED,
                        field_name="content",
                        old_value=line[1:].rstrip(),
                        new_value="",
                        description="Content removed"
                    ))
            
            # Calculate content statistics
            lines1 = content1.splitlines()
            lines2 = content2.splitlines()
            words1 = content1.split()
            words2 = content2.split()
            
            changes.append(DocumentChange(
                change_id=str(uuid4()),
                change_type=ChangeType.MODIFIED,
                field_name="content_stats",
                old_value={
                    "lines": len(lines1),
                    "words": len(words1),
                    "characters": len(content1)
                },
                new_value={
                    "lines": len(lines2),
                    "words": len(words2),
                    "characters": len(content2)
                },
                description="Content statistics changed"
            ))
            
        except Exception as e:
            logger.error(f"Error comparing content: {str(e)}")
        
        return changes
    
    async def _compare_structure(
        self,
        content1: str,
        content2: str
    ) -> List[DocumentChange]:
        """Compare document structure."""
        
        changes = []
        
        try:
            # Analyze structure elements
            structure1 = self._analyze_structure(content1)
            structure2 = self._analyze_structure(content2)
            
            # Compare structure elements
            for element in ["headings", "paragraphs", "lists", "links", "images"]:
                if structure1[element] != structure2[element]:
                    changes.append(DocumentChange(
                        change_id=str(uuid4()),
                        change_type=ChangeType.MODIFIED,
                        field_name=f"structure.{element}",
                        old_value=structure1[element],
                        new_value=structure2[element],
                        description=f"{element.title()} count changed"
                    ))
            
            # Compare heading hierarchy
            if structure1["heading_hierarchy"] != structure2["heading_hierarchy"]:
                changes.append(DocumentChange(
                    change_id=str(uuid4()),
                    change_type=ChangeType.MODIFIED,
                    field_name="structure.heading_hierarchy",
                    old_value=structure1["heading_hierarchy"],
                    new_value=structure2["heading_hierarchy"],
                    description="Heading hierarchy changed"
                ))
            
        except Exception as e:
            logger.error(f"Error comparing structure: {str(e)}")
        
        return changes
    
    async def _compare_metadata(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> List[DocumentChange]:
        """Compare document metadata."""
        
        changes = []
        
        try:
            # Get all keys from both metadata
            all_keys = set(metadata1.keys()) | set(metadata2.keys())
            
            for key in all_keys:
                value1 = metadata1.get(key)
                value2 = metadata2.get(key)
                
                if value1 != value2:
                    if key not in metadata1:
                        changes.append(DocumentChange(
                            change_id=str(uuid4()),
                            change_type=ChangeType.ADDED,
                            field_name=f"metadata.{key}",
                            old_value=None,
                            new_value=value2,
                            description=f"Metadata field '{key}' added"
                        ))
                    elif key not in metadata2:
                        changes.append(DocumentChange(
                            change_id=str(uuid4()),
                            change_type=ChangeType.REMOVED,
                            field_name=f"metadata.{key}",
                            old_value=value1,
                            new_value=None,
                            description=f"Metadata field '{key}' removed"
                        ))
                    else:
                        changes.append(DocumentChange(
                            change_id=str(uuid4()),
                            change_type=ChangeType.MODIFIED,
                            field_name=f"metadata.{key}",
                            old_value=value1,
                            new_value=value2,
                            description=f"Metadata field '{key}' modified"
                        ))
            
        except Exception as e:
            logger.error(f"Error comparing metadata: {str(e)}")
        
        return changes
    
    async def _compare_style(
        self,
        content1: str,
        content2: str
    ) -> List[DocumentChange]:
        """Compare document style."""
        
        changes = []
        
        try:
            # Analyze style metrics
            style1 = self._analyze_style(content1)
            style2 = self._analyze_style(content2)
            
            # Compare style metrics
            for metric in ["avg_sentence_length", "avg_word_length", "passive_voice_ratio"]:
                if abs(style1[metric] - style2[metric]) > 0.1:
                    changes.append(DocumentChange(
                        change_id=str(uuid4()),
                        change_type=ChangeType.MODIFIED,
                        field_name=f"style.{metric}",
                        old_value=style1[metric],
                        new_value=style2[metric],
                        description=f"Style metric '{metric}' changed"
                    ))
            
            # Compare writing patterns
            if style1["writing_patterns"] != style2["writing_patterns"]:
                changes.append(DocumentChange(
                    change_id=str(uuid4()),
                    change_type=ChangeType.MODIFIED,
                    field_name="style.writing_patterns",
                    old_value=style1["writing_patterns"],
                    new_value=style2["writing_patterns"],
                    description="Writing patterns changed"
                ))
            
        except Exception as e:
            logger.error(f"Error comparing style: {str(e)}")
        
        return changes
    
    async def _calculate_similarity(
        self,
        content1: str,
        content2: str
    ) -> float:
        """Calculate similarity score between documents."""
        
        try:
            # Use SequenceMatcher for similarity
            matcher = difflib.SequenceMatcher(None, content1, content2)
            similarity = matcher.ratio()
            
            # Adjust for length difference
            length_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
            adjusted_similarity = similarity * length_ratio
            
            return adjusted_similarity * 100
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def _generate_comparison_summary(
        self,
        changes: List[DocumentChange],
        similarity_score: float
    ) -> Dict[str, Any]:
        """Generate comparison summary."""
        
        # Count changes by type
        change_counts = {}
        for change in changes:
            change_counts[change.change_type] = change_counts.get(change.change_type, 0) + 1
        
        # Count changes by field
        field_counts = {}
        for change in changes:
            field_counts[change.field_name] = field_counts.get(change.field_name, 0) + 1
        
        # Determine change severity
        total_changes = len(changes)
        if total_changes == 0:
            severity = "none"
        elif total_changes < 5:
            severity = "minor"
        elif total_changes < 15:
            severity = "moderate"
        elif total_changes < 30:
            severity = "major"
        else:
            severity = "extensive"
        
        return {
            "total_changes": total_changes,
            "change_counts": change_counts,
            "field_counts": field_counts,
            "similarity_score": similarity_score,
            "change_severity": severity,
            "most_changed_field": max(field_counts.items(), key=lambda x: x[1])[0] if field_counts else None,
            "change_types": list(change_counts.keys())
        }
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure."""
        
        import re
        
        return {
            "headings": len(re.findall(r'^#+\s', content, re.MULTILINE)),
            "paragraphs": len(content.split('\n\n')),
            "lists": len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE)),
            "links": len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)),
            "images": len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)),
            "heading_hierarchy": self._get_heading_hierarchy(content)
        }
    
    def _analyze_style(self, content: str) -> Dict[str, Any]:
        """Analyze document style."""
        
        import re
        
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        return {
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "passive_voice_ratio": len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', content, re.IGNORECASE)) / len(sentences) if sentences else 0,
            "writing_patterns": {
                "contractions": len(re.findall(r"\b\w+'\w+\b", content)),
                "exclamations": len(re.findall(r'!', content)),
                "questions": len(re.findall(r'\?', content))
            }
        }
    
    def _get_heading_hierarchy(self, content: str) -> List[str]:
        """Get heading hierarchy."""
        
        import re
        
        headings = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
        return [f"{len(level)}: {text.strip()}" for level, text in headings]
    
    def _generate_cache_key(
        self,
        content1: str,
        content2: str,
        comparison_type: ComparisonType
    ) -> str:
        """Generate cache key for comparison."""
        
        hash1 = hashlib.md5(content1.encode()).hexdigest()
        hash2 = hashlib.md5(content2.encode()).hexdigest()
        return f"{hash1}_{hash2}_{comparison_type.value}"
    
    async def generate_diff_report(
        self,
        comparison_result: ComparisonResult,
        format: str = "html"
    ) -> str:
        """Generate diff report in specified format."""
        
        try:
            if format == "html":
                return await self._generate_html_diff(comparison_result)
            elif format == "json":
                return await self._generate_json_diff(comparison_result)
            elif format == "text":
                return await self._generate_text_diff(comparison_result)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error generating diff report: {str(e)}")
            raise
    
    async def _generate_html_diff(self, comparison_result: ComparisonResult) -> str:
        """Generate HTML diff report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .changes {{ margin: 20px 0; }}
                .change {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .added {{ border-left-color: #4CAF50; background: #f1f8e9; }}
                .removed {{ border-left-color: #f44336; background: #ffebee; }}
                .modified {{ border-left-color: #ff9800; background: #fff3e0; }}
                .score {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Comparison Report</h1>
                <p><strong>Comparison ID:</strong> {comparison_result.comparison_id}</p>
                <p><strong>Created:</strong> {comparison_result.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Processing Time:</strong> {comparison_result.processing_time:.2f}s</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p class="score">Similarity Score: {comparison_result.similarity_score:.1f}%</p>
                <p><strong>Total Changes:</strong> {comparison_result.summary['total_changes']}</p>
                <p><strong>Change Severity:</strong> {comparison_result.summary['change_severity'].title()}</p>
            </div>
            
            <div class="changes">
                <h2>Changes</h2>
        """
        
        for change in comparison_result.changes:
            change_class = change.change_type.value
            html += f"""
                <div class="change {change_class}">
                    <h3>{change.change_type.value.title()}: {change.field_name}</h3>
                    <p><strong>Description:</strong> {change.description}</p>
                    {f'<p><strong>Old Value:</strong> {change.old_value}</p>' if change.old_value else ''}
                    {f'<p><strong>New Value:</strong> {change.new_value}</p>' if change.new_value else ''}
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def _generate_json_diff(self, comparison_result: ComparisonResult) -> str:
        """Generate JSON diff report."""
        
        data = {
            "comparison_id": comparison_result.comparison_id,
            "document1_id": comparison_result.document1_id,
            "document2_id": comparison_result.document2_id,
            "comparison_type": comparison_result.comparison_type.value,
            "similarity_score": comparison_result.similarity_score,
            "summary": comparison_result.summary,
            "changes": [
                {
                    "change_id": change.change_id,
                    "change_type": change.change_type.value,
                    "field_name": change.field_name,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "description": change.description,
                    "confidence": change.confidence
                }
                for change in comparison_result.changes
            ],
            "created_at": comparison_result.created_at.isoformat(),
            "processing_time": comparison_result.processing_time
        }
        
        return json.dumps(data, indent=2, default=str)
    
    async def _generate_text_diff(self, comparison_result: ComparisonResult) -> str:
        """Generate text diff report."""
        
        lines = [
            "DOCUMENT COMPARISON REPORT",
            "=" * 50,
            f"Comparison ID: {comparison_result.comparison_id}",
            f"Created: {comparison_result.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Processing Time: {comparison_result.processing_time:.2f}s",
            "",
            "SUMMARY",
            "-" * 20,
            f"Similarity Score: {comparison_result.similarity_score:.1f}%",
            f"Total Changes: {comparison_result.summary['total_changes']}",
            f"Change Severity: {comparison_result.summary['change_severity'].title()}",
            "",
            "CHANGES",
            "-" * 20
        ]
        
        for change in comparison_result.changes:
            lines.extend([
                f"{change.change_type.value.upper()}: {change.field_name}",
                f"  Description: {change.description}",
                f"  Old Value: {change.old_value}" if change.old_value else "",
                f"  New Value: {change.new_value}" if change.new_value else "",
                ""
            ])
        
        return "\n".join(lines)
    
    async def get_comparison_history(
        self,
        document_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get comparison history for a document."""
        
        # This would typically query a database
        # For now, return mock data
        
        return [
            {
                "comparison_id": str(uuid4()),
                "compared_with": "document_123",
                "similarity_score": 85.5,
                "total_changes": 12,
                "created_at": datetime.now().isoformat()
            }
            for _ in range(min(limit, 5))
        ]



























