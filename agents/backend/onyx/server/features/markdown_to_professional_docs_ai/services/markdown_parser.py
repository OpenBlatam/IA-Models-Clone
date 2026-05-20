"""Markdown Parser - Parse and structure Markdown content"""
import re
from typing import Dict, List, Any, Optional
from markdown import Markdown
from markdown.extensions import tables, fenced_code, codehilite, toc
import yaml


class MarkdownParser:
    """Parse Markdown content and extract structured data"""
    
    def __init__(self):
        self.md = Markdown(
            extensions=[
                'tables',
                'fenced_code',
                'codehilite',
                'toc',
                'nl2br',
                'sane_lists',
                'attr_list',
                'def_list',
                'abbr',
                'footnotes'
            ]
        )
    
    def parse(self, markdown_content: str) -> Dict[str, Any]:
        """
        Parse Markdown content and return structured data
        
        Args:
            markdown_content: Raw Markdown content
            
        Returns:
            Dictionary with parsed content structure
        """
        # Convert to HTML first
        html_content = self.md.convert(markdown_content)
        
        # Extract components
        parsed = {
            "raw": markdown_content,
            "html": html_content,
            "title": self._extract_title(markdown_content),
            "headings": self._extract_headings(markdown_content),
            "tables": self._extract_tables(markdown_content),
            "code_blocks": self._extract_code_blocks(markdown_content),
            "images": self._extract_images(markdown_content),
            "links": self._extract_links(markdown_content),
            "lists": self._extract_lists(markdown_content),
            "paragraphs": self._extract_paragraphs(markdown_content),
            "blockquotes": self._extract_blockquotes(markdown_content),
            "horizontal_rules": self._extract_horizontal_rules(markdown_content),
            "emphasis": self._extract_emphasis(markdown_content),
            "metadata": self._extract_metadata(markdown_content),
            "statistics": self._calculate_statistics(markdown_content),
            "mermaid_diagrams": self._extract_mermaid_diagrams(markdown_content),
            "math_formulas": self._extract_math_formulas(markdown_content)
        }
        
        return parsed
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract main title (first H1)"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else None
    
    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """Extract all headings with levels"""
        headings = []
        pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                "level": level,
                "text": text,
                "id": self._slugify(text)
            })
        
        return headings
    
    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract all tables"""
        tables = []
        # Match markdown tables
        pattern = r'\|(.+)\|\n\|[-\s|]+\|\n((?:\|.+\|\n?)+)'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            header_row = match.group(1)
            data_rows = match.group(2)
            
            headers = [h.strip() for h in header_row.split('|') if h.strip()]
            rows = []
            
            for row in data_rows.strip().split('\n'):
                if row.strip():
                    cells = [c.strip() for c in row.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        rows.append(cells)
            
            if headers and rows:
                tables.append({
                    "headers": headers,
                    "rows": rows
                })
        
        return tables
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks"""
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append({
                "language": language,
                "code": code
            })
        
        return code_blocks
    
    def _extract_images(self, content: str) -> List[Dict[str, Any]]:
        """Extract image references"""
        images = []
        pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        
        for match in re.finditer(pattern, content):
            alt_text = match.group(1)
            url = match.group(2)
            images.append({
                "alt": alt_text,
                "url": url
            })
        
        return images
    
    def _extract_links(self, content: str) -> List[Dict[str, Any]]:
        """Extract links"""
        links = []
        pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        
        for match in re.finditer(pattern, content):
            text = match.group(1)
            url = match.group(2)
            links.append({
                "text": text,
                "url": url
            })
        
        return links
    
    def _extract_lists(self, content: str) -> List[Dict[str, Any]]:
        """Extract lists (ordered and unordered)"""
        lists = []
        lines = content.split('\n')
        current_list = None
        
        for line in lines:
            # Unordered list
            if re.match(r'^[\*\-\+]\s+', line):
                if not current_list or current_list['type'] != 'unordered':
                    if current_list:
                        lists.append(current_list)
                    current_list = {
                        "type": "unordered",
                        "items": []
                    }
                item = re.sub(r'^[\*\-\+]\s+', '', line).strip()
                current_list['items'].append(item)
            # Ordered list
            elif re.match(r'^\d+\.\s+', line):
                if not current_list or current_list['type'] != 'ordered':
                    if current_list:
                        lists.append(current_list)
                    current_list = {
                        "type": "ordered",
                        "items": []
                    }
                item = re.sub(r'^\d+\.\s+', '', line).strip()
                current_list['items'].append(item)
            else:
                if current_list:
                    lists.append(current_list)
                    current_list = None
        
        if current_list:
            lists.append(current_list)
        
        return lists
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """Extract plain text paragraphs"""
        # Remove code blocks, headers, lists, etc.
        cleaned = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'^#{1,6}\s+.+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^[\*\-\+]\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\d+\.\s+', '', cleaned, flags=re.MULTILINE)
        
        paragraphs = []
        for para in cleaned.split('\n\n'):
            para = para.strip()
            if para and len(para) > 10:  # Filter out very short lines
                paragraphs.append(para)
        
        return paragraphs
    
    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract frontmatter metadata if present"""
        metadata = {}
        
        # Check for YAML frontmatter
        if content.startswith('---'):
            match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if match:
                try:
                    yaml_content = match.group(1)
                    metadata = yaml.safe_load(yaml_content) or {}
                except:
                    # Fallback to simple parsing
                    yaml_content = match.group(1)
                    for line in yaml_content.split('\n'):
                        if ':' in line and not line.strip().startswith('#'):
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip().strip('"\'')
        
        return metadata
    
    def _extract_blockquotes(self, content: str) -> List[str]:
        """Extract blockquotes"""
        blockquotes = []
        pattern = r'^>\s+(.+)$'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            blockquotes.append(match.group(1).strip())
        
        return blockquotes
    
    def _extract_horizontal_rules(self, content: str) -> List[int]:
        """Extract horizontal rule positions (line numbers)"""
        rules = []
        lines = content.split('\n')
        
        for idx, line in enumerate(lines):
            if re.match(r'^[-*_]{3,}$', line.strip()):
                rules.append(idx + 1)
        
        return rules
    
    def _extract_emphasis(self, content: str) -> Dict[str, List[str]]:
        """Extract emphasis (bold, italic, strikethrough)"""
        emphasis = {
            "bold": [],
            "italic": [],
            "strikethrough": []
        }
        
        # Bold (**text** or __text__)
        for match in re.finditer(r'\*\*(.+?)\*\*|__(.+?)__', content):
            text = match.group(1) or match.group(2)
            if text not in emphasis["bold"]:
                emphasis["bold"].append(text)
        
        # Italic (*text* or _text_)
        for match in re.finditer(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', content):
            text = match.group(1) or match.group(2)
            if text and text not in emphasis["italic"]:
                emphasis["italic"].append(text)
        
        # Strikethrough (~~text~~)
        for match in re.finditer(r'~~(.+?)~~', content):
            text = match.group(1)
            if text not in emphasis["strikethrough"]:
                emphasis["strikethrough"].append(text)
        
        return emphasis
    
    def _calculate_statistics(self, content: str) -> Dict[str, Any]:
        """Calculate document statistics"""
        lines = content.split('\n')
        words = content.split()
        
        return {
            "total_lines": len(lines),
            "total_words": len(words),
            "total_characters": len(content),
            "total_headings": len(self._extract_headings(content)),
            "total_tables": len(self._extract_tables(content)),
            "total_images": len(self._extract_images(content)),
            "total_links": len(self._extract_links(content)),
            "total_code_blocks": len(self._extract_code_blocks(content)),
            "total_lists": len(self._extract_lists(content)),
            "total_mermaid_diagrams": len(self._extract_mermaid_diagrams(content)),
            "total_math_formulas": len(self._extract_math_formulas(content))
        }
    
    def _extract_math_formulas(self, content: str) -> List[Dict[str, Any]]:
        """Extract mathematical formulas"""
        math_formulas = []
        
        # Inline math: $...$
        inline_pattern = r'\$([^$]+)\$'
        for match in re.finditer(inline_pattern, content):
            math_formulas.append({
                "type": "inline",
                "formula": match.group(1),
                "position": match.start()
            })
        
        # Block math: $$...$$
        block_pattern = r'\$\$([^$]+)\$\$'
        for match in re.finditer(block_pattern, content):
            math_formulas.append({
                "type": "block",
                "formula": match.group(1),
                "position": match.start()
            })
        
        return math_formulas
        }
    
    def _extract_mermaid_diagrams(self, content: str) -> List[Dict[str, Any]]:
        """Extract Mermaid diagram code blocks"""
        mermaid_diagrams = []
        pattern = r'```mermaid\n(.*?)```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            code = match.group(1).strip()
            diagram_type = self._detect_mermaid_type(code)
            mermaid_diagrams.append({
                "code": code,
                "type": diagram_type
            })
        
        return mermaid_diagrams
    
    def _detect_mermaid_type(self, code: str) -> str:
        """Detect Mermaid diagram type"""
        code_lower = code.lower().strip()
        
        if code_lower.startswith('graph') or code_lower.startswith('flowchart'):
            return "flowchart"
        elif code_lower.startswith('sequence'):
            return "sequence"
        elif code_lower.startswith('class'):
            return "class"
        elif code_lower.startswith('state'):
            return "state"
        elif code_lower.startswith('er'):
            return "er"
        elif code_lower.startswith('gantt'):
            return "gantt"
        elif code_lower.startswith('pie'):
            return "pie"
        else:
            return "unknown"
    
    def _slugify(self, text: str) -> str:
        """Convert text to URL-friendly slug"""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')

