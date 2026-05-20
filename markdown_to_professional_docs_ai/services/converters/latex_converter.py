"""LaTeX Converter - Convert Markdown to LaTeX format"""
from typing import Dict, Any, Optional
import re

from .base_converter import BaseConverter


class LaTeXConverter(BaseConverter):
    """Convert Markdown to LaTeX format"""
    
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_path: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convert to LaTeX format"""
        latex_parts = []
        
        # Document class and packages
        latex_parts.append("\\documentclass[11pt,a4paper]{article}\n")
        latex_parts.append("\\usepackage[utf8]{inputenc}\n")
        latex_parts.append("\\usepackage[T1]{fontenc}\n")
        latex_parts.append("\\usepackage{amsmath}\n")
        latex_parts.append("\\usepackage{graphicx}\n")
        latex_parts.append("\\usepackage{booktabs}\n")
        latex_parts.append("\\usepackage{hyperref}\n")
        latex_parts.append("\\usepackage{xcolor}\n")
        latex_parts.append("\\usepackage{geometry}\n")
        latex_parts.append("\\geometry{margin=1in}\n\n")
        latex_parts.append("\\begin{document}\n\n")
        
        # Title
        if parsed_content.get("title"):
            latex_parts.append(f"\\title{{{self._escape_latex(parsed_content['title'])}}}\n")
            latex_parts.append("\\maketitle\n\n")
        
        # Headings
        for heading in parsed_content.get("headings", []):
            level = heading["level"]
            text = self._escape_latex(heading["text"])
            
            if level == 1:
                latex_parts.append(f"\\section{{{text}}}\n\n")
            elif level == 2:
                latex_parts.append(f"\\subsection{{{text}}}\n\n")
            elif level == 3:
                latex_parts.append(f"\\subsubsection{{{text}}}\n\n")
            else:
                latex_parts.append(f"\\paragraph{{{text}}}\n\n")
        
        # Tables
        if include_tables:
            for table in parsed_content.get("tables", []):
                latex_parts.append("\\begin{table}[h]\n")
                latex_parts.append("\\centering\n")
                latex_parts.append("\\begin{tabular}{" + "l" * len(table.get("headers", [])) + "}\n")
                latex_parts.append("\\toprule\n")
                
                # Headers
                headers = table.get("headers", [])
                if headers:
                    header_row = " & ".join(self._escape_latex(h) for h in headers)
                    latex_parts.append(f"{header_row} \\\\\n")
                    latex_parts.append("\\midrule\n")
                
                # Rows
                for row in table.get("rows", []):
                    row_text = " & ".join(self._escape_latex(str(cell)) for cell in row)
                    latex_parts.append(f"{row_text} \\\\\n")
                
                latex_parts.append("\\bottomrule\n")
                latex_parts.append("\\end{tabular}\n")
                latex_parts.append("\\end{table}\n\n")
        
        # Paragraphs
        for para in parsed_content.get("paragraphs", []):
            # Convert markdown links to LaTeX
            para = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\\href{\2}{\1}', para)
            # Convert bold
            para = re.sub(r'\*\*([^\*]+)\*\*', r'\\textbf{\1}', para)
            # Convert italic
            para = re.sub(r'\*([^\*]+)\*', r'\\textit{\1}', para)
            
            latex_parts.append(f"{self._escape_latex(para)}\n\n")
        
        # Lists
        for list_data in parsed_content.get("lists", []):
            if list_data["type"] == "unordered":
                latex_parts.append("\\begin{itemize}\n")
                for item in list_data["items"]:
                    latex_parts.append(f"\\item {self._escape_latex(item)}\n")
                latex_parts.append("\\end{itemize}\n\n")
            else:
                latex_parts.append("\\begin{enumerate}\n")
                for item in list_data["items"]:
                    latex_parts.append(f"\\item {self._escape_latex(item)}\n")
                latex_parts.append("\\end{enumerate}\n\n")
        
        # Math formulas (already in LaTeX format)
        for formula in parsed_content.get("math_formulas", []):
            if formula["type"] == "inline":
                latex_parts.append(f"${formula['formula']}$\n\n")
            else:
                latex_parts.append(f"$${formula['formula']}$$\n\n")
        
        latex_parts.append("\\end{document}\n")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(latex_parts))
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        replacements = {
            '\\': '\\textbackslash{}',
            '{': '\\{',
            '}': '\\}',
            '$': '\\$',
            '&': '\\&',
            '%': '\\%',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '~': '\\textasciitilde{}',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text

