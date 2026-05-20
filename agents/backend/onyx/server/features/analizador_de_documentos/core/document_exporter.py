"""
Document Exporter - Exportación de Resultados
=============================================

Exportar resultados de análisis en múltiples formatos.
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentAnalysisExporter:
    """Exportador de resultados de análisis."""
    
    def __init__(self):
        """Inicializar exportador."""
        self.supported_formats = ['json', 'csv', 'xml', 'html', 'markdown', 'txt']
    
    def export(
        self,
        results: Any,  # DocumentAnalysisResult o lista
        output_path: str,
        format: str = 'json',
        include_raw: bool = False
    ) -> str:
        """
        Exportar resultados.
        
        Args:
            results: Resultado(s) de análisis
            output_path: Ruta de salida
            format: Formato de exportación
            include_raw: Incluir contenido raw
        
        Returns:
            Ruta del archivo exportado
        """
        format = format.lower()
        
        if format not in self.supported_formats:
            raise ValueError(f"Formato no soportado: {format}. Formatos: {self.supported_formats}")
        
        if format == 'json':
            return self._export_json(results, output_path, include_raw)
        elif format == 'csv':
            return self._export_csv(results, output_path)
        elif format == 'xml':
            return self._export_xml(results, output_path)
        elif format == 'html':
            return self._export_html(results, output_path)
        elif format == 'markdown':
            return self._export_markdown(results, output_path)
        elif format == 'txt':
            return self._export_txt(results, output_path)
    
    def _export_json(
        self,
        results: Any,
        output_path: str,
        include_raw: bool
    ) -> str:
        """Exportar a JSON."""
        if hasattr(results, '__dict__'):
            # Single result
            data = self._result_to_dict(results, include_raw)
        elif isinstance(results, list):
            # Multiple results
            data = [self._result_to_dict(r, include_raw) for r in results]
        else:
            data = results
        
        output_file = output_path if output_path.endswith('.json') else f"{output_path}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return output_file
    
    def _export_csv(self, results: Any, output_path: str) -> str:
        """Exportar a CSV."""
        output_file = output_path if output_path.endswith('.csv') else f"{output_path}.csv"
        
        if isinstance(results, list):
            if not results:
                return output_file
            
            # Determinar campos
            fields = ['document_id', 'document_type', 'confidence', 'processing_time']
            if hasattr(results[0], 'classification'):
                fields.append('classification_top')
            if hasattr(results[0], 'sentiment'):
                fields.append('sentiment_top')
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'document_id': getattr(result, 'document_id', ''),
                        'document_type': getattr(result, 'document_type', ''),
                        'confidence': getattr(result, 'confidence', 0),
                        'processing_time': getattr(result, 'processing_time', 0)
                    }
                    
                    if hasattr(result, 'classification') and result.classification:
                        row['classification_top'] = max(
                            result.classification.items(),
                            key=lambda x: x[1]
                        )[0]
                    
                    if hasattr(result, 'sentiment') and result.sentiment:
                        row['sentiment_top'] = max(
                            result.sentiment.items(),
                            key=lambda x: x[1]
                        )[0]
                    
                    writer.writerow(row)
        else:
            # Single result
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Field', 'Value'])
                writer.writerow(['document_id', getattr(results, 'document_id', '')])
                writer.writerow(['confidence', getattr(results, 'confidence', 0)])
        
        return output_file
    
    def _export_xml(self, results: Any, output_path: str) -> str:
        """Exportar a XML."""
        output_file = output_path if output_path.endswith('.xml') else f"{output_path}.xml"
        
        root = ET.Element('document_analysis')
        root.set('timestamp', datetime.now().isoformat())
        
        if isinstance(results, list):
            for result in results:
                self._add_result_to_xml(root, result)
        else:
            self._add_result_to_xml(root, results)
        
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        return output_file
    
    def _add_result_to_xml(self, parent: ET.Element, result: Any):
        """Agregar resultado a XML."""
        result_elem = ET.SubElement(parent, 'result')
        
        if hasattr(result, 'document_id'):
            result_elem.set('id', str(result.document_id))
            result_elem.set('type', getattr(result, 'document_type', ''))
            result_elem.set('confidence', str(getattr(result, 'confidence', 0)))
            
            if hasattr(result, 'summary') and result.summary:
                summary_elem = ET.SubElement(result_elem, 'summary')
                summary_elem.text = result.summary
            
            if hasattr(result, 'classification') and result.classification:
                class_elem = ET.SubElement(result_elem, 'classification')
                for label, score in result.classification.items():
                    item = ET.SubElement(class_elem, 'item')
                    item.set('label', label)
                    item.set('score', str(score))
    
    def _export_html(self, results: Any, output_path: str) -> str:
        """Exportar a HTML."""
        output_file = output_path if output_path.endswith('.html') else f"{output_path}.html"
        
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document Analysis Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .result { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
        .summary { background: #f5f5f5; padding: 10px; }
        .classification { margin-top: 10px; }
        .item { margin: 5px 0; }
    </style>
</head>
<body>
    <h1>Document Analysis Results</h1>
    <p>Generated: {timestamp}</p>
"""
        
        if isinstance(results, list):
            for result in results:
                html += self._result_to_html(result)
        else:
            html += self._result_to_html(results)
        
        html += """
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html.format(timestamp=datetime.now().isoformat()))
        
        return output_file
    
    def _result_to_html(self, result: Any) -> str:
        """Convertir resultado a HTML."""
        html = f"""
    <div class="result">
        <h2>Document: {getattr(result, 'document_id', 'Unknown')}</h2>
        <p><strong>Type:</strong> {getattr(result, 'document_type', 'Unknown')}</p>
        <p><strong>Confidence:</strong> {getattr(result, 'confidence', 0):.2%}</p>
"""
        
        if hasattr(result, 'summary') and result.summary:
            html += f"""
        <div class="summary">
            <h3>Summary</h3>
            <p>{result.summary}</p>
        </div>
"""
        
        if hasattr(result, 'classification') and result.classification:
            html += """
        <div class="classification">
            <h3>Classification</h3>
"""
            for label, score in result.classification.items():
                html += f"""
            <div class="item">
                <strong>{label}:</strong> {score:.2%}
            </div>
"""
            html += """
        </div>
"""
        
        html += """
    </div>
"""
        return html
    
    def _export_markdown(self, results: Any, output_path: str) -> str:
        """Exportar a Markdown."""
        output_file = output_path if output_path.endswith('.md') else f"{output_path}.md"
        
        md = f"""# Document Analysis Results

Generated: {datetime.now().isoformat()}

"""
        
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                md += self._result_to_markdown(result, i)
        else:
            md += self._result_to_markdown(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return output_file
    
    def _result_to_markdown(self, result: Any, index: Optional[int] = None) -> str:
        """Convertir resultado a Markdown."""
        title = f"## Document {index}" if index else "## Document Analysis"
        md = f"""
{title}

**ID:** {getattr(result, 'document_id', 'Unknown')}  
**Type:** {getattr(result, 'document_type', 'Unknown')}  
**Confidence:** {getattr(result, 'confidence', 0):.2%}  
**Processing Time:** {getattr(result, 'processing_time', 0):.2f}s

"""
        
        if hasattr(result, 'summary') and result.summary:
            md += f"""
### Summary

{result.summary}

"""
        
        if hasattr(result, 'classification') and result.classification:
            md += """
### Classification

"""
            for label, score in result.classification.items():
                md += f"- **{label}**: {score:.2%}\n"
            md += "\n"
        
        if hasattr(result, 'keywords') and result.keywords:
            md += f"""
### Keywords

{', '.join(result.keywords[:10])}

"""
        
        return md
    
    def _export_txt(self, results: Any, output_path: str) -> str:
        """Exportar a texto plano."""
        output_file = output_path if output_path.endswith('.txt') else f"{output_path}.txt"
        
        txt = f"""Document Analysis Results
Generated: {datetime.now().isoformat()}

"""
        
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                txt += f"\n{'='*60}\n"
                txt += f"Document {i}\n"
                txt += f"{'='*60}\n"
                txt += self._result_to_txt(result)
        else:
            txt += self._result_to_txt(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(txt)
        
        return output_file
    
    def _result_to_txt(self, result: Any) -> str:
        """Convertir resultado a texto."""
        txt = f"""
Document ID: {getattr(result, 'document_id', 'Unknown')}
Type: {getattr(result, 'document_type', 'Unknown')}
Confidence: {getattr(result, 'confidence', 0):.2%}
Processing Time: {getattr(result, 'processing_time', 0):.2f}s

"""
        
        if hasattr(result, 'summary') and result.summary:
            txt += f"Summary:\n{result.summary}\n\n"
        
        if hasattr(result, 'classification') and result.classification:
            txt += "Classification:\n"
            for label, score in result.classification.items():
                txt += f"  {label}: {score:.2%}\n"
            txt += "\n"
        
        return txt
    
    def _result_to_dict(self, result: Any, include_raw: bool) -> Dict[str, Any]:
        """Convertir resultado a diccionario."""
        data = {
            'document_id': getattr(result, 'document_id', None),
            'document_type': getattr(result, 'document_type', None),
            'confidence': getattr(result, 'confidence', 0),
            'processing_time': getattr(result, 'processing_time', 0),
            'timestamp': getattr(result, 'timestamp', None)
        }
        
        if include_raw or hasattr(result, 'content'):
            data['content'] = getattr(result, 'content', None)
        
        if hasattr(result, 'summary'):
            data['summary'] = result.summary
        
        if hasattr(result, 'classification'):
            data['classification'] = result.classification
        
        if hasattr(result, 'keywords'):
            data['keywords'] = result.keywords
        
        if hasattr(result, 'sentiment'):
            data['sentiment'] = result.sentiment
        
        if hasattr(result, 'entities'):
            data['entities'] = result.entities
        
        if hasattr(result, 'topics'):
            data['topics'] = result.topics
        
        if hasattr(result, 'metadata'):
            data['metadata'] = result.metadata
        
        return data


__all__ = [
    "DocumentAnalysisExporter"
]
















