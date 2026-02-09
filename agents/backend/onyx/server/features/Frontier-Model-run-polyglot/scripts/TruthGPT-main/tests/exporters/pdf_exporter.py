"""
PDF Exporter
Export test results to PDF format
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime

class PDFExporter:
    """Export test results to PDF"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def export_to_pdf(
        self,
        test_results: Dict[str, Any],
        output_file: str = "test_results.pdf"
    ) -> Path:
        """Export test results to PDF"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        except ImportError:
            print("⚠️  reportlab not installed. Install with: pip install reportlab")
            return None
        
        output_path = self.project_root / output_file
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30
        )
        story.append(Paragraph("🧪 TruthGPT Test Report", title_style))
        story.append(Spacer(1, 12))
        
        # Summary
        story.append(Paragraph("Summary", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Summary table
        total = test_results.get('total_tests', 0)
        passed = total - test_results.get('failures', 0) - test_results.get('errors', 0) - test_results.get('skipped', 0)
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Tests', str(total)],
            ['Passed', str(passed)],
            ['Failed', str(test_results.get('failures', 0))],
            ['Errors', str(test_results.get('errors', 0))],
            ['Skipped', str(test_results.get('skipped', 0))],
            ['Success Rate', f"{test_results.get('success_rate', 0):.1f}%"],
            ['Execution Time', f"{test_results.get('execution_time', 0):.2f}s"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Failures section
        if test_results.get('failures'):
            story.append(Paragraph("Failures", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for test, traceback in test_results.get('failures', [])[:10]:  # Limit to 10
                story.append(Paragraph(f"<b>{test}</b>", styles['Normal']))
                story.append(Paragraph(traceback[:500], styles['Code']))
                story.append(Spacer(1, 12))
        
        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        
        return output_path

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    exporter = PDFExporter(project_root)
    
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3,
        'failures': [('test_example', 'Traceback...')]
    }
    
    pdf_path = exporter.export_to_pdf(test_results)
    if pdf_path:
        print(f"✅ PDF exported to: {pdf_path}")

if __name__ == "__main__":
    main()







