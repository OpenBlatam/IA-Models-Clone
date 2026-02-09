"""
HTML Report Generator
Generates beautiful HTML reports for test results
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

class HTMLReportGenerator:
    """Generate HTML reports from test results"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate(self, test_results: Dict[str, Any], output_file: str = "test_report.html"):
        """Generate HTML report from test results"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TruthGPT Test Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .timestamp {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card.success .value {{
            color: #28a745;
        }}
        
        .stat-card.failure .value {{
            color: #dc3545;
        }}
        
        .stat-card.warning .value {{
            color: #ffc107;
        }}
        
        .stat-card.info .value {{
            color: #17a2b8;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .test-list {{
            list-style: none;
        }}
        
        .test-item {{
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid;
            background: #f8f9fa;
        }}
        
        .test-item.passed {{
            border-left-color: #28a745;
            background: #d4edda;
        }}
        
        .test-item.failed {{
            border-left-color: #dc3545;
            background: #f8d7da;
        }}
        
        .test-item.error {{
            border-left-color: #dc3545;
            background: #f8d7da;
        }}
        
        .test-item.skipped {{
            border-left-color: #ffc107;
            background: #fff3cd;
        }}
        
        .test-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .test-details {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        
        .traceback {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            overflow-x: auto;
            margin-top: 10px;
            white-space: pre-wrap;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            background: #f8f9fa;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TruthGPT Test Report</h1>
            <div class="timestamp">Generated: {self.timestamp}</div>
        </div>
        
        <div class="summary">
            {self._generate_summary_cards(test_results)}
        </div>
        
        <div class="content">
            {self._generate_test_details(test_results)}
        </div>
        
        <div class="footer">
            <p>TruthGPT Testing Framework | Generated by HTML Report Generator</p>
        </div>
    </div>
</body>
</html>"""
        
        # Write to file
        output_path = Path(output_file)
        output_path.write_text(html, encoding='utf-8')
        return output_path
    
    def _generate_summary_cards(self, results: Dict[str, Any]) -> str:
        """Generate summary stat cards"""
        total = results.get('total_tests', 0)
        passed = total - results.get('failures', 0) - results.get('errors', 0) - results.get('skipped', 0)
        failures = results.get('failures', 0)
        errors = results.get('errors', 0)
        skipped = results.get('skipped', 0)
        success_rate = results.get('success_rate', 0)
        execution_time = results.get('execution_time', 0)
        
        cards = f"""
            <div class="stat-card info">
                <div class="value">{total}</div>
                <div class="label">Total Tests</div>
            </div>
            <div class="stat-card success">
                <div class="value">{passed}</div>
                <div class="label">Passed</div>
            </div>
            <div class="stat-card failure">
                <div class="value">{failures + errors}</div>
                <div class="label">Failed</div>
            </div>
            <div class="stat-card warning">
                <div class="value">{skipped}</div>
                <div class="label">Skipped</div>
            </div>
            <div class="stat-card success">
                <div class="value">{success_rate:.1f}%</div>
                <div class="label">Success Rate</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {success_rate}%">{success_rate:.1f}%</div>
                </div>
            </div>
            <div class="stat-card info">
                <div class="value">{execution_time:.2f}s</div>
                <div class="label">Execution Time</div>
            </div>
        """
        return cards
    
    def _generate_test_details(self, results: Dict[str, Any]) -> str:
        """Generate detailed test results"""
        sections = []
        
        # Failures
        if results.get('failures'):
            failures_html = '<div class="section"><h2>❌ Failures</h2><ul class="test-list">'
            for test, traceback in results.get('failures', []):
                failures_html += f'''
                    <li class="test-item failed">
                        <div class="test-name">{test}</div>
                        <div class="test-details">Test failed</div>
                        <div class="traceback">{self._escape_html(traceback)}</div>
                    </li>
                '''
            failures_html += '</ul></div>'
            sections.append(failures_html)
        
        # Errors
        if results.get('errors'):
            errors_html = '<div class="section"><h2>💥 Errors</h2><ul class="test-list">'
            for test, traceback in results.get('errors', []):
                errors_html += f'''
                    <li class="test-item error">
                        <div class="test-name">{test}</div>
                        <div class="test-details">Test error</div>
                        <div class="traceback">{self._escape_html(traceback)}</div>
                    </li>
                '''
            errors_html += '</ul></div>'
            sections.append(errors_html)
        
        # Skipped
        if results.get('skipped'):
            skipped_html = '<div class="section"><h2>⏭️ Skipped</h2><ul class="test-list">'
            for test, reason in results.get('skipped', []):
                skipped_html += f'''
                    <li class="test-item skipped">
                        <div class="test-name">{test}</div>
                        <div class="test-details">Reason: {reason}</div>
                    </li>
                '''
            skipped_html += '</ul></div>'
            sections.append(skipped_html)
        
        if not sections:
            sections.append('<div class="section"><h2>✅ All Tests Passed!</h2><p>No failures, errors, or skipped tests.</p></div>')
        
        return '\n'.join(sections)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))







