"""
HTML Report Generator for Test Results
Generates beautiful HTML reports from test execution results
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def generate_html_report(results: Dict[str, Any], output_file: str = "test_report.html"):
    """Generate an HTML report from test results"""
    
    total_tests = results.get('total_tests', 0)
    passed = results.get('passed', 0)
    failures = results.get('failures', 0)
    errors = results.get('errors', 0)
    skipped = results.get('skipped', 0)
    success_rate = results.get('success_rate', 0.0)
    execution_time = results.get('execution_time', 0.0)
    tests_per_second = results.get('tests_per_second', 0.0)
    
    # Calculate percentages
    passed_pct = (passed / total_tests * 100) if total_tests > 0 else 0
    failed_pct = ((failures + errors) / total_tests * 100) if total_tests > 0 else 0
    skipped_pct = (skipped / total_tests * 100) if total_tests > 0 else 0
    
    # Determine status color
    if failures == 0 and errors == 0:
        status_color = "#28a745"  # Green
        status_text = "PASSED"
    else:
        status_color = "#dc3545"  # Red
        status_text = "FAILED"
    
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
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
            font-size: 1.1em;
        }}
        
        .status-badge {{
            display: inline-block;
            background: {status_color};
            color: white;
            padding: 10px 30px;
            border-radius: 25px;
            font-size: 1.2em;
            font-weight: bold;
            margin-top: 20px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
            text-align: center;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-card.passed {{
            border-left-color: #28a745;
        }}
        
        .stat-card.failed {{
            border-left-color: #dc3545;
        }}
        
        .stat-card.skipped {{
            border-left-color: #ffc107;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .progress-bar {{
            background: #e9ecef;
            border-radius: 10px;
            height: 30px;
            margin: 20px 0;
            overflow: hidden;
            position: relative;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            border-radius: 10px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        
        .progress-fill.failed {{
            background: linear-gradient(90deg, #dc3545, #c82333);
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        
        .section {{
            margin-top: 40px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TruthGPT Test Report</h1>
            <div class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="status-badge">{status_text}</div>
        </div>
        
        <div class="content">
            <div class="summary-grid">
                <div class="stat-card passed">
                    <div class="stat-value">{passed}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-card failed">
                    <div class="stat-value">{failures + errors}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card skipped">
                    <div class="stat-value">{skipped}</div>
                    <div class="stat-label">Skipped</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_tests}</div>
                    <div class="stat-label">Total Tests</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Success Rate</div>
                <div class="progress-bar">
                    <div class="progress-fill {'failed' if failures + errors > 0 else ''}" style="width: {success_rate}%">
                        {success_rate:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Execution Time</div>
                    <div class="metric-value">{execution_time:.2f}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tests per Second</div>
                    <div class="metric-value">{tests_per_second:.1f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Success Rate</div>
                    <div class="metric-value">{success_rate:.1f}%</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Test Distribution</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {passed_pct}%; background: #28a745;">
                        Passed ({passed_pct:.1f}%)
                    </div>
                </div>
                <div class="progress-bar" style="margin-top: 10px;">
                    <div class="progress-fill failed" style="width: {failed_pct}%;">
                        Failed ({failed_pct:.1f}%)
                    </div>
                </div>
                <div class="progress-bar" style="margin-top: 10px;">
                    <div class="progress-fill" style="width: {skipped_pct}%; background: #ffc107;">
                        Skipped ({skipped_pct:.1f}%)
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Generated by TruthGPT Test Runner | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML report generated: {output_file}")
    return output_file

def main():
    """Main function to generate HTML report from JSON results"""
    if len(sys.argv) < 2:
        print("Usage: python generate_html_report.py <results.json> [output.html]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "test_report.html"
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        generate_html_report(results, output_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
