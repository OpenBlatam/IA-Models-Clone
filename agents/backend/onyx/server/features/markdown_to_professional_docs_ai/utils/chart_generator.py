"""Chart Generator - Generate charts and diagrams from data"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, Any, Optional, List
import base64
import numpy as np


class ChartGenerator:
    """Generate charts and diagrams from table data"""
    
    def create_chart_from_table(
        self,
        table: Dict[str, Any],
        chart_type: str = "bar"
    ) -> Optional[bytes]:
        """
        Create a chart image from table data
        
        Args:
            table: Table data with headers and rows
            chart_type: Type of chart (bar, line, pie)
            
        Returns:
            Chart image as bytes or None
        """
        try:
            if not table.get("rows") or len(table["rows"]) == 0:
                return None
            
            headers = table.get("headers", [])
            if len(headers) < 2:
                return None
            
            # Extract data
            x_data = []
            y_data = []
            
            for row in table["rows"]:
                if len(row) >= 2:
                    x_data.append(str(row[0]))
                    try:
                        # Try to convert to number
                        value = float(str(row[1]).replace(',', '').replace('$', '').strip())
                        y_data.append(value)
                    except:
                        y_data.append(0)
            
            if not x_data or not y_data:
                return None
            
            # Create chart
            plt.figure(figsize=(10, 6))
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.facecolor'] = 'white'
            
            if chart_type == "bar":
                bars = plt.bar(x_data, y_data, color='#366092', alpha=0.8, edgecolor='#2c4a6b', linewidth=1.2)
                plt.ylabel(headers[1] if len(headers) > 1 else "Value", fontsize=11, fontweight='bold')
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            elif chart_type == "line":
                plt.plot(x_data, y_data, marker='o', color='#366092', linewidth=2.5, 
                        markersize=8, markerfacecolor='white', markeredgewidth=2)
                plt.fill_between(x_data, y_data, alpha=0.3, color='#366092')
                plt.ylabel(headers[1] if len(headers) > 1 else "Value", fontsize=11, fontweight='bold')
            elif chart_type == "pie":
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(x_data)))
                plt.pie(y_data, labels=x_data, autopct='%1.1f%%', startangle=90,
                       colors=colors_pie, textprops={'fontsize': 10})
            elif chart_type == "scatter":
                plt.scatter(x_data, y_data, s=100, c='#366092', alpha=0.6, edgecolors='#2c4a6b', linewidth=1.5)
                plt.ylabel(headers[1] if len(headers) > 1 else "Value", fontsize=11, fontweight='bold')
            elif chart_type == "area":
                plt.fill_between(x_data, y_data, alpha=0.5, color='#366092')
                plt.plot(x_data, y_data, color='#2c4a6b', linewidth=2)
                plt.ylabel(headers[1] if len(headers) > 1 else "Value", fontsize=11, fontweight='bold')
            elif chart_type == "histogram":
                plt.hist(y_data, bins=min(10, len(y_data)), color='#366092', alpha=0.7, edgecolor='#2c4a6b')
                plt.ylabel("Frequency", fontsize=11, fontweight='bold')
            
            plt.xlabel(headers[0] if headers else "Category", fontsize=11, fontweight='bold')
            title = f"{headers[0]} vs {headers[1]}" if len(headers) >= 2 else "Chart"
            plt.title(title, fontsize=13, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            chart_bytes = buf.read()
            buf.close()
            plt.close()
            
            return chart_bytes
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
    
    def create_plotly_chart_data(
        self,
        table: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create Plotly chart data structure for HTML charts
        
        Args:
            table: Table data with headers and rows
            
        Returns:
            Plotly chart data dictionary
        """
        try:
            if not table.get("rows") or len(table["rows"]) == 0:
                return None
            
            headers = table.get("headers", [])
            if len(headers) < 2:
                return None
            
            # Extract data
            x_data = []
            y_data = []
            
            for row in table["rows"]:
                if len(row) >= 2:
                    x_data.append(str(row[0]))
                    try:
                        value = float(str(row[1]).replace(',', '').replace('$', '').strip())
                        y_data.append(value)
                    except:
                        y_data.append(0)
            
            if not x_data or not y_data:
                return None
            
            # Create Plotly data structure with enhanced styling
            data = [{
                "x": x_data,
                "y": y_data,
                "type": "bar",
                "marker": {
                    "color": "#366092",
                    "line": {
                        "color": "#2c4a6b",
                        "width": 1.5
                    }
                },
                "text": [f"{val:.1f}" for val in y_data],
                "textposition": "outside",
                "hovertemplate": "<b>%{x}</b><br>Value: %{y}<extra></extra>"
            }]
            
            layout = {
                "title": {
                    "text": f"{headers[0]} vs {headers[1]}" if len(headers) >= 2 else "Chart",
                    "font": {"size": 16, "color": "#1a1a1a"}
                },
                "xaxis": {
                    "title": {"text": headers[0] if headers else "Category", "font": {"size": 12}},
                    "tickangle": -45
                },
                "yaxis": {
                    "title": {"text": headers[1] if len(headers) > 1 else "Value", "font": {"size": 12}}
                },
                "template": "plotly_white",
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"family": "Arial, sans-serif"},
                "margin": {"l": 60, "r": 20, "t": 60, "b": 80}
            }
            
            return {
                "data": data,
                "layout": layout
            }
        except Exception as e:
            print(f"Error creating Plotly chart data: {e}")
            return None
    
    def create_diagram(
        self,
        diagram_type: str,
        data: Dict[str, Any]
    ) -> Optional[bytes]:
        """
        Create diagrams (flowcharts, etc.) from data
        
        Args:
            diagram_type: Type of diagram
            data: Diagram data
            
        Returns:
            Diagram image as bytes
        """
        try:
            if diagram_type == "flowchart" and "nodes" in data:
                # Simple flowchart using matplotlib
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.axis('off')
                
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                
                # Draw nodes
                for node in nodes:
                    x, y = node.get("position", (0, 0))
                    label = node.get("label", "")
                    ax.text(x, y, label, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#366092', alpha=0.7),
                           fontsize=10, color='white', fontweight='bold')
                
                # Draw edges
                for edge in edges:
                    from_node = edge.get("from")
                    to_node = edge.get("to")
                    # Simple line drawing (would need actual node positions)
                    pass
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                diagram_bytes = buf.read()
                buf.close()
                plt.close()
                
                return diagram_bytes
        except Exception as e:
            print(f"Error creating diagram: {e}")
        
        return None
    
    def auto_detect_chart_type(self, table: Dict[str, Any]) -> str:
        """
        Automatically detect best chart type for table data
        
        Args:
            table: Table data
            
        Returns:
            Recommended chart type
        """
        if not table.get("rows"):
            return "bar"
        
        headers = table.get("headers", [])
        if len(headers) < 2:
            return "bar"
        
        # Analyze data to determine best chart type
        try:
            # Check if second column is numeric
            numeric_count = 0
            for row in table["rows"][:5]:  # Sample first 5 rows
                if len(row) >= 2:
                    try:
                        float(str(row[1]).replace(',', '').replace('$', '').strip())
                        numeric_count += 1
                    except:
                        pass
            
            if numeric_count < 3:
                return "bar"
            
            # Check data characteristics
            values = []
            for row in table["rows"]:
                if len(row) >= 2:
                    try:
                        values.append(float(str(row[1]).replace(',', '').replace('$', '').strip()))
                    except:
                        pass
            
            if len(values) > 10:
                return "line"  # Better for many data points
            elif len(values) <= 5:
                return "pie"  # Better for few categories
            else:
                return "bar"  # Default
        except:
            return "bar"

