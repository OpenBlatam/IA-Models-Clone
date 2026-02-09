"""Advanced table processing utilities"""
from typing import Dict, Any, List, Optional
import re


class TableProcessor:
    """Process and enhance tables"""
    
    def detect_table_type(self, table: Dict[str, Any]) -> str:
        """
        Detect table type and structure
        
        Args:
            table: Table data
            
        Returns:
            Table type (data, matrix, pivot, etc.)
        """
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        if not headers or not rows:
            return "simple"
        
        # Check if it's a matrix (all numeric)
        all_numeric = True
        for row in rows:
            for cell in row:
                try:
                    float(str(cell).replace(',', '').replace('$', '').strip())
                except:
                    all_numeric = False
                    break
            if not all_numeric:
                break
        
        if all_numeric and len(headers) == len(rows[0]) if rows else False:
            return "matrix"
        
        # Check if it's a pivot table (first column as categories)
        if len(headers) > 2:
            return "pivot"
        
        return "data"
    
    def extract_formulas(self, table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract formulas from table cells
        
        Args:
            table: Table data
            
        Returns:
            List of formulas found
        """
        formulas = []
        rows = table.get("rows", [])
        
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell_str = str(cell)
                # Look for Excel-like formulas
                if cell_str.startswith('='):
                    formulas.append({
                        "row": row_idx,
                        "col": col_idx,
                        "formula": cell_str,
                        "type": "excel"
                    })
                # Look for LaTeX math
                elif '$' in cell_str:
                    formulas.append({
                        "row": row_idx,
                        "col": col_idx,
                        "formula": cell_str,
                        "type": "latex"
                    })
        
        return formulas
    
    def calculate_statistics(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics for numeric columns
        
        Args:
            table: Table data
            
        Returns:
            Statistics dictionary
        """
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        
        if not headers or not rows:
            return {}
        
        stats = {}
        
        for col_idx, header in enumerate(headers):
            values = []
            for row in rows:
                if col_idx < len(row):
                    try:
                        value = float(str(row[col_idx]).replace(',', '').replace('$', '').strip())
                        values.append(value)
                    except:
                        pass
            
            if values:
                stats[header] = {
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2] if values else 0
                }
        
        return stats
    
    def enhance_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance table with additional information
        
        Args:
            table: Table data
            
        Returns:
            Enhanced table
        """
        enhanced = table.copy()
        
        enhanced["type"] = self.detect_table_type(table)
        enhanced["formulas"] = self.extract_formulas(table)
        enhanced["statistics"] = self.calculate_statistics(table)
        enhanced["row_count"] = len(table.get("rows", []))
        enhanced["col_count"] = len(table.get("headers", []))
        
        return enhanced


# Global table processor
_table_processor: Optional[TableProcessor] = None


def get_table_processor() -> TableProcessor:
    """Get global table processor"""
    global _table_processor
    if _table_processor is None:
        _table_processor = TableProcessor()
    return _table_processor

