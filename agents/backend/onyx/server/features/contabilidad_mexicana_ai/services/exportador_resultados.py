"""
Exportador de Resultados
=========================

Servicio para exportar resultados de cálculos y asesorías a diferentes formatos.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class ExportadorResultados:
    """Exporta resultados a diferentes formatos."""
    
    def exportar_json(self, resultado: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Exportar resultado a JSON.
        
        Args:
            resultado: Resultado a exportar
            filename: Nombre del archivo (opcional)
        
        Returns:
            Contenido JSON como string
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resultado_{timestamp}.json"
        
        # Convertir Decimal a float para JSON
        resultado_serializable = self._make_json_serializable(resultado)
        
        json_content = json.dumps(resultado_serializable, indent=2, ensure_ascii=False)
        
        return json_content
    
    def exportar_markdown(self, resultado: Dict[str, Any]) -> str:
        """
        Exportar resultado a Markdown.
        
        Args:
            resultado: Resultado a exportar
        
        Returns:
            Contenido Markdown como string
        """
        lines = []
        lines.append("# Resultado de Cálculo Fiscal\n")
        lines.append(f"**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if "regimen" in resultado:
            lines.append(f"**Régimen**: {resultado['regimen']}")
        
        if "tipo_impuesto" in resultado:
            lines.append(f"**Tipo de Impuesto**: {resultado['tipo_impuesto']}")
        
        if "ingresos_anuales" in resultado:
            lines.append(f"**Ingresos Anuales**: ${resultado['ingresos_anuales']:,.2f}")
        
        if "impuesto_total" in resultado:
            lines.append(f"**Impuesto Total**: ${resultado['impuesto_total']:,.2f}")
        
        if "tasa_efectiva" in resultado:
            lines.append(f"**Tasa Efectiva**: {resultado['tasa_efectiva']:.2f}%")
        
        if "desglose" in resultado:
            lines.append("\n## Desglose por Tramos\n")
            lines.append("| Tramo | Base | Tasa | Impuesto |")
            lines.append("|-------|------|------|----------|")
            for tramo in resultado["desglose"]:
                lines.append(
                    f"| {tramo['tramo']} | "
                    f"${tramo['base']:,.2f} | "
                    f"{tramo['tasa']*100:.2f}% | "
                    f"${tramo['impuesto']:,.2f} |"
                )
        
        if "resultado" in resultado:
            lines.append("\n## Detalle del Cálculo\n")
            lines.append(resultado["resultado"])
        
        return "\n".join(lines)
    
    def exportar_csv(self, resultado: Dict[str, Any]) -> str:
        """
        Exportar resultado a CSV.
        
        Args:
            resultado: Resultado a exportar
        
        Returns:
            Contenido CSV como string
        """
        lines = []
        lines.append("Concepto,Valor")
        
        if "regimen" in resultado:
            lines.append(f"Régimen,{resultado['regimen']}")
        
        if "tipo_impuesto" in resultado:
            lines.append(f"Tipo de Impuesto,{resultado['tipo_impuesto']}")
        
        if "ingresos_anuales" in resultado:
            lines.append(f"Ingresos Anuales,{resultado['ingresos_anuales']}")
        
        if "impuesto_total" in resultado:
            lines.append(f"Impuesto Total,{resultado['impuesto_total']}")
        
        if "tasa_efectiva" in resultado:
            lines.append(f"Tasa Efectiva,{resultado['tasa_efectiva']}")
        
        if "desglose" in resultado:
            lines.append("\nTramo,Base,Tasa,Impuesto")
            for tramo in resultado["desglose"]:
                lines.append(
                    f"{tramo['tramo']},"
                    f"{tramo['base']},"
                    f"{tramo['tasa']},"
                    f"{tramo['impuesto']}"
                )
        
        return "\n".join(lines)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convertir objetos a formato JSON serializable."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
