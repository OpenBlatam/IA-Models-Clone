"""
Storage Service - Servicio para persistencia de diseños
"""

import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from ..core.models import StoreDesign
from ..core.logging_config import get_logger
from ..core.exceptions import StorageError
from ..config.settings import settings

logger = get_logger(__name__)


class StorageService:
    """Servicio para almacenar y recuperar diseños"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.designs_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage service inicializado: {self.storage_path}")
    
    def save_design(self, design: StoreDesign) -> bool:
        """Guardar diseño en archivo"""
        try:
            file_path = self.storage_path / f"{design.store_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(design.dict(), f, indent=2, default=str, ensure_ascii=False)
            logger.info(
                f"Diseño guardado: {design.store_id}",
                extra={"store_id": design.store_id, "store_name": design.store_name}
            )
            return True
        except PermissionError as e:
            logger.error(
                f"Error de permisos guardando diseño: {e}",
                extra={"store_id": design.store_id, "path": str(file_path)},
                exc_info=True
            )
            raise StorageError(f"No se pudo guardar el diseño: permisos insuficientes", details={"store_id": design.store_id})
        except Exception as e:
            logger.error(
                f"Error guardando diseño: {e}",
                extra={"store_id": design.store_id},
                exc_info=True
            )
            raise StorageError(f"Error guardando diseño: {str(e)}", details={"store_id": design.store_id})
    
    def load_design(self, store_id: str) -> Optional[StoreDesign]:
        """Cargar diseño desde archivo"""
        try:
            file_path = self.storage_path / f"{store_id}.json"
            if not file_path.exists():
                logger.debug(f"Diseño no encontrado: {store_id}", extra={"store_id": store_id})
                return None
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            design = StoreDesign(**data)
            logger.debug(f"Diseño cargado: {store_id}", extra={"store_id": store_id})
            return design
        except json.JSONDecodeError as e:
            logger.error(
                f"Error de formato JSON al cargar diseño: {e}",
                extra={"store_id": store_id, "path": str(file_path)},
                exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"Error cargando diseño: {e}",
                extra={"store_id": store_id},
                exc_info=True
            )
            return None
    
    def list_designs(self) -> List[Dict[str, Any]]:
        """Listar todos los diseños guardados"""
        designs = []
        try:
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    designs.append({
                        "store_id": data.get("store_id"),
                        "store_name": data.get("store_name"),
                        "store_type": data.get("store_type"),
                        "style": data.get("style"),
                        "created_at": data.get("created_at")
                    })
                except Exception as e:
                    logger.warning(f"Error leyendo {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error listando diseños: {e}")
        
        return designs
    
    def delete_design(self, store_id: str) -> bool:
        """Eliminar diseño"""
        try:
            file_path = self.storage_path / f"{store_id}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Diseño eliminado: {store_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando diseño: {e}")
            return False
    
    def export_design(self, store_id: str, format: str = "json") -> Optional[str]:
        """Exportar diseño en formato específico"""
        design = self.load_design(store_id)
        if not design:
            return None
        
        if format == "json":
            return json.dumps(design.dict(), indent=2, default=str, ensure_ascii=False)
        elif format == "markdown":
            return self._export_markdown(design)
        elif format == "html":
            return self._export_html(design)
        else:
            return None
    
    def _export_markdown(self, design: StoreDesign) -> str:
        """Exportar diseño en formato Markdown"""
        md = f"""# Diseño de Local: {design.store_name}

## Información General
- **ID**: {design.store_id}
- **Tipo**: {design.store_type.value}
- **Estilo**: {design.style.value}
- **Creado**: {design.created_at}

## Descripción
{design.description}

## Layout

### Dimensiones
- Ancho: {design.layout.dimensions.get('width', 'N/A')}m
- Largo: {design.layout.dimensions.get('length', 'N/A')}m
- Alto: {design.layout.dimensions.get('height', 'N/A')}m

### Zonas
"""
        for zone in design.layout.zones:
            md += f"- **{zone.get('name', 'Zona')}**: {zone.get('size', 'N/A')} - {zone.get('purpose', '')}\n"
        
        md += f"""
## Plan de Marketing

### Audiencia Objetivo
{design.marketing_plan.target_audience}

### Estrategias de Marketing
"""
        for strategy in design.marketing_plan.marketing_strategy:
            md += f"- {strategy}\n"
        
        md += f"""
### Tácticas de Ventas
"""
        for tactic in design.marketing_plan.sales_tactics:
            md += f"- {tactic}\n"
        
        md += f"""
### Estrategia de Precios
{design.marketing_plan.pricing_strategy}

## Plan de Decoración

### Esquema de Colores
"""
        for key, value in design.decoration_plan.color_scheme.items():
            md += f"- **{key}**: {value}\n"
        
        md += f"""
### Presupuesto Estimado
"""
        total = sum(design.decoration_plan.budget_estimate.values())
        for key, value in design.decoration_plan.budget_estimate.items():
            md += f"- **{key}**: ${value:,.0f}\n"
        md += f"- **Total**: ${total:,.0f}\n"
        
        return md
    
    def _export_html(self, design: StoreDesign) -> str:
        """Exportar diseño en formato HTML"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Diseño: {design.store_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2C3E50; }}
        h2 {{ color: #34495E; border-bottom: 2px solid #3498DB; }}
        .info {{ background: #ECF0F1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <h1>🏪 Diseño de Local: {design.store_name}</h1>
    
    <div class="info">
        <p><strong>Tipo:</strong> {design.store_type.value}</p>
        <p><strong>Estilo:</strong> {design.style.value}</p>
        <p><strong>Creado:</strong> {design.created_at}</p>
    </div>
    
    <h2>Descripción</h2>
    <p>{design.description}</p>
    
    <h2>Layout</h2>
    <p><strong>Dimensiones:</strong> {design.layout.dimensions.get('width', 'N/A')}m x {design.layout.dimensions.get('length', 'N/A')}m</p>
    
    <h2>Plan de Marketing</h2>
    <h3>Audiencia Objetivo</h3>
    <p>{design.marketing_plan.target_audience}</p>
    
    <h3>Estrategias</h3>
    <ul>
"""
        for strategy in design.marketing_plan.marketing_strategy:
            html += f"        <li>{strategy}</li>\n"
        
        html += """    </ul>
    
    <h2>Plan de Decoración</h2>
    <h3>Presupuesto Estimado</h3>
    <ul>
"""
        total = sum(design.decoration_plan.budget_estimate.values())
        for key, value in design.decoration_plan.budget_estimate.items():
            html += f"        <li><strong>{key}</strong>: ${value:,.0f}</li>\n"
        html += f"        <li><strong>Total</strong>: ${total:,.0f}</li>\n"
        html += """    </ul>
</body>
</html>"""
        
        return html




