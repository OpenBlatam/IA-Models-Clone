"""
Advanced Export System
======================
Sistema de exportación avanzada en múltiples formatos
"""

import json
import base64
import zipfile
import io
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import time


class ExportFormat(Enum):
    """Formatos de exportación"""
    JSON = "json"
    ZIP = "zip"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class ExportConfig:
    """Configuración de exportación"""
    format: ExportFormat
    include_images: bool = True
    include_metadata: bool = True
    include_history: bool = False
    include_statistics: bool = False
    compress: bool = False
    custom_fields: Optional[List[str]] = None


class AdvancedExporter:
    """
    Exportador avanzado con múltiples formatos
    """
    
    def __init__(self):
        self.export_history: List[Dict] = []
    
    def export_result(
        self,
        result_data: Dict,
        config: ExportConfig,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Exportar resultado en formato especificado
        
        Args:
            result_data: Datos del resultado
            config: Configuración de exportación
            output_path: Ruta de salida (opcional)
        """
        if config.format == ExportFormat.JSON:
            return self._export_json(result_data, config)
        elif config.format == ExportFormat.ZIP:
            return self._export_zip(result_data, config)
        elif config.format == ExportFormat.HTML:
            return self._export_html(result_data, config)
        elif config.format == ExportFormat.MARKDOWN:
            return self._export_markdown(result_data, config)
        elif config.format == ExportFormat.CSV:
            return self._export_csv(result_data, config)
        else:
            raise ValueError(f"Formato no soportado: {config.format}")
    
    def _export_json(self, result_data: Dict, config: ExportConfig) -> bytes:
        """Exportar como JSON"""
        export_data = self._prepare_export_data(result_data, config)
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        return json_str.encode('utf-8')
    
    def _export_zip(self, result_data: Dict, config: ExportConfig) -> bytes:
        """Exportar como ZIP"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Agregar JSON con metadata
            export_data = self._prepare_export_data(result_data, config)
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            zip_file.writestr('result.json', json_str)
            
            # Agregar imágenes si están incluidas
            if config.include_images:
                if 'original_image' in result_data:
                    zip_file.writestr('original.png', self._decode_image(result_data['original_image']))
                if 'result_image' in result_data:
                    zip_file.writestr('result.png', self._decode_image(result_data['result_image']))
                if 'mask' in result_data:
                    zip_file.writestr('mask.png', self._decode_image(result_data['mask']))
            
            # Agregar README
            readme = self._generate_readme(result_data, config)
            zip_file.writestr('README.md', readme)
        
        zip_buffer.seek(0)
        return zip_buffer.read()
    
    def _export_html(self, result_data: Dict, config: ExportConfig) -> bytes:
        """Exportar como HTML"""
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resultado - Character Clothing Changer AI</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .content {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .image-container {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }}
        .image-box {{
            flex: 1;
            text-align: center;
        }}
        .image-box img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metadata {{
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .metadata h3 {{
            margin-top: 0;
        }}
        .metadata-item {{
            margin: 10px 0;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>👔 Character Clothing Changer AI</h1>
        <p>Resultado de Procesamiento</p>
    </div>
    
    <div class="content">
        <h2>Imágenes</h2>
        <div class="image-container">
            <div class="image-box">
                <h3>Original</h3>
                <img src="data:image/png;base64,{result_data.get('original_image', '')}" alt="Original">
            </div>
            <div class="image-box">
                <h3>Resultado</h3>
                <img src="data:image/png;base64,{result_data.get('result_image', '')}" alt="Resultado">
            </div>
        </div>
        
        <div class="metadata">
            <h3>Metadata</h3>
            <div class="metadata-item">
                <span class="metadata-label">Descripción:</span>
                {result_data.get('clothing_description', 'N/A')}
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Personaje:</span>
                {result_data.get('character_name', 'N/A')}
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Fecha:</span>
                {result_data.get('timestamp', 'N/A')}
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Calidad:</span>
                {result_data.get('quality_score', 'N/A')}
            </div>
        </div>
    </div>
</body>
</html>
        """
        return html.encode('utf-8')
    
    def _export_markdown(self, result_data: Dict, config: ExportConfig) -> bytes:
        """Exportar como Markdown"""
        md = f"""# Resultado - Character Clothing Changer AI

## Información General

- **Descripción:** {result_data.get('clothing_description', 'N/A')}
- **Personaje:** {result_data.get('character_name', 'N/A')}
- **Fecha:** {result_data.get('timestamp', 'N/A')}
- **Calidad:** {result_data.get('quality_score', 'N/A')}

## Imágenes

### Original
![Original](data:image/png;base64,{result_data.get('original_image', '')[:100]}...)

### Resultado
![Resultado](data:image/png;base64,{result_data.get('result_image', '')[:100]}...)

## Metadata Técnica

```json
{json.dumps(result_data.get('metadata', {}), indent=2)}
```
"""
        return md.encode('utf-8')
    
    def _export_csv(self, result_data: Dict, config: ExportConfig) -> bytes:
        """Exportar como CSV"""
        import csv
        
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        
        # Headers
        writer.writerow(['Campo', 'Valor'])
        
        # Data
        writer.writerow(['Descripción', result_data.get('clothing_description', '')])
        writer.writerow(['Personaje', result_data.get('character_name', '')])
        writer.writerow(['Fecha', result_data.get('timestamp', '')])
        writer.writerow(['Calidad', result_data.get('quality_score', '')])
        
        if config.include_metadata and 'metadata' in result_data:
            for key, value in result_data['metadata'].items():
                writer.writerow([key, str(value)])
        
        return csv_buffer.getvalue().encode('utf-8')
    
    def _prepare_export_data(self, result_data: Dict, config: ExportConfig) -> Dict:
        """Preparar datos para exportación"""
        export_data = {
            'exported_at': time.time(),
            'export_format': config.format.value,
            'clothing_description': result_data.get('clothing_description'),
            'character_name': result_data.get('character_name'),
            'timestamp': result_data.get('timestamp')
        }
        
        if config.include_images:
            export_data['images'] = {
                'original': result_data.get('original_image'),
                'result': result_data.get('result_image'),
                'mask': result_data.get('mask')
            }
        
        if config.include_metadata:
            export_data['metadata'] = result_data.get('metadata', {})
        
        if config.include_statistics:
            export_data['statistics'] = result_data.get('statistics', {})
        
        if config.custom_fields:
            for field in config.custom_fields:
                if field in result_data:
                    export_data[field] = result_data[field]
        
        return export_data
    
    def _decode_image(self, image_data: str) -> bytes:
        """Decodificar imagen base64"""
        if isinstance(image_data, str):
            return base64.b64decode(image_data)
        return image_data
    
    def _generate_readme(self, result_data: Dict, config: ExportConfig) -> str:
        """Generar README para ZIP"""
        return f"""# Resultado - Character Clothing Changer AI

## Información

- **Descripción:** {result_data.get('clothing_description', 'N/A')}
- **Personaje:** {result_data.get('character_name', 'N/A')}
- **Fecha:** {result_data.get('timestamp', 'N/A')}

## Archivos

- `result.json` - Datos completos en JSON
- `original.png` - Imagen original
- `result.png` - Imagen resultado
- `mask.png` - Máscara utilizada (si disponible)

## Exportado

Formato: {config.format.value}
Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""


# Instancia global
advanced_exporter = AdvancedExporter()

