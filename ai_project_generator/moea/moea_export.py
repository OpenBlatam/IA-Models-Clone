"""
MOEA Export Tool - Herramienta de exportación avanzada
======================================================
Exporta proyectos, resultados y configuraciones en múltiples formatos
"""
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import requests


class MOEAExporter:
    """Exportador de proyectos MOEA"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def export_project(
        self,
        project_id: str,
        output_format: str = "zip",
        include_results: bool = True,
        include_config: bool = True
    ) -> Optional[str]:
        """Exportar proyecto completo"""
        print(f"📦 Exportando proyecto: {project_id}")
        
        # Obtener información del proyecto
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/project/{project_id}",
                timeout=10
            )
            if response.status_code != 200:
                print(f"❌ Error obteniendo información del proyecto")
                return None
            
            project_data = response.json()
            project_dir = Path(project_data.get('result', {}).get('project_dir', ''))
            
            if not project_dir.exists():
                print(f"❌ Directorio del proyecto no encontrado: {project_dir}")
                return None
            
            # Crear archivo de exportación
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"moea_export_{project_id}_{timestamp}.{output_format}"
            
            if output_format == "zip":
                self._create_zip(project_dir, output_file, include_results, include_config)
            elif output_format == "tar":
                self._create_tar(project_dir, output_file, include_results, include_config)
            elif output_format == "tar.gz":
                self._create_tar(project_dir, output_file, include_results, include_config, compress=True)
            else:
                print(f"❌ Formato no soportado: {output_format}")
                return None
            
            print(f"✅ Proyecto exportado: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error exportando proyecto: {e}")
            return None
    
    def _create_zip(
        self,
        project_dir: Path,
        output_file: str,
        include_results: bool,
        include_config: bool
    ):
        """Crear archivo ZIP"""
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in project_dir.rglob('*'):
                if file_path.is_file():
                    # Filtrar archivos
                    if not include_results and 'results' in str(file_path):
                        continue
                    if not include_config and file_path.name in ['.env', 'config.json']:
                        continue
                    
                    arcname = file_path.relative_to(project_dir.parent)
                    zipf.write(file_path, arcname)
    
    def _create_tar(
        self,
        project_dir: Path,
        output_file: str,
        include_results: bool,
        include_config: bool,
        compress: bool = False
    ):
        """Crear archivo TAR"""
        mode = 'w:gz' if compress else 'w'
        with tarfile.open(output_file, mode) as tarf:
            for file_path in project_dir.rglob('*'):
                if file_path.is_file():
                    # Filtrar archivos
                    if not include_results and 'results' in str(file_path):
                        continue
                    if not include_config and file_path.name in ['.env', 'config.json']:
                        continue
                    
                    arcname = file_path.relative_to(project_dir.parent)
                    tarf.add(file_path, arcname=arcname)
    
    def export_results(
        self,
        project_id: str,
        format: str = "json"
    ) -> Optional[str]:
        """Exportar solo resultados"""
        print(f"📊 Exportando resultados: {project_id}")
        
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/moea/export/{project_id}",
                params={"format": format},
                timeout=30
            )
            
            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"moea_results_{project_id}_{timestamp}.{format}"
                
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Resultados exportados: {filename}")
                return filename
            else:
                print(f"❌ Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error exportando resultados: {e}")
            return None
    
    def export_config(self, output_file: str = "moea_config_backup.json"):
        """Exportar configuración del sistema"""
        print("⚙️  Exportando configuración...")
        
        config = {
            "exported_at": datetime.now().isoformat(),
            "api_url": self.base_url,
            "config": {}
        }
        
        # Intentar obtener configuración del servidor
        try:
            response = requests.get(f"{self.base_url}/api/v1/system/info", timeout=5)
            if response.status_code == 200:
                config["config"] = response.json()
        except:
            pass
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuración exportada: {output_file}")
        return output_file


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Export Tool")
    parser.add_argument(
        'project_id',
        help='ID del proyecto a exportar'
    )
    parser.add_argument(
        '--format',
        choices=['zip', 'tar', 'tar.gz'],
        default='zip',
        help='Formato de exportación'
    )
    parser.add_argument(
        '--no-results',
        action='store_true',
        help='Excluir resultados'
    )
    parser.add_argument(
        '--no-config',
        action='store_true',
        help='Excluir configuración'
    )
    parser.add_argument(
        '--results-only',
        action='store_true',
        help='Exportar solo resultados'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='URL base de la API'
    )
    
    args = parser.parse_args()
    
    exporter = MOEAExporter(args.url)
    
    if args.results_only:
        exporter.export_results(args.project_id)
    else:
        exporter.export_project(
            args.project_id,
            output_format=args.format,
            include_results=not args.no_results,
            include_config=not args.no_config
        )


if __name__ == "__main__":
    main()

