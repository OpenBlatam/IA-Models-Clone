"""
MOEA Integration - Herramientas de integración
===============================================
Herramientas para integrar MOEA con otros sistemas
"""
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class MOEAIntegration:
    """Herramientas de integración MOEA"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def export_to_prometheus(self, output_file: str = "moea_metrics.prom") -> str:
        """Exportar métricas en formato Prometheus"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/stats", timeout=5)
            if response.status_code != 200:
                return None
            
            stats = response.json()
            
            prom_metrics = []
            prom_metrics.append(f"# MOEA Metrics - {datetime.now().isoformat()}")
            prom_metrics.append(f"moea_processed_projects_total {stats.get('processed_count', 0)}")
            prom_metrics.append(f"moea_queue_size {stats.get('queue_size', 0)}")
            prom_metrics.append(f"moea_avg_time_seconds {stats.get('avg_time', 0)}")
            
            with open(output_file, 'w') as f:
                f.write('\n'.join(prom_metrics))
            
            return output_file
            
        except Exception as e:
            print(f"❌ Error exportando a Prometheus: {e}")
            return None
    
    def create_webhook_config(self, webhook_url: str, events: List[str], output_file: str = "webhook_config.json") -> str:
        """Crear configuración de webhook"""
        config = {
            "webhook_url": webhook_url,
            "events": events,
            "created_at": datetime.now().isoformat(),
            "headers": {
                "Content-Type": "application/json"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return output_file
    
    def generate_grafana_dashboard(self, output_file: str = "grafana_dashboard.json") -> str:
        """Generar dashboard de Grafana"""
        dashboard = {
            "dashboard": {
                "title": "MOEA System Dashboard",
                "panels": [
                    {
                        "title": "Processed Projects",
                        "targets": [
                            {
                                "expr": "moea_processed_projects_total",
                                "legendFormat": "Processed"
                            }
                        ]
                    },
                    {
                        "title": "Queue Size",
                        "targets": [
                            {
                                "expr": "moea_queue_size",
                                "legendFormat": "Queue"
                            }
                        ]
                    },
                    {
                        "title": "Average Time",
                        "targets": [
                            {
                                "expr": "moea_avg_time_seconds",
                                "legendFormat": "Avg Time"
                            }
                        ]
                    }
                ],
                "refresh": "5s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                }
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        return output_file
    
    def generate_docker_compose_monitoring(self, output_file: str = "docker-compose.monitoring.yml") -> str:
        """Generar docker-compose para monitoreo"""
        compose = """version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
"""
        
        with open(output_file, 'w') as f:
            f.write(compose)
        
        return output_file
    
    def generate_github_actions_ci(self, output_file: str = ".github/workflows/moea-ci.yml") -> str:
        """Generar workflow de GitHub Actions"""
        workflow = """name: MOEA CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        cd backend
        pytest
    
    - name: Health check
      run: |
        python moea_health.py
    
    - name: Security audit
      run: |
        python moea_security.py generated_projects/moea_optimization_system
"""
        
        workflow_path = Path(output_file)
        workflow_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(workflow_path, 'w') as f:
            f.write(workflow)
        
        return output_file


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Integration Tools")
    subparsers = parser.add_subparsers(dest='command', help='Comandos')
    
    # Prometheus
    prom_parser = subparsers.add_parser('prometheus', help='Exportar a Prometheus')
    prom_parser.add_argument('--output', default='moea_metrics.prom')
    prom_parser.add_argument('--url', default='http://localhost:8000')
    
    # Webhook
    webhook_parser = subparsers.add_parser('webhook', help='Crear configuración de webhook')
    webhook_parser.add_argument('url', help='URL del webhook')
    webhook_parser.add_argument('--events', nargs='+', default=['project.completed'])
    webhook_parser.add_argument('--output', default='webhook_config.json')
    
    # Grafana
    grafana_parser = subparsers.add_parser('grafana', help='Generar dashboard de Grafana')
    grafana_parser.add_argument('--output', default='grafana_dashboard.json')
    
    # Docker
    docker_parser = subparsers.add_parser('docker', help='Generar docker-compose para monitoreo')
    docker_parser.add_argument('--output', default='docker-compose.monitoring.yml')
    
    # CI
    ci_parser = subparsers.add_parser('ci', help='Generar workflow de CI')
    ci_parser.add_argument('--output', default='.github/workflows/moea-ci.yml')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    integration = MOEAIntegration(getattr(args, 'url', 'http://localhost:8000'))
    
    if args.command == 'prometheus':
        output = integration.export_to_prometheus(args.output)
        if output:
            print(f"✅ Métricas exportadas: {output}")
    
    elif args.command == 'webhook':
        output = integration.create_webhook_config(args.url, args.events, args.output)
        print(f"✅ Configuración de webhook creada: {output}")
    
    elif args.command == 'grafana':
        output = integration.generate_grafana_dashboard(args.output)
        print(f"✅ Dashboard de Grafana generado: {output}")
    
    elif args.command == 'docker':
        output = integration.generate_docker_compose_monitoring(args.output)
        print(f"✅ Docker compose generado: {output}")
    
    elif args.command == 'ci':
        output = integration.generate_github_actions_ci(args.output)
        print(f"✅ Workflow de CI generado: {output}")


if __name__ == "__main__":
    main()

