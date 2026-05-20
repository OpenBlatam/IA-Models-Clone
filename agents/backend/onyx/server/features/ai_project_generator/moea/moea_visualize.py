"""
MOEA Visualization Helper
==========================
Script para generar visualizaciones de resultados MOEA
"""
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

API_BASE = "http://localhost:8000"


class MOEAVisualizer:
    """Visualizador de resultados MOEA"""
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
    
    def get_pareto_front(self, project_id: str) -> Optional[Dict]:
        """Obtener frente de Pareto"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/moea/pareto/{project_id}",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def plot_pareto_2d(
        self,
        pareto_data: Dict,
        title: str = "Pareto Front",
        save_path: Optional[str] = None
    ):
        """Graficar frente de Pareto 2D"""
        solutions = pareto_data.get('solutions', [])
        if not solutions:
            print("❌ No hay soluciones para graficar")
            return
        
        # Extraer objetivos
        f1_values = [s.get('f1', s.get('objectives', [{}])[0].get('value', 0)) 
                    for s in solutions]
        f2_values = [s.get('f2', s.get('objectives', [{}])[1].get('value', 0)) 
                    if len(s.get('objectives', [])) > 1 
                    else s.get('objectives', [{}])[0].get('value', 0)
                    for s in solutions]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(f1_values, f2_values, alpha=0.6, s=50)
        plt.xlabel('Objective 1 (f1)', fontsize=12)
        plt.ylabel('Objective 2 (f2)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pareto_3d(
        self,
        pareto_data: Dict,
        title: str = "Pareto Front 3D",
        save_path: Optional[str] = None
    ):
        """Graficar frente de Pareto 3D"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("❌ matplotlib 3D no disponible")
            return
        
        solutions = pareto_data.get('solutions', [])
        if not solutions:
            print("❌ No hay soluciones para graficar")
            return
        
        # Extraer objetivos (necesitamos al menos 3)
        objectives = []
        for s in solutions:
            obj_values = s.get('objectives', [])
            if len(obj_values) >= 3:
                objectives.append([
                    obj_values[0].get('value', 0),
                    obj_values[1].get('value', 0),
                    obj_values[2].get('value', 0)
                ])
        
        if not objectives:
            print("❌ Se necesitan al menos 3 objetivos para visualización 3D")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        objectives = np.array(objectives)
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                  alpha=0.6, s=50)
        
        ax.set_xlabel('Objective 1 (f1)', fontsize=12)
        ax.set_ylabel('Objective 2 (f2)', fontsize=12)
        ax.set_zlabel('Objective 3 (f3)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico 3D guardado: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_algorithm_comparison(
        self,
        results: List[Dict],
        metric: str = "hypervolume",
        save_path: Optional[str] = None
    ):
        """Comparar algoritmos en un gráfico"""
        algorithms = {}
        
        for result in results:
            algo = result.get('algorithm', 'unknown')
            if algo not in algorithms:
                algorithms[algo] = []
            
            if metric in result:
                algorithms[algo].append(result[metric])
        
        if not algorithms:
            print(f"❌ No hay datos de {metric}")
            return
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        means = [np.mean(algorithms[algo]) for algo in algorithms.keys()]
        stds = [np.std(algorithms[algo]) for algo in algorithms.keys()]
        
        plt.bar(x, means, width, yerr=stds, alpha=0.7, capsize=5)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'Algorithm Comparison - {metric.capitalize()}', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x, list(algorithms.keys()))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparación guardada: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_project(
        self,
        project_id: str,
        output_dir: str = "moea_visualizations"
    ):
        """Visualizar proyecto completo"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📊 Visualizando proyecto: {project_id}")
        
        # Obtener datos
        pareto_data = self.get_pareto_front(project_id)
        
        if not pareto_data:
            print("❌ No se pudieron obtener datos del proyecto")
            return
        
        # Determinar número de objetivos
        solutions = pareto_data.get('solutions', [])
        if not solutions:
            print("❌ No hay soluciones")
            return
        
        num_objectives = len(solutions[0].get('objectives', []))
        
        # Visualizar según número de objetivos
        if num_objectives == 2:
            self.plot_pareto_2d(
                pareto_data,
                title=f"Pareto Front - {project_id}",
                save_path=str(output_path / f"{project_id}_pareto_2d.png")
            )
        elif num_objectives >= 3:
            self.plot_pareto_2d(
                pareto_data,
                title=f"Pareto Front (f1 vs f2) - {project_id}",
                save_path=str(output_path / f"{project_id}_pareto_2d.png")
            )
            self.plot_pareto_3d(
                pareto_data,
                title=f"Pareto Front 3D - {project_id}",
                save_path=str(output_path / f"{project_id}_pareto_3d.png")
            )
        
        print(f"✅ Visualizaciones guardadas en: {output_dir}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Visualization")
    parser.add_argument(
        "project_id",
        help="Project ID to visualize"
    )
    parser.add_argument(
        "--output",
        default="moea_visualizations",
        help="Output directory"
    )
    parser.add_argument(
        "--url",
        default=API_BASE,
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    visualizer = MOEAVisualizer(args.url)
    visualizer.visualize_project(args.project_id, args.output)


if __name__ == "__main__":
    main()

