"""
Comparador de Resultados de Pruebas
Compara resultados de diferentes ejecuciones
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class TestComparison:
    """Compara resultados de pruebas."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def load_result(self, filepath: str) -> bool:
        """Carga un archivo de resultados."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['filepath'] = filepath
                data['loaded_at'] = datetime.now().isoformat()
                self.results.append(data)
                return True
        except Exception as e:
            print(f"Error cargando {filepath}: {e}")
            return False
    
    def compare_summaries(self) -> Dict[str, Any]:
        """Compara los resúmenes de todas las ejecuciones."""
        if len(self.results) < 2:
            return {"error": "Se necesitan al menos 2 resultados para comparar"}
        
        summaries = [r.get("summary", {}) for r in self.results]
        
        comparison = {
            "total_runs": len(self.results),
            "comparison_date": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Comparar métricas clave
        metrics_to_compare = [
            "total", "passed", "failed", "success_rate", "duration"
        ]
        
        for metric in metrics_to_compare:
            values = [s.get(metric, 0) for s in summaries]
            comparison["metrics"][metric] = {
                "values": values,
                "average": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "improvement": values[-1] - values[0] if len(values) >= 2 else 0
            }
        
        return comparison
    
    def find_regressions(self) -> List[Dict[str, Any]]:
        """Encuentra regresiones entre ejecuciones."""
        if len(self.results) < 2:
            return []
        
        regressions = []
        
        # Comparar tests entre ejecuciones
        previous_tests = {test["name"]: test for test in self.results[-2].get("tests", [])}
        current_tests = {test["name"]: test for test in self.results[-1].get("tests", [])}
        
        for test_name, current_test in current_tests.items():
            if test_name in previous_tests:
                previous_test = previous_tests[test_name]
                
                # Test que pasaba ahora falla
                if previous_test.get("passed") and not current_test.get("passed"):
                    regressions.append({
                        "type": "FAILURE_REGRESSION",
                        "test": test_name,
                        "previous": "PASSED",
                        "current": "FAILED",
                        "severity": "HIGH"
                    })
                
                # Test que se volvió más lento
                prev_duration = previous_test.get("duration", 0)
                curr_duration = current_test.get("duration", 0)
                if curr_duration > prev_duration * 1.5:  # 50% más lento
                    regressions.append({
                        "type": "PERFORMANCE_REGRESSION",
                        "test": test_name,
                        "previous_duration": prev_duration,
                        "current_duration": curr_duration,
                        "increase": f"{((curr_duration / prev_duration - 1) * 100):.1f}%",
                        "severity": "MEDIUM"
                    })
        
        return regressions
    
    def find_improvements(self) -> List[Dict[str, Any]]:
        """Encuentra mejoras entre ejecuciones."""
        if len(self.results) < 2:
            return []
        
        improvements = []
        
        previous_tests = {test["name"]: test for test in self.results[-2].get("tests", [])}
        current_tests = {test["name"]: test for test in self.results[-1].get("tests", [])}
        
        for test_name, current_test in current_tests.items():
            if test_name in previous_tests:
                previous_test = previous_tests[test_name]
                
                # Test que fallaba ahora pasa
                if not previous_test.get("passed") and current_test.get("passed"):
                    improvements.append({
                        "type": "FIXED_TEST",
                        "test": test_name,
                        "previous": "FAILED",
                        "current": "PASSED"
                    })
                
                # Test que se volvió más rápido
                prev_duration = previous_test.get("duration", 0)
                curr_duration = current_test.get("duration", 0)
                if prev_duration > 0 and curr_duration < prev_duration * 0.8:  # 20% más rápido
                    improvements.append({
                        "type": "PERFORMANCE_IMPROVEMENT",
                        "test": test_name,
                        "previous_duration": prev_duration,
                        "current_duration": curr_duration,
                        "improvement": f"{((1 - curr_duration / prev_duration) * 100):.1f}%"
                    })
        
        return improvements
    
    def generate_comparison_report(self, output_file: str = "comparison_report.json"):
        """Genera un reporte de comparación."""
        comparison = self.compare_summaries()
        regressions = self.find_regressions()
        improvements = self.find_improvements()
        
        report = {
            "comparison": comparison,
            "regressions": regressions,
            "improvements": improvements,
            "summary": {
                "total_regressions": len(regressions),
                "total_improvements": len(improvements),
                "overall_trend": "IMPROVING" if len(improvements) > len(regressions) else "REGRESSING" if len(regressions) > len(improvements) else "STABLE"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Reporte de comparación generado: {output_file}")
        return report
    
    def print_comparison(self):
        """Imprime comparación en consola."""
        if len(self.results) < 2:
            print("❌ Se necesitan al menos 2 resultados para comparar")
            return
        
        comparison = self.compare_summaries()
        regressions = self.find_regressions()
        improvements = self.find_improvements()
        
        print("\n" + "="*70)
        print("  COMPARACIÓN DE RESULTADOS")
        print("="*70)
        
        print(f"\n📊 Ejecuciones comparadas: {len(self.results)}")
        
        # Métricas
        print("\n📈 Métricas:")
        for metric, data in comparison.get("metrics", {}).items():
            print(f"  {metric}:")
            print(f"    Promedio: {data.get('average', 0):.2f}")
            print(f"    Min: {data.get('min', 0):.2f}")
            print(f"    Max: {data.get('max', 0):.2f}")
            if data.get('improvement', 0) != 0:
                trend = "📈" if data['improvement'] > 0 else "📉"
                print(f"    Cambio: {trend} {data['improvement']:.2f}")
        
        # Regresiones
        if regressions:
            print(f"\n❌ Regresiones encontradas: {len(regressions)}")
            for reg in regressions:
                severity = "🔴" if reg["severity"] == "HIGH" else "🟡"
                print(f"  {severity} {reg['test']}: {reg.get('type', 'Unknown')}")
        else:
            print("\n✅ No se encontraron regresiones")
        
        # Mejoras
        if improvements:
            print(f"\n✅ Mejoras encontradas: {len(improvements)}")
            for imp in improvements:
                print(f"  ✓ {imp['test']}: {imp.get('type', 'Unknown')}")
        else:
            print("\n⚠️  No se encontraron mejoras")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    import sys
    import glob
    
    comparator = TestComparison()
    
    # Buscar archivos de resultados
    result_files = glob.glob("test_results_*.json") + glob.glob("test_results.json")
    
    if not result_files:
        print("❌ No se encontraron archivos de resultados")
        print("   Ejecuta las pruebas primero para generar resultados")
        sys.exit(1)
    
    # Cargar resultados
    print(f"📂 Cargando {len(result_files)} archivo(s) de resultados...")
    for file in result_files:
        if comparator.load_result(file):
            print(f"  ✓ Cargado: {file}")
    
    # Comparar
    comparator.print_comparison()
    comparator.generate_comparison_report()
































