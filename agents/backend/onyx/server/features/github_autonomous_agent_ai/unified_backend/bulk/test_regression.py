"""
Tests de Regresión Automatizados
Compara resultados actuales con baseline y detecta regresiones
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class RegressionTester:
    """Tester de regresión."""
    
    def __init__(self, baseline_file: str = "test_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline: Optional[Dict[str, Any]] = None
        self.current_results: Optional[Dict[str, Any]] = None
    
    def load_baseline(self) -> bool:
        """Carga el baseline."""
        if not os.path.exists(self.baseline_file):
            print(f"⚠️ Baseline no encontrado: {self.baseline_file}")
            print("   Ejecuta las pruebas primero para crear el baseline")
            return False
        
        try:
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                self.baseline = json.load(f)
            print(f"✅ Baseline cargado: {self.baseline_file}")
            return True
        except Exception as e:
            print(f"❌ Error cargando baseline: {e}")
            return False
    
    def load_current_results(self, results_file: str = "test_results.json") -> bool:
        """Carga resultados actuales."""
        if not os.path.exists(results_file):
            print(f"❌ Resultados no encontrados: {results_file}")
            return False
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.current_results = json.load(f)
            print(f"✅ Resultados actuales cargados: {results_file}")
            return True
        except Exception as e:
            print(f"❌ Error cargando resultados: {e}")
            return False
    
    def create_baseline(self, results_file: str = "test_results.json"):
        """Crea baseline desde resultados actuales."""
        if not os.path.exists(results_file):
            print(f"❌ No se encontraron resultados: {results_file}")
            return False
        
        try:
            import shutil
            shutil.copy(results_file, self.baseline_file)
            
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            
            baseline_data["baseline_created"] = datetime.now().isoformat()
            baseline_data["baseline_version"] = "1.0"
            
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Baseline creado: {self.baseline_file}")
            return True
        except Exception as e:
            print(f"❌ Error creando baseline: {e}")
            return False
    
    def compare_tests(self) -> List[Dict[str, Any]]:
        """Compara tests actuales con baseline."""
        if not self.baseline or not self.current_results:
            return []
        
        baseline_tests = {t["name"]: t for t in self.baseline.get("tests", [])}
        current_tests = {t["name"]: t for t in self.current_results.get("tests", [])}
        
        regressions = []
        
        for test_name, current_test in current_tests.items():
            if test_name in baseline_tests:
                baseline_test = baseline_tests[test_name]
                
                if baseline_test.get("passed") and not current_test.get("passed"):
                    regressions.append({
                        "type": "FAILURE_REGRESSION",
                        "test": test_name,
                        "baseline": "PASSED",
                        "current": "FAILED",
                        "severity": "HIGH"
                    })
                
                baseline_duration = baseline_test.get("duration", 0)
                current_duration = current_test.get("duration", 0)
                if baseline_duration > 0 and current_duration > baseline_duration * 1.5:
                    regressions.append({
                        "type": "PERFORMANCE_REGRESSION",
                        "test": test_name,
                        "baseline_duration": baseline_duration,
                        "current_duration": current_duration,
                        "increase_percent": ((current_duration / baseline_duration - 1) * 100),
                        "severity": "MEDIUM"
                    })
        
        return regressions
    
    def compare_summary(self) -> Dict[str, Any]:
        """Compara resumen."""
        if not self.baseline or not self.current_results:
            return {}
        
        baseline_summary = self.baseline.get("summary", {})
        current_summary = self.current_results.get("summary", {})
        
        comparison = {
            "baseline_date": baseline_summary.get("timestamp", "Unknown"),
            "current_date": current_summary.get("timestamp", "Unknown"),
            "metrics": {}
        }
        
        metrics = ["total", "passed", "failed", "success_rate", "duration"]
        
        for metric in metrics:
            baseline_val = baseline_summary.get(metric, 0)
            current_val = current_summary.get(metric, 0)
            
            change = current_val - baseline_val
            change_percent = (change / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison["metrics"][metric] = {
                "baseline": baseline_val,
                "current": current_val,
                "change": change,
                "change_percent": change_percent
            }
        
        return comparison
    
    def run_regression_test(self) -> bool:
        """Ejecuta test de regresión completo."""
        print("\n" + "="*70)
        print("  🔍 TEST DE REGRESIÓN")
        print("="*70 + "\n")
        
        if not self.load_baseline():
            print("\n💡 Para crear un baseline:")
            print("   1. Ejecuta las pruebas: python test_api_responses.py")
            print("   2. Crea baseline: python test_regression.py --create-baseline\n")
            return False
        
        if not self.load_current_results():
            return False
        
        regressions = self.compare_tests()
        comparison = self.compare_summary()
        
        print("\n📊 Comparación de Métricas:")
        print("-" * 70)
        for metric, data in comparison.get("metrics", {}).items():
            change = data["change"]
            change_pct = data["change_percent"]
            trend = "📈" if change > 0 and metric in ["passed", "success_rate"] else "📉" if change < 0 and metric in ["passed", "success_rate"] else "➡️"
            
            print(f"{metric.upper():<20} Baseline: {data['baseline']:<10} Current: {data['current']:<10} {trend} {change:+.2f} ({change_pct:+.1f}%)")
        
        print()
        
        if regressions:
            print(f"❌ REGRESIONES ENCONTRADAS: {len(regressions)}")
            print("-" * 70)
            for reg in regressions:
                severity_icon = "🔴" if reg["severity"] == "HIGH" else "🟡"
                print(f"{severity_icon} {reg['test']}: {reg['type']}")
                if reg["type"] == "PERFORMANCE_REGRESSION":
                    print(f"   Aumento: {reg['increase_percent']:.1f}%")
            print()
            return False
        else:
            print("✅ No se encontraron regresiones\n")
            return True
    
    def export_report(self, output_file: str = "regression_report.json"):
        """Exporta reporte de regresión."""
        if not self.baseline or not self.current_results:
            return
        
        regressions = self.compare_tests()
        comparison = self.compare_summary()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "baseline_file": self.baseline_file,
            "comparison": comparison,
            "regressions": regressions,
            "has_regressions": len(regressions) > 0
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Reporte exportado: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests de Regresión")
    parser.add_argument("--create-baseline", action="store_true", 
                       help="Crear baseline desde resultados actuales")
    parser.add_argument("--baseline", default="test_baseline.json",
                       help="Archivo de baseline")
    
    args = parser.parse_args()
    
    tester = RegressionTester(args.baseline)
    
    if args.create_baseline:
        tester.create_baseline()
    else:
        success = tester.run_regression_test()
        tester.export_report()
        exit(0 if success else 1)
































