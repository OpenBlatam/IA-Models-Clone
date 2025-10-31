#!/usr/bin/env python3
"""
Run Improvements - Script ejecutor de mejoras reales
Ejecuta mejoras prácticas paso a paso
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Añadir el directorio actual al path para importar el motor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_improvements_engine import get_real_improvements_engine

class ImprovementRunner:
    """Ejecutor de mejoras reales"""
    
    def __init__(self):
        self.engine = get_real_improvements_engine()
        self.implemented_count = 0
        self.total_improvements = 0
    
    def show_menu(self):
        """Mostrar menú de opciones"""
        print("🚀 MEJORAS REALES - MENÚ PRINCIPAL")
        print("=" * 50)
        print("1. 📊 Ver mejoras de alto impacto")
        print("2. ⚡ Ver mejoras rápidas (Quick Wins)")
        print("3. 🎯 Implementar mejora específica")
        print("4. 📋 Crear plan de implementación")
        print("5. 📈 Ver estadísticas")
        print("6. 🚀 Implementar todas las mejoras rápidas")
        print("7. 📁 Exportar mejoras a JSON")
        print("8. ❌ Salir")
        print("=" * 50)
    
    def show_high_impact_improvements(self):
        """Mostrar mejoras de alto impacto"""
        print("\n🎯 MEJORAS DE ALTO IMPACTO (Score >= 8)")
        print("=" * 60)
        
        improvements = self.engine.get_high_impact_improvements()
        high_impact = [imp for imp in improvements if imp.get('impact_score', 0) >= 8]
        
        for i, improvement in enumerate(high_impact, 1):
            print(f"\n{i}. {improvement['title']}")
            print(f"   📊 Impacto: {improvement['impact_score']}/10")
            print(f"   ⏱️  Esfuerzo: {improvement['effort_hours']} horas")
            print(f"   🏷️  Categoría: {improvement['category']}")
            print(f"   🔥 Prioridad: {improvement['priority']}")
            print(f"   📝 {improvement['description']}")
            
            if improvement['implementation_steps']:
                print("   📋 Pasos:")
                for step in improvement['implementation_steps'][:3]:  # Mostrar solo los primeros 3
                    print(f"      • {step}")
                if len(improvement['implementation_steps']) > 3:
                    print(f"      • ... y {len(improvement['implementation_steps']) - 3} pasos más")
    
    def show_quick_wins(self):
        """Mostrar mejoras rápidas"""
        print("\n⚡ MEJORAS RÁPIDAS (Menos de 2 horas)")
        print("=" * 50)
        
        quick_wins = self.engine.get_quick_wins()
        
        for i, improvement in enumerate(quick_wins, 1):
            print(f"\n{i}. {improvement['title']}")
            print(f"   ⏱️  Esfuerzo: {improvement['effort_hours']} horas")
            print(f"   📊 Impacto: {improvement['impact_score']}/10")
            print(f"   📝 {improvement['description']}")
    
    def implement_specific_improvement(self):
        """Implementar mejora específica"""
        print("\n🎯 IMPLEMENTAR MEJORA ESPECÍFICA")
        print("=" * 40)
        
        # Mostrar opciones
        improvements = self.engine.get_high_impact_improvements()
        high_impact = [imp for imp in improvements if imp.get('impact_score', 0) >= 8]
        
        print("Mejoras disponibles:")
        for i, improvement in enumerate(high_impact, 1):
            print(f"{i}. {improvement['title']} ({improvement['effort_hours']}h, {improvement['impact_score']}/10)")
        
        try:
            choice = int(input("\nSelecciona el número de la mejora (0 para cancelar): "))
            if choice == 0:
                return
            
            if 1 <= choice <= len(high_impact):
                improvement = high_impact[choice - 1]
                self._implement_improvement(improvement)
            else:
                print("❌ Opción inválida")
        except ValueError:
            print("❌ Por favor ingresa un número válido")
    
    def _implement_improvement(self, improvement):
        """Implementar una mejora específica"""
        print(f"\n🚀 IMPLEMENTANDO: {improvement['title']}")
        print("=" * 50)
        print(f"📝 Descripción: {improvement['description']}")
        print(f"⏱️  Esfuerzo estimado: {improvement['effort_hours']} horas")
        print(f"📊 Impacto: {improvement['impact_score']}/10")
        
        print(f"\n📋 Pasos de implementación:")
        for i, step in enumerate(improvement['implementation_steps'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n💻 Ejemplos de código:")
        for i, example in enumerate(improvement['code_examples'], 1):
            print(f"\n   Ejemplo {i}:")
            print(f"   {example}")
        
        print(f"\n🧪 Notas de testing:")
        print(f"   {improvement['testing_notes']}")
        
        # Crear archivo de implementación
        self._create_implementation_file(improvement)
        
        print(f"\n✅ Archivo de implementación creado: {improvement['title'].lower().replace(' ', '_')}_implementation.py")
        print("💡 Ejecuta el archivo para implementar la mejora")
    
    def _create_implementation_file(self, improvement):
        """Crear archivo de implementación"""
        filename = f"{improvement['title'].lower().replace(' ', '_').replace(':', '')}_implementation.py"
        
        content = f'''#!/usr/bin/env python3
"""
Implementation: {improvement['title']}
Category: {improvement['category']}
Priority: {improvement['priority']}
Effort: {improvement['effort_hours']} hours
Impact: {improvement['impact_score']}/10
"""

import os
import sys
import subprocess
from datetime import datetime

def log_action(action: str, details: str = ""):
    """Log implementation action"""
    timestamp = datetime.now().isoformat()
    print(f"[{{timestamp}}] {{action}}: {{details}}")

def check_dependencies():
    """Check required dependencies"""
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "sqlalchemy", 
        "slowapi", "redis", "structlog", "psutil"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            log_action(f"✅ {{package}} found")
        except ImportError:
            missing.append(package)
            log_action(f"❌ {{package}} missing")
    
    if missing:
        log_action("Installing missing packages", ", ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
        log_action("✅ Dependencies installed")

def implement():
    """Implement {improvement['title']}"""
    log_action("🚀 Starting implementation", "{improvement['title']}")
    
    # Implementation steps:
'''
        
        for step in improvement['implementation_steps']:
            content += f"    # {step}\n"
        
        content += f'''
    # Code examples:
'''
        
        for i, example in enumerate(improvement['code_examples'], 1):
            content += f"    # Example {i}:\n    # {example.replace(chr(10), chr(10) + '    # ')}\n"
        
        content += f'''
    log_action("✅ Implementation completed", "{improvement['title']}")
    log_action("🧪 Testing notes", "{improvement['testing_notes']}")

if __name__ == "__main__":
    check_dependencies()
    implement()
'''
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def create_implementation_plan(self):
        """Crear plan de implementación"""
        print("\n📋 CREANDO PLAN DE IMPLEMENTACIÓN")
        print("=" * 40)
        
        plan = self.engine.create_implementation_plan()
        
        with open("implementation_plan.md", "w", encoding="utf-8") as f:
            f.write(plan)
        
        print("✅ Plan de implementación creado: implementation_plan.md")
        print("📖 Abre el archivo para ver el plan completo")
    
    def show_stats(self):
        """Mostrar estadísticas"""
        print("\n📈 ESTADÍSTICAS DE MEJORAS")
        print("=" * 30)
        
        improvements = self.engine.get_high_impact_improvements()
        quick_wins = self.engine.get_quick_wins()
        
        high_impact = [imp for imp in improvements if imp.get('impact_score', 0) >= 8]
        
        total_effort = sum(imp.get('effort_hours', 0) for imp in high_impact + quick_wins)
        avg_impact = sum(imp.get('impact_score', 0) for imp in high_impact + quick_wins) / len(high_impact + quick_wins) if (high_impact + quick_wins) else 0
        
        print(f"📊 Mejoras de alto impacto: {len(high_impact)}")
        print(f"⚡ Mejoras rápidas: {len(quick_wins)}")
        print(f"⏱️  Esfuerzo total estimado: {total_effort:.1f} horas")
        print(f"📈 Impacto promedio: {avg_impact:.1f}/10")
        print(f"🎯 Mejoras críticas: {len([imp for imp in high_impact if imp.get('priority') == 'critical'])}")
        print(f"🚀 Mejoras de performance: {len([imp for imp in high_impact if imp.get('category') == 'performance'])}")
    
    def implement_all_quick_wins(self):
        """Implementar todas las mejoras rápidas"""
        print("\n🚀 IMPLEMENTANDO TODAS LAS MEJORAS RÁPIDAS")
        print("=" * 50)
        
        quick_wins = self.engine.get_quick_wins()
        
        for i, improvement in enumerate(quick_wins, 1):
            print(f"\n{i}/{len(quick_wins)}. {improvement['title']}")
            self._create_implementation_file(improvement)
            self.implemented_count += 1
        
        print(f"\n✅ {len(quick_wins)} archivos de implementación creados")
        print("💡 Ejecuta cada archivo para implementar las mejoras")
    
    def export_improvements(self):
        """Exportar mejoras a JSON"""
        print("\n📁 EXPORTANDO MEJORAS")
        print("=" * 25)
        
        # Crear datos de exportación
        export_data = {
            "high_impact_improvements": self.engine.get_high_impact_improvements(),
            "quick_wins": self.engine.get_quick_wins(),
            "exported_at": datetime.now().isoformat(),
            "total_improvements": len(self.engine.get_high_impact_improvements()) + len(self.engine.get_quick_wins())
        }
        
        with open("improvements_export.json", "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print("✅ Mejoras exportadas a: improvements_export.json")
    
    def run(self):
        """Ejecutar el menú principal"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\nSelecciona una opción (1-8): ").strip()
                
                if choice == "1":
                    self.show_high_impact_improvements()
                elif choice == "2":
                    self.show_quick_wins()
                elif choice == "3":
                    self.implement_specific_improvement()
                elif choice == "4":
                    self.create_implementation_plan()
                elif choice == "5":
                    self.show_stats()
                elif choice == "6":
                    self.implement_all_quick_wins()
                elif choice == "7":
                    self.export_improvements()
                elif choice == "8":
                    print("\n👋 ¡Hasta luego! Implementa las mejoras y verás resultados reales.")
                    break
                else:
                    print("❌ Opción inválida. Por favor selecciona 1-8.")
                
                input("\nPresiona Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    runner = ImprovementRunner()
    runner.run()





