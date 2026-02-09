"""
MOEA AI Assistant - Asistente inteligente
==========================================
Asistente con IA para ayudar con el uso del sistema MOEA
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class MOEAAIAssistant:
    """Asistente inteligente MOEA"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.command_history: List[Dict] = []
    
    def _load_knowledge_base(self) -> Dict:
        """Cargar base de conocimiento"""
        return {
            "commands": {
                "generate": {
                    "description": "Generar un nuevo proyecto MOEA",
                    "command": "python quick_moea.py",
                    "alternatives": ["python moea_wrapper.py generate", "python generate_moea.py"]
                },
                "setup": {
                    "description": "Configurar proyecto generado",
                    "command": "python moea_setup.py",
                    "alternatives": ["python moea_wrapper.py setup"]
                },
                "test": {
                    "description": "Probar la API del proyecto",
                    "command": "python moea_test_api.py",
                    "alternatives": ["python moea_wrapper.py test"]
                },
                "monitor": {
                    "description": "Monitorear el sistema en tiempo real",
                    "command": "python moea_monitor.py",
                    "alternatives": ["python moea_wrapper.py monitor"]
                },
                "dashboard": {
                    "description": "Abrir dashboard web",
                    "command": "python moea_dashboard.py",
                    "alternatives": []
                }
            },
            "troubleshooting": {
                "server_not_running": {
                    "problem": "El servidor no está corriendo",
                    "solution": "Ejecuta: cd backend && uvicorn app.main:app --reload",
                    "check": "python moea_health.py"
                },
                "project_not_found": {
                    "problem": "Proyecto no encontrado",
                    "solution": "Genera el proyecto primero: python quick_moea.py",
                    "check": "python moea_utils.py list"
                },
                "dependencies_missing": {
                    "problem": "Dependencias faltantes",
                    "solution": "Instala dependencias: python moea_setup.py",
                    "check": "pip list"
                }
            },
            "best_practices": [
                "Siempre verifica el proyecto después de generarlo",
                "Crea backups regulares de tus proyectos",
                "Monitorea el sistema regularmente",
                "Ejecuta health checks antes de producción",
                "Revisa la documentación generada"
            ]
        }
    
    def suggest_command(self, intent: str) -> Optional[Dict]:
        """Sugerir comando basado en intención"""
        intent_lower = intent.lower()
        
        # Buscar en comandos
        for cmd_name, cmd_info in self.knowledge_base["commands"].items():
            if cmd_name in intent_lower or any(word in intent_lower for word in cmd_info["description"].lower().split()):
                return {
                    "command": cmd_info["command"],
                    "description": cmd_info["description"],
                    "alternatives": cmd_info.get("alternatives", [])
                }
        
        return None
    
    def troubleshoot(self, problem: str) -> Optional[Dict]:
        """Ayudar con troubleshooting"""
        problem_lower = problem.lower()
        
        for issue_key, issue_info in self.knowledge_base["troubleshooting"].items():
            if issue_key.replace("_", " ") in problem_lower or issue_info["problem"].lower() in problem_lower:
                return {
                    "problem": issue_info["problem"],
                    "solution": issue_info["solution"],
                    "check": issue_info.get("check", "")
                }
        
        return None
    
    def get_help(self, topic: Optional[str] = None) -> str:
        """Obtener ayuda"""
        if not topic:
            help_text = "📚 MOEA AI Assistant - Comandos Disponibles:\n\n"
            for cmd_name, cmd_info in self.knowledge_base["commands"].items():
                help_text += f"  {cmd_name}: {cmd_info['description']}\n"
                help_text += f"    → {cmd_info['command']}\n\n"
            return help_text
        
        topic_lower = topic.lower()
        
        # Buscar comando específico
        suggestion = self.suggest_command(topic)
        if suggestion:
            return f"💡 {suggestion['description']}\n   Comando: {suggestion['command']}"
        
        # Buscar troubleshooting
        solution = self.troubleshoot(topic)
        if solution:
            return f"🔧 Problema: {solution['problem']}\n   Solución: {solution['solution']}"
        
        return f"❓ No encontré información sobre: {topic}"
    
    def interactive_mode(self):
        """Modo interactivo"""
        print("=" * 70)
        print("MOEA AI Assistant - Modo Interactivo".center(70))
        print("=" * 70)
        print("\nEscribe 'help' para ver comandos, 'quit' para salir\n")
        
        while True:
            try:
                user_input = input("MOEA> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if user_input.lower() == 'help':
                    print(self.get_help())
                    continue
                
                # Intentar sugerir comando
                suggestion = self.suggest_command(user_input)
                if suggestion:
                    print(f"\n💡 {suggestion['description']}")
                    print(f"   Comando sugerido: {suggestion['command']}")
                    if suggestion.get('alternatives'):
                        print(f"   Alternativas: {', '.join(suggestion['alternatives'])}")
                    print()
                    continue
                
                # Intentar troubleshooting
                solution = self.troubleshoot(user_input)
                if solution:
                    print(f"\n🔧 Problema detectado: {solution['problem']}")
                    print(f"   Solución: {solution['solution']}")
                    if solution.get('check'):
                        print(f"   Verificar: {solution['check']}")
                    print()
                    continue
                
                # Respuesta genérica
                print(f"❓ No estoy seguro de cómo ayudarte con: {user_input}")
                print("   Escribe 'help' para ver comandos disponibles\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA AI Assistant")
    parser.add_argument(
        'query',
        nargs='?',
        help='Consulta o comando'
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Modo interactivo'
    )
    
    args = parser.parse_args()
    
    assistant = MOEAAIAssistant()
    
    if args.interactive:
        assistant.interactive_mode()
    elif args.query:
        result = assistant.get_help(args.query)
        print(result)
    else:
        print(assistant.get_help())


if __name__ == "__main__":
    main()

