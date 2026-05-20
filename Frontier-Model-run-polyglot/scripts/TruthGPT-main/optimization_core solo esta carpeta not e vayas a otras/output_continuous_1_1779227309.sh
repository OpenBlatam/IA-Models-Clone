# Modo demo
python main_refactored_system.py --mode demo

# Procesamiento individual
python main_refactored_system.py --mode single --prompt "¿Qué es ML?"

# Procesamiento en lotes
python main_refactored_system.py --mode batch --batch-file requests.json

# Con exportación de reportes
python main_refactored_system.py --mode demo --export ./reports