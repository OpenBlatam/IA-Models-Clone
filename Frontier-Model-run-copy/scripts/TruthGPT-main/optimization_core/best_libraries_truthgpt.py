"""
BEST LIBRARIES FOR TRUTHGPT
Mejor selección de librerías especializadas para TruthGPT
Optimizaciones específicas con mejores prácticas de la industria
"""

import sys
import importlib.util

class BestLibrariesTruthGPT:
    """
    Mejores librerías especializadas para desarrollo TruthGPT.
    Incluye todas las herramientas necesarias para modelos LLM avanzados.
    """
    
    def __init__(self):
        self.libraries = self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Inicializar las mejores librerías disponibles."""
        return {
            # ===========================================
            # CORE DEEP LEARNING
            # ===========================================
            'torch': {
                'name': 'PyTorch',
                'version': '2.1.0',
                'description': 'Framework principal de deep learning',
                'priority': 'CRITICAL',
                'install': 'pip install torch>=2.1.0',
                'features': [
                    'nn.Module para arquitecturas',
                    'Autograd para diferenciación',
                    'torch.cuda.amp para mixed precision',
                    'DistributedDataParallel para multi-GPU'
                ]
            },
            
            'transformers': {
                'name': 'Transformers',
                'version': '4.35.0',
                'description': 'Modelos pre-entrenados para LLM',
                'priority': 'CRITICAL',
                'install': 'pip install transformers>=4.35.0',
                'features': [
                    'AutoModelForCausalLM',
                    'AutoTokenizer',
                    'TrainingArguments',
                    'Trainer'
                ]
            },
            
            'accelerate': {
                'name': 'Accelerate',
                'version': '0.25.0',
                'description': 'Optimización de entrenamiento',
                'priority': 'HIGH',
                'install': 'pip install accelerate>=0.25.0',
                'features': [
                    'Multi-GPU training',
                    'Mixed precision automático',
                    'Gradient accumulation'
                ]
            },
            
            'peft': {
                'name': 'PEFT',
                'version': '0.6.0',
                'description': 'Parameter-Efficient Fine-Tuning',
                'priority': 'HIGH',
                'install': 'pip install peft>=0.6.0',
                'features': [
                    'LoRA (Low-Rank Adaptation)',
                    'QLoRA (Quantized LoRA)',
                    'AdaLoRA',
                    'P-tuning v2'
                ]
            },
            
            'bitsandbytes': {
                'name': 'bitsandbytes',
                'version': '0.41.0',
                'description': 'Quantización 8-bit eficiente',
                'priority': 'HIGH',
                'install': 'pip install bitsandbytes>=0.41.0',
                'features': [
                    '8-bit optimizers',
                    'Quantization-aware training',
                    'Memory efficient training'
                ]
            },
            
            # ===========================================
            # LLM DEVELOPMENT
            # ===========================================
            'langchain': {
                'name': 'LangChain',
                'version': '0.1.0',
                'description': 'Desarrollo de aplicaciones LLM',
                'priority': 'MEDIUM',
                'install': 'pip install langchain>=0.1.0',
                'features': [
                    'Chain composition',
                    'Memory management',
                    'Agent framework',
                    'Document loaders'
                ]
            },
            
            'openai': {
                'name': 'OpenAI SDK',
                'version': '1.6.0',
                'description': 'Integración con OpenAI API',
                'priority': 'MEDIUM',
                'install': 'pip install openai>=1.6.0',
                'features': [
                    'GPT-4 access',
                    'Embeddings',
                    'Chat completion'
                ]
            },
            
            'anthropic': {
                'name': 'Anthropic SDK',
                'version': '0.7.0',
                'description': 'Integración con Claude',
                'priority': 'MEDIUM',
                'install': 'pip install anthropic>=0.7.0',
                'features': [
                    'Claude API',
                    'Message handling'
                ]
            },
            
            # ===========================================
            # DIFFUSION MODELS
            # ===========================================
            'diffusers': {
                'name': 'Diffusers',
                'version': '0.25.0',
                'description': 'Pipelines de diffusion',
                'priority': 'MEDIUM',
                'install': 'pip install diffusers>=0.25.0',
                'features': [
                    'Stable Diffusion',
                    'ControlNet',
                    'Schedulers'
                ]
            },
            
            # ===========================================
            # GPU OPTIMIZATION
            # ===========================================
            'flash_attn': {
                'name': 'Flash Attention',
                'version': '2.4.0',
                'description': 'Atención optimizada para GPU',
                'priority': 'HIGH',
                'install': 'pip install flash-attn>=2.4.0',
                'features': [
                    '2-3x faster attention',
                    'Memory efficient',
                    'Exact attention computation'
                ]
            },
            
            'xformers': {
                'name': 'XFormers',
                'version': '0.0.23',
                'description': 'Operaciones optimizadas',
                'priority': 'HIGH',
                'install': 'pip install xformers>=0.0.23',
                'features': [
                    'Memory efficient operations',
                    'Sparse attention patterns'
                ]
            },
            
            'triton': {
                'name': 'Triton',
                'version': '3.0.0',
                'description': 'Compilador JIT para GPU',
                'priority': 'MEDIUM',
                'install': 'pip install triton>=3.0.0',
                'features': [
                    'Custom GPU kernels',
                    'High performance'
                ]
            },
            
            # ===========================================
            # INTERACTIVE UIS
            # ===========================================
            'gradio': {
                'name': 'Gradio',
                'version': '4.7.0',
                'description': 'Interfaces interactivas',
                'priority': 'MEDIUM',
                'install': 'pip install gradio>=4.7.0',
                'features': [
                    'Web interfaces',
                    'Model demos',
                    'Real-time inference'
                ]
            },
            
            'streamlit': {
                'name': 'Streamlit',
                'version': '1.28.0',
                'description': 'Dashboards de datos',
                'priority': 'LOW',
                'install': 'pip install streamlit>=1.28.0',
                'features': [
                    'Rapid prototyping',
                    'Data visualization'
                ]
            },
            
            # ===========================================
            # EXPERIMENT TRACKING
            # ===========================================
            'wandb': {
                'name': 'Weights & Biases',
                'version': '0.16.0',
                'description': 'Experiment tracking',
                'priority': 'HIGH',
                'install': 'pip install wandb>=0.16.0',
                'features': [
                    'Hyperparameter tracking',
                    'Model versioning',
                    'Team collaboration'
                ]
            },
            
            'mlflow': {
                'name': 'MLflow',
                'version': '2.9.0',
                'description': 'ML lifecycle management',
                'priority': 'MEDIUM',
                'install': 'pip install mlflow>=2.9.0',
                'features': [
                    'Model registry',
                    'Experiment tracking',
                    'Model serving'
                ]
            },
            
            'tensorboard': {
                'name': 'TensorBoard',
                'version': '2.15.0',
                'description': 'Visualization de entrenamiento',
                'priority': 'MEDIUM',
                'install': 'pip install tensorboard>=2.15.0',
                'features': [
                    'Training metrics',
                    'Graph visualization',
                    'Embeddings visualization'
                ]
            },
            
            # ===========================================
            # DISTRIBUTED TRAINING
            # ===========================================
            'deepspeed': {
                'name': 'DeepSpeed',
                'version': '0.12.0',
                'description': 'Distributed training optimization',
                'priority': 'HIGH',
                'install': 'pip install deepspeed>=0.12.0',
                'features': [
                    'ZeRO optimization',
                    'Gradient checkpointing',
                    'Mixed precision'
                ]
            },
            
            'fairscale': {
                'name': 'FairScale',
                'version': '0.4.13',
                'description': 'Fair distributed training',
                'priority': 'MEDIUM',
                'install': 'pip install fairscale>=0.4.13',
                'features': [
                    'Sharded data parallel',
                    'Gradient scaling'
                ]
            },
            
            'horovod': {
                'name': 'Horovod',
                'version': '0.28.1',
                'description': 'Distributed training framework',
                'priority': 'MEDIUM',
                'install': 'pip install horovod>=0.28.1',
                'features': [
                    'Multi-GPU training',
                    'Ring-allreduce'
                ]
            },
            
            # ===========================================
            # SCIENTIFIC COMPUTING
            # ===========================================
            'numpy': {
                'name': 'NumPy',
                'version': '1.26.0',
                'description': 'Computación numérica',
                'priority': 'CRITICAL',
                'install': 'pip install numpy>=1.26.0',
                'features': [
                    'N-dimensional arrays',
                    'Mathematical operations'
                ]
            },
            
            'pandas': {
                'name': 'Pandas',
                'version': '2.1.0',
                'description': 'Manipulación de datos',
                'priority': 'HIGH',
                'install': 'pip install pandas>=2.1.0',
                'features': [
                    'DataFrame operations',
                    'Data analysis'
                ]
            },
            
            'scikit-learn': {
                'name': 'scikit-learn',
                'version': '1.3.0',
                'description': 'Machine learning clásico',
                'priority': 'MEDIUM',
                'install': 'pip install scikit-learn>=1.3.0',
                'features': [
                    'Preprocessing',
                    'Model evaluation'
                ]
            },
            
            # ===========================================
            # MONITORING & PROFILING
            # ===========================================
            'psutil': {
                'name': 'psutil',
                'version': '5.9.6',
                'description': 'System monitoring',
                'priority': 'MEDIUM',
                'install': 'pip install psutil>=5.9.6',
                'features': [
                    'CPU usage',
                    'Memory usage',
                    'GPU monitoring'
                ]
            },
            
            'memory_profiler': {
                'name': 'memory-profiler',
                'version': '0.61.0',
                'description': 'Memory profiling',
                'priority': 'LOW',
                'install': 'pip install memory-profiler>=0.61.0',
                'features': [
                    'Line-by-line profiling',
                    'Memory leak detection'
                ]
            },
            
            # ===========================================
            # TESTING & QUALITY
            # ===========================================
            'pytest': {
                'name': 'pytest',
                'version': '7.4.3',
                'description': 'Testing framework',
                'priority': 'MEDIUM',
                'install': 'pip install pytest>=7.4.3',
                'features': [
                    'Test discovery',
                    'Fixtures',
                    'Parametrization'
                ]
            },
            
            'black': {
                'name': 'Black',
                'version': '23.12.0',
                'description': 'Code formatter',
                'priority': 'MEDIUM',
                'install': 'pip install black>=23.12.0',
                'features': [
                    'Automatic formatting',
                    'Consistent style'
                ]
            },
            
            'ruff': {
                'name': 'Ruff',
                'version': '0.1.6',
                'description': 'Fast linter',
                'priority': 'MEDIUM',
                'install': 'pip install ruff>=0.1.6',
                'features': [
                    'Fast linting',
                    'Import sorting',
                    'Auto-fix'
                ]
            },
            
            'mypy': {
                'name': 'mypy',
                'version': '1.7.0',
                'description': 'Type checker',
                'priority': 'LOW',
                'install': 'pip install mypy>=1.7.0',
                'features': [
                    'Static type checking',
                    'Type inference'
                ]
            }
        }
    
    def get_library(self, name: str):
        """Obtener información de una librería específica."""
        return self.libraries.get(name, None)
    
    def get_by_priority(self, priority: str):
        """Obtener librerías por prioridad."""
        return {
            k: v for k, v in self.libraries.items()
            if v['priority'] == priority
        }
    
    def get_critical_libraries(self):
        """Obtener librerías críticas."""
        return self.get_by_priority('CRITICAL')
    
    def get_high_priority_libraries(self):
        """Obtener librerías de alta prioridad."""
        return self.get_by_priority('HIGH')
    
    def get_install_commands(self, priority: str = None):
        """Obtener comandos de instalación."""
        libs = self.libraries if not priority else self.get_by_priority(priority)
        commands = [lib['install'] for lib in libs.values()]
        return '\n'.join(commands)
    
    def check_installation(self, library_name: str) -> bool:
        """Verificar si una librería está instalada."""
        try:
            spec = importlib.util.find_spec(library_name)
            return spec is not None
        except ImportError:
            return False
    
    def get_missing_libraries(self):
        """Obtener librerías no instaladas."""
        missing = []
        for lib_name in self.libraries.keys():
            if not self.check_installation(lib_name):
                missing.append(lib_name)
        return missing
    
    def get_statistics(self):
        """Obtener estadísticas de las librerías."""
        total = len(self.libraries)
        priorities = {}
        for lib in self.libraries.values():
            prio = lib['priority']
            priorities[prio] = priorities.get(prio, 0) + 1
        
        return {
            'total_libraries': total,
            'by_priority': priorities,
            'critical': len(self.get_critical_libraries()),
            'high': len(self.get_high_priority_libraries())
        }


# Funciones de utilidad
def get_best_libraries():
    """Obtener instancia de mejores librerías."""
    return BestLibrariesTruthGPT()

def print_library_info(name: str):
    """Imprimir información de una librería."""
    lib_manager = get_best_libraries()
    lib_info = lib_manager.get_library(name)
    
    if lib_info:
        print(f"\n{'='*60}")
        print(f"Library: {lib_info['name']}")
        print(f"Version: {lib_info['version']}")
        print(f"Priority: {lib_info['priority']}")
        print(f"Description: {lib_info['description']}")
        print(f"\nInstall: {lib_info['install']}")
        print(f"\nFeatures:")
        for feature in lib_info['features']:
            print(f"  - {feature}")
        print(f"{'='*60}\n")
    else:
        print(f"Library '{name}' not found!")

def print_all_critical():
    """Imprimir todas las librerías críticas."""
    lib_manager = get_best_libraries()
    critical = lib_manager.get_critical_libraries()
    
    print("\n🚨 CRITICAL LIBRARIES:")
    print("="*60)
    for name, info in critical.items():
        print(f"{name}: {info['name']} (v{info['version']})")
    print("="*60)

def print_install_script(priority: str = None):
    """Imprimir script de instalación."""
    lib_manager = get_best_libraries()
    commands = lib_manager.get_install_commands(priority)
    
    print("\n📦 INSTALLATION SCRIPT:")
    print("="*60)
    print(commands)
    print("="*60)

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia
    lib_manager = get_best_libraries()
    
    # Mostrar estadísticas
    stats = lib_manager.get_statistics()
    print(f"\n📊 STATISTICS:")
    print(f"Total libraries: {stats['total_libraries']}")
    print(f"Critical: {stats['critical']}")
    print(f"High priority: {stats['high']}")
    
    # Mostrar librerías críticas
    print_all_critical()
    
    # Mostrar información de una librería específica
    print_library_info('transformers')
    
    # Verificar instalación
    missing = lib_manager.get_missing_libraries()
    if missing:
        print(f"\n⚠️ Missing libraries: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
    else:
        print("\n✅ All libraries are installed!")
    
    # Mostrar script de instalación para librerías críticas
    print_install_script('CRITICAL')

